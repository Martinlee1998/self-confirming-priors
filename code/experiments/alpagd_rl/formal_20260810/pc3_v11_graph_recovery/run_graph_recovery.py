"""Run the real PC3-v11 graph-private acquisition and evaluation protocol."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
V10_SRC = Path(os.environ.get(
    "PC3_V10_SRC", str(HERE.parent / "pc3_v10_real_integration")
))
if str(V10_SRC) not in sys.path:
    sys.path.insert(0, str(V10_SRC))
V9_ROOT = Path(os.environ.get(
    "PC3_V9_ROOT", str(HERE.parents[3])
))
if str(V9_ROOT) not in sys.path:
    sys.path.insert(0, str(V9_ROOT))

from four_controller_fixed_contrast import (  # noqa: E402
    CONTROLLER_SEEDS,
    FORMAL_HORIZON,
    TARGET_ORDER,
    FixedContrastError,
    _achievement_state,
    _canonical_depth,
    _final_checkpoint_model,
    _inverse_cdf,
    _move_to_device,
    _online_inputs,
)

TARGETS = tuple(TARGET_ORDER)
ARMS = ("O", "F", "U", "D")
STREAMS = (0, 1)
TARGET_WINDOW = 1_000
HORIZON = 10_000
SCHEMA = "pc3-v11-graph-recovery/v1"
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
CORRUPTIONS = {
    name: (tuple(value["delete"]), tuple(value["insert"]))
    for name, value in PROTOCOL["permutations"].items()
}


class V11Error(FixedContrastError):
    pass


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _validate_permutation(name: str) -> None:
    if name not in CORRUPTIONS:
        raise V11Error("permutation must be P1 or P2")


def _worlds(permutation: str, phase: str) -> tuple[int, ...]:
    _validate_permutation(permutation)
    key = f"{phase}_worlds"
    lower, upper = PROTOCOL["permutations"][permutation][key]
    values = tuple(range(int(lower), int(upper) + 1))
    expected = 40 if phase == "acquisition" else 100
    if len(values) != expected:
        raise V11Error("frozen world namespace has the wrong size")
    return values


def _edge_index(source: int, target: int) -> int:
    return source * len(TARGETS) + target


@dataclass
class GraphState:
    arm: str
    alpha: list[float]
    beta: list[float]
    pending_positive: list[float]
    pending_counter: list[float]
    update_count: int = 0
    first_hit_count: int = 0

    def probability(self, source: int, target: int) -> float:
        index = _edge_index(source, target)
        return self.alpha[index] / (self.alpha[index] + self.beta[index])

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "GraphState":
        state = cls(
            arm=str(value["arm"]),
            alpha=[float(x) for x in value["alpha"]],
            beta=[float(x) for x in value["beta"]],
            pending_positive=[float(x) for x in value["pending_positive"]],
            pending_counter=[float(x) for x in value["pending_counter"]],
            update_count=int(value["update_count"]),
            first_hit_count=int(value["first_hit_count"]),
        )
        state.validate()
        return state

    def validate(self) -> None:
        size = len(TARGETS) ** 2
        if self.arm not in ARMS or any(len(x) != size for x in (
            self.alpha, self.beta, self.pending_positive, self.pending_counter
        )):
            raise V11Error("corrupt graph state shape")
        if not all(math.isfinite(x) and x >= 0 for values in (
            self.alpha, self.beta, self.pending_positive, self.pending_counter
        ) for x in values):
            raise V11Error("non-finite graph state")


def initial_graph(arm: str, permutation: str) -> GraphState:
    _validate_permutation(permutation)
    if arm not in ARMS:
        raise V11Error("unknown arm")
    size = len(TARGETS) ** 2
    alpha, beta = [1.0] * size, [1.0] * size
    if arm != "D":
        active = {(i, j) for i in range(5) for j in range(5) if i < j}
        if arm in {"F", "U"}:
            deleted, inserted = CORRUPTIONS[permutation]
            active.remove((TARGETS.index(deleted[0]), TARGETS.index(deleted[1])))
            active.add((TARGETS.index(inserted[0]), TARGETS.index(inserted[1])))
        for i, j in itertools.permutations(range(5), 2):
            index = _edge_index(i, j)
            alpha[index], beta[index] = ((9.0, 1.0) if (i, j) in active else (1.0, 9.0))
    state = GraphState(arm, alpha, beta, [0.0] * size, [0.0] * size)
    state.validate()
    return state


class TargetProvider:
    """Graph-private provider; no method exposes graph fields to the policy."""

    def __init__(self, graph: GraphState, *, mutable: bool):
        self.graph = graph
        self.mutable = bool(mutable)

    def distribution(self, achieved: Mapping[str, bool], target_age: int) -> tuple[float, ...]:
        if target_age < 0:
            raise V11Error("negative target age")
        unfinished = [i for i, code in enumerate(TARGETS) if not bool(achieved[code])]
        if not unfinished:
            return (0.0,) * len(TARGETS)
        eligible = []
        for target in unfinished:
            blocked = any(
                source != target
                and self.graph.probability(source, target) > 0.5
                and not bool(achieved[TARGETS[source]])
                for source in range(len(TARGETS))
            )
            if not blocked:
                eligible.append(target)
        if not eligible:
            # A transient learned cycle is resolved symmetrically from the least
            # posterior obstruction; index order never acts as a tie-break.
            obstruction = {
                target: sum(
                    self.graph.probability(source, target)
                    for source in unfinished if source != target
                ) for target in unfinished
            }
            minimum = min(obstruction.values())
            eligible = [t for t in unfinished if math.isclose(obstruction[t], minimum, abs_tol=1e-12)]
        mass = 1.0 / len(eligible)
        return tuple(mass if i in eligible else 0.0 for i in range(len(TARGETS)))

    def select(self, achieved: Mapping[str, bool], target_age: int, uniform: float) -> str | None:
        probabilities = self.distribution(achieved, target_age)
        if sum(probabilities) == 0.0:
            return None
        if not 0.0 <= float(uniform) < 1.0:
            raise V11Error("invalid target uniform")
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if uniform < cumulative:
                return TARGETS[index]
        return TARGETS[max(i for i, x in enumerate(probabilities) if x > 0)]

    def observe_first_hit(self, target: str, earlier: Sequence[str]) -> None:
        if not self.mutable:
            return
        target_index = TARGETS.index(target)
        for source in earlier:
            source_index = TARGETS.index(source)
            self.graph.pending_positive[_edge_index(source_index, target_index)] += 1.0
            self.graph.pending_counter[_edge_index(target_index, source_index)] += 1.0
        self.graph.first_hit_count += 1

    def update(self) -> dict[str, Any]:
        if not self.mutable:
            return {"mutable": False, "changed": False}
        changed = False
        observations = 0.0
        for index in range(len(self.graph.alpha)):
            positive = self.graph.pending_positive[index]
            counter = self.graph.pending_counter[index]
            observations += positive + counter
            if positive or counter:
                self.graph.alpha[index] += positive
                self.graph.beta[index] += counter
                self.graph.pending_positive[index] = 0.0
                self.graph.pending_counter[index] = 0.0
                changed = True
        self.graph.update_count += 1
        self.graph.validate()
        return {"mutable": True, "changed": changed, "observations": observations,
                "update_count": self.graph.update_count}


def _uniforms(controller: int, world: int, stream: int, kind: int) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([controller, world, stream, kind, 20260826]))
    return np.asarray(generator.random(HORIZON + 1), dtype=np.float64)


def _run_arm(*, model: Any, handoff: Any, provider: TargetProvider, device: Any,
             target_uniforms: np.ndarray, action_uniforms: np.ndarray) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_v9_observation_consistent.live_adapter import LiveObservableAdapter

    adapter = LiveObservableAdapter()
    achieved = _achievement_state(handoff.achievements)
    first_hits: dict[str, int | None] = {code: None for code in TARGETS}
    hit_order: list[str] = []
    target = provider.select(achieved, 0, float(target_uniforms[0]))
    hidden = model.initial_hidden(1, device=device)
    previous_action = -1
    target_age = 0
    window_actions = 0
    target_counts = {code: 0 for code in TARGETS}
    reselections = 1
    native_return = 0.0
    stop_reason = "HORIZON"
    model.eval()
    with torch.no_grad():
        for global_step in range(HORIZON):
            if target is None:
                stop_reason = "ALL_TARGETS_COMPLETE"
                break
            observable = adapter.capture(handoff, prev_action=previous_action)
            inputs = _online_inputs(
                observable, target=target,
                age=float(np.float32(min(target_age, TARGET_WINDOW) / TARGET_WINDOW)),
                episode_boundary=(global_step == 0),
            )
            result = model(**_move_to_device(inputs, device), hidden=hidden)
            logits = result["policy_logits"][0, 0]
            probabilities = logits.softmax(-1)
            if not all(bool(torch.isfinite(x).all().item()) for x in (
                logits, probabilities, result["value"], result["final_hidden"]
            )):
                raise V11Error("non-finite policy output")
            action = _inverse_cdf(probabilities, float(action_uniforms[global_step]))
            hidden = result["final_hidden"]
            _observation, reward, done, _info = handoff.step(action)
            native_return += float(reward)
            previous_action = action
            target_counts[target] += 1
            target_age += 1
            window_actions += 1
            achieved_after = _achievement_state(handoff.achievements)
            new_hits = []
            for code in TARGETS:
                if achieved_after[code] and first_hits[code] is None:
                    first_hits[code] = global_step + 1
                    new_hits.append(code)
            for code in new_hits:
                provider.observe_first_hit(code, tuple(hit_order))
            # Hits on the same public transition are tied: do not manufacture
            # an A--E order from container iteration.
            hit_order.extend(new_hits)
            terminal = bool(done) or bool(handoff.terminal)
            if terminal:
                update = provider.update()
                stop_reason = "E_HIT" if achieved_after["E"] else "TERMINAL"
                achieved = achieved_after
                break
            reselect = False
            if target in new_hits:
                reselect = True
            if window_actions == TARGET_WINDOW:
                provider.update()
                reselect = True
            if reselect:
                target = provider.select(
                    achieved_after, 0, float(target_uniforms[min(global_step + 1, HORIZON)])
                )
                target_age = 0
                window_actions = 0
                reselections += 1
            achieved = achieved_after
        else:
            global_step = HORIZON - 1
    steps = (global_step + 1) if HORIZON else 0
    depth = _canonical_depth(achieved)
    q = sum(bool(achieved[code]) for code in TARGETS) / len(TARGETS)
    return {
        "step_count": int(steps), "stop_reason": stop_reason,
        "final_achievements": achieved, "first_hit_steps": first_hits,
        "first_hit_order": hit_order, "canonical_prefix_depth": depth,
        "R": float(depth / len(TARGETS)), "Q": float(q),
        "full_chain": int(depth == len(TARGETS)), "E_hit": bool(achieved["E"]),
        "native_return": float(native_return), "target_action_counts": target_counts,
        "target_reselections": int(reselections), "optimizer_steps": 0,
        "terminal_update": update if terminal else None,
    }


def _checkpoint(controller: int) -> Path:
    root = Path(os.environ.get(
        "PC3_CHECKPOINT_DIR", str(HERE.parents[4] / "checkpoints")
    ))
    return root / f"controller_{controller}" / "ppo_latest.pt"


def _row_path(root: Path, phase: str, permutation: str, controller: int, stream: int, world: int) -> Path:
    return root / permutation / phase / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"


def run_acquisition(*, permutation: str, controller: int, stream_only: int | None,
                    output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs

    if controller not in CONTROLLER_SEEDS:
        raise V11Error("unknown controller")
    device = torch.device(device_name)
    model = _final_checkpoint_model(checkpoint=_checkpoint(controller), controller_seed=controller, device=device)
    completed = 0
    selected_streams = STREAMS if stream_only is None else (stream_only,)
    if any(stream not in STREAMS for stream in selected_streams):
        raise V11Error("stream must be 0 or 1")
    for stream in selected_streams:
        graph_u, graph_d = initial_graph("U", permutation), initial_graph("D", permutation)
        for world in _worlds(permutation, "acquisition"):
            path = _row_path(output_root, "acquisition", permutation, controller, stream, world)
            if path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("schema") != SCHEMA or old.get("identity") != [permutation, controller, stream, world]:
                    raise V11Error("corrupt or mixed acquisition row")
                graph_u = GraphState.from_json(old["graphs_after"]["U"])
                graph_d = GraphState.from_json(old["graphs_after"]["D"])
                completed += 1
                continue
            target_u = _uniforms(controller, world, stream, 11)
            action_u = _uniforms(controller, world, stream, 12)
            _prefix, handoffs = open_arm_handoffs(world, ("U", "D"))
            try:
                arms = {
                    "U": _run_arm(model=model, handoff=handoffs["U"],
                                  provider=TargetProvider(graph_u, mutable=True), device=device,
                                  target_uniforms=target_u, action_uniforms=action_u),
                    "D": _run_arm(model=model, handoff=handoffs["D"],
                                  provider=TargetProvider(graph_d, mutable=True), device=device,
                                  target_uniforms=target_u, action_uniforms=action_u),
                }
            finally:
                for handoff in handoffs.values():
                    handoff.close()
            row = {
                "schema": SCHEMA, "phase": "acquisition",
                "identity": [permutation, controller, stream, world],
                "paired_uniforms": {"target": [controller, world, stream, 11],
                                    "action": [controller, world, stream, 12]},
                "arms": arms,
                "graphs_after": {"U": graph_u.to_json(), "D": graph_d.to_json()},
            }
            _atomic_json(row, path)
            completed += 1
    return {"status": "ACQUISITION_COMPLETE", "permutation": permutation,
            "controller": controller, "rows": completed}


def _load_frozen_graph(output_root: Path, permutation: str, controller: int, stream: int, arm: str) -> GraphState:
    world = _worlds(permutation, "acquisition")[-1]
    path = _row_path(output_root, "acquisition", permutation, controller, stream, world)
    if not path.is_file():
        raise V11Error("acquisition is incomplete")
    value = json.loads(path.read_text(encoding="utf-8"))
    state = GraphState.from_json(value["graphs_after"][arm])
    state.arm = arm
    return state


def _standardized_tv(graph_left: GraphState, graph_right: GraphState) -> float:
    left, right = TargetProvider(graph_left, mutable=False), TargetProvider(graph_right, mutable=False)
    values = []
    for bits in itertools.product((False, True), repeat=len(TARGETS)):
        if all(bits):
            continue
        context = {code: bool(bits[i]) for i, code in enumerate(TARGETS)}
        p = left.distribution(context, 0)
        q = right.distribution(context, 0)
        values.append(0.5 * sum(abs(x - y) for x, y in zip(p, q)))
    return float(fmean(values))


def run_evaluation(*, permutation: str, controller: int, stream_only: int | None,
                   output_root: Path, device_name: str) -> dict[str, Any]:
    import copy
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs

    if controller not in CONTROLLER_SEEDS:
        raise V11Error("unknown controller")
    device = torch.device(device_name)
    model = _final_checkpoint_model(checkpoint=_checkpoint(controller), controller_seed=controller, device=device)
    completed = 0
    selected_streams = STREAMS if stream_only is None else (stream_only,)
    if any(stream not in STREAMS for stream in selected_streams):
        raise V11Error("stream must be 0 or 1")
    for stream in selected_streams:
        frozen = {
            "O": initial_graph("O", permutation), "F": initial_graph("F", permutation),
            "U": _load_frozen_graph(output_root, permutation, controller, stream, "U"),
            "D": _load_frozen_graph(output_root, permutation, controller, stream, "D"),
        }
        tv_fo = _standardized_tv(frozen["F"], frozen["O"])
        tv_uo = _standardized_tv(frozen["U"], frozen["O"])
        for world in _worlds(permutation, "evaluation"):
            path = _row_path(output_root, "evaluation", permutation, controller, stream, world)
            if path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("schema") != SCHEMA or old.get("identity") != [permutation, controller, stream, world]:
                    raise V11Error("corrupt or mixed evaluation row")
                completed += 1
                continue
            target_u = _uniforms(controller, world, stream, 21)
            action_u = _uniforms(controller, world, stream, 22)
            _prefix, handoffs = open_arm_handoffs(world, ARMS)
            try:
                arms = {
                    arm: _run_arm(model=model, handoff=handoffs[arm],
                                  provider=TargetProvider(copy.deepcopy(frozen[arm]), mutable=False),
                                  device=device, target_uniforms=target_u, action_uniforms=action_u)
                    for arm in ARMS
                }
            finally:
                for handoff in handoffs.values():
                    handoff.close()
            row = {
                "schema": SCHEMA, "phase": "evaluation",
                "identity": [permutation, controller, stream, world],
                "paired_uniforms": {"target": [controller, world, stream, 21],
                                    "action": [controller, world, stream, 22]},
                "optimizer_steps": 0, "graphs_frozen": True, "arms": arms,
                "standardized_target_tv": {"F_O": tv_fo, "U_O": tv_uo,
                    "recovery": tv_fo - tv_uo, "contexts": 31,
                    "context_definition": "all_noncomplete_achieved_masks_age_zero_equal_weight"},
            }
            _atomic_json(row, path)
            completed += 1
    return {"status": "EVALUATION_COMPLETE", "permutation": permutation,
            "controller": controller, "rows": completed}


def _mean_ci(values: Sequence[float], *, critical: float) -> dict[str, float]:
    mean = fmean(values)
    if len(values) < 2:
        return {"estimate": mean, "se": 0.0, "lower": mean, "upper": mean}
    se = stdev(values) / math.sqrt(len(values))
    return {"estimate": mean, "se": se, "lower": mean - critical * se,
            "upper": mean + critical * se, "n": len(values), "t_critical": critical}


def run_merge(*, permutation: str, output_root: Path) -> dict[str, Any]:
    rows = []
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            for world in _worlds(permutation, "evaluation"):
                path = _row_path(output_root, "evaluation", permutation, controller, stream, world)
                if not path.is_file():
                    raise V11Error(f"evaluation row unavailable: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("identity") != [permutation, controller, stream, world]:
                    raise V11Error("evaluation arm/data mixing")
                rows.append(row)
    world_metrics: dict[int, dict[str, float]] = {}
    for world in _worlds(permutation, "evaluation"):
        group = [row for row in rows if row["identity"][3] == world]
        def contrast(left: str, right: str, metric: str = "R") -> float:
            return fmean(float(row["arms"][left][metric]) - float(row["arms"][right][metric]) for row in group)
        world_metrics[world] = {
            "damage_O_minus_F": contrast("O", "F"),
            "behavior_U_minus_F": contrast("U", "F"),
            "language_U_minus_D": contrast("U", "D"),
            "target_recovery": fmean(float(row["standardized_target_tv"]["recovery"]) for row in group),
        }
    primary = {
        key: {**_mean_ci([value[key] for value in world_metrics.values()], critical=1.9842169515),
              "independent_level": "evaluation_world"}
        for key in ("damage_O_minus_F", "behavior_U_minus_F", "language_U_minus_D")
    }
    chain_recovery = []
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            chain_rows = [row for row in rows
                          if row["identity"][1] == controller and row["identity"][2] == stream]
            chain_recovery.append(float(chain_rows[0]["standardized_target_tv"]["recovery"]))
    primary["target_recovery"] = {
        **_mean_ci(chain_recovery, critical=2.3646242516),
        "independent_level": "frozen_controller_stream_acquisition_chain",
        "note": "one provider-derived value per chain; not 100 replicated world observations",
    }
    damage = primary["damage_O_minus_F"]["estimate"]
    behavior = primary["behavior_U_minus_F"]["estimate"]
    primary["recovery_fraction"] = {
        "estimate": (behavior / damage) if damage != 0 else None,
        "definition": "behavior_U_minus_F / damage_O_minus_F",
    }
    descriptive = {
        arm: {metric: fmean(float(row["arms"][arm][metric]) for row in rows)
              for metric in ("R", "Q", "full_chain", "E_hit", "native_return")}
        for arm in ARMS
    }
    result = {
        "schema": SCHEMA, "phase": "merged_evaluation", "permutation": permutation,
        "controllers": list(CONTROLLER_SEEDS), "streams": list(STREAMS),
        "worlds": list(_worlds(permutation, "evaluation")), "rows": len(rows),
        "optimizer_steps": 0, "primary": primary,
        "world_metrics": {str(k): v for k, v in world_metrics.items()},
        "descriptive": descriptive,
    }
    output = output_root / permutation / "result.json"
    _atomic_json(result, output)
    return {"status": "MERGE_COMPLETE", "permutation": permutation,
            "rows": len(rows), "output": str(output), "primary": primary}


def run_smoke(*, output_root: Path, device_name: str) -> dict[str, Any]:
    """One real checkpoint/model/env wiring smoke in a non-evidence namespace."""
    import copy
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs

    global HORIZON
    original = HORIZON
    HORIZON = 8
    try:
        controller, world, stream = CONTROLLER_SEEDS[0], 1_299_999, 0
        device = torch.device(device_name)
        model = _final_checkpoint_model(checkpoint=_checkpoint(controller), controller_seed=controller, device=device)
        graphs = {arm: initial_graph(arm, "P1") for arm in ARMS}
        target_u, action_u = _uniforms(controller, world, stream, 91), _uniforms(controller, world, stream, 92)
        _prefix, handoffs = open_arm_handoffs(world, ARMS)
        try:
            arms = {arm: _run_arm(model=model, handoff=handoffs[arm],
                                  provider=TargetProvider(copy.deepcopy(graphs[arm]), mutable=False),
                                  device=device, target_uniforms=target_u, action_uniforms=action_u)
                    for arm in ARMS}
        finally:
            for handoff in handoffs.values():
                handoff.close()
        result = {"schema": SCHEMA, "phase": "non_evidence_wiring_smoke",
                  "identity": ["SMOKE", controller, stream, world], "actions": 8,
                  "optimizer_steps": 0, "arms": arms}
        _atomic_json(result, output_root / "smoke" / "wiring.json")
    finally:
        HORIZON = original
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "acquire", "evaluate", "merge"))
    parser.add_argument("--permutation", choices=("P1", "P2"))
    parser.add_argument("--controller", type=int)
    parser.add_argument("--stream", type=int, choices=STREAMS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage != "smoke" and args.permutation is None:
        parser.error("--permutation is required except for smoke")
    if args.stage in {"acquire", "evaluate"} and args.controller is None:
        parser.error("--controller is required for acquire/evaluate")
    if args.stage == "smoke":
        result = run_smoke(output_root=args.output_root, device_name=args.device)
    elif args.stage == "acquire":
        result = run_acquisition(permutation=args.permutation, controller=args.controller,
                                 stream_only=args.stream, output_root=args.output_root,
                                 device_name=args.device)
    elif args.stage == "evaluate":
        result = run_evaluation(permutation=args.permutation, controller=args.controller,
                                stream_only=args.stream, output_root=args.output_root,
                                device_name=args.device)
    else:
        result = run_merge(permutation=args.permutation, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
