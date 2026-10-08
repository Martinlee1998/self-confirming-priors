"""Run the frozen real PC3-v12 evidence-availability protocol."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
import os
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
V11_SRC = Path(os.environ.get(
    "PC3_V11_SRC",
    str(HERE.parent / "pc3_v11_graph_recovery"),
))
V10_SRC = Path(os.environ.get(
    "PC3_V10_SRC", str(HERE.parent / "pc3_v10_real_integration")
))
V9_ROOT = Path(os.environ.get(
    "PC3_V9_ROOT", str(HERE.parents[3])
))
for value in (V10_SRC, V9_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


def _load_v11() -> Any:
    path = V11_SRC / "run_graph_recovery.py"
    spec = importlib.util.spec_from_file_location("pc3_v11_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen v11 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v11 = _load_v11()
from four_controller_fixed_contrast import (  # noqa: E402
    CONTROLLER_SEEDS,
    _achievement_state,
    _canonical_depth,
    _final_checkpoint_model,
    _inverse_cdf,
    _move_to_device,
    _online_inputs,
)

TARGETS = tuple(v11.TARGETS)
STREAMS = (0, 1)
ACQUISITION_ARMS = ("N", "E")
EVALUATION_ARMS = ("O", "F", "N", "E")
TARGET_WINDOW = 1_000
HORIZON = 10_000
SCHEMA = "pc3-v12-evidence-availability/v1"
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


class V12Error(v11.V11Error):
    pass


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _worlds(phase: str) -> tuple[int, ...]:
    lower, upper = PROTOCOL[f"{phase}_worlds"]
    values = tuple(range(int(lower), int(upper) + 1))
    expected = 40 if phase == "acquisition" else 100
    if len(values) != expected:
        raise V12Error("frozen world namespace has the wrong size")
    return values


def _uniforms(controller: int, world: int, stream: int, kind: int) -> np.ndarray:
    generator = np.random.default_rng(
        np.random.SeedSequence([controller, world, stream, kind, 20260826, 12])
    )
    return np.asarray(generator.random(HORIZON + 1), dtype=np.float64)


def _initial_corrupt_graph() -> Any:
    # v11 U/P2 is exactly the frozen A-C-B-D-E graph and updater prior.
    return v11.initial_graph("U", "P2")


def _initial_eval_graph(arm: str) -> Any:
    if arm == "O":
        return v11.initial_graph("O", "P2")
    if arm == "F":
        return v11.initial_graph("F", "P2")
    raise V12Error("only O/F have fixed initial evaluation graphs")


def _instrument_target(achieved: Mapping[str, bool]) -> str | None:
    """Return the forced target using public achievement booleans only."""
    if not bool(achieved["A"]):
        return "A"
    if not bool(achieved["B"]):
        return "B"
    if not bool(achieved["C"]):
        return "C"
    return None


def _select_target(provider: Any, achieved: Mapping[str, bool], target_age: int,
                   uniform: float, instrument: bool) -> tuple[str | None, bool]:
    forced_code = _instrument_target(achieved) if instrument else None
    if forced_code is not None:
        return TARGETS[TARGETS.index(forced_code)], True
    return provider.select(achieved, target_age, uniform), False


def _edge_total(graph: Any, source: str, target: str) -> float:
    index = v11._edge_index(TARGETS.index(source), TARGETS.index(target))
    return float(graph.alpha[index] + graph.pending_positive[index])


def _run_arm(*, model: Any, handoff: Any, provider: Any, device: Any,
             target_uniforms: np.ndarray, action_uniforms: np.ndarray,
             instrument: bool) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_v9_observation_consistent.live_adapter import LiveObservableAdapter

    if instrument and not provider.mutable:
        raise V12Error("instrument is forbidden for frozen/evaluation providers")
    adapter = LiveObservableAdapter()
    achieved = _achievement_state(handoff.achievements)
    first_hits: dict[str, int | None] = {code: None for code in TARGETS}
    hit_order: list[str] = []
    before_bc = _edge_total(provider.graph, "B", "C")
    before_cb_counter = float(provider.graph.beta[v11._edge_index(2, 1)] + provider.graph.pending_counter[v11._edge_index(2, 1)])
    target, forced = _select_target(provider, achieved, 0, float(target_uniforms[0]), instrument)
    hidden = model.initial_hidden(1, device=device)
    previous_action = -1
    target_age = 0
    window_actions = 0
    target_counts = {code: 0 for code in TARGETS}
    instrument_counts = {code: 0 for code in ("A", "B", "C")}
    instrument_reselections = int(forced)
    reselections = 1
    native_return = 0.0
    stop_reason = "HORIZON"
    terminal = False
    terminal_update = None
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
            # The only policy fields are those returned by frozen v10 _online_inputs.
            if "instrument" in inputs or "graph" in inputs or "evidence" in inputs:
                raise V12Error("private acquisition state entered policy inputs")
            result = model(**_move_to_device(inputs, device), hidden=hidden)
            logits = result["policy_logits"][0, 0]
            probabilities = logits.softmax(-1)
            if not all(bool(torch.isfinite(x).all().item()) for x in (
                logits, probabilities, result["value"], result["final_hidden"]
            )):
                raise V12Error("non-finite policy output")
            action = _inverse_cdf(probabilities, float(action_uniforms[global_step]))
            hidden = result["final_hidden"]
            _observation, reward, done, _info = handoff.step(action)
            native_return += float(reward)
            previous_action = action
            target_counts[target] += 1
            if forced:
                instrument_counts[target] += 1
            target_age += 1
            window_actions += 1
            achieved_after = _achievement_state(handoff.achievements)
            new_hits = []
            for code in TARGETS:
                if achieved_after[code] and first_hits[code] is None:
                    first_hits[code] = global_step + 1
                    new_hits.append(code)
            # Both arms use this exact updater path; the instrument never calls it.
            for code in new_hits:
                provider.observe_first_hit(code, tuple(hit_order))
            hit_order.extend(new_hits)  # same-transition public hits remain tied
            terminal = bool(done) or bool(handoff.terminal)
            if terminal:
                terminal_update = provider.update()
                stop_reason = "E_HIT" if achieved_after["E"] else "TERMINAL"
                achieved = achieved_after
                break
            reselect = target in new_hits
            if window_actions == TARGET_WINDOW:
                provider.update()
                reselect = True
            if reselect:
                target, forced = _select_target(
                    provider, achieved_after, 0,
                    float(target_uniforms[min(global_step + 1, HORIZON)]), instrument,
                )
                target_age = 0
                window_actions = 0
                reselections += 1
                instrument_reselections += int(forced)
            achieved = achieved_after
        else:
            global_step = HORIZON - 1
    # Protocol requires the last partial acquisition window to be flushed.
    final_flush = provider.update() if provider.mutable and not terminal else None
    after_bc = _edge_total(provider.graph, "B", "C")
    after_cb_counter = float(provider.graph.beta[v11._edge_index(2, 1)] + provider.graph.pending_counter[v11._edge_index(2, 1)])
    steps = global_step + 1
    depth = _canonical_depth(achieved)
    q = sum(bool(achieved[code]) for code in TARGETS) / len(TARGETS)
    bc_pair = int(
        first_hits["B"] is not None and first_hits["C"] is not None
        and int(first_hits["B"]) < int(first_hits["C"])
    )
    return {
        "step_count": int(steps), "stop_reason": stop_reason,
        "final_achievements": achieved, "first_hit_steps": first_hits,
        "first_hit_order": hit_order, "canonical_prefix_depth": depth,
        "R": float(depth / len(TARGETS)), "Q": float(q),
        "full_chain": int(depth == len(TARGETS)), "E_hit": bool(achieved["E"]),
        "native_return": float(native_return), "target_action_counts": target_counts,
        "target_reselections": int(reselections), "optimizer_steps": 0,
        "instrument_enabled": bool(instrument),
        "instrument_public_rule": "A_then_B_then_C_from_public_achievements" if instrument else None,
        "instrument_forced_action_counts": instrument_counts,
        "instrument_forced_reselections": int(instrument_reselections),
        "real_B_before_C_pair": bc_pair,
        "B_to_C_positive_evidence_added": after_bc - before_bc,
        "C_to_B_counterevidence_added": after_cb_counter - before_cb_counter,
        "terminal_update": terminal_update, "final_flush": final_flush,
    }


def _checkpoint(controller: int) -> Path:
    root = Path(os.environ.get(
        "PC3_CHECKPOINT_DIR", str(HERE.parents[4] / "checkpoints")
    ))
    return root / f"controller_{controller}" / "ppo_latest.pt"


def _row_path(root: Path, phase: str, controller: int, stream: int, world: int) -> Path:
    return root / phase / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"


def run_acquisition(*, controller: int, stream_only: int | None,
                    output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs

    if controller not in CONTROLLER_SEEDS:
        raise V12Error("unknown controller")
    device = torch.device(device_name)
    model = _final_checkpoint_model(
        checkpoint=_checkpoint(controller), controller_seed=controller, device=device
    )
    completed = 0
    selected_streams = STREAMS if stream_only is None else (stream_only,)
    for stream in selected_streams:
        graphs = {arm: _initial_corrupt_graph() for arm in ACQUISITION_ARMS}
        for world in _worlds("acquisition"):
            path = _row_path(output_root, "acquisition", controller, stream, world)
            if path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("schema") != SCHEMA or old.get("identity") != [controller, stream, world]:
                    raise V12Error("corrupt or mixed acquisition row")
                graphs = {arm: v11.GraphState.from_json(old["graphs_after"][arm]) for arm in ACQUISITION_ARMS}
                completed += 1
                continue
            target_u = _uniforms(controller, world, stream, 11)
            action_u = _uniforms(controller, world, stream, 12)
            _prefix, handoffs = open_arm_handoffs(world, ACQUISITION_ARMS)
            try:
                arms = {
                    arm: _run_arm(
                        model=model, handoff=handoffs[arm],
                        provider=v11.TargetProvider(graphs[arm], mutable=True), device=device,
                        target_uniforms=target_u, action_uniforms=action_u,
                        instrument=(arm == "E"),
                    ) for arm in ACQUISITION_ARMS
                }
            finally:
                for handoff in handoffs.values():
                    handoff.close()
            row = {
                "schema": SCHEMA, "phase": "acquisition",
                "identity": [controller, stream, world],
                "paired_uniforms": {"target": [controller, world, stream, 11, 20260826, 12],
                                    "action": [controller, world, stream, 12, 20260826, 12]},
                "same_initial_corrupt_graph": world == _worlds("acquisition")[0],
                "arms": arms,
                "graphs_after": {arm: graphs[arm].to_json() for arm in ACQUISITION_ARMS},
            }
            _atomic_json(row, path)
            completed += 1
    return {"status": "ACQUISITION_COMPLETE", "controller": controller, "rows": completed}


def _load_frozen(output_root: Path, controller: int, stream: int, arm: str) -> Any:
    path = _row_path(output_root, "acquisition", controller, stream, _worlds("acquisition")[-1])
    if not path.is_file():
        raise V12Error("acquisition is incomplete")
    value = json.loads(path.read_text(encoding="utf-8"))
    return v11.GraphState.from_json(value["graphs_after"][arm])


def _standardized_tv(left_graph: Any, right_graph: Any) -> float:
    left = v11.TargetProvider(left_graph, mutable=False)
    right = v11.TargetProvider(right_graph, mutable=False)
    values = []
    for bits in itertools.product((False, True), repeat=len(TARGETS)):
        if all(bits):
            continue
        context = {code: bool(bits[i]) for i, code in enumerate(TARGETS)}
        p, q = left.distribution(context, 0), right.distribution(context, 0)
        values.append(0.5 * sum(abs(x - y) for x, y in zip(p, q)))
    return float(fmean(values))


def run_evaluation(*, controller: int, stream_only: int | None,
                   output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs

    if controller not in CONTROLLER_SEEDS:
        raise V12Error("unknown controller")
    device = torch.device(device_name)
    model = _final_checkpoint_model(
        checkpoint=_checkpoint(controller), controller_seed=controller, device=device
    )
    completed = 0
    selected_streams = STREAMS if stream_only is None else (stream_only,)
    for stream in selected_streams:
        frozen = {"O": _initial_eval_graph("O"), "F": _initial_eval_graph("F"),
                  "N": _load_frozen(output_root, controller, stream, "N"),
                  "E": _load_frozen(output_root, controller, stream, "E")}
        tv_no = _standardized_tv(frozen["N"], frozen["O"])
        tv_eo = _standardized_tv(frozen["E"], frozen["O"])
        for world in _worlds("evaluation"):
            path = _row_path(output_root, "evaluation", controller, stream, world)
            if path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("schema") != SCHEMA or old.get("identity") != [controller, stream, world]:
                    raise V12Error("corrupt or mixed evaluation row")
                completed += 1
                continue
            target_u = _uniforms(controller, world, stream, 21)
            action_u = _uniforms(controller, world, stream, 22)
            _prefix, handoffs = open_arm_handoffs(world, EVALUATION_ARMS)
            try:
                arms = {
                    arm: _run_arm(
                        model=model, handoff=handoffs[arm],
                        provider=v11.TargetProvider(copy.deepcopy(frozen[arm]), mutable=False),
                        device=device, target_uniforms=target_u, action_uniforms=action_u,
                        instrument=False,
                    ) for arm in EVALUATION_ARMS
                }
            finally:
                for handoff in handoffs.values():
                    handoff.close()
            row = {
                "schema": SCHEMA, "phase": "evaluation", "identity": [controller, stream, world],
                "paired_uniforms": {"target": [controller, world, stream, 21, 20260826, 12],
                                    "action": [controller, world, stream, 22, 20260826, 12]},
                "optimizer_steps": 0, "graphs_frozen": True, "instrument_enabled": False,
                "arms": arms,
                "standardized_target_tv": {
                    "N_O": tv_no, "E_O": tv_eo, "recovery": tv_no - tv_eo,
                    "contexts": 31,
                    "context_definition": "all_noncomplete_achieved_masks_age_zero_equal_weight",
                },
            }
            _atomic_json(row, path)
            completed += 1
    return {"status": "EVALUATION_COMPLETE", "controller": controller, "rows": completed}


def _mean_ci(values: Sequence[float], critical: float) -> dict[str, Any]:
    mean = fmean(values)
    if len(values) < 2:
        return {"estimate": mean, "se": 0.0, "lower": mean, "upper": mean, "n": len(values)}
    se = stdev(values) / math.sqrt(len(values))
    return {"estimate": mean, "se": se, "lower": mean - critical * se,
            "upper": mean + critical * se, "n": len(values), "t_critical": critical}


def run_merge(*, output_root: Path) -> dict[str, Any]:
    acquisition = []
    evaluation = []
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            for world in _worlds("acquisition"):
                path = _row_path(output_root, "acquisition", controller, stream, world)
                if not path.is_file():
                    raise V12Error(f"acquisition row unavailable: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("identity") != [controller, stream, world]:
                    raise V12Error("acquisition arm/data mixing")
                acquisition.append(row)
            for world in _worlds("evaluation"):
                path = _row_path(output_root, "evaluation", controller, stream, world)
                if not path.is_file():
                    raise V12Error(f"evaluation row unavailable: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("identity") != [controller, stream, world]:
                    raise V12Error("evaluation arm/data mixing")
                evaluation.append(row)

    world_metrics = {}
    for world in _worlds("evaluation"):
        group = [row for row in evaluation if row["identity"][2] == world]
        def contrast(left: str, right: str) -> float:
            return fmean(float(row["arms"][left]["R"]) - float(row["arms"][right]["R"]) for row in group)
        world_metrics[world] = {
            "behavior_E_minus_N": contrast("E", "N"),
            "E_minus_F": contrast("E", "F"),
            "O_minus_F": contrast("O", "F"),
        }
    primary = {
        key: {**_mean_ci([v[key] for v in world_metrics.values()], 1.9842169515),
              "independent_level": "evaluation_world"}
        for key in ("behavior_E_minus_N", "E_minus_F", "O_minus_F")
    }
    damage = primary["O_minus_F"]["estimate"]
    recovery = primary["E_minus_F"]["estimate"]
    primary["recovery_fraction"] = {
        "estimate": recovery / damage if damage != 0 else None,
        "definition": "E_minus_F / O_minus_F",
    }
    chain_tv = []
    chain_exposure = []
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            eval_chain = [row for row in evaluation if row["identity"][:2] == [controller, stream]]
            acq_chain = [row for row in acquisition if row["identity"][:2] == [controller, stream]]
            chain_tv.append(float(eval_chain[0]["standardized_target_tv"]["recovery"]))
            n_pairs = sum(float(row["arms"]["N"]["B_to_C_positive_evidence_added"]) for row in acq_chain)
            e_pairs = sum(float(row["arms"]["E"]["B_to_C_positive_evidence_added"]) for row in acq_chain)
            chain_exposure.append({"controller": controller, "stream": stream,
                                   "N_B_before_C_evidence": n_pairs,
                                   "E_B_before_C_evidence": e_pairs,
                                   "E_minus_N": e_pairs - n_pairs})
    primary["target_recovery"] = {
        **_mean_ci(chain_tv, 2.3646242516),
        "definition": "TV(N,O)-TV(E,O)",
        "independent_level": "frozen_controller_stream_acquisition_chain",
    }
    primary["exposure_manipulation"] = {
        **_mean_ci([x["E_minus_N"] for x in chain_exposure], 2.3646242516),
        "unit": "real_public_B_before_C_pair_observations_per_chain",
        "independent_level": "controller_stream_acquisition_chain",
        "chains": chain_exposure,
    }
    descriptive = {
        arm: {metric: fmean(float(row["arms"][arm][metric]) for row in evaluation)
              for metric in ("R", "Q", "full_chain", "E_hit", "native_return")}
        for arm in EVALUATION_ARMS
    }
    posteriors = {}
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            final = next(row for row in acquisition
                         if row["identity"] == [controller, stream, _worlds("acquisition")[-1]])
            key = f"{controller}_s{stream}"
            posteriors[key] = {}
            for arm in ACQUISITION_ARMS:
                graph = v11.GraphState.from_json(final["graphs_after"][arm])
                posteriors[key][arm] = {
                    "P_B_to_C": graph.probability(1, 2),
                    "P_C_to_B": graph.probability(2, 1),
                    "first_hit_count": graph.first_hit_count,
                    "update_count": graph.update_count,
                }
    result = {
        "schema": SCHEMA, "phase": "merged_evaluation",
        "controllers": list(CONTROLLER_SEEDS), "streams": list(STREAMS),
        "acquisition_worlds": list(_worlds("acquisition")),
        "evaluation_worlds": list(_worlds("evaluation")),
        "acquisition_rows": len(acquisition), "evaluation_rows": len(evaluation),
        "optimizer_steps": 0, "primary": primary,
        "world_metrics": {str(k): v for k, v in world_metrics.items()},
        "descriptive": descriptive, "frozen_posteriors": posteriors,
        "analysis_policy": "confidence intervals are descriptive; no statistical gate, rerun, sample addition, or instrument tuning",
    }
    output = output_root / "result.json"
    _atomic_json(result, output)
    return {"status": "MERGE_COMPLETE", "rows": len(evaluation),
            "output": str(output), "primary": primary}


def run_smoke(*, output_root: Path, device_name: str) -> dict[str, Any]:
    """Tiny real Crafter/checkpoint run; explicitly outside evidence namespaces."""
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs

    global HORIZON
    original = HORIZON
    HORIZON = 16
    try:
        controller, world, stream = CONTROLLER_SEEDS[0], 1_339_999, 0
        device = torch.device(device_name)
        model = _final_checkpoint_model(
            checkpoint=_checkpoint(controller), controller_seed=controller, device=device
        )
        graphs = {arm: _initial_corrupt_graph() for arm in ACQUISITION_ARMS}
        initial_equal = graphs["N"].to_json() == graphs["E"].to_json()
        target_u, action_u = _uniforms(controller, world, stream, 91), _uniforms(controller, world, stream, 92)
        _prefix, handoffs = open_arm_handoffs(world, ACQUISITION_ARMS)
        try:
            arms = {
                arm: _run_arm(
                    model=model, handoff=handoffs[arm],
                    provider=v11.TargetProvider(graphs[arm], mutable=True), device=device,
                    target_uniforms=target_u, action_uniforms=action_u,
                    instrument=(arm == "E"),
                ) for arm in ACQUISITION_ARMS
            }
        finally:
            for handoff in handoffs.values():
                handoff.close()
        eval_graph = copy.deepcopy(graphs["E"])
        _prefix, eval_handoffs = open_arm_handoffs(world + 1, ("EVAL",))
        try:
            eval_arm = _run_arm(
                model=model, handoff=eval_handoffs["EVAL"],
                provider=v11.TargetProvider(eval_graph, mutable=False), device=device,
                target_uniforms=target_u, action_uniforms=action_u, instrument=False,
            )
        finally:
            eval_handoffs["EVAL"].close()
        checks = {
            "real_checkpoint_loaded": True,
            "real_crafter_steps_positive": all(arm["step_count"] > 0 for arm in arms.values()),
            "N_E_initial_graph_identical": initial_equal,
            "E_target_public_rule": arms["E"]["instrument_public_rule"] == "A_then_B_then_C_from_public_achievements",
            "instrument_flag_absent_from_policy_inputs": True,
            "graph_updates_only_via_real_first_hits": all(
                arm["B_to_C_positive_evidence_added"] in (0.0, 1.0) for arm in arms.values()
            ),
            "eval_instrument_disabled": eval_arm["instrument_enabled"] is False,
            "optimizer_steps_zero": all(arm["optimizer_steps"] == 0 for arm in arms.values()) and eval_arm["optimizer_steps"] == 0,
        }
        if not all(checks.values()):
            raise V12Error(f"real wiring smoke failed: {checks}")
        result = {"schema": SCHEMA, "phase": "non_evidence_real_wiring_smoke",
                  "identity": ["SMOKE", controller, stream, world], "actions_per_arm": 16,
                  "checks": checks, "arms": arms, "evaluation_probe": eval_arm}
        _atomic_json(result, output_root / "smoke" / "real_wiring.json")
        return result
    finally:
        HORIZON = original


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "acquire", "evaluate", "merge"))
    parser.add_argument("--controller", type=int)
    parser.add_argument("--stream", type=int, choices=STREAMS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage in {"acquire", "evaluate"} and args.controller is None:
        parser.error("--controller is required for acquire/evaluate")
    if args.stage == "smoke":
        result = run_smoke(output_root=args.output_root, device_name=args.device)
    elif args.stage == "acquire":
        result = run_acquisition(controller=args.controller, stream_only=args.stream,
                                 output_root=args.output_root, device_name=args.device)
    elif args.stage == "evaluate":
        result = run_evaluation(controller=args.controller, stream_only=args.stream,
                                output_root=args.output_root, device_name=args.device)
    else:
        result = run_merge(output_root=args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
