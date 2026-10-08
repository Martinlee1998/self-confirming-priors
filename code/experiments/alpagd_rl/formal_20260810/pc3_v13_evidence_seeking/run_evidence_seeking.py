"""PC3-v13: real non-oracle frontier evidence-seeking acquisition."""
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
V11_SRC = Path(os.environ.get("PC3_V11_SRC", str(HERE.parent / "pc3_v11_graph_recovery")))
V10_SRC = Path(os.environ.get("PC3_V10_SRC", str(HERE.parent / "pc3_v10_real_integration")))
V9_ROOT = Path(os.environ.get("PC3_V9_ROOT", str(HERE.parents[3])))
for value in (V10_SRC, V9_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


def _load_v11() -> Any:
    path = V11_SRC / "run_graph_recovery.py"
    spec = importlib.util.spec_from_file_location("pc3_v11_frozen_for_v13", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v11 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v11 = _load_v11()
from four_controller_fixed_contrast import (  # noqa: E402
    CONTROLLER_SEEDS, _achievement_state, _canonical_depth, _final_checkpoint_model,
    _inverse_cdf, _move_to_device, _online_inputs,
)

TARGETS = tuple(v11.TARGETS)
STREAMS = (0, 1)
ACQUISITION_ARMS = ("N", "ES")
EVALUATION_ARMS = ("O", "F", "N", "ES")
HORIZON = 10_000
TARGET_WINDOW = 1_000
SCHEMA = "pc3-v13-evidence-seeking/v1"
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


class V13Error(v11.V11Error):
    pass


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _worlds(phase: str) -> tuple[int, ...]:
    lower, upper = PROTOCOL[f"{phase}_worlds"]
    worlds = tuple(range(int(lower), int(upper) + 1))
    if len(worlds) != (40 if phase == "acquisition" else 100):
        raise V13Error("unexpected frozen world namespace")
    return worlds


def _uniforms(controller: int, world: int, stream: int, kind: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([controller, world, stream, kind, 20260827, 13]))
    return np.asarray(rng.random(HORIZON + 1), dtype=np.float64)


def _initial_corrupt_graph() -> Any:
    return v11.initial_graph("U", "P2")


def _initial_eval_graph(arm: str) -> Any:
    if arm not in {"O", "F"}:
        raise V13Error("only O/F are fixed evaluation graphs")
    return v11.initial_graph(arm, "P2")


def _checkpoint(controller: int) -> Path:
    root = Path(os.environ.get("PC3_CHECKPOINT_DIR", str(HERE.parents[4] / "checkpoints")))
    return root / f"controller_{controller}" / "ppo_latest.pt"


def _row_path(root: Path, phase: str, controller: int, stream: int, world: int) -> Path:
    return root / phase / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"


def _mask_key(achieved: Mapping[str, bool]) -> str:
    return "".join("1" if achieved[code] else "0" for code in TARGETS)


def _decision_signature(provider: Any) -> tuple[tuple[float, ...], ...]:
    """All 31 registered provider decisions; this never reaches the policy."""
    values = []
    for bits in itertools.product((False, True), repeat=len(TARGETS)):
        if all(bits):
            continue
        mask = {code: bool(bits[i]) for i, code in enumerate(TARGETS)}
        values.append(tuple(round(x, 12) for x in provider.distribution(mask, 0)))
    return tuple(values)


def _pair_key(source: str, target: str) -> str:
    return "|".join(sorted((source, target)))


def _history_record(history: dict[str, Any], pair: tuple[str, str]) -> dict[str, Any]:
    key = _pair_key(*pair)
    return history.setdefault(key, {
        "pair": list(pair), "attempts": 0, "first_hit_yield": 0,
        "discriminating_yield": 0, "consecutive_zero_yield": 0,
        "cooldown_until_world": 0, "retired": False,
    })


def _frontier_candidates(provider: Any, achieved: Mapping[str, bool], history: Mapping[str, Any],
                         world: int, used_masks: set[str], used_pairs: set[str]) -> list[dict[str, Any]]:
    """Return graph-private blocker probes without referring to any true edge/order."""
    mask = _mask_key(achieved)
    if mask in used_masks or len(used_pairs) >= int(PROTOCOL["pair_memory"]["max_challenges_per_world"]):
        return []
    distribution = provider.distribution(achieved, 0)
    unfinished = [i for i, code in enumerate(TARGETS) if not achieved[code]]
    candidates: list[dict[str, Any]] = []
    for target_i in unfinished:
        blockers = [source_i for source_i in unfinished if source_i != target_i
                    and provider.graph.probability(source_i, target_i) > 0.5]
        if len(blockers) != 1:
            continue
        blocker_i = blockers[0]
        if distribution[blocker_i] <= 0.0:
            continue
        # The stored direction is the discriminating event ES is trying to
        # observe: blocked target before its sole provider blocker.
        pair = (TARGETS[target_i], TARGETS[blocker_i])
        key = _pair_key(*pair)
        record = history.get(key, {})
        if key in used_pairs or record.get("retired", False) or int(record.get("cooldown_until_world", 0)) >= world:
            continue
        # Score the provider decision effect of removing this *current* blocker
        # edge.  The copy is only a scorer; ES never writes this counterfactual
        # graph and never sees a reference/correct edge.
        counterfactual = copy.deepcopy(provider.graph)
        edge = v11._edge_index(blocker_i, target_i)
        counterfactual.alpha[edge], counterfactual.beta[edge] = 1.0, 9.0
        after_distribution = v11.TargetProvider(counterfactual, mutable=False).distribution(achieved, 0)
        impact = 0.5 * sum(abs(a - b) for a, b in zip(distribution, after_distribution))
        # A cooled pair is excluded; the remaining score therefore has no outcome input.
        candidates.append({"pair": pair, "pair_key": key, "mask": mask,
                           "probe_target": TARGETS[target_i], "blocker": TARGETS[blocker_i],
                           "evidence_edge": [TARGETS[target_i], TARGETS[blocker_i]],
                           "evidence_edge_probability_before": provider.graph.probability(target_i, blocker_i),
                           "decision_impact": float(impact)})
    return candidates


def _choose_frontier(provider: Any, achieved: Mapping[str, bool], history: Mapping[str, Any],
                     world: int, uniform: float, used_masks: set[str], used_pairs: set[str]) -> dict[str, Any] | None:
    candidates = _frontier_candidates(provider, achieved, history, world, used_masks, used_pairs)
    if not candidates:
        return None
    best = max(item["decision_impact"] for item in candidates)
    tied = sorted((item for item in candidates if math.isclose(item["decision_impact"], best, abs_tol=1e-12)),
                 key=lambda item: item["pair_key"])
    return tied[min(int(float(uniform) * len(tied)), len(tied) - 1)]


def _select_target(provider: Any, achieved: Mapping[str, bool], uniform: float, mode: str,
                   es_history: Mapping[str, Any], world: int, used_masks: set[str],
                   used_pairs: set[str]) -> tuple[str | None, dict[str, Any] | None, bool]:
    if mode == "ES":
        challenge = _choose_frontier(provider, achieved, es_history, world, uniform, used_masks, used_pairs)
        if challenge is not None:
            return str(challenge["probe_target"]), challenge, True
    return provider.select(achieved, 0, uniform), None, False


def _run_arm(*, model: Any, handoff: Any, provider: Any, device: Any, target_uniforms: np.ndarray,
             action_uniforms: np.ndarray, mode: str, world: int, es_history: dict[str, Any] | None,
             selection_mode: str | None = None) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_v9_observation_consistent.live_adapter import LiveObservableAdapter

    if mode not in ACQUISITION_ARMS + EVALUATION_ARMS:
        raise V13Error("unknown run arm")
    selection_mode = mode if selection_mode is None else selection_mode
    if selection_mode == "ES" and (es_history is None or not provider.mutable):
        raise V13Error("ES is acquisition-only and requires its chain history")
    adapter = LiveObservableAdapter()
    achieved = _achievement_state(handoff.achievements)
    first_hits: dict[str, int | None] = {code: None for code in TARGETS}
    hit_order: list[str] = []
    used_masks: set[str] = set()
    used_pairs: set[str] = set()
    challenges: list[dict[str, Any]] = []
    pre_signature = _decision_signature(provider) if selection_mode == "ES" else ()
    target, challenge, forced = _select_target(provider, achieved, float(target_uniforms[0]), selection_mode,
                                                es_history or {}, world, used_masks, used_pairs)
    if challenge and selection_mode == "ES":
        used_masks.add(str(challenge["mask"])); used_pairs.add(str(challenge["pair_key"])); challenges.append(dict(challenge))
    hidden = model.initial_hidden(1, device=device)
    previous_action, target_age, window_actions, native_return = -1, 0, 0, 0.0
    target_counts = {code: 0 for code in TARGETS}
    stop_reason, terminal = "HORIZON", False
    model.eval()
    with torch.no_grad():
        for step in range(HORIZON):
            if target is None:
                stop_reason = "ALL_TARGETS_COMPLETE"; break
            observable = adapter.capture(handoff, prev_action=previous_action)
            inputs = _online_inputs(observable, target=target,
                                    age=float(np.float32(min(target_age, TARGET_WINDOW) / TARGET_WINDOW)),
                                    episode_boundary=(step == 0))
            if any(name in inputs for name in ("graph", "evidence", "challenge", "instrument", "history")):
                raise V13Error("private ES state entered policy inputs")
            output = model(**_move_to_device(inputs, device), hidden=hidden)
            logits, probabilities = output["policy_logits"][0, 0], output["policy_logits"][0, 0].softmax(-1)
            if not all(bool(torch.isfinite(item).all().item()) for item in (logits, probabilities, output["value"], output["final_hidden"])):
                raise V13Error("non-finite frozen policy output")
            action = _inverse_cdf(probabilities, float(action_uniforms[step]))
            hidden = output["final_hidden"]
            _obs, reward, done, _info = handoff.step(action)
            native_return += float(reward); previous_action = action; target_counts[target] += 1
            target_age += 1; window_actions += 1
            achieved_after = _achievement_state(handoff.achievements)
            new_hits = [code for code in TARGETS if achieved_after[code] and first_hits[code] is None]
            for code in new_hits:
                first_hits[code] = step + 1; provider.observe_first_hit(code, tuple(hit_order))
            hit_order.extend(new_hits)
            terminal = bool(done) or bool(handoff.terminal)
            if terminal:
                provider.update(); achieved = achieved_after; stop_reason = "E_HIT" if achieved_after["E"] else "TERMINAL"; break
            reselect = target in new_hits
            if window_actions == TARGET_WINDOW:
                provider.update(); reselect = True
            if reselect:
                target, challenge, forced = _select_target(provider, achieved_after,
                    float(target_uniforms[min(step + 1, HORIZON)]), selection_mode, es_history or {}, world, used_masks, used_pairs)
                if challenge and selection_mode == "ES":
                    used_masks.add(str(challenge["mask"])); used_pairs.add(str(challenge["pair_key"])); challenges.append(dict(challenge))
                target_age, window_actions = 0, 0
            achieved = achieved_after
        else:
            step = HORIZON - 1
    if provider.mutable and not terminal:
        provider.update()
    post_signature = _decision_signature(provider) if selection_mode == "ES" else ()
    moved = bool(selection_mode == "ES" and pre_signature != post_signature)
    pair_evidence: dict[str, bool] = {}
    for challenge_item in challenges:
        source, target_code = challenge_item["pair"]
        pair_evidence[str(challenge_item["pair_key"])] = bool(first_hits[source] is not None and first_hits[target_code] is not None and first_hits[source] < first_hits[target_code])
    if selection_mode == "ES":
        assert es_history is not None
        for challenge_item in challenges:
            key = str(challenge_item["pair_key"]); record = _history_record(es_history, tuple(challenge_item["pair"]))
            source, target_code = challenge_item["pair"]
            first_hit = bool(first_hits[source] is not None or first_hits[target_code] is not None)
            pair_hit = pair_evidence[key]
            record["attempts"] += 1; record["first_hit_yield"] += int(first_hit); record["discriminating_yield"] += int(pair_hit)
            record["consecutive_zero_yield"] = 0 if pair_hit else int(record["consecutive_zero_yield"]) + 1
            if record["consecutive_zero_yield"] >= int(PROTOCOL["pair_memory"]["zero_yield_attempts_before_cooldown"]):
                record["cooldown_until_world"] = world + int(PROTOCOL["pair_memory"]["cooldown_worlds"])
                record["consecutive_zero_yield"] = 0
            edge_source, edge_target = challenge_item["evidence_edge"]
            edge_after = provider.graph.probability(TARGETS.index(edge_source), TARGETS.index(edge_target))
            # The provider blocks only at p > 0.5, so reaching exactly 0.5 is
            # already a decision-boundary crossing and should retire the pair.
            threshold_crossed = (
                float(challenge_item["evidence_edge_probability_before"]) < 0.5
                and edge_after >= 0.5
            )
            if moved and pair_hit and threshold_crossed:
                record["retired"] = True
    depth = _canonical_depth(achieved)
    return {
        "step_count": step + 1, "stop_reason": stop_reason, "final_achievements": achieved,
        "first_hit_steps": first_hits, "first_hit_order": hit_order, "canonical_prefix_depth": depth,
        "R": float(depth / len(TARGETS)), "Q": float(sum(bool(achieved[x]) for x in TARGETS) / len(TARGETS)),
        "full_chain": int(depth == len(TARGETS)), "E_hit": bool(achieved["E"]), "native_return": float(native_return),
        "target_action_counts": target_counts, "optimizer_steps": 0, "mode": mode, "selection_mode": selection_mode,
        "es_challenges": challenges if selection_mode == "ES" else [], "es_pair_evidence": pair_evidence if selection_mode == "ES" else {},
        "es_provider_decision_movement": moved if selection_mode == "ES" else False,
        "private_state_absent_from_policy_inputs": True,
    }


def _load_previous_chain(root: Path, controller: int, stream: int, world: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if world == _worlds("acquisition")[0]:
        return ({arm: _initial_corrupt_graph() for arm in ACQUISITION_ARMS}, {})
    previous = _row_path(root, "acquisition", controller, stream, world - 1)
    if not previous.is_file():
        raise V13Error("acquisition must resume sequentially within a chain")
    value = json.loads(previous.read_text(encoding="utf-8"))
    graphs = {arm: v11.GraphState.from_json(value["graphs_after"][arm]) for arm in ACQUISITION_ARMS}
    return graphs, dict(value.get("es_history_after", {}))


def _early_summary(root: Path, controller: int, stream: int, world: int, history: Mapping[str, Any]) -> None:
    if world != int(PROTOCOL["soft_progress_policy"]["early_world_id"]):
        return
    pairs = sum(int(item.get("discriminating_yield", 0)) for item in history.values())
    movement = any(item.get("retired", False) for item in history.values())
    _atomic_json({"schema": SCHEMA, "controller": controller, "stream": stream, "world": world,
                  "es_pair_evidence": pairs, "provider_decision_movement": movement,
                  "interpretation": "CONTINUE_IF_CROSS_CHAIN_CONDITION_MET_ELSE_REVIEW_ONE_MINIMAL_REVISION"},
                 root / "progress" / f"controller_{controller}_stream_{stream}_world10.json")


def run_acquisition(*, controller: int, stream: int, output_root: Path, device_name: str, until_world: int | None) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    if controller not in CONTROLLER_SEEDS or stream not in STREAMS:
        raise V13Error("unknown controller/stream")
    device = torch.device(device_name)
    model = _final_checkpoint_model(checkpoint=_checkpoint(controller), controller_seed=controller, device=device)
    completed = 0
    for world in _worlds("acquisition"):
        if until_world is not None and world > until_world:
            break
        path = _row_path(output_root, "acquisition", controller, stream, world)
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("schema") != SCHEMA or old.get("identity") != [controller, stream, world]:
                raise V13Error("corrupt or mixed acquisition row")
            completed += 1; continue
        graphs, es_history = _load_previous_chain(output_root, controller, stream, world)
        target_u, action_u = _uniforms(controller, world, stream, 11), _uniforms(controller, world, stream, 12)
        _prefix, handoffs = open_arm_handoffs(world, ACQUISITION_ARMS)
        try:
            arms = {arm: _run_arm(model=model, handoff=handoffs[arm], provider=v11.TargetProvider(graphs[arm], mutable=True),
                                  device=device, target_uniforms=target_u, action_uniforms=action_u, mode=arm, world=world,
                                  es_history=es_history if arm == "ES" else None) for arm in ACQUISITION_ARMS}
        finally:
            for handoff in handoffs.values(): handoff.close()
        _atomic_json({"schema": SCHEMA, "phase": "acquisition", "identity": [controller, stream, world],
                      "paired_uniforms": {"target": [controller, world, stream, 11, 20260827, 13], "action": [controller, world, stream, 12, 20260827, 13]},
                      "arms": arms, "graphs_after": {arm: graphs[arm].to_json() for arm in ACQUISITION_ARMS},
                      "es_history_after": es_history}, path)
        _early_summary(output_root, controller, stream, world, es_history); completed += 1
    return {"status": "ACQUISITION_COMPLETE_OR_PAUSED", "controller": controller, "stream": stream, "rows": completed}


def _load_frozen(root: Path, controller: int, stream: int, arm: str) -> Any:
    path = _row_path(root, "acquisition", controller, stream, _worlds("acquisition")[-1])
    if not path.is_file(): raise V13Error("all 40 acquisition worlds are required before evaluation")
    return v11.GraphState.from_json(json.loads(path.read_text(encoding="utf-8"))["graphs_after"][arm])


def _tv(left_graph: Any, right_graph: Any) -> float:
    left, right = v11.TargetProvider(left_graph, mutable=False), v11.TargetProvider(right_graph, mutable=False)
    values = []
    for bits in itertools.product((False, True), repeat=len(TARGETS)):
        if all(bits): continue
        mask = {code: bool(bits[i]) for i, code in enumerate(TARGETS)}
        values.append(0.5 * sum(abs(a - b) for a, b in zip(left.distribution(mask, 0), right.distribution(mask, 0))))
    return float(fmean(values))


def run_evaluation(*, controller: int, stream: int, output_root: Path, device_name: str, world_start: int, world_end: int) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    worlds = [w for w in _worlds("evaluation") if world_start <= w <= world_end]
    if not worlds: raise V13Error("empty evaluation shard")
    device = torch.device(device_name)
    model = _final_checkpoint_model(checkpoint=_checkpoint(controller), controller_seed=controller, device=device)
    frozen = {"O": _initial_eval_graph("O"), "F": _initial_eval_graph("F"), **{a: _load_frozen(output_root, controller, stream, a) for a in ACQUISITION_ARMS}}
    completed = 0
    for world in worlds:
        path = _row_path(output_root, "evaluation", controller, stream, world)
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("schema") != SCHEMA or old.get("identity") != [controller, stream, world]: raise V13Error("corrupt evaluation row")
            completed += 1; continue
        target_u, action_u = _uniforms(controller, world, stream, 21), _uniforms(controller, world, stream, 22)
        _prefix, handoffs = open_arm_handoffs(world, EVALUATION_ARMS)
        try:
            arms = {arm: _run_arm(model=model, handoff=handoffs[arm], provider=v11.TargetProvider(copy.deepcopy(frozen[arm]), mutable=False),
                                  device=device, target_uniforms=target_u, action_uniforms=action_u, mode=arm, world=world, es_history=None,
                                  selection_mode="ordinary")
                    for arm in EVALUATION_ARMS}
        finally:
            for handoff in handoffs.values(): handoff.close()
        _atomic_json({"schema": SCHEMA, "phase": "evaluation", "identity": [controller, stream, world], "arms": arms,
                      "optimizer_steps": 0, "graphs_frozen": True, "evidence_rules_absent": True,
                      "target_tv_to_O": {arm: _tv(frozen[arm], frozen["O"]) for arm in ("N", "ES")}}, path)
        completed += 1
    return {"status": "EVALUATION_SHARD_COMPLETE", "controller": controller, "stream": stream, "rows": completed}


def _mean_ci(values: Sequence[float], critical: float) -> dict[str, Any]:
    mean = fmean(values)
    if len(values) < 2: return {"estimate": mean, "n": len(values), "lower": mean, "upper": mean}
    se = stdev(values) / math.sqrt(len(values))
    return {"estimate": mean, "n": len(values), "se": se, "lower": mean - critical * se, "upper": mean + critical * se}


def run_merge(*, output_root: Path) -> dict[str, Any]:
    acq, evaluation = [], []
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            for phase, sink in (("acquisition", acq), ("evaluation", evaluation)):
                for world in _worlds(phase):
                    path = _row_path(output_root, phase, controller, stream, world)
                    if not path.is_file(): raise V13Error(f"missing {phase} row: {path}")
                    sink.append(json.loads(path.read_text(encoding="utf-8")))
    world_effects = {}
    for world in _worlds("evaluation"):
        rows = [row for row in evaluation if row["identity"][2] == world]
        def contrast(left: str, right: str) -> float: return fmean(float(r["arms"][left]["R"]) - float(r["arms"][right]["R"]) for r in rows)
        world_effects[str(world)] = {"ES_minus_N": contrast("ES", "N"), "ES_minus_F": contrast("ES", "F"), "O_minus_F": contrast("O", "F")}
    chains = []
    for controller in CONTROLLER_SEEDS:
        for stream in STREAMS:
            rows = [r for r in acq if r["identity"][:2] == [controller, stream]]
            es_rows = [r["arms"]["ES"] for r in rows]
            final = rows[-1]
            chains.append({"controller": controller, "stream": stream,
                           "es_pair_evidence": sum(sum(bool(x) for x in r["es_pair_evidence"].values()) for r in es_rows),
                           "es_provider_movement": any(r["es_provider_decision_movement"] for r in es_rows),
                           "tv_N_to_O": _tv(v11.GraphState.from_json(final["graphs_after"]["N"]), _initial_eval_graph("O")),
                           "tv_ES_to_O": _tv(v11.GraphState.from_json(final["graphs_after"]["ES"]), _initial_eval_graph("O")),
                           "es_history": final["es_history_after"]})
    result = {"schema": SCHEMA, "phase": "merged_evaluation", "acquisition_rows": len(acq), "evaluation_rows": len(evaluation),
              "world_effects": world_effects,
              "behavior": {key: _mean_ci([v[key] for v in world_effects.values()], 1.9842169515) for key in next(iter(world_effects.values()))},
              "chains": chains,
              "analysis_policy": "CI is descriptive only; early look is an operational resource decision, not an inferential gate."}
    _atomic_json(result, output_root / "result.json")
    return {"status": "MERGE_COMPLETE", "output": str(output_root / "result.json"), "chains": len(chains)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "evaluate", "merge"))
    parser.add_argument("--controller", type=int); parser.add_argument("--stream", type=int, choices=STREAMS)
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--until-world", type=int); parser.add_argument("--world-start", type=int); parser.add_argument("--world-end", type=int)
    args = parser.parse_args(argv)
    if args.stage in {"acquire", "evaluate"} and (args.controller is None or args.stream is None): parser.error("controller and stream are required")
    if args.stage == "acquire": result = run_acquisition(controller=args.controller, stream=args.stream, output_root=args.output_root, device_name=args.device, until_world=args.until_world)
    elif args.stage == "evaluate":
        result = run_evaluation(controller=args.controller, stream=args.stream, output_root=args.output_root, device_name=args.device,
                                world_start=args.world_start or _worlds("evaluation")[0], world_end=args.world_end or _worlds("evaluation")[-1])
    else: result = run_merge(output_root=args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
