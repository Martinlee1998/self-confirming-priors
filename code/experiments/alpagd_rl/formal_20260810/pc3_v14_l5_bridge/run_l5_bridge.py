"""PC3-v14 retained-completion L5 bridge; no API client is present."""
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
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
NODES = tuple(PROTOCOL["l5_nodes"])
ACQ = ("L5_N", "L5_ES")
EVAL = ("O", "L5_0", "L5_N", "L5_ES")
SCHEMA = "pc3-v14-l5-bridge/v1"
HORIZON = int(PROTOCOL["horizon"])


class V14Error(RuntimeError):
    pass


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _worlds(phase: str) -> tuple[int, ...]:
    if phase == "evaluation" and "evaluation_world_ids" in PROTOCOL:
        values = tuple(int(value) for value in PROTOCOL["evaluation_world_ids"])
    else:
        low, high = PROTOCOL[f"{phase}_worlds"]
        values = tuple(range(int(low), int(high) + 1))
    if len(values) != (40 if phase == "acquisition" else 100):
        raise V14Error("unexpected world namespace")
    if len(values) != len(set(values)):
        raise V14Error("world namespace contains duplicates")
    return values


def _uniforms(controller: int, world: int, stream: int, kind: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([controller, world, stream, kind, 20260828, 14]))
    return np.asarray(rng.random(HORIZON + 1), dtype=np.float64)


def _source_root() -> Path:
    return Path(os.environ.get("PC3_L5_SOURCE_ROOT", str(HERE.parents[4])))


def _read_completion_q() -> tuple[dict[str, float], dict[str, list[float]]]:
    missing = float(PROTOCOL["missing_completion_edge_probability"])
    pairs = [(left, right) for left in NODES for right in NODES if left != right]
    values = {f"{left}->{right}": [] for left, right in pairs}
    for relative in PROTOCOL["source_completions"]:
        path = _source_root() / relative
        if not path.is_file():
            raise V14Error(f"retained completion unavailable: {path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        edges = {(str(edge["from_achievement"]), str(edge["to_achievement"])): float(edge["prior_probability"])
                 for edge in parsed["edge_priors"]}
        for left, right in pairs:
            values[f"{left}->{right}"].append(float(edges.get((left, right), missing)))
    aggregate = {key: float(fmean(items)) for key, items in values.items()}
    return aggregate, values


def verify_projection_sources() -> dict[str, Any]:
    aggregate, per_completion = _read_completion_q()
    frozen = {str(key): float(value) for key, value in PROTOCOL["qL"].items()}
    mismatch = {key: {"computed": aggregate.get(key), "frozen": frozen.get(key)} for key in set(aggregate) | set(frozen)
                if key not in aggregate or key not in frozen or not math.isclose(aggregate[key], frozen[key], abs_tol=1e-12)}
    if mismatch:
        raise V14Error(f"retained completion projection differs from protocol: {mismatch}")
    concentration = float(PROTOCOL["beta_concentration"])
    return {"status": "PROJECTION_MATCH", "source_completion_count": 3,
            "omission_probability": float(PROTOCOL["missing_completion_edge_probability"]),
            "qL": aggregate, "per_completion_values": per_completion,
            "beta_counts": {key: {"alpha": concentration * value, "beta": concentration * (1.0 - value)} for key, value in aggregate.items()}}


def _load_v13() -> Any:
    source = Path(os.environ.get("PC3_V13_SRC", str(HERE.parent / "pc3_v13_evidence_seeking")))
    path = source / "run_evidence_seeking.py"
    spec = importlib.util.spec_from_file_location("pc3_v13_for_l5_bridge", path)
    if spec is None or spec.loader is None:
        raise V14Error(f"cannot import v13 selector: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _l5_graph(v13: Any) -> Any:
    projection = verify_projection_sources()
    q = projection["qL"]; kappa = float(PROTOCOL["beta_concentration"])
    alpha, beta = [], []
    for left in NODES:
        for right in NODES:
            if left == right:
                alpha.append(1.0); beta.append(1.0)
            else:
                value = float(q[f"{left}->{right}"])
                alpha.append(kappa * value); beta.append(kappa * (1.0 - value))
    # Use frozen v11's allowed internal arm label; outer rows retain L5 labels.
    graph = v13.v11.GraphState("U", alpha, beta, [0.0] * 25, [0.0] * 25)
    graph.validate()
    return graph


def _checkpoint(v13: Any, controller: int) -> Path:
    return v13._checkpoint(controller)


def _row(root: Path, phase: str, controller: int, stream: int, world: int) -> Path:
    return root / phase / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"


def _load_previous(root: Path, v13: Any, controller: int, stream: int, world: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if world == _worlds("acquisition")[0]:
        return {arm: _l5_graph(v13) for arm in ACQ}, {}
    value = json.loads(_row(root, "acquisition", controller, stream, world - 1).read_text(encoding="utf-8"))
    return ({arm: v13.v11.GraphState.from_json(value["graphs_after"][arm]) for arm in ACQ},
            dict(value.get("es_history_after", {})))


def _write_projection_manifest(root: Path) -> None:
    path = root / "projection_manifest.json"
    if not path.exists():
        _atomic_json({"schema": SCHEMA, "projection": verify_projection_sources(),
                      "scope_boundary": PROTOCOL["scope_boundary"]}, path)


def run_acquisition(*, controller: int, stream: int, output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    v13 = _load_v13()
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    if controller not in v13.CONTROLLER_SEEDS or stream not in v13.STREAMS:
        raise V14Error("unknown controller/stream")
    _write_projection_manifest(output_root)
    model = v13._final_checkpoint_model(checkpoint=_checkpoint(v13, controller), controller_seed=controller, device=torch.device(device_name))
    complete = 0
    for world in _worlds("acquisition"):
        path = _row(output_root, "acquisition", controller, stream, world)
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("schema") != SCHEMA or old.get("identity") != [controller, stream, world]: raise V14Error("mixed acquisition row")
            complete += 1; continue
        graphs, history = _load_previous(output_root, v13, controller, stream, world)
        target_u, action_u = _uniforms(controller, world, stream, 11), _uniforms(controller, world, stream, 12)
        _prefix, handoffs = open_arm_handoffs(world, ACQ)
        try:
            arms = {
                "L5_N": v13._run_arm(model=model, handoff=handoffs["L5_N"], provider=v13.v11.TargetProvider(graphs["L5_N"], mutable=True), device=torch.device(device_name), target_uniforms=target_u, action_uniforms=action_u, mode="N", world=world, es_history=None),
                "L5_ES": v13._run_arm(model=model, handoff=handoffs["L5_ES"], provider=v13.v11.TargetProvider(graphs["L5_ES"], mutable=True), device=torch.device(device_name), target_uniforms=target_u, action_uniforms=action_u, mode="ES", world=world, es_history=history),
            }
        finally:
            for handoff in handoffs.values(): handoff.close()
        _atomic_json({"schema": SCHEMA, "phase": "acquisition", "identity": [controller, stream, world], "arms": arms,
                      "graphs_after": {arm: graphs[arm].to_json() for arm in ACQ}, "es_history_after": history,
                      "paired_uniforms": {"target": [controller, world, stream, 11, 20260828, 14], "action": [controller, world, stream, 12, 20260828, 14]}}, path)
        complete += 1
    return {"status": "ACQUISITION_COMPLETE", "controller": controller, "stream": stream, "rows": complete}


def _frozen(root: Path, v13: Any, controller: int, stream: int, arm: str) -> Any:
    path = _row(root, "acquisition", controller, stream, _worlds("acquisition")[-1])
    if not path.is_file(): raise V14Error("acquisition incomplete")
    return v13.v11.GraphState.from_json(json.loads(path.read_text(encoding="utf-8"))["graphs_after"][arm])


def _tv(v13: Any, left: Any, right: Any) -> float:
    return v13._tv(left, right)


def _mask_signature(v13: Any, graph: Any) -> dict[str, list[float]]:
    provider = v13.v11.TargetProvider(graph, mutable=False)
    signature = {}
    for bits in itertools.product((False, True), repeat=5):
        if all(bits):
            continue
        mask = {code: bool(bits[index]) for index, code in enumerate(v13.TARGETS)}
        signature["".join("1" if bit else "0" for bit in bits)] = [float(x) for x in provider.distribution(mask, 0)]
    return signature


def _threshold_movements(v13: Any, initial: Any, final: Any) -> list[dict[str, Any]]:
    movements = []
    for source in range(5):
        for target in range(5):
            if source == target:
                continue
            before, after = initial.probability(source, target), final.probability(source, target)
            if (before > 0.5) != (after > 0.5):
                movements.append({"edge": f"{NODES[source]}->{NODES[target]}", "before": before, "after": after})
    return movements


def run_evaluation(*, controller: int, stream: int, output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    v13 = _load_v13()
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    model = v13._final_checkpoint_model(checkpoint=_checkpoint(v13, controller), controller_seed=controller, device=torch.device(device_name))
    initial = _l5_graph(v13)
    frozen = {"O": v13._initial_eval_graph("O"), "L5_0": initial, "L5_N": _frozen(output_root, v13, controller, stream, "L5_N"), "L5_ES": _frozen(output_root, v13, controller, stream, "L5_ES")}
    complete = 0
    for world in _worlds("evaluation"):
        path = _row(output_root, "evaluation", controller, stream, world)
        if path.exists(): complete += 1; continue
        target_u, action_u = _uniforms(controller, world, stream, 21), _uniforms(controller, world, stream, 22)
        _prefix, handoffs = open_arm_handoffs(world, EVAL)
        try:
            arms = {arm: v13._run_arm(model=model, handoff=handoffs[arm], provider=v13.v11.TargetProvider(copy.deepcopy(frozen[arm]), mutable=False), device=torch.device(device_name), target_uniforms=target_u, action_uniforms=action_u, mode="O" if arm == "O" else "N", world=world, es_history=None, selection_mode="ordinary") for arm in EVAL}
        finally:
            for handoff in handoffs.values(): handoff.close()
        _atomic_json({"schema": SCHEMA, "phase": "evaluation", "identity": [controller, stream, world], "arms": arms, "graphs_frozen": True, "evidence_rules_absent": True, "optimizer_steps": 0,
                      "target_tv_to_O": {arm: _tv(v13, frozen[arm], frozen["O"]) for arm in ("L5_0", "L5_N", "L5_ES")}}, path)
        complete += 1
    return {"status": "EVALUATION_COMPLETE", "controller": controller, "stream": stream, "rows": complete}


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    mean = fmean(values)
    if len(values) < 2: return {"estimate": mean, "n": len(values), "lower": mean, "upper": mean}
    se = stdev(values) / math.sqrt(len(values))
    return {"estimate": mean, "n": len(values), "se": se, "lower": mean - 1.9842169515 * se, "upper": mean + 1.9842169515 * se}


def run_merge(*, output_root: Path) -> dict[str, Any]:
    v13 = _load_v13()
    acquisition = []
    evaluation = []
    for controller in (2026081911, 2026081912, 2026081913, 2026081914):
        for stream in (0, 1):
            for world in _worlds("acquisition"):
                path = _row(output_root, "acquisition", controller, stream, world)
                if not path.is_file(): raise V14Error(f"missing acquisition row: {path}")
                acquisition.append(json.loads(path.read_text(encoding="utf-8")))
            for world in _worlds("evaluation"):
                path = _row(output_root, "evaluation", controller, stream, world)
                if not path.is_file(): raise V14Error(f"missing evaluation row: {path}")
                evaluation.append(json.loads(path.read_text(encoding="utf-8")))
    effects = {}
    for world in _worlds("evaluation"):
        rows = [item for item in evaluation if item["identity"][2] == world]
        def contrast(left: str, right: str) -> float: return fmean(float(row["arms"][left]["R"]) - float(row["arms"][right]["R"]) for row in rows)
        effects[str(world)] = {"L5_ES_minus_L5_N": contrast("L5_ES", "L5_N"), "L5_N_minus_L5_0": contrast("L5_N", "L5_0"), "L5_ES_minus_L5_0": contrast("L5_ES", "L5_0"), "O_minus_L5_0": contrast("O", "L5_0")}
    initial = _l5_graph(v13)
    chains = []
    for controller in v13.CONTROLLER_SEEDS:
        for stream in v13.STREAMS:
            acq_rows = [row for row in acquisition if row["identity"][:2] == [controller, stream]]
            eval_rows = [row for row in evaluation if row["identity"][:2] == [controller, stream]]
            final = acq_rows[-1]
            final_n = v13.v11.GraphState.from_json(final["graphs_after"]["L5_N"])
            final_es = v13.v11.GraphState.from_json(final["graphs_after"]["L5_ES"])
            mean_r = {arm: fmean(float(row["arms"][arm]["R"]) for row in eval_rows) for arm in EVAL}
            chains.append({"controller": controller, "stream": stream,
                           "real_pair_evidence": sum(sum(bool(value) for value in row["arms"]["L5_ES"]["es_pair_evidence"].values()) for row in acq_rows),
                           "es_provider_decision_movement": any(row["arms"]["L5_ES"]["es_provider_decision_movement"] for row in acq_rows),
                           "threshold_movements": {"L5_N": _threshold_movements(v13, initial, final_n), "L5_ES": _threshold_movements(v13, initial, final_es)},
                           "tv_to_O": {"L5_0": float(eval_rows[0]["target_tv_to_O"]["L5_0"]), "L5_N": float(eval_rows[0]["target_tv_to_O"]["L5_N"]), "L5_ES": float(eval_rows[0]["target_tv_to_O"]["L5_ES"])},
                           "mean_R": mean_r,
                           "behavior_contrasts": {"L5_ES_minus_L5_N": mean_r["L5_ES"] - mean_r["L5_N"], "L5_N_minus_L5_0": mean_r["L5_N"] - mean_r["L5_0"], "L5_ES_minus_L5_0": mean_r["L5_ES"] - mean_r["L5_0"]},
                           "signature_31_masks": {"L5_0": _mask_signature(v13, initial), "L5_N": _mask_signature(v13, final_n), "L5_ES": _mask_signature(v13, final_es)},
                           "freeze_flags": {"graphs_frozen": all(bool(row["graphs_frozen"]) for row in eval_rows), "evidence_rules_absent": all(bool(row["evidence_rules_absent"]) for row in eval_rows), "optimizer_steps_zero": all(int(row["optimizer_steps"]) == 0 for row in eval_rows), "ordinary_selection_only": all(all(value["selection_mode"] == "ordinary" for value in row["arms"].values()) for row in eval_rows)}})
    arm_summary = {}
    for arm in EVAL:
        values = [row["arms"][arm] for row in evaluation]
        depths = {str(depth): sum(int(value["canonical_prefix_depth"]) == depth for value in values) for depth in range(6)}
        arm_summary[arm] = {"mean_R": fmean(float(value["R"]) for value in values), "mean_Q": fmean(float(value["Q"]) for value in values), "mean_native_return": fmean(float(value["native_return"]) for value in values), "E_hits": sum(bool(value["E_hit"]) for value in values), "full_chains": sum(int(value["full_chain"]) for value in values), "prefix_depth_counts": depths}
    result = {"schema": SCHEMA, "phase": "merged_evaluation", "acquisition_rows": len(acquisition), "evaluation_rows": len(evaluation), "world_effects": effects, "behavior": {key: _mean_ci([row[key] for row in effects.values()]) for key in next(iter(effects.values()))}, "arm_summary": arm_summary, "chains": chains, "scope_boundary": PROTOCOL["scope_boundary"], "analysis_policy": "descriptive intervals only; no CI or SESOI stop gate"}
    _atomic_json(result, output_root / "result.json")
    return {"status": "MERGE_COMPLETE", "output": str(output_root / "result.json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check-projection", "acquire", "evaluate", "merge")); parser.add_argument("--controller", type=int); parser.add_argument("--stream", type=int); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage in {"acquire", "evaluate"} and (args.controller is None or args.stream is None): parser.error("controller and stream are required")
    if args.stage == "check-projection": result = verify_projection_sources()
    elif args.stage == "acquire": result = run_acquisition(controller=args.controller, stream=args.stream, output_root=args.output_root, device_name=args.device)
    elif args.stage == "evaluate": result = run_evaluation(controller=args.controller, stream=args.stream, output_root=args.output_root, device_name=args.device)
    else: result = run_merge(output_root=args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
