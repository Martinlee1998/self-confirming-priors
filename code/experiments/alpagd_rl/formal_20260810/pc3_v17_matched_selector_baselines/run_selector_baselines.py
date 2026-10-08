"""PC3-v17: matched-random and uncertainty-only frozen-provider baselines.

This wrapper deliberately reuses v13's real handoff, frozen policy, first-hit
updater, row schema shape, and evaluation implementation.  Its only changed
component is how a common graph-private challenge frontier is selected.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from typing import Any, Iterator, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
V13_PATH = HERE.parent / "pc3_v13_evidence_seeking" / "run_evidence_seeking.py"
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
SCHEMA = "pc3-v17-selector-baselines/v1"
ACQUISITION_ARMS = ("N", "R", "UQ", "ES")
EVALUATION_ARMS = ("O", "F", "N", "R", "UQ", "ES")
CHALLENGE_ARMS = frozenset({"R", "UQ", "ES"})


def _load_v13() -> Any:
    spec = importlib.util.spec_from_file_location("pc3_v13_frozen_for_v16", V13_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v13 implementation: {V13_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v13 = _load_v13()
from pc3_selectors import choose_es, choose_random, choose_uncertainty, common_frontier  # noqa: E402

# v13's real rollout body consults these module globals for arm validation and
# pair-memory bookkeeping.  They are rebound only in this process; v13 files
# and results remain untouched.
v13.ACQUISITION_ARMS = ACQUISITION_ARMS
v13.EVALUATION_ARMS = EVALUATION_ARMS
v13.PROTOCOL = PROTOCOL


class V16Error(v13.V13Error):
    pass


def _worlds(phase: str) -> tuple[int, ...]:
    low, high = PROTOCOL[f"{phase}_worlds"]
    return tuple(range(int(low), int(high) + 1))


def _row_path(root: Path, phase: str, controller: int, stream: int, world: int) -> Path:
    return root / phase / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"


def _selector_uniforms(controller: int, world: int, stream: int) -> np.ndarray:
    """Independent preregistered selector/tie stream; never feeds provider.select."""
    seed = np.random.SeedSequence([controller, world, stream, 31, 20260828, 16])
    return np.asarray(np.random.default_rng(seed).random(v13.HORIZON + 1), dtype=np.float64)


_ACTIVE: dict[str, Any] | None = None


def _select_target(provider: Any, achieved: dict[str, bool], target_uniform: float, mode: str,
                   history: dict[str, Any], world: int, used_masks: set[str], used_pairs: set[str]) -> tuple[str | None, dict[str, Any] | None, bool]:
    """Hook for v13._run_arm. Ordinary fallback retains its original uniform."""
    if mode != "ES" or _ACTIVE is None:
        return provider.select(achieved, 0, target_uniform), None, False
    selector_uniforms = _ACTIVE["selector_uniforms"]
    index = _ACTIVE["selection_index"]
    _ACTIVE["selection_index"] += 1
    candidates = common_frontier(provider=provider, targets=v13.TARGETS, achieved=achieved, history=history,
                                 world=world, used_masks=used_masks, used_pairs=used_pairs,
                                 max_challenges=int(PROTOCOL["pair_memory"]["max_challenges_per_world"]))
    strategy = _ACTIVE["strategy"]
    selector_uniform = float(selector_uniforms[min(index, v13.HORIZON)])
    if strategy == "R":
        challenge = choose_random(candidates, selector_uniform)
    elif strategy == "UQ":
        challenge = choose_uncertainty(candidates, selector_uniform)
    elif strategy == "ES":
        challenge = choose_es(candidates=candidates, provider=provider, targets=v13.TARGETS, achieved=achieved,
                              uniform=selector_uniform, edge_index=v13.v11._edge_index,
                              target_provider=v13.v11.TargetProvider)
    else:
        raise V16Error(f"unknown challenge strategy: {strategy}")
    if challenge is not None:
        challenge = dict(challenge)
        challenge["selector_strategy"] = strategy
        return str(challenge["probe_target"]), challenge, True
    return provider.select(achieved, 0, target_uniform), None, False


v13._select_target = _select_target


@contextmanager
def _strategy(strategy: str, selector_uniforms: np.ndarray) -> Iterator[None]:
    global _ACTIVE
    if _ACTIVE is not None:
        raise V16Error("nested selector strategy context")
    _ACTIVE = {"strategy": strategy, "selector_uniforms": selector_uniforms, "selection_index": 0}
    try:
        yield
    finally:
        _ACTIVE = None


def _initial_graph() -> Any:
    return v13.v11.initial_graph("U", "P2")


def _load_previous(root: Path, controller: int, stream: int, world: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if world == _worlds("acquisition")[0]:
        return ({arm: _initial_graph() for arm in ACQUISITION_ARMS}, {arm: {} for arm in CHALLENGE_ARMS})
    prior = _row_path(root, "acquisition", controller, stream, world - 1)
    if not prior.is_file():
        raise V16Error("acquisition must resume sequentially within a chain")
    row = json.loads(prior.read_text(encoding="utf-8"))
    if row.get("schema") != SCHEMA:
        raise V16Error("prior row is not a v16 acquisition row")
    graphs = {arm: v13.v11.GraphState.from_json(row["graphs_after"][arm]) for arm in ACQUISITION_ARMS}
    histories = {arm: dict(row.get("challenge_histories_after", {}).get(arm, {})) for arm in CHALLENGE_ARMS}
    return graphs, histories


def _run_acquisition_arm(*, arm: str, model: Any, handoff: Any, graph: Any, device: Any,
                         target_uniforms: np.ndarray, action_uniforms: np.ndarray, selector_uniforms: np.ndarray,
                         world: int, history: dict[str, Any] | None) -> dict[str, Any]:
    if arm == "N":
        row = v13._run_arm(model=model, handoff=handoff, provider=v13.v11.TargetProvider(graph, mutable=True), device=device,
                           target_uniforms=target_uniforms, action_uniforms=action_uniforms, mode="N", world=world,
                           es_history=None)
    else:
        # Passing ES into the frozen v13 rollout body retains its pair-memory
        # and first-hit update mechanics; the hook above selects R/UQ/ES.
        with _strategy(arm, selector_uniforms):
            row = v13._run_arm(model=model, handoff=handoff, provider=v13.v11.TargetProvider(graph, mutable=True), device=device,
                               target_uniforms=target_uniforms, action_uniforms=action_uniforms, mode=arm, world=world,
                               es_history=history, selection_mode="ES")
        row["mode"], row["selection_mode"] = arm, arm
        row["challenges"] = row.pop("es_challenges")
        row["pair_evidence"] = row.pop("es_pair_evidence")
        row["provider_decision_movement"] = row.pop("es_provider_decision_movement")
    return row


def run_acquisition(*, controller: int, stream: int, output_root: Path, device_name: str, until_world: int | None) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    if controller not in v13.CONTROLLER_SEEDS or stream not in v13.STREAMS:
        raise V16Error("unknown controller or stream")
    model = v13._final_checkpoint_model(checkpoint=v13._checkpoint(controller), controller_seed=controller, device=torch.device(device_name))
    completed = 0
    for world in _worlds("acquisition"):
        if until_world is not None and world > until_world:
            break
        path = _row_path(output_root, "acquisition", controller, stream, world)
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")).get("schema") != SCHEMA:
                raise V16Error("corrupt or mixed acquisition row")
            completed += 1; continue
        graphs, histories = _load_previous(output_root, controller, stream, world)
        target_u, action_u = v13._uniforms(controller, world, stream, 11), v13._uniforms(controller, world, stream, 12)
        selector_u = _selector_uniforms(controller, world, stream)
        _prefix, handoffs = open_arm_handoffs(world, ACQUISITION_ARMS)
        try:
            arms = {arm: _run_acquisition_arm(arm=arm, model=model, handoff=handoffs[arm], graph=graphs[arm], device=torch.device(device_name),
                                               target_uniforms=target_u, action_uniforms=action_u, selector_uniforms=selector_u, world=world,
                                               history=histories.get(arm)) for arm in ACQUISITION_ARMS}
        finally:
            for handoff in handoffs.values(): handoff.close()
        v13._atomic_json({"schema": SCHEMA, "phase": "acquisition", "identity": [controller, stream, world],
                          "paired_uniforms": {"target": [controller, world, stream, 11, 20260827, 13], "action": [controller, world, stream, 12, 20260827, 13], "selector": [controller, world, stream, 31, 20260828, 16]},
                          "arms": arms, "graphs_after": {arm: graphs[arm].to_json() for arm in ACQUISITION_ARMS},
                          "challenge_histories_after": histories}, path)
        completed += 1
    return {"status": "ACQUISITION_COMPLETE_OR_PAUSED", "controller": controller, "stream": stream, "rows": completed}


def _load_frozen(root: Path, controller: int, stream: int, arm: str) -> Any:
    path = _row_path(root, "acquisition", controller, stream, _worlds("acquisition")[-1])
    if not path.is_file(): raise V16Error("all acquisition worlds are required before evaluation")
    return v13.v11.GraphState.from_json(json.loads(path.read_text(encoding="utf-8"))["graphs_after"][arm])


def run_evaluation(*, controller: int, stream: int, output_root: Path, device_name: str, world_start: int, world_end: int) -> dict[str, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    worlds = [w for w in _worlds("evaluation") if world_start <= w <= world_end]
    if not worlds: raise V16Error("empty evaluation shard")
    device = torch.device(device_name)
    model = v13._final_checkpoint_model(checkpoint=v13._checkpoint(controller), controller_seed=controller, device=device)
    frozen = {"O": v13._initial_eval_graph("O"), "F": v13._initial_eval_graph("F"), **{arm: _load_frozen(output_root, controller, stream, arm) for arm in ACQUISITION_ARMS}}
    completed = 0
    for world in worlds:
        path = _row_path(output_root, "evaluation", controller, stream, world)
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")).get("schema") != SCHEMA: raise V16Error("corrupt or mixed evaluation row")
            completed += 1; continue
        target_u, action_u = v13._uniforms(controller, world, stream, 21), v13._uniforms(controller, world, stream, 22)
        _prefix, handoffs = open_arm_handoffs(world, EVALUATION_ARMS)
        try:
            arms = {arm: v13._run_arm(model=model, handoff=handoffs[arm], provider=v13.v11.TargetProvider(copy.deepcopy(frozen[arm]), mutable=False), device=device,
                                       target_uniforms=target_u, action_uniforms=action_u, mode=arm, world=world, es_history=None,
                                       selection_mode="ordinary") for arm in EVALUATION_ARMS}
        finally:
            for handoff in handoffs.values(): handoff.close()
        v13._atomic_json({"schema": SCHEMA, "phase": "evaluation", "identity": [controller, stream, world], "arms": arms,
                          "optimizer_steps": 0, "graphs_frozen": True, "challenge_rules_absent": True}, path)
        completed += 1
    return {"status": "EVALUATION_SHARD_COMPLETE", "controller": controller, "stream": stream, "rows": completed}


def run_merge(*, output_root: Path) -> dict[str, Any]:
    rows = []
    for controller in v13.CONTROLLER_SEEDS:
        for stream in v13.STREAMS:
            for world in _worlds("evaluation"):
                path = _row_path(output_root, "evaluation", controller, stream, world)
                if not path.is_file(): raise V16Error(f"missing evaluation row: {path}")
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    effects = {str(world): {f"{arm}_minus_N": fmean(float(row["arms"][arm]["R"]) - float(row["arms"]["N"]["R"]) for row in rows if row["identity"][2] == world) for arm in ("R", "UQ", "ES")} for world in _worlds("evaluation")}
    result = {"schema": SCHEMA, "phase": "merged_evaluation", "evaluation_rows": len(rows), "world_effects": effects,
              "behavior": {name: {"estimate": fmean(effect[name] for effect in effects.values()), "n": len(effects)} for name in next(iter(effects.values()))},
              "analysis_policy": "All complete valid arms are reported. Intervals and effect sizes describe uncertainty; they are not selection gates."}
    v13._atomic_json(result, output_root / "result.json")
    return {"status": "MERGE_COMPLETE", "output": str(output_root / "result.json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "evaluate", "merge")); parser.add_argument("--controller", type=int); parser.add_argument("--stream", type=int, choices=v13.STREAMS)
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--until-world", type=int); parser.add_argument("--world-start", type=int); parser.add_argument("--world-end", type=int)
    args = parser.parse_args(argv)
    if args.stage in {"acquire", "evaluate"} and (args.controller is None or args.stream is None): parser.error("controller and stream are required")
    if args.stage == "acquire": result = run_acquisition(controller=args.controller, stream=args.stream, output_root=args.output_root, device_name=args.device, until_world=args.until_world)
    elif args.stage == "evaluate": result = run_evaluation(controller=args.controller, stream=args.stream, output_root=args.output_root, device_name=args.device, world_start=args.world_start or _worlds("evaluation")[0], world_end=args.world_end or _worlds("evaluation")[-1])
    else: result = run_merge(output_root=args.output_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
