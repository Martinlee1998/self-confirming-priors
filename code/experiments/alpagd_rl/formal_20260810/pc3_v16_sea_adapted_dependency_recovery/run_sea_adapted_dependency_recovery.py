"""PC3-v16 SEA-adapted / SEA-style achievement-labelled dependency recovery.

This intentionally does *not* invoke official SEA.  It is a compact mechanism
baseline: infer directed precedence only from public first-hit labels in retained
acquisition records, then freeze the recovered graph for the existing provider
and fresh-world evaluation interfaces.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
V11_DEFAULT = HERE.parent / "pc3_v11_graph_recovery" / "run_graph_recovery.py"
V15_DEFAULT = HERE.parent / "pc3_v15_p1_evidence_seeking" / "run_p1_evidence_seeking.py"
V13_DEFAULT = HERE.parent / "pc3_v13_evidence_seeking" / "run_evidence_seeking.py"
FORBIDDEN = frozenset(PROTOCOL["forbidden_inputs"])
FIXTURES = {"p1": ("PC3_V15_SRC", V15_DEFAULT), "p2": ("PC3_V13_SRC", V13_DEFAULT)}


class SeaAdaptedError(RuntimeError):
    pass


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SeaAdaptedError(f"cannot import compatibility module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _v11() -> Any:
    path = Path(os.environ.get("PC3_V11_SRC", str(V11_DEFAULT))) / "run_graph_recovery.py"
    if path == V11_DEFAULT / "run_graph_recovery.py":  # Default is already the file.
        path = V11_DEFAULT
    return _load_module("pc3_v11_for_sea_adapted", path)


def _runner(fixture: str, runner_file: Path | None = None) -> Any:
    if fixture not in FIXTURES:
        raise SeaAdaptedError("fixture must be p1 or p2")
    env_name, default = FIXTURES[fixture]
    if runner_file is None:
        configured = Path(os.environ.get(env_name, str(default)))
        path = configured if configured.suffix == ".py" else configured / default.name
    else:
        path = runner_file
    if not path.is_file():
        raise SeaAdaptedError(f"fixture runner is missing: {path}")
    return _load_module(f"pc3_{fixture}_runner_for_sea_adapted", path)


def _rows(root: Path, controller: int, stream: int) -> Iterable[Path]:
    # Rows are selected only by the declared chain identity, never by a
    # corruption descriptor or other semantic label.
    yield from sorted((root / "acquisition" / f"controller_{controller}" / f"stream_{stream}").glob("world_*.json"))


def _public_orders(root: Path, source_arm: str, targets: Sequence[str], controller: int, stream: int,
                   expected_worlds: Sequence[int]) -> list[tuple[str, ...]]:
    files = list(_rows(root, controller, stream))
    if len(files) != len(expected_worlds):
        raise SeaAdaptedError(f"expected exactly {len(expected_worlds)} acquisition rows for one chain; found {len(files)}")
    orders: list[tuple[str, ...]] = []
    found_worlds: list[int] = []
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        identity = row.get("identity")
        if identity is None or list(identity[:2]) != [controller, stream]:
            raise SeaAdaptedError(f"cross-chain or malformed acquisition identity: {path}")
        try:
            world = int(identity[2])
        except (IndexError, TypeError, ValueError) as exc:
            raise SeaAdaptedError(f"malformed acquisition identity: {path}") from exc
        found_worlds.append(world)
        try:
            order = row["arms"][source_arm]["first_hit_order"]
        except (KeyError, TypeError) as exc:
            raise SeaAdaptedError(f"missing public first_hit_order for arm {source_arm}: {path}") from exc
        if not isinstance(order, list) or len(order) != len(set(order)) or any(x not in targets for x in order):
            raise SeaAdaptedError(f"invalid public achievement order: {path}")
        orders.append(tuple(str(x) for x in order))
    if tuple(sorted(found_worlds)) != tuple(expected_worlds) or len(set(found_worlds)) != len(found_worlds):
        raise SeaAdaptedError("acquisition rows do not exactly match this fixture's 40-world chain namespace")
    return orders


def recover_graph_from_orders(orders: Sequence[Sequence[str]], v11: Any) -> Any:
    """Recover independent directed precedence posteriors from labelled orders.

    The constructor is intentionally data-only: no corrupted graph is used as a
    seed.  A pair contributes only when both achievements occurred in the same
    acquisition world; unseen directions retain the symmetric Beta(1,1) prior.
    """
    targets = tuple(v11.TARGETS)
    state = v11.initial_graph("D", "P1")  # D is an uninformative Beta(1,1) carrier.
    state.arm = "D"  # Internal type constraint; output metadata carries SEA-adapted identity.
    forward = [0.0] * (len(targets) ** 2)
    reverse = [0.0] * (len(targets) ** 2)
    for order in orders:
        unique = tuple(order)
        if len(unique) != len(set(unique)) or any(code not in targets for code in unique):
            raise SeaAdaptedError("orders must contain unique registered achievement labels")
        for later_index, later in enumerate(unique):
            for earlier in unique[:later_index]:
                edge = v11._edge_index(targets.index(earlier), targets.index(later))
                opposite = v11._edge_index(targets.index(later), targets.index(earlier))
                forward[edge] += 1.0
                reverse[opposite] += 1.0
    for edge in range(len(state.alpha)):
        state.alpha[edge] = 1.0 + forward[edge]
        state.beta[edge] = 1.0 + reverse[edge]
    state.pending_positive = [0.0] * len(state.alpha)
    state.pending_counter = [0.0] * len(state.alpha)
    state.update_count = 0
    state.first_hit_count = sum(len(order) for order in orders)
    state.validate()
    return state


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def recover(*, acquisition_root: Path, output: Path, source_arm: str, fixture: str,
            controller: int, stream: int, runner_file: Path | None) -> dict[str, Any]:
    v11, runner = _v11(), _runner(fixture, runner_file)
    expected_worlds = runner._worlds("acquisition")
    if len(expected_worlds) != 40:
        raise SeaAdaptedError("fixture no longer exposes a 40-world acquisition chain")
    orders = _public_orders(acquisition_root, source_arm, tuple(v11.TARGETS), controller, stream, expected_worlds)
    graph = recover_graph_from_orders(orders, v11)
    artifact = {
        "schema": PROTOCOL["schema"], "method_label": PROTOCOL["method_label"],
        "official_sea_claim": PROTOCOL["official_sea_claim"], "fixture": fixture,
        "identity": [controller, stream], "source_arm": source_arm,
        "input_fields_read": ["arms.<source_arm>.first_hit_order"],
        "forbidden_inputs_not_read": sorted(FORBIDDEN), "acquisition_root": str(acquisition_root),
        "rows_consumed": len(orders), "labelled_orders_with_any_hit": sum(bool(x) for x in orders),
        "graph": graph.to_json(), "graph_frozen": True,
    }
    _atomic_json(artifact, output)
    return {"status": "RECOVERED_FROZEN_SEA_ADAPTED_GRAPH", "rows": len(orders), "output": str(output)}


def evaluate(*, recovered_graph: Path, output_root: Path, controller: int, stream: int, fixture: str,
             device: str, world_start: int | None, world_end: int | None, runner_file: Path | None) -> dict[str, Any]:
    """Use the retained fresh-world runner with ordinary frozen selection only."""
    v11, runner = _v11(), _runner(fixture, runner_file)
    artifact = json.loads(recovered_graph.read_text(encoding="utf-8"))
    if artifact.get("schema") != PROTOCOL["schema"] or artifact.get("method_label") != PROTOCOL["method_label"]:
        raise SeaAdaptedError("recovered graph artifact is not a SEA-adapted artifact")
    if artifact.get("fixture") != fixture or artifact.get("identity") != [controller, stream]:
        raise SeaAdaptedError("recovered graph fixture/identity does not match requested evaluation chain")
    graph = v11.GraphState.from_json(artifact["graph"])
    if not artifact.get("graph_frozen"):
        raise SeaAdaptedError("evaluation requires a frozen recovered graph")
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import open_arm_handoffs
    worlds = [w for w in runner._worlds("evaluation") if (world_start is None or w >= world_start) and (world_end is None or w <= world_end)]
    if not worlds:
        raise SeaAdaptedError("empty evaluation shard")
    model = runner._final_checkpoint_model(checkpoint=runner._checkpoint(controller), controller_seed=controller, device=torch.device(device))
    completed = 0
    for world in worlds:
        path = output_root / "evaluation" / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"
        if path.exists():
            completed += 1; continue
        _prefix, handoffs = open_arm_handoffs(world, ("SEA-A",))
        try:
            target_u, action_u = runner._uniforms(controller, world, stream, 21), runner._uniforms(controller, world, stream, 22)
            arm = runner._run_arm(model=model, handoff=handoffs["SEA-A"], provider=v11.TargetProvider(copy.deepcopy(graph), mutable=False),
                               device=torch.device(device), target_uniforms=target_u, action_uniforms=action_u,
                               mode="N", world=world, es_history=None, selection_mode="ordinary")
        finally:
            for handoff in handoffs.values(): handoff.close()
        arm["mode"] = "SEA-A"
        _atomic_json({"schema": PROTOCOL["schema"], "method_label": PROTOCOL["method_label"], "phase": "evaluation",
                      "fixture": fixture, "identity": [controller, stream, world], "arms": {"SEA-A": arm}, "graphs_frozen": True,
                      "evidence_rules_absent": True, "optimizer_steps": 0, "selection_mode": "ordinary"}, path)
        completed += 1
    return {"status": "EVALUATION_SHARD_COMPLETE", "rows": completed, "method_label": PROTOCOL["method_label"]}


def check() -> dict[str, Any]:
    v11 = _v11()
    graph = recover_graph_from_orders((("A", "B", "C"), ("A", "B"), ("B", "C")), v11)
    a, b = v11.TARGETS.index("A"), v11.TARGETS.index("B")
    if graph.probability(a, b) <= 0.5 or graph.probability(b, a) >= 0.5:
        raise SeaAdaptedError("synthetic labelled-order recovery check failed")
    return {"status": "STATIC_AND_SYNTHETIC_CHECK_PASS", "method_label": PROTOCOL["method_label"], "behavior_rows_executed": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("recover", "evaluate", "check"))
    parser.add_argument("--acquisition-root", type=Path); parser.add_argument("--recovered-graph", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-arm", default=PROTOCOL["source_arm"]); parser.add_argument("--fixture", choices=tuple(FIXTURES))
    parser.add_argument("--runner-file", type=Path); parser.add_argument("--controller", type=int)
    parser.add_argument("--stream", type=int, choices=(0, 1)); parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-start", type=int); parser.add_argument("--world-end", type=int)
    args = parser.parse_args(argv)
    if args.stage == "check": result = check()
    elif args.stage == "recover":
        if None in (args.acquisition_root, args.output, args.fixture, args.controller, args.stream): parser.error("recover requires acquisition root, output, fixture, controller, and stream")
        result = recover(acquisition_root=args.acquisition_root, output=args.output, source_arm=args.source_arm, fixture=args.fixture,
                         controller=args.controller, stream=args.stream, runner_file=args.runner_file)
    else:
        if None in (args.recovered_graph, args.output_root, args.fixture, args.controller, args.stream): parser.error("evaluate requires recovered graph, output root, fixture, controller, and stream")
        result = evaluate(recovered_graph=args.recovered_graph, output_root=args.output_root, controller=args.controller, stream=args.stream,
                          fixture=args.fixture, device=args.device, world_start=args.world_start, world_end=args.world_end,
                          runner_file=args.runner_file)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
