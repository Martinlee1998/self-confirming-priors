"""Merge and analyse the PC3 matched repair-strategy comparison.

This is an analysis-only program.  It consumes completed JSON rows from the
selector experiment, the retained PC3 fixture, and the SEA-adapted baseline;
it never opens an environment or changes a graph.  The output keeps chain and
world aggregation distinct: chains describe acquisition/recovery and worlds
describe behaviour conditional on the acquired frozen graphs.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence


TARGETS = ("A", "B", "C", "D", "E")
CONTROLLERS = (2026081911, 2026081912, 2026081913, 2026081914)
STREAMS = (0, 1)
N_CHAINS = len(CONTROLLERS) * len(STREAMS)
N_ACQUISITION = 40
N_EVALUATION = 100
EPSILON = 1e-12


class ComparativeAnalysisError(RuntimeError):
    """Raised when an input is not a complete, frozen PC3 result."""


@dataclass(frozen=True)
class Graph:
    alpha: tuple[float, ...]
    beta: tuple[float, ...]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Graph":
        try:
            alpha = tuple(float(x) for x in value["alpha"])
            beta = tuple(float(x) for x in value["beta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ComparativeAnalysisError("malformed graph state") from exc
        size = len(TARGETS) ** 2
        if len(alpha) != size or len(beta) != size:
            raise ComparativeAnalysisError("graph state has the wrong edge count")
        if any(not math.isfinite(x) or x < 0.0 for x in (*alpha, *beta)):
            raise ComparativeAnalysisError("graph state contains a non-finite value")
        if any(a + b <= 0.0 for a, b in zip(alpha, beta)):
            raise ComparativeAnalysisError("graph state has an undefined probability")
        return cls(alpha=alpha, beta=beta)

    def probability(self, source: int, target: int) -> float:
        index = source * len(TARGETS) + target
        return self.alpha[index] / (self.alpha[index] + self.beta[index])


def _correct_graph() -> Graph:
    """The design-defined correct graph, independent of fixture corruption."""
    alpha, beta = [], []
    for source in range(len(TARGETS)):
        for target in range(len(TARGETS)):
            active = source < target
            alpha.append(9.0 if active else 1.0)
            beta.append(1.0 if active else 9.0)
    return Graph(tuple(alpha), tuple(beta))


def _fixed_graph(fixture: str) -> Graph:
    if fixture not in {"p1", "p2"}:
        raise ComparativeAnalysisError("fixture must be p1 or p2")
    deleted, inserted = (("A", "B"), ("B", "A")) if fixture == "p1" else (("B", "C"), ("C", "B"))
    active = {(source, target) for source in range(5) for target in range(5) if source < target}
    active.remove((TARGETS.index(deleted[0]), TARGETS.index(deleted[1])))
    active.add((TARGETS.index(inserted[0]), TARGETS.index(inserted[1])))
    alpha, beta = [], []
    for source in range(5):
        for target in range(5):
            alpha.append(9.0 if (source, target) in active else 1.0)
            beta.append(1.0 if (source, target) in active else 9.0)
    return Graph(tuple(alpha), tuple(beta))


def _distribution(graph: Graph, achieved: Mapping[str, bool]) -> tuple[float, ...]:
    unfinished = [index for index, name in enumerate(TARGETS) if not achieved[name]]
    if not unfinished:
        return (0.0,) * len(TARGETS)
    eligible = []
    for target in unfinished:
        blocked = any(source != target and graph.probability(source, target) > 0.5 and not achieved[TARGETS[source]]
                      for source in range(len(TARGETS)))
        if not blocked:
            eligible.append(target)
    if not eligible:
        obstruction = {target: sum(graph.probability(source, target) for source in unfinished if source != target)
                       for target in unfinished}
        minimum = min(obstruction.values())
        eligible = [target for target in unfinished if math.isclose(obstruction[target], minimum, abs_tol=EPSILON)]
    mass = 1.0 / len(eligible)
    return tuple(mass if index in eligible else 0.0 for index in range(len(TARGETS)))


def _provider_signature(graph: Graph) -> tuple[tuple[float, ...], ...]:
    signatures = []
    for bits in itertools.product((False, True), repeat=len(TARGETS)):
        if all(bits):
            continue
        achieved = {name: bool(bits[index]) for index, name in enumerate(TARGETS)}
        signatures.append(tuple(round(value, 12) for value in _distribution(graph, achieved)))
    if len(signatures) != 31:
        raise ComparativeAnalysisError("provider mask enumeration did not produce 31 masks")
    return tuple(signatures)


def _tv(left: Graph, right: Graph) -> float:
    return float(fmean(0.5 * sum(abs(a - b) for a, b in zip(x, y))
                       for x, y in zip(_provider_signature(left), _provider_signature(right))))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparativeAnalysisError(f"cannot read JSON row: {path}") from exc
    if not isinstance(value, dict):
        raise ComparativeAnalysisError(f"JSON row is not an object: {path}")
    return value


def _chain_paths(root: Path, phase: str, controller: int, stream: int) -> list[Path]:
    folder = root / phase / f"controller_{controller}" / f"stream_{stream}"
    return sorted(folder.glob("world_*.json"))


def _identity(row: Mapping[str, Any], path: Path, phase: str, controller: int, stream: int) -> int:
    identity = row.get("identity")
    if not isinstance(identity, list) or len(identity) != 3 or identity[:2] != [controller, stream]:
        raise ComparativeAnalysisError(f"mixed or malformed identity in {path}")
    if row.get("phase") != phase:
        raise ComparativeAnalysisError(f"wrong phase in {path}")
    try:
        return int(identity[2])
    except (TypeError, ValueError) as exc:
        raise ComparativeAnalysisError(f"malformed world identity in {path}") from exc


def _finite_metric(value: Any, field: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparativeAnalysisError(f"{field} is missing or nonnumeric in {path}") from exc
    if not math.isfinite(number):
        raise ComparativeAnalysisError(f"{field} is non-finite in {path}")
    return number


def _validate_rows(root: Path, phase: str, arms: set[str], *, require_freeze: bool) -> dict[tuple[int, int, int], dict[str, Any]]:
    expected_count = N_ACQUISITION if phase == "acquisition" else N_EVALUATION
    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    schema: str | None = None
    reference_worlds: set[int] | None = None
    for controller in CONTROLLERS:
        for stream in STREAMS:
            files = _chain_paths(root, phase, controller, stream)
            if len(files) != expected_count:
                raise ComparativeAnalysisError(f"{root}: expected {expected_count} {phase} rows for chain {(controller, stream)}, found {len(files)}")
            worlds: set[int] = set()
            for path in files:
                row = _read_json(path)
                world = _identity(row, path, phase, controller, stream)
                if world in worlds:
                    raise ComparativeAnalysisError(f"duplicate world identity in {path}")
                worlds.add(world)
                if schema is None:
                    schema = str(row.get("schema"))
                elif str(row.get("schema")) != schema:
                    raise ComparativeAnalysisError(f"mixed schemas within {root}")
                row_arms = row.get("arms")
                if not isinstance(row_arms, dict) or not arms.issubset(row_arms):
                    raise ComparativeAnalysisError(f"missing expected arms in {path}")
                if phase == "evaluation":
                    if row.get("graphs_frozen") is not True or int(row.get("optimizer_steps", -1)) != 0:
                        raise ComparativeAnalysisError(f"evaluation is not frozen/no-optimizer in {path}")
                    if require_freeze and not (row.get("evidence_rules_absent", row.get("challenge_rules_absent", False)) is True):
                        raise ComparativeAnalysisError(f"evaluation rule flag missing in {path}")
                for arm in arms:
                    arm_row = row_arms[arm]
                    if not isinstance(arm_row, dict):
                        raise ComparativeAnalysisError(f"malformed arm {arm} in {path}")
                    for metric in ("R", "Q", "native_return"):
                        _finite_metric(arm_row.get(metric), metric, path)
                    if int(arm_row.get("optimizer_steps", 0)) != 0:
                        raise ComparativeAnalysisError(f"arm optimizer step is nonzero in {path}")
                    hits = arm_row.get("final_achievements")
                    if not isinstance(hits, dict) or any(name not in hits for name in TARGETS):
                        raise ComparativeAnalysisError(f"missing A-E achievement values in {path}")
                rows[(controller, stream, world)] = row
            if len(worlds) != expected_count:
                raise ComparativeAnalysisError(f"world namespace collision in {root}")
            if reference_worlds is None:
                reference_worlds = worlds
            elif worlds != reference_worlds:
                raise ComparativeAnalysisError(f"world namespaces are mixed across chains in {root}")
            if phase == "acquisition":
                final_row = rows[(controller, stream, max(worlds))]
                graphs = final_row.get("graphs_after")
                if not isinstance(graphs, dict) or not arms.issubset(graphs):
                    raise ComparativeAnalysisError(f"acquisition does not retain final graphs for chain {(controller, stream)}")
                for arm in arms:
                    Graph.from_json(graphs[arm])
    if len(rows) != N_CHAINS * expected_count:
        raise ComparativeAnalysisError(f"wrong total {phase} row count in {root}")
    return rows


def _arm_summary(rows: Mapping[tuple[int, int, int], Mapping[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        values = {metric: [] for metric in ("R", "Q", "native_return")}
        values.update({name: [] for name in TARGETS})
        for row in rows.values():
            record = row["arms"][arm]
            for metric in ("R", "Q", "native_return"):
                values[metric].append(float(record[metric]))
            for name in TARGETS:
                values[name].append(float(bool(record["final_achievements"][name])))
        result[arm] = {metric: float(fmean(numbers)) for metric, numbers in values.items()}
        result[arm]["n_rows"] = len(rows)
    return result


def _paired_world_contrasts(rows: Mapping[tuple[int, int, int], Mapping[str, Any]], arms: Sequence[str]) -> list[dict[str, Any]]:
    worlds = sorted({world for _, _, world in rows})
    contrasts = []
    for world in worlds:
        block = [rows[(controller, stream, world)] for controller in CONTROLLERS for stream in STREAMS]
        for left in arms:
            for right in arms:
                if left == right:
                    continue
                contrasts.append({"world": world, "left": left, "right": right, "metric": "R",
                                  "estimate": float(fmean(float(row["arms"][left]["R"]) - float(row["arms"][right]["R"]) for row in block)),
                                  "n_rows_averaged": len(block)})
    return contrasts


def _world_namespace(rows: Mapping[tuple[int, int, int], Mapping[str, Any]]) -> tuple[int, ...]:
    """Return the already-validated shared bank namespace for one source."""
    worlds = tuple(sorted({world for _controller, _stream, world in rows}))
    if len(worlds) != N_EVALUATION:
        raise ComparativeAnalysisError("evaluation source does not contain one 100-world bank")
    return worlds


def _fixture_sea_contrasts(fixture_rows: Mapping[tuple[int, int, int], Mapping[str, Any]],
                           sea_rows: Mapping[tuple[int, int, int], Mapping[str, Any]],
                           fixture_arms: Sequence[str]) -> list[dict[str, Any]]:
    """Pair SEA-A only with rows from its own fixture world bank."""
    fixture_worlds, sea_worlds = _world_namespace(fixture_rows), _world_namespace(sea_rows)
    if fixture_worlds != sea_worlds:
        raise ComparativeAnalysisError("SEA-A evaluation bank does not match its fixture evaluation bank")
    output = []
    for world in fixture_worlds:
        sea_mean = float(fmean(float(sea_rows[(controller, stream, world)]["arms"]["SEA-A"]["R"])
                               for controller in CONTROLLERS for stream in STREAMS))
        for arm in fixture_arms:
            fixture_mean = float(fmean(float(fixture_rows[(controller, stream, world)]["arms"][arm]["R"])
                                        for controller in CONTROLLERS for stream in STREAMS))
            output.append({"world": world, "left": "SEA-A", "right": arm, "metric": "R",
                           "estimate": sea_mean - fixture_mean, "n_rows_averaged": N_CHAINS,
                           "world_bank": "fixture_and_sea"})
    return output


def _descriptive_interval(values: Sequence[float]) -> dict[str, Any]:
    mean = float(fmean(values))
    if len(values) < 2:
        return {"estimate": mean, "n": len(values), "lower": mean, "upper": mean}
    se = float(stdev(values) / math.sqrt(len(values)))
    return {"estimate": mean, "n": len(values), "se": se, "lower": mean - 1.96 * se, "upper": mean + 1.96 * se}


def _selector_chain_summary(root: Path, fixture: str, selector_arms: Sequence[str]) -> list[dict[str, Any]]:
    correct, fixed = _correct_graph(), _fixed_graph(fixture)
    fixed_tv = _tv(fixed, correct)
    output = []
    for controller in CONTROLLERS:
        for stream in STREAMS:
            entries = [(_identity(_read_json(path), path, "acquisition", controller, stream), _read_json(path))
                       for path in _chain_paths(root, "acquisition", controller, stream)]
            entries.sort(key=lambda item: item[0])
            final = entries[-1][1]
            # O and F are fixed evaluation references rather than acquired
            # chains.  Keeping them here makes the provider table complete
            # without pretending that either generated challenge evidence.
            for arm, graph in (("O", correct), ("F", fixed)):
                output.append({"method": arm, "controller": controller, "stream": stream,
                               "graph_source": "fixed evaluation reference", "provider_tv_to_O": _tv(graph, correct),
                               "provider_decision_equivalent_to_O": _provider_signature(graph) == _provider_signature(correct),
                               "provider_repaired_toward_O": False, "challenge_evidence": 0.0,
                               "challenge_attempts": 0, "first_repair_world": None,
                               "final_provider_signature_masks": 31})
            for arm in selector_arms:
                graph = Graph.from_json(final["graphs_after"][arm])
                signature = _provider_signature(graph)
                evidence = 0.0
                attempts = 0
                first_repair_world: int | None = None
                for world, row in entries:
                    record = row["arms"][arm]
                    pairs = record.get("pair_evidence", record.get("es_pair_evidence", {}))
                    if isinstance(pairs, dict):
                        evidence += sum(float(value) for value in pairs.values() if isinstance(value, (int, float)))
                    challenges = record.get("challenges", record.get("es_challenges", []))
                    if isinstance(challenges, list):
                        attempts += len(challenges)
                    state = Graph.from_json(row["graphs_after"][arm])
                    if first_repair_world is None and _tv(state, correct) + EPSILON < fixed_tv:
                        first_repair_world = world
                tv_to_o = _tv(graph, correct)
                output.append({"method": arm, "controller": controller, "stream": stream,
                               "graph_source": "acquired frozen graph",
                               "provider_tv_to_O": tv_to_o,
                               "provider_decision_equivalent_to_O": _provider_signature(graph) == _provider_signature(correct),
                               "provider_repaired_toward_O": tv_to_o + EPSILON < fixed_tv,
                               "challenge_evidence": evidence, "challenge_attempts": attempts,
                               "first_repair_world": first_repair_world,
                               "final_provider_signature_masks": len(signature)})
    return output


def _selector_challenge_audit(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe realised selector opportunities without reinterpreting outcomes.

    A comparison is eligible only when the three selectors faced the same
    candidate keys at the same chain/world/challenge index and there was a real
    choice (frontier size greater than one).  This is an execution audit, not a
    test of treatment success.
    """
    arms = ("R", "UQ", "ES")
    frontier_sizes: dict[str, list[int]] = {arm: [] for arm in arms}
    pair_counts: dict[str, Counter[str]] = {arm: Counter() for arm in arms}
    by_opportunity: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for controller in CONTROLLERS:
        for stream in STREAMS:
            for path in _chain_paths(root, "acquisition", controller, stream):
                row = _read_json(path)
                world = _identity(row, path, "acquisition", controller, stream)
                for arm in arms:
                    challenges = row["arms"][arm].get("challenges", [])
                    if not isinstance(challenges, list):
                        raise ComparativeAnalysisError(f"selector challenges are malformed in {path}")
                    for index, challenge in enumerate(challenges):
                        if not isinstance(challenge, dict):
                            raise ComparativeAnalysisError(f"selector challenge is malformed in {path}")
                        try:
                            frontier_size = int(challenge["frontier_size"])
                            keys = tuple(str(key) for key in challenge["candidate_pair_keys"])
                            selected = str(challenge["pair_key"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ComparativeAnalysisError(f"selector audit fields are missing in {path}") from exc
                        if frontier_size < 1 or len(keys) != frontier_size or tuple(sorted(keys)) != keys or selected not in keys:
                            raise ComparativeAnalysisError(f"invalid selector frontier record in {path}")
                        frontier_sizes[arm].append(frontier_size); pair_counts[arm][selected] += 1
                        by_opportunity[(controller, stream, world, index)][arm] = {"frontier_size": frontier_size,
                                                                                      "candidate_pair_keys": keys, "pair_key": selected}
    summaries, frequencies = [], []
    for arm in arms:
        sizes = frontier_sizes[arm]
        total = len(sizes)
        multi = sum(size > 1 for size in sizes)
        summaries.append({"source": "selector", "arm": arm, "total_attempts": total,
                          "mean_attempts_per_chain": float(total / N_CHAINS),
                          "frontier_size_mean": float(fmean(sizes)) if sizes else 0.0,
                          "frontier_size_min": min(sizes) if sizes else None,
                          "frontier_size_max": max(sizes) if sizes else None,
                          "frontier_size_gt_1_count": multi,
                          "frontier_size_gt_1_share": float(multi / total) if total else 0.0,
                          "description": "descriptive realised challenge audit; not a continuation or significance gate"})
        frequencies.extend({"source": "selector", "arm": arm, "pair_key": pair, "count": count}
                           for pair, count in sorted(pair_counts[arm].items()))
    comparisons = []
    for left, right in (("R", "UQ"), ("R", "ES"), ("UQ", "ES")):
        comparable = differing = 0
        for opportunity in by_opportunity.values():
            a, b = opportunity.get(left), opportunity.get(right)
            if a is None or b is None:
                continue
            if a["frontier_size"] > 1 and a["frontier_size"] == b["frontier_size"] and a["candidate_pair_keys"] == b["candidate_pair_keys"]:
                comparable += 1
                differing += int(a["pair_key"] != b["pair_key"])
        comparisons.append({"source": "selector", "left": left, "right": right, "comparable_count": comparable,
                            "selected_pair_differing_count": differing,
                            "selected_pair_differing_share": float(differing / comparable) if comparable else None,
                            "description": "same chain/world/challenge-index and identical multi-candidate frontier"})
    return summaries, frequencies, comparisons


def _controller_aggregates(rows: Mapping[tuple[int, int, int], Mapping[str, Any]], arms: Sequence[str], source: str) -> list[dict[str, Any]]:
    output = []
    for controller in CONTROLLERS:
        controller_rows = [row for (seed, _stream, _world), row in rows.items() if seed == controller]
        for arm in arms:
            record: dict[str, Any] = {"source": source, "controller": controller, "method": arm, "n_rows": len(controller_rows)}
            for metric in ("R", "Q", "native_return"):
                record[metric] = float(fmean(float(row["arms"][arm][metric]) for row in controller_rows))
            for name in TARGETS:
                record[f"hit_{name}"] = float(fmean(float(bool(row["arms"][arm]["final_achievements"][name])) for row in controller_rows))
            output.append(record)
    return output


def _sea_artifacts(root: Path) -> dict[tuple[int, int], Graph]:
    candidates = sorted(root.rglob("*.json"))
    found: dict[tuple[int, int], Graph] = {}
    for path in candidates:
        value = _read_json(path)
        if "graph" not in value or "identity" not in value:
            continue
        identity = value["identity"]
        if not isinstance(identity, list) or len(identity) != 2:
            continue
        try:
            key = (int(identity[0]), int(identity[1]))
        except (TypeError, ValueError):
            continue
        if key not in {(controller, stream) for controller in CONTROLLERS for stream in STREAMS}:
            continue
        if value.get("graph_frozen") is not True:
            raise ComparativeAnalysisError(f"SEA artifact is not frozen: {path}")
        if key in found:
            raise ComparativeAnalysisError(f"duplicate SEA recovered artifact for chain {key}")
        found[key] = Graph.from_json(value["graph"])
    if len(found) != N_CHAINS:
        raise ComparativeAnalysisError(f"expected {N_CHAINS} recovered SEA graphs, found {len(found)} under {root}")
    return found


def _sea_summary(eval_rows: Mapping[tuple[int, int, int], Mapping[str, Any]], artifacts: Mapping[tuple[int, int], Graph], fixture: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    correct = _correct_graph()
    chains, controllers = [], []
    for controller in CONTROLLERS:
        per_controller = []
        for stream in STREAMS:
            graph = artifacts[(controller, stream)]
            rows = [eval_rows[(controller, stream, world)] for world in sorted({world for _, _, world in eval_rows})]
            chain = {"method": "SEA-A", "controller": controller, "stream": stream,
                     "provider_tv_to_O": _tv(graph, correct),
                     "provider_decision_equivalent_to_O": _provider_signature(graph) == _provider_signature(correct),
                     "provider_repaired_toward_O": _tv(graph, correct) + EPSILON < _tv(_fixed_graph(fixture), correct),
                     "evaluation_mean_R": float(fmean(float(row["arms"]["SEA-A"]["R"]) for row in rows)),
                     "evaluation_mean_Q": float(fmean(float(row["arms"]["SEA-A"]["Q"]) for row in rows)),
                     "evaluation_mean_native_return": float(fmean(float(row["arms"]["SEA-A"]["native_return"]) for row in rows)),
                     "n_evaluation_worlds": len(rows), "provider_signature_masks": 31}
            chains.append(chain); per_controller.append(chain)
        controllers.append({"source": "sea", "controller": controller, "method": "SEA-A", "n_chains": len(per_controller),
                            "provider_tv_to_O": float(fmean(item["provider_tv_to_O"] for item in per_controller)),
                            "evaluation_mean_R": float(fmean(item["evaluation_mean_R"] for item in per_controller)),
                            "evaluation_mean_Q": float(fmean(item["evaluation_mean_Q"] for item in per_controller)),
                            "evaluation_mean_native_return": float(fmean(item["evaluation_mean_native_return"] for item in per_controller))})
    return chains, controllers


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def analyze(*, selector_root: Path, fixture_root: Path, sea_evaluation_root: Path,
            sea_artifact_root: Path, output_root: Path, fixture: str) -> dict[str, Any]:
    selector_arms = ("O", "F", "N", "R", "UQ", "ES")
    fixture_arms = ("O", "F", "N", "ES")
    selector_eval = _validate_rows(selector_root, "evaluation", set(selector_arms), require_freeze=True)
    selector_acq = _validate_rows(selector_root, "acquisition", {"N", "R", "UQ", "ES"}, require_freeze=False)
    fixture_eval = _validate_rows(fixture_root, "evaluation", set(fixture_arms), require_freeze=True)
    _validate_rows(fixture_root, "acquisition", {"N", "ES"}, require_freeze=False)
    sea_eval = _validate_rows(sea_evaluation_root, "evaluation", {"SEA-A"}, require_freeze=True)
    artifacts = _sea_artifacts(sea_artifact_root)

    selector_chains = [{"source": "selector", **row} for row in _selector_chain_summary(selector_root, fixture, ("N", "R", "UQ", "ES"))]
    fixture_chains = [{"source": "fixture", **row} for row in _selector_chain_summary(fixture_root, fixture, ("N", "ES"))]
    selector_challenge_summary, selector_pair_frequencies, selector_comparability = _selector_challenge_audit(selector_root)
    sea_chains, sea_controller = _sea_summary(sea_eval, artifacts, fixture)
    selector_controllers = _controller_aggregates(selector_eval, selector_arms, "selector")
    fixture_controllers = _controller_aggregates(fixture_eval, fixture_arms, "fixture")
    selector_contrasts = [{**row, "world_bank": "selector"} for row in _paired_world_contrasts(selector_eval, selector_arms)]
    fixture_contrasts = [{**row, "world_bank": "fixture_and_sea"} for row in _paired_world_contrasts(fixture_eval, fixture_arms)]
    sea_fixture_contrasts = _fixture_sea_contrasts(fixture_eval, sea_eval, fixture_arms)
    def summarize(contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        output = {}
        for left, right in sorted({(str(row["left"]), str(row["right"])) for row in contrasts}):
            values = [float(row["estimate"]) for row in contrasts if row["left"] == left and row["right"] == right]
            output[f"{left}_minus_{right}"] = _descriptive_interval(values)
        return output
    selector_summary = _arm_summary(selector_eval, selector_arms)
    fixture_summary = _arm_summary(fixture_eval, fixture_arms)
    sea_summary = _arm_summary(sea_eval, ("SEA-A",))
    result = {
        "schema": "pc3-v19-comparative-analysis/v1",
        "fixture": fixture,
        "statistical_units": {"acquisition_recovery": "8 controller-stream chains", "selector_behavior": "100 selector-bank world aggregates after averaging 8 matched rows", "fixture_and_sea_behavior": "a separate 100-world fixture/SEA bank after averaging 8 matched rows", "rows": "descriptive only"},
        "world_banks": {"selector": list(_world_namespace(selector_eval)), "fixture_and_sea": list(_world_namespace(fixture_eval)),
                        "independent_banks": _world_namespace(selector_eval) != _world_namespace(fixture_eval),
                        "pairing_rule": "Selector arms are paired only within the selector bank. SEA-A is paired only with O/F/N/ES from the matching fixture bank."},
        "analysis_policy": "All valid arms are reported. Intervals are descriptive summaries, not p-value or continuation gates.",
        "input_roots": {"selector": str(selector_root), "fixture": str(fixture_root), "sea_evaluation": str(sea_evaluation_root), "sea_artifacts": str(sea_artifact_root)},
        "validation": {"expected_chains": N_CHAINS, "expected_acquisition_per_chain": N_ACQUISITION, "expected_evaluation_per_chain": N_EVALUATION,
                       "selector_acquisition_rows": len(selector_acq), "selector_evaluation_rows": len(selector_eval),
                       "fixture_evaluation_rows": len(fixture_eval), "sea_evaluation_rows": len(sea_eval), "sea_artifacts": len(artifacts)},
        "selector_arm_summary": selector_summary,
        "fixture_arm_summary": fixture_summary,
        "sea_arm_summary": sea_summary,
        "selector_world_level_paired_contrast_summary": summarize(selector_contrasts),
        "fixture_world_level_paired_contrast_summary": summarize(fixture_contrasts),
        "fixture_sea_world_level_paired_contrast_summary": summarize(sea_fixture_contrasts),
        "selector_chain_provider_summary": selector_chains,
        "selector_challenge_audit": {"policy": "Descriptive audit of realised selector choices; it does not select, stop, or discard any arm.",
                                     "by_arm": selector_challenge_summary, "selected_pair_frequencies": selector_pair_frequencies,
                                     "cross_selector_comparability": selector_comparability},
        "fixture_chain_provider_summary": fixture_chains,
        "sea_adapted_chain_summary": sea_chains,
        "controller_level_aggregates": selector_controllers + fixture_controllers + sea_controller,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparative_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    _write_csv(output_root / "selector_arm_summary.csv", [{"source": "selector", "method": arm, **values} for arm, values in selector_summary.items()])
    _write_csv(output_root / "fixture_arm_summary.csv", [{"source": "fixture", "method": arm, **values} for arm, values in fixture_summary.items()])
    _write_csv(output_root / "sea_arm_summary.csv", [{"source": "sea", "method": arm, **values} for arm, values in sea_summary.items()])
    _write_csv(output_root / "selector_world_level_paired_contrasts.csv", selector_contrasts)
    _write_csv(output_root / "fixture_world_level_paired_contrasts.csv", fixture_contrasts)
    _write_csv(output_root / "fixture_sea_world_level_paired_contrasts.csv", sea_fixture_contrasts)
    _write_csv(output_root / "selector_chain_provider_summary.csv", selector_chains)
    _write_csv(output_root / "selector_challenge_audit.csv", selector_challenge_summary)
    _write_csv(output_root / "selector_selected_pair_frequencies.csv", selector_pair_frequencies)
    _write_csv(output_root / "selector_challenge_comparability.csv", selector_comparability)
    _write_csv(output_root / "fixture_chain_provider_summary.csv", fixture_chains)
    _write_csv(output_root / "sea_adapted_chain_summary.csv", sea_chains)
    _write_csv(output_root / "controller_level_aggregates.csv", result["controller_level_aggregates"])
    return result


def _test_rows(root: Path, phase: str, arms: Sequence[str], *, acquisition_start: int = 1001,
               evaluation_start: int = 2001) -> None:
    for controller in CONTROLLERS:
        for stream in STREAMS:
            start, count = (acquisition_start, N_ACQUISITION) if phase == "acquisition" else (evaluation_start, N_EVALUATION)
            for world in range(start, start + count):
                arm_values = {}
                for arm in arms:
                    arm_values[arm] = {"R": 0.4, "Q": 0.4, "native_return": 1.0,
                                       "final_achievements": {target: target in {"A", "B"} for target in TARGETS}}
                    if phase == "acquisition" and arm in {"R", "UQ", "ES"}:
                        # This shared two-edge frontier is a real synthetic
                        # comparable opportunity.  UQ deliberately selects a
                        # different edge, proving the audit detects divergence.
                        chosen = "B|C" if arm == "UQ" else "A|B"
                        arm_values[arm].update({"pair_evidence": {chosen: 1}, "challenges": [{
                            "pair": chosen.split("|"), "pair_key": chosen, "frontier_size": 2,
                            "candidate_pair_keys": ["A|B", "B|C"]}]})
                row = {"schema": "synthetic/v1", "phase": phase, "identity": [controller, stream, world], "arms": arm_values}
                if phase == "evaluation":
                    row.update({"graphs_frozen": True, "optimizer_steps": 0, "evidence_rules_absent": True})
                else:
                    row["graphs_after"] = {arm: {"arm": "D", "alpha": list(_correct_graph().alpha), "beta": list(_correct_graph().beta),
                                                  "pending_positive": [0.0] * 25, "pending_counter": [0.0] * 25,
                                                  "update_count": 0, "first_hit_count": 0} for arm in arms}
                path = root / phase / f"controller_{controller}" / f"stream_{stream}" / f"world_{world}.json"
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(row), encoding="utf-8")


def schema_test() -> dict[str, Any]:
    """Small synthetic-schema test; it deliberately runs no environment code."""
    with tempfile.TemporaryDirectory(prefix="pc3-v19-schema-") as temporary:
        base = Path(temporary); selector, fixture, sea_eval, artifacts, output = (base / name for name in ("selector", "fixture", "sea_eval", "artifacts", "output"))
        # Deliberately use a different selector world namespace.  The prior
        # implementation incorrectly indexed selector rows using SEA worlds;
        # this test therefore succeeds only when cross-bank pairing is absent.
        _test_rows(selector, "acquisition", ("N", "R", "UQ", "ES"), acquisition_start=3001, evaluation_start=4001)
        _test_rows(selector, "evaluation", ("O", "F", "N", "R", "UQ", "ES"), acquisition_start=3001, evaluation_start=4001)
        _test_rows(fixture, "acquisition", ("N", "ES"), acquisition_start=1001, evaluation_start=2001)
        _test_rows(fixture, "evaluation", ("O", "F", "N", "ES"), acquisition_start=1001, evaluation_start=2001)
        _test_rows(sea_eval, "evaluation", ("SEA-A",), acquisition_start=1001, evaluation_start=2001)
        correct = _correct_graph()
        for controller in CONTROLLERS:
            for stream in STREAMS:
                payload = {"identity": [controller, stream], "graph_frozen": True,
                           "graph": {"arm": "D", "alpha": list(correct.alpha), "beta": list(correct.beta), "pending_positive": [0.0] * 25,
                                     "pending_counter": [0.0] * 25, "update_count": 0, "first_hit_count": 0}}
                path = artifacts / f"controller_{controller}_stream_{stream}.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload), encoding="utf-8")
        result = analyze(selector_root=selector, fixture_root=fixture, sea_evaluation_root=sea_eval, sea_artifact_root=artifacts, output_root=output, fixture="p2")
        selector_pairs = result["selector_world_level_paired_contrast_summary"]
        selector_audit = result["selector_challenge_audit"]["cross_selector_comparability"]
        r_uq = next(row for row in selector_audit if row["left"] == "R" and row["right"] == "UQ")
        if (result["validation"]["sea_artifacts"] != N_CHAINS or len(result["sea_adapted_chain_summary"]) != N_CHAINS
                or len(result["fixture_chain_provider_summary"]) != N_CHAINS * 4
                or not result["world_banks"]["independent_banks"]
                or any("SEA-A" in name for name in selector_pairs)
                or r_uq["comparable_count"] < 1 or r_uq["selected_pair_differing_count"] < 1):
            raise ComparativeAnalysisError("synthetic schema test did not produce all chains")
    return {"status": "SCHEMA_TEST_PASS", "environment_rollouts": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "schema-test"))
    parser.add_argument("--selector-root", type=Path); parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--sea-evaluation-root", type=Path); parser.add_argument("--sea-artifact-root", type=Path)
    parser.add_argument("--output-root", type=Path); parser.add_argument("--fixture", choices=("p1", "p2"), default="p2")
    args = parser.parse_args(argv)
    if args.stage == "schema-test":
        result = schema_test()
    else:
        required = (args.selector_root, args.fixture_root, args.sea_evaluation_root, args.sea_artifact_root, args.output_root)
        if any(value is None for value in required):
            parser.error("analyze requires all four input roots and --output-root")
        result = analyze(selector_root=args.selector_root, fixture_root=args.fixture_root, sea_evaluation_root=args.sea_evaluation_root,
                         sea_artifact_root=args.sea_artifact_root, output_root=args.output_root, fixture=args.fixture)
    summary = result if args.stage == "schema-test" else {"status": "COMPLETE", "schema": result.get("schema"), "validation": result.get("validation")}
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
