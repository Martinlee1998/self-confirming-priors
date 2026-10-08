"""The sole v18 change: an edge-level, graph-private common frontier."""
from __future__ import annotations
import copy
import math
from typing import Any, Mapping, Sequence

def _key(source: str, target: str) -> str: return "|".join(sorted((source, target)))
def _pick(items: Sequence[dict[str, Any]], uniform: float) -> dict[str, Any]:
    ordered = sorted(items, key=lambda item: item["pair_key"])
    return ordered[min(int(uniform * len(ordered)), len(ordered) - 1)]
def common_frontier(*, provider: Any, targets: Sequence[str], achieved: Mapping[str, bool], history: Mapping[str, Any], world: int, used_masks: set[str], used_pairs: set[str], max_challenges: int) -> list[dict[str, Any]]:
    mask = "".join("1" if achieved[code] else "0" for code in targets)
    if mask in used_masks or len(used_pairs) >= max_challenges: return []
    distribution, unfinished = provider.distribution(achieved, 0), [i for i, code in enumerate(targets) if not achieved[code]]
    out: list[dict[str, Any]] = []
    for target_i in unfinished:
        for blocker_i in unfinished:
            if blocker_i == target_i or provider.graph.probability(blocker_i, target_i) <= .5 or distribution[blocker_i] <= 0.: continue
            pair = (targets[target_i], targets[blocker_i]); key = _key(*pair); record = history.get(key, {})
            if key in used_pairs or record.get("retired", False) or int(record.get("cooldown_until_world", 0)) >= world: continue
            out.append({"pair": pair, "pair_key": key, "mask": mask, "probe_target": targets[target_i], "blocker": targets[blocker_i], "evidence_edge": [targets[target_i], targets[blocker_i]], "evidence_edge_probability_before": float(provider.graph.probability(target_i, blocker_i))})
    keys = sorted(item["pair_key"] for item in out)
    for item in out: item["frontier_size"], item["candidate_pair_keys"] = len(out), keys
    return out
def _annotate(item: dict[str, Any], score: float) -> dict[str, Any]:
    value = dict(item); value["selected_selector_score"] = float(score); value["selected_evidence_edge_probability"] = float(value["evidence_edge_probability_before"]); return value
def choose_random(candidates: Sequence[dict[str, Any]], uniform: float) -> dict[str, Any] | None:
    return _annotate(_pick(candidates, uniform), int(uniform * len(candidates))) if candidates else None
def choose_uncertainty(candidates: Sequence[dict[str, Any]], uniform: float) -> dict[str, Any] | None:
    if not candidates: return None
    def h(item: dict[str, Any]) -> float:
        p = item["evidence_edge_probability_before"]; return -p*math.log(p)-(1-p)*math.log(1-p) if 0 < p < 1 else 0.
    best=max(h(x) for x in candidates); return _annotate(_pick([x for x in candidates if math.isclose(h(x), best, abs_tol=1e-12)], uniform), best)
def choose_es(*, candidates: Sequence[dict[str, Any]], provider: Any, targets: Sequence[str], achieved: Mapping[str, bool], uniform: float, edge_index: Any, target_provider: Any) -> dict[str, Any] | None:
    if not candidates: return None
    before=provider.distribution(achieved,0); scored=[]
    for item in candidates:
        graph=copy.deepcopy(provider.graph); edge=edge_index(targets.index(item["blocker"]),targets.index(item["probe_target"])); graph.alpha[edge],graph.beta[edge]=1.,9.
        after=target_provider(graph,mutable=False).distribution(achieved,0); scored.append((.5*sum(abs(a-b) for a,b in zip(before,after)),item))
    best=max(score for score,_ in scored); return _annotate(_pick([item for score,item in scored if math.isclose(score,best,abs_tol=1e-12)],uniform),best)
