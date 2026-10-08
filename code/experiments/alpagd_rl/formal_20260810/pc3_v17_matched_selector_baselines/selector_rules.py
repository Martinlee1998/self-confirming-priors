"""Graph-private selector rules for PC3-v17 (no environment execution)."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence


def pair_key(source: str, target: str) -> str:
    return "|".join(sorted((source, target)))


def _pick_tie(items: Sequence[dict[str, Any]], uniform: float) -> dict[str, Any]:
    ordered = sorted(items, key=lambda item: str(item["pair_key"]))
    return ordered[min(int(float(uniform) * len(ordered)), len(ordered) - 1)]


def common_frontier(*, provider: Any, targets: Sequence[str], achieved: Mapping[str, bool],
                    history: Mapping[str, Any], world: int, used_masks: set[str],
                    used_pairs: set[str], max_challenges: int) -> list[dict[str, Any]]:
    """Return the v13 eligible frontier, without computing decision impact."""
    mask = "".join("1" if achieved[code] else "0" for code in targets)
    if mask in used_masks or len(used_pairs) >= max_challenges:
        return []
    distribution = provider.distribution(achieved, 0)
    unfinished = [i for i, code in enumerate(targets) if not achieved[code]]
    candidates: list[dict[str, Any]] = []
    for target_i in unfinished:
        blockers = [source_i for source_i in unfinished if source_i != target_i
                    and provider.graph.probability(source_i, target_i) > 0.5]
        if len(blockers) != 1 or distribution[blockers[0]] <= 0.0:
            continue
        blocker_i = blockers[0]
        pair = (targets[target_i], targets[blocker_i])
        key = pair_key(*pair)
        record = history.get(key, {})
        if key in used_pairs or record.get("retired", False) or int(record.get("cooldown_until_world", 0)) >= world:
            continue
        candidates.append({"pair": pair, "pair_key": key, "mask": mask,
                           "probe_target": targets[target_i], "blocker": targets[blocker_i],
                           "evidence_edge": [targets[target_i], targets[blocker_i]],
                           "evidence_edge_probability_before": float(provider.graph.probability(target_i, blocker_i))})
    return candidates


def choose_random(candidates: Sequence[dict[str, Any]], uniform: float) -> dict[str, Any] | None:
    return _pick_tie(candidates, uniform) if candidates else None


def choose_uncertainty(candidates: Sequence[dict[str, Any]], uniform: float) -> dict[str, Any] | None:
    """Maximize Bernoulli entropy using only the current candidate belief."""
    if not candidates:
        return None
    def entropy(item: dict[str, Any]) -> float:
        p = float(item["evidence_edge_probability_before"])
        return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p) if 0.0 < p < 1.0 else 0.0
    best = max(entropy(item) for item in candidates)
    return _pick_tie([item for item in candidates if math.isclose(entropy(item), best, abs_tol=1e-12)], uniform)


def choose_es(*, candidates: Sequence[dict[str, Any]], provider: Any, targets: Sequence[str], achieved: Mapping[str, bool],
              uniform: float, edge_index: Any, target_provider: Any) -> dict[str, Any] | None:
    """v13 impact rule; only ES may construct this counterfactual score."""
    if not candidates:
        return None
    before = provider.distribution(achieved, 0)
    scored = []
    for candidate in candidates:
        graph = copy.deepcopy(provider.graph)
        blocker_i, target_i = targets.index(candidate["blocker"]), targets.index(candidate["probe_target"])
        edge = edge_index(blocker_i, target_i)
        graph.alpha[edge], graph.beta[edge] = 1.0, 9.0
        after = target_provider(graph, mutable=False).distribution(achieved, 0)
        scored.append((0.5 * sum(abs(a - b) for a, b in zip(before, after)), candidate))
    best = max(score for score, _ in scored)
    return _pick_tie([candidate for score, candidate in scored if math.isclose(score, best, abs_tol=1e-12)], uniform)
