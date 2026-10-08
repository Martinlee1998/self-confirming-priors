"""Mask-only deterministic frontier computation for Crafter 1.8.3."""

from dataclasses import dataclass
from typing import Sequence


CANONICAL_ACHIEVEMENT_IDS = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)

# Every entry contains all achievement IDs that mechanics force to have been
# unlocked earlier, not merely the transitive reduction of that relation.
_HARD_PREREQUISITES = (
    frozenset({"collect_wood", "make_wood_pickaxe", "place_table"}),
    frozenset({"collect_coal", "collect_iron", "collect_stone", "collect_wood", "make_iron_pickaxe", "make_stone_pickaxe", "make_wood_pickaxe", "place_furnace", "place_table"}),
    frozenset(),
    frozenset({"collect_stone", "collect_wood", "make_stone_pickaxe", "make_wood_pickaxe", "place_table"}),
    frozenset(),
    frozenset({"collect_wood", "make_wood_pickaxe", "place_table"}),
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset(),
    frozenset({"collect_sapling", "place_plant"}),
    frozenset({"collect_coal", "collect_iron", "collect_stone", "collect_wood", "make_stone_pickaxe", "make_wood_pickaxe", "place_furnace", "place_table"}),
    frozenset({"collect_coal", "collect_iron", "collect_stone", "collect_wood", "make_stone_pickaxe", "make_wood_pickaxe", "place_furnace", "place_table"}),
    frozenset({"collect_stone", "collect_wood", "make_wood_pickaxe", "place_table"}),
    frozenset({"collect_stone", "collect_wood", "make_wood_pickaxe", "place_table"}),
    frozenset({"collect_wood", "place_table"}),
    frozenset({"collect_wood", "place_table"}),
    frozenset({"collect_stone", "collect_wood", "make_wood_pickaxe", "place_table"}),
    frozenset({"collect_sapling"}),
    frozenset({"collect_stone", "collect_wood", "make_wood_pickaxe", "place_table"}),
    frozenset({"collect_wood"}),
    frozenset(),
)


@dataclass(frozen=True)
class FrontierDecision:
    """Candidate IDs and the uniform sampling contract for the caller."""

    candidates: tuple[str, ...]
    mode: str


def frontier_from_achieved_mask(achieved_mask: Sequence[object]) -> FrontierDecision:
    """Return frontier candidates using only a canonical-order achieved mask.

    ``mode == 'frontier'`` means the caller samples uniformly from hard-
    prerequisite-eligible candidates. ``uniform_unachieved`` is the specified
    fail-safe fallback. ``null`` means all achievements are already complete.
    The function itself is deterministic and consumes no random state.
    """
    if isinstance(achieved_mask, (str, bytes)):
        raise TypeError("achieved_mask must be a length-22 sequence of binary values")
    try:
        values = tuple(achieved_mask)
    except TypeError as exc:
        raise TypeError("achieved_mask must be a length-22 sequence") from exc
    if len(values) != len(CANONICAL_ACHIEVEMENT_IDS):
        raise ValueError("achieved_mask must contain exactly 22 values")
    if any(value not in (0, 1, False, True) for value in values):
        raise ValueError("achieved_mask values must be binary")

    achieved = {
        achievement_id
        for achievement_id, value in zip(CANONICAL_ACHIEVEMENT_IDS, values)
        if bool(value)
    }
    unachieved = tuple(
        achievement_id
        for achievement_id in CANONICAL_ACHIEVEMENT_IDS
        if achievement_id not in achieved
    )
    frontier = tuple(
        achievement_id
        for achievement_id, prerequisites in zip(
            CANONICAL_ACHIEVEMENT_IDS, _HARD_PREREQUISITES
        )
        if achievement_id not in achieved and prerequisites <= achieved
    )
    if frontier:
        return FrontierDecision(frontier, "frontier")
    if unachieved:
        return FrontierDecision(unachieved, "uniform_unachieved")
    return FrontierDecision((), "null")
