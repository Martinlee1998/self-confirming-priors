"""Immutable capability object forming the sole PC3-v9 teacher boundary."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

INVENTORY_NAMES = (
    "health", "food", "drink", "energy", "sapling", "wood", "stone", "coal",
    "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)
FORBIDDEN_NAMES = frozenset({
    "env", "view", "seed", "world_seed", "absolute_position", "position", "full_map",
    "world_cells", "rng", "future", "planner", "planner_state", "step_fn",
})


@dataclass(frozen=True, slots=True)
class LocalSemanticCell:
    dx: int
    dy: int
    material: str | None
    object_name: str | None

    def __post_init__(self) -> None:
        if not -4 <= int(self.dx) <= 4 or not -3 <= int(self.dy) <= 3:
            raise ValueError("local semantic cell escaped the frozen 9x7 crop")


@dataclass(frozen=True, slots=True)
class ObservableStep:
    """No env/view/seed/absolute-position capability can be attached."""

    pixels: np.ndarray
    local_semantic_crop: tuple[LocalSemanticCell, ...]
    inventory16: tuple[int, ...]
    facing: tuple[int, int]
    achieved22: tuple[int, ...]
    public_event22: tuple[int, ...]
    prev_action: int
    ego_delta: tuple[int, int]
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels)
        if pixels.shape != (64, 64, 3) or pixels.dtype != np.uint8:
            raise ValueError("pixels must be an immutable uint8 64x64x3 rendered viewport")
        copied = pixels.copy(); copied.flags.writeable = False
        object.__setattr__(self, "pixels", copied)
        if len(self.local_semantic_crop) != 63:
            raise ValueError("local semantic crop must contain exactly 9x7 cells")
        coords = {(int(cell.dx), int(cell.dy)) for cell in self.local_semantic_crop}
        if coords != {(x, y) for x in range(-4, 5) for y in range(-3, 4)}:
            raise ValueError("local semantic crop coordinates are incomplete or duplicated")
        if len(self.inventory16) != 16 or any(int(v) < 0 for v in self.inventory16):
            raise ValueError("public HUD inventory must contain 16 nonnegative counts")
        if self.facing not in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            raise ValueError("facing must be a public unit direction")
        for label, values in (("achieved22", self.achieved22), ("public_event22", self.public_event22)):
            if len(values) != 22 or any(v not in (0, 1, False, True) for v in values):
                raise ValueError(f"{label} must be a public binary length-22 vector")
        if not -1 <= int(self.prev_action) < 17:
            raise ValueError("previous action must be -1 or a valid action")
        if self.ego_delta not in {(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)}:
            raise ValueError("ego delta must be an adjacent public motion delta")

    @property
    def inventory(self) -> Mapping[str, int]:
        return MappingProxyType(dict(zip(INVENTORY_NAMES, map(int, self.inventory16))))


def adjacent_ego_delta(previous_absolute_position: Sequence[int] | None,
                       current_absolute_position: Sequence[int]) -> tuple[int, int]:
    """Boundary-only conversion. Absolute inputs are neither returned nor retained."""
    if len(current_absolute_position) != 2 or (previous_absolute_position is not None and len(previous_absolute_position) != 2):
        raise ValueError("adapter positions must be 2D")
    if previous_absolute_position is None:
        return (0, 0)
    delta = (int(current_absolute_position[0]) - int(previous_absolute_position[0]),
             int(current_absolute_position[1]) - int(previous_absolute_position[1]))
    if delta not in {(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)}:
        raise ValueError("adjacent observations produced a nonlocal ego delta")
    return delta


def reject_forbidden_capabilities(value: object) -> None:
    names = set(vars(value)) if hasattr(value, "__dict__") else set()
    leaked = sorted(names & FORBIDDEN_NAMES)
    if leaked:
        raise TypeError(f"forbidden teacher capabilities: {leaked}")
