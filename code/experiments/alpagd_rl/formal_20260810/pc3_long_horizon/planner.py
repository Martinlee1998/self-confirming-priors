"""Pure one-step mechanics labeler for PC-3.

``propose_action`` reads an immutable state value and returns one action index.
It does not step an environment, mutate state, complete prerequisites, or inspect
policy/controller information.  Infeasible targets deliberately yield ``noop``;
the caller owns the frozen 1000-step target window and must not immediately
reselect a different target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from experiments.alpagd_rl.formal_20260810.q0_state_bank.schema import ACTION_NAMES

ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
DIRECTIONS = (
    ("move_left", (-1, 0)), ("move_right", (1, 0)),
    ("move_up", (0, -1)), ("move_down", (0, 1)),
)
TARGETS = ("place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe", "collect_iron")
PAIRED_WRONG_TARGETS = ("place_plant", "make_wood_sword", "collect_coal", "make_stone_sword")
PLACEABLE = frozenset(("grass", "sand", "path"))
WALKABLE = frozenset(("grass", "sand", "path"))


@dataclass(frozen=True)
class MechanicsState:
    player_position: tuple[int, int]
    player_facing: tuple[int, int]
    area: tuple[int, int]
    cells: tuple[tuple[tuple[str | None, str | None], ...], ...]
    inventory: Mapping[str, int]
    achievements: Mapping[str, int]

    def cell(self, position: tuple[int, int]) -> tuple[str | None, str | None]:
        return self.cells[position[0]][position[1]]


def capture_state(view) -> MechanicsState:
    """Copy the allowed live mechanics fields into an immutable planner value."""
    area = tuple(map(int, view.world_area()))
    cells = tuple(tuple(view.cell((x, y)) for y in range(area[1])) for x in range(area[0]))
    return MechanicsState(
        tuple(map(int, view.player_position())), tuple(map(int, view.player_facing())), area,
        cells, dict(view.inventory()), dict(view.achievement_counts()),
    )


def _inside(state: MechanicsState, position: tuple[int, int]) -> bool:
    return 0 <= position[0] < state.area[0] and 0 <= position[1] < state.area[1]


def _add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] + right[0], left[1] + right[1]


def _walkable(state: MechanicsState, position: tuple[int, int]) -> bool:
    material, obj = state.cell(position)
    return material in WALKABLE and (obj is None or (obj == "Player" and position == state.player_position))


def _bfs_first(state: MechanicsState, goals: set[tuple[int, int]]) -> str | None:
    start = state.player_position
    if start in goals:
        return "noop"
    queue = deque([start]); first: dict[tuple[int, int], str | None] = {start: None}
    while queue:
        current = queue.popleft()
        for action, delta in DIRECTIONS:
            nxt = _add(current, delta)
            if nxt in first or not _inside(state, nxt) or not _walkable(state, nxt):
                continue
            first[nxt] = action if current == start else first[current]
            if nxt in goals:
                return first[nxt]
            queue.append(nxt)
    return None


def _adjacent_action(state: MechanicsState, material: str) -> str:
    """Navigate to, face, then interact with the nearest required material."""
    goals: dict[tuple[int, int], str] = {}
    for x in range(state.area[0]):
        for y in range(state.area[1]):
            if state.cell((x, y))[0] != material:
                continue
            for action, delta in DIRECTIONS:
                stand = (x - delta[0], y - delta[1])
                if _inside(state, stand) and _walkable(state, stand):
                    goals.setdefault(stand, action)
    desired = goals.get(state.player_position)
    if desired is not None:
        delta = dict(DIRECTIONS)[desired]
        return "do" if state.player_facing == delta else desired
    return _bfs_first(state, set(goals)) or "noop"


def _near_material(state: MechanicsState, material: str) -> bool:
    x, y = state.player_position
    return any(
        state.cell((nx, ny))[0] == material
        for nx in range(max(0, x - 1), min(state.area[0], x + 2))
        for ny in range(max(0, y - 1), min(state.area[1], y + 2))
    )


def _navigate_near(state: MechanicsState, material: str) -> str:
    goals = {
        (x, y) for x in range(state.area[0]) for y in range(state.area[1])
        if _walkable(state, (x, y)) and any(
            state.cell((nx, ny))[0] == material
            for nx in range(max(0, x - 1), min(state.area[0], x + 2))
            for ny in range(max(0, y - 1), min(state.area[1], y + 2))
        )
    }
    return _bfs_first(state, goals) or "noop"


def _place_table(state: MechanicsState) -> str:
    if int(state.inventory.get("wood", 0)) < 2:
        return "noop"
    front = _add(state.player_position, state.player_facing)
    if _inside(state, front):
        material, obj = state.cell(front)
        if material in PLACEABLE and obj is None:
            return "place_table"
    # A placement action targets the cell in front. Move to a walkable stand
    # adjacent to a legal empty cell, with deterministic L/R/U/D tie-breaking.
    goals: dict[tuple[int, int], str] = {}
    for x in range(state.area[0]):
        for y in range(state.area[1]):
            if state.cell((x, y))[0] not in PLACEABLE or state.cell((x, y))[1] is not None:
                continue
            for action, delta in DIRECTIONS:
                stand = (x - delta[0], y - delta[1])
                if _inside(state, stand) and _walkable(state, stand):
                    goals.setdefault(stand, action)
    desired = goals.get(state.player_position)
    if desired is not None:
        return desired
    return _bfs_first(state, set(goals)) or "noop"


def _place_plant(state: MechanicsState) -> str:
    if int(state.inventory.get("sapling", 0)) < 1:
        return "noop"
    front = _add(state.player_position, state.player_facing)
    if _inside(state, front):
        material, obj = state.cell(front)
        if material == "grass" and obj is None:
            return "place_plant"
    goals: dict[tuple[int, int], str] = {}
    for x in range(state.area[0]):
        for y in range(state.area[1]):
            if state.cell((x, y)) != ("grass", None):
                continue
            for action, delta in DIRECTIONS:
                stand = (x - delta[0], y - delta[1])
                if _inside(state, stand) and _walkable(state, stand): goals.setdefault(stand, action)
    desired = goals.get(state.player_position)
    return desired if desired is not None else (_bfs_first(state, set(goals)) or "noop")


def _action_name(state: MechanicsState, target: str) -> str:
    if target == "place_table":
        return _place_table(state)
    if target == "place_plant":
        return _place_plant(state)
    if target == "make_wood_pickaxe":
        if int(state.inventory.get("wood", 0)) < 1:
            return "noop"
        return "make_wood_pickaxe" if _near_material(state, "table") else _navigate_near(state, "table")
    if target == "make_wood_sword":
        if int(state.inventory.get("wood", 0)) < 1:
            return "noop"
        return "make_wood_sword" if _near_material(state, "table") else _navigate_near(state, "table")
    if target == "collect_stone":
        if int(state.inventory.get("wood_pickaxe", 0)) < 1:
            return "noop"
        return _adjacent_action(state, "stone")
    if target == "collect_coal":
        if int(state.inventory.get("wood_pickaxe", 0)) < 1:
            return "noop"
        return _adjacent_action(state, "coal")
    if target == "make_stone_pickaxe":
        if int(state.inventory.get("wood", 0)) < 1 or int(state.inventory.get("stone", 0)) < 1:
            return "noop"
        return "make_stone_pickaxe" if _near_material(state, "table") else _navigate_near(state, "table")
    if target == "make_stone_sword":
        if int(state.inventory.get("wood", 0)) < 1 or int(state.inventory.get("stone", 0)) < 1:
            return "noop"
        return "make_stone_sword" if _near_material(state, "table") else _navigate_near(state, "table")
    if target == "collect_iron":
        if int(state.inventory.get("stone_pickaxe", 0)) < 1:
            return "noop"
        return _adjacent_action(state, "iron")
    raise ValueError(f"unsupported PC-3 target:{target}")


def propose_action(state: MechanicsState, target: str) -> int:
    """Return a deterministic action label without mutating ``state``."""
    if target not in TARGETS + PAIRED_WRONG_TARGETS:
        raise ValueError(f"unsupported PC-3 target:{target}")
    return ACTION_INDEX[_action_name(state, target)]
