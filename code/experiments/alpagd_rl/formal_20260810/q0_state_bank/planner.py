"""Version-pinned deterministic mechanics-guided planner for Q0A.

The planner consumes only live Crafter mechanics state through ``MechanicsView``.
It never imports or calls a learned policy.  Every returned action sequence is
generated online from the fresh replay on which it is executed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schema import ACTION_NAMES, CANONICAL_ACHIEVEMENTS

PLANNER_VERSION = "q0-deterministic-mechanics-controller/v2"
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
DIRECTIONS = (
    ("move_left", (-1, 0)),
    ("move_right", (1, 0)),
    ("move_up", (0, -1)),
    ("move_down", (0, 1)),
)

COLLECT_TARGET_MATERIAL = {
    "collect_wood": "tree", "collect_stone": "stone", "collect_coal": "coal",
    "collect_iron": "iron", "collect_diamond": "diamond",
    "collect_drink": "water", "collect_sapling": "grass",
}
TARGET_OBJECT = {
    "defeat_skeleton": "Skeleton", "defeat_zombie": "Zombie",
    "eat_cow": "Cow", "eat_plant": "Plant",
}
PLACE_TARGET_ACTION = {
    "place_stone": "place_stone", "place_table": "place_table",
    "place_furnace": "place_furnace", "place_plant": "place_plant",
}
MAKE_TARGET_ACTION = {
    "make_wood_pickaxe": "make_wood_pickaxe",
    "make_stone_pickaxe": "make_stone_pickaxe",
    "make_iron_pickaxe": "make_iron_pickaxe",
    "make_wood_sword": "make_wood_sword",
    "make_stone_sword": "make_stone_sword",
    "make_iron_sword": "make_iron_sword",
}

MAKE_RECIPE = {
    "make_wood_pickaxe": ({"wood": 1}, ("table",)),
    "make_wood_sword": ({"wood": 1}, ("table",)),
    "make_stone_pickaxe": ({"wood": 1, "stone": 1}, ("table",)),
    "make_stone_sword": ({"wood": 1, "stone": 1}, ("table",)),
    "make_iron_pickaxe": ({"wood": 1, "coal": 1, "iron": 1}, ("table", "furnace")),
    "make_iron_sword": ({"wood": 1, "coal": 1, "iron": 1}, ("table", "furnace")),
}
PLACE_RECIPE = {
    "place_stone": ({"stone": 1}, ("grass", "sand", "path", "water", "lava")),
    "place_table": ({"wood": 2}, ("grass", "sand", "path")),
    "place_furnace": ({"stone": 4}, ("grass", "sand", "path")),
    "place_plant": ({"sapling": 1}, ("grass",)),
}
COLLECT_REQUIRE = {
    "stone": {"wood_pickaxe": 1}, "coal": {"wood_pickaxe": 1},
    "iron": {"stone_pickaxe": 1}, "diamond": {"iron_pickaxe": 1},
}


class PlannerError(RuntimeError):
    pass


class MechanicsView(Protocol):
    """The planner's only source of world information."""

    def player_position(self) -> tuple[int, int]: ...
    def player_facing(self) -> tuple[int, int]: ...
    def inventory(self) -> Mapping[str, int]: ...
    def achievement_counts(self) -> Mapping[str, int]: ...
    def world_area(self) -> tuple[int, int]: ...
    def cell(self, position: tuple[int, int]) -> tuple[str | None, str | None]: ...


StepFn = Callable[[int], tuple[Any, float, bool, Mapping[str, Any]]]


class PinnedCrafterMechanicsView:
    """Read live mechanics from the source-hash-pinned official Crafter 1.8.3.

    Upstream 1.8.3 exposes no public full-world query.  Q0 therefore uses the
    official ``_world``/``_player`` mechanics objects only after the existing
    formal source fingerprint and worker patch have been validated.  No model,
    checkpoint, trajectory, or learned representation is involved.
    """

    MATERIALS = (
        None, "water", "grass", "stone", "path", "sand", "tree", "lava",
        "coal", "iron", "diamond", "table", "furnace",
    )
    OBJECT_NAMES = {
        "cow": "Cow", "zombie": "Zombie", "skeleton": "Skeleton",
        "arrow": "Arrow", "plant": "Plant", "fence": "Fence",
    }

    EXPECTED_ENGINE_SHA256 = "6ba8e03e7e141b730378a286e3f3cc2778f88f785bd60992b36cfce142789936"
    EXPECTED_ENV_SHA256 = "a728d30479c4c0b2a3fe39505169af4a861e114132bde76d6c05d235da0171b6"
    EXPECTED_OBJECTS_SHA256 = "415aa49964639f3ae6b03372d588476fcfc4cafd6fd8b04a2d19b0fd6752b161"
    EXPECTED_WORLDGEN_SHA256 = "135599bea2a2f05c5e83be19a73cab7e0fe79ec15bdb72123671d44d351e1cfa"
    EXPECTED_DATA_SHA256 = "41ec28a266e74d7f98e0b7381ff5f11ed491c2a6cf39a460eda8bbd751b8630f"

    def __init__(self, env: Any, *, source_evidence: Mapping[str, Any] | None = None):
        required = ("_world", "_player")
        if any(not hasattr(env, name) for name in required):
            raise PlannerError("pinned Crafter 1.8.3 internal mechanics interface missing")
        if source_evidence is None:
            source_evidence = self.source_fingerprint()
        expected = {
            "crafter_version": "1.8.3",
            "engine_sha256": self.EXPECTED_ENGINE_SHA256,
            "env_sha256": self.EXPECTED_ENV_SHA256,
            "objects_sha256": self.EXPECTED_OBJECTS_SHA256,
            "worldgen_sha256": self.EXPECTED_WORLDGEN_SHA256,
            "data_sha256": self.EXPECTED_DATA_SHA256,
        }
        if {key: source_evidence.get(key) for key in expected} != expected:
            raise PlannerError("Crafter 1.8.3 five-source fingerprint mismatch")
        if source_evidence.get("formal_patch_applied") is not True:
            raise PlannerError("formal deterministic patch evidence missing")
        self.env = env
        self.source_evidence = dict(source_evidence)

    @classmethod
    def source_fingerprint(cls) -> dict[str, Any]:
        import crafter.engine as engine
        import crafter.env as env_module
        import crafter.objects as objects
        import crafter.worldgen as worldgen
        from crafter import constants

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        engine_path = Path(inspect.getfile(engine)).resolve()
        env_path = Path(inspect.getfile(env_module)).resolve()
        objects_path = Path(inspect.getfile(objects)).resolve()
        worldgen_path = Path(inspect.getfile(worldgen)).resolve()
        data_path = Path(constants.root / "data.yaml").resolve()
        chunks = engine.World.chunks
        getter = chunks.fget if isinstance(chunks, property) else None
        patch_id = getattr(getter, "_alpagd_formal_patch_id", None)
        patch_contract = getattr(getter, "_alpagd_formal_patch_contract_sha256", None)
        expected_patch_id = "alpagd_formal_20260810_crafter_deterministic_chunks_v1"
        expected_patch_contract = "da07b7d1281e2716a39ad8295155c34d78e6b8c29b02d98dc6a68fd55a31e2ba"
        return {
            "crafter_version": importlib.metadata.version("crafter"),
            "engine_path": str(engine_path), "engine_sha256": digest(engine_path),
            "env_path": str(env_path), "env_sha256": digest(env_path),
            "objects_path": str(objects_path), "objects_sha256": digest(objects_path),
            "worldgen_path": str(worldgen_path), "worldgen_sha256": digest(worldgen_path),
            "data_path": str(data_path), "data_sha256": digest(data_path),
            "formal_patch_applied": patch_id == expected_patch_id and patch_contract == expected_patch_contract,
            "formal_patch_id": patch_id,
            "formal_patch_contract_sha256": patch_contract,
            "formal_patch_module_sha256": "da02d77843735e3419a7a2c5dd74132079fa575950e8eaaed473ca2c7c61b50e",
        }

    def player_position(self) -> tuple[int, int]:
        values = self.env._player.pos
        return int(values[0]), int(values[1])

    def player_facing(self) -> tuple[int, int]:
        values = self.env._player.facing
        return int(values[0]), int(values[1])

    def inventory(self) -> Mapping[str, int]:
        return {str(key): int(value) for key, value in self.env._player.inventory.items()}

    def achievement_counts(self) -> Mapping[str, int]:
        counts = self.env._player.achievements
        if set(counts) != set(CANONICAL_ACHIEVEMENTS):
            raise PlannerError("Crafter export achievement IDs mismatch")
        return {name: int(counts[name]) for name in CANONICAL_ACHIEVEMENTS}

    def world_area(self) -> tuple[int, int]:
        area = self.env._world.area
        return int(area[0]), int(area[1])

    def cell(self, position: tuple[int, int]) -> tuple[str | None, str | None]:
        material, obj = self.env._world[position]
        object_name = None if obj is None else type(obj).__name__
        if object_name is not None and object_name not in set(self.OBJECT_NAMES.values()) | {"Player"}:
            raise PlannerError(f"unknown Crafter object type:{object_name}")
        return material, object_name


@dataclass(frozen=True)
class PlanResult:
    target: str
    success: bool
    actions: tuple[int, ...]
    action_names: tuple[str, ...]
    initial_count: int
    final_count: int
    terminal: bool
    failure_reason: str | None


def planner_sha256() -> str:
    return hashlib.sha256(Path(inspect.getfile(DeterministicMechanicsPlanner)).read_bytes()).hexdigest()


def planner_provenance() -> dict[str, Any]:
    return {
        "planner_version": PLANNER_VERSION,
        "planner_sha256": planner_sha256(),
        "policy_blind": True,
        "allowed_inputs": [
            "live_world_materials", "live_world_objects", "live_inventory",
            "live_player_position_and_facing", "raw_achievement_counts",
        ],
        "forbidden_inputs": [
            "u0_checkpoint", "u0_policy_call", "u0_logit", "u0_trajectory",
            "u0_outcome", "historical_result", "manual_action_sequence",
        ],
        "navigation": "deterministic_bfs_left_right_up_down_tie_break",
        "required_crafter_source_sha256": {
            "engine.py": PinnedCrafterMechanicsView.EXPECTED_ENGINE_SHA256,
            "env.py": PinnedCrafterMechanicsView.EXPECTED_ENV_SHA256,
            "objects.py": PinnedCrafterMechanicsView.EXPECTED_OBJECTS_SHA256,
            "worldgen.py": PinnedCrafterMechanicsView.EXPECTED_WORLDGEN_SHA256,
            "data.yaml": PinnedCrafterMechanicsView.EXPECTED_DATA_SHA256,
        },
        "required_formal_patch_sha256": {
            "contract": "da07b7d1281e2716a39ad8295155c34d78e6b8c29b02d98dc6a68fd55a31e2ba",
            "module": "da02d77843735e3419a7a2c5dd74132079fa575950e8eaaed473ca2c7c61b50e",
        },
    }


def _add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] + right[0], left[1] + right[1]


def _inside(position: tuple[int, int], area: tuple[int, int]) -> bool:
    return 0 <= position[0] < area[0] and 0 <= position[1] < area[1]


class DeterministicMechanicsPlanner:
    """Online deterministic controller; actions are applied as they are planned."""

    def __init__(self, view: MechanicsView, step: StepFn, *, horizon: int):
        self.view = view
        self.step_fn = step
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("planner horizon must be positive")
        self.actions: list[int] = []
        self.done = False

    def _act(self, name: str) -> None:
        if self.done or len(self.actions) >= self.horizon:
            raise PlannerError("terminal_or_horizon")
        if name not in ACTION_INDEX:
            raise PlannerError(f"unknown_action:{name}")
        _, _, done, info = self.step_fn(ACTION_INDEX[name])
        if "achievements" not in info or not isinstance(info["achievements"], Mapping):
            raise PlannerError("raw_achievement_counts_missing")
        self.actions.append(ACTION_INDEX[name])
        self.done = bool(done)

    def _walkable(self, position: tuple[int, int], *, allow_goal: bool = False) -> bool:
        material, obj = self.view.cell(position)
        player_here = position == self.view.player_position()
        if obj == "Player" and not player_here:
            return False
        unoccupied = obj is None or (obj == "Player" and player_here)
        return (unoccupied and material in {"grass", "path", "sand"}) or allow_goal

    def _bfs(self, goals: set[tuple[int, int]]) -> list[str] | None:
        start = self.view.player_position()
        if start in goals:
            return []
        area = self.view.world_area()
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
        found: tuple[int, int] | None = None
        while queue:
            current = queue.popleft()
            for action, delta in DIRECTIONS:
                nxt = _add(current, delta)
                if nxt in previous or not _inside(nxt, area):
                    continue
                if nxt not in goals and not self._walkable(nxt):
                    continue
                previous[nxt] = (current, action)
                if nxt in goals:
                    found = nxt
                    queue.clear()
                    break
                queue.append(nxt)
        if found is None:
            return None
        reverse: list[str] = []
        cursor = found
        while previous[cursor] is not None:
            parent, action = previous[cursor]  # type: ignore[misc]
            reverse.append(action)
            cursor = parent
        return list(reversed(reverse))

    def _adjacent_goals(self, predicate: Callable[[str | None, str | None], bool]) -> dict[tuple[int, int], str]:
        area = self.view.world_area()
        goals: dict[tuple[int, int], str] = {}
        for x in range(area[0]):
            for y in range(area[1]):
                position = (x, y)
                material, obj = self.view.cell(position)
                if not predicate(material, obj):
                    continue
                for action, delta in DIRECTIONS:
                    stand = (x - delta[0], y - delta[1])
                    if _inside(stand, area) and self._walkable(stand):
                        goals.setdefault(stand, action)
        return goals

    def _go_adjacent(self, predicate: Callable[[str | None, str | None], bool]) -> str:
        while True:
            goals = self._adjacent_goals(predicate)
            position = self.view.player_position()
            if position in goals:
                break
            path = self._bfs(set(goals))
            if path is None:
                raise PlannerError("no_reachable_goal")
            if not path:
                raise PlannerError("navigation_state_drift")
            # Dynamic objects can invalidate any cached suffix.  Execute only
            # the next deterministic action, then rebuild goals and BFS from
            # the resulting live mechanics state.
            self._act(path[0])
        face_action = goals[position]
        desired = dict(DIRECTIONS)[face_action]
        if self.view.player_facing() != desired:
            # Moving toward a blocked material/object only changes facing.
            self._act(face_action)
        return face_action

    def _collect(self, material: str) -> None:
        required = COLLECT_REQUIRE.get(material, {})
        inventory = self.view.inventory()
        if any(int(inventory.get(name, 0)) < amount for name, amount in required.items()):
            raise PlannerError("missing_tool")
        self._go_adjacent(lambda cell_material, _obj: cell_material == material)
        self._act("do")

    def _collect_sapling(self) -> None:
        initial = int(self.view.achievement_counts()["collect_sapling"])
        while int(self.view.achievement_counts()["collect_sapling"]) == initial:
            # Crafter 1.8.3 gives saplings stochastically from grass.  Each
            # failed grass cell is consumed into path, so re-run a fresh BFS.
            self._go_adjacent(lambda material, obj: material == "grass" and obj is None)
            self._act("do")

    def _interaction_lanes(self, target_predicate: Callable[[str | None, str | None], bool],
                           *, middle_predicate: Callable[[tuple[int, int]], bool] | None = None
                           ) -> dict[tuple[int, int], str]:
        area = self.view.world_area()
        lanes: dict[tuple[int, int], str] = {}
        for x in range(area[0]):
            for y in range(area[1]):
                launch = (x, y)
                if not self._walkable(launch):
                    continue
                for action, delta in DIRECTIONS:
                    middle = _add(launch, delta)
                    target = _add(middle, delta)
                    if not _inside(target, area) or not self._walkable(middle):
                        continue
                    if middle_predicate is not None and not middle_predicate(middle):
                        continue
                    if target_predicate(*self.view.cell(target)):
                        lanes.setdefault(launch, action)
        return lanes

    def _move_into_interaction_lane(
        self, target_predicate: Callable[[str | None, str | None], bool],
        *, middle_predicate: Callable[[tuple[int, int]], bool] | None = None,
    ) -> None:
        while True:
            lanes = self._interaction_lanes(target_predicate, middle_predicate=middle_predicate)
            launch = self.view.player_position()
            if launch not in lanes:
                path = self._bfs(set(lanes))
                if path is None:
                    raise PlannerError("no_reachable_interaction_lane")
                if not path:
                    raise PlannerError("interaction_lane_state_drift")
                self._act(path[0])
                continue

            action = lanes[launch]
            delta = dict(DIRECTIONS)[action]
            required_position = _add(launch, delta)
            self._act(action)
            actual_position = self.view.player_position()
            target_position = _add(actual_position, delta)
            if actual_position != required_position:
                # A dynamic blocker converted the movement into a facing-only
                # action.  Recompute all lanes instead of placing from launch.
                continue
            if (self.view.player_facing() != delta or not _inside(target_position, self.view.world_area())
                    or not target_predicate(*self.view.cell(target_position))
                    or (middle_predicate is not None and not middle_predicate(actual_position))):
                continue
            return

    def _place(self, target: str, *, near_material: str | None = None) -> None:
        uses, allowed = PLACE_RECIPE[target]
        inventory = self.view.inventory()
        if any(int(inventory.get(name, 0)) < amount for name, amount in uses.items()):
            raise PlannerError("missing_inventory")
        def middle_ok(position: tuple[int, int]) -> bool:
            if near_material is None:
                return True
            area = self.view.world_area()
            return any(
                self.view.cell((x, y))[0] == near_material
                for x in range(max(0, position[0] - 1), min(area[0], position[0] + 2))
                for y in range(max(0, position[1] - 1), min(area[1], position[1] + 2))
            )
        self._move_into_interaction_lane(
            lambda material, obj: material in allowed and obj is None,
            middle_predicate=middle_ok,
        )
        inventory = self.view.inventory()
        position = self.view.player_position()
        front = _add(position, self.view.player_facing())
        if any(int(inventory.get(name, 0)) < amount for name, amount in uses.items()):
            raise PlannerError("placement_inventory_precondition_drift")
        if (not _inside(front, self.view.world_area())
                or self.view.cell(front)[0] not in allowed or self.view.cell(front)[1] is not None
                or not middle_ok(position)):
            raise PlannerError("placement_position_precondition_drift")
        self._act(PLACE_TARGET_ACTION[target])

    def _make(self, target: str) -> None:
        uses, nearby = MAKE_RECIPE[target]
        inventory = self.view.inventory()
        if any(int(inventory.get(name, 0)) < amount for name, amount in uses.items()):
            raise PlannerError("missing_inventory")

        def required_positions() -> set[tuple[int, int]]:
            area = self.view.world_area()
            positions: set[tuple[int, int]] = set()
            for x in range(area[0]):
                for y in range(area[1]):
                    materials = {
                        self.view.cell((nx, ny))[0]
                        for nx in range(max(0, x - 1), min(area[0], x + 2))
                        for ny in range(max(0, y - 1), min(area[1], y + 2))
                    }
                    if all(name in materials for name in nearby) and self._walkable((x, y)):
                        positions.add((x, y))
            return positions

        while True:
            positions = required_positions()
            if self.view.player_position() in positions:
                break
            path = self._bfs(positions)
            if path is None:
                raise PlannerError("required_utility_not_reachable")
            if not path:
                raise PlannerError("utility_navigation_state_drift")
            self._act(path[0])
        inventory = self.view.inventory()
        if any(int(inventory.get(name, 0)) < amount for name, amount in uses.items()):
            raise PlannerError("make_inventory_precondition_drift")
        if self.view.player_position() not in required_positions():
            raise PlannerError("make_position_precondition_drift")
        self._act(MAKE_TARGET_ACTION[target])

    def _attack(self, object_name: str) -> None:
        target = {
            "Cow": "eat_cow", "Zombie": "defeat_zombie",
            "Skeleton": "defeat_skeleton", "Plant": "eat_plant",
        }[object_name]
        initial = int(self.view.achievement_counts()[target])
        while int(self.view.achievement_counts()[target]) == initial:
            self._go_adjacent(lambda _material, obj: obj == object_name)
            position = self.view.player_position()
            front = _add(position, self.view.player_facing())
            if self.view.cell(front)[1] == object_name:
                self._act("do")

    def _has_material(self, material: str) -> bool:
        area = self.view.world_area()
        return any(self.view.cell((x, y))[0] == material
                   for x in range(area[0]) for y in range(area[1]))

    def _ensure_utility(self, material: str) -> None:
        if self._has_material(material):
            return
        if material == "table":
            self._ensure_inventory("wood", 2)
            self._place("place_table")
        elif material == "furnace":
            self._ensure_utility("table")
            self._ensure_inventory("stone", 4)
            self._place("place_furnace", near_material="table")
        else:
            raise PlannerError(f"unknown_utility:{material}")
        if not self._has_material(material):
            raise PlannerError(f"utility_not_created:{material}")

    def _ensure_tool(self, item: str) -> None:
        if int(self.view.inventory().get(item, 0)) > 0:
            return
        target = f"make_{item}"
        if target not in MAKE_RECIPE:
            raise PlannerError(f"unknown_tool:{item}")
        uses, utilities = MAKE_RECIPE[target]
        for utility in utilities:
            self._ensure_utility(utility)
        # Acquiring stone/coal/iron may recursively craft prerequisite tools
        # and consume wood.  Gather those dependency-heavy ingredients first,
        # then replenish the final recipe's wood immediately before crafting.
        for ingredient, amount in uses.items():
            if ingredient == "wood":
                continue
            self._ensure_inventory(ingredient, amount)
        if "wood" in uses:
            self._ensure_inventory("wood", uses["wood"])
        inventory = self.view.inventory()
        if any(int(inventory.get(name, 0)) < amount for name, amount in uses.items()):
            raise PlannerError(f"tool_recipe_precondition_missing:{item}")
        self._make(target)
        if int(self.view.inventory().get(item, 0)) <= 0:
            raise PlannerError(f"tool_not_created:{item}")

    def _ensure_inventory(self, item: str, amount: int) -> None:
        if amount < 0:
            raise PlannerError("negative_inventory_goal")
        while int(self.view.inventory().get(item, 0)) < amount:
            if item == "wood":
                self._collect("tree")
            elif item in {"stone", "coal"}:
                self._ensure_tool("wood_pickaxe")
                self._collect(item)
            elif item == "iron":
                self._ensure_tool("stone_pickaxe")
                self._collect("iron")
            elif item == "diamond":
                self._ensure_tool("iron_pickaxe")
                self._collect("diamond")
            elif item == "sapling":
                self._collect_sapling()
            elif item.endswith("_pickaxe") or item.endswith("_sword"):
                self._ensure_tool(item)
            else:
                raise PlannerError(f"unsupported_inventory_goal:{item}")

    def run(self, target: str) -> PlanResult:
        if target not in CANONICAL_ACHIEVEMENTS or target == "wake_up":
            raise ValueError("planner supports exactly the 21 action-directed achievements")
        counts = self.view.achievement_counts()
        if set(counts) != set(CANONICAL_ACHIEVEMENTS):
            raise PlannerError("raw_achievement_count_key_mismatch")
        initial = int(counts[target])
        if initial != 0:
            return PlanResult(target, False, (), (), initial, initial, self.done, "already_achieved")
        failure: str | None = None
        try:
            if target in COLLECT_TARGET_MATERIAL:
                if target == "collect_sapling":
                    self._collect_sapling()
                else:
                    self._collect(COLLECT_TARGET_MATERIAL[target])
            elif target in PLACE_TARGET_ACTION:
                self._place(target)
            elif target in MAKE_TARGET_ACTION:
                self._make(target)
            elif target in TARGET_OBJECT:
                self._attack(TARGET_OBJECT[target])
            else:
                raise PlannerError("unsupported_target")
        except PlannerError as exc:
            failure = str(exc)
        final = int(self.view.achievement_counts()[target])
        success = final > initial
        if not success and failure is None:
            failure = "achievement_count_did_not_increase"
        return PlanResult(
            target=target,
            success=success,
            actions=tuple(self.actions),
            action_names=tuple(ACTION_NAMES[index] for index in self.actions),
            initial_count=initial,
            final_count=final,
            terminal=self.done,
            failure_reason=None if success else failure,
        )


class DeterministicHandoffPrefixPlanner(DeterministicMechanicsPlanner):
    """Prepare prerequisites without unlocking the designated target."""

    def prepare(self, designated_target: str) -> tuple[int, ...]:
        # Always produce at least one action so the handoff fingerprint is based
        # on raw post-step Crafter info rather than a synthetic reset record.
        self._act("noop")
        if designated_target == "collect_wood":
            # No prerequisite can be collected without violating the target.
            pass
        elif designated_target == "place_table":
            self._ensure_inventory("wood", 2)
            # A second placement skill is deliberately left feasible.
            self._ensure_inventory("sapling", 1)
        elif designated_target == "collect_stone":
            self._ensure_tool("wood_pickaxe")
        elif designated_target == "make_stone_pickaxe":
            self._ensure_utility("table")
            # Stone acquisition may craft a wood pickaxe and consume wood;
            # therefore recipe wood is replenished last.
            self._ensure_inventory("stone", 1)
            self._ensure_inventory("wood", 1)
        elif designated_target == "place_furnace":
            # place_stone remains a same-state feasible decoy.
            self._ensure_inventory("stone", 5)
        elif designated_target == "collect_iron":
            self._ensure_tool("stone_pickaxe")
        elif designated_target == "make_iron_pickaxe":
            self._ensure_utility("table")
            self._ensure_utility("furnace")
            # Coal/iron acquisition may construct prerequisite tools and spend
            # wood.  Acquire the deep materials first, then replenish wood.
            self._ensure_inventory("coal", 1)
            self._ensure_inventory("iron", 1)
            self._ensure_inventory("wood", 1)
        elif designated_target == "collect_diamond":
            self._ensure_tool("iron_pickaxe")
            # Preserve an immediately feasible iron-sword decoy after making
            # the pickaxe, rather than consuming the only deep ingredients.
            self._ensure_inventory("wood", 1)
            self._ensure_inventory("coal", 1)
            self._ensure_inventory("iron", 1)
        else:
            raise PlannerError("unsupported_designated_target")
        if int(self.view.achievement_counts()[designated_target]) != 0:
            raise PlannerError("prefix_unlocked_designated_target")
        return tuple(self.actions)


def mechanics_prefix_planner(env: Any, view: MechanicsView, designated_target: str,
                             _target_horizon: int) -> Sequence[int]:
    """Callable matching the builder's ``PrefixPlanner`` protocol."""
    def step(action: int):
        result = env.step(action)
        if not isinstance(result, tuple) or len(result) != 4:
            raise PlannerError("formal Crafter must expose old Gym four-tuple step API")
        return result
    # Prefixes are separately counted and bounded; leave the scientific rollout
    # horizon untouched and retain the frozen episode-length guard.
    return DeterministicHandoffPrefixPlanner(view, step, horizon=9_700).prepare(designated_target)
