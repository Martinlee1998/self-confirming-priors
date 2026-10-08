"""Fresh deterministic PC-3 handoff construction without snapshot restore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.alpagd_rl.formal_20260810.q0_state_bank.builder import PublicCrafterAdapter
from experiments.alpagd_rl.formal_20260810.q0_state_bank.planner import (
    DeterministicHandoffPrefixPlanner, PinnedCrafterMechanicsView,
)
from experiments.alpagd_rl.formal_20260810.q0_state_bank.schema import CANONICAL_ACHIEVEMENTS
from experiments.alpagd_rl.formal_20260810.training.environment_contract import make_formal_env

from .contract import PC3_TARGETS


class HandoffError(RuntimeError):
    pass


def _make_env(seed: int, ordinal: int = 0) -> PublicCrafterAdapter:
    import crafter
    env = PublicCrafterAdapter(make_formal_env(crafter.Env, seed=int(seed), length=20_000))
    env.bind_factory_ordinal(int(ordinal))
    return env


class PC3PrefixPlanner(DeterministicHandoffPrefixPlanner):
    """Collect exactly four wood, then face one legal table cell."""

    def prepare_handoff(self) -> tuple[int, ...]:
        self._act("noop")
        self._ensure_inventory("wood", 4)
        self._move_into_interaction_lane(
            lambda material, obj: material in {"grass", "sand", "path"} and obj is None,
        )
        inventory = self.view.inventory(); counts = self.view.achievement_counts()
        if int(inventory.get("wood", 0)) != 4:
            raise HandoffError("PC-3 prefix requires exactly four wood")
        if int(counts.get("collect_wood", 0)) <= 0:
            raise HandoffError("PC-3 prefix did not complete collect_wood")
        if any(int(counts.get(target, 0)) != 0 for target in PC3_TARGETS):
            raise HandoffError("PC-3 prefix completed a designated target")
        return tuple(self.actions)


@dataclass
class LiveHandoff:
    world_seed: int
    env: PublicCrafterAdapter
    view: PinnedCrafterMechanicsView
    _observation: np.ndarray
    semantic: np.ndarray
    inventory: dict[str, int]
    achievements: dict[str, int]
    player_position: tuple[int, int]
    player_facing: tuple[int, int]
    environment_time: int
    health: int
    terminal: bool
    environment_rng: Any
    prefix_length: int
    post_handoff_step: int = 0

    def observation(self) -> np.ndarray:
        return self._observation

    def step(self, action: int):
        observation, reward, done, info = self.env.step(int(action))
        self._observation = np.asarray(observation).copy()
        self.semantic = np.asarray(info["semantic"]).copy()
        self.inventory = {str(k): int(v) for k, v in info["inventory"].items()}
        self.achievements = {str(k): int(v) for k, v in info["achievements"].items()}
        self.player_position = tuple(map(int, self.view.player_position()))
        self.player_facing = tuple(map(int, self.view.player_facing()))
        self.environment_time = int(self.env._env._step)
        self.health = int(self.env._env._player.health)
        self.terminal = bool(done)
        self.environment_rng = self.env._env._world.random.get_state()
        self.post_handoff_step += 1
        return observation, reward, done, info

    def close(self) -> None:
        self.env.close()


def plan_prefix(world_seed: int) -> tuple[int, ...]:
    env = _make_env(world_seed)
    try:
        env.reset(); view = PinnedCrafterMechanicsView(env._env)
        return PC3PrefixPlanner(view, env.step, horizon=9_500).prepare_handoff()
    finally:
        env.close()


def replay_prefix(world_seed: int, prefix: Sequence[int], *, ordinal: int = 0) -> LiveHandoff:
    """Replay in a fresh env; never deserialize or restore an environment state."""
    env = _make_env(world_seed, ordinal)
    try:
        observation = env.reset(); info: Mapping[str, Any] | None = None; done = False
        for action in prefix:
            if done:
                raise HandoffError("PC-3 prefix continued after terminal")
            observation, _reward, done, info = env.step(int(action))
        if info is None or done:
            raise HandoffError("PC-3 replay did not reach a live stepped handoff")
        view = PinnedCrafterMechanicsView(env._env)
        inventory = {str(k): int(v) for k, v in view.inventory().items()}
        achievements = {str(k): int(v) for k, v in view.achievement_counts().items()}
        if set(achievements) != set(CANONICAL_ACHIEVEMENTS):
            raise HandoffError("PC-3 achievement schema drift")
        position = view.player_position(); facing = view.player_facing()
        front = (position[0] + facing[0], position[1] + facing[1])
        material, obj = view.cell(front)
        forbidden = ("wood_pickaxe", "stone", "stone_pickaxe", "iron")
        if (int(inventory.get("wood", 0)) != 4
                or any(int(inventory.get(name, 0)) != 0 for name in forbidden)
                or any(int(achievements[target]) != 0 for target in PC3_TARGETS)
                or int(achievements["collect_wood"]) <= 0
                or material not in {"grass", "sand", "path"} or obj is not None
                or any(view.cell((x, y))[0] == "table"
                       for x in range(view.world_area()[0]) for y in range(view.world_area()[1]))):
            raise HandoffError("PC-3 replay handoff qualification failed")
        semantic = np.asarray(info["semantic"]).copy()
        return LiveHandoff(
            int(world_seed), env, view, np.asarray(observation).copy(), semantic,
            inventory, achievements, tuple(map(int, view.player_position())),
            tuple(map(int, view.player_facing())), int(env._env._step),
            int(env._env._player.health), bool(done), env._env._world.random.get_state(),
            len(prefix),
        )
    except BaseException:
        env.close()
        raise


def assert_equal_handoff(left: LiveHandoff, right: LiveHandoff) -> None:
    """Required in-memory equality only; no serialization/fingerprint/hash layer."""
    def rng_equal(a, b):
        return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    if (not np.array_equal(left.observation(), right.observation())
            or not np.array_equal(left.semantic, right.semantic)
            or left.inventory != right.inventory
            or left.achievements != right.achievements
            or left.player_position != right.player_position
            or left.player_facing != right.player_facing
            or left.environment_time != right.environment_time
            or left.health != right.health
            or left.terminal != right.terminal
            or not rng_equal(left.environment_rng, right.environment_rng)
            or left.post_handoff_step != 0 or right.post_handoff_step != 0):
        raise HandoffError("PC-3 fresh replay equality failed")


def open_arm_handoffs(world_seed: int, arms: Sequence[str]) -> tuple[tuple[int, ...], dict[str, LiveHandoff]]:
    """Plan once, then replay the same prefix into one fresh env per arm."""
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("PC-3 arms must be a nonempty unique sequence")
    prefix = plan_prefix(world_seed)
    result: dict[str, LiveHandoff] = {}
    try:
        for ordinal, arm in enumerate(arms):
            result[str(arm)] = replay_prefix(world_seed, prefix, ordinal=ordinal)
        reference = next(iter(result.values()))
        for handoff in result.values():
            assert_equal_handoff(reference, handoff)
        return prefix, result
    except BaseException:
        for handoff in result.values(): handoff.close()
        raise
