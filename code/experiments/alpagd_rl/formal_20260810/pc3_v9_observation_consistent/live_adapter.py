"""Pinned handoff/view copier; the sole live-to-ObservableStep boundary."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.alpagd_rl.formal_20260810.oracle.frontier import CANONICAL_ACHIEVEMENT_IDS

from .observable import INVENTORY_NAMES, LocalSemanticCell, ObservableStep, adjacent_ego_delta


class LiveObservableAdapter:
    """Keeps one private previous absolute position; emits no absolute provenance."""

    __slots__ = ("_previous_absolute_position", "_previous_achieved", "_closed_episode")

    def __init__(self) -> None:
        self._previous_absolute_position: tuple[int, int] | None = None
        self._previous_achieved: tuple[int, ...] | None = None
        self._closed_episode = False

    def reset_episode(self) -> None:
        self._previous_absolute_position = None
        self._previous_achieved = None
        self._closed_episode = False

    def capture(self, handoff, *, prev_action: int, terminated: bool = False,
                truncated: bool = False) -> ObservableStep:
        if self._closed_episode:
            raise RuntimeError("adapter requires explicit reset before a new episode")
        view = handoff.view
        current_absolute = tuple(map(int, view.player_position()))
        delta = adjacent_ego_delta(self._previous_absolute_position, current_absolute)
        inventory: Mapping[str, int] = view.inventory()
        achieved = tuple(int(handoff.achievements.get(name, 0) > 0)
                         for name in CANONICAL_ACHIEVEMENT_IDS)
        event = (0,) * 22 if self._previous_achieved is None else tuple(
            max(0, current - previous) for current, previous in zip(achieved, self._previous_achieved))
        cells = tuple(LocalSemanticCell(dx, dy, *view.cell((current_absolute[0] + dx,
                                                            current_absolute[1] + dy)))
                      for dx in range(-4, 5) for dy in range(-3, 4))
        result = ObservableStep(
            np.asarray(handoff.observation(), dtype=np.uint8).copy(), cells,
            tuple(int(inventory[name]) for name in INVENTORY_NAMES),
            tuple(map(int, view.player_facing())), achieved, event, int(prev_action), delta,
            bool(terminated), bool(truncated),
        )
        self._previous_absolute_position = current_absolute
        self._previous_achieved = achieved
        self._closed_episode = bool(terminated or truncated)
        return result
