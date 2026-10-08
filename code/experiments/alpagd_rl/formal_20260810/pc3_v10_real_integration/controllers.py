"""Shared architecture and inputs for the four frozen controllers.

Author: Yanqi Li
"""
from __future__ import annotations

from typing import Any, Mapping

CONTROLLER_SEEDS = (2026081911, 2026081912, 2026081913, 2026081914)
TARGET_ORDER = ("A", "B", "C", "D", "E")
TEACHER_TARGETS = {
    "A": "place_table",
    "B": "make_wood_pickaxe",
    "C": "collect_stone",
    "D": "make_stone_pickaxe",
    "E": "collect_iron",
}


def _move_to_device(values: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {name: value.to(device) for name, value in values.items()}


def _finite(*values: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(value).all().item()) for value in values)


def _new_model_optimizer(*, controller_seed: int, device: Any) -> tuple[Any, Any]:
    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_v9_observation_consistent.model import (
        TargetFreeRecurrentPolicy,
    )

    torch.manual_seed(int(controller_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(controller_seed))
    model = TargetFreeRecurrentPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    return model, optimizer
