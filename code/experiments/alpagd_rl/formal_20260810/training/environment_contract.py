from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable


FORMAL_PATCH_CONTRACT_SHA256 = "da07b7d1281e2716a39ad8295155c34d78e6b8c29b02d98dc6a68fd55a31e2ba"
FORMAL_PATCH_MODULE_SHA256 = "da02d77843735e3419a7a2c5dd74132079fa575950e8eaaed473ca2c7c61b50e"


@dataclass(frozen=True)
class FormalEnvironmentContract:
    crafter_version: str = "1.8.3"
    observation_shape: tuple[int, int, int] = (64, 64, 3)
    action_count: int = 17
    achievement_count: int = 22
    episode_steps: int = 10_000


def patch_fingerprint(patch_bytes: bytes) -> str:
    return hashlib.sha256(patch_bytes).hexdigest()


def make_formal_env(
    env_constructor: Callable,
    *,
    patch_installer: Callable[[], dict[str, Any]] | None = None,
    **env_kwargs,
):
    """Install the single frozen runtime patch before constructing Crafter."""
    if patch_installer is None:
        from experiments.alpagd_rl.formal_20260810.runtime.patching import (
            install_worker_determinism,
        )

        patch_installer = install_worker_determinism
    evidence = patch_installer()
    expected = {
        "patch_contract_sha256": FORMAL_PATCH_CONTRACT_SHA256,
        "patch_module_sha256": FORMAL_PATCH_MODULE_SHA256,
        "patch_applied": True,
    }
    observed = {key: evidence.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"formal deterministic patch evidence mismatch: {observed!r}")
    env = env_constructor(**env_kwargs)
    return env


def successes_from_info(info: dict, achievement_names: tuple[str, ...]):
    """Runtime adapter expected by E1; no reward-threshold fallback is accepted."""
    if "successes" in info:
        successes = info["successes"]
    elif "achievements" in info:
        successes = [int(info["achievements"].get(name, 0) > 0) for name in achievement_names]
    else:
        raise KeyError("Crafter info must expose successes or achievements")
    if len(successes) != 22:
        raise ValueError("E1 requires exactly 22 success bits")
    return successes
