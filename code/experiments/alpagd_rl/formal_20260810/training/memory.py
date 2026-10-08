from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # Allows configuration/static-contract tests without Torch.
    torch = None
    torch_functional = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required; run bootstrap_formal_env.ps1/.sh first")


@dataclass
class AchievementMemoryState:
    latent: object
    successes: object

    def state_dict(self) -> dict[str, object]:
        return {"latent": self.latent, "successes": self.successes}


class AchievementLatentMemory:
    """Port of the pinned AD training-rollout event memory.

    A newly changed bit in ``info['successes']`` replaces memory with the
    normalized next-latent minus the pre-action latent. Episode termination
    clears both memory and the success mask. This deliberately does not use
    reward thresholds from the upstream standalone evaluator.
    """

    def __init__(self, latent_dim: int, achievement_count: int = 22):
        require_torch()
        self.latent_dim = latent_dim
        self.achievement_count = achievement_count

    def zeros(self, batch_size: int, *, device=None, dtype=None) -> AchievementMemoryState:
        latent = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        successes = torch.zeros(
            batch_size, self.achievement_count, device=device, dtype=torch.bool
        )
        return AchievementMemoryState(latent=latent, successes=successes)

    def update(
        self,
        state: AchievementMemoryState,
        pre_action_latent,
        next_observation_latent,
        successes,
        episode_done,
    ) -> AchievementMemoryState:
        successes = successes.to(dtype=torch.bool)
        done = episode_done.to(dtype=torch.bool).reshape(-1, 1)
        if successes.shape != state.successes.shape:
            raise ValueError("successes must have shape [batch, 22]")
        changed = (successes != state.successes).any(dim=-1, keepdim=True)
        event_latent = torch_functional.normalize(
            next_observation_latent - pre_action_latent, dim=-1
        )
        latent = torch.where(changed, event_latent, state.latent)
        latent = torch.where(done, torch.zeros_like(latent), latent)
        retained_successes = torch.where(done, torch.zeros_like(successes), successes)
        return AchievementMemoryState(latent=latent, successes=retained_successes)

    def reset(self, state: AchievementMemoryState, reset_mask=None) -> AchievementMemoryState:
        if reset_mask is None:
            return self.zeros(
                state.latent.shape[0], device=state.latent.device, dtype=state.latent.dtype
            )
        mask = reset_mask.to(dtype=torch.bool).reshape(-1, 1)
        return AchievementMemoryState(
            latent=torch.where(mask, torch.zeros_like(state.latent), state.latent),
            successes=torch.where(mask, torch.zeros_like(state.successes), state.successes),
        )
