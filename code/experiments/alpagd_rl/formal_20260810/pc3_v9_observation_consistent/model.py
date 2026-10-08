"""Target-free recurrent belief core with post-GRU A-E option heads."""
from __future__ import annotations

import torch
from torch import nn

from experiments.alpagd_rl.formal_20260810.training.models import ImpalaResNetEncoder

ACTION_COUNT = 17
HUD_COUNT = 16
ACHIEVEMENT_COUNT = 22
OPTION_COUNT = 5


class TargetFreeRecurrentPolicy(nn.Module):
    """The recurrent core sees pixels and public agent state/ego-motion only."""

    def __init__(self, *, hidden_dim: int = 256, encoder_dim: int = 256,
                 encoder: nn.Module | None = None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = encoder if encoder is not None else ImpalaResNetEncoder(output_dim=encoder_dim)
        self.prev_action_embedding = nn.Embedding(ACTION_COUNT + 1, 32)
        side_dim = HUD_COUNT + ACHIEVEMENT_COUNT * 2 + 2 + 2 + 32
        self.public_side = nn.Sequential(nn.Linear(side_dim, 160), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(encoder_dim + 160, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # Target identity and target age first enter here, strictly after GRU.
        post_dim = hidden_dim + OPTION_COUNT + 1
        self.option_heads = nn.ModuleList([nn.Linear(post_dim, ACTION_COUNT) for _ in range(OPTION_COUNT)])
        self.value_head = nn.Linear(post_dim, 1)

    def initial_hidden(self, batch_size: int, *, device=None) -> torch.Tensor:
        return torch.zeros(int(batch_size), self.hidden_dim, device=device)

    @staticmethod
    def normalize_hud(hud: torch.Tensor) -> torch.Tensor:
        return hud.float().clamp(min=0.0, max=9.0) / 9.0

    def forward(self, pixels: torch.Tensor, hud16: torch.Tensor, achieved22: torch.Tensor,
                prev_action: torch.Tensor, prev_observable_achievement_delta: torch.Tensor,
                ego_delta: torch.Tensor, facing: torch.Tensor, episode_boundary: torch.Tensor,
                target_onehot5: torch.Tensor, normalized_target_age: torch.Tensor,
                hidden: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if pixels.ndim != 5 or pixels.shape[-3:] != (64, 64, 3):
            raise ValueError("pixels must be [T,B,64,64,3]")
        time, batch = pixels.shape[:2]
        if hud16.shape != (time, batch, HUD_COUNT): raise ValueError("hud16 must be [T,B,16]")
        if achieved22.shape != (time, batch, ACHIEVEMENT_COUNT): raise ValueError("achieved22 must be [T,B,22]")
        if prev_observable_achievement_delta.shape != achieved22.shape: raise ValueError("public event must be [T,B,22]")
        if ego_delta.shape != (time, batch, 2) or facing.shape != (time, batch, 2):
            raise ValueError("ego delta and facing must be [T,B,2]")
        if prev_action.shape != (time, batch) or episode_boundary.shape != (time, batch):
            raise ValueError("previous action and boundary must be [T,B]")
        if target_onehot5.shape != (time, batch, OPTION_COUNT): raise ValueError("target must be [T,B,5]")
        if normalized_target_age.shape not in {(time, batch), (time, batch, 1)}:
            raise ValueError("target age must be [T,B] or [T,B,1]")
        if torch.any(target_onehot5.sum(-1) != 1): raise ValueError("exactly one A-E option must be selected")
        if prev_action.numel() and (prev_action.min().item() < -1 or prev_action.max().item() >= ACTION_COUNT):
            raise ValueError("previous action out of range")
        image = pixels.reshape(time * batch, 64, 64, 3).permute(0, 3, 1, 2).float() / 255.0
        visual = self.encoder(image).reshape(time, batch, -1)
        action = self.prev_action_embedding((prev_action.long() + 1).clamp(0, ACTION_COUNT))
        public = torch.cat((self.normalize_hud(hud16), achieved22.float(),
                            prev_observable_achievement_delta.float(), ego_delta.float(),
                            facing.float(), action), dim=-1)
        fused = self.fusion(torch.cat((visual, self.public_side(public)), dim=-1))
        state = self.initial_hidden(batch, device=pixels.device) if hidden is None else hidden
        if state.shape != (batch, self.hidden_dim): raise ValueError("hidden must be [B,hidden_dim]")
        states, before = [], []
        for index in range(time):
            keep = (~episode_boundary[index].bool()).float().unsqueeze(-1)
            state = state * keep
            before.append(state)
            state = self.gru(fused[index], state)
            states.append(state)
        recurrent = torch.stack(states)
        age = normalized_target_age.reshape(time, batch, 1).float()
        post = torch.cat((recurrent, target_onehot5.float(), age), dim=-1)
        all_logits = torch.stack([head(post) for head in self.option_heads], dim=2)
        option_index = target_onehot5.argmax(-1)
        gather = option_index[..., None, None].expand(time, batch, 1, ACTION_COUNT)
        selected_logits = all_logits.gather(2, gather).squeeze(2)
        return {"policy_logits": selected_logits, "all_option_logits": all_logits,
                "value": self.value_head(post).squeeze(-1), "hidden_sequence": recurrent,
                "hidden_before_step": torch.stack(before), "final_hidden": state}
