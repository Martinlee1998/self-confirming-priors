from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .memory import require_torch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except ImportError:  # Import remains usable for dependency diagnostics.
    torch = None
    nn = None
    functional = None
    activation_checkpoint = None


ModuleBase = nn.Module if nn is not None else object


class FanInInitReLULayer(ModuleBase):
    """Exact initialization/normalization semantics from pinned AD torch_util.py."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        require_torch()
        super().__init__()
        self.norm = None
        if group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, in_channels)
        elif layer_norm:
            self.norm = nn.LayerNorm(in_channels)
        layer_cls = {"conv": nn.Conv2d, "linear": nn.Linear}[layer_type]
        self.layer = layer_cls(
            in_channels, out_channels, bias=self.norm is None, **layer_kwargs
        )
        self.use_activation = use_activation
        with torch.no_grad():
            axes = tuple(range(1, self.layer.weight.ndim))
            self.layer.weight.mul_(
                init_scale / self.layer.weight.norm(dim=axes, p=2, keepdim=True)
            )
            if self.layer.bias is not None:
                self.layer.bias.zero_()

    def forward(self, inputs):
        if self.norm is not None:
            inputs = self.norm(inputs)
        outputs = self.layer(inputs)
        return functional.relu(outputs, inplace=True) if self.use_activation else outputs


class CnnBasicBlock(ModuleBase):
    def __init__(self, channels: int, init_scale: float, group_norm_groups: int = 1):
        require_torch()
        super().__init__()
        scale = math.sqrt(init_scale)
        kwargs = {"group_norm_groups": group_norm_groups}
        self.conv0 = FanInInitReLULayer(
            channels, channels, kernel_size=3, padding=1, init_scale=scale, **kwargs
        )
        self.conv1 = FanInInitReLULayer(
            channels, channels, kernel_size=3, padding=1, init_scale=scale, **kwargs
        )

    def forward(self, inputs):
        return inputs + self.conv1(self.conv0(inputs))


class CnnDownStack(ModuleBase):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        blocks: int,
        *,
        init_scale: float,
        first_conv_norm: bool,
    ):
        require_torch()
        super().__init__()
        self.firstconv = FanInInitReLULayer(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            group_norm_groups=1 if first_conv_norm else None,
        )
        self.n = nn.GroupNorm(1, output_channels)
        self.blocks = nn.ModuleList(
            CnnBasicBlock(
                output_channels,
                init_scale=init_scale / math.sqrt(blocks),
                group_norm_groups=1,
            )
            for _ in range(blocks)
        )

    def forward(self, inputs):
        outputs = self.firstconv(inputs)
        outputs = functional.max_pool2d(outputs, kernel_size=3, stride=2, padding=1)
        outputs = self.n(outputs)
        for block in self.blocks:
            outputs = block(outputs)
        return outputs


class ImpalaResNetEncoder(ModuleBase):
    """Pinned AD `ImpalaCNN`: [64,128,128], two residual blocks, output 256."""

    def __init__(self, output_dim: int = 256, channels=(64, 128, 128), blocks=2):
        require_torch()
        super().__init__()
        stacks = []
        input_channels = 3
        spatial = 64
        for index, output_channels in enumerate(channels):
            stacks.append(
                CnnDownStack(
                    input_channels,
                    output_channels,
                    blocks,
                    init_scale=1.0 / math.sqrt(len(channels)),
                    first_conv_norm=index > 0,
                )
            )
            input_channels = output_channels
            spatial = (spatial + 1) // 2
        self.stacks = nn.ModuleList(stacks)
        self.dense = FanInInitReLULayer(
            input_channels * spatial * spatial,
            output_dim,
            layer_type="linear",
            init_scale=1.4,
            layer_norm=True,
        )

    def forward(self, inputs):
        for stack in self.stacks:
            inputs = stack(inputs)
        return self.dense(inputs.reshape(inputs.shape[0], -1))


class CategoricalActionHead(ModuleBase):
    def __init__(self, input_dim: int, action_count: int):
        require_torch()
        super().__init__()
        self.linear = nn.Linear(input_dim, action_count)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return functional.log_softmax(self.linear(inputs), dim=-1)


class NormalizeEwma(ModuleBase):
    def __init__(self, size: int, beta: float = 0.99, epsilon: float = 1e-2):
        require_torch()
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.running_mean = nn.Parameter(torch.zeros(size), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(size), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def running_mean_var(self):
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        return mean, (mean_sq - mean**2).clamp(min=1e-2)

    def forward(self, inputs):
        if self.training:
            detached = inputs.detach()
            self.running_mean.mul_(self.beta).add_(detached.mean(0) * (1.0 - self.beta))
            self.running_mean_sq.mul_(self.beta).add_((detached**2).mean(0) * (1.0 - self.beta))
            self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)
        mean, variance = self.running_mean_var()
        return (inputs - mean[None]) / torch.sqrt(variance[None])

    def denormalize(self, inputs):
        mean, variance = self.running_mean_var()
        return inputs * torch.sqrt(variance[None]) + mean[None]


class ScaledMSEHead(ModuleBase):
    def __init__(self, input_dim: int):
        require_torch()
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.orthogonal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
        self.normalizer = NormalizeEwma(1)

    def forward(self, inputs):
        return self.linear(inputs)

    def normalize(self, inputs):
        return self.normalizer(inputs)

    def denormalize(self, inputs):
        return self.normalizer.denormalize(inputs)

    def mse_loss(self, prediction, target):
        return functional.mse_loss(prediction, self.normalizer(target), reduction="none")


class MLP(ModuleBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int = 1):
        require_torch()
        super().__init__()
        sizes = [input_dim] + hidden_layers * [hidden_dim] + [output_dim]
        self.layers = nn.ModuleList()
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            use_activation = index < hidden_layers
            self.layers.append(
                FanInInitReLULayer(
                    input_size,
                    output_size,
                    layer_type="linear",
                    init_scale=1.4 if use_activation else 1.0,
                    layer_norm=use_activation,
                    use_activation=use_activation,
                )
            )

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def _formal_image_to_float(image):
    if image.ndim != 4 or tuple(image.shape[1:]) not in {(64, 64, 3), (3, 64, 64)}:
        raise ValueError("formal image must be [B,64,64,3] or [B,3,64,64]")
    if image.dtype != torch.uint8:
        raise TypeError("formal model input contract is uint8; runtime must not pass inferred-scale floats")
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    return image.float().div_(255.0)


def build_u0_side_vector(target, achieved, target_age):
    """Build side vector from the deployment contract's normalized age in [0, 1]."""
    require_torch()
    if target.ndim != 2 or target.shape[-1] != 22:
        raise ValueError("target must have shape [batch,22]")
    if achieved.shape != target.shape:
        raise ValueError("achieved must have the same [batch,22] shape as target")
    if target_age.ndim == 1:
        target_age = target_age.unsqueeze(-1)
    if target_age.shape != (target.shape[0], 1):
        raise ValueError("target_age must have shape [batch] or [batch,1]")
    age = target_age.float().clamp_(0.0, 1.0)
    return torch.cat([target.float(), achieved.float(), age], dim=-1)


def apply_null_target_dropout(target, probability: float, *, generator=None, training=True):
    require_torch()
    if not training or probability == 0.0:
        return target
    if not 0.0 <= probability <= 1.0:
        raise ValueError("dropout probability must be in [0,1]")
    draws = torch.rand((target.shape[0], 1), device=target.device, generator=generator)
    return torch.where(draws < probability, torch.zeros_like(target), target)


class PolicyValueNet(ModuleBase):
    """Exact E0/E1 head path; U0 adds only post-encoder goal fusion."""

    def __init__(self, system_id: str, action_count: int = 17, hidden_dim: int = 1024):
        require_torch()
        super().__init__()
        if system_id not in {"E0", "E1", "U0"}:
            raise ValueError(f"unknown system {system_id!r}")
        self.system_id = system_id
        self.hidden_dim = hidden_dim
        self.enc = ImpalaResNetEncoder(output_dim=256)
        self.linear = FanInInitReLULayer(
            256, hidden_dim, layer_type="linear", layer_norm=True
        )
        if system_id == "U0":
            self.side_encoder = MLP(45, 128, hidden_dim=128, hidden_layers=1)
            self.goal_fusion = FanInInitReLULayer(
                hidden_dim + 128, hidden_dim, layer_type="linear", layer_norm=True
            )
            head_input = hidden_dim
        elif system_id == "E1":
            head_input = 2 * hidden_dim
        else:
            head_input = hidden_dim
        self.pi_head = CategoricalActionHead(head_input, action_count)
        self.vf_head = ScaledMSEHead(head_input)
        if system_id == "E1":
            self.action_mlp = MLP(action_count, 2 * hidden_dim, hidden_dim)
            self.next_goal_pred_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim)

    def _encode_without_checkpoint(self, image):
        return self.linear(self.enc(_formal_image_to_float(image)))

    def encode(self, image, *, checkpoint_encoder: bool = False):
        """Encode pixels, optionally recomputing the deterministic E1 CNN in backward.

        Activation checkpointing is deliberately opt-in.  Formal PPO, inference,
        the frozen old model, and every no-grad auxiliary target retain the exact
        eager path.  The encoder contains only convolutions, GroupNorm/LayerNorm,
        ReLU, max-pooling, and linear layers: it has no stochastic operation or
        running-state mutation, so non-reentrant recomputation is algorithmically
        neutral.
        """
        if not checkpoint_encoder:
            return self._encode_without_checkpoint(image)
        if self.system_id != "E1":
            raise ValueError("encoder activation checkpointing is restricted to E1 auxiliary")
        if not torch.is_grad_enabled():
            raise RuntimeError("checkpoint_encoder requires a gradient-enabled auxiliary path")
        return activation_checkpoint(
            self._encode_without_checkpoint, image, use_reentrant=False
        )

    def forward(
        self,
        image,
        *,
        target=None,
        achieved=None,
        target_age=None,
        memory=None,
        target_dropout_probability: float = 0.0,
        dropout_generator=None,
        checkpoint_encoder: bool = False,
    ) -> dict[str, Any]:
        image_latent = self.encode(image, checkpoint_encoder=checkpoint_encoder)
        if self.system_id == "U0":
            if target is None or achieved is None or target_age is None:
                raise ValueError("U0 requires target, achieved, and target_age")
            target = apply_null_target_dropout(
                target, target_dropout_probability, generator=dropout_generator, training=self.training
            )
            side = build_u0_side_vector(target, achieved, target_age)
            policy_latent = self.goal_fusion(torch.cat([image_latent, self.side_encoder(side)], -1))
        elif self.system_id == "E1":
            if memory is None or memory.shape != (image.shape[0], self.hidden_dim):
                raise ValueError(f"E1 memory must have shape [batch,{self.hidden_dim}]")
            policy_latent = torch.cat([image_latent, memory], -1)
        else:
            if any(item is not None for item in (target, achieved, target_age, memory)):
                raise ValueError("E0 is pixels-only")
            policy_latent = image_latent
        return {
            "image_latent": image_latent,
            "latent": policy_latent,
            "policy_logits": self.pi_head(policy_latent),
            "value": self.vf_head(policy_latent),
        }

    def achievement_state(
        self, goal_image, next_goal_image, *, checkpoint_encoder: bool = False
    ):
        return functional.normalize(
            self.encode(next_goal_image, checkpoint_encoder=checkpoint_encoder)
            - self.encode(goal_image, checkpoint_encoder=checkpoint_encoder),
            dim=-1,
        )

    def act(self, image, **kwargs):
        if self.training:
            raise RuntimeError("act requires eval mode")
        with torch.no_grad():
            outputs = self.forward(image, **kwargs)
            distribution = torch.distributions.Categorical(logits=outputs["policy_logits"])
            actions = distribution.sample().unsqueeze(-1)
            outputs["actions"] = actions
            outputs["log_probabilities"] = outputs["policy_logits"].gather(-1, actions)
            outputs["denormalized_value"] = self.vf_head.denormalize(outputs["value"])
        return outputs

    def predict_next_achievement_state(self, image_latent, actions, memory):
        if self.system_id != "E1":
            raise RuntimeError("achievement distillation is only defined for E1")
        if actions.ndim == 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        one_hot = functional.one_hot(actions.long(), num_classes=self.pi_head.linear.out_features).float()
        gamma, beta = self.action_mlp(one_hot).chunk(2, dim=-1)
        conditioned = (1.0 + gamma) * image_latent + beta
        prediction = self.next_goal_pred_mlp(torch.cat([conditioned, memory], dim=-1))
        return functional.normalize(prediction, dim=-1)

    def load_upstream_state_dict(self, state_dict):
        if self.system_id not in {"E0", "E1"}:
            raise RuntimeError("upstream weight mapping is defined only for E0/E1")
        return self.load_state_dict(state_dict, strict=True)


def achievement_distillation_contrastive_loss(
    anchor_state, positive_prediction, negative_prediction, temperature=0.1
):
    require_torch()
    positive = torch.einsum("bi,bi->b", anchor_state, positive_prediction)
    negative = torch.einsum("bi,bi->b", anchor_state, negative_prediction)
    logits = torch.stack([positive, negative], dim=-1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return functional.cross_entropy(logits, labels)


@dataclass(frozen=True)
class PPOLoss:
    total: Any
    policy: Any
    value: Any
    entropy: Any


def clipped_ppo_loss(
    policy_logits,
    values,
    actions,
    old_log_probabilities,
    advantages,
    returns,
    *,
    clip_ratio=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    value_head=None,
) -> PPOLoss:
    require_torch()
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)
    if old_log_probabilities.ndim == 1:
        old_log_probabilities = old_log_probabilities.unsqueeze(-1)
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    new_log_probabilities = policy_logits.gather(-1, actions.long())
    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    if value_head is None:
        raise ValueError("formal PPO loss requires the model value_head normalizer")
    if returns.ndim == 1:
        returns = returns.unsqueeze(-1)
    if returns.shape != values.shape:
        raise ValueError("returns and normalized value predictions must both be [batch,1]")
    value = value_head.mse_loss(values, returns).mean()
    entropy = -(policy_logits.exp() * policy_logits).sum(dim=-1).mean()
    total = policy + value_coefficient * value - entropy_coefficient * entropy
    return PPOLoss(total=total, policy=policy, value=value, entropy=entropy)
