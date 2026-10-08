"""Actual fixed-order contrast evaluator for the four PC3-v10 controllers.

This is deliberately independent of the zero-world runtime and of the
training entrypoints.  A formal invocation accepts only four *final* PPO
behaviour-policy checkpoints and runs the matched five-arm experiment:

* Correct = A,B,C,D,E;
* P1 = B,A,C,D,E; P2 = A,C,B,D,E; P3 = A,B,D,C,E;
  P4 = A,B,C,E,D.

Each controller/world/stream row plans one real long-horizon handoff, replays
that identical base state into five fresh Crafter environments, gives every
arm the same action-uniform at every global step, and preserves every arm.
The endpoint is always canonical A->E prefix depth / 5; it is never an
arm-relative achievement score.  E-hit, true terminal, and the exact 10,000
post-handoff actions are the only stopping conditions.

The ``--development-wiring`` mode is intentionally much shorter and uses a
separate non-formal controller/world.  It exercises only the real handoff,
adapter, recurrent carry, scheduler and common-U path.  It neither loads a
BC intermediate checkpoint nor emits a formal outcome.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from controllers import (
    CONTROLLER_SEEDS,
    TARGET_ORDER,
    TEACHER_TARGETS,
    _finite,
    _move_to_device,
    _new_model_optimizer,
)


ARMS = ("Correct", "P1", "P2", "P3", "P4")
ARM_ORDERS = {
    "Correct": ("A", "B", "C", "D", "E"),
    "P1": ("B", "A", "C", "D", "E"),
    "P2": ("A", "C", "B", "D", "E"),
    "P3": ("A", "B", "D", "C", "E"),
    "P4": ("A", "B", "C", "E", "D"),
}
TARGET_WINDOW = 1_000
FORMAL_HORIZON = 10_000
FORMAL_STREAMS = (0, 1)
FORMAL_WORLD_LOWER = 1_180_001
FORMAL_WORLD_UPPER = 1_189_999
FORMAL_WORLD_COUNT = 100
PPO_FINAL_ROLLOUTS = 2_560
PPO_MICROSTEPS_PER_ROLLOUT = 32
PPO_CHECKPOINT_SCHEMA = "pc3-v10-real-ppo-checkpoint/v1"

DEVELOPMENT_NAMESPACE = "pc3-v10-real-fixed-contrast-development-20260820"
DEVELOPMENT_CONTROLLER_SEED = 2_026_089_001
DEVELOPMENT_WORLD_SEED = 2_026_089_501
DEVELOPMENT_STREAM = 0
DEVELOPMENT_SHORT_ACTIONS = 32
WIRING_SCHEMA = "pc3-v10-real-fixed-contrast-wiring/v1"
RESULT_SCHEMA = "pc3-v10-real-fixed-contrast-result/v1"
SHARD_SCHEMA = "pc3-v10-real-fixed-contrast-controller-shard/v1"

# Exact t(.975, 99), fixed by the prospective four-controller / 100-world
# endpoint.  It is a reporting constant, never a training or selection gate.
T_975_DF99 = 1.9842169515086827


class FixedContrastError(RuntimeError):
    """A real contrast runtime or final-checkpoint prerequisite failed."""


def _achievement_state(achievements: Mapping[str, Any]) -> dict[str, bool]:
    """Project the public real achievement counts to the canonical A--E map."""

    state: dict[str, bool] = {}
    for code in TARGET_ORDER:
        name = TEACHER_TARGETS[code]
        if name not in achievements:
            raise FixedContrastError(f"real handoff is missing achievement {name}")
        state[code] = bool(int(achievements[name]) > 0)
    return state


def _canonical_depth(achieved: Mapping[str, bool]) -> int:
    depth = 0
    for code in TARGET_ORDER:
        if achieved.get(code) is not True:
            break
        depth += 1
    return depth


def _arm_frontier(arm: str, achieved: Mapping[str, bool]) -> str | None:
    if arm not in ARM_ORDERS:
        raise FixedContrastError("unknown fixed contrast arm")
    for code in ARM_ORDERS[arm]:
        if achieved.get(code) is not True:
            return code
    return None


@dataclass(frozen=True)
class FixedContrastScheduler:
    """Local v10-equivalent target-age state for any fixed arm order.

    The public formal scheduler is intentionally range-restricted to formal
    worlds.  This tiny state machine has the same A--E window/age semantics
    while being valid for the isolated development wiring world too.
    """

    arm: str
    target: str | None = None
    window_actions: int = 0
    target_age_actions: int = 0

    def __post_init__(self) -> None:
        if self.arm not in ARM_ORDERS:
            raise ValueError("arm must be Correct or P1--P4")
        if self.target is not None and self.target not in TARGET_ORDER:
            raise ValueError("target must be A--E or None")
        if type(self.window_actions) is not int or not 0 <= self.window_actions < TARGET_WINDOW:
            raise ValueError("v10 target window must be in [0, 1000)")
        if type(self.target_age_actions) is not int or self.target_age_actions < 0:
            raise ValueError("v10 target age must be a nonnegative exact int")
        if self.target is None and (self.window_actions or self.target_age_actions):
            raise ValueError("inactive scheduler cannot retain age/window state")

    @property
    def pre_action_target_age(self) -> float:
        # The policy receives exactly float32 min(age, 1000) / 1000.
        return float(np.float32(min(self.target_age_actions, TARGET_WINDOW) / TARGET_WINDOW))

    def select(self, achieved: Mapping[str, bool]) -> "FixedContrastScheduler":
        selected = _arm_frontier(self.arm, achieved)
        if selected == self.target:
            return self
        # Only a true identity change clears age.  This handles all ordinary
        # target hits and skips already-achieved out-of-order targets.
        return FixedContrastScheduler(self.arm, selected, 0, 0)

    def after_action(self, achieved_after: Mapping[str, bool]) -> "FixedContrastScheduler":
        """Apply one non-terminal post-action transition.

        A 1000-action timeout retains the target and its age but starts a new
        window.  If the action changes the selected target, that identity
        change wins over timeout and resets only the new target's age/window.
        """

        if self.target is None:
            raise FixedContrastError("inactive scheduler cannot consume an action")
        next_target = _arm_frontier(self.arm, achieved_after)
        next_age = self.target_age_actions + 1
        next_window = self.window_actions + 1
        if next_target != self.target:
            return FixedContrastScheduler(self.arm, next_target, 0, 0)
        if next_window == TARGET_WINDOW:
            return FixedContrastScheduler(self.arm, self.target, 0, next_age)
        return FixedContrastScheduler(self.arm, self.target, next_window, next_age)


def _online_inputs(
    observable: Any,
    *,
    target: str,
    age: float,
    episode_boundary: bool,
) -> dict[str, Any]:
    """Build a one-step [T=1,B=1] policy input from public observations only."""

    import torch

    if target not in TARGET_ORDER or not math.isfinite(float(age)):
        raise FixedContrastError("invalid public target/age input")
    target_onehot = np.zeros((1, 1, len(TARGET_ORDER)), dtype=np.float32)
    target_onehot[0, 0, TARGET_ORDER.index(target)] = 1.0
    return {
        "pixels": torch.from_numpy(
            np.asarray(observable.pixels, dtype=np.uint8)[None, None].copy()
        ),
        "hud16": torch.from_numpy(
            np.asarray(observable.inventory16, dtype=np.float32)[None, None].copy()
        ),
        "achieved22": torch.from_numpy(
            np.asarray(observable.achieved22, dtype=np.float32)[None, None].copy()
        ),
        "prev_action": torch.as_tensor([[int(observable.prev_action)]], dtype=torch.long),
        "prev_observable_achievement_delta": torch.from_numpy(
            np.asarray(observable.public_event22, dtype=np.float32)[None, None].copy()
        ),
        "ego_delta": torch.from_numpy(
            np.asarray(observable.ego_delta, dtype=np.float32)[None, None].copy()
        ),
        "facing": torch.from_numpy(
            np.asarray(observable.facing, dtype=np.float32)[None, None].copy()
        ),
        "episode_boundary": torch.as_tensor([[bool(episode_boundary)]], dtype=torch.bool),
        "target_onehot5": torch.from_numpy(target_onehot),
        "normalized_target_age": torch.from_numpy(
            np.asarray([[np.float32(age)]], dtype=np.float32)
        ),
    }


def _uniform_seed(*, controller_seed: int, world_seed: int, stream: int) -> np.random.SeedSequence:
    """One arm-independent common-random-number stream for a paired row.

    This is intentionally an ordinary reproducible RNG stream, not a hash
    ledger.  The same precomputed U[t] is supplied to every arm at global step
    t and each arm then applies its own categorical inverse CDF.
    """

    if type(controller_seed) is not int or type(world_seed) is not int or type(stream) is not int:
        raise FixedContrastError("paired uniform identity must use exact integers")
    return np.random.SeedSequence([
        int(controller_seed), int(world_seed), int(stream), 20_260_820,
    ])


def _paired_uniforms(
    *, controller_seed: int, world_seed: int, stream: int, horizon: int
) -> np.ndarray:
    if type(horizon) is not int or not 1 <= horizon <= FORMAL_HORIZON:
        raise FixedContrastError("contrast horizon must be within the fixed 10k treatment")
    generator = np.random.default_rng(
        _uniform_seed(
            controller_seed=controller_seed, world_seed=world_seed, stream=stream
        )
    )
    return np.asarray(generator.random(horizon), dtype=np.float64)


def _inverse_cdf(probabilities: Any, uniform: float) -> int:
    """Use one supplied U rather than independent categorical draws per arm."""

    values = np.asarray(probabilities.detach().float().cpu().numpy(), dtype=np.float64)
    if values.shape != (17,) or not np.isfinite(values).all() or np.any(values < 0.0):
        raise FixedContrastError("real policy did not emit finite 17-action probabilities")
    if not 0.0 <= float(uniform) < 1.0 or not math.isclose(
        float(values.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise FixedContrastError("common action U or policy probability mass drifted")
    cumulative = 0.0
    for action, probability in enumerate(values):
        cumulative += float(probability)
        if float(uniform) < cumulative:
            return int(action)
    # Floating-point roundoff can leave a tiny final gap; categorical sampling
    # assigns it to the final action, matching the formal sampler convention.
    return 16


def _evaluate_arm(
    *,
    model: Any,
    handoff: Any,
    arm: str,
    device: Any,
    uniforms: np.ndarray,
) -> dict[str, Any]:
    """Run one real arm online without reconstructing recurrent history."""

    import torch
    from experiments.alpagd_rl.formal_20260810.pc3_v9_observation_consistent.live_adapter import (
        LiveObservableAdapter,
    )

    if arm not in ARM_ORDERS:
        raise FixedContrastError("fixed contrast arm order is unavailable")
    adapter = LiveObservableAdapter()
    achieved = _achievement_state(handoff.achievements)
    scheduler = FixedContrastScheduler(arm).select(achieved)
    if scheduler.target != ARM_ORDERS[arm][0]:
        raise FixedContrastError("fresh real handoff did not begin at an arm's first target")
    hidden = model.initial_hidden(1, device=device)
    previous_action = -1
    first_hits: dict[str, int | None] = {code: None for code in TARGET_ORDER}
    forward_finite = True
    stop_reason = "HORIZON"
    model.eval()
    with torch.no_grad():
        for global_step, uniform in enumerate(uniforms):
            observable = adapter.capture(handoff, prev_action=previous_action)
            current_target = scheduler.target
            if current_target is None:
                raise FixedContrastError("arm exhausted its targets before endpoint stop")
            inputs = _online_inputs(
                observable,
                target=current_target,
                age=scheduler.pre_action_target_age,
                # Target switches and timeouts retain the live GRU state.
                episode_boundary=(global_step == 0),
            )
            result = model(**_move_to_device(inputs, device), hidden=hidden)
            logits = result["policy_logits"][0, 0]
            probabilities = logits.softmax(-1)
            forward_finite = forward_finite and _finite(
                logits, probabilities, result["value"], result["final_hidden"]
            )
            if not forward_finite:
                raise FixedContrastError("non-finite real contrast policy output")
            action = _inverse_cdf(probabilities, float(uniform))
            # This is direct online carry.  There is no episode-prefix hidden
            # reconstruction and no reset except a true endpoint stop.
            hidden = result["final_hidden"]
            _observation, _reward, done, _info = handoff.step(action)
            previous_action = action
            achieved_after = _achievement_state(handoff.achievements)
            for code in TARGET_ORDER:
                if achieved_after[code] and first_hits[code] is None:
                    first_hits[code] = global_step + 1
            e_hit = bool(achieved_after["E"])
            terminal = bool(done) or bool(handoff.terminal)
            # Required precedence: an E hit on a terminal transition remains
            # an E endpoint, not a terminal failure.
            if e_hit:
                stop_reason = "E_HIT"
                achieved = achieved_after
                break
            if terminal:
                stop_reason = "TERMINAL"
                achieved = achieved_after
                break
            scheduler = scheduler.after_action(achieved_after)
            achieved = achieved_after
        else:
            global_step = len(uniforms) - 1
    steps = int(global_step) + 1
    depth = _canonical_depth(achieved)
    return {
        "arm": arm,
        "step_count": steps,
        "stop_reason": stop_reason,
        "final_achievements": achieved,
        "canonical_prefix_depth": depth,
        "R": float(depth / len(TARGET_ORDER)),
        # `raw_E_hit` and full canonical `Y` are intentionally separate:
        # a permuted arm can hit E without completing A->E canonically.
        "raw_E_hit": bool(achieved["E"]),
        "Y": int(depth == len(TARGET_ORDER)),
        "first_hit_steps": first_hits,
        "forward_finite": bool(forward_finite),
    }


def evaluate_paired_row(
    *,
    model: Any,
    controller_seed: int,
    world_seed: int,
    stream: int,
    device: Any,
    horizon: int = FORMAL_HORIZON,
) -> dict[str, Any]:
    """Evaluate all five fresh arms from an identical real base replay."""

    from experiments.alpagd_rl.formal_20260810.pc3_long_horizon.handoff import (
        open_arm_handoffs,
    )

    if type(stream) is not int or stream not in FORMAL_STREAMS:
        raise FixedContrastError("fixed contrast stream must be 0 or 1")
    uniforms = _paired_uniforms(
        controller_seed=int(controller_seed),
        world_seed=int(world_seed),
        stream=int(stream),
        horizon=int(horizon),
    )
    prefix, handoffs = open_arm_handoffs(int(world_seed), ARMS)
    try:
        results = {
            arm: _evaluate_arm(
                model=model,
                handoff=handoffs[arm],
                arm=arm,
                device=device,
                uniforms=uniforms,
            )
            for arm in ARMS
        }
    finally:
        for handoff in handoffs.values():
            handoff.close()
    correct_r = float(results["Correct"]["R"])
    mean_permuted_r = fmean(float(results[arm]["R"]) for arm in ARMS[1:])
    return {
        "controller_seed": int(controller_seed),
        "world_seed": int(world_seed),
        "stream": int(stream),
        "horizon": int(horizon),
        "base_prefix_length": len(prefix),
        "common_action_uniform": {
            "method": "numpy_seedsequence_arm_independent_inverse_cdf",
            "identity": [int(controller_seed), int(world_seed), int(stream)],
            "values_supplied_per_arm": len(uniforms),
        },
        "arms": results,
        "paired_R_difference": float(correct_r - mean_permuted_r),
    }


def _final_checkpoint_model(*, checkpoint: Path, controller_seed: int, device: Any) -> Any:
    """Load only an exact completed PPO behaviour policy; never a BC prefix."""

    import torch

    if not checkpoint.is_file():
        raise FixedContrastError(f"final PPO checkpoint is unavailable: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    expected_steps = PPO_FINAL_ROLLOUTS * 16 * 256
    expected_optimizer_steps = PPO_FINAL_ROLLOUTS * PPO_MICROSTEPS_PER_ROLLOUT
    if (
        payload.get("schema") != PPO_CHECKPOINT_SCHEMA
        or int(payload.get("controller_seed", -1)) != int(controller_seed)
        or int(payload.get("completed_rollouts", -1)) != PPO_FINAL_ROLLOUTS
        or int(payload.get("global_environment_steps", -1)) != expected_steps
        or int(payload.get("optimizer_steps", -1)) != expected_optimizer_steps
        or payload.get("environment_serialized") is not False
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("source_dagger_checkpoint"), str)
        or not payload["source_dagger_checkpoint"]
    ):
        raise FixedContrastError(
            "fixed contrast requires an exact completed final PPO behaviour checkpoint"
        )
    model, _optimizer = _new_model_optimizer(controller_seed=controller_seed, device=device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    with torch.no_grad():
        if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
            raise FixedContrastError("final PPO behaviour model contains non-finite weights")
    return model


def _parse_checkpoint_specs(specifications: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise FixedContrastError("checkpoint must be expressed as CONTROLLER_SEED=PATH")
        raw_seed, raw_path = specification.split("=", 1)
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise FixedContrastError("checkpoint controller seed is not an integer") from exc
        if seed not in CONTROLLER_SEEDS or seed in result or not raw_path:
            raise FixedContrastError("checkpoint specifications must cover each frozen controller once")
        result[seed] = Path(raw_path)
    if set(result) != set(CONTROLLER_SEEDS):
        raise FixedContrastError("formal contrast needs exactly the four frozen controller checkpoints")
    return result


def _parse_single_checkpoint(specifications: Sequence[str], *, controller_seed: int) -> Path:
    """Parse one strict final-PPO input for a controller collection shard."""

    if controller_seed not in CONTROLLER_SEEDS or len(specifications) != 1:
        raise FixedContrastError("a controller shard needs exactly one frozen controller/checkpoint")
    specification = specifications[0]
    if "=" not in specification:
        # The strict PPO payload check below still binds this direct path to
        # `controller_seed`; the explicit form remains useful in shell grids.
        return Path(specification)
    raw_seed, raw_path = specification.split("=", 1)
    try:
        parsed_seed = int(raw_seed)
    except ValueError as exc:
        raise FixedContrastError("checkpoint controller seed is not an integer") from exc
    if parsed_seed != int(controller_seed) or not raw_path:
        raise FixedContrastError("controller shard checkpoint identity does not match its controller")
    return Path(raw_path)


def _load_formal_worlds(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FixedContrastError(f"formal contrast world list is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    values = value.get("worlds") if isinstance(value, Mapping) else value
    if not isinstance(values, list):
        raise FixedContrastError("formal contrast world list must be a JSON list or {worlds: [...]}")
    worlds = tuple(int(world) for world in values)
    if (
        len(worlds) != FORMAL_WORLD_COUNT
        or len(set(worlds)) != FORMAL_WORLD_COUNT
        or any(not FORMAL_WORLD_LOWER <= world <= FORMAL_WORLD_UPPER for world in worlds)
    ):
        raise FixedContrastError("formal contrast requires exactly 100 in-namespace worlds")
    return worlds


def _collect_controller_rows(
    *,
    controller_seed: int,
    checkpoint: Path,
    worlds: Sequence[int],
    device: Any,
) -> list[dict[str, Any]]:
    """Collect one controller's exact 100-world x two-stream paired rows."""

    model = _final_checkpoint_model(
        checkpoint=checkpoint, controller_seed=controller_seed, device=device
    )
    rows: list[dict[str, Any]] = []
    for world_seed in worlds:
        for stream in FORMAL_STREAMS:
            rows.append(evaluate_paired_row(
                model=model,
                controller_seed=controller_seed,
                world_seed=int(world_seed),
                stream=stream,
                device=device,
                horizon=FORMAL_HORIZON,
            ))
    expected = {(int(world), stream) for world in worlds for stream in FORMAL_STREAMS}
    observed = {(int(row["world_seed"]), int(row["stream"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise FixedContrastError("controller shard did not retain its exact paired world/stream rows")
    return rows


def reduce_primary(paired_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce the exact 4-controller x 2-stream x 100-world paired grid.

    All three labels are descriptive output only.  This function has no gate,
    no controller/world selection, and no ability to alter training.
    """

    expected = {
        (controller_seed, world_seed, stream)
        for controller_seed in CONTROLLER_SEEDS
        for world_seed in {int(row["world_seed"]) for row in paired_rows}
        for stream in FORMAL_STREAMS
    }
    rows = {
        (int(row["controller_seed"]), int(row["world_seed"]), int(row["stream"])): row
        for row in paired_rows
    }
    if len(rows) != len(paired_rows):
        raise FixedContrastError("primary reduction received duplicate paired identities")
    worlds = tuple(sorted({identity[1] for identity in rows}))
    if (
        len(worlds) != FORMAL_WORLD_COUNT
        or len(rows) != FORMAL_WORLD_COUNT * len(CONTROLLER_SEEDS) * len(FORMAL_STREAMS)
        or set(rows) != expected
    ):
        raise FixedContrastError("primary reduction requires the complete exact 4x2x100 paired grid")
    effects: dict[int, float] = {}
    individual: dict[str, dict[int, float]] = {arm: {} for arm in ARMS[1:]}
    for world_seed in worlds:
        row_set = [rows[(controller, world_seed, stream)] for controller in CONTROLLER_SEEDS for stream in FORMAL_STREAMS]
        if any(set(row.get("arms", {})) != set(ARMS) for row in row_set):
            raise FixedContrastError("one paired contrast row does not retain all five arms")
        effects[world_seed] = fmean(float(row["paired_R_difference"]) for row in row_set)
        for arm in ARMS[1:]:
            individual[arm][world_seed] = fmean(
                float(row["arms"]["Correct"]["R"]) - float(row["arms"][arm]["R"])
                for row in row_set
            )
    ordered_effects = [effects[world_seed] for world_seed in worlds]
    estimate = fmean(ordered_effects)
    sample_sd = stdev(ordered_effects)
    if sample_sd == 0.0:
        lower = upper = estimate
    else:
        margin = T_975_DF99 * sample_sd / math.sqrt(FORMAL_WORLD_COUNT)
        lower, upper = estimate - margin, estimate + margin
    point_positive = estimate > 0.0
    ci_positive = lower > 0.0
    sesoi_established = lower > 0.05
    highest_tier = (
        "MEANINGFUL_ORDER_PROGRESS_ESTABLISHED" if sesoi_established
        else "ORDER_PROGRESS_ESTABLISHED" if ci_positive
        else "POINT_POSITIVE_DESCRIPTIVE" if point_positive
        else "NOT_POINT_POSITIVE"
    )
    return {
        "endpoint": "canonical_A_to_E_prefix_depth_divided_by_5",
        "world_effects": {str(world): effects[world] for world in worlds},
        "estimate_R": float(estimate),
        "sample_sd": float(sample_sd),
        "ci95": {"lower": float(lower), "upper": float(upper), "t_critical": T_975_DF99},
        "labels": {
            "point_positive": bool(point_positive),
            "ci_positive": bool(ci_positive),
            "sesoi_established": bool(sesoi_established),
            "highest_tier": highest_tier,
            "nonblocking": True,
        },
        "individual_permutation_world_effects_descriptive": {
            arm: {str(world): float(values[world]) for world in worlds}
            for arm, values in individual.items()
        },
    }


def _atomic_json(value: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)


def run_controller_shard(
    *,
    controller_seed: int,
    checkpoint_spec: Sequence[str],
    worlds_json: Path,
    output: Path,
    device_name: str,
) -> dict[str, Any]:
    """Collect one final controller's independent 100x2 paired shard."""

    import torch

    if controller_seed not in CONTROLLER_SEEDS:
        raise FixedContrastError("controller shard must use one frozen controller seed")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FixedContrastError("requested CUDA is unavailable")
    checkpoint = _parse_single_checkpoint(checkpoint_spec, controller_seed=controller_seed)
    worlds = _load_formal_worlds(worlds_json)
    paired_rows = _collect_controller_rows(
        controller_seed=controller_seed,
        checkpoint=checkpoint,
        worlds=worlds,
        device=device,
    )
    shard = {
        "schema": SHARD_SCHEMA,
        "mode": "formal_fixed_contrast_controller_shard",
        "controller_seed": int(controller_seed),
        "checkpoint": str(checkpoint.resolve()),
        "worlds": list(worlds),
        "streams": list(FORMAL_STREAMS),
        "arms": list(ARMS),
        "horizon": FORMAL_HORIZON,
        "paired_rows": paired_rows,
    }
    _atomic_json(shard, output)
    return {
        "status": "FIXED_CONTRAST_CONTROLLER_SHARD_COMPLETED",
        "controller_seed": int(controller_seed),
        "paired_rows": len(paired_rows),
        "output": str(output),
    }


def _load_controller_shard(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FixedContrastError(f"controller shard is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    controller_seed = int(value.get("controller_seed", -1))
    worlds_value = value.get("worlds")
    rows = value.get("paired_rows")
    if (
        value.get("schema") != SHARD_SCHEMA
        or value.get("mode") != "formal_fixed_contrast_controller_shard"
        or controller_seed not in CONTROLLER_SEEDS
        or tuple(value.get("streams", ())) != FORMAL_STREAMS
        or tuple(value.get("arms", ())) != ARMS
        or int(value.get("horizon", -1)) != FORMAL_HORIZON
        or not isinstance(worlds_value, list)
        or not isinstance(rows, list)
    ):
        raise FixedContrastError("controller shard envelope drift")
    worlds = tuple(int(world) for world in worlds_value)
    if (
        len(worlds) != FORMAL_WORLD_COUNT
        or len(set(worlds)) != FORMAL_WORLD_COUNT
        or any(not FORMAL_WORLD_LOWER <= world <= FORMAL_WORLD_UPPER for world in worlds)
    ):
        raise FixedContrastError("controller shard world grid drift")
    expected = {(controller_seed, world, stream) for world in worlds for stream in FORMAL_STREAMS}
    identities = [
        (int(row.get("controller_seed", -1)), int(row.get("world_seed", -1)), int(row.get("stream", -1)))
        for row in rows if isinstance(row, Mapping)
    ]
    if (
        len(rows) != len(expected)
        or len(identities) != len(rows)
        or len(set(identities)) != len(identities)
        or set(identities) != expected
        or any(set(row.get("arms", {})) != set(ARMS) for row in rows)
    ):
        raise FixedContrastError("controller shard does not retain the exact 100x2 five-arm grid")
    return value


def run_merge(*, shard_paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Merge exactly four controller shards, then do the sole primary reduce."""

    if len(shard_paths) != len(CONTROLLER_SEEDS) or len({path.resolve() for path in shard_paths}) != len(shard_paths):
        raise FixedContrastError("merge requires exactly four distinct controller shards")
    shards = [_load_controller_shard(path) for path in shard_paths]
    by_controller = {int(shard["controller_seed"]): shard for shard in shards}
    if len(by_controller) != len(shards) or set(by_controller) != set(CONTROLLER_SEEDS):
        raise FixedContrastError("merge shards must cover the four frozen controllers exactly once")
    worlds = tuple(int(world) for world in shards[0]["worlds"])
    if any(tuple(int(world) for world in shard["worlds"]) != worlds for shard in shards[1:]):
        raise FixedContrastError("controller shards do not declare the same ordered 100-world grid")
    paired_rows = [
        row for controller_seed in CONTROLLER_SEEDS
        for row in by_controller[controller_seed]["paired_rows"]
    ]
    identities = [
        (int(row["controller_seed"]), int(row["world_seed"]), int(row["stream"]))
        for row in paired_rows
    ]
    expected = {
        (controller_seed, world, stream)
        for controller_seed in CONTROLLER_SEEDS
        for world in worlds
        for stream in FORMAL_STREAMS
    }
    if len(paired_rows) != len(expected) or len(set(identities)) != len(identities) or set(identities) != expected:
        raise FixedContrastError("merge did not form the exact non-overlapping 4x2x100 paired grid")
    primary = reduce_primary(paired_rows)
    result = {
        "schema": RESULT_SCHEMA,
        "mode": "formal_fixed_contrast_merged",
        "controllers": list(CONTROLLER_SEEDS),
        "worlds": list(worlds),
        "streams": list(FORMAL_STREAMS),
        "arms": list(ARMS),
        "horizon": FORMAL_HORIZON,
        "source_shards": [str(path.resolve()) for path in shard_paths],
        "paired_rows": paired_rows,
        "primary": primary,
    }
    _atomic_json(result, output)
    return {
        "status": "FIXED_CONTRAST_MERGE_COMPLETED",
        "paired_rows": len(paired_rows),
        "output": str(output),
    }


def run_formal(
    *, checkpoint_specs: Sequence[str], worlds_json: Path, output: Path, device_name: str
) -> dict[str, Any]:
    """Execute the full fixed treatment only when final PPO inputs exist."""

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FixedContrastError("requested CUDA is unavailable")
    checkpoints = _parse_checkpoint_specs(checkpoint_specs)
    worlds = _load_formal_worlds(worlds_json)
    paired_rows: list[dict[str, Any]] = []
    for controller_seed in CONTROLLER_SEEDS:
        paired_rows.extend(_collect_controller_rows(
            controller_seed=controller_seed,
            checkpoint=checkpoints[controller_seed],
            worlds=worlds,
            device=device,
        ))
    primary = reduce_primary(paired_rows)
    result = {
        "schema": RESULT_SCHEMA,
        "mode": "formal_fixed_contrast",
        "controllers": list(CONTROLLER_SEEDS),
        "worlds": list(worlds),
        "streams": list(FORMAL_STREAMS),
        "arms": list(ARMS),
        "horizon": FORMAL_HORIZON,
        "paired_rows": paired_rows,
        "primary": primary,
    }
    _atomic_json(result, output)
    return {
        "status": "FIXED_CONTRAST_FORMAL_COMPLETED",
        "paired_rows": len(paired_rows),
        "output": str(output),
    }


def run_development_wiring(*, output: Path, device_name: str, short_actions: int) -> dict[str, Any]:
    """Run a small non-formal actual paired wiring check without any checkpoint."""

    import torch

    if not 1 <= int(short_actions) <= 128:
        raise FixedContrastError("development paired wiring is intentionally limited to 1..128 actions")
    if FORMAL_WORLD_LOWER <= DEVELOPMENT_WORLD_SEED <= FORMAL_WORLD_UPPER:
        raise FixedContrastError("development contrast world overlaps the formal namespace")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FixedContrastError("requested CUDA is unavailable")
    # This is a fresh isolated policy solely for interface wiring; no BC or
    # other intermediate training checkpoint is opened in this code path.
    model, _optimizer = _new_model_optimizer(
        controller_seed=DEVELOPMENT_CONTROLLER_SEED, device=device
    )
    paired = evaluate_paired_row(
        model=model,
        controller_seed=DEVELOPMENT_CONTROLLER_SEED,
        world_seed=DEVELOPMENT_WORLD_SEED,
        stream=DEVELOPMENT_STREAM,
        device=device,
        horizon=int(short_actions),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # Preserve operational evidence only.  The development wiring does not
    # print or classify endpoint outcomes and cannot stand in for formal data.
    result = {
        "schema": WIRING_SCHEMA,
        "namespace": DEVELOPMENT_NAMESPACE,
        "mode": "nonformal_interface_wiring_only",
        "controller_seed": DEVELOPMENT_CONTROLLER_SEED,
        "world_seed": DEVELOPMENT_WORLD_SEED,
        "stream": DEVELOPMENT_STREAM,
        "requested_actions_per_arm": int(short_actions),
        "base_prefix_length": int(paired["base_prefix_length"]),
        "arms": list(ARMS),
        "action_counts": {arm: int(paired["arms"][arm]["step_count"]) for arm in ARMS},
        "forward_finite": bool(all(paired["arms"][arm]["forward_finite"] for arm in ARMS)),
        "common_action_uniform": paired["common_action_uniform"],
    }
    _atomic_json(result, output)
    return {
        "status": "FIXED_CONTRAST_WIRING_OK",
        "arms": len(ARMS),
        "requested_actions_per_arm": int(short_actions),
        "forward_finite": bool(result["forward_finite"]),
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--development-wiring", action="store_true")
    parser.add_argument("--short-actions", type=int, default=DEVELOPMENT_SHORT_ACTIONS)
    parser.add_argument("--controller-shard", "--controller-seed", dest="controller_shard", type=int)
    parser.add_argument("--merge-shard", type=Path, action="append", default=[])
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append", default=[],
        metavar="CONTROLLER_SEED=PATH",
        help="one final PPO behaviour checkpoint per frozen controller",
    )
    parser.add_argument("--worlds-json", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.development_wiring:
            if args.checkpoints or args.worlds_json is not None or args.controller_shard is not None or args.merge_shard:
                raise FixedContrastError("development wiring never opens formal checkpoints or worlds")
            result = run_development_wiring(
                output=args.output, device_name=args.device, short_actions=args.short_actions
            )
        elif args.merge_shard:
            if args.checkpoints or args.worlds_json is not None or args.controller_shard is not None:
                raise FixedContrastError("contrast merge accepts only four completed controller shards")
            result = run_merge(shard_paths=args.merge_shard, output=args.output)
        elif args.controller_shard is not None:
            if args.worlds_json is None:
                raise FixedContrastError("controller shard requires --worlds-json")
            result = run_controller_shard(
                controller_seed=args.controller_shard,
                checkpoint_spec=args.checkpoints,
                worlds_json=args.worlds_json,
                output=args.output,
                device_name=args.device,
            )
        else:
            if args.worlds_json is None:
                raise FixedContrastError("formal fixed contrast requires --worlds-json")
            result = run_formal(
                checkpoint_specs=args.checkpoints,
                worlds_json=args.worlds_json,
                output=args.output,
                device_name=args.device,
            )
    except Exception as exc:
        print(json.dumps({
            "status": "ERROR", "error_type": type(exc).__name__, "message": str(exc),
        }, sort_keys=True), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
