"""Policy-blind Q0 state-bank builder over fresh deterministic replays."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .planner import (
    DeterministicMechanicsPlanner, MechanicsView, PlanResult,
    planner_provenance, planner_sha256,
)
from .schema import (
    ACTION_DIRECTED_ACHIEVEMENTS, ACTION_NAMES, CANONICAL_ACHIEVEMENTS,
    CANONICAL_INDEX, DEV_WORLD_TARGET, FORMAL_WORLD_TARGET, STATE_ROW_SCHEMA_ID, TARGET_SPEC,
    PHASE_Q0A_TRANSITION_UPPER_BOUND, canonical_bytes, canonical_sha256, validate_state_row,
    validate_state_bank_manifest, validate_runtime_qualification_binding,
    expected_verification_suffix,
)

BUILDER_VERSION = "q0-policy-blind-replay-builder/v1"
VERIFICATION_SUFFIX_LENGTH = 16
RECONSTRUCTIONS_REQUIRED = 3
PRIOR_RESET_ATTEMPT = {
    "world_seed": 759001,
    "operation": "official_adapter_engineering_smoke_construct_plus_single_reset",
    "environment_steps": 0,
    "planner_called": False,
    "checkpoint_loaded_or_called": False,
    "used_for_design_or_state_selection": False,
    "receipt_created": False,
    "stdout_summary": "(32, 32) (64, 64) 22 True",
    "disclosure": "dev namespace is not pristine; retain in every later dev receipt",
}


class BuildError(RuntimeError):
    pass


class RuntimeQualificationGateError(BuildError):
    """The live qualification changed before the first world factory."""


class ReplayEnvironment(Protocol):
    action_names: Sequence[str]
    def reset(self) -> Any: ...
    def step(self, action: int) -> tuple[Any, float, bool, Mapping[str, Any]]: ...
    def close(self) -> None: ...
    def reset_audit(self) -> Mapping[str, Any]: ...
    def bind_factory_ordinal(self, ordinal: int) -> None: ...


EnvironmentFactory = Callable[[int], ReplayEnvironment]
PrefixPlanner = Callable[[ReplayEnvironment, MechanicsView, str, int], Sequence[int]]
ViewFactory = Callable[[ReplayEnvironment, Mapping[str, Any] | None], MechanicsView]


@dataclass(frozen=True)
class ReplaySnapshot:
    observation: Any
    info: Mapping[str, Any]
    trace: tuple[Mapping[str, Any], ...]
    fingerprint: Mapping[str, Any]
    done: bool
    reset_audit: Mapping[str, Any]


class PublicCrafterAdapter:
    """Normalize pinned Crafter 1.8.3 into the builder's strict evidence API."""

    def __init__(self, env: Any):
        self._env = env
        self._reset_calls = 0
        self._episode_before_first_reset = getattr(env, "_episode", None)
        self._episode_after_first_reset = None
        self._factory_ordinal: int | None = None

    def bind_factory_ordinal(self, ordinal: int) -> None:
        if self._factory_ordinal is not None or not isinstance(ordinal, int) or ordinal < 0:
            raise BuildError("factory ordinal must be bound exactly once")
        self._factory_ordinal = ordinal

    @property
    def action_names(self) -> Sequence[str]:
        return tuple(self._env.action_names)

    def reset(self) -> Any:
        self._reset_calls += 1
        if self._reset_calls != 1:
            raise BuildError("adapter permits exactly one reset per fresh environment")
        observation = self._env.reset()
        self._episode_after_first_reset = getattr(self._env, "_episode", None)
        return observation

    def reset_audit(self) -> Mapping[str, Any]:
        return {
            "factory_ordinal": self._factory_ordinal,
            "reset_calls": self._reset_calls,
            "underlying_episode_before_first_reset": self._episode_before_first_reset,
            "underlying_episode_after_first_reset": self._episode_after_first_reset,
            "exact_episode_transition": [0, 1],
            "qualified": self._reset_calls == 1
            and self._factory_ordinal is not None
            and self._episode_before_first_reset == 0
            and self._episode_after_first_reset == 1,
        }

    def step(self, action: int) -> tuple[Any, float, bool, Mapping[str, Any]]:
        result = self._env.step(int(action))
        if not isinstance(result, tuple) or len(result) != 4:
            raise BuildError("formal Crafter requires old Gym four-tuple API")
        observation, reward, done, info = result
        if not isinstance(info, Mapping):
            raise BuildError("Crafter info must be a mapping")
        return observation, float(reward), bool(done), dict(info)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


def builder_sha256() -> str:
    return hashlib.sha256(Path(inspect.getfile(build_candidate_state)).read_bytes()).hexdigest()


def builder_provenance() -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "builder_sha256": builder_sha256(),
        "policy_blind": True,
        "fresh_environment_per_replay": True,
        "resets_per_environment": 1,
        "crafter_episode_index": 1,
        "checkpoint_import_or_call": False,
        "prior_reset_attempt": PRIOR_RESET_ATTEMPT,
    }


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _normal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, set):
        return sorted(_normal(item) for item in value)
    return value


def raw_counts(info: Mapping[str, Any]) -> dict[str, int]:
    achievements = info.get("achievements")
    if not isinstance(achievements, Mapping) or set(achievements) != set(CANONICAL_ACHIEVEMENTS):
        raise BuildError("raw info['achievements'] must contain exactly 22 canonical counts")
    counts: dict[str, int] = {}
    for name in CANONICAL_ACHIEVEMENTS:
        value = achievements[name]
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or int(value) < 0:
            raise BuildError("raw achievement counts must be nonnegative integers")
        counts[name] = int(value)
    return counts


def _initial_info(env: ReplayEnvironment, view_factory: ViewFactory) -> Mapping[str, Any]:
    view = view_factory(env, None)
    return {
        "inventory": dict(view.inventory()),
        "achievements": dict(view.achievement_counts()),
        "semantic": [[view.cell((x, y))[0], view.cell((x, y))[1]]
                     for x in range(view.world_area()[0]) for y in range(view.world_area()[1])],
        "player_pos": list(view.player_position()),
    }


def _trace_row(step: int, action: int | None, observation: Any, reward: float,
               done: bool, info: Mapping[str, Any]) -> dict[str, Any]:
    if "semantic" not in info or info.get("player_pos") is None or "inventory" not in info:
        raise BuildError("Crafter info must expose semantic, player_pos, and inventory")
    counts = raw_counts(info)
    return {
        "step": int(step),
        "action": action,
        "observation_sha256": _array_sha256(observation),
        "reward": float(reward),
        "done": bool(done),
        "achievement_counts": counts,
        "inventory": _normal(info["inventory"]),
        "player_position": [int(value) for value in np.asarray(info["player_pos"]).tolist()],
        "semantic_array_sha256": _array_sha256(info["semantic"]),
    }


def _fingerprint(observation: Any, info: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = raw_counts(info)
    return {
        "observation_sha256": _array_sha256(observation),
        "achieved_mask": [int(counts[name] > 0) for name in CANONICAL_ACHIEVEMENTS],
        "achievement_counts": counts,
        "inventory_state_canonical_sha256": hashlib.sha256(canonical_bytes(_normal(info["inventory"]))).hexdigest(),
        "player_position": [int(value) for value in np.asarray(info["player_pos"]).tolist()],
        "semantic_array_sha256": _array_sha256(info["semantic"]),
        "prefix_trace_sha256": canonical_sha256(list(trace)),
        "target_age_start_normalized": 0.0,
    }


def reconstruct(factory: EnvironmentFactory, view_factory: ViewFactory, world_seed: int,
                actions: Sequence[int]) -> tuple[ReplayEnvironment, MechanicsView, ReplaySnapshot]:
    env = factory(int(world_seed))
    reset_count = 0
    try:
        observation = env.reset()
        reset_count += 1
        if reset_count != 1:
            raise BuildError("each environment must be reset exactly once")
        info = _initial_info(env, view_factory)
        trace: list[Mapping[str, Any]] = [_trace_row(0, None, observation, 0.0, False, info)]
        done = False
        for step, action in enumerate(actions, start=1):
            if done:
                raise BuildError("prefix continued after terminal")
            if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(ACTION_NAMES):
                raise BuildError("invalid prefix action")
            observation, reward, done, info = env.step(action)
            trace.append(_trace_row(step, action, observation, reward, done, info))
        audit = dict(env.reset_audit())
        if audit.get("reset_calls") != 1 or audit.get("underlying_episode_before_first_reset") != 0 \
                or audit.get("underlying_episode_after_first_reset") != 1 or audit.get("qualified") is not True:
            raise BuildError("fresh environment reset audit must prove episode 0 to 1 exactly once")
        snapshot = ReplaySnapshot(observation, info, tuple(trace), _fingerprint(observation, info, trace), bool(done), audit)
        if done:
            raise BuildError("handoff prefix terminated environment")
        return env, view_factory(env, info), snapshot
    except BaseException:
        env.close()
        raise


def verification_suffix(world_seed: int) -> list[int]:
    """Frozen policy-independent suffix derived only from the world seed."""
    return expected_verification_suffix(int(world_seed))


def _run_suffix(env: ReplayEnvironment, snapshot: ReplaySnapshot, actions: Sequence[int]) -> tuple[list[Mapping[str, Any]], str]:
    rows: list[Mapping[str, Any]] = []
    for step, action in enumerate(actions, start=1):
        observation, reward, done, info = env.step(action)
        rows.append(_trace_row(step, action, observation, reward, done, info))
        if done and step != len(actions):
            raise BuildError("verification suffix terminated early")
    return rows, canonical_sha256(rows)


def _run_witness(factory: EnvironmentFactory, view_factory: ViewFactory, world_seed: int,
                 prefix: Sequence[int], target: str, horizon: int,
                 expected_fingerprint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int], Mapping[str, Any]]:
    role_binder = getattr(factory, "bind_operation_role", None)
    if callable(role_binder):
        role_binder(f"witness_fresh_prefix:{target}")
    env, view, snapshot = reconstruct(factory, view_factory, world_seed, prefix)
    try:
        if snapshot.fingerprint != expected_fingerprint:
            raise BuildError("fresh witness replay fingerprint mismatch")
        trace_rows: list[Mapping[str, Any]] = []
        if callable(role_binder):
            role_binder(f"witness_planner:{target}")
        def step(action: int):
            observation, reward, done, info = env.step(action)
            trace_rows.append(_trace_row(len(trace_rows) + 1, action, observation, reward, done, info))
            return observation, reward, done, info
        result: PlanResult = DeterministicMechanicsPlanner(view, step, horizon=horizon).run(target)
        attempt = {
            "target": target,
            "attempted": True,
            "success": result.success,
            "actions": list(result.actions),
            "action_names": list(result.action_names),
            "first_success_step": len(result.actions) if result.success else None,
            "initial_count": result.initial_count,
            "final_count": result.final_count,
            "terminal": result.terminal,
            "failure_reason": result.failure_reason,
            "trace": trace_rows,
            "trace_sha256": canonical_sha256(trace_rows),
            "fresh_replay_fingerprint_sha256": canonical_sha256(snapshot.fingerprint),
        }
        return attempt, {"fresh_prefix": len(prefix), "planner_actions": len(trace_rows)}, snapshot.reset_audit
    finally:
        env.close()


def build_candidate_state(*, factory: EnvironmentFactory, view_factory: ViewFactory,
                          prefix_planner: PrefixPlanner, world_seed: int,
                          designated_target: str, phase: str,
                          primary: bool = True, reserve_rank: int | None = None) -> dict[str, Any]:
    """Build one state; never selects or loads a learned policy."""
    namespaces = {"dev": DEV_WORLD_TARGET, "formal": FORMAL_WORLD_TARGET}
    if phase not in namespaces or namespaces[phase].get(world_seed) != designated_target:
        raise BuildError("phase world namespace or designated target mismatch")
    if designated_target not in TARGET_SPEC:
        raise BuildError("unknown designated target")
    horizon = TARGET_SPEC[designated_target]["horizon"]

    # Retaining strong references for the complete build makes identity reuse
    # impossible to hide.  The emitted ordinal is deterministic because every
    # factory call follows a frozen role/canonical-candidate order.
    runtime_environments: list[ReplayEnvironment] = []
    role_binder = getattr(factory, "bind_operation_role", None)
    def bind_role(role: str) -> None:
        if callable(role_binder):
            role_binder(role)
    def tracked_factory(seed: int) -> ReplayEnvironment:
        env = factory(seed)
        if any(env is prior for prior in runtime_environments):
            raise BuildError("factory reused an environment object")
        binder = getattr(env, "bind_factory_ordinal", None)
        if not callable(binder):
            raise BuildError("environment lacks factory-ordinal audit binding")
        binder(len(runtime_environments))
        runtime_environments.append(env)
        return env
    # _run_witness receives tracked_factory rather than the outer counted
    # factory.  Forward the role hook so failed witness transitions retain the
    # same frozen witness_fresh_prefix/witness_planner grammar as successful
    # rows instead of the temporary caller labels.
    tracked_factory.bind_operation_role = bind_role  # type: ignore[attr-defined]

    bind_role("prefix_planning")
    seed_env = tracked_factory(int(world_seed))
    prefix: list[int] = []
    prefix_plan_steps = 0
    prefix_planning_trace: list[Mapping[str, Any]] = []
    try:
        observation = seed_env.reset()
        seed_reset_audit = dict(seed_env.reset_audit())
        if seed_reset_audit.get("reset_calls") != 1 \
                or seed_reset_audit.get("underlying_episode_before_first_reset") != 0 \
                or seed_reset_audit.get("underlying_episode_after_first_reset") != 1 \
                or seed_reset_audit.get("qualified") is not True:
            raise BuildError("prefix-planning environment reset audit failed")
        seed_info = _initial_info(seed_env, view_factory)
        seed_view = view_factory(seed_env, seed_info)
        original_step = seed_env.step
        def counted_step(action: int):
            nonlocal prefix_plan_steps
            prefix_plan_steps += 1
            result = original_step(action)
            observation, reward, done, info = result
            prefix_planning_trace.append(_trace_row(prefix_plan_steps, int(action), observation, reward, done, info))
            return result
        seed_env.step = counted_step  # type: ignore[method-assign]
        prefix = list(prefix_planner(seed_env, seed_view, designated_target, horizon))
        if prefix_plan_steps != len(prefix):
            raise BuildError("prefix planner action record does not equal executed transition count")
        if len(prefix) + horizon > 10_000:
            raise BuildError("prefix plus horizon exceeds 10000")
    finally:
        seed_env.close()

    replays: list[ReplaySnapshot] = []
    suffix_hashes: list[str] = []
    suffix_traces: list[list[Mapping[str, Any]]] = []
    suffix = verification_suffix(world_seed)
    prefix_planning_transitions = prefix_plan_steps
    for replay_index in range(RECONSTRUCTIONS_REQUIRED):
        bind_role(f"reconstruction:{replay_index + 1}")
        env, _view, snapshot = reconstruct(tracked_factory, view_factory, world_seed, prefix)
        try:
            replays.append(snapshot)
            if len(replays) <= 2:
                bind_role(f"verification_suffix:{len(replays)}")
                suffix_trace, suffix_hash = _run_suffix(env, snapshot, suffix)
                suffix_traces.append(suffix_trace)
                suffix_hashes.append(suffix_hash)
        finally:
            env.close()
    fingerprints = [canonical_sha256(snapshot.fingerprint) for snapshot in replays]
    if len(set(fingerprints)) != 1:
        raise BuildError("three fresh handoff reconstructions differ")
    if len(set(suffix_hashes)) != 1:
        raise BuildError("two fresh verification suffix traces differ")
    handoff = replays[0].fingerprint
    counts = handoff["achievement_counts"]
    if counts[designated_target] != 0:
        raise BuildError("designated target already achieved at handoff")

    bind_role("designated_witness")
    designated_attempt, designated_counts, designated_reset_audit = _run_witness(
        tracked_factory, view_factory, world_seed, prefix, designated_target, horizon, handoff
    )
    if not designated_attempt["success"]:
        raise BuildError(f"designated witness failed: {designated_attempt['failure_reason']}")

    eligible = [
        name for name in ACTION_DIRECTED_ACHIEVEMENTS
        if name != designated_target and counts[name] == 0
    ]
    attempts: dict[str, Any] = {}
    candidate_transition_counts: dict[str, dict[str, int]] = {}
    candidate_reset_audits: dict[str, Mapping[str, Any]] = {}
    for candidate in eligible:
        bind_role(f"candidate:{candidate}")
        attempt, attempt_counts, attempt_reset_audit = _run_witness(
            tracked_factory, view_factory, world_seed, prefix, candidate, horizon, handoff
        )
        attempts[candidate] = attempt
        candidate_transition_counts[candidate] = attempt_counts
        candidate_reset_audits[candidate] = attempt_reset_audit

    emitted_ordinals = ([seed_reset_audit.get("factory_ordinal")]
        + [snapshot.reset_audit.get("factory_ordinal") for snapshot in replays]
        + [designated_reset_audit.get("factory_ordinal")]
        + [audit.get("factory_ordinal") for audit in candidate_reset_audits.values()])
    if emitted_ordinals != list(range(len(runtime_environments))):
        raise BuildError("factory audit ordinals do not match frozen call order")

    designated_length = len(designated_attempt["actions"])
    matches = sorted(
        (abs(designated_length - len(attempt["actions"])), CANONICAL_INDEX[name], name)
        for name, attempt in attempts.items()
        if attempt["success"]
        and abs(designated_length - len(attempt["actions"])) <= 16
        and abs(designated_length - len(attempt["actions"]))
        / max(designated_length, len(attempt["actions"])) <= 0.2
    )
    if not matches:
        raise BuildError("no physically feasible witness-length-matched decoy")
    decoy = matches[0][2]
    replay_prefix_transitions = RECONSTRUCTIONS_REQUIRED * len(prefix)
    suffix_transitions = 2 * len(suffix)
    designated_total = designated_counts["fresh_prefix"] + designated_counts["planner_actions"]
    candidate_total = sum(values["fresh_prefix"] + values["planner_actions"]
                          for values in candidate_transition_counts.values())
    total = (prefix_planning_transitions + replay_prefix_transitions + suffix_transitions
             + designated_total + candidate_total)
    row: dict[str, Any] = {
        "schema": STATE_ROW_SCHEMA_ID,
        "state_id": f"q0-{phase}-{designated_target}-w{world_seed}",
        "designated_target": designated_target,
        "matched_decoy": decoy,
        "tier": TARGET_SPEC[designated_target]["tier"],
        "horizon": horizon,
        "world_seed": int(world_seed),
        "primary": bool(primary),
        "reserve_rank": reserve_rank,
        "prefix_actions": prefix,
        "prefix_actions_sha256": canonical_sha256(prefix),
        "prefix_trace": list(replays[0].trace),
        "prefix_planning_trace": prefix_planning_trace,
        "reconstruction_prefix_traces": [list(snapshot.trace) for snapshot in replays],
        "designated_witness_actions": designated_attempt["actions"],
        "designated_attempt": designated_attempt,
        "decoy_witness_actions": attempts[decoy]["actions"],
        "handoff_fingerprint": handoff,
        "verification_suffix_actions": suffix,
        "verification_suffix_trace_sha256": suffix_hashes[0],
        "verification_suffix_traces": suffix_traces,
        "candidate_attempts": attempts,
        "planner_sha256": planner_sha256(),
        "planner_provenance": planner_provenance(),
        "builder_sha256": builder_sha256(),
        "builder_provenance": builder_provenance(),
        "prior_reset_attempt": PRIOR_RESET_ATTEMPT,
        "transition_counts": {
            "prefix_planning": prefix_planning_transitions,
            "three_reconstruction_prefixes": replay_prefix_transitions,
            "two_verification_suffixes": suffix_transitions,
            "designated_fresh_prefix": designated_counts["fresh_prefix"],
            "designated_planner_actions": designated_counts["planner_actions"],
            "candidate_fresh_prefixes": sum(value["fresh_prefix"] for value in candidate_transition_counts.values()),
            "candidate_planner_actions": sum(value["planner_actions"] for value in candidate_transition_counts.values()),
            "candidate_breakdown": candidate_transition_counts,
            "total_q0a_transitions": total,
        },
        "reset_audits": {
            "prefix_planning": seed_reset_audit,
            "three_reconstructions": [dict(snapshot.reset_audit) for snapshot in replays],
            "designated_witness": dict(designated_reset_audit),
            "candidate_witnesses": {key: dict(value) for key, value in candidate_reset_audits.items()},
            "all_instances_unique": True,
        },
        "qualification": "QUALIFIED_POLICY_BLIND_HANDOFF",
    }
    row["canonical_sha256"] = canonical_sha256(row)
    validate_state_row(row, phase=phase)
    return row


def build_phase(*, factory: EnvironmentFactory, view_factory: ViewFactory,
                prefix_planner: PrefixPlanner, phase: str,
                pre_factory_gate: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build the complete fixed namespace for one phase without adaptive dropping."""
    if phase == "dev":
        namespace = DEV_WORLD_TARGET
        expected_per_target = 10
    elif phase == "formal":
        namespace = FORMAL_WORLD_TARGET
        expected_per_target = 20
    else:
        raise BuildError("phase must be dev or formal")
    qualified: dict[str, list[dict[str, Any]]] = {name: [] for name in TARGET_SPEC}
    failures: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    phase_transition_total = 0
    if pre_factory_gate is None:
        raise RuntimeQualificationGateError(
            "runtime qualification gate is required before the first world factory")
    qualification_binding: Mapping[str, Any] | None = None
    gate_checked = False
    def qualified_factory(seed: int) -> ReplayEnvironment:
        nonlocal gate_checked, qualification_binding
        if not gate_checked:
            try:
                candidate = pre_factory_gate()
                validate_runtime_qualification_binding(candidate)
            except Exception as error:
                raise RuntimeQualificationGateError(
                    "runtime qualification revalidation failed immediately before world factory"
                ) from error
            qualification_binding = dict(candidate)
            gate_checked = True
        return factory(seed)
    def successful_operation_counts(row: Mapping[str, Any]) -> dict[str, int]:
        prefix_len = len(row["prefix_actions"])
        result = {"prefix_planning": prefix_len,
                  "reconstruction:1": prefix_len, "verification_suffix:1": 16,
                  "reconstruction:2": prefix_len, "verification_suffix:2": 16,
                  "reconstruction:3": prefix_len,
                  f"witness_fresh_prefix:{row['designated_target']}": prefix_len,
                  f"witness_planner:{row['designated_target']}": len(row["designated_attempt"]["trace"])}
        for name, attempt in row["candidate_attempts"].items():
            result[f"witness_fresh_prefix:{name}"] = prefix_len
            result[f"witness_planner:{name}"] = len(attempt["trace"])
        return result
    for world_seed, target in namespace.items():
        transition_count = 0
        partial_journal: list[dict[str, Any]] = []
        phase_factory_ordinal = 0
        operation_role = "unbound"
        def bind_operation_role(role: str) -> None:
            nonlocal operation_role
            operation_role = role
        def counted_factory(seed: int) -> ReplayEnvironment:
            nonlocal transition_count, phase_factory_ordinal, phase_transition_total
            env = qualified_factory(seed)
            ordinal = phase_factory_ordinal
            phase_factory_ordinal += 1
            original_step = env.step
            def counted_step(action: int):
                nonlocal transition_count, phase_transition_total
                if phase_transition_total >= PHASE_Q0A_TRANSITION_UPPER_BOUND[phase]:
                    raise BuildError("phase Q0A transition upper bound reached before step")
                result = original_step(action)
                transition_count += 1
                phase_transition_total += 1
                try:
                    observation, reward, done, info = result
                    outcome = _trace_row(transition_count, int(action), observation, reward, done, info)
                except Exception as exc:
                    outcome = {"step": transition_count, "action": int(action), "error_type": type(exc).__name__,
                               "error_message": str(exc), "outcome_status": "TRACE_ENCODING_ERROR"}
                    partial_journal.append({
                        "phase": phase, "operation_role": operation_role, "world_seed": world_seed,
                        "environment_factory_ordinal": ordinal, "transition_ordinal": transition_count,
                        "action": int(action), "outcome": outcome,
                    })
                    raise
                partial_journal.append({
                    "phase": phase, "operation_role": operation_role, "world_seed": world_seed,
                    "environment_factory_ordinal": ordinal, "transition_ordinal": transition_count,
                    "action": int(action), "outcome": outcome,
                })
                return result
            env.step = counted_step  # type: ignore[method-assign]
            return env
        counted_factory.bind_operation_role = bind_operation_role  # type: ignore[attr-defined]
        try:
            row = build_candidate_state(
                factory=counted_factory, view_factory=view_factory,
                prefix_planner=prefix_planner, world_seed=world_seed,
                designated_target=target, phase=phase,
            )
            if row["transition_counts"]["total_q0a_transitions"] != transition_count:
                raise BuildError("phase counter and state row transition counts differ")
            qualified[target].append(row)
            disposition = "QUALIFIED"
            error_type = error_message = None
            operation_counts = successful_operation_counts(row)
        except RuntimeQualificationGateError:
            # Infrastructure qualification is not a scientific candidate
            # failure and may never be converted into a partial state bank.
            raise
        except Exception as exc:  # preserve every candidate disposition
            error_type, error_message = type(exc).__name__, str(exc)
            operation_counts: dict[str, int] = {}
            for entry in partial_journal:
                role = entry["operation_role"]
                operation_counts[role] = operation_counts.get(role, 0) + 1
            failures.append({
                "world_seed": world_seed, "designated_target": target,
                "error_type": error_type, "error_message": error_message,
                "q0a_transitions": transition_count,
                "q0a_transition_counts_by_operation": operation_counts,
                "partial_transition_journal": partial_journal,
            })
            disposition = "FAILED"
        dispositions.append({"world_seed": world_seed, "designated_target": target,
                             "status": disposition, "q0a_transitions": transition_count,
                             "error_type": error_type, "error_message": error_message,
                             "q0a_transition_counts_by_operation": operation_counts,
                             "partial_transition_journal": partial_journal if disposition == "FAILED" else []})
    if phase == "formal":
        for target, rows in qualified.items():
            rows.sort(key=lambda row: row["world_seed"])
            for index, row in enumerate(rows):
                row["primary"] = index < 16
                row["reserve_rank"] = None if index < 16 else index - 15
                row["canonical_sha256"] = canonical_sha256(row, omit=("canonical_sha256",))
                validate_state_row(row, phase=phase)
    flat = [row for target in TARGET_SPEC for row in qualified[target]]
    per_target = {target: len(rows) for target, rows in qualified.items()}
    if phase == "dev":
        status = "DEV_FEASIBILITY_PASS" if all(
            count == expected_per_target for count in per_target.values()
        ) else "DEV_FEASIBILITY_FAIL"
    else:
        status = "FORMAL_STATE_BANK_QUALIFIED" if all(
            count >= 16 for count in per_target.values()
        ) else "STATE_BANK_NOT_QUALIFIED"
    payload = {
        "schema": "alpagd-q0-policy-blind-state-bank/v1",
        "phase": phase,
        "status": status,
        "states": flat,
        "failures": failures,
        "counts_by_target": per_target,
        "expected_candidates_per_target": expected_per_target,
        "formal_worlds_opened": phase == "formal",
        "planner_provenance": planner_provenance(),
        "builder_provenance": builder_provenance(),
        "prior_reset_attempt": PRIOR_RESET_ATTEMPT,
        "runtime_qualification": dict(qualification_binding or {}),
        "world_dispositions": dispositions,
        "q0a_transitions_total": sum(row["q0a_transitions"] for row in dispositions),
        "q0a_transition_upper_bound": PHASE_Q0A_TRANSITION_UPPER_BOUND[phase],
    }
    if payload["q0a_transitions_total"] > payload["q0a_transition_upper_bound"]:
        raise BuildError("phase Q0A transition upper bound exceeded")
    payload["canonical_sha256"] = canonical_sha256(payload)
    validate_state_bank_manifest(payload, phase=phase)
    return payload


def write_json_atomic(path: Path, value: Any) -> None:
    """Create a fresh artifact durably; existing evidence is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Destination was checked fresh above; link preserves no-clobber even
        # if another writer races between the check and publication.
        os.link(temporary, path)
        if path.read_bytes() != payload:
            raise OSError("atomic JSON readback mismatch")
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass
