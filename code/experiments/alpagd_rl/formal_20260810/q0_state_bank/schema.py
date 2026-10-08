"""Pure-stdlib schemas and validators for Q0 state-bank evidence."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

SCHEMA_ID = "alpagd-q0-policy-blind-state-bank/v1"
STATE_ROW_SCHEMA_ID = "alpagd-q0-policy-blind-state/v1"
RUNTIME_QUALIFICATION_BINDING_SCHEMA = "alpagd-q0-runtime-qualification-binding/v1"

CANONICAL_ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
)
ACTION_DIRECTED_ACHIEVEMENTS = CANONICAL_ACHIEVEMENTS[:-1]
CANONICAL_INDEX = {name: index for index, name in enumerate(CANONICAL_ACHIEVEMENTS)}
ACTION_NAMES = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)
TARGETS = (
    ("collect_wood", "shallow", 128),
    ("place_table", "shallow", 128),
    ("collect_stone", "medium", 128),
    ("make_stone_pickaxe", "medium", 128),
    ("place_furnace", "medium", 128),
    ("collect_iron", "deep", 256),
    ("make_iron_pickaxe", "deep", 256),
    ("collect_diamond", "deep", 256),
)
TARGET_SPEC = {name: {"tier": tier, "horizon": horizon} for name, tier, horizon in TARGETS}
DEV_WORLD_TARGET = {
    759001 + target_index * 10 + offset: target
    for target_index, (target, _, _) in enumerate(TARGETS)
    for offset in range(10)
}
FORMAL_WORLD_TARGET = {
    760001 + target_index * 20 + offset: target
    for target_index, (target, _, _) in enumerate(TARGETS)
    for offset in range(20)
}

HANDOFF_FINGERPRINT_FIELDS = (
    "observation_sha256", "achieved_mask", "achievement_counts",
    "inventory_state_canonical_sha256", "player_position",
    "semantic_array_sha256", "prefix_trace_sha256", "target_age_start_normalized",
)
PHASE_Q0A_TRANSITION_UPPER_BOUND = {"dev": 20_000_000, "formal": 40_000_000}
VERIFICATION_SUFFIX_LENGTH = 16
STATE_ROW_REQUIRED = (
    "schema", "state_id", "designated_target", "matched_decoy", "tier",
    "horizon", "world_seed", "primary", "reserve_rank", "prefix_actions", "prefix_actions_sha256",
    "prefix_trace", "prefix_planning_trace", "reconstruction_prefix_traces", "verification_suffix_traces",
    "designated_witness_actions", "decoy_witness_actions",
    "designated_attempt",
    "handoff_fingerprint", "verification_suffix_actions",
    "verification_suffix_trace_sha256", "candidate_attempts",
    "planner_sha256", "planner_provenance", "builder_sha256",
    "builder_provenance", "transition_counts", "qualification",
    "reset_audits", "prior_reset_attempt",
    "canonical_sha256",
)


class StateBankValidationError(ValueError):
    """The state-bank artifact does not satisfy the frozen Q0 contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any, *, omit: Sequence[str] = ()) -> str:
    if omit and isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in set(omit)}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def expected_verification_suffix(world_seed: int) -> list[int]:
    """Pure-stdlib authority for the preregistered policy-independent suffix."""
    if not isinstance(world_seed, int) or isinstance(world_seed, bool):
        raise StateBankValidationError("verification suffix world seed type")
    digest = hashlib.sha256(f"q0-verification-suffix/v1:{world_seed}".encode()).digest()
    return [digest[index] % 5 for index in range(VERIFICATION_SUFFIX_LENGTH)]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def validate_runtime_qualification_binding(binding: Mapping[str, Any]) -> None:
    fields = {
        "schema", "receipt_path", "receipt_file_sha256", "receipt_canonical_sha256",
        "profile_path", "profile_sha256", "environment_lock_path", "environment_lock_sha256",
        "pip_freeze_path", "pip_freeze_sha256", "source_inventory_canonical_sha256",
        "python_executable", "canonical_sha256",
    }
    hash_fields = fields - {"schema", "receipt_path", "profile_path",
                            "environment_lock_path", "pip_freeze_path",
                            "python_executable", "canonical_sha256"}
    if (not isinstance(binding, Mapping) or set(binding) != fields
            or binding.get("schema") != RUNTIME_QUALIFICATION_BINDING_SCHEMA
            or any(not isinstance(binding.get(field), str) or not binding[field]
                   for field in ("receipt_path", "profile_path", "environment_lock_path",
                                 "pip_freeze_path", "python_executable"))
            or any(not _is_sha256(binding.get(field)) for field in hash_fields)
            or not _is_sha256(binding.get("canonical_sha256"))
            or binding.get("canonical_sha256") != canonical_sha256(
                binding, omit=("canonical_sha256",))):
        raise StateBankValidationError("runtime qualification binding invalid")


def _validate_actions(name: str, actions: Any, *, maximum: int | None = None) -> None:
    if not isinstance(actions, list) or any(
        not isinstance(action, int) or isinstance(action, bool)
        or not 0 <= action < len(ACTION_NAMES) for action in actions
    ):
        raise StateBankValidationError(f"{name} must be a list of Crafter action indices")
    if maximum is not None and len(actions) > maximum:
        raise StateBankValidationError(f"{name} exceeds its frozen bound")


TRACE_ROW_FIELDS = {"step", "action", "observation_sha256", "reward", "done",
                    "achievement_counts", "inventory", "player_position", "semantic_array_sha256"}


def _validate_trace_row(name: str, row: Any, *, expected_step: int,
                        expected_action: int | None) -> None:
    if not isinstance(row, Mapping) or set(row) != TRACE_ROW_FIELDS:
        raise StateBankValidationError(f"{name} trace row exact keys")
    counts = row["achievement_counts"]
    if (row["step"] != expected_step or row["action"] != expected_action
            or not _is_sha256(row["observation_sha256"])
            or not _is_sha256(row["semantic_array_sha256"])
            or not isinstance(row["reward"], (int, float)) or isinstance(row["reward"], bool)
            or not math.isfinite(float(row["reward"])) or not isinstance(row["done"], bool)
            or not isinstance(counts, Mapping) or set(counts) != set(CANONICAL_ACHIEVEMENTS)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values())
            or not isinstance(row["inventory"], Mapping)
            or not isinstance(row["player_position"], list) or len(row["player_position"]) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in row["player_position"])):
        raise StateBankValidationError(f"{name} trace row semantics")


def _validate_trace(name: str, trace: Any, *, actions: Sequence[int], includes_reset: bool = False) -> None:
    if not isinstance(trace, list) or len(trace) != len(actions) + int(includes_reset):
        raise StateBankValidationError(f"{name} trace length mismatch")
    for index, row in enumerate(trace):
        expected_step = index if includes_reset else index + 1
        expected_action = None if includes_reset and index == 0 else actions[index - int(includes_reset)]
        _validate_trace_row(name, row, expected_step=expected_step, expected_action=expected_action)


def validate_candidate_attempt(name: str, attempt: Any, horizon: int,
                               handoff_fingerprint_sha256: str) -> None:
    if not isinstance(attempt, Mapping):
        raise StateBankValidationError(f"candidate attempt {name} must be an object")
    expected = {
        "target", "attempted", "success", "actions", "action_names",
        "first_success_step", "initial_count", "final_count", "terminal",
        "failure_reason", "trace", "trace_sha256", "fresh_replay_fingerprint_sha256",
    }
    if set(attempt) != expected or attempt.get("target") != name or attempt.get("attempted") is not True:
        raise StateBankValidationError(f"candidate attempt {name} shape/identity mismatch")
    _validate_actions(f"candidate_attempts.{name}.actions", attempt["actions"], maximum=horizon)
    if attempt["action_names"] != [ACTION_NAMES[index] for index in attempt["actions"]]:
        raise StateBankValidationError(f"candidate attempt {name} action names mismatch")
    if not isinstance(attempt["success"], bool) or not isinstance(attempt["terminal"], bool):
        raise StateBankValidationError(f"candidate attempt {name} boolean fields")
    if (not isinstance(attempt["initial_count"], int) or isinstance(attempt["initial_count"], bool)
            or not isinstance(attempt["final_count"], int) or isinstance(attempt["final_count"], bool)):
        raise StateBankValidationError(f"candidate attempt {name} count fields")
    _validate_trace(f"candidate_attempts.{name}", attempt["trace"], actions=attempt["actions"])
    trace_counts = [row["achievement_counts"][name] for row in attempt["trace"]]
    recomputed_final = trace_counts[-1] if trace_counts else attempt["initial_count"]
    first_increase = next((index for index, count in enumerate(trace_counts, 1)
                           if count > attempt["initial_count"]), None)
    recomputed_success = first_increase is not None
    recomputed_terminal = bool(attempt["trace"] and attempt["trace"][-1]["done"])
    if (attempt["final_count"] != recomputed_final or attempt["success"] != recomputed_success
            or attempt["first_success_step"] != first_increase or attempt["terminal"] != recomputed_terminal
            or (recomputed_success and first_increase != len(attempt["trace"]))):
        raise StateBankValidationError(f"candidate attempt {name} trace-derived disposition mismatch")
    if attempt["success"]:
        if attempt["final_count"] <= attempt["initial_count"]:
            raise StateBankValidationError(f"candidate attempt {name} success without count increase")
        if attempt["first_success_step"] != len(attempt["actions"]):
            raise StateBankValidationError(f"candidate attempt {name} is padded or step is inconsistent")
        if attempt["failure_reason"] is not None:
            raise StateBankValidationError(f"candidate attempt {name} successful failure reason")
    elif attempt["first_success_step"] is not None or not isinstance(attempt["failure_reason"], str):
        raise StateBankValidationError(f"candidate attempt {name} failure disposition")
    for field in ("trace_sha256", "fresh_replay_fingerprint_sha256"):
        if not _is_sha256(attempt[field]):
            raise StateBankValidationError(f"candidate attempt {name} invalid {field}")
    if (attempt["trace_sha256"] != canonical_sha256(attempt["trace"])
            or attempt["fresh_replay_fingerprint_sha256"] != handoff_fingerprint_sha256):
        raise StateBankValidationError(f"candidate attempt {name} hash binding mismatch")


def validate_state_row(row: Mapping[str, Any], *, phase: str | None = None) -> None:
    if not isinstance(row, Mapping) or set(row) != set(STATE_ROW_REQUIRED):
        missing = sorted(set(STATE_ROW_REQUIRED) - set(row)) if isinstance(row, Mapping) else []
        extra = sorted(set(row) - set(STATE_ROW_REQUIRED)) if isinstance(row, Mapping) else []
        raise StateBankValidationError(f"state row exact keys mismatch: missing={missing}, extra={extra}")
    if row["schema"] != STATE_ROW_SCHEMA_ID:
        raise StateBankValidationError("state row schema mismatch")
    target = row["designated_target"]
    if target not in TARGET_SPEC:
        raise StateBankValidationError("unknown designated target")
    spec = TARGET_SPEC[target]
    if row["tier"] != spec["tier"] or row["horizon"] != spec["horizon"]:
        raise StateBankValidationError("target tier/horizon mismatch")
    world = row["world_seed"]
    if not isinstance(world, int) or isinstance(world, bool):
        raise StateBankValidationError("world seed type")
    if phase == "dev" and DEV_WORLD_TARGET.get(world) != target:
        raise StateBankValidationError("dev world/target namespace mismatch")
    if phase == "formal" and FORMAL_WORLD_TARGET.get(world) != target:
        raise StateBankValidationError("formal world/target namespace mismatch")
    if phase is not None and row["state_id"] != f"q0-{phase}-{target}-w{world}":
        raise StateBankValidationError("state id mismatch")
    if row["matched_decoy"] not in ACTION_DIRECTED_ACHIEVEMENTS or row["matched_decoy"] == target:
        raise StateBankValidationError("invalid matched decoy")
    if not isinstance(row["primary"], bool):
        raise StateBankValidationError("primary must be boolean")
    if row["reserve_rank"] is not None and (
        not isinstance(row["reserve_rank"], int) or isinstance(row["reserve_rank"], bool)
        or not 1 <= row["reserve_rank"] <= 4
    ):
        raise StateBankValidationError("reserve rank")
    if row["primary"] and row["reserve_rank"] is not None:
        raise StateBankValidationError("primary row cannot have reserve rank")
    _validate_actions("prefix_actions", row["prefix_actions"])
    _validate_actions("designated_witness_actions", row["designated_witness_actions"], maximum=spec["horizon"])
    _validate_actions("decoy_witness_actions", row["decoy_witness_actions"], maximum=spec["horizon"])
    _validate_actions("verification_suffix_actions", row["verification_suffix_actions"], maximum=16)
    if len(row["verification_suffix_actions"]) != 16:
        raise StateBankValidationError("verification suffix must contain exactly 16 actions")
    if row["verification_suffix_actions"] != expected_verification_suffix(world):
        raise StateBankValidationError("verification suffix does not match world-seed derivation")
    if len(row["prefix_actions"]) + spec["horizon"] > 10_000:
        raise StateBankValidationError("prefix plus horizon exceeds environment length")
    fingerprint = row["handoff_fingerprint"]
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != set(HANDOFF_FINGERPRINT_FIELDS):
        raise StateBankValidationError("handoff fingerprint exact fields")
    if len(fingerprint["achieved_mask"]) != 22 or any(value not in (0, 1) for value in fingerprint["achieved_mask"]):
        raise StateBankValidationError("handoff achieved mask")
    if (not isinstance(fingerprint["target_age_start_normalized"], (int, float))
            or isinstance(fingerprint["target_age_start_normalized"], bool)
            or float(fingerprint["target_age_start_normalized"]) != 0.0):
        raise StateBankValidationError("handoff target age must start at normalized zero")
    if row["prefix_actions_sha256"] != canonical_sha256(row["prefix_actions"]):
        raise StateBankValidationError("prefix actions hash mismatch")
    if (not isinstance(row["prefix_trace"], list)
            or fingerprint["prefix_trace_sha256"] != canonical_sha256(row["prefix_trace"])):
        raise StateBankValidationError("prefix raw trace binding mismatch")
    planning_trace = row["prefix_planning_trace"]
    if (not isinstance(planning_trace, list) or len(planning_trace) != len(row["prefix_actions"])
            or [entry.get("action") for entry in planning_trace] != row["prefix_actions"]):
        raise StateBankValidationError("online prefix planning trace mismatch")
    _validate_trace("prefix_trace", row["prefix_trace"], actions=row["prefix_actions"], includes_reset=True)
    _validate_trace("prefix_planning_trace", planning_trace, actions=row["prefix_actions"])
    if planning_trace != row["prefix_trace"][1:]:
        raise StateBankValidationError("online prefix planning and reconstruction evidence differ")
    replay_traces = row["reconstruction_prefix_traces"]
    if (not isinstance(replay_traces, list) or len(replay_traces) != 3
            or any(trace != row["prefix_trace"] for trace in replay_traces)):
        raise StateBankValidationError("three reconstruction prefix traces mismatch")
    for index, trace in enumerate(replay_traces, 1):
        _validate_trace(f"reconstruction_prefix_traces.{index}", trace,
                        actions=row["prefix_actions"], includes_reset=True)
    suffix_traces = row["verification_suffix_traces"]
    if (not isinstance(suffix_traces, list) or len(suffix_traces) != 2
            or any(not isinstance(trace, list) for trace in suffix_traces)
            or len({canonical_sha256(trace) for trace in suffix_traces}) != 1
            or canonical_sha256(suffix_traces[0]) != row["verification_suffix_trace_sha256"]):
        raise StateBankValidationError("verification suffix raw trace binding mismatch")
    for index, trace in enumerate(suffix_traces, 1):
        _validate_trace(f"verification_suffix_traces.{index}", trace,
                        actions=row["verification_suffix_actions"])
    if suffix_traces[0] != suffix_traces[1]:
        raise StateBankValidationError("verification suffix traces not byte-identical")
    if set(fingerprint["achievement_counts"]) != set(CANONICAL_ACHIEVEMENTS) or any(
        not isinstance(value, int) or value < 0 for value in fingerprint["achievement_counts"].values()
    ):
        raise StateBankValidationError("handoff raw achievement counts")
    if fingerprint["achievement_counts"][target] != 0 or fingerprint["achievement_counts"][row["matched_decoy"]] != 0:
        raise StateBankValidationError("designated or decoy already achieved at handoff")
    if not (isinstance(fingerprint["player_position"], list) and len(fingerprint["player_position"]) == 2
            and all(isinstance(value, int) for value in fingerprint["player_position"])):
        raise StateBankValidationError("player position")
    for field in (
        "observation_sha256", "inventory_state_canonical_sha256",
        "semantic_array_sha256", "prefix_trace_sha256",
    ):
        if not _is_sha256(fingerprint[field]):
            raise StateBankValidationError(f"handoff fingerprint invalid {field}")
    handoff_row = row["prefix_trace"][-1]
    recomputed_fingerprint = {
        "observation_sha256": handoff_row["observation_sha256"],
        "achieved_mask": [int(handoff_row["achievement_counts"][name] > 0)
                          for name in CANONICAL_ACHIEVEMENTS],
        "achievement_counts": handoff_row["achievement_counts"],
        "inventory_state_canonical_sha256": canonical_sha256(handoff_row["inventory"]),
        "player_position": handoff_row["player_position"],
        "semantic_array_sha256": handoff_row["semantic_array_sha256"],
        "prefix_trace_sha256": canonical_sha256(row["prefix_trace"]),
        "target_age_start_normalized": 0.0,
    }
    if fingerprint != recomputed_fingerprint:
        raise StateBankValidationError("handoff fingerprint does not recompute from prefix trace")
    handoff_sha256 = canonical_sha256(fingerprint)
    attempts = row["candidate_attempts"]
    expected_attempts = {
        candidate for candidate in ACTION_DIRECTED_ACHIEVEMENTS
        if candidate != target and fingerprint["achievement_counts"][candidate] == 0
    }
    if not isinstance(attempts, Mapping) or set(attempts) != expected_attempts:
        raise StateBankValidationError("candidate attempts must cover every eligible action-directed decoy")
    for name, attempt in attempts.items():
        validate_candidate_attempt(name, attempt, spec["horizon"], handoff_sha256)
        if attempt["initial_count"] != fingerprint["achievement_counts"][name]:
            raise StateBankValidationError("candidate initial count not bound to handoff")
        if (not isinstance(attempt.get("trace"), list)
                or attempt.get("trace_sha256") != canonical_sha256(attempt["trace"])):
            raise StateBankValidationError("candidate witness raw trace binding mismatch")
    designated_attempt = row["designated_attempt"]
    validate_candidate_attempt(target, designated_attempt, spec["horizon"], handoff_sha256)
    if designated_attempt["initial_count"] != fingerprint["achievement_counts"][target]:
        raise StateBankValidationError("designated initial count not bound to handoff")
    if (not isinstance(designated_attempt.get("trace"), list)
            or designated_attempt.get("trace_sha256") != canonical_sha256(designated_attempt["trace"])
            or designated_attempt["actions"] != row["designated_witness_actions"]):
        raise StateBankValidationError("designated witness raw trace binding mismatch")
    for attempt in [designated_attempt, *attempts.values()]:
        if (len(attempt["trace"]) != len(attempt["actions"])
                or [item.get("step") for item in attempt["trace"]] != list(range(1, len(attempt["trace"]) + 1))
                or [item.get("action") for item in attempt["trace"]] != attempt["actions"]):
            raise StateBankValidationError("witness trace/action/step mismatch")
    if (len(row["prefix_trace"]) != len(row["prefix_actions"]) + 1
            or [item.get("step") for item in row["prefix_trace"]] != list(range(len(row["prefix_trace"])))
            or [item.get("action") for item in row["prefix_trace"]][1:] != row["prefix_actions"]):
        raise StateBankValidationError("prefix trace/action/step mismatch")
    counts = row["transition_counts"]
    prefix_len = len(row["prefix_actions"])
    candidate_prefix = prefix_len * len(attempts)
    candidate_planner = sum(len(attempt["trace"]) for attempt in attempts.values())
    recomputed = {
        "prefix_planning": prefix_len,
        "three_reconstruction_prefixes": 3 * prefix_len,
        "two_verification_suffixes": sum(len(trace) for trace in suffix_traces),
        "designated_fresh_prefix": prefix_len,
        "designated_planner_actions": len(designated_attempt["trace"]),
        "candidate_fresh_prefixes": candidate_prefix,
        "candidate_planner_actions": candidate_planner,
    }
    if any(counts.get(key) != value for key, value in recomputed.items()):
        raise StateBankValidationError("state transition counts do not recompute from raw traces")
    breakdown = counts.get("candidate_breakdown")
    expected_breakdown = {name: {"fresh_prefix": prefix_len, "planner_actions": len(attempt["trace"])}
                          for name, attempt in attempts.items()}
    if breakdown != expected_breakdown or counts.get("total_q0a_transitions") != sum(recomputed.values()):
        raise StateBankValidationError("candidate/total transition count mismatch")
    designated_length = len(row["designated_witness_actions"])
    decoy_length = len(row["decoy_witness_actions"])
    selected = attempts[row["matched_decoy"]]
    if not selected["success"] or selected["actions"] != row["decoy_witness_actions"]:
        raise StateBankValidationError("selected decoy witness mismatch")
    if designated_length < 1 or decoy_length < 1:
        raise StateBankValidationError("zero-length witness forbidden")
    difference = abs(designated_length - decoy_length)
    if difference > 16 or difference / max(designated_length, decoy_length) > 0.2:
        raise StateBankValidationError("witness lengths do not satisfy both frozen matching constraints")
    successful = sorted(
        (abs(designated_length - len(attempt["actions"])), CANONICAL_INDEX[name], name)
        for name, attempt in attempts.items()
        if attempt["success"]
        and abs(designated_length - len(attempt["actions"])) <= 16
        and abs(designated_length - len(attempt["actions"]))
        / max(designated_length, len(attempt["actions"])) <= 0.2
    )
    if not successful or successful[0][2] != row["matched_decoy"]:
        raise StateBankValidationError("matched decoy does not obey deterministic tie-break")
    if row["qualification"] != "QUALIFIED_POLICY_BLIND_HANDOFF":
        raise StateBankValidationError("state qualification disposition")
    for field in ("planner_sha256", "builder_sha256", "verification_suffix_trace_sha256"):
        if not _is_sha256(row[field]):
            raise StateBankValidationError(f"invalid {field}")
    if (not isinstance(row["planner_provenance"], Mapping)
            or row["planner_provenance"].get("planner_sha256") != row["planner_sha256"]
            or not isinstance(row["builder_provenance"], Mapping)
            or row["builder_provenance"].get("builder_sha256") != row["builder_sha256"]):
        raise StateBankValidationError("planner/builder provenance hash mismatch")
    if row["canonical_sha256"] != canonical_sha256(row, omit=("canonical_sha256",)):
        raise StateBankValidationError("state row canonical hash mismatch")
    prior = row["prior_reset_attempt"]
    if not isinstance(prior, Mapping) or prior.get("world_seed") != 759001 \
            or prior.get("environment_steps") != 0 \
            or prior.get("planner_called") is not False \
            or prior.get("checkpoint_loaded_or_called") is not False \
            or prior.get("used_for_design_or_state_selection") is not False:
        raise StateBankValidationError("prior dev reset attempt disclosure")
    audits = row["reset_audits"]
    if not isinstance(audits, Mapping) or set(audits) != {
        "prefix_planning", "three_reconstructions", "designated_witness",
        "candidate_witnesses", "all_instances_unique",
    } or audits["all_instances_unique"] is not True:
        raise StateBankValidationError("reset audit shape")
    all_audits = [audits["prefix_planning"], *audits["three_reconstructions"],
                  audits["designated_witness"], *audits["candidate_witnesses"].values()]
    if len(audits["three_reconstructions"]) != 3 or set(audits["candidate_witnesses"]) != set(attempts):
        raise StateBankValidationError("reset audit cardinality")
    if any(audit.get("reset_calls") != 1
           or audit.get("underlying_episode_before_first_reset") != 0
           or audit.get("underlying_episode_after_first_reset") != 1
           or audit.get("qualified") is not True for audit in all_audits):
        raise StateBankValidationError("reset audit episode semantics")
    ordinals = [audit.get("factory_ordinal") for audit in all_audits]
    if ordinals != list(range(len(all_audits))):
        raise StateBankValidationError("environment factory ordinals do not follow canonical fresh-call order")


def validate_state_bank_manifest(manifest: Mapping[str, Any], *, phase: str) -> None:
    expected_manifest_fields = {"schema", "phase", "status", "states", "failures",
        "counts_by_target", "expected_candidates_per_target", "formal_worlds_opened",
        "planner_provenance", "builder_provenance", "prior_reset_attempt",
        "world_dispositions", "q0a_transitions_total", "q0a_transition_upper_bound",
        "runtime_qualification",
        "canonical_sha256"}
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_fields:
        raise StateBankValidationError("state-bank manifest exact fields")
    namespace = DEV_WORLD_TARGET if phase == "dev" else FORMAL_WORLD_TARGET if phase == "formal" else None
    if namespace is None or manifest.get("schema") != SCHEMA_ID or manifest.get("phase") != phase:
        raise StateBankValidationError("state-bank phase/schema mismatch")
    validate_runtime_qualification_binding(manifest.get("runtime_qualification"))
    dispositions = manifest.get("world_dispositions")
    if not isinstance(dispositions, list) or len(dispositions) != len(namespace):
        raise StateBankValidationError("every world disposition must be retained")
    worlds, total = set(), 0
    state_rows, failure_rows = manifest.get("states", []), manifest.get("failures", [])
    if not isinstance(state_rows, list) or not isinstance(failure_rows, list):
        raise StateBankValidationError("state/failure rows must be lists")
    states = {row.get("world_seed"): row for row in state_rows if isinstance(row, Mapping)}
    failures = {row.get("world_seed"): row for row in failure_rows if isinstance(row, Mapping)}
    if len(states) != len(state_rows) or len(failures) != len(failure_rows):
        raise StateBankValidationError("state/failure world seeds must be unique mappings")
    candidate_order = list(ACTION_DIRECTED_ACHIEVEMENTS)
    def role_rank(role: str, target: str) -> tuple[int, int]:
        fixed = {"prefix_planning": (0, 0), "reconstruction:1": (1, 0),
                 "verification_suffix:1": (2, 0), "reconstruction:2": (3, 0),
                 "verification_suffix:2": (4, 0), "reconstruction:3": (5, 0),
                 f"witness_fresh_prefix:{target}": (6, 0),
                 f"witness_planner:{target}": (7, 0)}
        if role in fixed:
            return fixed[role]
        for prefix, offset in (("witness_fresh_prefix:", 0), ("witness_planner:", 1)):
            if role.startswith(prefix) and role[len(prefix):] in candidate_order \
                    and role[len(prefix):] != target:
                return (8 + 2 * candidate_order.index(role[len(prefix):]) + offset, 0)
        raise StateBankValidationError("partial journal operation role outside frozen grammar")

    def state_operation_counts(state: Mapping[str, Any]) -> dict[str, int]:
        prefix_len = len(state["prefix_actions"])
        result = {"prefix_planning": prefix_len,
                  "reconstruction:1": prefix_len, "verification_suffix:1": 16,
                  "reconstruction:2": prefix_len, "verification_suffix:2": 16,
                  "reconstruction:3": prefix_len,
                  f"witness_fresh_prefix:{state['designated_target']}": prefix_len,
                  f"witness_planner:{state['designated_target']}": len(state["designated_attempt"]["trace"])}
        for name, attempt in state["candidate_attempts"].items():
            result[f"witness_fresh_prefix:{name}"] = prefix_len
            result[f"witness_planner:{name}"] = len(attempt["trace"])
        return result
    for item in dispositions:
        if not isinstance(item, Mapping) or set(item) != {"world_seed", "designated_target", "status", "q0a_transitions", "error_type", "error_message", "q0a_transition_counts_by_operation", "partial_transition_journal"}:
            raise StateBankValidationError("world disposition exact fields")
        world, target, count = item["world_seed"], item["designated_target"], item["q0a_transitions"]
        journal = item["partial_transition_journal"]
        operation_counts = item["q0a_transition_counts_by_operation"]
        if (world in worlds or namespace.get(world) != target or item["status"] not in {"QUALIFIED", "FAILED"}
                or not isinstance(count, int) or isinstance(count, bool) or count < 0):
            raise StateBankValidationError("world disposition invalid")
        if (not isinstance(journal, list) or (item["status"] == "FAILED" and len(journal) != count)
                or (item["status"] == "QUALIFIED" and journal != [])):
            raise StateBankValidationError("failed world partial journal mismatch")
        if (not isinstance(operation_counts, Mapping)
                or any(not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for key, value in operation_counts.items())):
            raise StateBankValidationError("operation transition ledger shape")
        previous_rank = (-1, -1)
        journal_operation_counts: dict[str, int] = {}
        for index, entry in enumerate(journal, 1):
            outcome = entry.get("outcome")
            if (not isinstance(entry, Mapping) or set(entry) != {"phase", "operation_role", "world_seed",
                    "environment_factory_ordinal", "transition_ordinal", "action", "outcome"}
                    or entry.get("phase") != phase or entry.get("world_seed") != world
                    or not isinstance(entry.get("operation_role"), str) or not entry["operation_role"]
                    or not isinstance(entry.get("environment_factory_ordinal"), int)
                    or isinstance(entry.get("environment_factory_ordinal"), bool)
                    or entry.get("transition_ordinal") != index or not isinstance(outcome, Mapping)
                    or outcome.get("step") != index or outcome.get("action") != entry.get("action")):
                raise StateBankValidationError("partial journal transition identity")
            rank = role_rank(entry["operation_role"], target)
            if rank < previous_rank:
                raise StateBankValidationError("partial journal operation order")
            previous_rank = rank
            journal_operation_counts[entry["operation_role"]] = journal_operation_counts.get(entry["operation_role"], 0) + 1
            if outcome.get("outcome_status") == "TRACE_ENCODING_ERROR":
                if set(outcome) != {"step", "action", "error_type", "error_message", "outcome_status"} \
                        or not isinstance(outcome["error_type"], str) or not isinstance(outcome["error_message"], str):
                    raise StateBankValidationError("partial journal error outcome shape")
            else:
                _validate_trace_row("partial_transition_journal", outcome,
                                    expected_step=index, expected_action=entry["action"])
        if item["status"] == "QUALIFIED":
            state = states.get(world)
            if (state is None or count != state.get("transition_counts", {}).get("total_q0a_transitions")
                    or operation_counts != state_operation_counts(state)
                    or sum(operation_counts.values()) != count
                    or world in failures or item["error_type"] is not None or item["error_message"] is not None):
                raise StateBankValidationError("qualified disposition/state join mismatch")
        else:
            failure = failures.get(world)
            if (failure is None or set(failure) != {"world_seed", "designated_target", "error_type", "error_message", "q0a_transitions", "q0a_transition_counts_by_operation", "partial_transition_journal"}
                    or failure.get("world_seed") != world or failure.get("designated_target") != target
                    or failure.get("error_type") != item["error_type"]
                    or failure.get("error_message") != item["error_message"]
                    or failure.get("q0a_transitions") != count
                    or operation_counts != journal_operation_counts
                    or failure.get("q0a_transition_counts_by_operation") != operation_counts
                    or failure.get("partial_transition_journal") != journal or world in states):
                raise StateBankValidationError("failed disposition/failure join mismatch")
        worlds.add(world); total += count
    if (worlds != set(namespace) or manifest.get("q0a_transitions_total") != total
            or manifest.get("q0a_transition_upper_bound") != PHASE_Q0A_TRANSITION_UPPER_BOUND[phase]
            or total > PHASE_Q0A_TRANSITION_UPPER_BOUND[phase]):
        raise StateBankValidationError("Q0A resource ledger invalid")
    if set(states) | set(failures) != set(namespace) or set(states) & set(failures):
        raise StateBankValidationError("state/failure disposition partition mismatch")
    for row in manifest.get("states", []):
        validate_state_row(row, phase=phase)
    expected_counts = {target: sum(row["designated_target"] == target for row in state_rows)
                       for target in TARGET_SPEC}
    expected_per_target = 10 if phase == "dev" else 20
    expected_status = ("DEV_FEASIBILITY_PASS" if all(value == expected_per_target for value in expected_counts.values())
                       else "DEV_FEASIBILITY_FAIL") if phase == "dev" else (
                       "FORMAL_STATE_BANK_QUALIFIED" if all(value >= 16 for value in expected_counts.values())
                       else "STATE_BANK_NOT_QUALIFIED")
    if (manifest.get("counts_by_target") != expected_counts
            or manifest.get("expected_candidates_per_target") != expected_per_target
            or manifest.get("formal_worlds_opened") is not (phase == "formal")
            or manifest.get("status") != expected_status):
        raise StateBankValidationError("state-bank count/status semantics")
    if state_rows:
        if (manifest.get("planner_provenance") != state_rows[0]["planner_provenance"]
                or manifest.get("builder_provenance") != state_rows[0]["builder_provenance"]
                or any(row["planner_provenance"] != manifest["planner_provenance"]
                       or row["builder_provenance"] != manifest["builder_provenance"] for row in state_rows)):
            raise StateBankValidationError("manifest provenance/state binding")
    if manifest.get("canonical_sha256") != canonical_sha256(manifest, omit=("canonical_sha256",)):
        raise StateBankValidationError("state-bank canonical hash mismatch")


def state_row_json_schema() -> dict[str, Any]:
    """Return a Draft 2020-12 shape schema; semantic validation remains above."""
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    actions = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 16}}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": STATE_ROW_SCHEMA_ID,
        "type": "object",
        "required": list(STATE_ROW_REQUIRED),
        "additionalProperties": False,
        "properties": {
            "schema": {"const": STATE_ROW_SCHEMA_ID},
            "state_id": {"type": "string", "minLength": 1},
            "designated_target": {"enum": list(TARGET_SPEC)},
            "matched_decoy": {"enum": list(ACTION_DIRECTED_ACHIEVEMENTS)},
            "tier": {"enum": ["shallow", "medium", "deep"]},
            "horizon": {"enum": [128, 256]},
            "world_seed": {"type": "integer"},
            "primary": {"type": "boolean"},
            "reserve_rank": {"type": ["integer", "null"], "minimum": 1, "maximum": 4},
            "prefix_actions": actions,
            "prefix_actions_sha256": sha,
            "prefix_trace": {"type": "array", "items": {"type": "object"}},
            "prefix_planning_trace": {"type": "array", "items": {"type": "object"}},
            "reconstruction_prefix_traces": {"type": "array", "minItems": 3, "maxItems": 3},
            "verification_suffix_traces": {"type": "array", "minItems": 2, "maxItems": 2},
            "designated_witness_actions": actions,
            "designated_attempt": {"type": "object"},
            "decoy_witness_actions": actions,
            "handoff_fingerprint": {"type": "object"},
            "verification_suffix_actions": {**actions, "minItems": 16, "maxItems": 16},
            "verification_suffix_trace_sha256": sha,
            "candidate_attempts": {"type": "object"},
            "planner_sha256": sha,
            "planner_provenance": {"type": "object"},
            "builder_sha256": sha,
            "builder_provenance": {"type": "object"},
            "transition_counts": {"type": "object"},
            "reset_audits": {"type": "object"},
            "prior_reset_attempt": {"type": "object"},
            "qualification": {"const": "QUALIFIED_POLICY_BLIND_HANDOFF"},
            "canonical_sha256": sha,
        },
    }
