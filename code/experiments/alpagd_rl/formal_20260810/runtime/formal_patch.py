"""Frozen formal-use deterministic patch for Crafter 1.8.3.

This module is self-contained and versioned for the 2026-08-10 formal
protocol.  It intentionally does not import or reuse legacy M2R patch code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path
from typing import Any, Mapping


PATCH_ID = "alpagd_formal_20260810_crafter_deterministic_chunks_v1"
PATCH_TARGET = "crafter.engine.World.chunks"
PATCH_SCOPE = "alpagd_rl_formal_20260810_worker_runtime"
EXPECTED_CRAFTER_VERSION = "1.8.3"
EXPECTED_ENGINE_SHA256 = "6ba8e03e7e141b730378a286e3f3cc2778f88f785bd60992b36cfce142789936"
EXPECTED_ENV_SHA256 = "a728d30479c4c0b2a3fe39505169af4a861e114132bde76d6c05d235da0171b6"
PATCH_SPEC = {
    "patch_id": PATCH_ID,
    "scope": PATCH_SCOPE,
    "target": PATCH_TARGET,
    "underlying_storage": "World._chunks remains unchanged as defaultdict(set)",
    "property_contract": (
        "World.chunks returns a shallow dict whose per-chunk values are tuples "
        "ordered by stable World._objects creation index"
    ),
    "ordering_key": "World._objects creation index",
    "expected_crafter_version": EXPECTED_CRAFTER_VERSION,
    "expected_engine_sha256": EXPECTED_ENGINE_SHA256,
    "expected_env_sha256": EXPECTED_ENV_SHA256,
}
PATCH_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(PATCH_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def formal_patch_contract() -> dict[str, str]:
    return {
        "patch_id": PATCH_ID,
        "scope": PATCH_SCOPE,
        "patch_target": PATCH_TARGET,
        "ordering_key": PATCH_SPEC["ordering_key"],
        "expected_crafter_version": EXPECTED_CRAFTER_VERSION,
        "expected_engine_sha256": EXPECTED_ENGINE_SHA256,
        "expected_env_sha256": EXPECTED_ENV_SHA256,
        "patch_contract_sha256": PATCH_CONTRACT_SHA256,
    }


def crafter_source_fingerprint() -> dict[str, str]:
    import crafter
    import crafter.engine as crafter_engine
    import crafter.env as crafter_env

    engine_path = Path(inspect.getfile(crafter_engine))
    env_path = Path(inspect.getfile(crafter_env))
    return {
        "crafter_version": importlib.metadata.version("crafter"),
        "crafter_package_file": str(Path(crafter.__file__).resolve()),
        "python_version": platform.python_version(),
        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "env_sha256": hashlib.sha256(env_path.read_bytes()).hexdigest(),
    }


def validate_source_fingerprint(observed: Mapping[str, str]) -> dict[str, str]:
    """Pure fail-closed validation, separated for dependency-free testing."""

    expected = {
        "crafter_version": EXPECTED_CRAFTER_VERSION,
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "env_sha256": EXPECTED_ENV_SHA256,
    }
    actual = {key: observed.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "Crafter source fingerprint does not match the frozen formal_20260810 "
            f"backend: observed={actual!r}, expected={expected!r}"
        )
    return dict(observed)


def apply_formal_deterministic_patch() -> dict[str, Any]:
    import crafter.engine as crafter_engine

    observed = validate_source_fingerprint(crafter_source_fingerprint())
    current = crafter_engine.World.chunks
    current_getter = current.fget if isinstance(current, property) else None
    installed_id = getattr(current_getter, "_alpagd_formal_patch_id", None)
    legacy_id = getattr(current_getter, "_alpagd_patch_id", None)
    if legacy_id is not None:
        raise RuntimeError(
            f"World.chunks already has a non-formal/legacy patch installed: {legacy_id}"
        )
    if installed_id is not None:
        if installed_id != PATCH_ID:
            raise RuntimeError(f"World.chunks already has a different formal patch: {installed_id}")
        if getattr(current_getter, "_alpagd_formal_patch_contract_sha256", None) != PATCH_CONTRACT_SHA256:
            raise RuntimeError("formal patch identity matches but contract hash differs")
        return {**formal_patch_contract(), "observed_crafter_source": observed, "patch_applied": True}
    if current_getter is None:
        raise RuntimeError("Crafter World.chunks is not the expected property")

    original_getter = current_getter

    def deterministic_chunks(world: Any) -> dict[Any, tuple[Any, ...]]:
        chunks = original_getter(world)
        creation_index = {
            id(obj): index
            for index, obj in enumerate(getattr(world, "_objects", []))
            if obj is not None
        }
        ordered: dict[Any, tuple[Any, ...]] = {}
        for chunk, objects in chunks.items():
            missing = [obj for obj in objects if id(obj) not in creation_index]
            if missing:
                raise RuntimeError("Crafter chunk contains an object absent from World._objects")
            ordered[chunk] = tuple(sorted(objects, key=lambda obj: creation_index[id(obj)]))
        return ordered

    deterministic_chunks._alpagd_formal_patch_id = PATCH_ID  # type: ignore[attr-defined]
    deterministic_chunks._alpagd_formal_patch_contract_sha256 = PATCH_CONTRACT_SHA256  # type: ignore[attr-defined]
    crafter_engine.World.chunks = property(
        deterministic_chunks, current.fset, current.fdel, current.__doc__
    )
    return {**formal_patch_contract(), "observed_crafter_source": observed, "patch_applied": True}
