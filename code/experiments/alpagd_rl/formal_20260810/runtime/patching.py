"""Worker-local installation of the formal_20260810 Crafter patch.

Call :func:`install_worker_determinism` inside every worker, before constructing
``crafter.Env``.  Importing this module alone deliberately does not mutate
Crafter.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .formal_patch import apply_formal_deterministic_patch


def _patch_module_path() -> Path:
    return Path(__file__).with_name("formal_patch.py").resolve()


def install_worker_determinism() -> dict[str, Any]:
    """Validate Crafter 1.8.3 source hashes and patch this worker's process."""

    path = _patch_module_path()
    evidence = apply_formal_deterministic_patch()
    return {
        **evidence,
        "worker_pid": os.getpid(),
        "installation_scope": "worker_process_before_env_construction",
        "patch_module": str(path),
        "patch_module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
