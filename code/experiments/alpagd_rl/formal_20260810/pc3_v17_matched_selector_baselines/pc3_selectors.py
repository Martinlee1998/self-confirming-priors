"""Compatibility export for the local selector rules."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("selector_rules.py")
_SPEC = importlib.util.spec_from_file_location("pc3_v17_local_selector_rules", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load selector implementation: {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

choose_es = _MODULE.choose_es
choose_random = _MODULE.choose_random
choose_uncertainty = _MODULE.choose_uncertainty
common_frontier = _MODULE.common_frontier
