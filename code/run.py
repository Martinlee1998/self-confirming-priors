"""Entry points for the paper experiments and figure scripts.

Author: Yanqi Li

Run ``python code/run.py <experiment> --help`` for the original runner's
arguments. Acquisition and evaluation require the four final checkpoints;
merge, comparison analysis, and projection checks use retained records only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

CODE = Path(__file__).resolve().parent
REPO = CODE.parent
SOURCE = CODE / "experiments/alpagd_rl/formal_20260810"
RUNNERS = {
    "study1": "pc3_v10_real_integration/four_controller_fixed_contrast.py",
    "study2": "pc3_v11_graph_recovery/run_graph_recovery.py",
    "study3": "pc3_v12_evidence_availability/run_evidence_availability.py",
    "study4-p2": "pc3_v13_evidence_seeking/run_evidence_seeking.py",
    "study4-p1": "pc3_v15_p1_evidence_seeking/run_p1_evidence_seeking.py",
    "study5": "pc3_v14_l5_bridge/run_l5_bridge.py",
    "sea-a": "pc3_v16_sea_adapted_dependency_recovery/run_sea_adapted_dependency_recovery.py",
    "selectors": "pc3_v18_discriminating_selector_baselines/run_selector_baselines.py",
    "comparison": "pc3_v19_comparative_analysis/analyze_comparative_results.py",
}


def configure_paths() -> None:
    for path in (CODE, SOURCE / "pc3_v10_real_integration"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    defaults = {
        "PC3_V10_SRC": SOURCE / "pc3_v10_real_integration",
        "PC3_V9_ROOT": CODE,
        "PC3_V11_SRC": SOURCE / "pc3_v11_graph_recovery",
        "PC3_V13_SRC": SOURCE / "pc3_v13_evidence_seeking",
        "PC3_CHECKPOINT_DIR": REPO / "checkpoints",
        "PC3_L5_SOURCE_ROOT": REPO,
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, str(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=tuple(RUNNERS) + ("figures",))
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    configure_paths()
    if args.experiment == "figures":
        if remaining:
            parser.error("figures does not take additional arguments")
        for name in ("figs_a.py", "figs_b.py"):
            subprocess.run(
                [sys.executable, str(CODE / "figures_src" / name)],
                cwd=CODE / "figures_src", check=True,
            )
        print("Figures 3–10 saved in code/generated_figures/.")
        return 0
    source = SOURCE / RUNNERS[args.experiment]
    sys.argv = [str(source), *remaining]
    runpy.run_path(str(source), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
