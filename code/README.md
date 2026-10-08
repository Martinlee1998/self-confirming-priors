# Experiment code

Author: Yanqi Li

The implementation contains the five controlled Crafter studies, the comparison rules, and scripts for Figures 3–10. It retains the algorithms and random-number streams used for the reported results. The root README describes the retained data and record-only reproduction workflow.

## Entry points

Run commands from the repository root:

```bash
python code/run.py --help
python code/run.py study2 --help
python code/run.py comparison schema-test
python code/run.py study5 check-projection --output-root outputs/study5
python -m unittest discover -s code/tests -v
```

The experiment names are `study1`, `study2`, `study3`, `study4-p1`, `study4-p2`, `study5`, `sea-a`, `selectors`, and `comparison`. Each runner exposes its own arguments through `--help`. The last three commands above need no environment rollouts, policy training, or LLM requests.

The original record labels are preserved: Study 2 uses `U` for the paper's natural-update condition `N` and `D` for symmetric condition `S`; Study 3 uses `E` for the order-exposure condition `X`. Study 5 uses `L5_0`, `L5_N`, and `L5_ES` for the initial, naturally acquired, and evidence-seeking language priors. The selector records use `R`, `UQ`, and `ES` for Random, Uncertainty, and DI.

## Running acquisition or evaluation

These stages need Crafter and the four final controller checkpoints, which are available on request. Place them at:

```text
checkpoints/controller_2026081911/ppo_latest.pt
checkpoints/controller_2026081912/ppo_latest.pt
checkpoints/controller_2026081913/ppo_latest.pt
checkpoints/controller_2026081914/ppo_latest.pt
```

Set `PC3_CHECKPOINT_DIR` to use another checkpoint directory. Study 1 accepts checkpoints explicitly with `--checkpoint` and additionally needs the retained 100-world list through `--worlds-json`. Only load checkpoints from a trusted source.

`requirements-rollout.txt` records the currently observed Ubuntu runtime. It is not a cryptographic lock of the original training environment. Select a compatible Torch build for the execution device; the observed build was `2.13.0+cu130`, while the requirements file pins the package version without prescribing a build index. No training launchers or API clients are included.

## Figures

The plotting dependencies are separate from the rollout dependencies. In a plotting environment:

```bash
python -m pip install -r code/requirements-figures.txt
python code/run.py figures
```

The scripts save Figures 3–10 to `code/generated_figures/` using the retained plotting inputs. They do not overwrite the published figures. The selected figure PDFs are provided separately; editable generators for Figures 0–2 are not part of this release.

The deterministic Crafter patch is retained byte-for-byte. Its `.gitattributes` rule preserves the module fingerprint across operating systems.
