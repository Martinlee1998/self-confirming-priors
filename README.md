# Self-confirming priors

Research code and data accompanying **Self-Confirming Errors in LLM-Derived
Priors for Reinforcement Learning in Open-Ended Games**.

Author: **Yanqi Li**

The experiments study how an incorrect achievement dependency can prevent a
reinforcement-learning agent from collecting the evidence needed to correct
it. MEND maintains a structural prior, updates it with observed achievement
order, and uses frontier challenges to seek missing evidence. Five studies
examine target-order sensitivity, natural repair, order-informed exposure,
frontier evidence seeking, and an LLM-derived prior. Comparators include
Random, Uncertainty, Decision Impact, and a precedence-count reconstruction
inspired by SEA.

## Contents

- [`data/`](data/README.md): acquisition and evaluation records, result
  summaries, comparator analyses, retained LLM proposals and prompts, and
  pairwise annotations.
- [`analysis/reproduce.py`](analysis/reproduce.py): archive verification and
  reanalysis of recorded outcomes, using only the Python standard library.
- [`code/`](code/README.md): graph updates, target providers, acquisition
  rules, frozen-controller evaluation, and figure-generation scripts.
- [`figures/`](figures/README.md): three selected figures. The manuscript
  itself is not included.

## Reproduce the recorded results

Use Python 3.10 or later:

```bash
git clone https://github.com/Martinlee1998/self-confirming-priors.git
cd self-confirming-priors
python analysis/reproduce.py
```

This command verifies and unpacks the seven data archives, recomputes the
world-level contrasts for Studies 1–5, regenerates both comparator analyses,
and checks the retained LLM projection. It writes the results to
`outputs/reproduction/`. No environment interaction, model training, or LLM
API call is needed.

For extraction alone:

```bash
python analysis/reproduce.py --unpack-only
```

## Run the implementation

The original experiments used Ubuntu. See [`code/README.md`](code/README.md)
for dependencies, checks, and runner arguments. Generating new trajectories
requires the four final controller checkpoints, available on request from
the author. They are not needed to reproduce the recorded analyses.

## Interpret the data

The behavioral measure `R` is the achieved canonical prefix length divided
by five. A–E correspond to placing a table, making a wood pickaxe, collecting
stone, making a stone pickaxe, and collecting iron. P1 and P2 reverse A→B
and B→C, respectively. The controlled five-achievement experiments and the
22-achievement LLM proposals are distinct; Study 5 connects them by retaining
the LLM probabilities among the five achievements.

Behavioral contrasts average the eight controller–replicate outcomes within
each world before estimating uncertainty across 100 worlds. Acquisition and
graph-recovery quantities use eight runs. The selector comparators and the
Study 4/SEA-A comparisons use separate world sets; the reanalysis preserves
their respective pairings. All conditions are included, including negative
and zero effects. [`data/README.md`](data/README.md) explains the archived
condition names and record schemas.

Machine-identifying paths have been removed from the public data. Outcomes,
seeds, condition names, evidence counts, and graph probabilities are retained.

## Citation and attribution

Use [`CITATION.cff`](CITATION.cff) to cite this repository. The source and
licensing notice for the Crafter manual material is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
