# Research data

These records accompany *Self-Confirming Errors in LLM-Derived Priors for
Reinforcement Learning in Open-Ended Games*. The five studies use frozen
controllers in Crafter. Acquisition updates a private achievement graph;
evaluation measures behavior after that graph has been frozen.

## Study map

| Paper study | Question | Raw records | Result summary |
| --- | --- | --- | --- |
| Study 1 | How does changing the imposed target order affect progress? | `raw/study1.tar.gz` | `summaries/study1.json` |
| Study 2, P1 | Can natural acquisition repair the reversed A–B relation? | `raw/study2.tar.gz` (`P1/`) | `summaries/study2_p1.json` |
| Study 2, P2 | Can natural acquisition repair the reversed B–C relation? | `raw/study2.tar.gz` (`P2/`) | `summaries/study2_p2.json` |
| Study 3 | Does supplying the missing ordering evidence repair P2? | `raw/study3.tar.gz` | `summaries/study3.json` |
| Study 4, P1 | What does frontier evidence seeking do when natural acquisition already exposes the relation? | `raw/study4.tar.gz` (`P1/`) | `summaries/study4_p1.json` |
| Study 4, P2 | Can frontier evidence seeking expose the otherwise missing evidence? | `raw/study4.tar.gz` (`P2/`) | `summaries/study4_p2.json` |
| Study 5 | What happens when the provider is initialized with the LLM-derived five-achievement projection? | `raw/study5.tar.gz` | `summaries/study5.json` |

A–E denote `place_table`, `make_wood_pickaxe`, `collect_stone`,
`make_stone_pickaxe`, and `collect_iron`, respectively. P1 reverses A→B;
P2 reverses B→C. The five-achievement experiment is a controlled fixture,
not a claim that all 22 Crafter achievements have been solved.

## Reading the records

Raw records are distributed as compressed JSON archives. From the repository
root, extract and verify them with:

```sh
python analysis/reproduce.py --unpack-only
```

The records are written to `outputs/reproduction/raw/`, grouped by dataset.
`raw/manifest.json` lists archive and record checksums and record counts.
Study 1 stores its 800 paired rows within the fixed-contrast result and the
four controller shards; the later studies store one JSON record per world
and run.

One acquisition run is one controller and one random-number replicate. Four
controllers and two replicates give eight runs per condition. Each run has 40
acquisition worlds and 100 fresh evaluation worlds. The acquisition and
evaluation seed sets are disjoint. Conditions within a study share the start
state and random-number streams in each evaluation row.

Raw JSON records retain their original field and condition names. The last
three fields of an `identity` record the controller, replicate (`stream`),
and world seed. Study 2 additionally prefixes the identity with `P1` or `P2`.
An evaluation record contains the outcomes for its paired conditions under
`arms`. Acquisition records include observations and graph counts needed to
follow the update. A `result.json` or a file in `summaries/` is an aggregate,
not an additional independent experiment.

Some early records use different condition names from the final paper:

| Archived name | Paper name | Meaning |
| --- | --- | --- |
| `U` in Study 2 | N | Natural acquisition from the directed prior |
| `D` in Study 2 | S | Symmetric prior |
| `E` in Study 3 | X | Order-informed exposure during acquisition |
| `ES` in Studies 4–5 | Frontier evidence seeking | Acquisition challenges selected at the blocked frontier |
| `L5_0`, `L5_N`, `L5_ES` | L0, LN, LES | Initial, naturally updated, and evidence-seeking LLM projections |

In Study 1, archived `P1`–`P4` labels refer to the four adjacent target-order
swaps (AB, BC, CD, DE); this is distinct from the P1/P2 graph reversals used
in later studies.

The behavioral independent unit is a world: paired differences are averaged
over the eight controller–replicate runs within each of the 100 worlds.
Acquisition and provider quantities have eight run-level units. The 800
evaluation rows, 320 acquisition rows, and 31 provider-query masks are not
independent behavioral samples. A query mask is the achieved/not-achieved
five-bit state over A–E, excluding the all-complete state.

## Comparators

`raw/baseline_selectors.tar.gz` contains the Random, Uncertainty, and Decision
Impact comparator records, grouped under `p1/` and `p2/`.
`raw/baseline_sea.tar.gz` contains the SEA-A recovered graphs under
`recovered/p1/` and `recovered/p2/`, and evaluation records under `p1_full/`
and `p2_full/`. `comparisons/p1/` and `comparisons/p2/`
contain the CSV and JSON analyses used for the comparator figures.

The selector comparators have their own acquisition and evaluation seed sets.
SEA-A uses the natural-acquisition records and evaluation seed sets of Study 4.
Each comparator is paired only with N in its own evaluation population;
records from the two populations should not be paired or pooled. SEA-A is a
precedence-count reconstruction inspired by SEA, not a reproduction of SEA's
representation learning or policy training.

## LLM prior and annotations

`llm/parsed_outputs/` contains the three retained graph proposals.
`llm/prompts/` gives the elicitation templates, and `llm/context/` supplies
the Crafter manual context and achievement list inserted into the templates.
The prior probability of an ordered pair is the mean
across the three proposals; an omitted pair has probability 0.01 within that
proposal. Study 5 retains the 20 ordered-pair probabilities among A–E and
uses a Beta concentration of 10. No new LLM call is needed to analyze these
records.

`annotations/` contains the project reference graph and the two completed
pairwise annotation files. Each pairwise file covers the 462 ordered pairs
among 22 achievements, excluding self-pairs. Their `edge_present` columns
contain 73 and 81 positives, respectively, with 21 shared positives and 329
shared negatives. These annotations document reference agreement, not
verified causal ground truth.

This release provides the recorded observations and analyses supporting the
paper. Raw world-level outcome records are not frame-by-frame video or full
action traces; regenerating trajectories additionally requires the frozen
controller checkpoints and the matching environment implementation.
