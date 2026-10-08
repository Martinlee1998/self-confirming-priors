# ALPAGD-RL LLM Prior Capture Prompt v1A

You are providing a fallible structural prior for ALPAGD-RL.

Use only the provided Crafter manual context and achievement list. Infer likely achievement dependency edges as uncertain language-derived priors. The graph is not final truth; future environment interaction evidence can retain, discount, or override each edge.

Environment: Crafter

Schema version: alpagd_graph_prior_schema_v1

Source context id: crafter_achievements_v1

Allowed edge types:

- none
- hard_prerequisite
- soft_facilitator
- unlock
- contextual_dependency

Probability rules:

- Every probability must be between 0.01 and 0.99.
- Do not use 0 or 1.
- Use lower confidence when the manual context is ambiguous.
- Include a concrete uncertainty note for every node and edge.

Return valid JSON only with these top-level fields:

- schema_version
- provider_model
- prompt_hash
- environment
- source_context_id
- node_priors
- edge_priors
- global_uncertainty_note

The `prompt_hash` field must be filled by the caller in the rendered prompt metadata. If unknown, set it to the exact placeholder string `sha256:TO_BE_FILLED_BY_CAPTURE_SCRIPT`.

For every achievement in the provided list, include one node prior.

For edge priors, include only candidate directed edges that the manual context makes plausible plus any explicit `none` edges needed to mark common but unsupported shortcuts. Each edge must include:

- from_achievement
- to_achievement
- edge_type
- prior_probability
- confidence
- rationale
- uncertainty_note
- known_counterexample_possibility

Do not output executable code, action instructions, or claims that the graph is true.

Manual context:

```text
{{MANUAL_CONTEXT}}
```

Achievement list:

```json
{{ACHIEVEMENT_LIST_JSON}}
```
