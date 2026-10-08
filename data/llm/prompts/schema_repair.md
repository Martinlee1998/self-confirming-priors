# ALPAGD-RL Strict Graph Prior Capture Prompt Schema Repair v1

You are producing a fallible structural prior for ALPAGD-RL.

Use only the Crafter manual context and the exact achievement ID enum below. Return JSON only. The output must match schema `alpagd_graph_prior_schema_v1`.

Top-level object must contain exactly these fields:

- schema_version
- provider_model
- prompt_hash
- environment
- source_context_id
- node_priors
- edge_priors
- global_uncertainty_note

Fixed top-level values:

- schema_version: `alpagd_graph_prior_schema_v1`
- provider_model: `deepseek-v4-pro`
- prompt_hash: `sha256:TO_BE_FILLED_BY_CAPTURE_SCRIPT`
- environment: `Crafter`
- source_context_id: `crafter_achievements_v1`

Each `node_priors` item must contain exactly these fields:

- achievement_id
- name
- description
- prior_relevance_probability
- confidence
- rationale
- uncertainty_note

Node field rules:

- Include exactly one node prior for every achievement ID in the enum.
- Use the exact achievement ID string from the enum.
- `prior_relevance_probability` must be a number between 0.01 and 0.99.
- `confidence` must be a number between 0.0 and 1.0.
- `name`, `description`, `rationale`, and `uncertainty_note` must be non-empty strings.

Each `edge_priors` item must contain exactly these fields:

- from_achievement
- to_achievement
- edge_type
- prior_probability
- confidence
- rationale
- uncertainty_note
- known_counterexample_possibility

Edge field rules:

- `from_achievement` and `to_achievement` must be exact achievement IDs from the enum.
- Do not create self-edges.
- Do not invent achievement IDs.
- `edge_type` must be one of: `none`, `hard_prerequisite`, `soft_facilitator`, `unlock`, `contextual_dependency`.
- `prior_probability` must be a number between 0.01 and 0.99.
- `confidence` must be a number between 0.0 and 1.0.
- `rationale`, `uncertainty_note`, and `known_counterexample_possibility` must be non-empty strings.
- Use concise strings.
- Include at most 80 directed edge priors.

Return compact JSON only. Do not include markdown fences or explanatory text.

Achievement ID enum:

```json
{{ACHIEVEMENT_ID_ENUM_JSON}}
```

Crafter manual context:

```text
{{MANUAL_CONTEXT}}
```
