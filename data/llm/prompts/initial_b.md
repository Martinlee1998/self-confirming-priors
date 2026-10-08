# ALPAGD-RL LLM Prior Capture Prompt v1B

Task: produce a conservative probabilistic achievement dependency prior for Crafter.

Use only the manual context and achievement identifiers below. Treat all labels as uncertain structural priors. Future environment interaction evidence is allowed to override them.

Use the schema `alpagd_graph_prior_schema_v1` and source context `crafter_achievements_v1`.

Allowed edge labels:

- none
- hard_prerequisite
- soft_facilitator
- unlock
- contextual_dependency

Conservatism rule:

- Use `hard_prerequisite` only when the manual context strongly implies that the target achievement is nearly impossible without the source achievement or the resource/state it produces.
- Use `unlock` when the source changes reachable tools, resources, or affordances.
- Use `soft_facilitator` when the source likely helps but is not necessary.
- Use `contextual_dependency` when the relation depends on layout, nearby objects, inventory, or seed context.
- Use `none` for unsupported apparent shortcuts.

Calibration rule:

- Prior probabilities must be non-dogmatic: 0.01 <= p <= 0.99.
- Confidence must be in [0.0, 1.0].
- High confidence requires a direct manual basis.
- Every entry must include rationale and uncertainty.

Return JSON only. Do not include markdown fences.

The caller will replace `sha256:TO_BE_FILLED_BY_CAPTURE_SCRIPT` with the rendered prompt hash when storing metadata.

Manual context:

```text
{{MANUAL_CONTEXT}}
```

Achievements:

```json
{{ACHIEVEMENT_LIST_JSON}}
```
