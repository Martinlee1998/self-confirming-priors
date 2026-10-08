# ALPAGD-RL LLM Prior Capture Prompt v1C

You are estimating uncertain directed relations among Crafter achievements for an RL research audit.

The output is a language prior, not the graph truth. The useful answer is one that exposes uncertainty, possible exceptions, and likely counterexamples.

Inputs available to you:

- Crafter manual context supplied below.
- The fixed achievement identifier list supplied below.
- The allowed edge types supplied below.

Allowed edge types:

- none
- hard_prerequisite
- soft_facilitator
- unlock
- contextual_dependency

Output requirements:

- JSON only.
- `schema_version` must be `alpagd_graph_prior_schema_v1`.
- `environment` must be `Crafter`.
- `source_context_id` must be `crafter_achievements_v1`.
- `provider_model` must describe the model used by the caller.
- `prompt_hash` may be `sha256:TO_BE_FILLED_BY_CAPTURE_SCRIPT` before capture metadata is attached.
- Include exactly one node prior for every listed achievement.
- Include directed edge priors only when there is a meaningful structural relation or a useful explicit negative edge.
- Use probabilities in [0.01, 0.99], never 0 or 1.
- Explain known uncertainty and counterexample possibility for each edge.

Do not output executable code or step-by-step action instructions.

Manual context:

```text
{{MANUAL_CONTEXT}}
```

Achievement list:

```json
{{ACHIEVEMENT_LIST_JSON}}
```
