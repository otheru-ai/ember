# Model cards

Ember accepts an optional JSON sidecar through `--model-card PATH`. The card
supplies model-specific output-budget and sampler defaults without requiring a
rebuild. Explicit request fields and CLI flags still take precedence.

Ember does not infer a sidecar from GGUF metadata. Without `--model-card`, the
server uses the DeepSeek defaults compiled into `src/model/model_card.c`.
Qwen3.8-Flash-Next deployments must pass
`--model-card share/model_cards/qwen3.8-flash-next.json`; its thinking and
non-thinking sampler recommendations differ, as recorded in the card notes.

## Fields

[`_schema.json`](_schema.json) is the authoritative authoring schema.

| Field | Required | Runtime use |
|---|---|---|
| `name` | yes | Provenance only. |
| `source` | yes | Provenance only. |
| `verified_at` | yes | Provenance only. |
| `max_tokens` | yes | Default combined reasoning and visible-output cap. |
| `hard_limit_reply_budget` | no | Visible-output reserve after `</think>`. |
| `thinking_terminator_hint` | no | Verbatim text injected when the reasoning budget expires. |
| `sampling` | no | Defaults for omitted sampler fields. |
| `reasoning_effort_tiers` | no | Explicit reasoning budgets for `low` through `max`. |
| `context_extension` | no | Advisory static-YaRN recipe and provenance. Loading the card never activates it; the architecture-specific CLI opt-in remains mandatory. |
| `notes` | no | Provenance and measured caveats. |

The C loader is intentionally permissive: missing or malformed runtime fields
fall back independently, and provenance fields are not parsed. Validate edited
cards against the schema before packaging them. For example, with any JSON
Schema draft-2020-12 validator:

```bash
npx --yes ajv-cli@5 validate \
  -s share/model_cards/_schema.json \
  -d share/model_cards/deepseek-v4-flash-src.json \
  --spec=draft2020
```

Validate the Qwen card with the same command after substituting
`share/model_cards/qwen3.8-flash-next.json` for the data path.
