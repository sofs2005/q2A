# Qwen 3.7 Model Aliases Design

## Goal

Add two Qwen 3.7 model entries so downstream clients can request the desired public names while upstream requests use the raw Qwen model identifiers.

## Scope

This change only updates the default model alias mapping in `backend/core/config.py` and adds minimal tests for model resolution and fallback model listing.

## Model Mapping

- Downstream `qwen3.7-max` resolves upstream to `qwen3.7-max`.
- Downstream `qwen3.7-plus-preview` resolves upstream to `qwen-latest-series-invite-beta-v16`.

Both models have a maximum context window of 1m tokens as a product constraint. The current codebase has no model context-window metadata table or response field, so this design does not add unused metadata.

## Data Flow

1. Downstream clients send requests with a model name.
2. Existing request builders call `resolve_model` or `resolve_request_model`.
3. The new `MODEL_MAP` entries translate the downstream names to upstream names.
4. `/v1/models` fallback output automatically includes the two downstream model names because it iterates `MODEL_MAP` keys.

## Error Handling

No new error path is required. Unknown models continue to follow the current `resolve_model` and `/v1/models/{model_id}` behavior.

## Testing

Add focused tests to verify:

- `resolve_model("qwen3.7-max")` returns `qwen3.7-max`.
- `resolve_model("qwen3.7-plus-preview")` returns `qwen-latest-series-invite-beta-v16`.
- The fallback `/v1/models` payload includes both downstream model IDs.

## Implementation Principles

- KISS: Use the existing alias map instead of adding a new registry.
- YAGNI: Do not introduce context-window metadata until an API consumes it.
- DRY: Reuse existing model-list generation behavior.
