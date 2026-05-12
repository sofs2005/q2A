# DSML/XML Tool Use Alignment Design

## Context

This project was originally adapted from `D:/Programs/ds2api`, but its tool-use path still treats `##TOOL_CALL##...##END_CALL##` as the primary model-facing protocol. The current `ds2api` implementation has moved to a DSML/XML-first tool protocol with stronger parsing and streaming safeguards.

The goal is to align this project with the latest `ds2api` tool-use behavior while preserving legacy compatibility for existing sessions and model outputs.

## Goals

- Make DSML/XML the primary tool-call protocol shown to models.
- Parse and stream-intercept `ds2api`-style DSML/XML tool calls in Python without introducing a JS or Go runtime dependency.
- Preserve existing API output semantics: parsed calls still become canonical `tool_use` blocks.
- Keep legacy `##TOOL_CALL##`, `<tool_call>{...}</tool_call>`, plain JSON, and text-kv parsing as fallbacks.
- Prevent XML/DSML tool markup from leaking into streamed text output.

## Non-goals

- Replacing OpenAI, Anthropic, or Responses response formatter semantics.
- Introducing cross-language subprocess calls to reuse `ds2api` JavaScript helpers directly.
- Refactoring unrelated runtime, account, quota, or attachment logic.

## Primary Protocol

Tool instructions should teach this DSML/XML form as the preferred format:

```xml
<|DSML|tool_calls>
  <|DSML|invoke name="TOOL_NAME">
    <|DSML|parameter name="ARG_NAME"><![CDATA[VALUE]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>
```

String values should prefer CDATA. Object values may use nested XML elements. Array values may repeat `<item>` children. Numbers, booleans, and null may remain plain text.

## Module Boundaries

### `backend/toolcore/prompt_contract.py`

- Generate DSML/XML-first tool instructions.
- Render history tool calls as DSML/XML so current context and new instructions agree.
- Use real tool names and schema-derived parameter hints in examples where possible.
- Keep legacy `##TOOL_CALL##` out of the primary instruction block.

### `backend/toolcall/`

- Own DSML/XML scanning, normalization, and parsing.
- Add a shared scanner capable of identifying tool markup outside ignored regions.
- Parse DSML/XML into `{"name": str, "input": dict}` calls.
- Keep `parse_tool_calls_detailed()` as the unified entry point.
- Prefer parse order: DSML/XML, legacy XML tool-call JSON, legacy hash wrapper, plain JSON, text-kv.

### `backend/toolcore/stream_sieve.py`

- Replace marker-only detection with an XML/DSML-aware streaming sieve.
- Hold partial tool markup until the wrapper is complete or until final flush proves it incomplete.
- Emit parsed tool calls as `{"type": "tool_calls", "calls": [...]}`.
- Emit non-tool text as `{"type": "content", "text": ...}`.
- Continue supporting legacy hash-wrapper detection as fallback.

### `backend/services/tool_parser.py`

- Continue handling tool-name case normalization, schema coercion, logging, and canonical `tool_use` block construction.
- Avoid adding low-level DSML scanning here to keep protocol parsing centralized.
- Reuse `parse_tool_calls_detailed()` for final text parsing.

## DSML/XML Compatibility Scope

The Python parser should support the `ds2api` compatibility surface that affects tool-call reliability:

- Canonical tags: `<|DSML|tool_calls>`, `<|DSML|invoke>`, `<|DSML|parameter>`.
- Legacy XML tags: `<tool_calls>`, `<invoke>`, `<parameter>`.
- DSML variants: `dsml-`, `dsml_`, repeated DSML-like prefixes, camel-style prefixes, and arbitrary prefixes immediately attached to `tool_calls`, `invoke`, or `parameter`.
- Fullwidth and CJK drift: angle brackets, slashes, equals signs, quotes, bang separators, ideographic comma separators, and similar punctuation seen in `ds2api` tests.
- Extra leading `<` characters before DSML tags.
- Missing opening wrapper repair when an `<invoke>` block appears before a closing `</tool_calls>` wrapper.
- CDATA restoration and XML entity unescaping.
- Nested parameters for objects and repeated `<item>` elements for arrays.

## Ignored Regions

Tool-markup detection must ignore:

- Markdown fenced code blocks.
- Markdown inline code spans.
- XML CDATA contents.
- XML comments and processing instructions.

This prevents examples, quoted code, and parameter bodies from being interpreted as live tool calls.

## Data Flow

1. Request normalization converts client tool schemas to normalized prompt tools.
2. Prompt construction emits DSML/XML-first instructions and DSML/XML history tool calls.
3. Model output is handled in one of two paths:
   - Non-streaming: final text is parsed by `parse_tool_calls_detailed()`.
   - Streaming: `ToolStreamSieve` incrementally captures and parses XML/DSML tool markup.
4. Parsed calls become canonical internal calls with `name` and `input`.
5. `tool_parser` converts calls to existing API-specific `tool_use` / function-call output semantics.

## Error Handling

- Incomplete tool wrapper during streaming: hold content until more chunks arrive.
- Incomplete tool wrapper at final flush: emit captured content as ordinary text.
- Unknown tool name: do not execute; fall back to text and log a warning through existing parser behavior.
- Invalid parameter structure: recover simple values where safe; otherwise fall back to text instead of returning a server error.
- Tool call already emitted during streaming: finalization must not emit the same call again.

## Testing Plan

### Prompt contract tests

Update `tests/test_toolcore_prompt_contract.py` to verify:

- Primary instructions contain `<|DSML|tool_calls>`.
- History tool calls render as DSML/XML.
- Primary instructions no longer teach `##TOOL_CALL##` as the model-facing format.
- Required and none tool-choice constraints still appear correctly.

### Parser tests

Add `tests/test_toolcall_dsml_parser.py` covering:

- Standard DSML tool calls.
- Legacy `<tool_calls>/<invoke>/<parameter>` calls.
- Hyphen, underscore, camel, repeated-prefix, and arbitrary-prefix DSML variants.
- Fullwidth and CJK punctuation drift.
- Missing opening wrapper repair.
- CDATA, entity unescaping, nested objects, repeated arrays, number, boolean, null, and empty strings.
- Ignoring markdown code fences, inline code spans, comments, and CDATA internals.
- Legacy hash-wrapper and JSON fallback still work.

### Streaming tests

Extend `tests/test_toolcore_stream_sieve.py` to verify:

- Chunk-split DSML calls are held until complete.
- Parsed DSML calls do not leak markup into text.
- Drifted DSML variants are intercepted without leakage.
- Fenced and inline-code examples remain text.
- Incomplete capture flushes as text.
- Legacy `##TOOL_CALL##` streaming fallback still works.

### Integration tests

Run targeted tests for:

- Tool parser repair.
- Toolcore prompt contract.
- Toolcore stream sieve.
- Anthropic toolcore integration.
- OpenAI/Responses roundtrip compatibility.

## Principles Applied

- KISS: implement in Python and avoid cross-language runtime coupling.
- DRY: centralize protocol scanning and parsing under `backend/toolcall/`.
- YAGNI: only port ds2api behavior directly related to tool-call stability.
- SOLID: keep prompt construction, protocol parsing, streaming interception, and API output formatting separate.
