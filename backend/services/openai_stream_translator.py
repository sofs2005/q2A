from __future__ import annotations

import json
from typing import Any, Callable

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.runtime.execution import RuntimeToolDirective
from backend.toolcore.stream_state_machine import ToolStreamStateMachine


STRICT_TOOL_TEXT_PREFIXES = ("{", "[", "`", "<")
BUFFERED_TOOL_CALLS_ONLY = "buffered_tool_calls_only"
DIRECTIVE_DRIVEN_TOOL_CALLS = "directive_driven_tool_calls"
TOOL_ARGUMENT_CHUNK_SIZE = 128
class OpenAIStreamTranslator:
    def __init__(
        self,
        *,
        completion_id: str,
        created: int,
        model_name: str,
        client_profile: str,
        build_final_directive: Callable[[str], RuntimeToolDirective] | None = None,
        allowed_tool_names: list[str] | None = None,
        toolcore_enabled: bool = True,
        tool_catalog=None,
    ):
        self.completion_id = completion_id
        self.created = created
        self.model_name = model_name
        self.client_profile = client_profile
        self.build_final_directive = build_final_directive
        self.allowed_tool_names = {name for name in (allowed_tool_names or []) if isinstance(name, str) and name}
        self.toolcore_enabled = toolcore_enabled
        self.tool_catalog = tool_catalog
        self.pending_chunks: list[str] = []
        self.role_chunk_sent = False
        self.emitted_tool_index = 0
        self.answer_fragments: list[str] = []
        self.pending_content_chunks: list[str] = []
        self.tool_calls_emitted = False
        self.tool_text_detection_mode = self._resolve_tool_text_detection_mode(client_profile)
        self.tool_call_finalize_mode = self._resolve_tool_call_finalize_mode(client_profile)
        self.state_machine = ToolStreamStateMachine(list(self.allowed_tool_names))

    @staticmethod
    def _resolve_tool_text_detection_mode(client_profile: str) -> str:
        if client_profile == OPENCLAW_OPENAI_PROFILE:
            return "strict_prefix"
        return "accept_any_tool_syntax"

    @staticmethod
    def _resolve_tool_call_finalize_mode(client_profile: str) -> str:
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
            return BUFFERED_TOOL_CALLS_ONLY
        return DIRECTIVE_DRIVEN_TOOL_CALLS

    def _should_finalize_tool_calls(self, directive: RuntimeToolDirective) -> bool:
        return directive.stop_reason == "tool_use"

    def _ensure_role_chunk(self) -> None:
        if self.role_chunk_sent:
            return
        yield_payload = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        self.pending_chunks.append(f"data: {json.dumps(yield_payload, ensure_ascii=False)}\n\n")
        self.role_chunk_sent = True

    def _emit_content_chunk(self, text_chunk: str) -> None:
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': text_chunk}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)
        self.pending_content_chunks.append(chunk)

    def _discard_pending_content_chunks(self) -> None:
        if not self.pending_content_chunks:
            return
        pending_content_ids = {id(chunk) for chunk in self.pending_content_chunks}
        self.pending_chunks = [chunk for chunk in self.pending_chunks if id(chunk) not in pending_content_ids]
        self.pending_content_chunks = []

    def on_delta(self, evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        self._ensure_role_chunk()

        if text_chunk and evt.get("phase") in ("think", "thinking_summary"):
            return

        if text_chunk and evt.get("phase") == "answer":
            self.answer_fragments.append(text_chunk)
            for event in self.state_machine.process_text_delta(text_chunk):
                if event.type == "content" and event.text:
                    self._emit_content_chunk(event.text)
                elif event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)
            return

        if tool_calls:
            for event in self.state_machine.process_tool_calls(tool_calls):
                if event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)

    def _client_tool_name(self, name: str) -> str:
        if self.tool_catalog is None:
            return name
        canonical = self.tool_catalog.get_canonical_name(name)
        if canonical is None:
            return name
        return self.tool_catalog.get_client_name(canonical)

    @staticmethod
    def _openai_tool_call_id(call_id: str) -> str:
        if call_id.startswith("call_"):
            return call_id
        return f"call_{call_id}"

    def emit_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        self._ensure_role_chunk()
        if tool_calls and not self.tool_calls_emitted:
            self._discard_pending_content_chunks()
        for tool_call in tool_calls:
            idx = self.emitted_tool_index
            self.emitted_tool_index += 1
            tool_name = self._client_tool_name(str(tool_call['name']))
            tool_call_id = self._openai_tool_call_id(str(tool_call["id"]))
            self.pending_chunks.append(
                f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'id': tool_call_id, 'type': 'function', 'function': {'name': tool_name, 'arguments': ''}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
            )
            arguments = json.dumps(tool_call['input'], ensure_ascii=False)
            for start in range(0, len(arguments), TOOL_ARGUMENT_CHUNK_SIZE):
                self.pending_chunks.append(
                    f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'function': {'arguments': arguments[start:start + TOOL_ARGUMENT_CHUNK_SIZE]}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                )
        if tool_calls:
            self.tool_calls_emitted = True

    def finalize(self, finish_reason: str, *, usage: dict[str, int] | None = None) -> list[str]:
        final_finish_reason = finish_reason
        if self.build_final_directive is not None and not self.tool_calls_emitted:
            directive = self.build_final_directive("".join(self.answer_fragments))
            for event in self.state_machine.flush(final_tool_use=directive.stop_reason == "tool_use"):
                if event.type == "content" and event.text:
                    self._emit_content_chunk(event.text)
                elif event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)
            if self._should_finalize_tool_calls(directive):
                self._discard_pending_content_chunks()
                tool_calls = [
                    {
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    }
                    for block in directive.tool_blocks
                    if block.get("type") == "tool_use"
                ]
                if tool_calls:
                    self.emit_tool_calls(tool_calls)
                    final_finish_reason = "tool_calls"
        else:
            for event in self.state_machine.flush(final_tool_use=finish_reason == "tool_calls"):
                if event.type == "content" and event.text:
                    self._emit_content_chunk(event.text)
                elif event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)

        chunks = list(self.pending_chunks)
        chunks.append(
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': final_finish_reason}]}, ensure_ascii=False)}\n\n"
        )
        if usage is not None:
            chunks.append(
                f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [], 'usage': usage}, ensure_ascii=False)}\n\n"
            )
        chunks.append("data: [DONE]\n\n")
        return chunks
