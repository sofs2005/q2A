from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.runtime.execution import RuntimeToolDirective
from backend.toolcall.markup_scan import find_tool_markup_tag_outside_ignored
from backend.toolcore.stream_state_machine import ToolStreamStateMachine


STRICT_TOOL_TEXT_PREFIXES = ("{", "[", "`", "<")
BUFFERED_TOOL_CALLS_ONLY = "buffered_tool_calls_only"
DIRECTIVE_DRIVEN_TOOL_CALLS = "directive_driven_tool_calls"
TOOL_ARGUMENT_CHUNK_SIZE = 128
TEXTUAL_TOOL_MARKERS = ("##TOOL_CALL##", "<tool_call>")
SAFE_TEXT_HOLD_MARKERS = ("<|DSML|", "</|DSML|", "<![CDATA[")
TOOL_MARKUP_SCAN_HINTS = (
    "<|dsml|",
    "</|dsml|",
    "<![cdata[",
    "＜！dsml！",
    "＜/！dsml！",
    "〈！dsml！",
    "〈/！dsml！",
    "<tool_calls",
    "</tool_calls",
    "<invoke",
    "</invoke",
)
DSML_CONTROL_TAG_RE = re.compile(
    r"[<＜﹤〈]\s*/?\s*[|！、␂]\s*/?\s*DSML\s*[|！、␂][^>＞﹥〉]*(?:[>＞﹥〉]|$)",
    re.IGNORECASE | re.DOTALL,
)
CDATA_MARKER_RE = re.compile(
    r"[<＜﹤〈]\s*[!！]\s*\[\s*CDATA\s*\[|\]\s*\]\s*[>＞﹥〉]",
    re.IGNORECASE,
)
DSML_CONTROL_START_RE = re.compile(
    r"[<＜﹤〈]\s*/?\s*[|！、␂]\s*/?\s*DSML\s*[|！、␂]",
    re.IGNORECASE,
)


def strip_dsml_control_markup(text: str) -> str:
    if not text:
        return ""
    dsml_start = DSML_CONTROL_START_RE.search(text)
    if dsml_start is not None:
        text = text[:dsml_start.start()]
    return CDATA_MARKER_RE.sub("", DSML_CONTROL_TAG_RE.sub("", text))


def _first_tool_markup_index(text: str) -> int:
    positions = [text.index(marker) for marker in TEXTUAL_TOOL_MARKERS if marker in text]
    lowered = text.lower()
    for marker in SAFE_TEXT_HOLD_MARKERS:
        pos = lowered.find(marker.lower())
        if pos >= 0:
            positions.append(pos)
    if any(hint in lowered for hint in TOOL_MARKUP_SCAN_HINTS):
        tag = find_tool_markup_tag_outside_ignored(text, 0)
        while tag is not None:
            if tag.name in {"tool_calls", "invoke"}:
                positions.append(tag.start)
                break
            tag = find_tool_markup_tag_outside_ignored(text, tag.end)
    return min(positions) if positions else -1


def _split_safe_text_tail(text: str) -> tuple[str, str]:
    lowered = text.lower()
    longest_hold = ""
    for marker in SAFE_TEXT_HOLD_MARKERS:
        marker_lower = marker.lower()
        max_prefix_len = min(len(marker) - 1, len(text))
        for prefix_len in range(1, max_prefix_len + 1):
            if lowered.endswith(marker_lower[:prefix_len]) and prefix_len > len(longest_hold):
                longest_hold = text[-prefix_len:]
    if not longest_hold:
        return text, ""
    return text[:-len(longest_hold)], longest_hold


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
        self.safe_text_hold = ""
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
        text_chunk = strip_dsml_control_markup(text_chunk)
        if not text_chunk:
            return
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': text_chunk}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)

    def on_delta(self, evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        self._ensure_role_chunk()

        if text_chunk and evt.get("phase") in ("think", "thinking_summary"):
            return

        if text_chunk and evt.get("phase") == "answer":
            if evt.get("_qwen2api_safe_text"):
                safe_candidate = f"{self.safe_text_hold}{text_chunk}"
                self.safe_text_hold = ""
                markup_index = _first_tool_markup_index(safe_candidate)
                if markup_index >= 0:
                    safe_text = safe_candidate[:markup_index].rstrip()
                else:
                    safe_text, self.safe_text_hold = _split_safe_text_tail(safe_candidate)
                    if self.safe_text_hold:
                        safe_text = safe_text.rstrip()
                if safe_text:
                    self.answer_fragments.append(safe_text)
                    self._emit_content_chunk(safe_text)
                return
            self.answer_fragments.append(text_chunk)
            for event in self.state_machine.process_text_delta(text_chunk):
                if event.type == "content" and event.text:
                    self._emit_content_chunk(event.text)
                elif event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)
            return

        if tool_calls:
            for event in self.state_machine.flush(final_tool_use=True):
                if event.type == "content" and event.text:
                    self._emit_content_chunk(event.text)
                elif event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)
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
    def _openai_tool_call_id(call_id: Any) -> str:
        text = str(call_id or "").strip()
        if text.startswith("call_"):
            return text
        return f"call_{uuid.uuid4().hex}"

    def emit_tool_calls(self, tool_calls: list[dict[str, Any]], *, split_arguments: bool = False) -> None:
        self._ensure_role_chunk()
        for tool_call in tool_calls:
            idx = self.emitted_tool_index
            self.emitted_tool_index += 1
            tool_name = self._client_tool_name(str(tool_call['name']))
            arguments = json.dumps(tool_call['input'], ensure_ascii=False)
            if split_arguments:
                self.pending_chunks.append(
                    f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'id': self._openai_tool_call_id(tool_call.get('id')), 'type': 'function', 'function': {'name': tool_name}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                )
                for start in range(0, len(arguments), TOOL_ARGUMENT_CHUNK_SIZE):
                    self.pending_chunks.append(
                        f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'function': {'arguments': arguments[start:start + TOOL_ARGUMENT_CHUNK_SIZE]}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                    )
            else:
                self.pending_chunks.append(
                    f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'id': self._openai_tool_call_id(tool_call.get('id')), 'type': 'function', 'function': {'name': tool_name, 'arguments': arguments}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
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
                    self.emit_tool_calls(tool_calls, split_arguments=False)
                    final_finish_reason = "tool_calls"
        else:
            for event in self.state_machine.flush(final_tool_use=finish_reason == "tool_calls"):
                if event.type == "content" and event.text:
                    self._emit_content_chunk(event.text)
                elif event.type == "tool_calls" and event.calls:
                    self.emit_tool_calls(event.calls)

        if self.tool_calls_emitted and final_finish_reason in (None, "stop"):
            final_finish_reason = "tool_calls"

        chunks = list(self.pending_chunks)
        finish_payload = {
            'id': self.completion_id,
            'object': 'chat.completion.chunk',
            'created': self.created,
            'model': self.model_name,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': final_finish_reason}],
        }
        chunks.append(f"data: {json.dumps(finish_payload, ensure_ascii=False)}\n\n")
        if usage is not None:
            usage_payload = {
                'id': self.completion_id,
                'object': 'chat.completion.chunk',
                'created': self.created,
                'model': self.model_name,
                'choices': [],
                'usage': usage,
            }
            chunks.append(f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return chunks
