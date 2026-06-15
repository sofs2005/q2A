from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.toolcall.formats_json import load_json_with_repair
from backend.toolcall.normalize import normalize_arguments
from backend.toolcall.parser import parse_tool_calls_detailed
from backend.toolcore.types import CanonicalToolCall


@dataclass(slots=True)
class ToolDirectiveParseResult:
    canonical_calls: list[CanonicalToolCall] = field(default_factory=list)
    tool_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"


def parse_state_tool_calls(state_tool_calls: list[dict[str, Any]], allowed_tool_names: list[str]) -> ToolDirectiveParseResult:
    canonical_calls: list[CanonicalToolCall] = []
    tool_blocks: list[dict[str, Any]] = []
    allowed_name_set = {str(name).strip() for name in allowed_tool_names if str(name).strip()}
    for tool_call in state_tool_calls:
        name = str(tool_call.get("name", "")).strip()
        if allowed_name_set and name not in allowed_name_set:
            continue
        if not name:
            continue
        call_id = str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
        if not call_id:
            continue
        input_data = tool_call.get("input", {}) if isinstance(tool_call.get("input", {}), dict) else {}
        canonical = CanonicalToolCall(call_id=call_id, name=name, input=input_data)
        canonical_calls.append(canonical)
        tool_blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": input_data})
    stop_reason = "tool_use" if canonical_calls else "end_turn"
    return ToolDirectiveParseResult(canonical_calls=canonical_calls, tool_blocks=tool_blocks, stop_reason=stop_reason)


def _resolve_tool_name(raw_name: str, tool_names: set[str]) -> str | None:
    """Case-insensitive tool name resolution.

    Models sometimes emit tool names with different casing (e.g. 'read' instead of 'Read').
    This helper matches against the allowed set case-insensitively and returns the canonical name.
    """
    if raw_name in tool_names:
        return raw_name
    lowered = raw_name.lower()
    for name in tool_names:
        if name.lower() == lowered:
            return name
    return None


def parse_textual_tool_calls(answer_text: str, tools: list[dict[str, Any]]) -> ToolDirectiveParseResult:
    if not tools or not answer_text:
        return ToolDirectiveParseResult(tool_blocks=[{"type": "text", "text": answer_text}], stop_reason="end_turn")
    tool_names = {str(tool.get("name", "")).strip() for tool in tools if isinstance(tool, dict) and str(tool.get("name", "")).strip()}
    tool_blocks: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    tc_matches = list(re.finditer(r'##TOOL_CALL##\s*(.*?)\s*##END_CALL##', answer_text, re.DOTALL | re.IGNORECASE))
    if tc_matches:
        cursor = 0
        for match in tc_matches:
            prefix = answer_text[cursor:match.start()].strip()
            if prefix:
                tool_blocks.append({"type": "text", "text": prefix})
            try:
                obj = load_json_with_repair(match.group(1))
                if isinstance(obj, dict) and obj.get("name"):
                    raw_name = str(obj.get("name", ""))
                    raw_input = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
                    if raw_name in tool_names:
                        tool_blocks.append(
                            {
                                "type": "tool_use",
                                "id": f"toolu_{len(tool_blocks)}",
                                "name": raw_name,
                                "input": normalize_arguments(raw_input),
                            }
                        )
                        stop_reason = "tool_use"
                    else:
                        tool_blocks.append({"type": "text", "text": answer_text[match.start():match.end()].strip()})
                else:
                    tool_blocks.append({"type": "text", "text": answer_text[match.start():match.end()].strip()})
            except Exception:
                tool_blocks.append({"type": "text", "text": answer_text[match.start():match.end()].strip()})
            cursor = match.end()
        suffix = answer_text[cursor:].strip()
        if suffix:
            tool_blocks.append({"type": "text", "text": suffix})
    else:
        detailed = parse_tool_calls_detailed(answer_text, tool_names)
        detailed_calls = detailed.get("calls", []) if isinstance(detailed, dict) else []
        if isinstance(detailed_calls, list) and detailed_calls:
            for index, call in enumerate(detailed_calls):
                if not isinstance(call, dict):
                    continue
                tool_blocks.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{index}",
                        "name": str(call.get("name", "")).strip(),
                        "input": call.get("input", {}) if isinstance(call.get("input", {}), dict) else {},
                    }
                )
            stop_reason = "tool_use"
        else:
            tool_blocks = [{"type": "text", "text": answer_text}]

    canonical_calls: list[CanonicalToolCall] = []
    for block in tool_blocks:
        if block.get("type") != "tool_use" or not block.get("id") or not block.get("name"):
            continue
        input_data = block.get("input", {})
        if not isinstance(input_data, dict):
            input_data = {}
        canonical_calls.append(
            CanonicalToolCall(
                call_id=str(block.get("id") or "").strip(),
                name=str(block.get("name") or "").strip(),
                input=input_data,
            )
        )
    return ToolDirectiveParseResult(canonical_calls=canonical_calls, tool_blocks=tool_blocks, stop_reason=stop_reason)
