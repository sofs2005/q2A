from __future__ import annotations

import json
import re
from typing import Any

from backend.runtime.execution import build_tool_directive
from backend.toolcall.markup_scan import find_tool_markup_tag_outside_ignored
from backend.toolcore.formatter import (
    build_canonical_anthropic_message,
    build_canonical_gemini_payload,
    build_canonical_openai_chat_payload,
    build_canonical_openai_responses_payload,
)


def _first_dsml_tool_markup_index(text: str) -> int:
    tag = find_tool_markup_tag_outside_ignored(text, 0)
    while tag is not None:
        if tag.name in {"tool_calls", "invoke"}:
            return tag.start
        tag = find_tool_markup_tag_outside_ignored(text, tag.end)
    return -1


def sanitize_visible_answer_text(answer_text: str, *, tool_use: bool) -> str:
    text = answer_text or ""
    if not tool_use or not text:
        return text
    text = re.sub(r"(?im)^Tool\s+[A-Za-z0-9_.:-]+\s+does not exists?\.?\s*", "", text).strip()
    positions = [text.index(marker) for marker in ("##TOOL_CALL##", "<tool_call>") if marker in text]
    dsml_index = _first_dsml_tool_markup_index(text)
    if dsml_index >= 0:
        positions.append(dsml_index)
    if not positions:
        return text
    return text[:min(positions)].rstrip()


def _client_visible_tool_name(name: str, tool_catalog) -> str:
    if tool_catalog is None:
        return name
    canonical = tool_catalog.get_canonical_name(name)
    if canonical is None:
        return name
    return tool_catalog.get_client_name(canonical)


def _client_visible_tools(tools: list[dict[str, Any]], tool_catalog) -> list[dict[str, Any]]:
    visible_tools: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        visible = dict(tool)
        if isinstance(visible.get("function"), dict):
            function_payload = dict(visible["function"])
            name = str(function_payload.get("name") or "")
            if name:
                function_payload["name"] = _client_visible_tool_name(name, tool_catalog)
            visible["function"] = function_payload
        else:
            name = str(visible.get("name") or "")
            if name:
                visible["name"] = _client_visible_tool_name(name, tool_catalog)
        visible_tools.append(visible)
    return visible_tools


def build_openai_completion_payload(*, completion_id: str, created: int, model_name: str, prompt: str, execution, standard_request) -> dict[str, Any]:
    directive = build_tool_directive(standard_request, execution.state)
    visible_answer_text = sanitize_visible_answer_text(
        execution.state.answer_text, tool_use=directive.stop_reason == "tool_use"
    )
    payload = build_canonical_openai_chat_payload(
        completion_id=completion_id,
        created=created,
        model_name=model_name,
        prompt=prompt,
        answer_text=visible_answer_text,
        reasoning_text=execution.state.reasoning_text,
        directives=directive.tool_blocks,
        tool_catalog=standard_request.tool_catalog,
        extra_prompt_tokens=standard_request.context_attachment_tokens,
    )
    oai_tool_calls = payload["choices"][0]["message"].get("tool_calls", [])
    finish_reason = payload["choices"][0]["finish_reason"]
    import logging
    logging.getLogger("qwen2api.chat").info(
        "[OAI] response finish_reason=%s tool_calls=%s text_preview=%r",
        finish_reason,
        [
            {
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
            for call in oai_tool_calls
        ],
        execution.state.answer_text[:300],
    )
    return payload


def build_openai_response_payload(
    *,
    response_id: str,
    created: int,
    model_name: str,
    prompt: str,
    execution,
    standard_request,
    previous_response_id: str | None = None,
    store: bool = True,
) -> dict[str, Any]:
    directive = build_tool_directive(standard_request, execution.state)
    raw_answer_text = execution.state.answer_text or ""
    answer_text = sanitize_visible_answer_text(raw_answer_text, tool_use=directive.stop_reason == "tool_use")
    payload = build_canonical_openai_responses_payload(
        response_id=response_id,
        created=created,
        model_name=model_name,
        prompt=prompt,
        answer_text=answer_text,
        reasoning_text=execution.state.reasoning_text,
        directives=directive.tool_blocks,
        tool_catalog=standard_request.tool_catalog,
        extra_prompt_tokens=standard_request.context_attachment_tokens,
    )
    if standard_request.required_tool_name:
        name = _client_visible_tool_name(standard_request.required_tool_name, standard_request.tool_catalog)
        payload["tool_choice"] = {"type": "function", "function": {"name": name}}
    elif standard_request.tool_choice_raw is not None:
        payload["tool_choice"] = standard_request.tool_choice_raw
    else:
        payload["tool_choice"] = standard_request.tool_choice_mode or "auto"
    payload["tools"] = _client_visible_tools(standard_request.tools or [], standard_request.tool_catalog)
    payload["previous_response_id"] = previous_response_id
    payload["store"] = store
    payload["reasoning"] = {"effort": None, "summary": None}
    payload["parallel_tool_calls"] = False
    payload["error"] = None
    payload["incomplete_details"] = None
    payload["instructions"] = None
    payload["max_output_tokens"] = None
    payload["temperature"] = 1.0
    payload["text"] = {"format": {"type": "text"}}
    payload["top_p"] = 1.0
    payload["truncation"] = "disabled"
    payload["metadata"] = {}
    payload["user"] = None
    return payload


def build_anthropic_message_payload(*, msg_id: str, model_name: str, prompt: str, execution, standard_request) -> dict[str, Any]:
    directive = build_tool_directive(standard_request, execution.state)
    visible_answer_text = sanitize_visible_answer_text(
        execution.state.answer_text, tool_use=directive.stop_reason == "tool_use"
    )
    return build_canonical_anthropic_message(
        msg_id=msg_id,
        model_name=model_name,
        prompt=prompt,
        answer_text=visible_answer_text,
        reasoning_text=execution.state.reasoning_text,
        directives=directive.tool_blocks,
        tool_catalog=standard_request.tool_catalog,
        extra_prompt_tokens=standard_request.context_attachment_tokens,
    )


def build_gemini_generate_payload(*, execution) -> dict[str, Any]:
    return build_canonical_gemini_payload(answer_text=execution.state.answer_text)
