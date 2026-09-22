from __future__ import annotations

import json
import uuid
from typing import Any

from backend.services.token_calc import (
    completion_text_for_usage,
    count_tokens,
    effective_extra_prompt_tokens,
    resolve_usage,
)
from backend.toolcall.markup_scan import find_tool_markup_tag_outside_ignored


def _client_tool_name(name: str, tool_catalog=None) -> str:
    if tool_catalog is None:
        return name
    canonical = tool_catalog.get_canonical_name(name)
    if canonical is None:
        return name
    return tool_catalog.get_client_name(canonical)


def _extra_prompt_tokens_for(extra_prompt_tokens: int, upstream_usage: dict[str, Any] | None) -> int:
    return effective_extra_prompt_tokens(extra_prompt_tokens, upstream_usage=upstream_usage)


def build_canonical_openai_chat_payload(*, completion_id: str, created: int, model_name: str, prompt: str, answer_text: str, reasoning_text: str, directives: list[dict[str, Any]], tool_catalog=None, extra_prompt_tokens: int = 0, upstream_usage: dict[str, Any] | None = None) -> dict[str, Any]:
    del reasoning_text
    tool_blocks = [block for block in directives if block.get("type") == "tool_use"]
    if tool_blocks:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": _client_tool_name(str(block["name"]), tool_catalog),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
                for block in tool_blocks
            ],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": answer_text}
        finish_reason = "stop"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": resolve_usage(
            prompt,
            answer_text,
            message.get("tool_calls", []),
            extra_prompt_tokens=extra_prompt_tokens,
            upstream_usage=upstream_usage,
        ),
    }


def build_canonical_openai_responses_payload(*, response_id: str, created: int, model_name: str, prompt: str, answer_text: str, reasoning_text: str, directives: list[dict[str, Any]], tool_catalog=None, extra_prompt_tokens: int = 0, upstream_usage: dict[str, Any] | None = None) -> dict[str, Any]:
    tool_blocks = [block for block in directives if block.get("type") == "tool_use"]
    output: list[dict[str, Any]] = []
    if tool_blocks:
        if answer_text:
            output.append(
                {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": answer_text, "annotations": []}],
                }
            )
        output.extend(
            {
                "id": block["id"],
                "type": "function_call",
                "status": "completed",
                "call_id": block["id"],
                "name": _client_tool_name(str(block["name"]), tool_catalog),
                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
            }
            for block in tool_blocks
        )
    else:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": answer_text, "annotations": []}],
            }
        )
    resolved = resolve_usage(
        prompt,
        completion_text_for_usage(answer_text, tool_blocks),
        None,
        extra_prompt_tokens=extra_prompt_tokens,
        upstream_usage=upstream_usage,
    )
    input_tokens = resolved["prompt_tokens"]
    output_tokens = resolved["completion_tokens"]
    details = (upstream_usage or {}).get("_upstream_details") if isinstance(upstream_usage, dict) else None
    details = details if isinstance(details, dict) else {}
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": resolved["total_tokens"],
        "output_tokens_details": details.get("output_tokens_details")
        or {"reasoning_tokens": count_tokens(reasoning_text)},
    }
    if isinstance(details.get("prompt_tokens_details"), dict):
        usage["input_tokens_details"] = details["prompt_tokens_details"]
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model_name,
        "output": output,
        "output_text": answer_text,
        "usage": usage,
    }


def build_canonical_anthropic_message(*, msg_id: str, model_name: str, prompt: str, answer_text: str, reasoning_text: str, directives: list[dict[str, Any]], tool_catalog=None, extra_prompt_tokens: int = 0, upstream_usage: dict[str, Any] | None = None) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    if reasoning_text:
        content_blocks.append({"type": "thinking", "thinking": reasoning_text})
    if directives:
        for directive in directives:
            block = dict(directive)
            if block.get("type") == "tool_use" and block.get("name"):
                block["name"] = _client_tool_name(str(block["name"]), tool_catalog)
            content_blocks.append(block)
    elif answer_text:
        content_blocks.append({"type": "text", "text": answer_text})
    resolved = resolve_usage(
        prompt,
        answer_text,
        None,
        extra_prompt_tokens=extra_prompt_tokens,
        upstream_usage=upstream_usage,
    )
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content_blocks,
        "stop_reason": "tool_use" if any(block.get("type") == "tool_use" for block in directives) else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": resolved["prompt_tokens"], "output_tokens": resolved["completion_tokens"]},
    }


def _strip_dsml_markup(text: str) -> str:
    if not text or "<|DSML|" not in text:
        return text
    tag = find_tool_markup_tag_outside_ignored(text, 0)
    while tag is not None:
        if tag.name in {"tool_calls", "invoke"}:
            return text[:tag.start].rstrip()
        tag = find_tool_markup_tag_outside_ignored(text, tag.end)
    return text


def build_canonical_gemini_payload(*, answer_text: str, tool_calls: list[dict[str, Any]] | None = None, prompt: str = "", prompt_tokens: int | None = None, completion_tokens: int | None = None, total_tokens: int | None = None) -> dict[str, Any]:
    if tool_calls:
        parts = [
            {
                "functionCall": {
                    "name": call.get("name", ""),
                    "args": call.get("input", {}),
                }
            }
            for call in tool_calls
            if call.get("name")
        ]
        if not parts:
            parts = [{"text": _strip_dsml_markup(answer_text) or ""}]
    else:
        parts = [{"text": _strip_dsml_markup(answer_text) or ""}]
    payload: dict[str, Any] = {
        "candidates": [
            {
                "content": {
                    "parts": parts,
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ]
    }
    # usageMetadata 为本次新增：此前 gemini 出口完全没有 token 统计
    if prompt_tokens is not None or completion_tokens is not None:
        in_tokens = int(prompt_tokens or 0)
        out_tokens = int(completion_tokens or 0)
        payload["usageMetadata"] = {
            "promptTokenCount": in_tokens,
            "candidatesTokenCount": out_tokens,
            "totalTokenCount": int(total_tokens if total_tokens is not None else in_tokens + out_tokens),
        }
    return payload
