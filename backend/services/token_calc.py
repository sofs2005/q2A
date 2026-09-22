import json
import logging
from typing import Any

import tiktoken

log = logging.getLogger("qwen2api.token")

try:
    # 默认使用 cl100k_base，因为目前这是最通用的 GPT-4 级分词器
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    log.warning(f"Failed to load tiktoken: {e}")
    encoder = None

def count_tokens(text: str) -> int:
    """计算文本的精确 Token 数"""
    if not text:
        return 0
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Fallback：每汉字 1 token，每 3 个英文字母 1 token 的粗略估算
    return max(1, len(text.encode('utf-8')) // 2)

def serialize_tool_calls_for_usage(tool_calls: list[dict[str, Any]] | None) -> str:
    if not tool_calls:
        return ""
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if call.get("type") == "tool_use":
            normalized.append({
                "id": call.get("id"),
                "type": "tool_use",
                "name": call.get("name"),
                "input": call.get("input", {}),
            })
            continue
        normalized.append(call)
    if not normalized:
        return ""
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def completion_text_for_usage(completion: str, tool_calls: list[dict[str, Any]] | None = None) -> str:
    parts = [completion] if completion else []
    tool_text = serialize_tool_calls_for_usage(tool_calls)
    if tool_text:
        parts.append(tool_text)
    return "\n".join(parts)


def calculate_usage(prompt: str, completion: str, tool_calls: list[dict[str, Any]] | None = None, *, extra_prompt_tokens: int = 0) -> dict:
    """结算：精确扣费"""
    prompt_tokens = count_tokens(prompt) + max(0, int(extra_prompt_tokens or 0))
    completion_tokens = count_tokens(completion_text_for_usage(completion, tool_calls))
    total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }


def _coerce_nonneg(value: Any) -> int | None:
    """宽松取非负整数；缺失/非法（None、bool、负数、非数字）返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def merge_upstream_usage(local: dict[str, int], upstream: dict[str, Any] | None) -> dict:
    """用上游累计 usage 覆盖本地估算；上游缺项时逐项回落本地值。

    上游完全不可用时原样返回 local（保证与未接入前逐键相等）。
    """
    if not isinstance(upstream, dict):
        return local
    prompt_tokens = _coerce_nonneg(upstream.get("prompt_tokens"))
    completion_tokens = _coerce_nonneg(upstream.get("completion_tokens"))
    if prompt_tokens is None and completion_tokens is None:
        return local
    if prompt_tokens is None:
        prompt_tokens = local["prompt_tokens"]
    if completion_tokens is None:
        completion_tokens = local["completion_tokens"]
    total_tokens = _coerce_nonneg(upstream.get("total_tokens"))
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    merged = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    details = upstream.get("_upstream_details")
    if isinstance(details, dict) and details:
        merged["_upstream_details"] = details
    return merged


def effective_extra_prompt_tokens(extra_prompt_tokens: int, *, upstream_usage: Any = None) -> int:
    """上游 usage 已含上下文附件 token 时不再叠加，避免重复计费。

    默认（QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS=false）保守叠加；
    实测确认上游 input_tokens 已含附件后再打开开关。
    """
    from backend.core.config import settings

    if isinstance(upstream_usage, dict) and getattr(
        settings, "QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS", False
    ):
        return 0
    return max(0, int(extra_prompt_tokens or 0))


def resolve_usage(
    prompt: str,
    completion: str,
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    extra_prompt_tokens: int = 0,
    upstream_usage: dict[str, Any] | None = None,
) -> dict:
    """单点决议：优先上游真实 usage，缺失/非法时回落到本地 tiktoken 估算。

    extra_prompt_tokens 默认仍叠加；当上游 input_tokens 已包含上下文附件时，
    调用方应传 0（见 QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS）。
    """
    local = calculate_usage(
        prompt,
        completion,
        tool_calls,
        extra_prompt_tokens=effective_extra_prompt_tokens(
            extra_prompt_tokens, upstream_usage=upstream_usage
        ),
    )
    return merge_upstream_usage(local, upstream_usage)
