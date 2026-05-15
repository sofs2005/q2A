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
