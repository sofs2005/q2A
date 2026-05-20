from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from backend.adapter.standard_request import normalize_tool_choice
from backend.services.client_profiles import (
    CLAUDE_CODE_OPENAI_PROFILE,
    OPENCLAW_OPENAI_PROFILE,
    QWEN_CODE_OPENAI_PROFILE,
    extract_system_prompt,
    sanitize_runtime_prompt_text,
    looks_like_opencode_system_prompt,
    sanitize_openclaw_user_text,
    user_role_system_text,
)
from backend.toolcore.prompt_contract import (
    build_tool_instruction_block,
    normalize_prompt_tools,
    render_history_tool_call,
)

log = logging.getLogger("qwen2api.prompt")

@dataclass(slots=True)
class PromptBuildResult:
    prompt: str
    tools: list[dict]
    tool_enabled: bool
    client_profile: str


def _sanitize_prompt_content(content, *, role: str, client_profile: str):
    if isinstance(content, str):
        if role == "system":
            return sanitize_runtime_prompt_text(content, role)
        return sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE and role == "user" else content
    if isinstance(content, list):
        sanitized_parts = []
        for part in content:
            if not isinstance(part, dict):
                sanitized_parts.append(part)
                continue
            if part.get("type") != "text":
                sanitized_parts.append(part)
                continue
            if role == "system":
                sanitized_text = sanitize_runtime_prompt_text(part.get("text", ""), role)
            else:
                sanitized_text = sanitize_openclaw_user_text(part.get("text", "")) if client_profile == OPENCLAW_OPENAI_PROFILE and role == "user" else part.get("text", "")
            sanitized_part = dict(part)
            sanitized_part["text"] = sanitized_text
            sanitized_parts.append(sanitized_part)
        return sanitized_parts
    return content


def _sanitize_openclaw_user_text(text: str) -> str:
    return sanitize_openclaw_user_text(text)


def _strip_user_role_system_text(text: str) -> str:
    cleaned = str(text or "").strip()
    system_text = user_role_system_text(cleaned)
    if system_text and cleaned.startswith(system_text):
        return cleaned[len(system_text):].strip()
    return cleaned


def _extract_user_text_only(content, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        text = _strip_user_role_system_text(content)
        return _sanitize_openclaw_user_text(text) if client_profile == OPENCLAW_OPENAI_PROFILE else text
    if isinstance(content, list):
        text_blocks = []
        for part in content:
            if not isinstance(part, dict) or part.get("type", "") != "text":
                continue
            block_text = _strip_user_role_system_text(part.get("text", ""))
            if client_profile == OPENCLAW_OPENAI_PROFILE:
                block_text = _sanitize_openclaw_user_text(block_text)
            if block_text:
                text_blocks.append(block_text)
        return "\n".join(text_blocks)
    return ""


def _extract_system_text_only(content) -> str:
    if isinstance(content, str):
        return sanitize_runtime_prompt_text(content, "system")
    if isinstance(content, list):
        text_blocks = []
        for part in content:
            if not isinstance(part, dict) or part.get("type", "") != "text":
                continue
            block_text = sanitize_runtime_prompt_text(part.get("text", ""), "system")
            if block_text:
                text_blocks.append(block_text)
        return "\n".join(text_blocks)
    return ""


def _render_history_tool_call(name: str, input_data: dict, client_profile: str) -> str:
    return render_history_tool_call(name, input_data, client_profile)


def _extract_text(content, user_tool_mode: bool = False, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        return _sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE else content
    if isinstance(content, list):
        parts = []
        text_blocks = []
        other_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                block_text = part.get("text", "")
                if client_profile == OPENCLAW_OPENAI_PROFILE:
                    block_text = _sanitize_openclaw_user_text(block_text)
                if block_text:
                    text_blocks.append(block_text)
            elif part_type == "tool_use":
                other_parts.append(_render_history_tool_call(part.get("name", ""), part.get("input", {}), client_profile))
            elif part_type == "tool_result":
                inner = part.get("content", "")
                tool_use_id = part.get("tool_use_id", "")
                if isinstance(inner, str):
                    other_parts.append(f"[Tool Result for call {tool_use_id}]\n{inner}\n[/Tool Result]")
                elif isinstance(inner, list):
                    texts = [p.get("text", "") for p in inner if isinstance(p, dict) and p.get("type") == "text"]
                    other_parts.append(f"[Tool Result for call {tool_use_id}]\n{''.join(texts)}\n[/Tool Result]")
            elif part_type == "input_file":
                other_parts.append(f"[Attachment file_id={part.get('file_id', '')} filename={part.get('filename', '')}]")
            elif part_type == "input_image":
                other_parts.append(f"[Attachment image file_id={part.get('file_id', '')} mime={part.get('mime_type', '')}]")

        if user_tool_mode and text_blocks:
            parts.append(text_blocks[-1])
        else:
            parts.extend(text_blocks)
        parts.extend(other_parts)
        return "\n".join(part for part in parts if part)
    return ""


def _safe_preview(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:limit] + ("...[truncated]" if len(compact) > limit else "")


def _is_heavy_tool_profile(client_profile: str) -> bool:
    return client_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}


def _has_tool_continuation_after_latest_user(messages: list, client_profile: str) -> bool:
    latest_user_index = -1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if _extract_user_text_only(message.get("content", ""), client_profile=client_profile).strip():
            latest_user_index = index
            break
    if latest_user_index < 0:
        return False

    for message in messages[latest_user_index + 1:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "") or "").lower()
        if role in {"tool", "function"}:
            return True
        if role == "assistant" and isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content):
            return True
    return False


def build_prompt_with_tools(
    system_prompt: str,
    messages: list,
    tools: list,
    *,
    client_profile: str = OPENCLAW_OPENAI_PROFILE,
    tool_choice_mode: str = "auto",
    required_tool_name: str | None = None,
    tool_catalog=None,
) -> str:
    if tools and client_profile == QWEN_CODE_OPENAI_PROFILE:
        max_chars = 24000
    elif tools and client_profile == CLAUDE_CODE_OPENAI_PROFILE:
        max_chars = 18000
    else:
        max_chars = 120000
    sys_part = f"<system>\n{system_prompt}\n</system>" if system_prompt else ""
    tools_part = ""
    if tools:
        tools_part = build_tool_instruction_block(
            tools,
            client_profile,
            tool_choice_mode=tool_choice_mode,
            required_tool_name=required_tool_name,
        )
    opencode_override = bool(tools and client_profile == OPENCLAW_OPENAI_PROFILE and looks_like_opencode_system_prompt(system_prompt))
    if opencode_override and tools_part:
        tools_part = "\n".join(
            [
                "=== OPENCODE TOOL FORMAT OVERRIDE ===",
                "The opencode system prompt may describe native or built-in tool syntax.",
                "IGNORE those native tool formats for this gateway.",
                "For this qwen2API bridge, EVERY tool call MUST use ONLY the DSML/XML tool_calls block below.",
                "If you need to inspect files or directories, call the tool immediately using that DSML/XML block.",
                "Never output plain-text plans such as 'I will inspect the directory first' before the tool call.",
                "=== END OVERRIDE ===",
                tools_part,
            ]
        )

    overhead = len(sys_part) + len(tools_part) + 50
    budget = max_chars - overhead
    history_parts = []
    used = 0
    needs_review_markers = ("需求回显", "已了解规则", "等待用户输入", "待执行任务", "待确认事项", "[需求回显]", "**需求回显**")
    msg_count = 0
    max_history_msgs = (
        16 if client_profile == QWEN_CODE_OPENAI_PROFILE else (12 if client_profile == CLAUDE_CODE_OPENAI_PROFILE else 200)
    ) if tools else 200
    for msg in reversed(messages):
        if msg_count >= max_history_msgs:
            break
        role = msg.get("role", "")
        if role not in ("user", "assistant", "system", "developer", "tool"):
            continue
        if role in {"system", "developer"} and system_prompt and _extract_system_text_only(msg.get("content", "")).strip() in system_prompt.strip():
            continue

        if role == "tool":
            tool_content = msg.get("content", "") or ""
            tool_call_id = msg.get("tool_call_id", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    p.get("text", "") for p in tool_content if isinstance(p, dict) and p.get("type") == "text"
                )
            elif not isinstance(tool_content, str):
                tool_content = str(tool_content)
            line = f"[Tool Result]{(' id=' + tool_call_id) if tool_call_id else ''}\n{tool_content}\n[/Tool Result]"
            if used + len(line) + 2 > budget and history_parts:
                break
            history_parts.insert(0, line)
            used += len(line) + 2
            msg_count += 1
            continue

        user_text_only = _extract_user_text_only(msg.get("content", ""), client_profile=client_profile) if role == "user" else ""
        if role in {"system", "developer"}:
            text = _extract_system_text_only(msg.get("content", ""))
        else:
            text = _extract_text(
                msg.get("content", ""),
                user_tool_mode=(bool(tools) and role == "user" and _is_heavy_tool_profile(client_profile)),
                client_profile=client_profile,
            )
            if role == "user":
                text = _strip_user_role_system_text(text)

        if role == "assistant" and not text and msg.get("tool_calls"):
            tc_parts = []
            for tc in msg["tool_calls"]:
                function_data = tc.get("function", {})
                name = function_data.get("name", "")
                if tool_catalog is not None:
                    name = tool_catalog.get_model_name(str(name)) or name
                args_str = function_data.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    args = {"raw": args_str}
                tc_parts.append(_render_history_tool_call(name, args, client_profile))
            text = "\n".join(tc_parts)

        if not str(text or "").strip():
            continue

        if tools and role == "assistant" and any(marker in text for marker in needs_review_markers):
            log.debug(f"[Prompt] 跳过需求回显式 assistant 消息 ({len(text)}字)")
            msg_count += 1
            continue
        lower_text = text.lower()
        is_tool_result = role == "user" and (
            "[tool result" in lower_text or text.startswith("{") or '"results"' in text[:100]
        )
        is_tool_result_only_user_msg = role == "user" and not user_text_only.strip() and bool(text.strip())
        prefix = "" if is_tool_result_only_user_msg else {"user": "Human: ", "assistant": "Assistant: ", "system": "System: ", "developer": "System: "}.get(role, "")
        line = text if is_tool_result_only_user_msg else f"{prefix}{text}"
        if used + len(line) + 2 > budget and history_parts:
            break
        history_parts.insert(0, line)
        used += len(line) + 2
        msg_count += 1

    if tools and messages and client_profile != CLAUDE_CODE_OPENAI_PROFILE:
        first_user = next(
            (
                message for message in messages
                if message.get("role") == "user"
                and _extract_user_text_only(message.get("content", ""), client_profile=client_profile).strip()
            ),
            None,
        )
        if first_user:
            first_text = _extract_user_text_only(first_user.get("content", ""), client_profile=client_profile)
            first_short = first_text
            first_line = f"Human: {first_short}"
            if not history_parts or not history_parts[0].startswith(f"Human: {first_text[:60]}"):
                first_line_cost = len(first_line) + 2
                if first_line_cost <= budget:
                    while history_parts and used + first_line_cost > budget:
                        removed = history_parts.pop()
                        used -= len(removed) + 2
                    history_parts.insert(0, first_line)
                    used += first_line_cost
                    log.debug(f"[Prompt] Restored original task context ({len(first_short)} chars)")

    latest_user_line = ""
    if tools and messages and not _has_tool_continuation_after_latest_user(messages, client_profile):
        latest_user = next(
            (
                message for message in reversed(messages)
                if message.get("role") == "user"
                and _extract_user_text_only(message.get("content", ""), client_profile=client_profile).strip()
            ),
            None,
        )
        if latest_user:
            latest_text = _extract_user_text_only(latest_user.get("content", ""), client_profile=client_profile).strip()
            if latest_text:
                latest_short = latest_text
                latest_user_line = f"Human (CURRENT TASK - TOP PRIORITY): {latest_short}"

    if latest_user_line and len(history_parts) > 1:
        duplicate_latest_line = latest_user_line.replace("Human (CURRENT TASK - TOP PRIORITY): ", "Human: ", 1)
        for index in range(len(history_parts) - 1, -1, -1):
            if history_parts[index] == duplicate_latest_line:
                history_parts.pop(index)
                break

    if tools and log.isEnabledFor(logging.DEBUG):
        tool_names = [tool.get("name", "") for tool in tools if tool.get("name")]
        tool_instruction_preview = _safe_preview(tools_part, 360)
        latest_user_preview = _safe_preview(latest_user_line, 220)
        first_user_preview = ""
        if messages:
            first_user = next((message for message in messages if message.get("role") == "user"), None)
            if first_user:
                first_user_preview = _safe_preview(
                    _extract_text(
                        first_user.get("content", ""),
                        user_tool_mode=_is_heavy_tool_profile(client_profile),
                        client_profile=client_profile,
                    ),
                    220,
                )
        log.debug(
            "[Prompt] 工具模式: history_msgs=%s history_chars=%s tool_count=%s tool_names=%s first_user=%r latest_user=%r tool_instr=%r",
            len(history_parts),
            used,
            len(tool_names),
            tool_names[:12],
            first_user_preview,
            latest_user_preview,
            tool_instruction_preview,
        )
    parts = []
    if tools_part and opencode_override:
        parts.append(tools_part)
    if sys_part:
        parts.append(sys_part)
    if tools_part and not opencode_override:
        parts.append(tools_part)
    parts.extend(history_parts)
    if latest_user_line:
        parts.append(latest_user_line)
    parts.append("Assistant:")
    return "\n\n".join(parts)


def messages_to_prompt(req_data: dict, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> PromptBuildResult:
    resolved_client_profile = client_profile
    tool_catalog = req_data.get("_tool_catalog")
    raw_messages = req_data.get("messages", [])
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        role = str(message.get("role", "") or "")
        sanitized_message = dict(message)
        sanitized_message["content"] = _sanitize_prompt_content(
            message.get("content", ""),
            role=role,
            client_profile=resolved_client_profile,
        )
        messages.append(sanitized_message)
    tools = normalize_prompt_tools(req_data.get("tools", []))
    tool_enabled = bool(tools)
    tool_choice = normalize_tool_choice(req_data.get("tool_choice"))
    system_prompt = extract_system_prompt(req_data, client_profile=resolved_client_profile)
    return PromptBuildResult(
        prompt=build_prompt_with_tools(
            system_prompt,
            messages,
            tools,
            client_profile=resolved_client_profile,
            tool_choice_mode=tool_choice.mode,
            required_tool_name=tool_choice.required_tool_name,
            tool_catalog=tool_catalog,
        ),
        tools=tools,
        tool_enabled=tool_enabled,
        client_profile=resolved_client_profile,
    )
