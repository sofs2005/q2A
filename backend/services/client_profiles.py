from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

CLAUDE_CODE_OPENAI_PROFILE = "claude_code_openai"
OPENCLAW_OPENAI_PROFILE = "openclaw_openai"
QWEN_CODE_OPENAI_PROFILE = "qwen_code_openai"

OPENCLAW_STARTUP_PATTERNS = (
    "A new session was started via /new or /reset.",
    "If runtime-provided startup context is included for this first turn",
)
UNTRUSTED_METADATA_PREFIXES = ("Sender (untrusted metadata):", "Conversation info (untrusted metadata):", "System (untrusted):")
OPENCLAW_UNTRUSTED_METADATA_PREFIX = UNTRUSTED_METADATA_PREFIXES[0]
USER_ROLE_SYSTEM_PREFIXES = ("## Memory Recall", "## Compiled Wiki", "System:")
OPENCODE_SYSTEM_PREFIX = "you are opencode"
AGENT_RUNTIME_SYSTEM_MARKERS = (
    "you are a personal assistant running inside",
    "tool availability (filtered by policy):",
    "## tooling",
    "the opencode system prompt may describe native or built-in tool syntax.",
)
SKILL_BOOTSTRAP_PREFIXES = (
    "the following skills provide specialized instructions for specific tasks.",
    "use the read tool to load a skill's file when the task matches its name.",
)

QWEN_CODE_SYSTEM_HINTS = ("qwen code", "qwen-code", "you are qwen code", "you are qwen-code")
QWEN_CODE_OPENAI_TOOL_NAMES = frozenset({
    "read_file",
    "list_directory",
    "write_file",
    "run_shell_command",
})
QWEN_CODE_PROFILE_TOOL_HINTS = {
    "readfile",
    "writefile",
    "editfile",
    "listdirectory",
    "listdir",
    "listfiles",
    "runshellcommand",
    "runcommand",
}
QWEN_CODE_TOOL_HINTS = {
    "read",
    "write",
    "edit",
    "multiedit",
    "notebookedit",
    "grep",
    "glob",
    "bash",
    "readfile",
    "writefile",
    "editfile",
    "notebookeditcell",
    "runcommand",
    "execcommand",
    "listdirectory",
    "listdir",
    "listfiles",
    "searchfiles",
}
QWEN_CODE_FILE_TOOL_HINTS = {
    "read",
    "write",
    "edit",
    "multiedit",
    "notebookedit",
    "readfile",
    "writefile",
    "editfile",
}
QWEN_CODE_NAV_TOOL_HINTS = {
    "grep",
    "glob",
    "bash",
    "runcommand",
    "execcommand",
    "listdirectory",
    "listdir",
    "listfiles",
    "searchfiles",
}
QWEN_CODE_TASK_REGEX = re.compile(
    r"(code|coding|program|repo|repository|refactor|debug|fix|implement|patch|file|files|terminal|shell|bash|command|test|build|编程|代码|仓库|文件|脚本|命令|调试|修复|实现|重构|测试)",
    re.IGNORECASE,
)
QWEN_CODE_OPENAI_HINT_HEADERS = (
    "user-agent",
    "x-openai-client-user-agent",
    "x-client-user-agent",
)
OPENAI_SDK_FINGERPRINT_HEADERS = (
    "x-openai-client-user-agent",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-runtime",
)


def header_value(headers: Mapping[str, Any] | Any, header_name: str) -> str:
    value = ""
    if hasattr(headers, "get"):
        value = headers.get(header_name, "")
    elif isinstance(headers, Mapping):
        value = headers.get(header_name, "")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def normalized_tool_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"]
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema") or tool.get("parameters") or {},
    }


def normalize_tools(tools: list[Any] | None) -> list[dict[str, Any]]:
    return [normalize_tool(tool) for tool in (tools or []) if isinstance(tool, dict)]


def extract_declared_tool_names(req_data: dict[str, Any] | None) -> set[str]:
    if not isinstance(req_data, dict):
        return set()

    tool_names: set[str] = set()
    for tool in req_data.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        candidate = tool.get("name")
        if not isinstance(candidate, str) or not candidate.strip():
            function_payload = tool.get("function")
            if isinstance(function_payload, dict):
                candidate = function_payload.get("name")
        if isinstance(candidate, str) and candidate.strip():
            tool_names.add(candidate.strip().lower())
    return tool_names


def has_qwen_code_header_hint(headers: Mapping[str, Any] | Any) -> bool:
    for header_name in QWEN_CODE_OPENAI_HINT_HEADERS:
        value = header_value(headers, header_name).lower()
        if "qwen" in value and "code" in value:
            return True
    return False


def has_openai_sdk_fingerprint(headers: Mapping[str, Any] | Any) -> bool:
    return any(header_value(headers, header_name) for header_name in OPENAI_SDK_FINGERPRINT_HEADERS)


def is_qwen_code_openai_request(headers: Mapping[str, Any] | Any, req_data: dict[str, Any] | None) -> bool:
    tool_names = extract_declared_tool_names(req_data)
    qwen_tool_matches = len(tool_names & QWEN_CODE_OPENAI_TOOL_NAMES)
    if qwen_tool_matches >= len(QWEN_CODE_OPENAI_TOOL_NAMES):
        return True
    if qwen_tool_matches >= 3 and (has_qwen_code_header_hint(headers) or has_openai_sdk_fingerprint(headers)):
        return True
    return has_qwen_code_header_hint(headers)


def user_role_system_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if any(cleaned.startswith(prefix) for prefix in USER_ROLE_SYSTEM_PREFIXES):
        return cleaned
    if is_agent_runtime_prose(cleaned, "user"):
        user_tail = re.sub(r"(?is)^.*?tool availability \(filtered by policy\):.*?(?:\n\n|$)", "", cleaned).strip()
        user_tail = re.sub(r"(?is)^you are a personal assistant running inside .*?(?:\n\n|$)", "", user_tail).strip()
        if user_tail and cleaned.endswith(user_tail):
            return cleaned[: -len(user_tail)].rstrip()
        return cleaned
    return ""


def is_user_role_system_text(text: str) -> bool:
    return bool(user_role_system_text(text))


def strip_openclaw_untrusted_metadata(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned or not any(cleaned.startswith(prefix) for prefix in UNTRUSTED_METADATA_PREFIXES):
        return cleaned

    kept_lines: list[str] = []
    lines = cleaned.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in UNTRUSTED_METADATA_PREFIXES):
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index < len(lines) and lines[index].strip().startswith("```"):
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    index += 1
                if index < len(lines):
                    index += 1
            continue
        kept_lines.append(line)
        index += 1

    return "\n".join(kept_lines).strip()


def sanitize_openclaw_user_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("System:"):
        return ""
    if cleaned.startswith(("## Memory Recall", "## Compiled Wiki")):
        parts = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        while parts and parts[0].startswith(("## Memory Recall", "## Compiled Wiki")):
            parts.pop(0)
        cleaned = "\n\n".join(parts).strip()
        if not cleaned:
            return ""
    if any(marker in cleaned for marker in OPENCLAW_STARTUP_PATTERNS):
        return ""
    if any(cleaned.startswith(prefix) for prefix in UNTRUSTED_METADATA_PREFIXES):
        cleaned = strip_openclaw_untrusted_metadata(cleaned)
        if not cleaned:
            return ""
    lowered = cleaned.lower()
    if all(prefix in lowered for prefix in SKILL_BOOTSTRAP_PREFIXES) and "<available_skills>" in lowered:
        cleaned = re.sub(r"(?is)^.*?</available_skills>\s*", "", cleaned).strip()
    if is_agent_runtime_prose(cleaned, "assistant") or is_agent_runtime_prose(cleaned, "user"):
        cleaned = re.sub(r"(?is)^.*?tool availability \(filtered by policy\):.*?(?:\n\n|$)", "", cleaned).strip()
        cleaned = re.sub(r"(?is)^you are a personal assistant running inside .*?(?:\n\n|$)", "", cleaned).strip()
    return cleaned


def is_agent_runtime_prose(text: str, role: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.strip().lower()
    if not lowered:
        return False
    if role in {"system", "assistant", "user"}:
        return any(marker in lowered for marker in AGENT_RUNTIME_SYSTEM_MARKERS)
    return False


def sanitize_runtime_prompt_text(text: str, _role: str) -> str:
    if not isinstance(text, str):
        return ""
    return text


def extract_user_text_only(content: Any, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    if isinstance(content, str):
        return sanitize_openclaw_user_text(content) if client_profile == OPENCLAW_OPENAI_PROFILE else content
    if not isinstance(content, list):
        return ""

    text_blocks: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type", "") != "text":
            continue
        block_text = part.get("text", "")
        if client_profile == OPENCLAW_OPENAI_PROFILE:
            block_text = sanitize_openclaw_user_text(block_text)
        if block_text:
            text_blocks.append(block_text)
    return "\n".join(text_blocks)


def _extract_system_text(content: Any) -> str:
    if isinstance(content, str):
        return sanitize_runtime_prompt_text(content, "system")
    if isinstance(content, list):
        return "\n".join(
            sanitize_runtime_prompt_text(part.get("text", ""), "system")
            for part in content
            if isinstance(part, dict) and part.get("type", "") in {"text", "input_text", "output_text"}
        )
    return ""


def extract_system_prompt(req_data: dict[str, Any], *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    system_parts: list[str] = []
    for field_name in ("system", "developer", "instructions"):
        system_text = _extract_system_text(req_data.get(field_name, ""))
        if system_text:
            system_parts.append(system_text)

    for msg in req_data.get("messages", []) or []:
        role = msg.get("role")
        system_text = _extract_system_text(msg.get("content", ""))
        if role in {"system", "developer"} and system_text:
            system_parts.append(system_text)
        elif role == "user":
            user_system_text = user_role_system_text(system_text)
            if user_system_text:
                system_parts.append(user_system_text)
    return "\n\n".join(system_parts)


def looks_like_opencode_system_prompt(system_prompt: str) -> bool:
    if not isinstance(system_prompt, str):
        return False
    return system_prompt.strip().lower().startswith(OPENCODE_SYSTEM_PREFIX)


def extract_latest_user_text(
    messages: list[dict[str, Any]] | None,
    *,
    client_profile: str = OPENCLAW_OPENAI_PROFILE,
) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        text = extract_user_text_only(msg.get("content", ""), client_profile=client_profile).strip()
        if text:
            return text
    return ""


def request_looks_like_coding_task(
    req_data: dict[str, Any],
    *,
    client_profile: str = OPENCLAW_OPENAI_PROFILE,
) -> bool:
    normalized_tools = normalize_tools(req_data.get("tools", []))
    normalized_names = {
        normalized_tool_name(tool.get("name", ""))
        for tool in normalized_tools
        if tool.get("name")
    }
    latest_user_text = extract_latest_user_text(
        req_data.get("messages", []),
        client_profile=client_profile,
    )
    if len(normalized_names & QWEN_CODE_FILE_TOOL_HINTS) >= 1 and len(normalized_names & QWEN_CODE_NAV_TOOL_HINTS) >= 1:
        return True
    if len(normalized_names & QWEN_CODE_TOOL_HINTS) >= 3:
        return True
    return bool(latest_user_text and QWEN_CODE_TASK_REGEX.search(latest_user_text))


def infer_client_profile(
    req_data: dict[str, Any],
    *,
    fallback_profile: str = OPENCLAW_OPENAI_PROFILE,
) -> str:
    if fallback_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}:
        return fallback_profile

    system_prompt = extract_system_prompt(req_data, client_profile=fallback_profile)
    system_lower = system_prompt.strip().lower()
    if looks_like_opencode_system_prompt(system_prompt):
        return fallback_profile
    if any(hint in system_lower for hint in QWEN_CODE_SYSTEM_HINTS):
        return QWEN_CODE_OPENAI_PROFILE

    normalized_names = {
        normalized_tool_name(tool.get("name", ""))
        for tool in normalize_tools(req_data.get("tools", []))
        if tool.get("name")
    }
    qwen_code_tool_matches = normalized_names & QWEN_CODE_PROFILE_TOOL_HINTS
    if len(qwen_code_tool_matches) >= 3:
        return QWEN_CODE_OPENAI_PROFILE
    if len(qwen_code_tool_matches) >= 2 and request_looks_like_coding_task(req_data, client_profile=fallback_profile):
        return QWEN_CODE_OPENAI_PROFILE
    return fallback_profile


def detect_openai_client_profile(headers: Mapping[str, Any] | Any, req_data: dict[str, Any] | None) -> str:
    if header_value(headers, "x-anthropic-billing-header"):
        return CLAUDE_CODE_OPENAI_PROFILE
    if is_qwen_code_openai_request(headers, req_data):
        return QWEN_CODE_OPENAI_PROFILE
    return OPENCLAW_OPENAI_PROFILE


__all__ = [
    "CLAUDE_CODE_OPENAI_PROFILE",
    "OPENCLAW_OPENAI_PROFILE",
    "QWEN_CODE_OPENAI_PROFILE",
    "detect_openai_client_profile",
    "extract_declared_tool_names",
    "extract_latest_user_text",
    "extract_system_prompt",
    "extract_user_text_only",
    "has_openai_sdk_fingerprint",
    "has_qwen_code_header_hint",
    "header_value",
    "infer_client_profile",
    "is_qwen_code_openai_request",
    "looks_like_opencode_system_prompt",
    "normalize_tool",
    "normalize_tools",
    "normalized_tool_name",
    "request_looks_like_coding_task",
    "sanitize_openclaw_user_text",
    "sanitize_runtime_prompt_text",
    "strip_openclaw_untrusted_metadata",
    "is_agent_runtime_prose",
]
