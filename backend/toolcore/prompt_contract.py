from __future__ import annotations

import json
from typing import Any

from backend.services.client_profiles import (
    CLAUDE_CODE_OPENAI_PROFILE,
    OPENCLAW_OPENAI_PROFILE,
    QWEN_CODE_OPENAI_PROFILE,
)


def _is_heavy_tool_profile(client_profile: str) -> bool:
    return client_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}


def compact_history_tool_input(name: str, input_data: dict[str, Any], client_profile: str) -> dict[str, Any]:
    if not _is_heavy_tool_profile(client_profile) or not isinstance(input_data, dict):
        return input_data
    compact = dict(input_data)
    large_text_keys = ("content", "new_string", "old_string", "insert_text", "text", "patch")
    for key in large_text_keys:
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 160:
            compact[key] = f"[omitted {len(value)} chars]"
    if name in {"Write", "Edit", "NotebookEdit"}:
        preferred: dict[str, Any] = {}
        for key in ("file_path", "path", "target_file", "filename", "old_string", "new_string", "content"):
            if key in compact:
                preferred[key] = compact[key]
        if preferred:
            compact = preferred
    return compact



def render_history_tool_call(name: str, input_data: dict[str, Any], client_profile: str) -> str:
    compact = compact_history_tool_input(name, input_data, client_profile)
    payload = {"name": str(name), "input": compact}
    return "##TOOL_CALL##\n" + json.dumps(payload, ensure_ascii=False) + "\n##END_CALL##"


def model_bridge_tool_name(index: int) -> str:
    return f"bridge-{index}"


def normalize_prompt_tool(tool: dict[str, Any]) -> dict[str, Any]:
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


def normalize_prompt_tools(tools: list[Any]) -> list[dict[str, Any]]:
    return [normalize_prompt_tool(tool) for tool in tools if isinstance(tool, dict)]


def _tool_param_hint(tool: dict[str, Any], *, max_keys: int = 3) -> str:
    params = tool.get("parameters", {}) or {}
    if not isinstance(params, dict):
        return ""
    props = params.get("properties", {}) or {}
    if not isinstance(props, dict) or not props:
        return ""
    required = params.get("required", []) or []
    ordered_keys: list[str] = []
    for key in required:
        if key in props and key not in ordered_keys:
            ordered_keys.append(key)
    for key in props:
        if key not in ordered_keys:
            ordered_keys.append(key)
    shown = ordered_keys[:max_keys]
    if not shown:
        return ""
    suffix = ", ..." if len(ordered_keys) > len(shown) else ""
    required_shown = [key for key in required if key in shown][:max_keys]
    required_suffix = f"; required: {', '.join(required_shown)}" if required_shown else ""
    return f" input keys: {', '.join(shown)}{suffix}{required_suffix}"


def _tool_usage_line(tool: dict[str, Any], *, max_desc: int = 40, max_keys: int = 3) -> str:
    name = tool.get("name", "")
    desc = (tool.get("description", "") or "")[:max_desc]
    hint = _tool_param_hint(tool, max_keys=max_keys)
    line = f"- {name}"
    if desc:
        line += f": {desc}"
    if hint:
        line += hint
    return line


def build_tool_instruction_block(
    tools: list[dict[str, Any]],
    client_profile: str,
    *,
    tool_choice_mode: str = "auto",
    required_tool_name: str | None = None,
) -> str:
    names = [t.get("name", "") for t in tools if t.get("name")]
    force_constraint_lines: list[str] = []
    if tool_choice_mode == "required":
        if required_tool_name:
            force_constraint_lines.extend([
                f'【强制】本轮必须调用工具 `{required_tool_name}`，不能仅回复普通文本，也不能改用其它工具。',
                f'MANDATORY: this turn MUST call the exact tool "{required_tool_name}". Plain text only is not allowed, and using a different tool is not allowed.',
            ])
        else:
            force_constraint_lines.extend([
                "【强制】本轮必须至少调用一个工具，不能只输出普通文本。",
                "MANDATORY: this turn MUST include at least one tool call. Plain text only is not allowed.",
            ])
    elif tool_choice_mode == "none":
        force_constraint_lines.extend([
            "【强制】本轮不要调用任何工具，直接给出普通文本回复。",
            "MANDATORY: do NOT call any tool on this turn. Respond with plain text only.",
        ])

    native_error_example = names[0] if names else "TOOL_NAME"
    lines = [
        "=== MANDATORY TOOL CALL INSTRUCTIONS ===",
        "This gateway-injected block only defines how to serialize tool calls for the bridge.",
        "These are gateway bridge tools, not upstream/native Qwen tools; do not invoke the platform's built-in tool system.",
        f"If you need a tool, output the safe JSON text block directly; never answer with platform errors such as `Tool {native_error_example} does not exists.`",
        "Follow the client's system/developer instructions for persona, style, language, and normal response format.",
        f"Bridge-call slots available: {', '.join(names)}",
        "",
        "TOOL CALL FORMAT — FOLLOW EXACTLY:",
        "##TOOL_CALL##",
        '{"name": "TOOL_NAME_HERE", "input": {"PARAMETER_NAME": "PARAMETER_VALUE"}}',
        "##END_CALL##",
        "",
        "Rules:",
        "- Use exactly one JSON object between ##TOOL_CALL## and ##END_CALL## when calling one tool.",
        "- Put the exact bridge slot id from the list above in the JSON name field.",
        "- Put every top-level argument under the JSON input object.",
        "- Strings, objects, arrays, numbers, booleans, and null must be valid JSON values.",
        "- Do not emit placeholder, blank, or whitespace-only parameters.",
        "- If a required parameter value is unknown, ask the user or answer normally instead of outputting an empty tool call.",
        "- Do NOT wrap the block in markdown fences. Do NOT output explanations, role markers, or internal monologue around the tool block.",
        "- If you call a tool, the first non-whitespace characters of that tool block must be exactly ##TOOL_CALL##.",
        "- Compatibility note: XML-like formats may be parsed, but the model-facing format is the safe JSON block only.",
        "",
        *force_constraint_lines,
        *([""] if force_constraint_lines else []),
        "Available bridge slots (copy the slot id exactly into the JSON name field):"
    ]
    for tool in tools:
        lines.append(
            _tool_usage_line(
                tool,
                max_desc=72 if client_profile == QWEN_CODE_OPENAI_PROFILE else 40,
                max_keys=6 if client_profile == QWEN_CODE_OPENAI_PROFILE else 3,
            )
        )
    lines.append("=== END TOOL INSTRUCTIONS ===")
    return "\n".join(lines)
