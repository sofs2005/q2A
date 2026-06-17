from __future__ import annotations

import html
import json
import re
from typing import Any

from backend.services.command_environment import CommandEnvironment
from backend.services.client_profiles import (
    CLAUDE_CODE_OPENAI_PROFILE,
    OPENCLAW_OPENAI_PROFILE,
    QWEN_CODE_OPENAI_PROFILE,
)


def _is_heavy_tool_profile(client_profile: str) -> bool:
    return client_profile in {CLAUDE_CODE_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE}


_MODIFY_TOOLS = frozenset({"Write", "Edit", "NotebookEdit", "Patch"})
_READ_TOOLS = frozenset({"Read", "Grep", "grep", "read"})
_MODIFY_THRESHOLD = 16_000
_READ_THRESHOLD = 4_000
_READ_SAMPLE_CHARS = 2_000
_OTHER_THRESHOLD = 160


def _head_tail_sample(text: str, threshold: int, sample_chars: int) -> str:
    """Keep first and last sample_chars of text, omitting the middle."""
    if len(text) <= threshold:
        return text
    omitted = len(text) - 2 * sample_chars
    if omitted <= 0:
        return text
    return text[:sample_chars] + f"\n... [omitted {omitted} chars] ...\n" + text[-sample_chars:]


def compact_history_tool_input(name: str, input_data: dict[str, Any], client_profile: str) -> dict[str, Any]:
    if not _is_heavy_tool_profile(client_profile) or not isinstance(input_data, dict):
        return input_data
    compact = dict(input_data)
    large_text_keys = ("content", "new_string", "old_string", "insert_text", "text", "patch", "command")

    if name in _MODIFY_TOOLS:
        # Modify tools: higher threshold, full omission when exceeded
        for key in large_text_keys:
            value = compact.get(key)
            if isinstance(value, str) and len(value) > _MODIFY_THRESHOLD:
                compact[key] = f"[omitted {len(value)} chars]"
        # Keep only preferred keys for modify tools
        preferred: dict[str, Any] = {}
        for key in ("file_path", "path", "target_file", "filename", "old_string", "new_string", "content"):
            if key in compact:
                preferred[key] = compact[key]
        if preferred:
            compact = preferred
    elif name in _READ_TOOLS:
        # Read tools: head/tail sampling to preserve file structure
        for key in large_text_keys:
            value = compact.get(key)
            if isinstance(value, str) and len(value) > _READ_THRESHOLD:
                compact[key] = _head_tail_sample(value, _READ_THRESHOLD, _READ_SAMPLE_CHARS)
    else:
        # Other tools: low threshold, full omission
        for key in large_text_keys:
            value = compact.get(key)
            if isinstance(value, str) and len(value) > _OTHER_THRESHOLD:
                compact[key] = f"[omitted {len(value)} chars]"

    return compact


def _cdata(value: str) -> str:
    return "<![CDATA[" + value.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _is_xmlish_name(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:-]*", value) is not None


def _render_dsml_value(value: Any) -> str:
    if isinstance(value, str):
        return _cdata(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return _cdata("[]")
        return "".join(f"<item>{_render_dsml_value(item)}</item>" for item in value)
    if isinstance(value, dict):
        if not value or not all(_is_xmlish_name(str(key)) for key in value):
            return _cdata(json.dumps(value, ensure_ascii=False))
        parts = []
        for key, item in value.items():
            safe_key = html.escape(str(key), quote=True)
            parts.append(f"<{safe_key}>{_render_dsml_value(item)}</{safe_key}>")
        return "".join(parts)
    return _cdata(str(value))


def render_history_tool_call(name: str, input_data: dict[str, Any], client_profile: str) -> str:
    compact = compact_history_tool_input(name, input_data, client_profile)
    safe_name = html.escape(str(name), quote=True)
    lines = ["<|DSML|tool_calls>", f'  <|DSML|invoke name="{safe_name}">']
    for key, value in compact.items():
        safe_key = html.escape(str(key), quote=True)
        lines.append(f'    <|DSML|parameter name="{safe_key}">{_render_dsml_value(value)}</|DSML|parameter>')
    lines.append("  </|DSML|invoke>")
    lines.append("</|DSML|tool_calls>")
    return "\n".join(lines)


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


def _build_name_alias_lines(name_alias_map: dict[str, str] | None) -> list[str]:
    """Render a client-name -> bridge-slot lookup table.

    The user (or earlier turns) may reference a tool by its real client-side
    name (e.g. "Bash"), but the bridge only exposes opaque slot ids
    (e.g. "bridge-2"). Without a lookup the model wrongly concludes the named
    tool is unavailable. This table lets it resolve the mention without us
    mutating free-form user text.
    """
    if not name_alias_map:
        return []
    # Sort by slot id then client name for stable, readable grouping.
    entries = sorted(name_alias_map.items(), key=lambda kv: (kv[1], kv[0]))
    lines = [
        "TOOL NAME RESOLUTION — the user or earlier turns may name a tool by its real client-side name.",
        "Resolve any such name to its bridge slot via this table before deciding a tool is unavailable:",
    ]
    for client_name, slot in entries:
        lines.append(f'- "{client_name}" => {slot}')
    lines.append(
        "When the user mentions a tool name listed here, call the mapped bridge slot; "
        "never reply that a listed tool does not exist."
    )
    lines.append("")
    return lines


def _shell_command_guidance(command_environment: CommandEnvironment | None) -> list[str]:
    shell = str(getattr(command_environment, "shell", "unknown") or "unknown").strip().lower()
    lines = [
        "- Shell command parameters must already be valid shell syntax after JSON parsing. Match the client's actual shell before choosing syntax.",
    ]
    if shell == "powershell":
        lines.append("- PowerShell note: POSIX here-documents are invalid. For multiline Python, prefer @' ... '@ | python - or create a temporary .py file.")
    elif shell in {"bash", "zsh", "sh"}:
        lines.append("- POSIX shell note: quoted here-documents are allowed only when the target shell is explicitly POSIX; otherwise prefer stdin or a temporary script file.")
    else:
        lines.append("- Unknown shell note: prefer stdin or a temporary script file over shell-specific syntax, and avoid complex nested inline quotes.")
    return lines


def build_tool_instruction_block(
    tools: list[dict[str, Any]],
    client_profile: str,
    *,
    tool_choice_mode: str = "auto",
    required_tool_name: str | None = None,
    command_environment: CommandEnvironment | None = None,
    name_alias_map: dict[str, str] | None = None,
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
        f"If you need a tool, output the DSML text block directly; never answer with platform errors such as `Tool {native_error_example} does not exists.`",
        "Follow the client's system/developer instructions for persona, style, language, and normal response format.",
        f"Bridge-call slots available: {', '.join(names)}",
        "",
        "TOOL CALL FORMAT — FOLLOW EXACTLY:",
        "<|DSML|tool_calls>",
        '  <|DSML|invoke name="TOOL_NAME_HERE">',
        '    <|DSML|parameter name="PARAMETER_NAME"><![CDATA[PARAMETER_VALUE]]></|DSML|parameter>',
        "  </|DSML|invoke>",
        "</|DSML|tool_calls>",
        "",
        "Rules:",
        "- Use one <|DSML|tool_calls> root when calling tools.",
        "- Put one or more <|DSML|invoke> entries under the root.",
        "- Use the exact bridge slot id from the list above in the invoke name attribute.",
        "- Every top-level argument must be a <|DSML|parameter name=\"ARG_NAME\"> node.",
        "- Use <![CDATA[...]]> for string values, including code, paths, prompts, and file contents.",
        "- CDATA preserves parameter text exactly; it does not add shell escaping, JSON escaping, or quote balancing.",
        *_shell_command_guidance(command_environment),
        "- Objects use nested XML elements inside the parameter body. Arrays may repeat <item> children.",
        "- Numbers, booleans, and null stay plain text.",
        "- Do not emit placeholder, blank, or whitespace-only parameters.",
        "- If a required parameter value is unknown, ask the user or answer normally instead of outputting an empty tool call.",
        "- Do NOT wrap XML in markdown fences. Do NOT output explanations, role markers, or internal monologue around the tool block.",
        "- If you call a tool, the first non-whitespace characters of that tool block must be exactly <|DSML|tool_calls>.",
        "- Compatibility note: legacy output formats may be parsed, but the model-facing format is DSML/XML only.",
        "",
        *force_constraint_lines,
        *([""] if force_constraint_lines else []),
        *_build_name_alias_lines(name_alias_map),
        "Available bridge slots (copy the slot id exactly into DSML invoke name):"
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
