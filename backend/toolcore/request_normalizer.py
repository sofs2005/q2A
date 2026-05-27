from __future__ import annotations

import json
from typing import Any

from backend.skills.adapters import extract_skill_catalog_from_messages
from backend.skills.catalog import normalize_skill_catalog
from backend.toolcore.prompt_contract import model_bridge_tool_name
from backend.toolcore.types import (
    CanonicalToolCall,
    CanonicalToolResult,
    ToolChoicePolicy,
    ToolCoreRequest,
    ToolDefinition,
)
from backend.toolcore.tool_catalog import ToolCatalog

__all__ = [
    "normalize_chat_request",
    "normalize_responses_request",
    "normalize_anthropic_request",
    "normalize_gemini_request",
    "to_prompt_payload",
]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_skill_catalog(raw_skill_catalog: Any, messages: list[dict[str, Any]], client_profile: str) -> Any:
    if raw_skill_catalog is not None:
        return normalize_skill_catalog(raw_skill_catalog)
    return extract_skill_catalog_from_messages(messages, client_profile=client_profile)


def _normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if raw_messages is None:
        return []
    if isinstance(raw_messages, str):
        return [{"role": "user", "content": raw_messages}]
    if isinstance(raw_messages, list):
        return [item for item in raw_messages if isinstance(item, dict)]
    if isinstance(raw_messages, dict):
        return [raw_messages]
    raise ValueError("messages/input must be a string, object, or list")


def _normalize_tools(raw_tools: Any, *, excluded_tool_names: set[str] | None = None) -> list[ToolDefinition]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list")

    excluded = {name.strip().lower() for name in (excluded_tool_names or set()) if name.strip()}
    tools: list[ToolDefinition] = []
    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, dict):
            continue
        function_payload = _dict_or_empty(raw_tool.get("function"))
        name = str(
            raw_tool.get("name")
            or function_payload.get("name")
            or ""
        ).strip()
        if not name or name.lower() in excluded:
            continue
        description = str(
            raw_tool.get("description")
            or function_payload.get("description")
            or ""
        )
        parameters = raw_tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = function_payload.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                client_name=name,
                model_name=model_bridge_tool_name(len(tools)),
                raw=raw_tool,
            )
        )
    return tools


def _normalize_anthropic_tools(raw_tools: Any) -> list[ToolDefinition]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list")

    tools: list[ToolDefinition] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        name = str(raw_tool.get("name") or "").strip()
        if not name:
            continue
        description = str(raw_tool.get("description") or "")
        parameters = _dict_or_empty(raw_tool.get("input_schema"))
        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                client_name=name,
                model_name=model_bridge_tool_name(len(tools)),
                raw=raw_tool,
            )
        )
    return tools


def _normalize_gemini_messages(contents: Any, *, tool_catalog: ToolCatalog | None = None) -> list[dict[str, Any]]:
    if contents is None:
        return []
    if not isinstance(contents, list):
        raise ValueError("contents must be a list")

    messages: list[dict[str, Any]] = []
    call_id_counter = 0
    for message in contents:
        if not isinstance(message, dict):
            continue
        role = "assistant" if message.get("role") == "model" else "user"
        text_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        function_responses: list[dict[str, Any]] = []
        for part in message.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
            elif isinstance(part.get("functionCall"), dict):
                function_calls.append(part["functionCall"])
            elif isinstance(part.get("functionResponse"), dict):
                function_responses.append(part["functionResponse"])
        if text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})
        if role == "assistant" and function_calls:
            tool_calls: list[dict[str, Any]] = []
            for fc in function_calls:
                call_id = f"call_{call_id_counter}"
                call_id_counter += 1
                name = str(fc.get("name", ""))
                args = fc.get("args", fc.get("arguments", {}))
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args or "{}")
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                })
            if tool_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        if role == "user" and function_responses:
            for fr in function_responses:
                name = str(fr.get("name", ""))
                response_data = fr.get("response")
                if isinstance(response_data, dict):
                    content = json.dumps(response_data, ensure_ascii=False)
                elif isinstance(response_data, str):
                    content = response_data
                else:
                    content = str(response_data or "")
                messages.append({"role": "tool", "tool_call_id": f"call_{call_id_counter}", "name": name, "content": content})
                call_id_counter += 1
    return messages


def _normalize_gemini_tools(raw_tools: Any) -> list[ToolDefinition]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list")

    tools: list[ToolDefinition] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        declarations = raw_tool.get("functionDeclarations", [])
        if not isinstance(declarations, list):
            continue
        for declaration in declarations:
            if not isinstance(declaration, dict):
                continue
            name = str(declaration.get("name") or "").strip()
            if not name:
                continue
            description = str(declaration.get("description") or "")
            parameters = _dict_or_empty(declaration.get("parameters"))
            tools.append(
                ToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                    client_name=name,
                    model_name=model_bridge_tool_name(len(tools)),
                    raw=declaration,
                )
            )
    return tools


def _normalize_gemini_tool_choice(body: dict[str, Any]) -> Any:
    tool_config = body.get("toolConfig")
    if not isinstance(tool_config, dict):
        return None
    function_calling = tool_config.get("functionCallingConfig")
    if not isinstance(function_calling, dict):
        return None
    mode = str(function_calling.get("mode") or "").strip().upper()
    if mode == "NONE":
        return "none"
    if mode == "ANY":
        return "required"
    if mode == "AUTO":
        return "auto"
    return None


def to_prompt_payload(request: ToolCoreRequest, *, model: str, stream: bool = False) -> dict[str, Any]:
    tools = [
        {
            "name": tool.model_name or tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in request.tools
    ]

    tool_choice: Any = request.raw_tool_choice
    if request.tool_choice_policy == ToolChoicePolicy.NONE:
        tool_choice = "none"
    elif request.tool_choice_policy == ToolChoicePolicy.REQUIRED:
        tool_choice = "required"
    elif request.tool_choice_policy == ToolChoicePolicy.FORCED and request.forced_tool_name:
        model_tool_name = request.tool_catalog.get_model_name(request.forced_tool_name) if request.tool_catalog else None
        tool_choice = {"type": "function", "function": {"name": model_tool_name or request.forced_tool_name}}

    return {
        "model": model,
        "messages": request.messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "stream": stream,
        "_tool_catalog": request.tool_catalog,
        "_skill_catalog": request.skill_catalog,
    }


def _parse_tool_choice(raw_tool_choice: Any, declared_tool_names: set[str]) -> tuple[ToolChoicePolicy, str | None]:
    if raw_tool_choice is None:
        return ToolChoicePolicy.AUTO, None

    if isinstance(raw_tool_choice, str):
        lowered = raw_tool_choice.strip().lower()
        if lowered == "auto":
            return ToolChoicePolicy.AUTO, None
        if lowered in {"required", "any"}:
            return ToolChoicePolicy.REQUIRED, None
        if lowered == "none":
            return ToolChoicePolicy.NONE, None
        raise ValueError(f"Invalid tool_choice: {raw_tool_choice}")

    if not isinstance(raw_tool_choice, dict):
        raise ValueError(f"Invalid tool_choice: {raw_tool_choice}")

    choice_type = str(raw_tool_choice.get("type") or "").strip().lower()
    if choice_type in {"", "auto"}:
        return ToolChoicePolicy.AUTO, None
    if choice_type in {"required", "any"}:
        return ToolChoicePolicy.REQUIRED, None
    if choice_type == "none":
        return ToolChoicePolicy.NONE, None
    if choice_type in {"function", "tool"}:
        function_payload = _dict_or_empty(raw_tool_choice.get("function"))
        forced_name = str(function_payload.get("name") or raw_tool_choice.get("name") or "").strip()
        if not forced_name:
            raise ValueError("Invalid tool_choice: missing forced tool name")
        if declared_tool_names and forced_name not in declared_tool_names:
            lowered_map = {name.lower(): name for name in declared_tool_names}
            canonical = lowered_map.get(forced_name.lower())
            if canonical is None:
                raise ValueError(f"Invalid tool_choice: undeclared tool {forced_name}")
            forced_name = canonical
        return ToolChoicePolicy.FORCED, forced_name
    raise ValueError(f"Invalid tool_choice: {raw_tool_choice}")


def _canonical_tool_call_from_assistant_message(message: dict[str, Any], declared_tool_names: set[str]) -> list[CanonicalToolCall]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []

    canonical_calls: list[CanonicalToolCall] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function_payload = _dict_or_empty(call.get("function"))
        arguments = function_payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_input = json.loads(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed_input = {"value": arguments}
        elif isinstance(arguments, dict):
            parsed_input = arguments
        else:
            parsed_input = {}
        if not isinstance(parsed_input, dict):
            parsed_input = {"value": parsed_input}
        name = str(function_payload.get("name") or call.get("name") or "").strip()
        if declared_tool_names and name not in declared_tool_names:
            continue
        canonical_calls.append(
            CanonicalToolCall(
                call_id=str(call.get("id") or call.get("call_id") or "").strip(),
                name=name,
                input=parsed_input,
            )
        )
    return [call for call in canonical_calls if call.call_id and call.name]


def _normalize_function_call_output(raw_output: Any) -> list[CanonicalToolResult]:
    if raw_output is None:
        return []
    items = raw_output if isinstance(raw_output, list) else [raw_output]
    out: list[CanonicalToolResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        if not call_id:
            continue
        out.append(
            CanonicalToolResult(
                call_id=call_id,
                output=item.get("output", ""),
                tool_name=str(item.get("name") or "").strip() or None,
            )
        )
    return out


def normalize_chat_request(req_data: dict[str, Any], *, excluded_tool_names: set[str] | None = None) -> ToolCoreRequest:
    messages = _normalize_messages(req_data.get("messages"))
    tools = _normalize_tools(req_data.get("tools"), excluded_tool_names=excluded_tool_names)
    tool_catalog = ToolCatalog(tools)
    declared_tool_names = {tool.name for tool in tools}
    raw_tool_choice = req_data.get("tool_choice")
    policy, forced_tool_name = _parse_tool_choice(raw_tool_choice, declared_tool_names)
    if raw_tool_choice is not None and not tools:
        raise ValueError("tool_choice provided but no tools declared")

    tool_calls: list[CanonicalToolCall] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() == "assistant":
            tool_calls.extend(_canonical_tool_call_from_assistant_message(message, declared_tool_names))

    skill_catalog = _normalize_skill_catalog(req_data.get("_skill_catalog"), messages, str(req_data.get("client_profile") or ""))

    return ToolCoreRequest(
        messages=messages,
        tools=tools,
        tool_choice_policy=policy,
        forced_tool_name=forced_tool_name,
        tool_calls=tool_calls,
        raw_tool_choice=raw_tool_choice,
        tool_catalog=tool_catalog,
        skill_catalog=skill_catalog,
    )


def normalize_responses_request(req_data: dict[str, Any]) -> ToolCoreRequest:
    messages = _normalize_messages(req_data.get("input", req_data.get("messages")))
    tools = _normalize_tools(req_data.get("tools"))
    tool_catalog = ToolCatalog(tools)
    declared_tool_names = {tool.name for tool in tools}
    raw_tool_choice = req_data.get("tool_choice")
    if raw_tool_choice is not None and not tools:
        raise ValueError("tool_choice provided but no tools declared")
    policy, forced_tool_name = _parse_tool_choice(raw_tool_choice, declared_tool_names)

    tool_calls: list[CanonicalToolCall] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() == "assistant":
            tool_calls.extend(_canonical_tool_call_from_assistant_message(message, declared_tool_names))

    tool_results = _normalize_function_call_output(req_data.get("function_call_output"))
    skill_catalog = _normalize_skill_catalog(req_data.get("_skill_catalog"), messages, str(req_data.get("client_profile") or ""))

    return ToolCoreRequest(
        messages=messages,
        tools=tools,
        tool_choice_policy=policy,
        forced_tool_name=forced_tool_name,
        tool_calls=tool_calls,
        tool_results=tool_results,
        raw_tool_choice=raw_tool_choice,
        tool_catalog=tool_catalog,
        skill_catalog=skill_catalog,
    )


def normalize_anthropic_request(req_data: dict[str, Any]) -> ToolCoreRequest:
    messages = _normalize_messages(req_data.get("messages"))
    tools = _normalize_anthropic_tools(req_data.get("tools"))
    tool_catalog = ToolCatalog(tools)
    declared_tool_names = {tool.name for tool in tools}
    raw_tool_choice = req_data.get("tool_choice")
    policy, forced_tool_name = _parse_tool_choice(raw_tool_choice, declared_tool_names)
    if raw_tool_choice is not None and not tools:
        raise ValueError("tool_choice provided but no tools declared")

    skill_catalog = _normalize_skill_catalog(req_data.get("_skill_catalog"), messages, str(req_data.get("client_profile") or ""))

    return ToolCoreRequest(
        messages=messages,
        tools=tools,
        tool_choice_policy=policy,
        forced_tool_name=forced_tool_name,
        raw_tool_choice=raw_tool_choice,
        tool_catalog=tool_catalog,
        skill_catalog=skill_catalog,
    )


def normalize_gemini_request(req_data: dict[str, Any], *, model: str, force_stream: bool | None = None) -> ToolCoreRequest:
    del model, force_stream
    tools = _normalize_gemini_tools(req_data.get("tools"))
    tool_catalog = ToolCatalog(tools)
    messages = _normalize_gemini_messages(req_data.get("contents"), tool_catalog=tool_catalog)
    tool_catalog = ToolCatalog(tools)
    declared_tool_names = {tool.name for tool in tools}
    raw_tool_choice = _normalize_gemini_tool_choice(req_data)
    if raw_tool_choice is not None and not tools:
        raise ValueError("tool_choice provided but no tools declared")
    policy, forced_tool_name = _parse_tool_choice(raw_tool_choice, declared_tool_names)

    skill_catalog = _normalize_skill_catalog(req_data.get("_skill_catalog"), messages, str(req_data.get("client_profile") or ""))

    return ToolCoreRequest(
        messages=messages,
        tools=tools,
        tool_choice_policy=policy,
        forced_tool_name=forced_tool_name,
        raw_tool_choice=raw_tool_choice,
        tool_catalog=tool_catalog,
        skill_catalog=skill_catalog,
    )
