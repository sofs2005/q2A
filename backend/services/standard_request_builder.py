from __future__ import annotations

from backend.adapter.standard_request import StandardRequest, enforce_declared_tool_choice, normalize_tool_choice
from backend.core.config import resolve_request_model
from backend.services.command_environment import CommandEnvironment
from backend.services.client_profiles import infer_client_profile, request_looks_like_coding_task
from backend.toolcore.prompt_builder import messages_to_prompt
from backend.toolcore.request_normalizer import normalize_chat_request, to_prompt_payload
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcall.normalize import build_tool_name_registry


SUBAGENT_COMMAND_ALIAS_NAMES = frozenset({"subagents"})
SUBAGENT_EXECUTABLE_TOOL_NAMES = frozenset({"agents_list", "sessions_spawn"})
NO_CLIENT_TOOLS_AUTHENTICITY_NOTICE = (
    "[System]\n"
    "Client-side tools are not available in this request. If the task requires file access, command execution, web access, browser actions, agents, skills, or artifact verification, do not claim that you performed those actions. Answer only from the supplied conversation context and clearly state any execution or verification that remains unavailable."
)


def _raw_declared_tool_names(req_data: dict) -> set[str]:
    names: set[str] = set()
    for raw_tool in req_data.get("tools", []) or []:
        if not isinstance(raw_tool, dict):
            continue
        function_payload = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else {}
        name = str(raw_tool.get("name") or function_payload.get("name") or "").strip().lower()
        if name:
            names.add(name)
    return names


def _excluded_command_like_tool_names(req_data: dict) -> set[str] | None:
    names = _raw_declared_tool_names(req_data)
    if names & SUBAGENT_COMMAND_ALIAS_NAMES and names & SUBAGENT_EXECUTABLE_TOOL_NAMES:
        return set(SUBAGENT_COMMAND_ALIAS_NAMES)
    return None


def _go_compatible_no_tool_prompt(prompt: str) -> str:
    parts = [NO_CLIENT_TOOLS_AUTHENTICITY_NOTICE]
    for block in str(prompt or "").split("\n\n"):
        block = block.strip()
        if not block or block == "Assistant:":
            continue
        if block.startswith("Human: "):
            parts.append("[User]\n" + block[len("Human: "):])
        elif block.startswith("Assistant: "):
            parts.append("[Assistant]\n" + block[len("Assistant: "):])
        elif block.startswith("System: "):
            parts.append("[System]\n" + block[len("System: "):])
        else:
            parts.append(block)
    return "\n\n".join(parts)


def build_chat_standard_request(
    req_data: dict,
    *,
    default_model: str,
    surface: str,
    client_profile: str = "openclaw_openai",
    command_environment: CommandEnvironment | None = None,
) -> StandardRequest:
    requested_model = req_data.get("model", default_model)
    effective_client_profile = infer_client_profile(req_data, fallback_profile=client_profile)
    normalized_request = normalize_chat_request(req_data, excluded_tool_names=_excluded_command_like_tool_names(req_data))
    request_tool_catalog = req_data.get("_tool_catalog")
    if not isinstance(request_tool_catalog, ToolCatalog):
        request_tool_catalog = normalized_request.tool_catalog
    normalized_payload = to_prompt_payload(normalized_request, model=requested_model, stream=bool(req_data.get("stream", False)))
    for field_name in ("system", "developer", "instructions"):
        if field_name in req_data:
            normalized_payload[field_name] = req_data.get(field_name, "")
    if command_environment is None:
        command_environment = CommandEnvironment()
    normalized_payload["_command_environment"] = command_environment
    prompt_result = messages_to_prompt(normalized_payload, client_profile=effective_client_profile)
    tools = prompt_result.tools
    tool_names = [tool_name for tool_name in (tool.get("name") for tool in tools) if isinstance(tool_name, str) and tool_name]
    coding_intent = request_looks_like_coding_task(req_data, client_profile=effective_client_profile)
    tool_choice = normalize_tool_choice(normalized_payload.get("tool_choice"))
    tool_choice = enforce_declared_tool_choice(tool_choice, tool_names)
    prompt = prompt_result.prompt
    if not prompt_result.tool_enabled and surface == "openai":
        prompt = _go_compatible_no_tool_prompt(prompt)

    return StandardRequest(
        prompt=prompt,
        response_model=requested_model,
        resolved_model=resolve_request_model(
            requested_model,
            client_profile=effective_client_profile,
            tool_enabled=prompt_result.tool_enabled,
            coding_intent=coding_intent,
        ),
        surface=surface,
        client_profile=effective_client_profile,
        requested_model=requested_model,
        stream=req_data.get("stream", False),
        tools=tools,
        tool_names=tool_names,
        tool_name_registry=build_tool_name_registry(tool_names),
        tool_catalog=request_tool_catalog,
        tool_enabled=prompt_result.tool_enabled,
        tool_choice_mode=tool_choice.mode,
        required_tool_name=tool_choice.required_tool_name,
        tool_choice_raw=normalized_request.raw_tool_choice,
        command_environment=command_environment,
    )
