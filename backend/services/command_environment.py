from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class CommandEnvironment:
    shell: str = "unknown"
    platform: str = "unknown"
    source: str = "unknown"


def _normalized(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _header_value(headers: Mapping[str, Any] | None, *names: str) -> str:
    if not headers:
        return ""
    for name in names:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
        if value is None:
            value = headers.get(name.upper())
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _metadata_value(metadata: dict[str, Any], *names: str) -> str:
    command_env = metadata.get("command_environment")
    if isinstance(command_env, dict):
        for name in names:
            value = command_env.get(name)
            if value is not None and str(value).strip():
                return str(value)
    for name in names:
        value = metadata.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def detect_command_environment(
    *,
    headers: Mapping[str, Any] | None = None,
    request_data: dict[str, Any] | None = None,
) -> CommandEnvironment:
    metadata = request_data.get("metadata") if isinstance(request_data, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}

    metadata_shell = _metadata_value(metadata, "shell", "terminal_shell", "command_shell")
    metadata_platform = _metadata_value(metadata, "platform", "os", "operating_system")
    if metadata_shell or metadata_platform:
        return CommandEnvironment(
            shell=_normalized(metadata_shell),
            platform=_normalized(metadata_platform),
            source="explicit",
        )

    header_shell = _header_value(headers, "x-shell", "x-terminal-shell", "x-command-shell")
    header_platform = _header_value(headers, "x-platform", "x-os", "x-operating-system")
    if header_shell or header_platform:
        return CommandEnvironment(
            shell=_normalized(header_shell),
            platform=_normalized(header_platform),
            source="headers",
        )

    return CommandEnvironment()


def format_command_environment_hint(env: CommandEnvironment | None) -> str:
    if env is None:
        return "unknown"
    shell = _normalized(env.shell)
    platform = _normalized(env.platform)
    if shell == "unknown" and platform == "unknown":
        return "unknown"
    if shell != "unknown" and platform != "unknown":
        return f"{shell}/{platform}"
    return shell if shell != "unknown" else platform
