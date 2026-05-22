from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolChoicePolicy(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"
    FORCED = "forced"


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    client_name: str | None = None
    model_name: str | None = None
    aliases: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class CanonicalToolCall:
    call_id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalToolResult:
    call_id: str
    output: Any
    tool_name: str | None = None


@dataclass(slots=True)
class ToolCoreRequest:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice_policy: ToolChoicePolicy = ToolChoicePolicy.AUTO
    forced_tool_name: str | None = None
    tool_calls: list[CanonicalToolCall] = field(default_factory=list)
    tool_results: list[CanonicalToolResult] = field(default_factory=list)
    raw_tool_choice: Any | None = None
    tool_catalog: Any | None = None
    skill_catalog: Any | None = None
