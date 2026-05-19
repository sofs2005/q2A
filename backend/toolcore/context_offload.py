from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE
from backend.toolcore.prompt_contract import model_bridge_tool_name, normalize_prompt_tool

SYSTEM_CONTEXT_FILE_PREFIX = "qwen2api_context"
SYSTEM_CONTEXT_PROMPT_NOTE = (
    "Generated system context files may be attached with opaque filenames. "
    "Use them as supporting context. User-uploaded files are separate user inputs and should also be respected."
)
TOOLS_CONTEXT_FILE_PREFIX = "qwen2api_tools"
TOOLS_CONTEXT_TITLE = "# QWEN2API_TOOLS.txt"
TOOLS_CONTEXT_SUMMARY = "Available tool descriptions and parameter schemas for this request."


@dataclass(slots=True)
class LocalContextFile:
    filename: str
    ext: str
    content_type: str
    text: str
    sha256: str
    purpose: str = "context"
    local_path: str = ""


@dataclass(slots=True)
class ContextOffloadPlan:
    mode: str
    inline_messages: list[dict[str, Any]]
    generated_files: list[LocalContextFile] = field(default_factory=list)
    summary_text: str = ""
    estimated_prompt_len: int = 0
    note: str = ""


class ContextOffloader:
    def __init__(self, settings):
        self.settings = settings

    def estimate_prompt_len(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, client_profile: str = "") -> int:
        total = 0
        for msg in messages or []:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += len(str(part.get("text", "")))
                        total += len(str(part.get("content", "")))
            total += 24
        total += sum(len(str(tool.get("name", ""))) + len(str(tool.get("description", ""))) for tool in (tools or []))
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
            total += 512
        return total

    def _extract_text(self, msg: dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
                    elif part.get("type") == "tool_result":
                        chunks.append(str(part.get("content", "")))
            return "\n".join(chunk for chunk in chunks if chunk)
        return str(content)

    def _make_file(self, base_name: str, ext: str, text: str, content_type: str) -> LocalContextFile:
        data = text.encode("utf-8")
        return LocalContextFile(
            filename=f"{base_name}.{ext}",
            ext=ext,
            content_type=content_type,
            text=text,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def _tools_context_text(self, tools: list[dict[str, Any]] | None) -> str:
        tool_blocks: list[str] = []
        for raw_tool in tools or []:
            if not isinstance(raw_tool, dict):
                continue
            tool = normalize_prompt_tool(raw_tool)
            raw_name = str(tool.get("name") or "").strip()
            if not raw_name:
                continue
            name = raw_name if raw_name.startswith("bridge-") else model_bridge_tool_name(len(tool_blocks))
            description = str(tool.get("description") or "No description available")
            parameters = tool.get("parameters") or {}
            schema = json.dumps(parameters if isinstance(parameters, dict) else {}, ensure_ascii=False)
            tool_blocks.append(f"Tool: {name}\nDescription: {description}\nParameters: {schema}")
        if not tool_blocks:
            return ""
        return "\n".join([TOOLS_CONTEXT_TITLE, TOOLS_CONTEXT_SUMMARY, "", "\n\n".join(tool_blocks), ""])

    def plan(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, client_profile: str = "") -> ContextOffloadPlan:
        estimated = self.estimate_prompt_len(messages, tools=tools, client_profile=client_profile)
        history_estimated = self.estimate_prompt_len(messages, tools=[], client_profile=client_profile)
        tools_text = self._tools_context_text(tools)
        history_needs_file = history_estimated > self.settings.CONTEXT_INLINE_MAX_CHARS
        if not history_needs_file and not tools_text:
            return ContextOffloadPlan(mode="inline", inline_messages=messages, estimated_prompt_len=estimated)

        user_messages = [message for message in messages if message.get("role") == "user"]
        latest_user = user_messages[-1] if user_messages else {"role": "user", "content": ""}
        latest_user_text = self._extract_text(latest_user)

        serialized_parts: list[str] = []
        if history_needs_file:
            for idx, msg in enumerate(messages or [], 1):
                role = msg.get("role", "unknown")
                text = self._extract_text(msg)
                if not text.strip():
                    continue
                serialized_parts.append(f"## Message {idx} [{role}]\n{text.strip()}\n")
        attachment_text = "\n".join(serialized_parts).strip()
        summary_text = attachment_text[:1200] if attachment_text else ""

        if history_estimated <= self.settings.CONTEXT_FORCE_FILE_MAX_CHARS:
            mode = "hybrid"
        else:
            mode = "file"

        generated_files: list[LocalContextFile] = []
        if attachment_text:
            generated_files.append(
                self._make_file(
                    f"{SYSTEM_CONTEXT_FILE_PREFIX}_history",
                    "txt",
                    attachment_text,
                    "text/plain",
                )
            )
        if tools_text:
            generated_files.append(
                self._make_file(
                    TOOLS_CONTEXT_FILE_PREFIX,
                    "txt",
                    tools_text,
                    "text/plain",
                )
            )

        rewritten_messages = [{"role": "user", "content": SYSTEM_CONTEXT_PROMPT_NOTE}]
        if latest_user_text.strip():
            rewritten_messages.append({"role": "user", "content": latest_user_text.strip()})

        return ContextOffloadPlan(
            mode=mode,
            inline_messages=rewritten_messages,
            generated_files=generated_files,
            summary_text=summary_text,
            estimated_prompt_len=estimated,
            note=SYSTEM_CONTEXT_PROMPT_NOTE,
        )
