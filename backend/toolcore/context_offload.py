from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE
from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE, sanitize_openclaw_user_text, strip_openclaw_untrusted_metadata
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

    def _split_openclaw_user_context(self, text: str) -> tuple[str, str]:
        cleaned = text.strip()
        if not cleaned:
            return "", ""
        cleaned = strip_openclaw_untrusted_metadata(cleaned)
        if not cleaned:
            return "", ""
        if not cleaned.startswith(("## Memory Recall", "## Compiled Wiki")):
            return "", sanitize_openclaw_user_text(cleaned).strip()
        parts = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        context_parts: list[str] = []
        while parts and parts[0].startswith(("## Memory Recall", "## Compiled Wiki")):
            context_parts.append(parts.pop(0))
        task_text = sanitize_openclaw_user_text("\n\n".join(parts)).strip()
        return "\n\n".join(context_parts).strip(), task_text

    def _latest_user_parts(self, messages: list[dict[str, Any]], *, client_profile: str) -> tuple[str, str]:
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = self._extract_text(message).strip()
            if client_profile == OPENCLAW_OPENAI_PROFILE:
                context_text, task_text = self._split_openclaw_user_context(text)
            else:
                context_text, task_text = "", text
            if task_text:
                return context_text, task_text
        return "", ""

    def _is_tool_result_message(self, message: dict[str, Any]) -> bool:
        if message.get("role") in {"tool", "function"}:
            return True
        content = message.get("content")
        return isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "tool_result"
            for part in content
        )

    def _recent_tool_continuation_tail(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = [message for message in messages or [] if isinstance(message, dict)]
        for index in range(len(items) - 1, -1, -1):
            message = items[index]
            if message.get("role") != "assistant" or not isinstance(message.get("tool_calls"), list) or not message.get("tool_calls"):
                continue
            tail = items[index:]
            if any(item.get("role") == "user" for item in tail[1:]):
                return []
            if any(self._is_tool_result_message(item) for item in tail[1:]):
                return tail
            return []
        return []

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

    def _calculate_dynamic_inline_threshold(self, total_chars: int) -> int:
        """Calculate a dynamic inline threshold that scales with conversation size.

        Uses deterministic hash-based jitter instead of random jitter to ensure
        identical inputs always produce identical offload decisions.
        """
        base_threshold = 8000
        max_threshold = self.settings.CONTEXT_INLINE_MAX_CHARS

        if total_chars <= 10000:
            base = base_threshold
        elif total_chars >= 50000:
            base = max_threshold
        else:
            scale = (total_chars - 10000) / (50000 - 10000)
            base = int(base_threshold + scale * (max_threshold - base_threshold))

        # Deterministic jitter based on content size hash (+/- 15%)
        jitter_hash = int(hashlib.md5(str(total_chars).encode()).hexdigest()[:8], 16)
        jitter_range = max(1, int(base * 0.15))
        jitter = (jitter_hash % (2 * jitter_range + 1)) - jitter_range
        result = base + jitter

        return max(base_threshold, min(max_threshold, result))

    def plan(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, client_profile: str = "") -> ContextOffloadPlan:
        estimated = self.estimate_prompt_len(messages, tools=tools, client_profile=client_profile)
        history_estimated = self.estimate_prompt_len(messages, tools=[], client_profile=client_profile)
        tools_text = self._tools_context_text(tools)
        dynamic_threshold = self._calculate_dynamic_inline_threshold(history_estimated)
        history_needs_file = history_estimated > dynamic_threshold
        if not history_needs_file and not tools_text:
            return ContextOffloadPlan(mode="inline", inline_messages=messages, estimated_prompt_len=estimated)

        latest_user_context, latest_user_text = self._latest_user_parts(messages, client_profile=client_profile)
        tool_continuation_tail = self._recent_tool_continuation_tail(messages)

        serialized_parts: list[str] = []
        if history_needs_file:
            for idx, msg in enumerate(messages or [], 1):
                role = msg.get("role", "unknown")
                text = self._extract_text(msg)
                if not text.strip():
                    continue
                serialized_parts.append(f"## Message {idx} [{role}]\n{text.strip()}\n")
        attachment_text = "\n".join(serialized_parts).strip()
        summary_text = attachment_text if attachment_text else ""

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
        if latest_user_context:
            rewritten_messages.append({"role": "user", "content": latest_user_context})
        if latest_user_text.strip():
            rewritten_messages.append({"role": "user", "content": latest_user_text.strip()})
        rewritten_messages.extend(dict(message) for message in tool_continuation_tail)

        return ContextOffloadPlan(
            mode=mode,
            inline_messages=rewritten_messages,
            generated_files=generated_files,
            summary_text=summary_text,
            estimated_prompt_len=estimated,
            note=SYSTEM_CONTEXT_PROMPT_NOTE,
        )
