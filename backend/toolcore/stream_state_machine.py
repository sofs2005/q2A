from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from backend.toolcore.stream_sieve import ToolStreamSieve, looks_like_tool_fragment


@dataclass(slots=True)
class ToolStreamEvent:
    type: str
    text: str | None = None
    calls: list[dict[str, Any]] | None = None


class ToolStreamStateMachine:
    def __init__(self, tool_names: list[str]) -> None:
        self.tool_names = [name for name in tool_names if isinstance(name, str) and name]
        self._sieve = ToolStreamSieve(self.tool_names)
        self._saw_tool_call = False
        self._call_sequence = 0

    def _attach_call_ids(self, calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        normalized_calls: list[dict[str, Any]] = []
        for call in calls or []:
            if not isinstance(call, dict):
                continue
            enriched = dict(call)
            if not enriched.get("id"):
                self._call_sequence += 1
                enriched["id"] = f"toolu_{uuid.uuid4().hex[:8]}_{self._call_sequence}"
            normalized_calls.append(enriched)
        return normalized_calls

    def process_text_delta(self, delta: str) -> list[ToolStreamEvent]:
        events: list[ToolStreamEvent] = []
        for event in self._sieve.process_chunk(delta):
            if event.get("type") == "tool_calls":
                self._saw_tool_call = True
                events.append(ToolStreamEvent(type="tool_calls", calls=self._attach_call_ids(event.get("calls", []))))
            elif event.get("type") == "content":
                events.append(ToolStreamEvent(type="content", text=event.get("text", "")))
        return events

    def process_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolStreamEvent]:
        if tool_calls:
            self._saw_tool_call = True
            return [ToolStreamEvent(type="tool_calls", calls=self._attach_call_ids(tool_calls))]
        return []

    def reset_attempt(self) -> None:
        self._sieve = ToolStreamSieve(self.tool_names)
        self._saw_tool_call = False

    def flush(self, *, final_tool_use: bool) -> list[ToolStreamEvent]:
        events: list[ToolStreamEvent] = []
        for event in self._sieve.flush():
            if event.get("type") == "tool_calls":
                self._saw_tool_call = True
                events.append(ToolStreamEvent(type="tool_calls", calls=self._attach_call_ids(event.get("calls", []))))
                continue

            text = event.get("text", "")
            if final_tool_use and text and looks_like_tool_fragment(text):
                continue
            events.append(ToolStreamEvent(type="content", text=text))
        return events
