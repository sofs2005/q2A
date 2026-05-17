from __future__ import annotations

import re
from typing import Any

from backend.services.tool_parser import parse_tool_calls_silent
from backend.toolcall.formats_dsml import consume_dsml_tool_capture, has_open_dsml_tool_tag
from backend.toolcall.markup_scan import find_partial_tool_markup_start, find_tool_markup_tag_outside_ignored


TOOL_START_MARKERS = ('{"name":', '<tool_call>', '##tool_call##', 'tool_call##', 'function.name:')
LEGACY_HOLD_CHARS = max(len(marker) for marker in TOOL_START_MARKERS) - 1
FENCE_OPEN_RE = re.compile(r"(?m)^[ \t]*(```+|~~~+)[^\n]*(?:\n|$)")


def _inside_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _markdown_code_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        opener = FENCE_OPEN_RE.search(text, pos)
        if opener is None:
            break
        fence = opener.group(1)
        closer = re.compile(rf"(?m)^[ \t]*{re.escape(fence)}[ \t]*(?:\n|$)").search(text, opener.end())
        if closer is None:
            spans.append((opener.start(), len(text)))
            break
        spans.append((opener.start(), closer.end()))
        pos = closer.end()

    fence_spans = list(spans)
    pos = 0
    while pos < len(text):
        skipped = False
        for start, end in fence_spans:
            if start <= pos < end:
                pos = end
                skipped = True
                break
        if skipped:
            continue
        if text.startswith("``", pos):
            end = text.find("``", pos + 2)
            if end == -1 or "\n" in text[pos:end + 2]:
                spans.append((pos, len(text)))
                break
            spans.append((pos, end + 2))
            pos = end + 2
            continue
        elif text[pos] == "`":
            end = text.find("`", pos + 1)
            if end == -1 or "\n" in text[pos:end + 1]:
                spans.append((pos, len(text)))
                break
            spans.append((pos, end + 1))
            pos = end + 1
            continue
        pos += 1
    return spans


def _unclosed_markdown_code_start(text: str) -> int:
    pos = 0
    while True:
        opener = FENCE_OPEN_RE.search(text, pos)
        if opener is None:
            break
        fence = opener.group(1)
        closer = re.compile(rf"(?m)^[ \t]*{re.escape(fence)}[ \t]*(?:\n|$)").search(text, opener.end())
        if closer is None:
            return opener.start()
        pos = closer.end()

    pos = text.rfind("\n") + 1
    while pos < len(text):
        if text.startswith("``", pos):
            end = text.find("``", pos + 2)
            if end == -1 or "\n" in text[pos:end + 2]:
                return pos
            pos = end + 2
            continue
        if text[pos] == "`":
            end = text.find("`", pos + 1)
            if end == -1 or "\n" in text[pos:end + 1]:
                return pos
            pos = end + 1
            continue
        pos += 1
    return -1


def _find_legacy_tool_start(text: str) -> int:
    lowered = text.lower()
    ignored = _markdown_code_spans(text)
    positions: list[int] = []
    for marker in TOOL_START_MARKERS:
        pos = lowered.find(marker)
        while pos >= 0:
            if not _inside_spans(pos, ignored):
                positions.append(pos)
                break
            pos = lowered.find(marker, pos + 1)
    return min(positions) if positions else -1


def looks_like_tool_fragment(text: str) -> bool:
    text = text or ""
    tag = find_tool_markup_tag_outside_ignored(text, 0)
    if tag is not None or has_open_dsml_tool_tag(text) or find_partial_tool_markup_start(text) >= 0:
        return True
    return _find_legacy_tool_start(text) >= 0


class ToolStreamSieve:
    def __init__(self, tool_names: list[str]):
        self.tool_names = [name for name in tool_names if isinstance(name, str) and name]
        self.pending = ""
        self.capture = ""
        self.capturing = False

    def process_chunk(self, chunk: str) -> list[dict[str, Any]]:
        if not chunk:
            return []

        self.pending += chunk
        return self._drain_pending()

    def _drain_pending(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        while True:
            if self.capturing:
                self.capture += self.pending
                self.pending = ""
                prefix, calls, suffix, ready = self._consume_capture()
                if not ready:
                    return events
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                self.pending = suffix
                self.capture = ""
                self.capturing = False
                if not self.pending:
                    return events
                continue

            start = self._find_tool_start(self.pending)
            if start >= 0:
                prefix = self.pending[:start]
                if prefix:
                    events.append({"type": "content", "text": prefix})
                self.capture = self.pending[start:]
                self.pending = ""
                self.capturing = True
                continue

            safe, hold = self._split_safe_content(self.pending)
            if safe:
                events.append({"type": "content", "text": safe})
            self.pending = hold
            return events

    def flush(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.capturing and self.capture:
            prefix, calls, suffix, ready = self._consume_capture()
            if ready:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                if suffix:
                    events.append({"type": "content", "text": suffix})
            elif not self._is_dsml_capture(self.capture):
                events.append({"type": "content", "text": self.capture})
            self.capture = ""
            self.capturing = False
        if self.pending:
            events.append({"type": "content", "text": self.pending})
            self.pending = ""
        return events

    def _is_dsml_capture(self, text: str) -> bool:
        return "<|DSML|" in text or "＜！DSML！" in text

    def _find_tool_start(self, text: str) -> int:
        positions: list[int] = []
        unclosed_code_start = _unclosed_markdown_code_start(text)
        scan_text = text[:unclosed_code_start] if unclosed_code_start >= 0 else text
        tag = find_tool_markup_tag_outside_ignored(scan_text, 0)
        while tag is not None:
            if not tag.closing and tag.name in {"tool_calls", "invoke"}:
                positions.append(tag.start)
                break
            tag = find_tool_markup_tag_outside_ignored(scan_text, tag.end)

        legacy_start = _find_legacy_tool_start(text)
        if legacy_start >= 0:
            positions.append(legacy_start)
        return min(positions) if positions else -1

    def _consume_capture(self) -> tuple[str, list[dict[str, Any]], str, bool]:
        if not self.capture:
            return "", [], "", False

        allowed_names = set(self.tool_names)
        first_tag = find_tool_markup_tag_outside_ignored(self.capture, 0)
        if first_tag is not None and first_tag.name == "tool_calls":
            prefix, calls, suffix, ready = consume_dsml_tool_capture(self.capture, allowed_names)
            if not ready or not calls:
                return "", [], "", False
            return prefix, calls, suffix, True

        lowered = self.capture.lower()
        if "##tool_call##" in lowered and "##end_call##" not in lowered:
            return "", [], "", False
        if "<tool_call>" in lowered and "</tool_call>" not in lowered:
            return "", [], "", False

        blocks, stop_reason = parse_tool_calls_silent(self.capture, [{"name": name} for name in self.tool_names])
        if stop_reason == "tool_use":
            tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
            text_blocks = [block for block in blocks if block.get("type") == "text"]
            prefix = text_blocks[0].get("text", "") if text_blocks else ""
            calls = [{"name": block["name"], "input": block.get("input", {})} for block in tool_blocks]
            if calls:
                return prefix, calls, "", True
        if looks_like_tool_fragment(self.capture):
            return "", [], "", False
        return self.capture, [], "", True

    def _split_safe_content(self, text: str) -> tuple[str, str]:
        hold_starts = [pos for pos in (find_partial_tool_markup_start(text), _unclosed_markdown_code_start(text)) if pos >= 0]
        if hold_starts:
            start = min(hold_starts)
            return text[:start], text[start:]
        if len(text) <= LEGACY_HOLD_CHARS:
            return "", text
        return text[:-LEGACY_HOLD_CHARS], text[-LEGACY_HOLD_CHARS:]
