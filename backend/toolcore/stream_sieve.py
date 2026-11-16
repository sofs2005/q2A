from __future__ import annotations

import asyncio
import functools
import re
from typing import Any

from backend.services.tool_parser import parse_tool_calls_silent
from backend.toolcall.formats_dsml import consume_dsml_tool_capture, has_open_dsml_tool_tag
from backend.toolcall.markup_scan import find_partial_tool_markup_start, find_tool_markup_tag_outside_ignored


TOOL_START_MARKERS = ('{"name":', '<tool_call>', '##tool_call##', 'tool_call##', 'function.name:')
LEGACY_HOLD_CHARS = max(len(marker) for marker in TOOL_START_MARKERS) - 1
MAX_CAPTURE_CHARS = 64 * 1024
FENCE_PASSTHROUGH_HOLD_CHARS = 32

# --- Capture-phase throttling -------------------------------------------------
# Re-parsing the whole capture buffer on every delta is O(n^2) in the length of
# the tool-call payload (16 KB of arguments arrives as ~2000 deltas). The sieve
# therefore only runs the expensive ``_consume_capture`` when the buffer *may*
# have become parseable:
#   * it is still small (cheap to parse, keeps small calls bit-for-bit identical)
#   * a closing marker showed up in the not-yet-scanned part of the buffer
#   * the buffer grew a lot since the last attempt (safety net / progress)
CAPTURE_ALWAYS_PARSE_CHARS = 512
CAPTURE_REPARSE_GROWTH = 512
CAPTURE_PROBE_OVERLAP = 32

# Below this many buffered characters the sieve always runs inline: a
# ``asyncio.to_thread`` hop costs more than the parse itself.
SIEVE_OFFLOAD_MIN_CHARS = 2048

# Closing markers that can make a capture parseable.  The DSML entries are a
# superset of ``_has_tool_calls_close_hint`` (they also cover the ``toolcall`` /
# ``tool-calls`` spellings accepted by ``markup_scan._canonicalize_tag_name``).
DSML_PROBE_HINTS = (
    "</|dsml|tool_calls",
    "<|/dsml|tool_calls",
    "<|/dsml|/tool_calls",
    "</tool_calls",
    "</|dsml|toolcall",
    "</|dsml|tool-calls",
    "<|/dsml|toolcall",
)
DSML_LIKE_MARKERS = ("<|dsml|", "</|dsml|", "<|/dsml|")
KV_LIKE_MARKERS = ("function.name:", "function.arguments:")


def _compact_for_probe(text: str) -> str:
    """Normalise *text* exactly the way ``_has_tool_calls_close_hint`` does.

    ``fold``, ``lower`` and whitespace removal are all per-character
    operations, so ``_compact_for_probe(a + b) == _compact_for_probe(a) +
    _compact_for_probe(b)`` — which lets the sieve maintain it incrementally.
    """
    folded = text.translate(DSML_CLOSE_HINT_TRANSLATION).replace("\x02", "|").lower()
    if not folded:
        return ""
    return "".join(ch for ch in folded if not ch.isspace())


class _CaptureProbe:
    """Incremental, O(delta)-per-chunk "is the capture parseable yet?" probe.

    Tracks the derived views of ``ToolStreamSieve.capture`` that closing-marker
    detection needs, plus how far those views have already been scanned, so a
    delta is never re-examined by the next probe.
    """

    __slots__ = (
        "_compact",
        "_lower",
        "_length",
        "_scan_pos",
        "_compact_scan_pos",
        "_last_parse_len",
        "_dsml_like",
        "_json_like",
        "_kv_like",
        "_markup_like",
        "_force",
    )

    def __init__(self) -> None:
        self._compact = ""
        self._lower = ""
        self._length = 0
        self._scan_pos = 0
        self._compact_scan_pos = 0
        self._last_parse_len = 0
        self._dsml_like = False
        self._json_like = False
        self._kv_like = False
        self._markup_like = False
        self._force = True

    def reset(self, capture: str = "") -> None:
        """Forget all state; the next probe attempt always parses once."""
        self._compact = ""
        self._lower = ""
        self._length = 0
        self._scan_pos = 0
        self._compact_scan_pos = 0
        self._last_parse_len = 0
        self._dsml_like = False
        self._json_like = False
        self._kv_like = False
        self._markup_like = False
        self._force = True
        if capture:
            self.append(capture)

    def append(self, delta: str) -> None:
        """Absorb *delta* (the newest slice of the capture buffer)."""
        if not delta:
            return

        if self._length == 0:
            head = delta.lstrip()
            if head:
                first = head[0]
                folded_first = first.translate(DSML_CLOSE_HINT_TRANSLATION)
                if folded_first == "<":
                    self._markup_like = True
                elif first == "{":
                    self._json_like = True

        self._length += len(delta)
        self._compact += _compact_for_probe(delta)
        self._lower += delta.lower()

        if not self._dsml_like:
            window = self._compact[max(0, len(self._compact) - len(delta) - CAPTURE_PROBE_OVERLAP):]
            if any(marker in window for marker in DSML_LIKE_MARKERS):
                self._dsml_like = True
                self._markup_like = True
        if not self._kv_like:
            window = self._lower[max(0, len(self._lower) - len(delta) - CAPTURE_PROBE_OVERLAP):]
            if any(marker in window for marker in KV_LIKE_MARKERS):
                self._kv_like = True

    def should_parse(self, capture: str) -> bool:
        """Return True when ``_consume_capture`` is worth running for *capture*.

        Consumes the pending "must parse" flag, so call it exactly once per
        delta.
        """
        forced = self._force
        self._force = False
        return forced or self.would_parse(capture)

    def would_parse(self, capture: str, pending_delta: str = "") -> bool:
        """Pure predicate: would :meth:`should_parse` run for this capture?

        *pending_delta* is treated as if it had already been appended, which
        lets callers decide whether to offload the work before mutating state.
        """
        if self._force:
            return True

        length = len(capture) + len(pending_delta)
        if length > MAX_CAPTURE_CHARS:
            return True
        if length <= CAPTURE_ALWAYS_PARSE_CHARS:
            return True
        if length - self._last_parse_len >= CAPTURE_REPARSE_GROWTH:
            return True

        delta_lower = pending_delta.lower() if pending_delta else ""
        lower_window = max(0, self._scan_pos - CAPTURE_PROBE_OVERLAP)
        lower_tail = self._lower[lower_window:] + delta_lower
        if "</tool_call" in lower_tail or "##end_call##" in lower_tail:
            return True

        compact_window = max(0, self._compact_scan_pos - CAPTURE_PROBE_OVERLAP)
        compact_tail = self._compact[compact_window:]
        if pending_delta:
            compact_tail += _compact_for_probe(pending_delta)
        if any(hint in compact_tail for hint in DSML_PROBE_HINTS):
            return True

        if self._json_like and ("}" in lower_tail or "]" in lower_tail):
            return True
        if self._kv_like and "\n" in lower_tail:
            return True
        if self._markup_like and ">" in lower_tail:
            return True
        return False

    def mark_parsed(self, capture: str) -> None:
        """Record that *capture* was just parsed (advance the scan cursors)."""
        self._last_parse_len = len(capture)
        self._scan_pos = len(capture)
        self._compact_scan_pos = len(self._compact)

FENCE_OPEN_RE = re.compile(r"(?m)^[ \t]*(```+|~~~+)[^\n]*(?:\n|$)")
DSML_CONTROL_START_RE = re.compile(
    r"[<＜﹤〈]\s*/?\s*[|！、␂]\s*/?\s*DSML\s*[|！、␂]",
    re.IGNORECASE,
)
DSML_CLOSE_HINT_TRANSLATION = str.maketrans({
    "＜": "<",
    "﹤": "<",
    "〈": "<",
    "＞": ">",
    "﹥": ">",
    "〉": ">",
    "／": "/",
    "∕": "/",
    "！": "|",
    "、": "|",
    "␂": "|",
})


def _inside_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _has_tool_calls_close_hint(text: str) -> bool:
    folded = text.translate(DSML_CLOSE_HINT_TRANSLATION).replace("\x02", "|").lower()
    compact = "".join(ch for ch in folded if not ch.isspace())
    return any(
        hint in compact
        for hint in (
            "</|dsml|tool_calls",
            "<|/dsml|tool_calls",
            "<|/dsml|/tool_calls",
            "</tool_calls",
        )
    )


def _is_dsml_control_prefix_fragment(text: str) -> bool:
    folded = text.translate(DSML_CLOSE_HINT_TRANSLATION).replace("\x02", "|").lower()
    compact = "".join(ch for ch in folded if not ch.isspace())
    return bool(compact) and any(marker.startswith(compact) for marker in ("<|dsml|", "</|dsml|", "<|/dsml|"))


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
    length = len(text)
    while pos < length:
        # Jump straight to the next backtick instead of probing every char.
        pos = text.find("`", pos)
        if pos < 0:
            break
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
                spans.append((pos, length))
                break
            spans.append((pos, end + 2))
            pos = end + 2
            continue
        end = text.find("`", pos + 1)
        if end == -1 or "\n" in text[pos:end + 1]:
            spans.append((pos, length))
            break
        spans.append((pos, end + 1))
        pos = end + 1
    return spans


def _fence_close_re(fence: str):
    return re.compile(rf"(?m)^[ \t]*{re.escape(fence)}[ \t]*(?:\n|$)")


def _unclosed_fenced_code(text: str) -> tuple[int, str] | None:
    pos = 0
    while True:
        opener = FENCE_OPEN_RE.search(text, pos)
        if opener is None:
            return None
        fence = opener.group(1)
        closer = _fence_close_re(fence).search(text, opener.end())
        if closer is None:
            return opener.start(), fence
        pos = closer.end()


def _unclosed_markdown_code_start(text: str) -> int:
    fenced = _unclosed_fenced_code(text)
    if fenced is not None:
        return fenced[0]

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


def _find_legacy_tool_start(text: str, ignored: list[tuple[int, int]] | None = None) -> int:
    lowered = text.lower()
    if ignored is None:
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
        self.dropping_dsml_remainder = False
        self.fenced_passthrough_fence: str | None = None
        self._probe = _CaptureProbe()

    def _drop_overlong_capture(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply the ``MAX_CAPTURE_CHARS`` overflow policy to the capture buffer."""
        if len(self.capture) <= MAX_CAPTURE_CHARS:
            return events
        if self._is_dsml_capture(self.capture):
            self.dropping_dsml_remainder = True
        else:
            events.append({"type": "content", "text": self.capture})
        self.capture = ""
        self.capturing = False
        self._probe.reset()
        return events

    def process_chunk(self, chunk: str) -> list[dict[str, Any]]:
        if not chunk:
            return []
        if self.dropping_dsml_remainder:
            return []

        self.pending += chunk
        return self._drain_pending()

    def _offload_needed(self, chunk: str) -> bool:
        """Whether *chunk* will trigger work heavy enough to leave the loop."""
        buffered = len(self.pending) + len(chunk)
        if not self.capturing:
            # Outside a capture the cost is bounded by the buffered text.
            return buffered >= SIEVE_OFFLOAD_MIN_CHARS
        if len(self.capture) + buffered < SIEVE_OFFLOAD_MIN_CHARS:
            return False
        return self._probe.would_parse(self.capture, self.pending + chunk)

    async def process_chunk_async(self, chunk: str) -> list[dict[str, Any]]:
        """Awaitable :meth:`process_chunk` that keeps heavy parses off the loop.

        A sieve instance is stateful, so callers must await one call at a time
        (the streaming loop in ``runtime.execution`` does exactly that).
        """
        if self._offload_needed(chunk):
            return await asyncio.to_thread(self.process_chunk, chunk)
        return self.process_chunk(chunk)

    async def flush_async(self) -> list[dict[str, Any]]:
        """Awaitable :meth:`flush`; offloads only when the tail is big."""
        if len(self.capture) + len(self.pending) >= SIEVE_OFFLOAD_MIN_CHARS:
            return await asyncio.to_thread(self.flush)
        return self.flush()

    def _drain_pending(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        while True:
            if self.fenced_passthrough_fence is not None:
                self._drain_fenced_passthrough(events)
                if self.fenced_passthrough_fence is not None or not self.pending:
                    return events
                continue

            if self.capturing:
                delta = self.pending
                self.capture += delta
                self.pending = ""
                if delta:
                    self._probe.append(delta)
                if not self._probe.should_parse(self.capture):
                    # Throttled: nothing can have become parseable yet, so the
                    # whole-capture re-parse is skipped for this delta.
                    return self._drop_overlong_capture(events)
                prefix, calls, suffix, ready = self._consume_capture()
                if not ready:
                    self._probe.mark_parsed(self.capture)
                    return self._drop_overlong_capture(events)
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                self.pending = suffix
                self.capture = ""
                self.capturing = False
                self._probe.reset()
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
                self._probe.reset(self.capture)
                continue

            fenced = _unclosed_fenced_code(self.pending)
            if fenced is not None:
                code_start, fence = fenced
                partial_start = find_partial_tool_markup_start(self.pending[:code_start])
                if partial_start >= 0:
                    safe = self.pending[:partial_start]
                    if safe:
                        events.append({"type": "content", "text": safe})
                    self.pending = self.pending[partial_start:]
                    return events
                if code_start > 0:
                    events.append({"type": "content", "text": self.pending[:code_start]})
                    self.pending = self.pending[code_start:]
                self.fenced_passthrough_fence = fence
                continue

            safe, hold = self._split_safe_content(self.pending)
            if safe:
                events.append({"type": "content", "text": safe})
            self.pending = hold
            return events

    def _drain_fenced_passthrough(self, events: list[dict[str, Any]]) -> None:
        fence = self.fenced_passthrough_fence
        if fence is None:
            return

        closer = _fence_close_re(fence).search(self.pending)
        if closer is not None:
            events.append({"type": "content", "text": self.pending[:closer.end()]})
            self.pending = self.pending[closer.end():]
            self.fenced_passthrough_fence = None
            return

        hold_chars = max(FENCE_PASSTHROUGH_HOLD_CHARS, len(fence) + 4)
        if len(self.pending) > hold_chars:
            events.append({"type": "content", "text": self.pending[:-hold_chars]})
            self.pending = self.pending[-hold_chars:]

    def flush(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        drop_pending_dsml_suffix = False
        if self.capturing and self.capture:
            is_dsml_capture = self._is_dsml_capture(self.capture)
            prefix, calls, suffix, ready = self._consume_capture()
            if ready:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                if calls:
                    events.append({"type": "tool_calls", "calls": calls})
                if suffix:
                    partial_suffix_start = find_partial_tool_markup_start(suffix)
                    if is_dsml_capture and (
                        (partial_suffix_start >= 0 and not suffix[:partial_suffix_start].strip())
                        or _is_dsml_control_prefix_fragment(suffix)
                    ):
                        safe_suffix = suffix[:partial_suffix_start] if partial_suffix_start >= 0 else ""
                        if safe_suffix.strip():
                            events.append({"type": "content", "text": safe_suffix})
                        drop_pending_dsml_suffix = True
                    else:
                        events.append({"type": "content", "text": suffix})
            elif not is_dsml_capture:
                events.append({"type": "content", "text": self.capture})
            else:
                drop_pending_dsml_suffix = True
            self.capture = ""
            self.capturing = False
        if self.pending:
            partial_pending_start = find_partial_tool_markup_start(self.pending)
            drop_pending = drop_pending_dsml_suffix and (
                (partial_pending_start >= 0 and not self.pending[:partial_pending_start].strip())
                or _is_dsml_control_prefix_fragment(self.pending)
            )
            if not drop_pending:
                events.append({"type": "content", "text": self.pending})
            self.pending = ""
        self.fenced_passthrough_fence = None
        self.dropping_dsml_remainder = False
        self._probe.reset()
        return events

    def _is_dsml_capture(self, text: str) -> bool:
        folded = text.translate(DSML_CLOSE_HINT_TRANSLATION).replace("\x02", "|").lower()
        compact = "".join(ch for ch in folded if not ch.isspace())
        return any(marker in compact for marker in ("<|dsml|", "</|dsml|", "<|/dsml|"))

    def _find_tool_start(self, text: str) -> int:
        positions: list[int] = []
        unclosed_code_start = _unclosed_markdown_code_start(text)
        scan_text = text[:unclosed_code_start] if unclosed_code_start >= 0 else text
        tag = find_tool_markup_tag_outside_ignored(scan_text, 0)
        while tag is not None:
            if not tag.closing and tag.name in {"tool_calls", "invoke", "parameter"}:
                positions.append(tag.start)
                break
            tag = find_tool_markup_tag_outside_ignored(scan_text, tag.end)

        ignored = _markdown_code_spans(scan_text)
        dsml_start = DSML_CONTROL_START_RE.search(scan_text)
        while dsml_start is not None:
            if not _inside_spans(dsml_start.start(), ignored):
                positions.append(dsml_start.start())
                break
            dsml_start = DSML_CONTROL_START_RE.search(scan_text, dsml_start.end())

        legacy_start = _find_legacy_tool_start(scan_text, ignored)
        if legacy_start >= 0:
            positions.append(legacy_start)
        return min(positions) if positions else -1

    def _consume_capture(self) -> tuple[str, list[dict[str, Any]], str, bool]:
        if not self.capture:
            return "", [], "", False

        allowed_names = set(self.tool_names)
        first_tag = find_tool_markup_tag_outside_ignored(self.capture, 0)
        if first_tag is not None and first_tag.name == "tool_calls":
            if not _has_tool_calls_close_hint(self.capture):
                return "", [], "", False
            prefix, calls, suffix, ready = consume_dsml_tool_capture(self.capture, allowed_names)
            if not ready:
                return "", [], "", False
            if not calls:
                return "", [], suffix, True
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
