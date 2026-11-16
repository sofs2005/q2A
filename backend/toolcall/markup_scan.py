from __future__ import annotations

import bisect
import re
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolMarkupTag:
    """Represents a DSML/XML tool markup tag found in text."""

    name: str
    """Canonical tag name: tool_calls, invoke, or parameter."""

    start: int
    """Start index of the tag in the original text (inclusive)."""

    end: int
    """End index of the tag in the original text (exclusive)."""

    closing: bool
    """True if this is a closing tag (e.g. </|DSML|tool_calls>)."""

    raw_name: str = ""
    """The original tag name as found in the text, before canonicalization."""


# ---------------------------------------------------------------------------
# Fullwidth / CJK character folding
# ---------------------------------------------------------------------------

_FULLWIDTH_TABLE = str.maketrans({
    '＜': '<',   # ＜ FULLWIDTH LESS-THAN SIGN
    '﹤': '<',   # ﹤ SMALL LESS-THAN SIGN
    '〈': '<',   # 〈 LEFT ANGLE BRACKET
    '＞': '>',   # ＞ FULLWIDTH GREATER-THAN SIGN
    '﹥': '>',   # ﹥ SMALL GREATER-THAN SIGN
    '〉': '>',   # 〉 RIGHT ANGLE BRACKET
    '／': '/',   # ／ FULLWIDTH SOLIDUS
    '∕': '/',   # ∕ DIVISION SLASH
    '＝': '=',   # ＝ FULLWIDTH EQUALS SIGN
    '“': '"',   # “ LEFT DOUBLE QUOTATION MARK
    '”': '"',   # ” RIGHT DOUBLE QUOTATION MARK
    '‘': "'",   # ‘ LEFT SINGLE QUOTATION MARK
    '’': "'",   # ’ RIGHT SINGLE QUOTATION MARK
    '！': '|',   # ！ FULLWIDTH EXCLAMATION MARK
    '、': '|',   # 、 IDEOGRAPHIC COMMA
    '␂': '|',   # ␂ SYMBOL FOR START OF TEXT
    # STX (0x02) is also a DSML separator — map it to pipe (fused into the
    # single-pass table so no second scan over the whole string is needed).
    '\x02': '|',
})


def _fold(text: str) -> str:
    """Fold fullwidth/CJK characters to their ASCII equivalents.

    Every mapping is one code point to one code point, so the folded string
    always has the same length as the input and positions map 1:1. Callers
    rely on that to index both strings with the same offsets.
    """
    return text.translate(_FULLWIDTH_TABLE)


# Characters that terminate tag-name scanning inside ``_parse_tag``: a quote
# opens/closes an attribute value, ``>`` ends the tag.
_QUOTE_OR_GT_RE = re.compile(r"[\"'>]")


# ---------------------------------------------------------------------------
# Canonical tag names
# ---------------------------------------------------------------------------

_CANONICAL: dict[str, str] = {
    'tool_calls': 'tool_calls',
    'tool-calls': 'tool_calls',
    'toolcalls': 'tool_calls',
    'invoke': 'invoke',
    'parameter': 'parameter',
}

_CANONICAL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ('toolcalls', 'tool_calls'),
    ('invoke', 'invoke'),
    ('parameter', 'parameter'),
)

_DSML_NAME_PREFIX = '|DSML|'
_DSML_CLOSE_NAME_PREFIX = '|/DSML|'
_DSML_NAME_PREFIXES = tuple(
    _DSML_NAME_PREFIX[:i] for i in range(1, len(_DSML_NAME_PREFIX) + 1)
)


# ---------------------------------------------------------------------------
# Tag-name matching helpers
# ---------------------------------------------------------------------------

def _compact_name(value: str) -> str:
    """Lowercase *value* and remove non-alphanumeric characters."""
    return ''.join(ch for ch in value.lower() if ch.isalnum())


def _candidate_name_starts(token: str) -> set[int]:
    """Return positions where a canonical tag segment may begin."""
    starts = {0}
    for i in range(1, len(token)):
        current = token[i]
        previous = token[i - 1]
        if not current.isalnum() or current.isspace():
            continue
        if not previous.isalnum() or (current.isupper() and previous.islower()):
            starts.add(i)
    return starts


def _alnum_groups(value: str) -> list[str]:
    """Return contiguous alphanumeric groups in *value*."""
    groups: list[str] = []
    current: list[str] = []
    for ch in value:
        if ch.isalnum():
            current.append(ch)
        elif current:
            groups.append(''.join(current))
            current = []
    if current:
        groups.append(''.join(current))
    return groups


def _prefix_allows_canonical_suffix(value: str) -> bool:
    """Return True when *value* is a safe prefix before a canonical tag."""
    groups = _alnum_groups(value)
    return len(groups) <= 1 or all(group.lower() == 'dsml' for group in groups)


def _canonicalize_tag_name(raw_name: str) -> Optional[str]:
    """Map a raw tag name to its canonical DSML form when recognised."""
    token = raw_name.strip()
    if not token:
        return None

    for start in sorted(_candidate_name_starts(token)):
        if start > 0 and not _prefix_allows_canonical_suffix(token[:start]):
            continue

        candidate = token[start:]
        direct = _CANONICAL.get(candidate.lower())
        if direct is not None:
            return direct

    return None


def _has_partial_canonical_suffix(raw_name: str) -> bool:
    """Return True when *raw_name* could still grow into a tool tag name."""
    token = raw_name.strip()
    if not token:
        return False

    for start in sorted(_candidate_name_starts(token)):
        compact = _compact_name(token[start:])
        if not compact:
            continue

        if start > 0 and not _prefix_allows_canonical_suffix(token[:start]):
            continue

        for suffix, _canonical in _CANONICAL_SUFFIXES:
            if suffix.startswith(compact):
                return True
    return False


def _extract_tag_name(inside: str) -> tuple[Optional[str], Optional[str]]:
    """Extract the raw tag name and its canonical form from tag contents."""
    for end in range(len(inside), 0, -1):
        candidate = inside[:end].rstrip()
        if not candidate:
            continue

        canonical = _canonicalize_tag_name(candidate)
        if canonical is None:
            continue

        remainder = inside[len(candidate):]
        if remainder and not (remainder[0].isspace() or remainder[0] == '/'):
            continue

        return candidate, canonical

    return None, None


# ---------------------------------------------------------------------------
# Partial-start detection (streaming hold)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ignored-region detection
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r'(?m)^[ \t]*(```+|~~~+)[^\n]*\n')
_FENCE_CLOSE_TEMPLATE = r'(?m)^[ \t]*{fence}[ \t]*(?:\n|$)'


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent spans."""
    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_fenced_code_spans(text: str) -> list[tuple[int, int]]:
    """Return markdown fenced-code spans, including unclosed fences to EOF."""
    spans: list[tuple[int, int]] = []
    pos = 0

    while True:
        opener = _FENCE_OPEN_RE.search(text, pos)
        if opener is None:
            break

        fence = opener.group(1)
        close_re = re.compile(_FENCE_CLOSE_TEMPLATE.format(fence=re.escape(fence)))
        closer = close_re.search(text, opener.end())
        if closer is None:
            spans.append((opener.start(), len(text)))
            break

        spans.append((opener.start(), closer.end()))
        pos = closer.end()

    return spans


def _find_inline_code_spans(text: str) -> list[tuple[int, int]]:
    """Return single-line inline code spans for one or two backticks.

    Candidate positions are located with ``str.find`` (C speed) instead of
    walking the string one character at a time.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    length = len(text)

    while pos < length:
        pos = text.find('`', pos)
        if pos < 0:
            break
        if text.startswith('``', pos):
            end = text.find('``', pos + 2)
            if end != -1 and '\n' not in text[pos:end + 2]:
                spans.append((pos, end + 2))
                pos = end + 2
                continue
        elif text[pos] == '`':
            end = text.find('`', pos + 1)
            if end != -1 and '\n' not in text[pos:end + 1]:
                spans.append((pos, end + 1))
                pos = end + 1
                continue
        pos += 1

    return spans


def _find_ignored_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted, merged list of (start, end) spans that should be ignored.

    Ignored regions include:
    - Markdown fenced code blocks (``` and ~~~)
    - Unclosed fenced code blocks (extend to EOF)
    - CDATA sections (<![CDATA[...]]>)
    - XML comments (<!-- ... -->)
    - XML processing instructions (<? ... ?>)
    - Markdown inline code spans using one or two backticks
    """
    spans = _find_fenced_code_spans(text)

    for m in re.finditer(r'<!\[CDATA\[[\s\S]*?\]\]>', text):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r'<!--[\s\S]*?-->', text):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r'<\?[\s\S]*?\?>', text):
        spans.append((m.start(), m.end()))
    spans.extend(_find_inline_code_spans(text))

    return _merge_spans(spans)


def _skip_ignored(pos: int, ignored: list[tuple[int, int]]) -> int:
    """If *pos* is inside an ignored span, jump to the end of that span."""
    for s, e in ignored:
        if s <= pos < e:
            return e
    return pos


class _SpanIndex:
    """Binary-searchable view over a sorted, merged ignored-span list.

    ``_skip_ignored`` used to be a linear scan over every span; during a full
    parse of a large DSML block that turns into quadratic behaviour because the
    span list grows with the number of fences/CDATA/comment sections.
    """

    __slots__ = ('spans', 'starts')

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        self.spans = spans
        self.starts = [start for start, _end in spans]

    def skip(self, pos: int) -> int:
        """Return *pos* unchanged, or the end of the span containing it."""
        starts = self.starts
        if not starts:
            return pos
        index = bisect.bisect_right(starts, pos) - 1
        if index < 0:
            return pos
        start, end = self.spans[index]
        if start <= pos < end:
            return end
        return pos


# ---------------------------------------------------------------------------
# Per-text scan cache
# ---------------------------------------------------------------------------
# A single ``parse_tool_calls_*`` call chain walks the *same* text object many
# times (once per block, once per nested tag search).  Recomputing the ignored
# spans and the folded copy on every hop is what made big DSML payloads
# quadratic.  We keep the last few (text, ignored, folded) triples per thread.
#
# Identity (``is``) is a safe cache key here because ``str`` is immutable *and*
# the cache holds a strong reference to the key, so the id cannot be recycled
# by the garbage collector while the entry is alive.

_SCAN_CACHE_LIMIT = 3
_scan_cache_local = threading.local()


def _scan_cache() -> dict[int, tuple[str, _SpanIndex, str]]:
    cache = getattr(_scan_cache_local, 'cache', None)
    if cache is None:
        cache = {}
        _scan_cache_local.cache = cache
    return cache


def _scan_artifacts(text: str) -> tuple[_SpanIndex, str]:
    """Return ``(span_index, folded_text)`` for *text*, memoised per thread."""
    cache = _scan_cache()
    key = id(text)
    entry = cache.get(key)
    if entry is not None and entry[0] is text:
        # Refresh LRU position (dicts keep insertion order).
        cache.pop(key)
        cache[key] = entry
        return entry[1], entry[2]

    artifacts = (_SpanIndex(_find_ignored_spans(text)), _fold(text))
    cache[key] = (text, artifacts[0], artifacts[1])
    while len(cache) > _SCAN_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    return artifacts


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

def _parse_tag(text: str, pos: int, folded: Optional[str] = None) -> Optional[ToolMarkupTag]:
    """Try to parse a DSML/XML tool tag starting at *pos* in the original text.

    *folded* is the pre-folded copy of *text* (same length, 1:1 positions).
    Passing it in avoids folding single characters inside the hot scan loop.

    Returns a ``ToolMarkupTag`` on success, or ``None`` if no valid tag
    starts at that position.
    """
    if pos >= len(text):
        return None
    if folded is None:
        folded = _fold(text)
    if folded[pos] != '<':
        return None

    idx = pos + 1
    is_closing = False
    if idx < len(text) and folded[idx] == '/':
        is_closing = True
        idx += 1

    name_start = idx
    if folded[idx:idx + len(_DSML_CLOSE_NAME_PREFIX)] == _DSML_CLOSE_NAME_PREFIX:
        is_closing = True
        name_start += len(_DSML_CLOSE_NAME_PREFIX)
    elif folded[idx:idx + len(_DSML_NAME_PREFIX)] == _DSML_NAME_PREFIX:
        name_start += len(_DSML_NAME_PREFIX)
    elif idx < len(text) and folded[idx] == '|':
        return None

    # Find the closing ``>`` in the original text, skipping over quoted
    # attribute values so that ``>`` inside a value is not mistaken for
    # the tag terminator.  Only quote characters and ``>`` matter, so jump
    # between them with the regex engine instead of walking every character.
    scan = name_start
    in_quote = False
    quote_char = ''

    while scan < len(text):
        match = _QUOTE_OR_GT_RE.search(folded, scan)
        if match is None:
            break
        scan = match.start()
        folded_ch = folded[scan]

        if in_quote:
            if folded_ch == quote_char:
                in_quote = False
        elif folded_ch == '"' or folded_ch == "'":
            in_quote = True
            quote_char = folded_ch
        else:  # folded_ch == '>'
            inside = text[name_start:scan]
            raw_name, canonical = _extract_tag_name(inside)
            if raw_name is None or canonical is None:
                return None
            return ToolMarkupTag(
                name=canonical,
                start=pos,
                end=scan + 1,
                closing=is_closing,
                raw_name=raw_name,
            )

        scan += 1

    # No closing ``>`` found — tag is incomplete.
    return None


# ---------------------------------------------------------------------------
# Internal helper — find tag reusing pre-computed ignored spans
# ---------------------------------------------------------------------------

def _find_tag_outside_ignored_with_spans(
    text: str,
    start: int,
    ignored: list[tuple[int, int]] | _SpanIndex,
    folded: Optional[str] = None,
) -> Optional[ToolMarkupTag]:
    """Same algorithm as ``find_tool_markup_tag_outside_ignored`` but
    accepts an externally-computed *ignored* span list so that callers
    can reuse it across multiple invocations.

    Candidate tag starts are located by jumping between ``<`` characters
    (``str.find`` on the folded text) instead of probing every position.
    """
    if start >= len(text):
        return None

    index = ignored if isinstance(ignored, _SpanIndex) else _SpanIndex(ignored)
    if folded is None:
        folded = _fold(text)

    length = len(text)
    pos = start
    while pos < length:
        candidate = folded.find('<', pos)
        if candidate < 0:
            return None
        skipped = index.skip(candidate)
        if skipped >= length:
            return None
        if skipped != candidate:
            # Candidate sits inside an ignored region: resume at its end, which
            # is itself a position worth probing.
            pos = skipped
            continue
        tag = _parse_tag(text, candidate, folded)
        if tag is not None:
            return tag
        pos = candidate + 1
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_tool_markup_tag_outside_ignored(
    text: str, start: int = 0,
) -> Optional[ToolMarkupTag]:
    """Find the first DSML/XML tool markup tag in *text* starting from *start*,
    skipping any content inside ignored regions (markdown fences, inline code,
    CDATA, XML comments, and processing instructions).

    Returns ``None`` when no tag is found.
    """
    if not text:
        return None
    index, folded = _scan_artifacts(text)
    return _find_tag_outside_ignored_with_spans(text, start, index, folded)


def find_matching_tool_markup_close(
    text: str, open_tag: ToolMarkupTag,
) -> Optional[ToolMarkupTag]:
    """Given an opening wrapper tag, find its matching closing tag.

    Handles nested tags of the same name (e.g. nested ``tool_calls``).
    Returns ``None`` when no matching close is found or when *open_tag*
    is already a closing tag.
    """
    if open_tag.closing:
        return None

    target = open_tag.name
    depth = 1
    pos = open_tag.end
    index, folded = _scan_artifacts(text)  # computed once, reused in loop

    while pos < len(text):
        tag = _find_tag_outside_ignored_with_spans(text, pos, index, folded)
        if tag is None:
            break

        if tag.name == target:
            if tag.closing:
                depth -= 1
                if depth == 0:
                    return tag
            else:
                depth += 1

        pos = tag.end

    return None


def contains_tool_markup_syntax_outside_ignored(text: str) -> bool:
    """Return ``True`` if *text* contains any DSML/XML tool markup syntax
    outside of ignored regions (fenced code, inline code, CDATA,
    comments, PIs).
    """
    return find_tool_markup_tag_outside_ignored(text, 0) is not None


def _is_partial_tag_candidate(fragment: str, folded: Optional[str] = None) -> bool:
    """Return True when *fragment* is a plausible partial tool tag start."""
    if folded is None:
        folded = _fold(fragment)
    if not folded.startswith('<'):
        return False

    rest = folded[1:]
    if rest == '':
        return True

    if rest.startswith('/'):
        rest = rest[1:]
        if rest == '':
            return True

    if rest.startswith('|'):
        if any(prefix.startswith(rest) for prefix in _DSML_NAME_PREFIXES):
            return True
        if not rest.startswith(_DSML_NAME_PREFIX):
            return False
        rest = rest[len(_DSML_NAME_PREFIX):]
        if rest == '':
            return True

    return _has_partial_canonical_suffix(rest)


def find_partial_tool_markup_start(text: str) -> int:
    """Return the index in *text* where a partial tool tag starts,
    or -1 if no partial tag is found (or if the partial tag falls
    inside an ignored region such as a fenced code block or inline
    code span).

    A partial tag is any plausible prefix of a recognised tool tag,
    including plain XML forms (``<tool_ca``), DSML forms
    (``<|DSML|too``), and bare angle brackets at end-of-stream.
    """
    if not text:
        return -1

    index, folded = _scan_artifacts(text)

    # Walk candidate ``<`` positions backwards instead of testing every index.
    pos = len(text) - 1
    while pos >= 0:
        candidate = folded.rfind('<', 0, pos + 1)
        if candidate < 0:
            return -1
        pos = candidate
        if index.skip(pos) == pos and _is_partial_tag_candidate(text[pos:], folded[pos:]):
            return pos
        pos -= 1

    return -1
