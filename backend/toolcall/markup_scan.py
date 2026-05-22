from __future__ import annotations

import re
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
})


def _fold(text: str) -> str:
    """Fold fullwidth/CJK characters to their ASCII equivalents."""
    folded = text.translate(_FULLWIDTH_TABLE)
    # STX (0x02) is also a DSML separator — map it to pipe.
    folded = folded.replace('\x02', '|')
    return folded


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
    """Return single-line inline code spans for one or two backticks."""
    spans: list[tuple[int, int]] = []
    pos = 0
    length = len(text)

    while pos < length:
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


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

def _parse_tag(text: str, pos: int) -> Optional[ToolMarkupTag]:
    """Try to parse a DSML/XML tool tag starting at *pos* in the original text.

    Returns a ``ToolMarkupTag`` on success, or ``None`` if no valid tag
    starts at that position.
    """
    if pos >= len(text) or _fold(text[pos]) != '<':
        return None

    idx = pos + 1
    is_closing = False
    if idx < len(text) and _fold(text[idx]) == '/':
        is_closing = True
        idx += 1

    name_start = idx
    if _fold(text[idx:idx + len(_DSML_CLOSE_NAME_PREFIX)]) == _DSML_CLOSE_NAME_PREFIX:
        is_closing = True
        name_start += len(_DSML_CLOSE_NAME_PREFIX)
    elif _fold(text[idx:idx + len(_DSML_NAME_PREFIX)]) == _DSML_NAME_PREFIX:
        name_start += len(_DSML_NAME_PREFIX)
    elif idx < len(text) and _fold(text[idx]) == '|':
        return None

    # Find the closing ``>`` in the original text, skipping over quoted
    # attribute values so that ``>`` inside a value is not mistaken for
    # the tag terminator.
    scan = name_start
    in_quote = False
    quote_char = ''

    while scan < len(text):
        ch = text[scan]
        folded_ch = _fold(ch)

        if in_quote:
            if folded_ch == quote_char:
                in_quote = False
        elif folded_ch in ('"', "'"):
            in_quote = True
            quote_char = folded_ch
        elif folded_ch == '>':
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
    ignored: list[tuple[int, int]],
) -> Optional[ToolMarkupTag]:
    """Same algorithm as ``find_tool_markup_tag_outside_ignored`` but
    accepts an externally-computed *ignored* span list so that callers
    can reuse it across multiple invocations.
    """
    pos = start
    while pos < len(text):
        pos = _skip_ignored(pos, ignored)
        if pos >= len(text):
            break
        tag = _parse_tag(text, pos)
        if tag is not None:
            return tag
        pos += 1
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
    ignored = _find_ignored_spans(text)
    return _find_tag_outside_ignored_with_spans(text, start, ignored)


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
    ignored = _find_ignored_spans(text)  # computed once, reused in loop

    while pos < len(text):
        tag = _find_tag_outside_ignored_with_spans(text, pos, ignored)
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


def _is_partial_tag_candidate(fragment: str) -> bool:
    """Return True when *fragment* is a plausible partial tool tag start."""
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
    ignored = _find_ignored_spans(text)

    for pos in range(len(text) - 1, -1, -1):
        if _fold(text[pos]) != '<':
            continue
        if _skip_ignored(pos, ignored) != pos:
            continue
        if _is_partial_tag_candidate(text[pos:]):
            return pos

    return -1
