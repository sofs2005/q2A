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
    '“': '"',   # " LEFT DOUBLE QUOTATION MARK
    '”': '"',   # " RIGHT DOUBLE QUOTATION MARK
    '‘': "'",   # ' LEFT SINGLE QUOTATION MARK
    '’': "'",   # ' RIGHT SINGLE QUOTATION MARK
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

# All prefixes of known tag names (used for partial-start detection).
_TAG_NAMES = ['tool_calls', 'tool-calls', 'toolcalls', 'invoke', 'parameter']
_TAG_PREFIXES: set[str] = set()
for _name in _TAG_NAMES:
    for _i in range(1, len(_name) + 1):
        _TAG_PREFIXES.add(_name[:_i])

# Regex to find a partial DSML tag at the *end* of the text.
_partial_pattern = (
    r'<\|DSML\|('
    + '|'.join(re.escape(p) for p in sorted(_TAG_PREFIXES, key=len, reverse=True))
    + r')$'
)
_PARTIAL_RE = re.compile(_partial_pattern, re.IGNORECASE)

# Regex that matches the *prefix* of a DSML tool tag (up to the tag name).
_DSML_PREFIX_RE = re.compile(
    r'<(/?)\|DSML\|(tool[_\-]?calls|toolcalls|invoke|parameter)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Ignored-region detection
# ---------------------------------------------------------------------------

def _find_ignored_spans(text: str) -> list[tuple[int, int]]:
    """Return sorted, merged list of (start, end) spans that should be ignored.

    Ignored regions include:
    - Markdown fenced code blocks (``` ... ```)
    - CDATA sections (<![CDATA[ ... ]]>)
    - XML comments (<!-- ... -->)
    - XML processing instructions (<? ... ?>)
    - Markdown inline code spans (`...`)
    """
    spans: list[tuple[int, int]] = []

    # Markdown fenced code blocks
    for m in re.finditer(r'(?m)^[ \t]*```[^\n]*\n[\s\S]*?\n[ \t]*```', text):
        spans.append((m.start(), m.end()))

    # CDATA sections
    for m in re.finditer(r'<!\[CDATA\[[\s\S]*?\]\]>', text):
        spans.append((m.start(), m.end()))

    # XML comments
    for m in re.finditer(r'<!--[\s\S]*?-->', text):
        spans.append((m.start(), m.end()))

    # Processing instructions
    for m in re.finditer(r'<\?[\s\S]*?\?>', text):
        spans.append((m.start(), m.end()))

    # Inline code spans
    for m in re.finditer(r'`[^`\n]+`', text):
        spans.append((m.start(), m.end()))

    # Sort and merge overlapping spans
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _in_ignored(pos: int, ignored: list[tuple[int, int]]) -> bool:
    """Return True if *pos* falls inside any ignored span."""
    for s, e in ignored:
        if s <= pos < e:
            return True
    return False


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
    """Try to parse a DSML tool tag starting at *pos* in the original text.

    Returns a ``ToolMarkupTag`` on success, or ``None`` if no valid tag
    starts at that position.
    """
    # Check the folded prefix — look at up to 200 chars ahead.
    remaining = _fold(text[pos:pos + 200])
    m = _DSML_PREFIX_RE.match(remaining)
    if not m:
        return None

    is_closing = bool(m.group(1))
    raw_name = m.group(2)
    canonical = _CANONICAL.get(raw_name.lower(), raw_name.lower())

    # Now find the closing ``>`` in the original text, skipping over
    # quoted attribute values so that ``>`` inside a value is not
    # mistaken for the tag terminator.
    prefix_len = len(m.group(0))
    idx = pos + prefix_len
    in_quote = False
    quote_char = ''

    while idx < len(text):
        ch = text[idx]
        folded_ch = _fold(ch)

        if in_quote:
            if folded_ch == quote_char:
                in_quote = False
        elif folded_ch in ('"', "'"):
            in_quote = True
            quote_char = folded_ch
        elif folded_ch == '>':
            return ToolMarkupTag(
                name=canonical,
                start=pos,
                end=idx + 1,
                closing=is_closing,
                raw_name=raw_name,
            )

        idx += 1

    # No closing ``>`` found — tag is incomplete.
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_tool_markup_tag_outside_ignored(
    text: str, start: int = 0,
) -> Optional[ToolMarkupTag]:
    """Find the first DSML tool markup tag in *text* starting from *start*,
    skipping any content inside markdown fences, inline code, CDATA,
    XML comments, and processing instructions.

    Returns ``None`` when no tag is found.
    """
    ignored = _find_ignored_spans(text)
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

    while pos < len(text):
        tag = find_tool_markup_tag_outside_ignored(text, pos)
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
    """Return ``True`` if *text* contains any DSML tool markup syntax
    outside of ignored regions (fenced code, inline code, CDATA,
    comments, PIs).
    """
    return find_tool_markup_tag_outside_ignored(text, 0) is not None


def find_partial_tool_markup_start(text: str) -> int:
    """Return the index in *text* where a partial DSML tool tag starts,
    or -1 if no partial tag is found.

    A partial tag is an opening ``<``, the DSML separator, and a prefix
    of a recognised tag name (e.g. ``<|DSML|tool``).  This is used by
    the streaming sieve to decide whether to hold trailing text.
    """
    folded = _fold(text)
    m = _PARTIAL_RE.search(folded)
    if m:
        return m.start()
    return -1
