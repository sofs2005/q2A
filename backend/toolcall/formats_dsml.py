from __future__ import annotations

import html
import re
from typing import Any

from backend.toolcall.formats_json import load_json_with_repair
from backend.toolcall.markup_scan import (
    ToolMarkupTag,
    find_matching_tool_markup_close,
    find_tool_markup_tag_outside_ignored,
)

CDATA_OPEN_RE = re.compile(r"(?is)<\s*[!！]\s*\[\s*cdata\s*\[")
CDATA_CLOSE_RE = re.compile(r"(?is)\]\s*\]\s*[>＞〉﹥]")
ATTR_RE = re.compile(r"(?is)\b([a-z0-9_:-]+)\s*=\s*([\"'])(.*?)\2")
ATTR_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "＝": "=",
})

__all__ = ["consume_dsml_tool_capture", "has_open_dsml_tool_tag", "parse_dsml_format"]


def _tag_source(text: str, tag: ToolMarkupTag) -> str:
    return text[tag.start:tag.end]


def _parse_attrs(raw: str) -> dict[str, str]:
    normalized = raw.translate(ATTR_TRANSLATION)
    return {match.group(1): html.unescape(match.group(3)) for match in ATTR_RE.finditer(normalized)}


def _normalize_cdata_markers(text: str) -> str:
    restored = CDATA_OPEN_RE.sub("<![CDATA[", text)
    return CDATA_CLOSE_RE.sub("]]>", restored)


def _is_wrapped_cdata(text: str) -> bool:
    restored = _normalize_cdata_markers(text)
    return restored.startswith("<![CDATA[") and restored.endswith("]]>")


def _restore_cdata(text: str) -> str:
    restored = _normalize_cdata_markers(text)
    if restored.startswith("<![CDATA[") and restored.endswith("]]>"):
        return restored[9:-3].replace("]]]]><![CDATA[>", "]]>")
    return restored


def _coerce_scalar(value: str) -> Any:
    raw = value.strip()
    restored = _restore_cdata(raw)
    # CDATA-wrapped values are literal strings — skip type coercion entirely
    if _is_wrapped_cdata(raw):
        return restored if restored != "" else ""
    stripped = html.unescape(restored)
    if stripped == "":
        return ""

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
            return float(stripped)
    except ValueError:
        return stripped

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return load_json_with_repair(stripped)
        except Exception:
            return stripped
    return stripped


def _find_blocks(text: str, tag_name: str, from_pos: int = 0) -> list[tuple[ToolMarkupTag, ToolMarkupTag, str]]:
    blocks: list[tuple[ToolMarkupTag, ToolMarkupTag, str]] = []
    pos = from_pos

    while pos < len(text):
        open_tag = find_tool_markup_tag_outside_ignored(text, pos)
        if open_tag is None:
            break
        if open_tag.closing or open_tag.name != tag_name:
            pos = open_tag.end
            continue

        close_tag = find_matching_tool_markup_close(text, open_tag)
        if close_tag is None:
            pos = open_tag.end
            continue

        blocks.append((open_tag, close_tag, text[open_tag.end:close_tag.start]))
        pos = close_tag.end

    return blocks


def _find_named_blocks(text: str, name: str) -> list[str]:
    pattern = re.compile(rf"(?is)<\s*{re.escape(name)}\b[^>]*>(.*?)<\s*/\s*{re.escape(name)}\s*>")
    return [match.group(1) for match in pattern.finditer(text)]


def _parse_generic_children(body: str) -> dict[str, Any]:
    pattern = re.compile(r"(?is)<\s*([a-zA-Z_][\w:-]*)\b[^>]*>(.*?)<\s*/\s*\1\s*>")
    out: dict[str, Any] = {}

    for match in pattern.finditer(body):
        key = match.group(1)
        value = _parse_parameter_value(match.group(2))
        if key in out:
            existing = out[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                out[key] = [existing, value]
        else:
            out[key] = value

    return out


def _parse_xmlish_children(body: str) -> Any:
    parameter_blocks = _find_blocks(body, "parameter")
    if parameter_blocks:
        out: dict[str, Any] = {}
        for open_tag, _close_tag, inner in parameter_blocks:
            name = _parse_attrs(_tag_source(body, open_tag)).get("name", "").strip()
            if name:
                out[name] = _parse_parameter_value(inner)
        return out

    item_blocks = _find_named_blocks(body, "item")
    if item_blocks:
        return [_parse_parameter_value(inner) for inner in item_blocks]

    child_map = _parse_generic_children(body)
    return child_map if child_map else _coerce_scalar(body)


def _parse_parameter_value(body: str) -> Any:
    stripped = body.strip()
    if stripped == "":
        return ""

    restored = _restore_cdata(stripped)
    if _is_wrapped_cdata(stripped):
        json_candidate = restored.strip()
        if json_candidate.startswith("{") or json_candidate.startswith("["):
            try:
                return load_json_with_repair(json_candidate)
            except Exception:
                pass
        return restored

    restored = restored.strip()
    if restored.startswith("{") or restored.startswith("["):
        try:
            return load_json_with_repair(restored)
        except Exception:
            pass

    return _parse_xmlish_children(stripped)


def _parse_invoke(text: str, open_tag: ToolMarkupTag, body: str, allowed_names: set[str]) -> dict[str, Any] | None:
    name = _parse_attrs(_tag_source(text, open_tag)).get("name", "").strip()
    if not name or name not in allowed_names:
        return None

    payload = _parse_parameter_value(body)
    return {"name": name, "input": payload if isinstance(payload, dict) else {}}


def _repair_missing_wrapper(text: str) -> str:
    first_invoke: ToolMarkupTag | None = None
    last_wrapper_close: ToolMarkupTag | None = None
    pos = 0

    while pos < len(text):
        tag = find_tool_markup_tag_outside_ignored(text, pos)
        if tag is None:
            break
        if tag.name == "invoke" and not tag.closing and first_invoke is None:
            first_invoke = tag
        if tag.name == "tool_calls" and tag.closing:
            last_wrapper_close = tag
        pos = tag.end

    if first_invoke is None or last_wrapper_close is None or first_invoke.start >= last_wrapper_close.start:
        return text

    return (
        text[:first_invoke.start]
        + "<tool_calls>"
        + text[first_invoke.start:last_wrapper_close.start]
        + "</tool_calls>"
        + text[last_wrapper_close.end:]
    )


def parse_dsml_format(text: str, allowed_names: set[str]) -> list[dict[str, Any]]:
    candidate = _repair_missing_wrapper(text.strip())
    wrappers = _find_blocks(candidate, "tool_calls")
    if not wrappers:
        return []

    calls: list[dict[str, Any]] = []
    for _wrapper_open, _wrapper_close, wrapper_body in wrappers:
        for invoke_open, _invoke_close, invoke_body in _find_blocks(wrapper_body, "invoke"):
            call = _parse_invoke(wrapper_body, invoke_open, invoke_body, allowed_names)
            if call is not None:
                calls.append(call)
    return calls


def has_open_dsml_tool_tag(text: str) -> bool:
    pos = 0
    while pos < len(text):
        tag = find_tool_markup_tag_outside_ignored(text, pos)
        if tag is None:
            return False
        if tag.name == "tool_calls" and not tag.closing and find_matching_tool_markup_close(text, tag) is None:
            return True
        pos = tag.end
    return False


def consume_dsml_tool_capture(captured: str, allowed_names: set[str]) -> tuple[str, list[dict[str, Any]], str, bool]:
    tag = find_tool_markup_tag_outside_ignored(captured, 0)
    while tag is not None:
        if tag.name != "tool_calls" or tag.closing:
            tag = find_tool_markup_tag_outside_ignored(captured, tag.end)
            continue

        close_tag = find_matching_tool_markup_close(captured, tag)
        if close_tag is None:
            return "", [], "", False

        block = captured[tag.start:close_tag.end]
        calls = parse_dsml_format(block, allowed_names)
        prefix = captured[:tag.start]
        suffix = captured[close_tag.end:]
        if calls:
            return prefix, calls, suffix, True
        return prefix + block, [], suffix, True

    return captured, [], "", True
