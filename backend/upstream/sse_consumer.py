import json
import logging

log = logging.getLogger("qwen2api.sse")


def _preview(value: object, limit: int = 300) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _first_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_qwen_event(evt: dict) -> list[dict]:
    if evt.get("choices"):
        delta = evt["choices"][0].get("delta", {})
        content = delta.get("content", "")

        if content and "Tool" in content and "does not exist" in content:
            log.warning(f"[SSE] Detected tool interception: content={content!r} phase={delta.get('phase')} status={delta.get('status')} extra={delta.get('extra')}")

        return [
            {
                "type": "delta",
                "phase": delta.get("phase", "answer"),
                "content": content,
                "status": delta.get("status", ""),
                "extra": delta.get("extra", {}),
            }
        ]

    parsed = []
    content = _first_string(evt.get("content"), evt.get("answer"), evt.get("text"), evt.get("delta"))
    status = _first_string(evt.get("status"))
    event_type = _first_string(evt.get("event"), evt.get("type"), status)
    if content or event_type:
        parsed.append(
            {
                "type": event_type or "delta",
                "phase": event_type or "answer",
                "content": content,
                "status": status,
                "extra": {},
            }
        )
    for key in ("data", "message"):
        nested = evt.get(key)
        if isinstance(nested, dict):
            parsed.extend(_parse_qwen_event(nested))
    return parsed


def parse_sse_chunk(chunk: str) -> list[dict]:
    events = []
    data_lines = []
    invalid_data_lines = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        data_lines.append(data)
        try:
            obj = json.loads(data)
            events.append(obj)
        except Exception:
            invalid_data_lines.append(data)

    parsed = []
    for evt in events:
        parsed.extend(_parse_qwen_event(evt))
    if not parsed and data_lines:
        if invalid_data_lines:
            log.warning(
                "[SSE] non-json data line count=%s preview=%r",
                len(invalid_data_lines),
                _preview(invalid_data_lines[0]),
            )
        elif events:
            first = events[0]
            keys = sorted(first.keys()) if isinstance(first, dict) else []
            log.warning(
                "[SSE] unparsed json event count=%s keys=%s preview=%r",
                len(events),
                keys,
                _preview(json.dumps(first, ensure_ascii=False) if isinstance(first, dict) else first),
            )
    return parsed
