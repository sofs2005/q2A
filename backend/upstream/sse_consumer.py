import json
import logging

log = logging.getLogger("qwen2api.sse")


def _preview(value: object, limit: int = 300) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


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
        if evt.get("choices"):
            delta = evt["choices"][0].get("delta", {})
            content = delta.get("content", "")

            # Log if content contains "Tool" and "does not exist"
            if content and "Tool" in content and "does not exist" in content:
                log.warning(f"[SSE] Detected tool interception: content={content!r} phase={delta.get('phase')} status={delta.get('status')} extra={delta.get('extra')}")

            parsed.append(
                {
                    "type": "delta",
                    "phase": delta.get("phase", "answer"),
                    "content": content,
                    "status": delta.get("status", ""),
                    "extra": delta.get("extra", {}),
                }
            )
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
