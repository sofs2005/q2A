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


def _as_int(value: object) -> int | None:
    """宽松取非负整数；非法/缺失返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_upstream_usage(raw: object) -> dict | None:
    """把上游每帧下发的累计 usage 归一化成 OpenAI 契约字段。

    上游字段为 input_tokens / output_tokens（累计值，非增量），此处只做映射，
    不做累加 —— 调用方按「覆盖最后一帧」处理。
    """
    if not isinstance(raw, dict):
        return None
    input_tokens = _as_int(raw.get("input_tokens"))
    output_tokens = _as_int(raw.get("output_tokens"))
    if input_tokens is None and output_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    total = _as_int(raw.get("total_tokens"))
    if total is None:
        total = input_tokens + output_tokens
    usage: dict = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total,
    }
    details = {
        key: raw.get(key)
        for key in ("output_tokens_details", "prompt_tokens_details")
        if isinstance(raw.get(key), dict)
    }
    if details:
        usage["_upstream_details"] = details
    return usage


def _attach_usage(evt: dict, source: dict) -> dict:
    """仅在帧内确实带 usage 时才挂键，避免污染无 usage 帧的结构。"""
    usage = _normalize_upstream_usage(source.get("usage"))
    if usage is not None:
        evt["usage"] = usage
    return evt


def _parse_upstream_error(evt: dict) -> dict | None:
    """识别官方 SSE 错误帧：{"error": {"code": "...", "details": "..."}, ...}。"""
    raw = evt.get("error")
    if not isinstance(raw, dict):
        return None
    code = _first_string(raw.get("code"), raw.get("type"), raw.get("error_code"))
    details = _first_string(raw.get("details"), raw.get("message"), raw.get("msg"), str(raw) if raw else "")
    if not code and not details:
        return None
    return {
        "type": "error",
        "phase": "error",
        "content": "",
        "status": "error",
        "code": code or "unknown",
        "details": details or code or "upstream error",
        "response_id": evt.get("response_id"),
        "response_index": evt.get("response_index"),
        "extra": {"error": raw},
    }


def _parse_response_created(evt: dict) -> dict | None:
    """识别流生命周期事件 response.created（非内容，不应当 unparsed）。"""
    created = evt.get("response.created")
    if not isinstance(created, dict):
        return None
    return _attach_usage(
        {
            "type": "lifecycle",
            "phase": "response.created",
            "content": "",
            "status": "created",
            "extra": created,
        },
        evt,
    )


def _parse_response_info(evt: dict) -> dict | None:
    """识别心跳帧 response.info（如 {"action": "keep_alive"}）。

    这类帧没有 content/status，落到兜底分支会不产出事件，进而被
    parse_sse_chunk 记为 unparsed 并打 WARNING，抬高 chunk_count 却不抬高
    parsed_event_count，可能误触发 stream_unparsed_preview 告警。
    """
    info = evt.get("response.info")
    if not isinstance(info, dict):
        return None
    action = _first_string(info.get("action")) or "info"
    return _attach_usage(
        {
            "type": "lifecycle",
            "phase": "response.info",
            "content": "",
            "status": action,
            "extra": info,
        },
        evt,
    )


def _parse_qwen_event(evt: dict) -> list[dict]:
    if not isinstance(evt, dict):
        return []

    lifecycle = _parse_response_created(evt)
    if lifecycle is not None:
        return [lifecycle]

    heartbeat = _parse_response_info(evt)
    if heartbeat is not None:
        return [heartbeat]

    error_evt = _parse_upstream_error(evt)
    if error_evt is not None:
        return [error_evt]

    if evt.get("choices"):
        delta = evt["choices"][0].get("delta", {})
        content = delta.get("content", "")

        if content and "Tool" in content and "does not exist" in content:
            log.warning(f"[SSE] Detected tool interception: content={content!r} phase={delta.get('phase')} status={delta.get('status')} extra={delta.get('extra')}")

        return [
            _attach_usage(
                {
                    "type": "delta",
                    "phase": delta.get("phase", "answer"),
                    "content": content,
                    "status": delta.get("status", ""),
                    "extra": delta.get("extra", {}),
                },
                evt,
            )
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
