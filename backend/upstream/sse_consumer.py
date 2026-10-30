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


def _parse_bare_json_error(chunk: str) -> dict | None:
    """识别「非 SSE 裸 JSON 错误体」（上游风控软拦的典型形态）。

    上游 AWSC 风控生效后，/api/v2/chat/completions 不再回 SSE 流，而是
    HTTP 200 + 一个裸 JSON 信封：

        {"success": false, "request_id": "...",
         "data": {"code": "Bad_Request", "details": "Internal error ..."}}

    旧实现只解析 `data:` 开头的行，这类响应一个事件都不产出、连 WARNING 都
    不打，错误被静默吞掉，调用方只能干等到 idle 超时。此处把它翻译成与
    `_parse_upstream_error` 完全同形的错误事件，让上层能立刻判错。

    Args:
        chunk: 单个 SSE 分片（此处约定分片内不含任何 `data:` 行）。

    Returns:
        错误事件 dict；不是裸 JSON 错误体时返回 None。
    """
    text = str(chunk or "").strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    # 形态一：上游信封 {"success": false, "data": {"code", "details"}}
    if payload.get("success") is False:
        data = payload.get("data")
        data = data if isinstance(data, dict) else {}
        code = _first_string(data.get("code"), data.get("type"), payload.get("code"))
        details = _first_string(
            data.get("details"),
            data.get("message"),
            data.get("msg"),
            payload.get("msg"),
            payload.get("message"),
            json.dumps(payload, ensure_ascii=False),
        )
        if not code and not details:
            return None
        # 复用 _parse_upstream_error 保证事件形状一致；完整信封放进 extra 便于运维排查
        error_obj: dict = {"code": code or "unknown", "details": details or code or "upstream error"}
        if payload.get("request_id") is not None:
            error_obj["request_id"] = payload.get("request_id")
        return _parse_upstream_error(
            {
                "error": error_obj,
                "response_id": _first_string(payload.get("request_id"), payload.get("response_id")) or None,
                "response_index": payload.get("response_index"),
            }
        )

    # 形态二：顶层直接带 error 对象
    return _parse_upstream_error(payload)


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

    # 本次分片里没有任何 data: 行 —— 可能整段就是一个裸 JSON 错误体（上游风控软拦）。
    # 必须放在正常 SSE 解析之后、且严格以「无 data: 行」为前提，保证正常流行为不变。
    if not data_lines:
        bare_error = _parse_bare_json_error(chunk)
        if bare_error is not None:
            log.warning(
                "[SSE] non-sse json error body code=%s details=%r preview=%r",
                bare_error.get("code"),
                bare_error.get("details"),
                _preview(chunk),
            )
            return [bare_error]

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
