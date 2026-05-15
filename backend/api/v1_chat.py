import asyncio
import hashlib
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable
from backend.adapter.standard_request import StandardRequest, detect_openai_client_profile
from backend.core.config import settings
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.services.attachment_preprocessor import preprocess_attachments
from backend.services.context_attachment_manager import prepare_context_attachments, derive_session_key
from backend.services.auth_quota import add_used_tokens, resolve_auth_context
from backend.services.completion_bridge import run_retryable_completion_bridge
from backend.services.openai_stream_translator import OpenAIStreamTranslator
from backend.services.response_formatters import build_openai_completion_payload
from backend.services.token_calc import calculate_usage
from backend.services.qwen_client import QwenClient
from backend.services.standard_request_builder import build_chat_standard_request
from backend.toolcore.task_session import (
    build_openai_assistant_history_message,
    clear_invalidated_session_chat,
    log_session_plan_reuse_cancelled,
    persist_session_turn,
    plan_persistent_session_turn,
)
from backend.runtime.execution import RuntimeAttemptState, build_tool_directive, build_usage_delta_factory, request_max_attempts

log = logging.getLogger("qwen2api.chat")
router = APIRouter()
OpenAIDeltaHandler = Callable[[dict[str, Any], str | None, list[dict[str, Any]] | None], Awaitable[None]]


def _stream_usage(result, prompt: str) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    if isinstance(usage, dict):
        return usage
    execution = getattr(result, "execution", None)
    state = getattr(execution, "state", None)
    answer_text = getattr(state, "answer_text", "") or ""
    tool_calls = getattr(state, "tool_calls", []) or []
    result_prompt = getattr(result, "prompt", prompt) or prompt
    return calculate_usage(result_prompt, answer_text, tool_calls)


def _detect_openai_client_profile(request: Request, req_data: dict) -> str:
    return detect_openai_client_profile(request.headers, req_data)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return str(content or "")


def _raw_openai_tool_names(req_data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in req_data.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        function = function if isinstance(function, dict) else {}
        name = tool.get("name") or function.get("name")
        if name:
            names.append(str(name))
    return names


def _tool_name_map_entries(standard_request: StandardRequest) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    catalog = standard_request.tool_catalog
    for tool in standard_request.tools:
        model_name = str(tool.get("name") or "")
        canonical_name = catalog.get_canonical_name(model_name) if catalog is not None else None
        canonical_name = canonical_name or model_name
        client_name = catalog.get_client_name(canonical_name) if catalog is not None else model_name
        entries.append({"model": model_name, "canonical": canonical_name, "client": client_name})
    return entries


def _build_openai_request_diagnostics(req_data: dict[str, Any], *, prompt: str) -> dict[str, Any]:
    messages = [message for message in (req_data.get("messages", []) or []) if isinstance(message, dict)]
    role_counts: dict[str, int] = {}
    assistant_tool_call_count = 0
    tool_result_count = 0
    latest_user_text = ""
    for message in messages:
        role = str(message.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            assistant_tool_call_count += len(message.get("tool_calls") or [])
        if role == "tool" or any(
            isinstance(part, dict) and part.get("type") == "tool_result"
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        ):
            tool_result_count += 1
        if role == "user":
            text = _text_from_message_content(message.get("content", "")).strip()
            if text:
                latest_user_text = text
    return {
        "message_count": len(messages),
        "role_counts": dict(sorted(role_counts.items())),
        "assistant_tool_call_count": assistant_tool_call_count,
        "tool_result_count": tool_result_count,
        "has_assistant_tool_calls": assistant_tool_call_count > 0,
        "has_tool_results": tool_result_count > 0,
        "latest_user_hash": _short_hash(latest_user_text) if latest_user_text else "",
        "prompt_hash": _short_hash(prompt),
    }


def _content_for_log(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part.get("text") or ""))
                elif "content" in part:
                    parts.append(_content_for_log(part.get("content")))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(text for text in parts if text)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


def _has_error_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in ("error", "failed", "exception", "traceback", "错误", "失败"))


def _has_image_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in ("data:image", "http://", "https://", ".png", ".jpg", ".jpeg", ".webp", "image", "图片"))


def _log_inbound_tool_turn_diagnostics(*, req_id: str, completion_id: str, prompt_hash: str, req_data: dict[str, Any]) -> None:
    messages = [message for message in (req_data.get("messages", []) or []) if isinstance(message, dict)]
    for message_index, message in enumerate(messages):
        role = str(message.get("role") or "")
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            for call_index, tool_call in enumerate(message.get("tool_calls") or []):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                function = function if isinstance(function, dict) else {}
                arguments = function.get("arguments")
                log.info(
                    "[OAI] inbound_assistant_tool_call req_id=%s completion_id=%s prompt_hash=%s message_index=%s call_index=%s id=%s type=%s name=%s arguments_chars=%s arguments_preview=%r",
                    req_id,
                    completion_id,
                    prompt_hash,
                    message_index,
                    call_index,
                    tool_call.get("id"),
                    tool_call.get("type"),
                    function.get("name"),
                    len(arguments) if isinstance(arguments, str) else 0,
                    _truncate_log_value(arguments) if isinstance(arguments, str) else "",
                )
        if role != "tool":
            continue
        content = _content_for_log(message.get("content"))
        log.info(
            "[OAI] inbound_tool_result req_id=%s completion_id=%s prompt_hash=%s message_index=%s tool_call_id=%s name=%s content_chars=%s empty=%s has_image_signal=%s has_error_signal=%s content_preview=%r",
            req_id,
            completion_id,
            prompt_hash,
            message_index,
            message.get("tool_call_id"),
            message.get("name"),
            len(content),
            not bool(content.strip()),
            _has_image_signal(content),
            _has_error_signal(content),
            _truncate_log_value(content),
        )


class _RepeatedToolRequestGuard:
    def __init__(self, *, ttl_seconds: float = 120.0, max_entries: int = 512, now: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.now = now
        self._entries: dict[tuple[str, str, str], tuple[float, list[str]]] = {}

    def record_tool_response(self, *, session_key: str, prompt_hash: str, latest_user_hash: str, tool_names: list[str]) -> None:
        names = [name for name in dict.fromkeys(tool_names) if name]
        if not session_key or not prompt_hash or not latest_user_hash or not names:
            return
        self._prune()
        self._entries[(session_key, prompt_hash, latest_user_hash)] = (self.now(), names)
        if len(self._entries) > self.max_entries:
            oldest_key = min(self._entries, key=lambda key: self._entries[key][0])
            self._entries.pop(oldest_key, None)

    def repeated_user_only_tool_request(self, session_key: str, diagnostics: dict[str, Any]) -> list[str] | None:
        if diagnostics.get("has_assistant_tool_calls") or diagnostics.get("has_tool_results"):
            return None
        if diagnostics.get("message_count") != 1 or diagnostics.get("role_counts") != {"user": 1}:
            return None
        prompt_hash = str(diagnostics.get("prompt_hash") or "")
        latest_user_hash = str(diagnostics.get("latest_user_hash") or "")
        key = (session_key, prompt_hash, latest_user_hash)
        if not all(key):
            return None
        exact = self._active_entry(key)
        if exact is not None:
            return exact
        for candidate_key in list(self._entries):
            candidate_session, _candidate_prompt_hash, candidate_user_hash = candidate_key
            if candidate_session != session_key or candidate_user_hash != latest_user_hash:
                continue
            fallback = self._active_entry(candidate_key)
            if fallback is not None:
                return fallback
        return None

    def _active_entry(self, key: tuple[str, str, str]) -> list[str] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        recorded_at, tool_names = entry
        if self.now() - recorded_at > self.ttl_seconds:
            self._entries.pop(key, None)
            return None
        return list(tool_names)

    def _prune(self) -> None:
        cutoff = self.now() - self.ttl_seconds
        expired = [key for key, (recorded_at, _tool_names) in self._entries.items() if recorded_at < cutoff]
        for key in expired:
            self._entries.pop(key, None)


_repeated_tool_request_guard = _RepeatedToolRequestGuard()


def _build_repeated_tool_request_notice(tool_names: list[str], *, prompt: str = "") -> str:
    del tool_names
    if "[SILENT]" in prompt:
        return "[SILENT]"
    return ""


def _build_openai_text_payload(*, completion_id: str, created: int, model_name: str, content: str, prompt: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": calculate_usage(prompt, content),
    }


def _openai_stream_chunk(*, completion_id: str, created: int, model_name: str, choices: list[dict[str, Any]], usage: dict[str, int] | None = None) -> str:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _log_openai_tool_name_map(*, req_id: str, completion_id: str, req_data: dict[str, Any], standard_request: StandardRequest) -> None:
    if not standard_request.tools:
        return
    log.info(
        "[OAI] tool_name_map req_id=%s completion_id=%s client_tools=%s model_tools=%s entries=%s",
        req_id,
        completion_id,
        _raw_openai_tool_names(req_data),
        standard_request.tool_names,
        _tool_name_map_entries(standard_request),
    )


def _truncate_log_value(value: str, *, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...[omitted {len(value) - limit} chars]"


def _openai_include_usage_requested(req_data: dict[str, Any]) -> bool:
    stream_options = req_data.get("stream_options")
    return isinstance(stream_options, dict) and stream_options.get("include_usage") is True


def _log_openai_stream_request_options(*, req_id: str, completion_id: str, req_data: dict[str, Any]) -> None:
    stream_options = req_data.get("stream_options")
    log.info(
        "[OAI] stream_request_options req_id=%s completion_id=%s stream=%s stream_options_type=%s include_usage_requested=%s raw_stream_options=%r",
        req_id,
        completion_id,
        req_data.get("stream"),
        type(stream_options).__name__,
        _openai_include_usage_requested(req_data),
        stream_options if isinstance(stream_options, dict) else None,
    )


def _log_openai_stream_finalize_options(
    *,
    req_id: str,
    completion_id: str,
    prompt_hash: str,
    req_data: dict[str, Any],
    finish_reason: str,
    usage: dict[str, int] | None,
    answer_text: str,
) -> None:
    log.info(
        "[OAI] stream_finalize_options req_id=%s completion_id=%s prompt_hash=%s finish_reason=%s include_usage_requested=%s usage_will_be_sent=%s usage=%s answer_chars=%s answer_preview=%r",
        req_id,
        completion_id,
        prompt_hash,
        finish_reason,
        _openai_include_usage_requested(req_data),
        usage is not None,
        usage,
        len(answer_text or ""),
        _truncate_log_value(answer_text or "", limit=300),
    )


def _log_outbound_tool_call_diagnostics(
    *,
    req_id: str,
    completion_id: str,
    prompt_hash: str,
    standard_request: StandardRequest,
    chunks: list[str],
) -> None:
    calls: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.startswith("data: "):
            continue
        data = chunk[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        delta = delta if isinstance(delta, dict) else {}
        for tool_call in delta.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            index = tool_call.get("index")
            if not isinstance(index, int):
                continue
            call = calls.setdefault(index, {"arguments_parts": []})
            if tool_call.get("id"):
                call["id"] = tool_call.get("id")
            if tool_call.get("type"):
                call["type"] = tool_call.get("type")
            function = tool_call.get("function")
            function = function if isinstance(function, dict) else {}
            if function.get("name"):
                call["name"] = function.get("name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                call["arguments_parts"].append(arguments)

    catalog = standard_request.tool_catalog
    for index, call in sorted(calls.items()):
        name = str(call.get("name") or "")
        canonical_name = catalog.get_canonical_name(name) if catalog is not None else None
        canonical_name = canonical_name or name
        client_name = catalog.get_client_name(canonical_name) if catalog is not None else name
        model_name = catalog.get_model_name(canonical_name) if catalog is not None else name
        arguments = "".join(call.get("arguments_parts") or [])
        json_valid = False
        input_keys: list[str] = []
        try:
            parsed_arguments = json.loads(arguments) if arguments else {}
            json_valid = True
            if isinstance(parsed_arguments, dict):
                input_keys = sorted(str(key) for key in parsed_arguments.keys())
        except json.JSONDecodeError:
            pass
        log.info(
            "[OAI] outbound_tool_call req_id=%s completion_id=%s prompt_hash=%s index=%s id=%s type=%s name=%s canonical_name=%s model_name=%s client_name=%s arguments_len=%s arguments_json_valid=%s input_keys=%s arguments_preview=%r",
            req_id,
            completion_id,
            prompt_hash,
            index,
            call.get("id"),
            call.get("type"),
            name,
            canonical_name,
            model_name,
            client_name,
            len(arguments),
            json_valid,
            input_keys,
            _truncate_log_value(arguments),
        )


def _log_openai_stream_sse_chunk(*, req_id: str, completion_id: str, prompt_hash: str, chunk: str) -> None:
    if not isinstance(chunk, str) or not chunk.startswith("data: "):
        return
    data = chunk[6:].strip()
    if not data or data == "[DONE]":
        log.info(
            "[OAI] stream_sse_chunk req_id=%s completion_id=%s prompt_hash=%s done=%s",
            req_id,
            completion_id,
            prompt_hash,
            data == "[DONE]",
        )
        return
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        log.info(
            "[OAI] stream_sse_chunk req_id=%s completion_id=%s prompt_hash=%s parse_error=True bytes=%s",
            req_id,
            completion_id,
            prompt_hash,
            len(chunk.encode("utf-8", errors="ignore")),
        )
        return
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        log.info(
            "[OAI] stream_sse_chunk req_id=%s completion_id=%s prompt_hash=%s choices=%s has_usage=%s",
            req_id,
            completion_id,
            prompt_hash,
            len(choices or []),
            isinstance(payload, dict) and payload.get("usage") is not None,
        )
        return
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice, dict) else {}
    delta = delta if isinstance(delta, dict) else {}
    tool_calls = delta.get("tool_calls")
    tool_names = []
    tool_details = []
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            function = function if isinstance(function, dict) else {}
            name = function.get("name")
            arguments = function.get("arguments")
            if name:
                tool_names.append(name)
            tool_details.append(
                {
                    "index": tool_call.get("index"),
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type"),
                    "name": name,
                    "arguments_chars": len(arguments) if isinstance(arguments, str) else 0,
                }
            )
    content = str(delta.get("content") or "")
    log.info(
        "[OAI] stream_sse_chunk req_id=%s completion_id=%s prompt_hash=%s choices=%s role=%s has_content=%s content_chars=%s content_preview=%r has_tool_calls=%s tool_names=%s tool_details=%s finish_reason=%s",
        req_id,
        completion_id,
        prompt_hash,
        len(choices),
        delta.get("role"),
        "content" in delta,
        len(content),
        _truncate_log_value(content, limit=160),
        bool(tool_calls),
        tool_names,
        tool_details,
        choice.get("finish_reason"),
    )



def _is_openai_content_delta_chunk(chunk: str) -> bool:
    if not isinstance(chunk, str) or not chunk.startswith("data: "):
        return False
    data = chunk[6:].strip()
    if not data or data == "[DONE]":
        return False
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice, dict) else {}
    return isinstance(delta, dict) and "content" in delta


def _filter_staged_chunks_for_tool_calls(chunks: list[str]) -> list[str]:
    return [chunk for chunk in chunks if not _is_openai_content_delta_chunk(chunk)]


def _openai_text_stream_chunks(*, completion_id: str, created: int, model_name: str, content: str, prompt: str):
    yield _openai_stream_chunk(
        completion_id=completion_id,
        created=created,
        model_name=model_name,
        choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    )
    yield _openai_stream_chunk(
        completion_id=completion_id,
        created=created,
        model_name=model_name,
        choices=[{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    )
    yield _openai_stream_chunk(
        completion_id=completion_id,
        created=created,
        model_name=model_name,
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
    )
    yield _openai_stream_chunk(
        completion_id=completion_id,
        created=created,
        model_name=model_name,
        choices=[],
        usage=calculate_usage(prompt, content),
    )
    yield "data: [DONE]\n\n"


def _record_repeated_tool_guard(
    *,
    session_key: str,
    diagnostics: dict[str, Any],
    tool_names: list[str],
    finish_reason: str,
    final_diagnostics: dict[str, Any] | None = None,
    guard: _RepeatedToolRequestGuard | None = None,
) -> None:
    if finish_reason != "tool_calls" or not tool_names:
        return
    target_guard = guard or _repeated_tool_request_guard
    for item in (diagnostics, final_diagnostics):
        if item is None:
            continue
        target_guard.record_tool_response(
            session_key=session_key,
            prompt_hash=str(item.get("prompt_hash") or ""),
            latest_user_hash=str(item.get("latest_user_hash") or ""),
            tool_names=tool_names,
        )


def _build_standard_request(req_data: dict, *, client_profile: str) -> StandardRequest:
    standard_request = build_chat_standard_request(
        req_data,
        default_model="gpt-3.5-turbo",
        surface="openai",
        client_profile=client_profile,
    )
    log.info("[OAI] normalized tools=%s profile=%s", standard_request.tool_names, client_profile)
    return standard_request


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    try:
        req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})

    client_profile = _detect_openai_client_profile(request, req_data)
    session_key = derive_session_key("openai", token, req_data)
    original_history_messages = req_data.get("messages", [])
    file_store = getattr(app.state, "file_store", None)
    preprocessed = None
    if file_store is not None:
        preprocessed = await preprocess_attachments(req_data, file_store, owner_token=token)
        req_data = preprocessed.payload

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    req_id = new_request_id()
    client_host = request.client.host if request.client else "-"
    early_standard_request = _build_standard_request(req_data, client_profile=client_profile)
    early_diagnostics = _build_openai_request_diagnostics(req_data, prompt=early_standard_request.prompt)
    repeated_tool_names = _repeated_tool_request_guard.repeated_user_only_tool_request(
        session_key,
        early_diagnostics,
    )
    if repeated_tool_names:
        with request_context(req_id=req_id, surface="openai", requested_model=early_standard_request.response_model, resolved_model=early_standard_request.resolved_model):
            log.warning(
                "[OAI] repeated_user_only_tool_request req_id=%s completion_id=%s session=%s prompt_hash=%s tool_names=%s before_context_upload=True action=log_only",
                req_id,
                completion_id,
                session_key,
                early_diagnostics["prompt_hash"],
                repeated_tool_names,
            )

    context_prepared = await prepare_context_attachments(app=app, payload=req_data, surface="openai", auth_token=token, client_profile=client_profile, existing_attachments=(preprocessed.attachments if preprocessed is not None else None))
    req_data = context_prepared["payload"]
    standard_request = _build_standard_request(req_data, client_profile=client_profile)
    if preprocessed is not None:
        standard_request.attachments = preprocessed.attachments
        standard_request.uploaded_file_ids = preprocessed.uploaded_file_ids
    standard_request.upstream_files = context_prepared["upstream_files"]
    standard_request.session_key = context_prepared["session_key"]
    standard_request.context_mode = context_prepared["context_mode"]
    standard_request.bound_account_email = context_prepared["bound_account_email"]
    standard_request.bound_account = context_prepared["bound_account"]

    session_plan = await plan_persistent_session_turn(app=app, request=standard_request, payload=req_data, surface="openai")
    if session_plan.enabled:
        standard_request.persistent_session = True
        standard_request.full_prompt = session_plan.full_prompt
        standard_request.prompt = session_plan.prompt
        standard_request.session_message_hashes = session_plan.current_hashes
        standard_request.upstream_chat_id = session_plan.existing_chat_id if session_plan.reuse_chat else None
        if standard_request.bound_account is None and session_plan.account_email:
            standard_request.bound_account = await app.state.account_pool.acquire_wait_preferred(session_plan.account_email, timeout=60)
            if standard_request.bound_account is not None:
                standard_request.bound_account_email = standard_request.bound_account.email
        elif standard_request.bound_account is not None and not standard_request.bound_account_email:
            standard_request.bound_account_email = standard_request.bound_account.email
        if standard_request.upstream_chat_id and standard_request.bound_account is None:
            log_session_plan_reuse_cancelled(
                request=standard_request,
                planned_chat_id=session_plan.existing_chat_id,
                reason="missing_bound_account",
            )
            standard_request.upstream_chat_id = None
            standard_request.prompt = standard_request.full_prompt or standard_request.prompt

    model_name = standard_request.response_model
    qwen_model = standard_request.resolved_model
    prompt = standard_request.prompt
    tools = standard_request.tools
    history_messages = original_history_messages

    diagnostics = _build_openai_request_diagnostics(req_data, prompt=prompt)
    guard_diagnostics = early_diagnostics

    with request_context(req_id=req_id, surface="openai", requested_model=model_name, resolved_model=qwen_model):
        log.info(
            "[OAI] model=%s stream=%s tool_enabled=%s profile=%s tools=%s prompt_len=%s prompt_tail=%r",
            qwen_model,
            standard_request.stream,
            standard_request.tool_enabled,
            standard_request.client_profile,
            [t.get('name') for t in tools],
            len(prompt),
            prompt[-500:],
        )
        log.info(
            "[OAI] request_diag req_id=%s client=%s session=%s prompt_hash=%s messages=%s roles=%s assistant_tool_calls=%s tool_results=%s latest_user_hash=%s context_mode=%s upstream_files=%s completion_id=%s",
            req_id,
            client_host,
            standard_request.session_key,
            diagnostics["prompt_hash"],
            diagnostics["message_count"],
            diagnostics["role_counts"],
            diagnostics["assistant_tool_call_count"],
            diagnostics["tool_result_count"],
            diagnostics["latest_user_hash"],
            standard_request.context_mode,
            len(standard_request.upstream_files or []),
            completion_id,
        )
        _log_inbound_tool_turn_diagnostics(
            req_id=req_id,
            completion_id=completion_id,
            prompt_hash=diagnostics["prompt_hash"],
            req_data=req_data,
        )
        _log_openai_stream_request_options(
            req_id=req_id,
            completion_id=completion_id,
            req_data=req_data,
        )
        _log_openai_tool_name_map(
            req_id=req_id,
            completion_id=completion_id,
            req_data=req_data,
            standard_request=standard_request,
        )
        repeated_tool_names = _repeated_tool_request_guard.repeated_user_only_tool_request(
            standard_request.session_key or session_key,
            diagnostics,
        )
        if repeated_tool_names:
            log.warning(
                "[OAI] repeated_user_only_tool_request req_id=%s completion_id=%s session=%s prompt_hash=%s tool_names=%s action=log_only",
                req_id,
                completion_id,
                standard_request.session_key,
                diagnostics["prompt_hash"],
                repeated_tool_names,
            )

        if standard_request.stream:
            async def generate():
                queue: asyncio.Queue[str | None] = asyncio.Queue()

                async def producer() -> None:
                    async with app.state.session_locks.hold(session_key):
                        try:
                            update_request_context(stream_attempt=1)
                            if standard_request.tools:
                                translator: OpenAIStreamTranslator | None = None
                                staged_chunks: list[str] = []

                                async def on_attempt_start(_attempt_index: int, _attempt_prompt: str) -> None:
                                    nonlocal translator, staged_chunks
                                    translator = OpenAIStreamTranslator(
                                        completion_id=completion_id,
                                        created=created,
                                        model_name=model_name,
                                        client_profile=standard_request.client_profile,
                                        build_final_directive=lambda answer_text: build_tool_directive(
                                            standard_request,
                                            RuntimeAttemptState(answer_text=answer_text),
                                        ),
                                        allowed_tool_names=standard_request.tool_names,
                                        toolcore_enabled=settings.TOOLCORE_V2_ENABLED,
                                        tool_catalog=standard_request.tool_catalog,
                                    )
                                    staged_chunks = []

                                async def on_retry(_attempt_index: int, _retry, _execution) -> None:
                                    nonlocal staged_chunks
                                    staged_chunks = []

                                async def on_delta(evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
                                    if translator is None:
                                        return
                                    translator.on_delta(evt, text_chunk, tool_calls)
                                    while translator.pending_chunks:
                                        staged_chunks.append(translator.pending_chunks.pop(0))

                                result = await run_retryable_completion_bridge(
                                    client=client,
                                    standard_request=standard_request,
                                    prompt=prompt,
                                    users_db=users_db,
                                    token=token,
                                    history_messages=history_messages,
                                    max_attempts=request_max_attempts(standard_request),
                                    usage_delta_factory=build_usage_delta_factory(prompt),
                                    allow_after_visible_output=True,
                                    capture_events=False,
                                    on_delta=on_delta,
                                    on_attempt_start=on_attempt_start,
                                    on_retry=on_retry,
                                )
                                execution = result.execution
                                directive = result.directive or build_tool_directive(standard_request, execution.state)
                                assistant_message = build_openai_assistant_history_message(
                                    execution=execution,
                                    request=standard_request,
                                    directive=directive,
                                )
                                await persist_session_turn(
                                    app=app,
                                    request=standard_request,
                                    surface="openai",
                                    execution=execution,
                                    assistant_message=assistant_message,
                                )
                                final_finish_reason = "tool_calls" if directive.stop_reason == "tool_use" else (execution.state.finish_reason or "stop")
                                tool_names = [block.get("name") for block in directive.tool_blocks if block.get("type") == "tool_use"]
                                _record_repeated_tool_guard(
                                    session_key=standard_request.session_key or session_key,
                                    diagnostics=guard_diagnostics,
                                    final_diagnostics=diagnostics,
                                    tool_names=tool_names,
                                    finish_reason=final_finish_reason,
                                )
                                log.info(
                                    "[OAI] stream_final req_id=%s completion_id=%s chat_id=%s prompt_hash=%s finish_reason=%s stop_reason=%s tool_names=%s answer_chars=%s staged_chunks=%s",
                                    req_id,
                                    completion_id,
                                    execution.chat_id,
                                    diagnostics["prompt_hash"],
                                    final_finish_reason,
                                    directive.stop_reason,
                                    tool_names,
                                    len(execution.state.answer_text or ""),
                                    len(staged_chunks),
                                )
                                output_staged_chunks = _filter_staged_chunks_for_tool_calls(staged_chunks) if final_finish_reason == "tool_calls" else staged_chunks
                                if final_finish_reason == "tool_calls":
                                    _log_outbound_tool_call_diagnostics(
                                        req_id=req_id,
                                        completion_id=completion_id,
                                        prompt_hash=diagnostics["prompt_hash"],
                                        standard_request=standard_request,
                                        chunks=output_staged_chunks,
                                    )
                                for chunk in output_staged_chunks:
                                    await queue.put(chunk)
                                if translator is not None:
                                    usage = _stream_usage(result, prompt)
                                    _log_openai_stream_finalize_options(
                                        req_id=req_id,
                                        completion_id=completion_id,
                                        prompt_hash=diagnostics["prompt_hash"],
                                        req_data=req_data,
                                        finish_reason=final_finish_reason,
                                        usage=usage,
                                        answer_text=execution.state.answer_text or "",
                                    )
                                    for chunk in translator.finalize(final_finish_reason, usage=usage):
                                        await queue.put(chunk)
                            else:
                                translator = OpenAIStreamTranslator(
                                    completion_id=completion_id,
                                    created=created,
                                    model_name=model_name,
                                    client_profile=standard_request.client_profile,
                                    build_final_directive=lambda answer_text: build_tool_directive(
                                        standard_request,
                                        RuntimeAttemptState(answer_text=answer_text),
                                    ),
                                    allowed_tool_names=standard_request.tool_names,
                                    toolcore_enabled=settings.TOOLCORE_V2_ENABLED,
                                )

                                async def on_delta(evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
                                    translator.on_delta(evt, text_chunk, tool_calls)
                                    while translator.pending_chunks:
                                        await queue.put(translator.pending_chunks.pop(0))

                                result = await run_retryable_completion_bridge(
                                    client=client,
                                    standard_request=standard_request,
                                    prompt=prompt,
                                    users_db=users_db,
                                    token=token,
                                    history_messages=history_messages,
                                    max_attempts=request_max_attempts(standard_request),
                                    usage_delta_factory=build_usage_delta_factory(prompt),
                                    allow_after_visible_output=True,
                                    capture_events=False,
                                    on_delta=on_delta,
                                )
                                execution = result.execution
                                directive = result.directive or build_tool_directive(standard_request, execution.state)
                                assistant_message = build_openai_assistant_history_message(
                                    execution=execution,
                                    request=standard_request,
                                    directive=directive,
                                )
                                await persist_session_turn(
                                    app=app,
                                    request=standard_request,
                                    surface="openai",
                                    execution=execution,
                                    assistant_message=assistant_message,
                                )
                                final_finish_reason = "tool_calls" if directive.stop_reason == "tool_use" else (execution.state.finish_reason or "stop")
                                tool_names = [block.get("name") for block in directive.tool_blocks if block.get("type") == "tool_use"]
                                _record_repeated_tool_guard(
                                    session_key=standard_request.session_key or session_key,
                                    diagnostics=guard_diagnostics,
                                    final_diagnostics=diagnostics,
                                    tool_names=tool_names,
                                    finish_reason=final_finish_reason,
                                )
                                log.info(
                                    "[OAI] stream_final req_id=%s completion_id=%s chat_id=%s prompt_hash=%s finish_reason=%s stop_reason=%s tool_names=%s answer_chars=%s staged_chunks=%s",
                                    req_id,
                                    completion_id,
                                    execution.chat_id,
                                    diagnostics["prompt_hash"],
                                    final_finish_reason,
                                    directive.stop_reason,
                                    tool_names,
                                    len(execution.state.answer_text or ""),
                                    len(translator.pending_chunks),
                                )
                                usage = _stream_usage(result, prompt)
                                _log_openai_stream_finalize_options(
                                    req_id=req_id,
                                    completion_id=completion_id,
                                    prompt_hash=diagnostics["prompt_hash"],
                                    req_data=req_data,
                                    finish_reason=final_finish_reason,
                                    usage=usage,
                                    answer_text=execution.state.answer_text or "",
                                )
                                for chunk in translator.finalize(final_finish_reason, usage=usage):
                                    await queue.put(chunk)
                        except HTTPException as he:
                            await clear_invalidated_session_chat(app=app, request=standard_request)
                            await queue.put(f"data: {json.dumps({'error': he.detail})}\n\n")
                        except Exception as e:
                            log.exception(
                                "[OAI] stream_error req_id=%s completion_id=%s prompt_hash=%s error=%s",
                                req_id,
                                completion_id,
                                diagnostics["prompt_hash"],
                                e,
                            )
                            await clear_invalidated_session_chat(app=app, request=standard_request)
                            await queue.put(f"data: {json.dumps({'error': str(e)})}\n\n")
                        finally:
                            log.info(
                                "[OAI] stream_producer_done req_id=%s completion_id=%s prompt_hash=%s",
                                req_id,
                                completion_id,
                                diagnostics["prompt_hash"],
                            )
                            await queue.put(None)

                producer_task = asyncio.create_task(producer())
                try:
                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        _log_openai_stream_sse_chunk(
                            req_id=req_id,
                            completion_id=completion_id,
                            prompt_hash=diagnostics["prompt_hash"],
                            chunk=chunk,
                        )
                        yield chunk
                finally:
                    if not producer_task.done():
                        log.warning(
                            "[OAI] stream_client_disconnect req_id=%s completion_id=%s prompt_hash=%s",
                            req_id,
                            completion_id,
                            diagnostics["prompt_hash"],
                        )
                        producer_task.cancel()
                        try:
                            await producer_task
                        except Exception:
                            pass

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            async with app.state.session_locks.hold(session_key):
                update_request_context(stream_attempt=1)
                result = await run_retryable_completion_bridge(
                    client=client,
                    standard_request=standard_request,
                    prompt=prompt,
                    users_db=users_db,
                    token=token,
                    history_messages=history_messages,
                    max_attempts=request_max_attempts(standard_request),
                    usage_delta_factory=build_usage_delta_factory(prompt),
                    allow_after_visible_output=True,
                )
                execution = result.execution
                directive = result.directive or build_tool_directive(standard_request, execution.state)
                assistant_message = build_openai_assistant_history_message(
                    execution=execution,
                    request=standard_request,
                    directive=directive,
                )
                await persist_session_turn(
                    app=app,
                    request=standard_request,
                    surface="openai",
                    execution=execution,
                    assistant_message=assistant_message,
                )
                tool_names = [block.get("name") for block in directive.tool_blocks if block.get("type") == "tool_use"]
                final_finish_reason = "tool_calls" if directive.stop_reason == "tool_use" else (execution.state.finish_reason or "stop")
                _record_repeated_tool_guard(
                    session_key=standard_request.session_key or session_key,
                    diagnostics=guard_diagnostics,
                    final_diagnostics=diagnostics,
                    tool_names=tool_names,
                    finish_reason=final_finish_reason,
                )
                log.info(
                    "[OAI] json_final req_id=%s completion_id=%s chat_id=%s prompt_hash=%s finish_reason=%s stop_reason=%s tool_names=%s answer_chars=%s",
                    req_id,
                    completion_id,
                    execution.chat_id,
                    diagnostics["prompt_hash"],
                    final_finish_reason,
                    directive.stop_reason,
                    tool_names,
                    len(execution.state.answer_text or ""),
                )

                return JSONResponse(build_openai_completion_payload(
                    completion_id=completion_id,
                    created=created,
                    model_name=model_name,
                    prompt=result.prompt,
                    execution=execution,
                    standard_request=standard_request,
                ))
        except Exception as e:
            await clear_invalidated_session_chat(app=app, request=standard_request)
            raise HTTPException(status_code=500, detail=str(e))
