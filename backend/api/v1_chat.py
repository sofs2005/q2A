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
    result_prompt = getattr(result, "prompt", prompt) or prompt
    return calculate_usage(result_prompt, answer_text)


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

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    req_id = new_request_id()
    diagnostics = _build_openai_request_diagnostics(req_data, prompt=prompt)
    client_host = request.client.host if request.client else "-"

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
                                for chunk in staged_chunks:
                                    await queue.put(chunk)
                                if translator is not None:
                                    for chunk in translator.finalize(final_finish_reason, usage=_stream_usage(result, prompt)):
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
                                for chunk in translator.finalize(final_finish_reason, usage=_stream_usage(result, prompt)):
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
