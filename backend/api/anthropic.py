import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.adapter.standard_request import StandardRequest, enforce_declared_tool_choice
from backend.core.config import resolve_model, settings
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.runtime import stream_presenter
from backend.runtime.execution import (
    build_tool_directive,
    cleanup_runtime_resources,
    collect_completion_run,
    evaluate_retry_directive,
    request_max_attempts,
)
from backend.services.auth_quota import resolve_auth_context
from backend.services.command_environment import detect_command_environment, format_command_environment_hint
from backend.services.context_attachment_manager import build_request_session_key, prepare_context_attachments
from backend.services.attachment_preprocessor import preprocess_attachments
from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE
from backend.toolcore.prompt_builder import messages_to_prompt
from backend.services.response_formatters import _client_visible_tool_name, build_anthropic_message_payload
from backend.services.qwen_client import QwenClient
from backend.services.token_calc import count_tokens
from backend.adapter.standard_request import normalize_tool_choice
from backend.toolcore.request_normalizer import normalize_anthropic_request, to_prompt_payload
from backend.services.completion_bridge import run_retryable_completion_bridge
from backend.toolcall.normalize import build_tool_name_registry

log = logging.getLogger("qwen2api.anthropic")
router = APIRouter()


class _AnthropicStreamState:
    def __init__(self, *, msg_id: str, model_name: str, prompt: str, extra_prompt_tokens: int = 0):
        self.msg_id = msg_id
        self.model_name = model_name
        self.prompt = prompt
        self.extra_prompt_tokens = extra_prompt_tokens
        self.pending_chunks: list[str] = []
        self.answer_text_buffer: list[tuple[int, str]] = []
        self.block_index = 0
        self.current_block: dict[str, object] = {"type": None, "index": None, "tool_call_id": None}
        self.opened_tool_calls: set[str] = set()

    def ensure_message_start(self) -> None:
        if not self.pending_chunks:
            self.pending_chunks.append(_message_start_event(
                self.msg_id,
                self.model_name,
                self.prompt,
                "",
                extra_prompt_tokens=self.extra_prompt_tokens,
            ))

    def close_current_block(self) -> None:
        index = self.current_block.get("index")
        if not isinstance(index, int):
            return
        self.pending_chunks.append(stream_presenter.anthropic_content_block_stop(index))
        self.current_block = {"type": None, "index": None, "tool_call_id": None}

    def open_textual_block(self, block_type: str) -> int:
        current_type = self.current_block.get("type")
        current_index = self.current_block.get("index")
        if current_type == block_type and isinstance(current_index, int):
            return current_index
        self.close_current_block()
        index = self.block_index
        self.block_index += 1
        if block_type == "thinking":
            content_block = {"type": "thinking", "thinking": ""}
        else:
            content_block = {"type": "text", "text": ""}
        self.pending_chunks.append(stream_presenter.anthropic_content_block_start(index, content_block))
        self.current_block = {"type": block_type, "index": index, "tool_call_id": None}
        return index

    def open_tool_block(self, tool_call_id: str, tool_name: str) -> int:
        current_index = self.current_block.get("index")
        if (
            self.current_block.get("type") == "tool_use"
            and self.current_block.get("tool_call_id") == tool_call_id
            and isinstance(current_index, int)
        ):
            return current_index
        self.close_current_block()
        index = self.block_index
        self.block_index += 1
        self.pending_chunks.append(
            f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': {'type': 'tool_use', 'id': tool_call_id, 'name': tool_name, 'input': {}}}, ensure_ascii=False)}\n\n"
        )
        self.current_block = {"type": "tool_use", "index": index, "tool_call_id": tool_call_id}
        self.opened_tool_calls.add(tool_call_id)
        return index

    def append_thinking_delta(self, text_chunk: str) -> None:
        index = self.open_textual_block("thinking")
        self.pending_chunks.append(
            stream_presenter.anthropic_content_block_delta(index, {"type": "thinking_delta", "thinking": text_chunk})
        )

    def buffer_answer_text(self, text_chunk: str) -> None:
        index = self.open_textual_block("text")
        self.answer_text_buffer.append((index, text_chunk))

    def append_tool_delta(self, *, tool_call_id: str, tool_name: str, partial_json: str) -> None:
        index = self.open_tool_block(tool_call_id, tool_name)
        if partial_json:
            self.pending_chunks.append(
                stream_presenter.anthropic_content_block_delta(index, {"type": "input_json_delta", "partial_json": partial_json})
            )

    def flush_answer_text(self) -> None:
        if not self.answer_text_buffer:
            return
        for index, text_chunk in self.answer_text_buffer:
            self.pending_chunks.append(
                stream_presenter.anthropic_content_block_delta(index, {"type": "text_delta", "text": text_chunk})
            )
        self.answer_text_buffer = []

    def clear_answer_text(self) -> None:
        self.answer_text_buffer = []


def _build_standard_request(req_data: dict, *, command_environment=None) -> StandardRequest:
    model_name = req_data.get("model", "claude-3-5-sonnet")
    normalized_request = normalize_anthropic_request(req_data)
    normalized_payload = to_prompt_payload(normalized_request, model=model_name, stream=bool(req_data.get("stream", False)))
    for field_name in ("system", "developer", "instructions"):
        if field_name in req_data:
            normalized_payload[field_name] = req_data.get(field_name, "")
    prompt_result = messages_to_prompt(normalized_payload, client_profile=CLAUDE_CODE_OPENAI_PROFILE)
    prompt = prompt_result.prompt
    tools = prompt_result.tools
    tool_names = [tool_name for tool_name in (tool.get("name") for tool in tools) if isinstance(tool_name, str) and tool_name]
    tool_choice = normalize_tool_choice(normalized_payload.get("tool_choice"))
    tool_choice = enforce_declared_tool_choice(tool_choice, tool_names)
    return StandardRequest(
        prompt=prompt,
        response_model=model_name,
        resolved_model=resolve_model(model_name),
        surface="anthropic",
        client_profile=CLAUDE_CODE_OPENAI_PROFILE,
        stream=req_data.get("stream", False),
        tools=tools,
        tool_names=tool_names,
        tool_name_registry=build_tool_name_registry(tool_names),
        tool_catalog=normalized_request.tool_catalog,
        tool_enabled=prompt_result.tool_enabled,
        tool_choice_mode=tool_choice.mode,
        required_tool_name=tool_choice.required_tool_name,
        tool_choice_raw=tool_choice.raw,
        command_environment=command_environment,
    )


def _anthropic_usage(prompt: str, answer_text: str, *, extra_prompt_tokens: int = 0) -> dict[str, int]:
    return {"input_tokens": count_tokens(prompt) + max(0, int(extra_prompt_tokens or 0)), "output_tokens": count_tokens(answer_text)}


def _message_start_event(msg_id: str, model_name: str, prompt: str, answer_text: str, *, extra_prompt_tokens: int = 0) -> str:
    return stream_presenter.anthropic_message_start(msg_id, model_name, _anthropic_usage(prompt, answer_text, extra_prompt_tokens=extra_prompt_tokens))


def _visible_answer_text_length(*, directive, execution, stream_state: _AnthropicStreamState | None = None) -> int:
    if directive.stop_reason == "tool_use":
        return 0
    if stream_state is not None:
        buffered_text = "".join(text_chunk for _, text_chunk in stream_state.answer_text_buffer)
        if buffered_text:
            return count_tokens(buffered_text)
    return count_tokens(execution.state.answer_text)


async def _add_used_tokens_for_prompt(*, users_db, token: str, prompt_text: str, answer_text_length: int, extra_prompt_tokens: int = 0) -> None:
    users = await users_db.get()
    for user in users:
        if user["id"] == token:
            user["used_tokens"] += answer_text_length + count_tokens(prompt_text) + max(0, int(extra_prompt_tokens or 0))
            break
    await users_db.save(users)


async def _reacquire_bound_account_if_needed(*, client: QwenClient, standard_request: StandardRequest) -> None:
    preferred_email = getattr(standard_request, "bound_account_email", None)
    if preferred_email:
        standard_request.bound_account = await client.account_pool.acquire_wait_preferred(preferred_email, timeout=60)
    else:
        standard_request.bound_account = None


@router.post("/messages/count_tokens")
@router.post("/v1/messages/count_tokens")
@router.post("/anthropic/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    try:
        req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})
    prompt_result = messages_to_prompt(req_data, client_profile=CLAUDE_CODE_OPENAI_PROFILE)
    return JSONResponse({"input_tokens": count_tokens(prompt_result.prompt)})


@router.post("/messages")
@router.post("/v1/messages")
@router.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    try:
        req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})

    req_id = new_request_id()
    session_key = build_request_session_key("anthropic", req_id)
    command_environment = detect_command_environment(headers=request.headers, request_data=req_data)
    command_environment_hint = format_command_environment_hint(command_environment)
    original_history_messages = req_data.get("messages", [])

    async def prepare_locked_request(payload: dict) -> tuple[StandardRequest, dict, str, str, str, str]:
        file_store = getattr(app.state, "file_store", None)
        preprocessed = None
        working_payload = payload
        if file_store is not None:
            preprocessed = await preprocess_attachments(working_payload, file_store, owner_token=token)
            working_payload = preprocessed.payload
        context_prepared = await prepare_context_attachments(
            app=app,
            payload=working_payload,
            surface="anthropic",
            auth_token=token,
            client_profile=CLAUDE_CODE_OPENAI_PROFILE,
            session_key=session_key,
            existing_attachments=(preprocessed.attachments if preprocessed is not None else None),
        )
        working_payload = context_prepared["payload"]
        standard_request = _build_standard_request(working_payload, command_environment=command_environment)
        if preprocessed is not None:
            standard_request.attachments = preprocessed.attachments
            standard_request.uploaded_file_ids = preprocessed.uploaded_file_ids
        standard_request.upstream_files = context_prepared["upstream_files"]
        standard_request.session_key = context_prepared["session_key"]
        standard_request.context_mode = context_prepared["context_mode"]
        standard_request.context_attachment_tokens = context_prepared.get("context_attachment_tokens", 0)
        standard_request.bound_account_email = context_prepared["bound_account_email"]
        standard_request.bound_account = context_prepared["bound_account"]

        model_name = standard_request.response_model
        qwen_model = standard_request.resolved_model
        prompt = standard_request.prompt
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        return standard_request, working_payload, model_name, qwen_model, prompt, msg_id

    with request_context(req_id=req_id, surface="anthropic", requested_model=req_data.get("model", "claude-3-5-sonnet"), resolved_model="-", command_environment=command_environment_hint):
        if request.headers.get("x-debug-session-key"):
            pass

        if req_data.get("stream", False):
            async def generate():
                async with app.state.session_locks.hold(session_key):
                    standard_request, effective_payload, model_name, qwen_model, prompt, msg_id = await prepare_locked_request(req_data)
                    update_request_context(requested_model=model_name, resolved_model=qwen_model)
                    log.info(f"[ANT] model={qwen_model}, stream={standard_request.stream}, tool_enabled={standard_request.tool_enabled}, tools={[t.get('name') for t in standard_request.tools]}, prompt_len={len(prompt)}")
                    history_messages = original_history_messages
                    current_prompt = prompt
                    max_attempts = request_max_attempts(standard_request)
                    for stream_attempt in range(max_attempts):
                        stream_state = _AnthropicStreamState(
                            msg_id=msg_id,
                            model_name=model_name,
                            prompt=current_prompt,
                            extra_prompt_tokens=standard_request.context_attachment_tokens,
                        )
                        try:
                            update_request_context(stream_attempt=stream_attempt + 1)

                            async def on_delta(evt: dict[str, Any], text_chunk: str | None, _: list[dict[str, Any]] | None) -> None:
                                stream_state.ensure_message_start()
                                phase = evt.get("phase")
                                if text_chunk and phase in ("think", "thinking_summary"):
                                    stream_state.append_thinking_delta(text_chunk)
                                    return
                                if text_chunk and phase == "answer":
                                    stream_state.buffer_answer_text(text_chunk)
                                    return
                                if phase == "tool_call":
                                    extra = evt.get("extra", {}) or {}
                                    tool_call_id = extra.get("tool_call_id")
                                    if tool_call_id is None:
                                        tool_call_id = f"tc_idx_{extra.get('index', 0)}"
                                    tool_name = extra.get("tool_name")
                                    if not tool_name:
                                        return
                                    stream_state.append_tool_delta(
                                        tool_call_id=str(tool_call_id),
                                        tool_name=_client_visible_tool_name(str(tool_name), standard_request.tool_catalog),
                                        partial_json=evt.get("content", ""),
                                    )

                            execution = await collect_completion_run(
                                client,
                                standard_request,
                                current_prompt,
                                capture_events=False,
                                on_delta=on_delta,
                            )
                            should_retry = False
                            try:
                                retry = evaluate_retry_directive(
                                    request=standard_request,
                                    current_prompt=current_prompt,
                                    history_messages=history_messages,
                                    attempt_index=stream_attempt,
                                    max_attempts=max_attempts,
                                    state=execution.state,
                                    allow_after_visible_output=True,
                                )
                                if retry.retry:
                                    current_prompt = retry.next_prompt
                                    should_retry = True
                                else:
                                    if not stream_state.pending_chunks:
                                        stream_state.pending_chunks.append(_message_start_event(
                                            msg_id,
                                            model_name,
                                            current_prompt,
                                            execution.state.answer_text,
                                            extra_prompt_tokens=standard_request.context_attachment_tokens,
                                        ))

                                    stream_state.close_current_block()
                                    directive = build_tool_directive(standard_request, execution.state)
                                    if directive.stop_reason == "tool_use":
                                        stream_state.clear_answer_text()
                                        stream_state.current_block = {"type": None, "index": None, "tool_call_id": None}
                                    else:
                                        stream_state.flush_answer_text()
                                    expected_tool_ids = {
                                        block.get("id")
                                        for block in directive.tool_blocks
                                        if block.get("type") == "tool_use" and block.get("id")
                                    }
                                    for block in directive.tool_blocks:
                                        if block.get("type") != "tool_use":
                                            continue
                                        tool_id = block.get("id")
                                        if tool_id in stream_state.opened_tool_calls:
                                            continue
                                        index = stream_state.open_tool_block(
                                            str(tool_id),
                                            _client_visible_tool_name(str(block.get("name", "")), standard_request.tool_catalog),
                                        )
                                        stream_state.pending_chunks.append(
                                            stream_presenter.anthropic_content_block_delta(index, {'type': 'input_json_delta', 'partial_json': json.dumps(block.get('input', {}), ensure_ascii=False)})
                                        )
                                        stream_state.close_current_block()

                                    visible_answer_length = _visible_answer_text_length(
                                        directive=directive,
                                        execution=execution,
                                        stream_state=stream_state,
                                    )
                                    stop_reason = "tool_use" if expected_tool_ids else "end_turn"
                                    stream_state.pending_chunks.append(stream_presenter.anthropic_message_delta(stop_reason, visible_answer_length))
                                    stream_state.pending_chunks.append(stream_presenter.anthropic_message_stop())

                                    await _add_used_tokens_for_prompt(
                                        users_db=users_db,
                                        token=token,
                                        prompt_text=current_prompt,
                                        answer_text_length=count_tokens(execution.state.answer_text),
                                        extra_prompt_tokens=standard_request.context_attachment_tokens,
                                    )
                            finally:
                                await cleanup_runtime_resources(
                                    client,
                                    execution.acc,
                                    execution.chat_id,
                                )

                            if should_retry:
                                await asyncio.sleep(0.15)
                                await _reacquire_bound_account_if_needed(client=client, standard_request=standard_request)
                                continue

                            heartbeat_interval = settings.STREAM_HEARTBEAT_INTERVAL_SECONDS
                            chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
                            for c in stream_state.pending_chunks:
                                await chunk_queue.put(c)
                            await chunk_queue.put(None)

                            while True:
                                try:
                                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=heartbeat_interval)
                                except asyncio.TimeoutError:
                                    yield ": heartbeat\n\n"
                                    continue
                                if chunk is None:
                                    break
                                yield chunk
                            return
                        except HTTPException as he:
                            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': he.detail}}, ensure_ascii=False)}\n\n"
                            return
                        except Exception as e:
                            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}}, ensure_ascii=False)}\n\n"
                            return

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async with app.state.session_locks.hold(session_key):
            standard_request, effective_payload, model_name, qwen_model, prompt, msg_id = await prepare_locked_request(req_data)
            update_request_context(requested_model=model_name, resolved_model=qwen_model)
            log.info(f"[ANT] model={qwen_model}, stream={standard_request.stream}, tool_enabled={standard_request.tool_enabled}, tools={[t.get('name') for t in standard_request.tools]}, prompt_len={len(prompt)}")
            history_messages = original_history_messages
            try:
                result = await run_retryable_completion_bridge(
                    client=client,
                    standard_request=standard_request,
                    prompt=prompt,
                    users_db=users_db,
                    token=token,
                    history_messages=history_messages,
                    max_attempts=request_max_attempts(standard_request),
                    allow_after_visible_output=True,
                )
                execution = result.execution
                directive = result.directive or build_tool_directive(standard_request, execution.state)
                return JSONResponse(
                    build_anthropic_message_payload(
                        msg_id=msg_id,
                        model_name=model_name,
                        prompt=result.prompt,
                        execution=execution,
                        standard_request=standard_request,
                    )
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
