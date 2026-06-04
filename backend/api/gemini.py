from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import logging
from typing import Any

from backend.core.config import resolve_model
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.runtime import stream_presenter
from backend.services.command_environment import detect_command_environment, format_command_environment_hint
from backend.services.completion_bridge import run_retryable_completion_bridge
from backend.services.auth_quota import resolve_auth_context
from backend.services.response_formatters import build_gemini_generate_payload
from backend.services.standard_request_builder import build_chat_standard_request
from backend.toolcore.request_normalizer import normalize_gemini_request, to_prompt_payload

log = logging.getLogger("qwen2api.gemini")
router = APIRouter()

GEMINI_STREAM_MEDIA_TYPE = "application/json"


def _gemini_to_chat_payload(model: str, body: dict[str, Any], *, force_stream: bool | None = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in body.get("contents", []) or []:
        role = "assistant" if message.get("role") == "model" else "user"
        text_parts: list[str] = []
        for part in message.get("parts", []) or []:
            text = part.get("text")
            if text:
                text_parts.append(text)
        if text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    tools: list[dict[str, Any]] = []
    for tool in body.get("tools", []) or []:
        declarations = tool.get("functionDeclarations", []) if isinstance(tool, dict) else []
        for declaration in declarations or []:
            if not isinstance(declaration, dict):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": declaration.get("name", ""),
                    "description": declaration.get("description", ""),
                    "parameters": declaration.get("parameters", {}),
                },
            })

    tool_choice: Any = None
    tool_config = body.get("toolConfig")
    if isinstance(tool_config, dict):
        function_calling = tool_config.get("functionCallingConfig")
        if isinstance(function_calling, dict):
            mode = str(function_calling.get("mode") or "").strip().upper()
            if mode == "NONE":
                tool_choice = "none"
            elif mode == "ANY":
                tool_choice = "required"
            elif mode == "AUTO":
                tool_choice = "auto"

    stream_requested = _is_gemini_stream_request(body) if force_stream is None else force_stream
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "stream": stream_requested,
    }


def _is_gemini_stream_request(body: dict[str, Any]) -> bool:
    if body.get("stream") is True:
        return True
    generation_config = body.get("generationConfig")
    if isinstance(generation_config, dict) and generation_config.get("stream") is True:
        return True
    return False


def _build_standard_request(model: str, body: dict, *, stream: bool | None = None, command_environment=None):
    normalized_request = normalize_gemini_request(body, model=model, force_stream=stream)
    payload = to_prompt_payload(normalized_request, model=model, stream=_is_gemini_stream_request(body) if stream is None else stream)
    return build_chat_standard_request(
        payload,
        default_model=model,
        surface="gemini",
        client_profile="openclaw_openai",
        command_environment=command_environment,
    )


def _gemini_chunk_payload(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                }
            }
        ]
    }


async def _load_and_validate_request(request: Request, model: str, *, force_stream: bool | None = None):
    app = request.app
    users_db = app.state.users_db
    client = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    body = await request.json()
    command_environment = detect_command_environment(headers=request.headers, request_data=body)
    standard_request = _build_standard_request(model, body, stream=force_stream, command_environment=command_environment)
    update_request_context(
        resolved_model=standard_request.resolved_model,
        command_environment=format_command_environment_hint(command_environment),
    )
    return users_db, client, token, standard_request


@router.post("/v1beta/models/{model}:generateContent")
@router.post("/v1/models/{model}:generateContent")
@router.post("/models/{model}:generateContent")
async def gemini_generate_content(model: str, request: Request):
    with request_context(req_id=new_request_id(), surface="gemini", requested_model=model):
        users_db, client, token, standard_request = await _load_and_validate_request(request, model, force_stream=False)
        content = standard_request.prompt
        log.info(f"[Gemini] route=generateContent model={standard_request.resolved_model}, stream={standard_request.stream}, prompt_len={len(content)}")

        try:
            result = await run_retryable_completion_bridge(
                client=client,
                standard_request=standard_request,
                prompt=content,
                users_db=users_db,
                token=token,
                history_messages=standard_request.tools and [] or [],
                max_attempts=2 if standard_request.tools else 3,
                allow_after_visible_output=True,
            )
            execution = result.execution
        except Exception as e:
            log.error(f"Gemini proxy failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        log.info(f"[Gemini] Request complete. Generated {len(execution.state.answer_text)} characters.")
        return JSONResponse(build_gemini_generate_payload(execution=execution))


@router.post("/v1beta/models/{model}:streamGenerateContent")
@router.post("/v1/models/{model}:streamGenerateContent")
@router.post("/models/{model}:streamGenerateContent")
async def gemini_stream_generate_content(model: str, request: Request):
    with request_context(req_id=new_request_id(), surface="gemini", requested_model=model):
        users_db, client, token, standard_request = await _load_and_validate_request(request, model, force_stream=True)
        content = standard_request.prompt
        log.info(f"[Gemini] route=streamGenerateContent model={standard_request.resolved_model}, stream={standard_request.stream}, prompt_len={len(content)}")

        async def generate():
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def on_delta(evt, text_chunk, _):
                if text_chunk and evt.get("phase") == "answer":
                    await queue.put(stream_presenter.gemini_text_chunk(text_chunk))

            async def runner():
                try:
                    result = await run_retryable_completion_bridge(
                        client=client,
                        standard_request=standard_request,
                        prompt=content,
                        users_db=users_db,
                        token=token,
                        history_messages=standard_request.tools and [] or [],
                        max_attempts=2 if standard_request.tools else 3,
                        allow_after_visible_output=True,
                        capture_events=False,
                        on_delta=on_delta,
                    )
                    log.info(f"[Gemini] Request complete. Generated {len(result.execution.state.answer_text)} characters.")
                except Exception as e:
                    await queue.put(stream_presenter.gemini_error_chunk(str(e)))
                finally:
                    await queue.put(None)

            task = asyncio.create_task(runner())
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
            await task

        return StreamingResponse(generate(), media_type=GEMINI_STREAM_MEDIA_TYPE)
