from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import (
    RuntimeToolDirective,
    build_tool_directive,
    cleanup_runtime_resources,
    collect_completion_run,
    detect_terminal_tool_loop,
    evaluate_retry_directive,
)
from backend.services.auth_quota import add_used_tokens
from backend.toolcore.task_session import build_retry_rebase_prompt
from backend.services.token_calc import calculate_usage
from backend.toolcall.runtime_tools import (
    is_list_directory_tool_name,
    is_read_tool_name,
    is_shell_tool_name,
    parse_tool_call_arguments,
    tool_target_preview,
)


@dataclass(slots=True)
class CompletionBridgeResult:
    execution: Any
    usage: dict[str, int]
    prompt: str
    attempt_index: int
    directive: Any | None = None


def _truncate_preview(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _build_terminal_tool_guard_message(loop_message: str, history_messages: list[dict[str, Any]] | None) -> str:
    tool_calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    seen_reads: list[str] = []
    seen_exploration: list[str] = []
    result_summaries: list[str] = []

    for msg in history_messages or []:
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id", "") or "")
                fn = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
                tool_name = str(fn.get("name", "") or "")
                args = parse_tool_call_arguments(tool_call)
                if call_id:
                    tool_calls_by_id[call_id] = (tool_name, args)
                target = tool_target_preview(tool_name, args)
                if target and is_read_tool_name(tool_name) and target not in seen_reads:
                    seen_reads.append(target)
                elif target and (is_list_directory_tool_name(tool_name) or is_shell_tool_name(tool_name)) and target not in seen_exploration:
                    seen_exploration.append(target)
        elif msg.get("role") == "tool":
            call_id = str(msg.get("tool_call_id", "") or "")
            call_info = tool_calls_by_id.get(call_id)
            if not call_info:
                continue
            tool_name, args = call_info
            target = tool_target_preview(tool_name, args)
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            preview = _truncate_preview(content)
            if preview:
                label = f"{tool_name}({target})" if target else tool_name
                summary = f"- {label}: {preview}"
                if summary not in result_summaries:
                    result_summaries.append(summary)

    lines = [loop_message]
    if seen_reads:
        lines.append("")
        lines.append("Already inspected file targets:")
        lines.extend(f"- {target}" for target in seen_reads[:6])
    if seen_exploration:
        lines.append("")
        lines.append("Recent exploration targets:")
        lines.extend(f"- {target}" for target in seen_exploration[:6])
    if result_summaries:
        lines.append("")
        lines.append("Recent tool result previews:")
        lines.extend(result_summaries[:4])
    lines.append("")
    lines.append("Suggested next step: continue from the existing tool results, identify the target file to edit, and write the improved script directly instead of calling the same discovery tool again.")
    return "\n".join(lines)


def _apply_terminal_tool_guard(*, execution: Any, directive: RuntimeToolDirective, history_messages: list[dict[str, Any]] | None) -> tuple[Any, RuntimeToolDirective]:
    loop_message = detect_terminal_tool_loop(history_messages, directive)
    if not loop_message:
        return execution, directive
    fallback_message = _build_terminal_tool_guard_message(loop_message, history_messages)

    if dataclasses.is_dataclass(execution.state):
        patched_state = dataclasses.replace(
            execution.state,
            answer_text=fallback_message,
            reasoning_text="",
            tool_calls=[],
            blocked_tool_names=[],
            finish_reason="stop",
        )
    else:
        patched_state = execution.state
        patched_state.answer_text = fallback_message
        patched_state.reasoning_text = ""
        patched_state.tool_calls = []
        patched_state.blocked_tool_names = []
        patched_state.finish_reason = "stop"

    if dataclasses.is_dataclass(execution):
        patched_execution = dataclasses.replace(execution, state=patched_state)
    else:
        patched_execution = execution
        patched_execution.state = patched_state
    patched_directive = RuntimeToolDirective(
        tool_blocks=[{"type": "text", "text": fallback_message}],
        stop_reason="end_turn",
    )
    return patched_execution, patched_directive


async def _reacquire_bound_account_if_needed(*, client, standard_request: StandardRequest) -> None:
    preferred_email = getattr(standard_request, 'bound_account_email', None)
    if preferred_email:
        standard_request.bound_account = await client.account_pool.acquire_wait_preferred(preferred_email, timeout=60)
        if standard_request.bound_account is None and getattr(standard_request, "upstream_files", None):
            raise RuntimeError(f"Unable to reacquire bound account for uploaded context files: {preferred_email}")
    else:
        standard_request.bound_account = None


async def run_completion_bridge(
    *,
    client,
    standard_request: StandardRequest,
    prompt: str,
    users_db,
    token: str,
    usage_delta: int | None = None,
    capture_events: bool = True,
    on_delta: Callable[[dict[str, Any], str | None, list[dict[str, Any]] | None], Awaitable[None]] | None = None,
) -> CompletionBridgeResult:
    execution = await collect_completion_run(
        client,
        standard_request,
        prompt,
        capture_events=capture_events,
        on_delta=on_delta,
    )
    usage = None
    execution_cleaned = False
    try:
        usage = calculate_usage(
            prompt,
            execution.state.answer_text,
            getattr(execution.state, "tool_calls", []),
            extra_prompt_tokens=getattr(standard_request, "context_attachment_tokens", 0),
        )
        await add_used_tokens(users_db, token, usage_delta if usage_delta is not None else usage["total_tokens"])
        await cleanup_runtime_resources(
            client,
            execution.acc,
            execution.chat_id,
            preserve_chat=bool(getattr(standard_request, 'persistent_session', False)),
        )
        execution_cleaned = True
        return CompletionBridgeResult(execution=execution, usage=usage, prompt=prompt, attempt_index=0)
    finally:
        if not execution_cleaned:
            await cleanup_runtime_resources(
                client,
                execution.acc,
                execution.chat_id,
                preserve_chat=bool(getattr(standard_request, 'persistent_session', False)),
            )


async def run_retryable_completion_bridge(
    *,
    client,
    standard_request: StandardRequest,
    prompt: str,
    users_db,
    token: str,
    history_messages: list[dict[str, Any]] | None,
    max_attempts: int,
    usage_delta_factory: Callable[[Any, str], int] | None = None,
    allow_after_visible_output: bool = False,
    capture_events: bool = True,
    on_delta: Callable[[dict[str, Any], str | None, list[dict[str, Any]] | None], Awaitable[None]] | None = None,
    on_attempt_start: Callable[[int, str], Awaitable[None]] | None = None,
    on_retry: Callable[[int, RuntimeRetryDirective, Any], Awaitable[None]] | None = None,
) -> CompletionBridgeResult:
    current_prompt = prompt
    if not getattr(standard_request, 'full_prompt', None):
        standard_request.full_prompt = prompt

    for attempt_index in range(max_attempts):
        if on_attempt_start is not None:
            await on_attempt_start(attempt_index, current_prompt)
        execution = await collect_completion_run(
            client,
            standard_request,
            current_prompt,
            capture_events=capture_events,
            on_delta=on_delta,
        )
        execution_cleaned = False
        try:
            retry = evaluate_retry_directive(
                request=standard_request,
                current_prompt=current_prompt,
                history_messages=history_messages,
                attempt_index=attempt_index,
                max_attempts=max_attempts,
                state=execution.state,
                allow_after_visible_output=allow_after_visible_output,
            )
            if retry.retry:
                if on_retry is not None:
                    await on_retry(attempt_index, retry, execution)
                preserve_chat = bool(getattr(standard_request, 'persistent_session', False))
                await cleanup_runtime_resources(client, execution.acc, execution.chat_id, preserve_chat=preserve_chat)
                execution_cleaned = True

                reused_persistent_chat = bool(getattr(standard_request, 'persistent_session', False) and getattr(standard_request, 'upstream_chat_id', None))
                if reused_persistent_chat:
                    current_prompt = build_retry_rebase_prompt(standard_request, reason=retry.reason)
                else:
                    current_prompt = retry.next_prompt

                if not preserve_chat:
                    await asyncio.sleep(0.15)
                await _reacquire_bound_account_if_needed(client=client, standard_request=standard_request)
                continue

            directive = build_tool_directive(standard_request, execution.state)
            execution, directive = _apply_terminal_tool_guard(
                execution=execution,
                directive=directive,
                history_messages=history_messages,
            )
            usage = calculate_usage(
                current_prompt,
                execution.state.answer_text,
                getattr(execution.state, "tool_calls", []),
                extra_prompt_tokens=getattr(standard_request, "context_attachment_tokens", 0),
            )
            usage_delta = usage_delta_factory(execution, current_prompt) if usage_delta_factory is not None else usage["total_tokens"]
            await add_used_tokens(users_db, token, usage_delta)
            await cleanup_runtime_resources(
                client,
                execution.acc,
                execution.chat_id,
                preserve_chat=bool(getattr(standard_request, 'persistent_session', False)),
            )
            execution_cleaned = True
            return CompletionBridgeResult(
                execution=execution,
                usage=usage,
                prompt=current_prompt,
                attempt_index=attempt_index,
                directive=directive,
            )
        finally:
            if not execution_cleaned:
                await cleanup_runtime_resources(
                    client,
                    execution.acc,
                    execution.chat_id,
                    preserve_chat=bool(getattr(standard_request, 'persistent_session', False)),
                )

    raise RuntimeError("Retryable completion bridge exhausted attempts")
