from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from backend.core.upstream_file_cache import UpstreamFileCacheEntry
from backend.services.client_profiles import user_role_system_text
from backend.services.token_calc import count_tokens
from backend.services.standard_request_builder import _excluded_command_like_tool_names
from backend.toolcore.context_offload import SYSTEM_CONTEXT_FILE_PREFIX, SYSTEM_CONTEXT_PROMPT_NOTE
from backend.toolcore.request_normalizer import normalize_chat_request, to_prompt_payload


log = logging.getLogger("qwen2api.context_attachment_manager")


def _is_retryable_attachment_upload_error(exc: Exception) -> bool:
    lowered = str(exc or "").lower()
    markers = (
        "temporary failure in name resolution",
        "failed to resolve",
        "nameresolutionerror",
        "connecterror",
        "connectionerror",
        "max retries exceeded",
        "newconnectionerror",
        "timed out",
        "timeout",
    )
    return any(marker in lowered for marker in markers)


def _text_from_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return str(content or "")


def build_request_session_key(surface: str, request_id: str) -> str:
    return f"{surface}:{request_id}"


def derive_session_key(surface: str, auth_token: str, payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    explicit = payload.get("session_key") or payload.get("conversation_id") or metadata.get("conversation_id")
    messages = payload.get("messages", []) or []
    first_user = next((m for m in messages if m.get("role") == "user"), {})
    first_text = _text_from_content(first_user.get("content", ""))
    system_text = _text_from_content(payload.get("system", ""))
    developer_parts = [_text_from_content(payload.get("developer", ""))]
    developer_parts.extend(
        _text_from_content(message.get("content", ""))
        for message in messages
        if message.get("role") in {"system", "developer"}
    )
    developer_parts.extend(
        user_system_text
        for message in messages
        if message.get("role") == "user"
        for user_system_text in [user_role_system_text(_text_from_content(message.get("content", "")))]
        if user_system_text
    )
    developer_text = "\n".join(part for part in developer_parts if part)
    instructions_text = _text_from_content(payload.get("instructions", ""))
    persona_basis = f"{system_text[:400]}::{developer_text[:400]}::{instructions_text[:400]}"
    explicit_text = str(explicit or "")
    user_basis = "" if explicit else f"::{first_text[:400]}"
    basis = f"{surface}::{auth_token}::{payload.get('model', '')}::{explicit_text}::{persona_basis}{user_basis}"
    return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _context_messages_for_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    context_messages: list[dict[str, Any]] = []
    for field_name, role in (("system", "system"), ("developer", "developer"), ("instructions", "system")):
        text = _text_from_content(payload.get(field_name, ""))
        if text.strip():
            context_messages.append({"role": role, "content": text})
    context_messages.extend(payload.get("messages", []) or [])
    return context_messages


def _generated_context_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext}"


def _fallback_context_attachment_result(
    *,
    payload: dict[str, Any],
    session_key: str,
    plan,
    use_generated_context_files: bool,
    manual_attachments: list[Any],
    fallback_message: str,
) -> dict[str, Any]:
    fallback_payload = dict(payload)
    summary_parts: list[str] = []
    if use_generated_context_files and plan.summary_text:
        summary_parts.append(plan.summary_text)
    if manual_attachments:
        names = ", ".join(att.filename for att in manual_attachments[:4])
        summary_parts.append(
            f"User attachments were provided but attachment upload failed. Attachment names: {names}"
        )
    fallback_messages: list[dict[str, Any]] = []
    if summary_parts:
        fallback_messages.append({
            "role": "user",
            "content": f"{summary_parts[0]}\n\n{SYSTEM_CONTEXT_PROMPT_NOTE}",
        })
    elif fallback_message:
        fallback_messages.append({"role": "user", "content": fallback_message})
    if use_generated_context_files and getattr(plan, "inline_messages", None):
        fallback_messages.extend(plan.inline_messages)
    else:
        fallback_messages.extend(payload.get("messages", []) or [])
    fallback_payload["messages"] = fallback_messages
    return {
        "payload": fallback_payload,
        "session_key": session_key,
        "context_mode": "inline",
        "upstream_files": list(payload.get("upstream_files", []) or []),
        "bound_account": None,
        "bound_account_email": None,
        "generated_local_files": [],
        "context_attachment_tokens": 0,
        "attachment_fallback": True,
    }


def _context_tools_for_payload(payload: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    tools = payload.get("tools", []) or []
    if surface not in {"openai", "responses"}:
        return tools
    try:
        normalized = normalize_chat_request(payload, excluded_tool_names=_excluded_command_like_tool_names(payload))
        prompt_payload = to_prompt_payload(
            normalized,
            model=str(payload.get("model") or ""),
            stream=bool(payload.get("stream", False)),
        )
        return prompt_payload.get("tools", []) or []
    except Exception:
        return tools


async def prepare_context_attachments(*, app, payload: dict[str, Any], surface: str, auth_token: str, client_profile: str, session_key: str, existing_attachments=None) -> dict[str, Any]:
    context_offloader = app.state.context_offloader
    account_pool = app.state.account_pool
    file_store = app.state.file_store
    affinity = app.state.session_affinity
    cache = app.state.upstream_file_cache
    uploader = app.state.upstream_file_uploader

    tools = _context_tools_for_payload(payload, surface)
    messages = payload.get("messages", []) or []
    context_messages = _context_messages_for_payload(payload)
    manual_attachments = list(existing_attachments or [])
    plan = context_offloader.plan(context_messages, tools=tools, client_profile=client_profile)
    use_generated_context_files = bool(plan.generated_files)
    if not use_generated_context_files and not manual_attachments:
        return {
            "payload": payload,
            "session_key": session_key,
            "context_mode": "inline",
            "upstream_files": list(payload.get("upstream_files", []) or []),
            "bound_account": None,
            "bound_account_email": None,
            "generated_local_files": [],
            "context_attachment_tokens": 0,
            "attachment_fallback": False,
        }

    requires_sticky_account = bool(manual_attachments or payload.get("upstream_files"))
    record = await affinity.get(session_key) if requires_sticky_account else None
    preferred_email = record.account_email if record else None
    if preferred_email:
        acc = await account_pool.acquire_wait_preferred(preferred_email, timeout=60)
    else:
        acc = await account_pool.acquire_wait(timeout=60)
    if not acc:
        log.warning(
            "[ContextAttachment] no upstream account available; falling back inline session_key=%s surface=%s manual_attachments=%s generated_files=%s",
            session_key,
            surface,
            [getattr(att, "filename", "") for att in manual_attachments],
            [getattr(item, "ext", "") for item in getattr(plan, "generated_files", [])],
        )
        return _fallback_context_attachment_result(
            payload=payload,
            session_key=session_key,
            plan=plan,
            use_generated_context_files=use_generated_context_files,
            manual_attachments=manual_attachments,
            fallback_message=(
                "No upstream account was available for attachment upload. "
                "Continue with the available inline context only."
            ),
        )
    await affinity.bind_account(session_key, surface, acc.email, context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS)

    upstream_files = list(payload.get("upstream_files", []) or [])
    local_file_records: list[dict[str, Any]] = []
    context_attachment_tokens = 0

    async def _switch_account_on_retry(current_acc, upload_exc: Exception):
        if not _is_retryable_attachment_upload_error(upload_exc):
            raise upload_exc

        exclude = {getattr(current_acc, "email", None)}
        account_pool.release(current_acc)
        next_acc = await account_pool.acquire_wait(timeout=10, exclude=exclude)
        if not next_acc:
            raise upload_exc
        await affinity.bind_account(session_key, surface, next_acc.email, context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS)
        log.warning(
            "[ContextAttachment] retrying upload with alternate account session_key=%s surface=%s previous_account=%s new_account=%s error=%s",
            session_key,
            surface,
            getattr(current_acc, "email", None),
            getattr(next_acc, "email", None),
            upload_exc,
        )
        return next_acc

    try:
        for attachment in manual_attachments:
            if getattr(attachment, "remote_ref", None):
                upstream_files.append(attachment.remote_ref)
                continue
            if not attachment.local_path:
                continue
            local_meta = {
                "id": attachment.file_id,
                "path": attachment.local_path,
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "sha256": attachment.sha256,
                "created_at": __import__("time").time(),
            }
            ext = Path(attachment.filename).suffix.lstrip(".").lower()
            cache_entry = await cache.get(session_key, acc.email, local_meta["sha256"], ext)
            if cache_entry is not None:
                remote = cache_entry.remote_file_meta
            else:
                try:
                    remote = await uploader.upload_local_file(acc, local_meta)
                except Exception as upload_exc:
                    acc = await _switch_account_on_retry(acc, upload_exc)
                    cache_entry = await cache.get(session_key, acc.email, local_meta["sha256"], ext)
                    if cache_entry is not None:
                        remote = cache_entry.remote_file_meta
                    else:
                        remote = await uploader.upload_local_file(acc, local_meta)
                await cache.set(UpstreamFileCacheEntry(
                    session_key=session_key,
                    account_email=acc.email,
                    sha256=local_meta["sha256"],
                    ext=ext,
                    filename=attachment.filename,
                    remote_file_meta=remote,
                    created_at=local_meta["created_at"],
                    expires_at=local_meta["created_at"] + context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS,
                ))
            upstream_files.append(remote["remote_ref"])
            await affinity.add_uploaded_file(session_key, remote)
            await file_store.delete_path(attachment.local_path)

        if use_generated_context_files:
            for generated in plan.generated_files:
                filename = _generated_context_filename(generated.ext)
                local_meta = await file_store.save_text(filename, generated.text, generated.content_type, purpose="context")
                cache_entry = await cache.get(session_key, acc.email, local_meta["sha256"], generated.ext)
                if cache_entry is not None:
                    remote = cache_entry.remote_file_meta
                else:
                    try:
                        remote = await uploader.upload_local_file(acc, local_meta)
                    except Exception as upload_exc:
                        acc = await _switch_account_on_retry(acc, upload_exc)
                        cache_entry = await cache.get(session_key, acc.email, local_meta["sha256"], generated.ext)
                        if cache_entry is not None:
                            remote = cache_entry.remote_file_meta
                        else:
                            remote = await uploader.upload_local_file(acc, local_meta)
                    await cache.set(UpstreamFileCacheEntry(
                        session_key=session_key,
                        account_email=acc.email,
                        sha256=local_meta["sha256"],
                        ext=generated.ext,
                        filename=filename,
                        remote_file_meta=remote,
                        created_at=local_meta["created_at"],
                        expires_at=local_meta["created_at"] + context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS,
                    ))
                upstream_files.append(remote["remote_ref"])
                context_attachment_tokens += count_tokens(generated.text)
                await affinity.add_uploaded_file(session_key, remote)
                await file_store.delete_path(local_meta["path"])
                local_file_records.append(local_meta)
    except Exception as exc:
        log.exception(
            "[ContextAttachment] upload failed session_key=%s surface=%s account=%s manual_attachments=%s generated_files=%s error=%s",
            session_key,
            surface,
            getattr(acc, "email", None),
            [getattr(att, "filename", "") for att in manual_attachments],
            [getattr(item, "ext", "") for item in getattr(plan, "generated_files", [])],
            exc,
        )
        account_pool.release(acc)
        return _fallback_context_attachment_result(
            payload=payload,
            session_key=session_key,
            plan=plan,
            use_generated_context_files=use_generated_context_files,
            manual_attachments=manual_attachments,
            fallback_message="Attachment upload failed. Continue with the available inline context only.",
        )

    rewritten = dict(payload)
    rewritten["messages"] = plan.inline_messages if use_generated_context_files else messages
    return {
        "payload": rewritten,
        "session_key": session_key,
        "context_mode": plan.mode if use_generated_context_files else "inline",
        "upstream_files": upstream_files,
        "bound_account": acc,
        "bound_account_email": acc.email,
        "generated_local_files": local_file_records,
        "context_attachment_tokens": context_attachment_tokens,
        "attachment_fallback": False,
    }
