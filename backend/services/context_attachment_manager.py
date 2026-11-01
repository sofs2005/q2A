from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from backend.core.upstream_file_cache import UpstreamFileCacheEntry
from backend.services.attachment_errors import InlineMediaTooLargeError
from backend.services.client_profiles import user_role_system_text
from backend.services.inline_media_ref import (
    KIND_AUDIO,
    KIND_IMAGE,
    KIND_VIDEO,
    build_inline_ref,
    classify_media,
)
from backend.services.token_calc import count_tokens
from backend.services.standard_request_builder import _excluded_command_like_tool_names
from backend.services.upstream_file_uploader import (
    FileUploadRateLimitedError,
    UpstreamWafChallengeError,
    safe_signed_url_cache_expires_at,
)
from backend.toolcore.context_offload import SYSTEM_CONTEXT_FILE_PREFIX, SYSTEM_CONTEXT_PROMPT_NOTE
from backend.toolcore.request_normalizer import normalize_chat_request, to_prompt_payload


log = logging.getLogger("qwen2api.context_attachment_manager")

# 无法解析签名过期时间时，对带 x-oss-signature 的 URL 使用保守复用窗口（官方常见 300s - 30s）。
_DEFAULT_SIGNED_URL_CACHE_TTL_SECONDS = 270.0


def _remote_url_from_meta(remote_meta: dict[str, Any] | None) -> str:
    if not isinstance(remote_meta, dict):
        return ""
    remote_ref = remote_meta.get("remote_ref")
    if isinstance(remote_ref, dict):
        return str(remote_ref.get("url") or "")
    return str(remote_meta.get("url") or "")


def _resolve_attachment_cache_expires_at(
    remote_meta: dict[str, Any],
    *,
    created_at: float,
    session_ttl_seconds: float,
) -> float:
    """缓存过期取 min(会话 TTL, 签名 URL 安全过期)。

    官方 file_url 通常 x-oss-expires=300；超过后复用会触发 invalid_input。
    """
    session_expires = float(created_at) + max(60.0, float(session_ttl_seconds or 0.0))
    candidates: list[float] = [session_expires]

    url = _remote_url_from_meta(remote_meta)
    safe = safe_signed_url_cache_expires_at(url, now=created_at)
    if safe is not None:
        candidates.append(safe)
    else:
        raw_url_expires = remote_meta.get("url_expires_at")
        if raw_url_expires is not None:
            try:
                candidates.append(float(raw_url_expires) - 30.0)
            except (TypeError, ValueError):
                pass
        elif "x-oss-signature" in url.lower() or "x-oss-expires" in url.lower():
            candidates.append(float(created_at) + _DEFAULT_SIGNED_URL_CACHE_TTL_SECONDS)

    return min(candidates)


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


def _upload_account_excludes(account_pool) -> set:
    """已知文件上传超限的账号邮箱集合；旧账号池无此方法时返回空集。"""
    getter = getattr(account_pool, "file_upload_blocked_emails", None)
    if callable(getter):
        return set(getter() or ())
    return set()


def _browser_upload_enabled(settings_obj: Any, browser_proxy: Any) -> bool:
    """浏览器上传链路是否可用。

    三个条件同时满足才启用：代理实例存在（`app.state.browser_upload_proxy`）、
    `UPLOAD_VIA_BROWSER != never`、`UPLOAD_CDP_ENDPOINT` 已配置。
    """
    if browser_proxy is None:
        return False
    mode = str(getattr(settings_obj, "UPLOAD_VIA_BROWSER", "auto") or "auto").strip().lower()
    if mode == "never":
        return False
    endpoint = str(getattr(settings_obj, "UPLOAD_CDP_ENDPOINT", "") or "").strip()
    return bool(endpoint)


def _text_from_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return str(content or "")


# 图片 / 音频 / 视频走「本地字节 → data URI 内联 ref」：不调上传、不调 parse、
# 不换号、不绑账号（实测上游 `files[i].url` 直接吃 data URI）。
_INLINE_MEDIA_KINDS = frozenset({KIND_IMAGE, KIND_AUDIO, KIND_VIDEO})
_DEFAULT_INLINE_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024


def _is_inline_media_attachment(attachment: Any) -> bool:
    """该附件是否可以直接内联（图片 / 音频 / 视频，且本地有字节）。"""
    if getattr(attachment, "remote_ref", None):
        return False
    if not getattr(attachment, "local_path", ""):
        return False
    kind = classify_media(
        str(getattr(attachment, "filename", "") or ""),
        str(getattr(attachment, "content_type", "") or ""),
    )
    return kind in _INLINE_MEDIA_KINDS


async def _inline_ref_for_attachment(
    attachment: Any,
    *,
    settings_obj: Any,
) -> tuple[dict[str, Any] | None, InlineMediaTooLargeError | None]:
    """读本地字节并构造内联 ref。

    返回 `(ref, too_large_error)`：

    - 成功：`(ref, None)`
    - 读不到 / 开关关闭 / 超过字节护栏：`(None, None)`，调用方回退上传
    - url 字符数超限且压不进上限：`(None, InlineMediaTooLargeError)`，
      调用方回退上传；若上传也不可用（WAF / 无账号）则把它抛出去换 4xx，
      绝不静默丢附件。
    """
    if not bool(getattr(settings_obj, "INLINE_ATTACHMENT_ENABLED", True)):
        return None, None
    max_bytes = int(getattr(settings_obj, "INLINE_ATTACHMENT_MAX_BYTES", _DEFAULT_INLINE_ATTACHMENT_MAX_BYTES) or 0)
    max_url_chars = getattr(settings_obj, "INLINE_ATTACHMENT_MAX_URL_CHARS", None)
    path = str(getattr(attachment, "local_path", "") or "")
    if not path:
        return None, None
    try:
        raw = await asyncio.to_thread(Path(path).read_bytes)
    except OSError as exc:
        log.warning(
            "[ContextAttachment] inline media read failed filename=%s error=%s",
            getattr(attachment, "filename", ""),
            exc,
        )
        return None, None
    if not raw:
        log.warning("[ContextAttachment] inline media is empty filename=%s", getattr(attachment, "filename", ""))
        return None, None
    if max_bytes > 0 and len(raw) > max_bytes:
        log.warning(
            "[ContextAttachment] inline media exceeds limit filename=%s size=%s limit=%s; falling back to upstream upload",
            getattr(attachment, "filename", ""),
            len(raw),
            max_bytes,
        )
        return None, None
    filename = str(getattr(attachment, "filename", "") or "")
    content_type = str(getattr(attachment, "content_type", "") or "")
    try:
        # 图片压缩是 CPU 活（Pillow 解码 + 多轮重编码），放到线程里跑。
        ref = await asyncio.to_thread(
            build_inline_ref,
            filename,
            content_type,
            raw,
            max_url_chars=(int(max_url_chars) if max_url_chars else None),
        )
    except InlineMediaTooLargeError as exc:
        log.warning(
            "[ContextAttachment] inline media too large filename=%s url_chars=%s limit=%s size=%s kind=%s; "
            "falling back to upstream upload",
            filename,
            exc.url_chars,
            exc.limit,
            exc.size,
            exc.kind,
        )
        return None, exc
    return ref, None


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
    # 浏览器上传代理（可选）：内联装不下时，用远程 CDP 浏览器过阿里云拼图验证码换
    # getstsToken 的 STS，再复用 uploader.put_bytes() 把字节 PUT 到 OSS。
    browser_proxy = getattr(app.state, "browser_upload_proxy", None)
    browser_upload_enabled = _browser_upload_enabled(context_offloader.settings, browser_proxy)

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

    # 图片 / 音频 / 视频不需要上游上传与解析，直接内联；文档仍走原 OSS 上传链路。
    inline_media_attachments: list[Any] = []
    upload_attachments: list[Any] = []
    for attachment in manual_attachments:
        if _is_inline_media_attachment(attachment):
            inline_media_attachments.append(attachment)
        else:
            upload_attachments.append(attachment)

    # 内联媒体不依赖上游账号：先处理，失败（读取失败 / 超限 / 开关关闭）才回退上传。
    upstream_files = list(payload.get("upstream_files", []) or [])
    fallback_uploads: list[Any] = []
    inline_too_large_errors: list[InlineMediaTooLargeError] = []
    inline_ok = 0
    for attachment in inline_media_attachments:
        inline_ref, too_large = await _inline_ref_for_attachment(
            attachment,
            settings_obj=context_offloader.settings,
        )
        if inline_ref is None:
            log.warning(
                "[ContextAttachment] inline media unavailable session_key=%s surface=%s filename=%s content_type=%s too_large=%s",
                session_key,
                surface,
                getattr(attachment, "filename", ""),
                getattr(attachment, "content_type", ""),
                too_large is not None,
            )
            if too_large is not None:
                inline_too_large_errors.append(too_large)
            fallback_uploads.append(attachment)
            continue
        upstream_files.append(inline_ref)
        inline_ok += 1
        # 已内联进 payload，本地副本不再需要
        await file_store.delete_path(attachment.local_path)

    # 纯内联媒体：不挑号、不绑号、不上传、不解析。
    if not upload_attachments and not fallback_uploads and not use_generated_context_files and not payload.get("upstream_files"):
        log.info(
            "[ContextAttachment] inline media only session_key=%s surface=%s inline=%s total=%s",
            session_key,
            surface,
            inline_ok,
            len(inline_media_attachments),
        )
        return {
            "payload": payload,
            "session_key": session_key,
            "context_mode": "inline",
            "upstream_files": upstream_files,
            "bound_account": None,
            "bound_account_email": None,
            "generated_local_files": [],
            "context_attachment_tokens": 0,
            "attachment_fallback": bool(inline_media_attachments) and inline_ok == 0,
        }

    requires_sticky_account = bool(manual_attachments or payload.get("upstream_files"))
    record = await affinity.get(session_key) if requires_sticky_account else None
    preferred_email = record.account_email if record else None
    upload_excludes = _upload_account_excludes(account_pool)
    if preferred_email:
        preferred_kwargs: dict[str, Any] = {"timeout": 60}
        if upload_excludes:
            preferred_kwargs["exclude"] = set(upload_excludes)
        acc = await account_pool.acquire_wait_preferred(preferred_email, **preferred_kwargs)
    else:
        acquire_kwargs: dict[str, Any] = {"timeout": 60}
        if upload_excludes:
            acquire_kwargs["exclude"] = set(upload_excludes)
        acc = await account_pool.acquire_wait(**acquire_kwargs)
    if not acc:
        log.warning(
            "[ContextAttachment] no upstream account available; falling back inline session_key=%s surface=%s manual_attachments=%s generated_files=%s",
            session_key,
            surface,
            [getattr(att, "filename", "") for att in manual_attachments],
            [getattr(item, "ext", "") for item in getattr(plan, "generated_files", [])],
        )
        if inline_too_large_errors:
            # 内联放不下 + 没有账号可上传 → 明确 4xx，绝不静默丢附件。
            raise inline_too_large_errors[0]
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

    local_file_records: list[dict[str, Any]] = []
    context_attachment_tokens = 0
    attachment_ttl = context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS

    async def _switch_account_on_retry(current_acc, upload_exc: Exception):
        if not _is_retryable_attachment_upload_error(upload_exc):
            raise upload_exc

        account_pool.release(current_acc)
        exclude = set(upload_excludes)
        current_email = getattr(current_acc, "email", None)
        if current_email:
            exclude.add(current_email)
        next_acc = await account_pool.acquire_wait(timeout=10, exclude=exclude)
        if not next_acc:
            raise upload_exc
        await affinity.bind_account(session_key, surface, next_acc.email, attachment_ttl)
        log.warning(
            "[ContextAttachment] retrying upload with alternate account session_key=%s surface=%s previous_account=%s new_account=%s error=%s",
            session_key,
            surface,
            current_email,
            getattr(next_acc, "email", None),
            upload_exc,
        )
        return next_acc

    async def _switch_account_on_upload_limit(current_acc, rate_exc: FileUploadRateLimitedError):
        account_pool.mark_file_upload_limited(current_acc, rate_exc.retry_after_seconds, str(rate_exc))
        await account_pool.save()
        current_email = getattr(current_acc, "email", None)
        if current_email:
            upload_excludes.add(current_email)
        account_pool.release(current_acc)
        next_acc = await account_pool.acquire_wait(timeout=10, exclude=set(upload_excludes))
        if not next_acc:
            raise rate_exc
        await affinity.bind_account(session_key, surface, next_acc.email, attachment_ttl)
        log.warning(
            "[ContextAttachment] upload quota exceeded; switching account session_key=%s surface=%s previous_account=%s new_account=%s retry_after=%ss",
            session_key,
            surface,
            current_email,
            getattr(next_acc, "email", None),
            int(getattr(rate_exc, "retry_after_seconds", 0) or 0),
        )
        return next_acc

    async def _try_browser_upload(current_acc, local_meta) -> dict[str, Any] | None:
        """浏览器上传：过验证码拿 STS → 复用 `put_bytes()` PUT 到 OSS。

        Returns:
            与 `uploader.upload_local_file()` 同结构的结果字典；
            浏览器侧任何异常 / 未启用都返回 None（调用方继续走原 HTTP 上传链路）。
        """
        if not browser_upload_enabled:
            return None
        # 延迟导入：playwright 是可选依赖，且多数单测不打桩浏览器代理。
        from backend.services.browser_upload_proxy import (
            BrowserUploadError,
            BrowserUploadRateLimited,
            BrowserUploadUnavailable,
        )

        filename = str(local_meta.get("filename") or "")
        content_type = str(local_meta.get("content_type") or "")
        try:
            raw = await asyncio.to_thread(Path(local_meta["path"]).read_bytes)
            if not raw:
                return None
            sts = await browser_proxy.fetch_sts(current_acc, filename, content_type, len(raw))
            if not sts:
                return None
            remote = await uploader.put_bytes(current_acc, sts, raw, filename, content_type)
        except BrowserUploadRateLimited as exc:
            # 上游每日上传配额用尽：翻译成既有异常，复用「标记超限号 → 换号」逻辑。
            raise FileUploadRateLimitedError(
                retry_after_seconds=exc.retry_after_seconds,
                message=str(exc),
            ) from exc
        except BrowserUploadUnavailable as exc:
            log.warning(
                "[ContextAttachment] browser upload unavailable session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                str(exc)[:200],
            )
            return None
        except BrowserUploadError as exc:
            log.warning(
                "[ContextAttachment] browser upload failed session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                str(exc)[:200],
            )
            return None
        except Exception as exc:
            log.warning(
                "[ContextAttachment] browser upload error session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                str(exc)[:200],
            )
            return None
        log.info(
            "[ContextAttachment] browser upload ok session_key=%s surface=%s account=%s filename=%s size=%s",
            session_key,
            surface,
            getattr(current_acc, "email", None),
            filename,
            len(raw),
        )
        return remote

    async def _upload_with_failover(current_acc, local_meta, ext):
        """上传单文件并处理换号回退，返回 (remote, acc, uploaded)。

        - 命中缓存：直接返回 (remote, acc, False)
        - 浏览器上传可用：先试浏览器（过验证码换 STS），失败再走 HTTP 上传
        - RateLimited（每日上传配额）：标记超限号→落盘→排除→换号循环，无号可用则抛出（外层回退内联）
        - 网络类错误：换号重试一次，仍失败则抛出
        """
        network_retry_used = False
        while True:
            cache_entry = await cache.get(session_key, current_acc.email, local_meta["sha256"], ext)
            if cache_entry is not None:
                return cache_entry.remote_file_meta, current_acc, False
            try:
                # 先试浏览器（过验证码换 STS），失败/未启用再走原 HTTP 上传。
                browser_remote = await _try_browser_upload(current_acc, local_meta)
                if browser_remote is not None:
                    return browser_remote, current_acc, True
                remote = await uploader.upload_local_file(current_acc, local_meta)
                return remote, current_acc, True
            except FileUploadRateLimitedError as rate_exc:
                current_acc = await _switch_account_on_upload_limit(current_acc, rate_exc)
            except Exception as upload_exc:
                if network_retry_used:
                    raise
                network_retry_used = True
                current_acc = await _switch_account_on_retry(current_acc, upload_exc)

    try:
        # 内联失败（读取失败 / 超限）的媒体在这里补走一次上传链路。
        for attachment in list(upload_attachments) + list(fallback_uploads):
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
            remote, acc, uploaded = await _upload_with_failover(acc, local_meta, ext)
            if uploaded:
                cache_expires = _resolve_attachment_cache_expires_at(
                    remote,
                    created_at=local_meta["created_at"],
                    session_ttl_seconds=attachment_ttl,
                )
                await cache.set(UpstreamFileCacheEntry(
                    session_key=session_key,
                    account_email=acc.email,
                    sha256=local_meta["sha256"],
                    ext=ext,
                    filename=attachment.filename,
                    remote_file_meta=remote,
                    created_at=local_meta["created_at"],
                    expires_at=cache_expires,
                ))
            upstream_files.append(remote["remote_ref"])
            await affinity.add_uploaded_file(session_key, remote)
            await file_store.delete_path(attachment.local_path)

        if use_generated_context_files:
            for generated in plan.generated_files:
                filename = _generated_context_filename(generated.ext)
                local_meta = await file_store.save_text(filename, generated.text, generated.content_type, purpose="context")
                remote, acc, uploaded = await _upload_with_failover(acc, local_meta, generated.ext)
                if uploaded:
                    cache_expires = _resolve_attachment_cache_expires_at(
                        remote,
                        created_at=local_meta["created_at"],
                        session_ttl_seconds=attachment_ttl,
                    )
                    await cache.set(UpstreamFileCacheEntry(
                        session_key=session_key,
                        account_email=acc.email,
                        sha256=local_meta["sha256"],
                        ext=generated.ext,
                        filename=filename,
                        remote_file_meta=remote,
                        created_at=local_meta["created_at"],
                        expires_at=cache_expires,
                    ))
                upstream_files.append(remote["remote_ref"])
                context_attachment_tokens += count_tokens(generated.text)
                await affinity.add_uploaded_file(session_key, remote)
                await file_store.delete_path(local_meta["path"])
                local_file_records.append(local_meta)
    except Exception as exc:
        # 上游上传被 WAF 人机验证拦住是**预期路径**（getstsToken 现在必被拦），
        # 只打一行 warning、不打整栈 ERROR；其它异常保留 exception 便于排查。
        if isinstance(exc, UpstreamWafChallengeError):
            log.warning(
                "[ContextAttachment] upstream upload blocked by WAF session_key=%s surface=%s account=%s error=%s",
                session_key,
                surface,
                getattr(acc, "email", None),
                str(exc)[:200],
            )
        else:
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
        if inline_too_large_errors:
            # 内联放不下 + 上传回退不可用 → 明确 4xx，绝不静默丢附件。
            raise inline_too_large_errors[0] from exc
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
