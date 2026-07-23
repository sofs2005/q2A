from __future__ import annotations

import asyncio
import logging

from backend.core.config import settings
from backend.core.request_logging import request_context

log = logging.getLogger("qwen2api.context_cleanup")


async def context_cleanup_loop(app, interval_seconds: int = 300):
    while True:
        try:
            with request_context(surface="context-cleanup"):
                ttl = app.state.context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS
                # 上下文附件：通用 TTL，但跳过 generated_image（其使用独立 TTL）。
                cleaned_context = await app.state.file_store.cleanup_expired(
                    ttl, exclude_purpose="generated_image"
                )
                image_ttl = int(getattr(settings, "GENERATED_IMAGE_TTL_SECONDS", ttl) or ttl)
                cleaned_images = await app.state.file_store.cleanup_expired(
                    image_ttl, purpose="generated_image"
                )
                expired_records = await app.state.session_affinity.cleanup_expired()
                await app.state.upstream_file_cache.cleanup_expired()
                if settings.UPSTREAM_AUTO_DELETE_ENABLED:
                    for record in expired_records:
                        acc = app.state.account_pool.get_by_email(record.account_email)
                        if not acc:
                            continue
                        if record.chat_id:
                            try:
                                await app.state.qwen_client.delete_chat(acc.token, record.chat_id, account=acc)
                            except Exception as exc:
                                log.warning("[ContextCleanup] chat delete failed session=%s chat_id=%s error=%s", record.session_key, record.chat_id, exc)
                        for remote_meta in record.uploaded_files:
                            try:
                                await app.state.upstream_file_uploader.delete_remote_file(acc, remote_meta)
                            except Exception as exc:
                                log.debug("[ContextCleanup] remote delete failed session=%s error=%s", record.session_key, exc)
                log.info(
                    "[ContextCleanup] ttl=%s image_ttl=%s cleaned_files=%s cleaned_images=%s expired_sessions=%s completed",
                    ttl,
                    image_ttl,
                    cleaned_context,
                    cleaned_images,
                    len(expired_records),
                )
        except Exception as exc:
            log.warning("[ContextCleanup] failed: %s", exc)
        await asyncio.sleep(max(60, interval_seconds))
