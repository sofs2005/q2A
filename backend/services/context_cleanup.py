from __future__ import annotations

import asyncio
import logging

from backend.core.config import settings
from backend.core.request_logging import request_context
from backend.services.media_purposes import MEDIA_PURPOSES

log = logging.getLogger("qwen2api.context_cleanup")


async def context_cleanup_loop(app, interval_seconds: int = 300):
    while True:
        try:
            with request_context(surface="context-cleanup"):
                ttl = app.state.context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS
                # 上下文附件：通用 TTL，但跳过所有生成物（生成物有独立的保留策略）。
                cleaned_context = await app.state.file_store.cleanup_expired(
                    ttl, exclude_purpose=list(MEDIA_PURPOSES)
                )
                # 生成物（图片/视频）：GENERATED_MEDIA_TTL_SECONDS>0 才按用途逐个清理；
                # ==0 表示永久保留（媒体库语义），完全跳过删除。
                media_ttl = int(getattr(settings, "GENERATED_MEDIA_TTL_SECONDS", 0) or 0)
                cleaned_media = 0
                if media_ttl > 0:
                    for purpose in MEDIA_PURPOSES:
                        cleaned_media += await app.state.file_store.cleanup_expired(
                            media_ttl, purpose=purpose
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
                    "[ContextCleanup] ttl=%s media_ttl=%s cleaned_files=%s cleaned_media=%s expired_sessions=%s completed",
                    ttl,
                    media_ttl,
                    cleaned_context,
                    cleaned_media,
                    len(expired_records),
                )
        except Exception as exc:
            log.warning("[ContextCleanup] failed: %s", exc)
        await asyncio.sleep(max(60, interval_seconds))
