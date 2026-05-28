import asyncio
import logging

from backend.core.config import settings

log = logging.getLogger("qwen2api.gc")


async def garbage_collect_chats(app):
    """
    ????????? 15 ???????????
    ??? API ??????????? (title ?? api_)?
    ??????? session ?? chat_id ???????????????
    """
    client = app.state.qwen_client
    while True:
        await asyncio.sleep(900)  # 15??
        if not settings.UPSTREAM_AUTO_DELETE_ENABLED:
            log.info("[GC] upstream auto delete disabled")
            continue
        log.info("[GC] ??????????...")
        pool = client.account_pool
        active_chat_ids = app.state.session_affinity.active_chat_ids()
        for acc in pool.accounts:
            if not acc.is_available():
                continue
            try:
                chats = await client.list_chats(acc.token, limit=50, account=acc)
                for c in chats:
                    if not isinstance(c, dict):
                        continue
                    chat_id = str(c.get("id") or "")
                    if not c.get("title", "").startswith("api_"):
                        continue
                    if chat_id and chat_id in active_chat_ids:
                        continue
                    asyncio.create_task(client.delete_chat(acc.token, chat_id, account=acc))
            except Exception as e:
                log.warning(f"[GC] ?? {acc.email} ????: {e}")
