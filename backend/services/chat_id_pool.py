from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from backend.core.config import settings

log = logging.getLogger("qwen2api.chat_id_pool")


def normalize_chat_type(chat_type: str | None) -> str:
    value = str(chat_type or "").strip()
    if value in ("image_gen", "t2i"):
        return "t2i"
    return value or "t2t"


def warm_chat_key(email: str, model: str, chat_type: str | None = "t2t") -> str:
    return f"{str(email or '').strip().lower()}|{str(model or '').strip()}|{normalize_chat_type(chat_type)}"


@dataclass
class WarmChat:
    email: str
    token: str
    model: str
    chat_type: str
    chat_id: str
    created_at: float


class ChatIDPool:
    def __init__(self, client, account_pool) -> None:
        self.client = client
        self.account_pool = account_pool
        self._items: dict[str, list[WarmChat]] = {}
        self._desired: dict[str, tuple[str, str]] = {}
        self._lock = asyncio.Lock()
        self._fill_task: asyncio.Task | None = None

    def enabled(self) -> bool:
        return int(getattr(settings, "CHAT_ID_PREWARM_TARGET_PER_ACCOUNT", 0) or 0) > 0

    async def remember_model(self, model: str, chat_type: str = "t2t") -> None:
        if not self.enabled():
            return
        key = f"{str(model or '').strip()}|{normalize_chat_type(chat_type)}"
        async with self._lock:
            self._desired[key] = (str(model or "").strip(), normalize_chat_type(chat_type))
        self.trigger_fill()

    async def take(self, email: str, model: str, chat_type: str = "t2t") -> tuple[str | None, bool]:
        if not self.enabled():
            return None, False
        await self.cleanup(delete_all=False)
        key = warm_chat_key(email, model, chat_type)
        async with self._lock:
            items = self._items.get(key) or []
            if not items:
                self._items.pop(key, None)
                self.trigger_fill()
                return None, False
            item = items.pop(0)
            if items:
                self._items[key] = items
            else:
                self._items.pop(key, None)
        log.info("[ChatIDPool] reused email=%s model=%s chat_type=%s chat_id=%s", email, model, normalize_chat_type(chat_type), item.chat_id)
        self.trigger_fill()
        return item.chat_id, True

    def trigger_fill(self) -> None:
        if not self.enabled():
            return
        if self._fill_task is not None and not self._fill_task.done():
            return
        self._fill_task = asyncio.create_task(self.fill())

    async def fill(self) -> None:
        if not self.enabled() or self.account_pool is None:
            return
        target = max(0, int(getattr(settings, "CHAT_ID_PREWARM_TARGET_PER_ACCOUNT", 0) or 0))
        max_concurrency = max(1, int(getattr(settings, "CHAT_ID_PREWARM_MAX_CONCURRENCY", 1) or 1))
        async with self._lock:
            desired = list(self._desired.values())
        if not desired:
            return
        accounts = [acc for acc in getattr(self.account_pool, "accounts", []) if acc.is_available()]
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []
        for acc in accounts:
            for model, chat_type in desired:
                missing = target - await self.count(acc.email, model, chat_type)
                for _ in range(max(0, missing)):
                    tasks.append(asyncio.create_task(self._create_warm_chat(semaphore, acc, model, chat_type)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _create_warm_chat(self, semaphore: asyncio.Semaphore, acc, model: str, chat_type: str) -> None:
        async with semaphore:
            try:
                chat_id = await self.client.executor.create_chat(acc, model, chat_type=chat_type)
            except Exception as exc:
                log.warning("[ChatIDPool] create_failed email=%s model=%s chat_type=%s error=%s", acc.email, model, chat_type, exc)
                return
            item = WarmChat(acc.email, acc.token, model, normalize_chat_type(chat_type), chat_id, time.time())
            key = warm_chat_key(item.email, item.model, item.chat_type)
            async with self._lock:
                self._items.setdefault(key, []).append(item)
                cached = len(self._items[key])
            log.info("[ChatIDPool] created email=%s chat_id=%s model=%s chat_type=%s cached=%s", item.email, item.chat_id, item.model, item.chat_type, cached)

    async def count(self, email: str, model: str, chat_type: str = "t2t") -> int:
        key = warm_chat_key(email, model, chat_type)
        async with self._lock:
            return len(self._items.get(key) or [])

    async def cleanup(self, *, delete_all: bool = False) -> list[WarmChat]:
        ttl = max(1, int(getattr(settings, "CHAT_ID_PREWARM_TTL_SECONDS", 120) or 120))
        now = time.time()
        expired: list[WarmChat] = []
        async with self._lock:
            for key, items in list(self._items.items()):
                keep = []
                for item in items:
                    if delete_all or now - item.created_at >= ttl:
                        expired.append(item)
                    else:
                        keep.append(item)
                if keep:
                    self._items[key] = keep
                else:
                    self._items.pop(key, None)
        for item in expired:
            asyncio.create_task(self.client.delete_chat(item.token, item.chat_id, account=self.account_pool.get_by_email(item.email)))
        if expired:
            log.info("[ChatIDPool] cleanup count=%s delete_all=%s", len(expired), delete_all)
        return expired
