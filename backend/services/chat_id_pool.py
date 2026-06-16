from __future__ import annotations

import asyncio
import hashlib
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
        self._desired: dict[str, tuple[str, str]] = self._configured_desired_models()
        self._lock = asyncio.Lock()
        self._fill_task: asyncio.Task | None = None
        self._fill_round = 0  # 预热轮次计数：用于每轮相位轮换，避免固定账号始终先打上游

    def _configured_desired_models(self) -> dict[str, tuple[str, str]]:
        desired: dict[str, tuple[str, str]] = {}
        raw_models = str(getattr(settings, "CHAT_ID_PREWARM_MODELS", "") or "")
        for raw_model in raw_models.split(","):
            model = raw_model.strip()
            if not model:
                continue
            chat_type = "t2t"
            desired[f"{model}|{chat_type}"] = (model, chat_type)
        return desired

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
        now = time.time()
        max_inflight = int(getattr(self.account_pool, "max_inflight", 1) or 1)
        accounts = [
            acc
            for acc in getattr(self.account_pool, "accounts", [])
            if acc.is_available()
            and int(getattr(acc, "inflight", 0) or 0) < max_inflight
            and float(getattr(acc, "next_available_at", lambda: 0.0)()) <= now
        ]
        semaphore = asyncio.Semaphore(max_concurrency)
        # 每轮相位轮换：按轮次旋转账号起跑顺序，避免同一账号每轮都第一个打上游
        self._fill_round += 1
        total = len(accounts)
        offset = self._fill_round % total if total else 0
        tasks = []
        for idx, acc in enumerate(accounts):
            # slot 决定该账号在错峰窗口内的相对起跑位置（0..total-1），随轮次旋转
            slot = (idx + offset) % total if total else 0
            for model, chat_type in desired:
                missing = target - await self.count(acc.email, model, chat_type)
                for _ in range(max(0, missing)):
                    tasks.append(asyncio.create_task(
                        self._create_warm_chat(semaphore, acc, model, chat_type, slot, total)
                    ))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _jitter(email: str, model: str, chat_type: str) -> float:
        """基于 email+model+chat_type 哈希的确定性抖动（0~JITTER_SECONDS）。

        在序位铺开的基础上叠加抖动，打散"同一序位、不同模型"的并发请求，
        并让相邻账号的起跑时间不至于过于规整，进一步弱化请求脉冲特征。
        """
        jitter_max = max(0.0, float(getattr(settings, "CHAT_ID_PREWARM_JITTER_SECONDS", 1.5) or 0))
        if jitter_max <= 0:
            return 0.0
        key = f"{str(email or '').strip().lower()}|{str(model or '').strip()}|{normalize_chat_type(chat_type)}"
        digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()
        return (int(digest, 16) % 1000) / 1000.0 * jitter_max

    @staticmethod
    def _spread_delay(slot: int, total: int, email: str, model: str, chat_type: str) -> float:
        """账号错峰延迟：在 SPREAD_SECONDS 窗口内按序位 slot 均匀铺开 + 抖动。

        相比旧的纯哈希抖动（固定 0~2s 且账号顺序永远相同），序位均匀铺开能
        随账号数量自适应拉开间距，配合 fill() 的轮次相位轮换实现真正的错开。
        """
        spread = max(0.0, float(getattr(settings, "CHAT_ID_PREWARM_SPREAD_SECONDS", 6) or 0))
        base = (spread * slot / total) if (spread > 0 and total > 0) else 0.0
        return base + ChatIDPool._jitter(email, model, chat_type)

    async def _create_warm_chat(self, semaphore: asyncio.Semaphore, acc, model: str, chat_type: str, slot: int = 0, total: int = 1) -> None:
        # 先错峰再获取信号量：按账号序位在错峰窗口内铺开起跑时间，
        # 避免高并发下所有任务同时拿到信号量后仍形成密集请求脉冲触发上游风控
        delay = self._spread_delay(slot, total, getattr(acc, "email", ""), model, chat_type)
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            try:
                chat_id = await self.client.executor.create_chat(acc, model, chat_type=chat_type)
            except Exception as exc:
                log.warning("[ChatIDPool] create_failed email=%s model=%s chat_type=%s error=%s", acc.email, model, chat_type, exc)
                # 预热路径遇到鉴权失败时也触发后台自愈：否则只在实际请求重试分支
                # 触发，仅被预热碰到的过期账号永远不会被刷新（healing 标志位保证幂等）。
                err = str(exc).lower()
                if "unauthorized" in err or "expired" in err or "401" in err or "403" in err:
                    resolver = getattr(self.client.executor, "auth_resolver", None)
                    if resolver is not None:
                        asyncio.create_task(resolver.auto_heal_account(acc))
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
