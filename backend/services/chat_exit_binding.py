"""chat_id ↔ 出口 IP 绑定表（内存 LRU + TTL）。

背景（2026-12 生产故障）
------------------------
阿里云 WAF 的惩罚是**出口 IP 级**的：同一个出口把 `chats/new` 打挂之后，该出口
后续所有请求都吃拼图挑战页，于是 `ChatIDPool` 建会话全面失败，网关直接不可用
（`No available accounts in pool`）。解法是把 `chats/new` 也挂到出口池上。

但上游的**会话与出口 IP 绑定**：在出口 A 建的 chat，用出口 B 发 completions
等于「异地登录」，不但不认，还会加重风控。所以必须记住「谁在哪个出口建的」，
后续同一 chat_id 的请求原路返回同一个出口。

本模块只做三件事：

1. `bind()` —— 建会话成功后登记 `chat_id → (exit_id, proxy_url)`；
2. `resolve()` —— 后续请求按 chat_id 查出口（命中即 LRU 提权，过期即失效）；
3. 容量控制 —— 超过 `max_entries` 按 LRU（最久未用）淘汰，避免长跑进程内存膨胀。

设计要点
--------
- **纯内存**：出口绑定是「运行期亲和性」，不需要跨重启持久化。进程重启后
  预热池里的 chat_id 也全废了（`ChatIDPool` 同样只在内存），重新建会话即可。
- **TTL 默认 1800s**：与上游会话的活跃窗口量级一致。过期后不再返回绑定，
  调用方据此重建会话（`QwenExecutor.get_chat_id`），不会拿旧出口硬打。
- **代理串不落日志**：与 `exit_pool` 同一口径，日志里只打出口后缀（`Default.g`）。
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("qwen2api.chat_exit")

#: 绑定默认存活时长（秒）：可被 `CHAT_EXIT_BINDING_TTL_SECONDS` 覆盖。
DEFAULT_TTL_SECONDS = 1800.0
#: 绑定表默认容量上限：可被 `CHAT_EXIT_BINDING_MAX_ENTRIES` 覆盖。
DEFAULT_MAX_ENTRIES = 512

#: `CHAT_VIA_EXIT_POOL` 的合法取值；非法值一律按 `auto` 处理
#: （写错一个 env 不应该把建会话链路打死）。
_VALID_MODES = ("auto", "always", "never")


def resolve_chat_exit_mode(settings_obj: Any) -> str:
    """解析 `CHAT_VIA_EXIT_POOL`：auto / always / never，非法值按 auto。

    Args:
        settings_obj: 配置对象（`backend.core.config.settings` 或单测替身）。

    Returns:
        `"auto"` / `"always"` / `"never"` 之一；`never` 表示建会话链路一行代理都不碰。
    """
    raw = str(getattr(settings_obj, "CHAT_VIA_EXIT_POOL", "auto") or "auto").strip().lower()
    return raw if raw in _VALID_MODES else "auto"


def _as_float(value: Any, default: float) -> float:
    """把配置值转成正浮点数；非法 / 负数退回默认值。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _as_int(value: Any, default: int) -> int:
    """把配置值转成正整数；非法 / 小于 1 退回默认值。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


@dataclass(frozen=True)
class ChatExitBinding:
    """一条 chat_id → 出口的绑定记录（不可变快照）。"""

    chat_id: str
    exit_id: str
    proxy_url: str
    bound_at: float
    expires_at: float

    def to_public_dict(self) -> dict[str, Any]:
        """给探针 / 日志用的安全视图：**不含**代理凭据与 host。"""
        return {
            "chat_id": self.chat_id,
            "exit_id": self.exit_id,
            "bound_at": round(self.bound_at, 1),
            "expires_at": round(self.expires_at, 1),
        }


class ChatExitBindings:
    """chat_id → 出口的内存映射（LRU + TTL）。

    并发安全：所有方法都是同步的纯 dict 读写（`OrderedDict` 的插入 / 提权 /
    弹出），在同一个事件循环里天然原子，不需要锁。
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """初始化绑定表。

        Args:
            ttl_seconds: 绑定存活时长（秒）。<=0 视为「未配置」，退回默认 1800s
                —— 建会话与 completions 之间若没有绑定，出口亲和性就没了。
            max_entries: 容量上限，超出后按最久未用淘汰。
            clock: 时钟函数（单测注入假时钟），默认 `time.monotonic`。
        """
        raw_ttl = _as_float(ttl_seconds, DEFAULT_TTL_SECONDS)
        self.ttl_seconds = raw_ttl if raw_ttl > 0 else DEFAULT_TTL_SECONDS
        self.max_entries = _as_int(max_entries, DEFAULT_MAX_ENTRIES)
        self._clock = clock or time.monotonic
        self._items: OrderedDict[str, ChatExitBinding] = OrderedDict()

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings_obj: Any) -> "ChatExitBindings":
        """按配置构造绑定表（TTL / 容量上限都可覆盖）。

        Args:
            settings_obj: 配置对象；缺字段时用模块默认值。

        Returns:
            绑定表实例（**永不返回 None**：即使出口池未启用也照常创建，
            只是个永远为空的表，调用方无需做 None 判断）。
        """
        return cls(
            ttl_seconds=_as_float(
                getattr(settings_obj, "CHAT_EXIT_BINDING_TTL_SECONDS", DEFAULT_TTL_SECONDS),
                DEFAULT_TTL_SECONDS,
            ),
            max_entries=_as_int(
                getattr(settings_obj, "CHAT_EXIT_BINDING_MAX_ENTRIES", DEFAULT_MAX_ENTRIES),
                DEFAULT_MAX_ENTRIES,
            ),
        )

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def bind(self, chat_id: str, exit_id: str, proxy_url: str) -> ChatExitBinding | None:
        """登记一条绑定（覆盖同名 chat_id），并把表压回容量上限。

        Args:
            chat_id: 上游会话 id。
            exit_id: 出口后缀（`Default.g`），只用于日志 / 观测。
            proxy_url: 该出口的完整代理串，不复用时会原样交给 curl_cffi。

        Returns:
            写入的绑定；`chat_id` / `proxy_url` 为空时返回 None（不写脏数据）。
        """
        chat_key = str(chat_id or "").strip()
        proxy = str(proxy_url or "").strip()
        if not chat_key or not proxy:
            return None
        now = self._clock()
        binding = ChatExitBinding(
            chat_id=chat_key,
            exit_id=str(exit_id or ""),
            proxy_url=proxy,
            bound_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._items[chat_key] = binding
        # 刚写入的是「最新用过的」，挪到队尾（若已存在则视为刷新）。
        self._items.move_to_end(chat_key)
        self._evict_overflow()
        return binding

    def resolve(self, chat_id: str) -> ChatExitBinding | None:
        """按 chat_id 查出口；过期条目顺手删除，命中则提升为「最近使用」。

        Args:
            chat_id: 上游会话 id。

        Returns:
            有效绑定，或 None（未绑定 / 已过期 / 表为空）。
        """
        chat_key = str(chat_id or "").strip()
        if not chat_key:
            return None
        binding = self._items.get(chat_key)
        if binding is None:
            return None
        if binding.expires_at <= self._clock():
            self._items.pop(chat_key, None)
            log.info(
                "[ChatExit] binding expired chat_id=%s exit=%s ttl=%.0fs",
                chat_key,
                binding.exit_id,
                self.ttl_seconds,
            )
            return None
        self._items.move_to_end(chat_key)
        return binding

    def drop(self, chat_id: str) -> bool:
        """删除一条绑定（出口被判死时调用，避免下次继续用烂出口）。

        Returns:
            真的删掉了才返回 True。
        """
        chat_key = str(chat_id or "").strip()
        if not chat_key:
            return False
        return self._items.pop(chat_key, None) is not None

    def clear(self) -> None:
        """清空整张表（进程收尾 / 单测复位用）。"""
        self._items.clear()

    def purge_expired(self) -> int:
        """扫一遍表，删掉所有过期条目。

        Returns:
            被删掉的条目数。
        """
        now = self._clock()
        stale = [key for key, binding in self._items.items() if binding.expires_at <= now]
        for key in stale:
            self._items.pop(key, None)
        return len(stale)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        """当前表内条目数（含尚未被扫出的过期条目）。"""
        return len(self._items)

    def snapshot(self) -> list[dict[str, Any]]:
        """只读快照（不含代理串），供探针 / 排障使用。"""
        return [binding.to_public_dict() for binding in self._items.values()]

    def _evict_overflow(self) -> None:
        """超出容量上限时从最久未使用的一端淘汰。"""
        while len(self._items) > self.max_entries:
            chat_key, binding = self._items.popitem(last=False)
            log.info(
                "[ChatExit] binding evicted chat_id=%s exit=%s max_entries=%s",
                chat_key,
                binding.exit_id,
                self.max_entries,
            )
