"""启动时并发探测出口健康：把「已经连不上」的出口提前踢出轮转。

背景（2026-12 生产实测）
------------------------
出口池上线后服务从「全挂」恢复成「能用但慢」：一次普通对话 108s，其中约 21s
浪费在挨个试连不通的出口上（`Default.g` / `d` / `c` / `e` 各等满一个超时）。
用户的代理系统不稳，任何时刻都有若干出口是死的 —— 只能靠**探测**把它们认出来，
而不是让真实请求去撞。

设计约束
--------
1. **不阻塞 startup**：`main.py` 只 `create_task`，探测在事件循环让出之后跑；
2. **轻量端点**：GET `/api/v2/users/status` 这类不会触发 WAF、不烧额度的接口，
   **绝不**用 `chat/completions`（烧额度 + 招风控）；
3. **并发**：`asyncio.gather`，总耗时受 `EXIT_PROBE_TOTAL_TIMEOUT_SECONDS`（默认
   15s）约束，超预算的出口按不可用处理；
4. **失败即冷却**：探测失败 → `mark_failure` → 达到阈值立刻进冷却，`pick()` 跳过。

日志口径（排障时直接 grep）：

    [ExitProbe] ok exit=Default.g 1.2s
    [ExitProbe] dead exit=Default.c reason=connect_failed → cooldown 600s
    [ExitProbe] done pool=chat exits=12 ready=8 dead=4 elapsed=3.4s
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.services.exit_pool import DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS

log = logging.getLogger("qwen2api.exit_probe")


def _probe_budget(settings_obj: Any) -> float:
    """启动探测的总耗时上限（秒）。"""
    raw = getattr(settings_obj, "EXIT_PROBE_TOTAL_TIMEOUT_SECONDS", DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS


def probe_enabled(settings_obj: Any) -> bool:
    """`EXIT_PROBE_ON_STARTUP` 是否开启（默认开）。"""
    raw = getattr(settings_obj, "EXIT_PROBE_ON_STARTUP", True)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def select_pool(app: Any) -> tuple[Any, str]:
    """挑出启动时要探测的出口池：建会话池优先（`chats/new` 是这次要救的链路）。

    Args:
        app: FastAPI 应用对象（读 `app.state`）。

    Returns:
        `(pool, label)`；没有可用池时 `(None, "")`。
    """
    client = getattr(app.state, "qwen_client", None)
    chat_pool = getattr(client, "chat_exit_pool", None)
    if chat_pool is not None and bool(getattr(chat_pool, "enabled", False)):
        return chat_pool, "chat"
    upload_pool = getattr(app.state, "upload_exit_pool", None)
    if upload_pool is not None and bool(getattr(upload_pool, "enabled", False)):
        return upload_pool, "upload"
    return None, ""


def _probe_credentials(account_pool: Any) -> tuple[str, Any]:
    """取「第一个可用账号」的 token 与账号对象（探测要带鉴权 + 同一套指纹）。

    探测端点即使不带 token 也能通，但带上一致的身份更接近真实链路，也不会
    因为「匿名请求」被上游区别对待。
    """
    accounts = list(getattr(account_pool, "accounts", []) or [])
    for account in accounts:
        if not getattr(account, "valid", False):
            continue
        token = str(getattr(account, "token", "") or "").strip()
        if token:
            return token, account
    return "", None


async def run_startup_exit_probe(app: Any, settings_obj: Any = None) -> dict[str, Any]:
    """后台任务入口：并发探测出口池，失败出口进冷却。**永不抛异常**。

    Args:
        app: FastAPI 应用对象。
        settings_obj: 配置对象（默认取 `backend.core.config.settings`）。

    Returns:
        汇总信息（单测用）：`{pool, total, ok, dead, elapsed_seconds, skipped}`。
    """
    if settings_obj is None:
        from backend.core.config import settings as settings_obj  # 延迟 import

    summary: dict[str, Any] = {
        "pool": "",
        "total": 0,
        "ok": 0,
        "dead": 0,
        "elapsed_seconds": 0.0,
        "skipped": "",
    }
    try:
        if not probe_enabled(settings_obj):
            summary["skipped"] = "disabled"
            log.info("[ExitProbe] skipped (EXIT_PROBE_ON_STARTUP=false)")
            return summary
        pool, label = select_pool(app)
        if pool is None:
            summary["skipped"] = "no_pool"
            log.info("[ExitProbe] skipped (出口池未启用)")
            return summary
        token, account = _probe_credentials(getattr(app.state, "account_pool", None))
        budget = _probe_budget(settings_obj)
        summary["pool"] = label
        started = time.monotonic()
        results = await pool.probe_all(token=token, account=account, limit_seconds=budget)
        elapsed = time.monotonic() - started
        ok = sum(1 for item in results if item.get("ok"))
        summary.update(
            {
                "total": len(results),
                "ok": ok,
                "dead": len(results) - ok,
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        log.info(
            "[ExitProbe] done pool=%s exits=%s ready=%s dead=%s elapsed=%.1fs",
            label,
            len(results),
            ok,
            len(results) - ok,
            elapsed,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - 探测失败绝不影响服务
        summary["skipped"] = f"error:{type(exc).__name__}"
        log.warning("[ExitProbe] startup probe failed error=%s", f"{type(exc).__name__}: {exc}")
    return summary


async def startup_exit_probe_task(app: Any) -> None:
    """`main.py` 用的后台任务包装：探测异常一律吞掉，绝不打断服务启动。"""
    try:
        await run_startup_exit_probe(app)
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - 双保险
        log.warning("[ExitProbe] background task crashed", exc_info=True)
