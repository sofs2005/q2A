"""预热 / 修复账号的浏览器会话与短效 token。

为什么需要它：短效 access_token 只有 900s，续期靠 `GET auth.qwen.ai/api/v2/auths/refresh`，
而该接口只认登录 cookie。cookie 落盘在 `data/browser_sessions/<sha1(email)>.json`，
只有「登录成功过」的账号才会有。本脚本把这件事一次做完：

- 有会话文件的账号 → 只发一次 refresh（无滑块、秒级）
- 没有会话文件的账号 → 走一次浏览器登录，成功后把 cookie 落盘

用法（容器内）：
    python -m backend.tools.warm_sessions
    python -m backend.tools.warm_sessions --only qw3
    python -m backend.tools.warm_sessions --all      # 含 disabled 账号
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from backend.core.account_pool import AccountPool
from backend.core.config import settings
from backend.core.database import AsyncJsonDB
from backend.services.token_provider import (
    SESSION_DIR,
    ensure_fresh_token,
    token_expires_at,
    token_kind,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("qwen2api.warm_sessions")


async def warm(only: str = "", include_all: bool = False) -> int:
    db = AsyncJsonDB(settings.ACCOUNTS_FILE, default_data=[])
    pool = AccountPool(db, max_inflight=1)
    await pool.load()

    targets = []
    for acc in pool.accounts:
        if only and only not in str(acc.email):
            continue
        if not include_all and not (acc.email and acc.password):
            continue
        if not include_all and str(acc.status_code or "") == "disabled":
            continue
        targets.append(acc)

    log.info("待处理账号 %d 个（会话目录 %s）", len(targets), SESSION_DIR)
    ok = 0
    for idx, acc in enumerate(targets, 1):
        started = time.time()
        log.info("[%d/%d] %s ...", idx, len(targets), acc.email)
        try:
            token = await ensure_fresh_token(acc, force=True)
        except Exception as exc:  # 单账号失败不影响整体
            log.warning("[%d/%d] %s 异常: %s", idx, len(targets), acc.email, str(exc)[:160])
            token = ""

        if not token:
            log.warning("[%d/%d] %s 失败（%.1fs）", idx, len(targets), acc.email, time.time() - started)
            continue

        acc.token = token
        acc.activation_pending = False
        acc.valid = True
        if str(acc.status_code or "") != "rate_limited":
            acc.status_code = "valid"
        acc.last_error = ""
        exp = token_expires_at(token)
        log.info(
            "[%d/%d] %s ok kind=%s ttl=%ds（%.1fs）",
            idx, len(targets), acc.email, token_kind(token),
            max(0, int(exp - time.time())), time.time() - started,
        )
        ok += 1

    if ok:
        await pool.save()
        log.info("已保存账号池（%d 个账号拿到可用 token）", ok)
    log.info("完成：成功 %d / 尝试 %d", ok, len(targets))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="预热账号浏览器会话与短效 token")
    parser.add_argument("--only", default="", help="只处理邮箱包含该子串的账号")
    parser.add_argument("--all", action="store_true", help="包含 disabled / 无密码账号")
    args = parser.parse_args()
    return asyncio.run(warm(only=args.only, include_all=args.all))


if __name__ == "__main__":
    raise SystemExit(main())
