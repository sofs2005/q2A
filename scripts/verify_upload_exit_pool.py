"""端到端验证出口池取 STS：真实代理出口 + 2MB 视频，直传 OSS。

跑法（生产容器里，工作目录=仓库根）：

    UPLOAD_EXIT_POOL='socks5h://Default.g@10.6.6.114:2260,socks5h://Default.j@10.6.6.114:2260' \
    python scripts/verify_upload_exit_pool.py

输出里会有一行 `[UploadExit] sts ok exit=Default.g elapsed=5.9s file=verify-2mb.mp4 size=2097152`
—— `exit=` 就是这次 STS 走的那个出口后缀（全链路零验证码）。

账号来源：默认读 `settings.ACCOUNTS_FILE`（生产即 `data/accounts.json`）的第一个账号；
文件不存在时用环境变量 `Q2A_EMAIL` / `Q2A_TOKEN`（`Q2A_COOKIES` 可选）。

退出码：0 = STS 拿到且 OSS PUT 成功；1 = 失败（原因已打在日志里）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile

# 必须在 import backend 之前设：Settings 在 import 期就读 env。
os.environ.setdefault(
    "UPLOAD_EXIT_POOL",
    "socks5h://Default.g@10.6.6.114:2260,"
    "socks5h://Default.j@10.6.6.114:2260,"
    "socks5h://Default.l@10.6.6.114:2260,"
    "socks5h://Default.c@10.6.6.114:2260",
)
os.environ.setdefault("UPLOAD_VIA_EXIT", "auto")

sys.path.insert(0, os.getcwd())

from backend.core.account_pool import Account  # noqa: E402
from backend.core.config import Settings, settings  # noqa: E402
from backend.services.qwen_client import QwenClient  # noqa: E402
from backend.services.upstream_file_uploader import UpstreamFileUploader  # noqa: E402

VIDEO_BYTES = 2 * 1024 * 1024


def _load_account() -> Account:
    """取一个能用的账号（优先 accounts.json，其次 Q2A_EMAIL / Q2A_TOKEN）。"""
    payload: dict = {}
    try:
        with open(settings.ACCOUNTS_FILE, encoding="utf-8") as handle:
            raw = json.load(handle)
        payload = raw[0] if isinstance(raw, list) and raw else {}
    except FileNotFoundError:
        payload = {}
    if not payload.get("token"):
        payload = {
            "email": os.environ.get("Q2A_EMAIL", ""),
            "token": os.environ.get("Q2A_TOKEN", ""),
            "cookies": os.environ.get("Q2A_COOKIES", ""),
        }
    return Account(
        email=str(payload.get("email") or ""),
        token=str(payload.get("token") or ""),
        cookies=str(payload.get("cookies") or ""),
    )


async def main() -> int:
    """跑一次「2MB 视频 → 出口池取 STS → OSS PUT」，打印出口与结果。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    account = _load_account()
    uploader = UpstreamFileUploader(QwenClient(None), Settings(_env_file=None))
    pool = uploader.exit_pool
    print(
        f"exit_pool enabled={pool is not None} "
        f"exits={getattr(pool, 'exit_ids', [])} account={account.email or '<none>'}"
    )
    if pool is None:
        print("UPLOAD_EXIT_POOL 为空或 UPLOAD_VIA_EXIT=never：本次不会走代理。")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
        handle.write(os.urandom(VIDEO_BYTES))
        local_path = handle.name
    code = 0
    try:
        result = await uploader.upload_local_file(
            account,
            {
                "filename": "verify-2mb.mp4",
                "path": local_path,
                "content_type": "video/mp4",
            },
        )
        print(
            "RESULT file_id=",
            result["remote_file_id"],
            "parse_status=",
            result.get("parse_status"),
            "url_expires_at=",
            result.get("url_expires_at"),
        )
    except Exception as exc:  # noqa: BLE001 - 验证脚本要把失败原因原样打出来
        print("FAILED", type(exc).__name__, exc)
        code = 1
    finally:
        os.unlink(local_path)
    if pool is not None:
        print("POOL", json.dumps(pool.snapshot(), ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
