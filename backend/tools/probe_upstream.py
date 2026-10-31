"""实测上游 Version / bx-v 是否真的可用。

为什么需要它：`backend/core/config.py` 里记着教训 —— 抓包只能证明"官网在发什么"，
证明不了"服务端认什么"（0.3.10 官网在发，服务端却 401；0.2.91 反而是可用的）。
所以「抓到新版本号」之后必须实测，实测通过才能写进代码。

本工具用真实账号跑三步，任意一步失败即视为该版本号不可用：
  1. refresh  —— 用已登录会话换短效 token（验证 auth 域的 Version/bx-v）
  2. models   —— 认证 GET /api/models（验证 chat 域的 Version/bx-v）
  3. chat     —— 建会话 + 取首块（真正走 completions，最权威）

Version/bx-v 通过环境变量覆盖，不需要改代码：
  QWEN_WEB_VERSION / QWEN_REFRESH_VERSION / QWEN_BX_VERSION

输出一行 JSON，例如：
  {"version": "0.3.28", "bx": "2.5.37", "refresh_ok": true, "models_ok": true,
   "chat_ok": true, "models": ["qwen3.8-max", ...], "error": null}

用法（容器内，脚本可走 stdin 传进去）：
  docker exec -i -e QWEN_WEB_VERSION=0.3.28 qwen2api python - --mode all < probe_upstream.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from backend.core.account_pool import AccountPool
from backend.core.config import settings
from backend.core.database import AsyncJsonDB
from backend.services.qwen_client import QwenClient
from backend.services.token_provider import ensure_fresh_token


def _pick_account(pool: AccountPool):
    """挑一个可用账号：优先 valid，其次任何有邮箱的。"""
    candidates = [a for a in pool.accounts if getattr(a, "email", "")]
    for acc in candidates:
        if str(getattr(acc, "status_code", "") or "") == "valid":
            return acc
    return candidates[0] if candidates else None


async def run(mode: str) -> dict:
    out: dict = {
        "version": str(getattr(settings, "QWEN_WEB_VERSION", "")),
        "bx": str(getattr(settings, "QWEN_BX_VERSION", "")),
        "refresh_ok": False,
        "models_ok": False,
        "chat_ok": False,
        "models": [],
        "error": None,
    }
    db = AsyncJsonDB(settings.ACCOUNTS_FILE, default_data=[])
    pool = AccountPool(db, max_inflight=1)
    await pool.load()
    account = _pick_account(pool)
    if account is None:
        out["error"] = "没有可用账号"
        return out

    want_refresh = mode in ("all", "refresh", "models", "chat")
    want_models = mode in ("all", "models")
    want_chat = mode in ("all", "chat")

    token = ""
    if want_refresh:
        try:
            token = await ensure_fresh_token(account, force=True)
        except Exception as exc:
            out["error"] = f"refresh 异常: {str(exc)[:160]}"
            return out
        out["refresh_ok"] = bool(token)
        if not token:
            out["error"] = "refresh 拿不到 token"
            return out

    client = QwenClient(pool)

    if want_models:
        try:
            models = await client.list_models(token, account)
        except Exception as exc:
            out["error"] = f"models 异常: {str(exc)[:160]}"
            models = []
        out["models_ok"] = bool(models)
        out["models"] = [
            str((m.get("id") or m.get("model") or m.get("name")) if isinstance(m, dict) else m)
            for m in models[:10]
        ]
        if not models:
            out["error"] = out["error"] or "models 返回空"

    if want_chat:
        model = str(getattr(settings, "QWEN_MAX_MODEL", "") or "qwen3.8-max")
        chat_id = ""
        try:
            chat_id = await client.executor.create_chat(account, model)
            got = False
            async for _event in client.executor.stream(account, chat_id, model, "ping", account=account):
                got = True
                break
            out["chat_ok"] = got
            if not got:
                out["error"] = out["error"] or "chat 没有返回任何块"
        except Exception as exc:
            out["error"] = out["error"] or f"chat 异常: {str(exc)[:200]}"
        finally:
            if chat_id:
                try:
                    await client.delete_chat(chat_id, token, account=account)
                except Exception:
                    pass

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="实测上游 Version/bx-v 可用性")
    parser.add_argument("--mode", default="all", choices=["all", "refresh", "models", "chat"])
    args = parser.parse_args()
    result = asyncio.run(run(args.mode))
    print(json.dumps(result, ensure_ascii=False))
    ok = result["refresh_ok"] and result["models_ok"] and (result["chat_ok"] or args.mode != "all")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
