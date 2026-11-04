from fastapi import APIRouter, Request, Depends, HTTPException
import logging

from backend.api.admin import verify_admin
from backend.core.database import AsyncJsonDB

logger = logging.getLogger("backend.api.probes")

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    required_state = (
        "accounts_db",
        "users_db",
        "captures_db",
        "account_pool",
        "qwen_client",
        "file_store",
        "session_affinity",
        "upstream_file_cache",
    )
    missing = [name for name in required_state if not hasattr(request.app.state, name)]
    if missing:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "missing": missing})
    return {"status": "ready"}


@router.get("/admin/dev/captures", dependencies=[Depends(verify_admin)])
async def get_captures(request: Request):
    db: AsyncJsonDB = request.app.state.captures_db
    return {"captures": await db.get()}


@router.delete("/admin/dev/captures", dependencies=[Depends(verify_admin)])
async def clear_captures(request: Request):
    db: AsyncJsonDB = request.app.state.captures_db
    await db.save([])
    return {"status": "cleared"}


@router.get("/admin/upload/probe", dependencies=[Depends(verify_admin)])
async def probe_browser_upload(request: Request):
    """只读探针：连一次浏览器，回报「浏览器上传」链路是否可用。

    排查「上传一直走内联 / 一直 400 attachment_too_large」时先看这里：

    - `mode`：`remote`（远程 CDP）/ `local`（进程内 Chromium）/ `off`（链路关闭）；
    - `connected=false`：浏览器连不上（endpoint / 网络 / 容器名写错 / 本机缺 Chromium）；
    - `captcha_passed_recently=false`：最近 10 分钟没成功过拼图验证码；
    - `calibration`：当前使用的拖动标定（`a` / `b` / 来源 / 标定时的 `mode`）；
    - `local` 模式额外给 `platform` / `webgl`，用于判断容器里的指纹环境；
    - `exit_pool`：出口池（`getstsToken` 主路径）的状态
      `{enabled, exits: [{id, state, failures, successes, cooldown_remaining, last_error}]}`。
      `state=cooldown` 表示该出口连续失败被临时剔除；`cooldown_remaining` 是剩余秒数。
      出口 id 只有后缀（如 `Default.g`），**不含**代理凭据。

    探针**永不**返回 5xx：异常也折叠成 `connected=false` + `error`。
    """
    exit_pool = _exit_pool_probe(request)
    proxy = getattr(request.app.state, "browser_upload_proxy", None)
    if proxy is None:
        return {
            "connected": False,
            "mode": "off",
            "browser_version": "",
            "captcha_passed_recently": False,
            "error": "browser_upload_proxy is not initialized",
            "exit_pool": exit_pool,
        }
    try:
        result = await proxy.probe()
    except Exception as exc:
        logger.warning("browser upload probe failed: %s", exc)
        return {
            "connected": False,
            "mode": "off",
            "browser_version": "",
            "captcha_passed_recently": False,
            "error": str(exc),
            "exit_pool": exit_pool,
        }
    if isinstance(result, dict):
        # 不要把浏览器探针的字段覆盖掉：出口池只是多挂一个 key。
        result.setdefault("exit_pool", exit_pool)
        return result
    return {"browser": result, "exit_pool": exit_pool}


def _exit_pool_probe(request: Request) -> dict:
    """出口池快照（不含凭据）；未启用时 `{enabled: false, exits: []}`。

    探针失败不影响浏览器部分：任何异常都折叠成 `enabled=false` + `error`。
    """
    pool = getattr(request.app.state, "upload_exit_pool", None)
    if pool is None:
        return {"enabled": False, "exits": []}
    try:
        snapshot = pool.snapshot()
    except Exception as exc:  # pragma: no cover - snapshot 只读内存，异常极罕见
        logger.warning("upload exit pool snapshot failed: %s", exc)
        return {"enabled": False, "exits": [], "error": str(exc)}
    return {"enabled": bool(getattr(pool, "enabled", bool(snapshot))), "exits": snapshot}
