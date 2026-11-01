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
    """只读探针：连一次 CDP，回报「浏览器上传」链路是否可用。

    排查「上传一直走内联 / 一直 400 attachment_too_large」时先看这里：

    - `connected=false`：CDP 连不上（endpoint / 网络 / 容器名写错）；
    - `captcha_passed_recently=false`：最近 10 分钟没成功过拼图验证码；
    - `calibration`：当前使用的拖动标定（`a` / `b` / 来源）。

    探针**永不**返回 5xx：异常也折叠成 `connected=false` + `error`。
    """
    proxy = getattr(request.app.state, "browser_upload_proxy", None)
    if proxy is None:
        return {
            "connected": False,
            "browser_version": "",
            "captcha_passed_recently": False,
            "error": "browser_upload_proxy is not initialized",
        }
    try:
        return await proxy.probe()
    except Exception as exc:
        logger.warning("browser upload probe failed: %s", exc)
        return {
            "connected": False,
            "browser_version": "",
            "captcha_passed_recently": False,
            "error": str(exc),
        }
