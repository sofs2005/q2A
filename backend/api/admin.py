import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel
from backend.core.config import settings
from backend.core.database import AsyncJsonDB
from backend.core.account_pool import AccountPool, Account

logger = logging.getLogger("backend.api.admin")

router = APIRouter()

def verify_admin(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split("Bearer ")[1]

    from backend.core.config import API_KEYS, settings as backend_settings

    # 允许使用默认管理员 Key (ADMIN_KEY) 或者任何已生成的 API_KEYS 作为管理凭证
    if token != backend_settings.ADMIN_KEY and token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Forbidden: Admin Key Mismatch")
    return token

class UserCreate(BaseModel):
    name: str
    quota: int = 1000000

class User(BaseModel):
    id: str
    name: str
    quota: int
    used_tokens: int


class ChatClearBatchRequest(BaseModel):
    emails: list[str]


def _normalize_selected_emails(emails: list[str]) -> list[str]:
    normalized_emails: list[str] = []
    seen: set[str] = set()

    for email in emails:
        if not email:
            continue
        normalized_email = email.strip()
        if not normalized_email or normalized_email in seen:
            continue
        normalized_emails.append(normalized_email)
        seen.add(normalized_email)

    return normalized_emails


def _summarize_clear_results(results: list[dict]) -> dict:
    summary = {"success": 0, "failed": 0, "skipped": 0}
    for item in results:
        status = item.get("status", "failed")
        if status in summary:
            summary[status] += 1
        else:
            summary["failed"] += 1
    return summary


def _get_account_by_email(pool: AccountPool, email: str):
    return next((account for account in pool.accounts if account.email == email), None)


async def _clear_single_account_chats(client, account):
    if not getattr(account, "cookies", "") and not getattr(account, "token", ""):
        return {"email": account.email, "status": "skipped", "reason": "missing_credentials"}

    return await client.clear_all_chats(account)


@router.get("/status", dependencies=[Depends(verify_admin)])
async def get_system_status(request: Request):
    pool = request.app.state.account_pool

    return {
        "accounts": pool.status(),
        "request_runtime": {
            "mode": "direct_http",
            "browser_required_for_requests": False,
            "description": "普通请求直连 HTTP，不经过浏览器",
        },
        "browser_automation": {
            "mode": "disabled",
            "available": False,
            "description": "轻量无浏览器镜像不包含注册/激活/刷新 Token 的浏览器自动化能力",
        }
    }

@router.get("/users", dependencies=[Depends(verify_admin)])
async def list_users(request: Request):
    db: AsyncJsonDB = request.app.state.users_db
    data = await db.get()
    return {"users": data}

@router.post("/users", dependencies=[Depends(verify_admin)])
async def create_user(user: UserCreate, request: Request):
    import uuid
    db: AsyncJsonDB = request.app.state.users_db
    data = await db.get()
    new_user = {
        "id": f"sk-{uuid.uuid4().hex}",
        "name": user.name,
        "quota": user.quota,
        "used_tokens": 0
    }
    data.append(new_user)
    await db.save(data)
    return new_user

@router.post("/accounts", dependencies=[Depends(verify_admin)])
async def add_account(request: Request):
    import time
    from backend.core.account_pool import Account, AccountPool
    from backend.services.qwen_client import QwenClient

    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON body")

    token = data.get("token", "")
    if not token:
        raise HTTPException(400, detail="token is required")

    acc = Account(
        email=data.get("email", f"manual_{int(time.time())}@qwen"),
        password=data.get("password", ""),
        token=token,
        cookies=data.get("cookies", ""),
        username=data.get("username", "")
    )

    is_valid = await client.verify_token(token)
    if not is_valid:
        return {"ok": False, "error": "Invalid token (验证失败，请确认Token有效)"}

    await pool.add(acc)
    return {"ok": True, "email": acc.email}


@router.get("/accounts", dependencies=[Depends(verify_admin)])
async def list_accounts(request: Request):
    pool: AccountPool = request.app.state.account_pool
    diagnostics_by_email = {item["email"]: item for item in pool.account_diagnostics()}
    accs = []
    for a in pool.accounts:
        d = a.to_dict()
        d.update(diagnostics_by_email.get(a.email, {}))
        accs.append(d)
    return {"accounts": accs}

@router.post("/accounts/register", dependencies=[Depends(verify_admin)])
async def register_new_account(request: Request):
    """无浏览器镜像不支持自动注册新千问账号。"""
    import logging

    log = logging.getLogger("backend.api.admin")
    client_ip = request.client.host if request.client else "127.0.0.1"
    log.info(f"[注册] 无浏览器模式拒绝自动注册请求，来源IP: {client_ip}")
    return {"ok": False, "error": "轻量无浏览器镜像不支持自动注册，请手动添加账号 token"}

@router.post("/verify", dependencies=[Depends(verify_admin)])
async def verify_all_accounts(request: Request):
    """验证所有账号的有效性 (完全复原单文件逻辑)"""
    from backend.core.account_pool import AccountPool
    from backend.services.qwen_client import QwenClient
    import logging

    log = logging.getLogger("qwen2api.admin")
    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    results = []
    for acc in pool.accounts:
        is_valid = await client.verify_token(acc.token)
        if not is_valid and acc.password:
            log.info(f"[校验] {acc.email} token失效，尝试自动刷新...")
            is_valid = await client.auth_resolver.refresh_token(acc)

        acc.valid = is_valid
        results.append({"email": acc.email, "valid": is_valid, "refreshed": not is_valid})

    await pool.save() # 直接保存全部状态，不调用 mark_invalid 以免熔断影响测试
    return {"ok": True, "results": results}

@router.post("/accounts/{email}/activate", dependencies=[Depends(verify_admin)])
async def activate_account(email: str, request: Request):
    """无浏览器镜像不支持页面式账号激活。"""
    import logging

    log = logging.getLogger("backend.api.admin")
    client_ip = request.client.host if request.client else "127.0.0.1"
    log.info(f"[激活] 无浏览器模式拒绝页面激活请求: {email}, 来源IP: {client_ip}")
    return {"ok": False, "error": "轻量无浏览器镜像不支持页面激活，请手动获取 token 后重新添加账号"}

@router.post("/accounts/{email}/verify", dependencies=[Depends(verify_admin)])
async def verify_account(email: str, request: Request):
    """单独验证某个账号的有效性 (完全复原单文件逻辑)"""
    from backend.services.qwen_client import QwenClient
    from backend.core.account_pool import AccountPool
    import logging

    log = logging.getLogger("qwen2api.admin")
    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    acc = next((a for a in pool.accounts if a.email == email), None)
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")

    is_valid = await client.verify_token(acc.token)
    if not is_valid and acc.password:
        log.info(f"[校验] {acc.email} token失效，尝试自动刷新...")
        is_valid = await client.auth_resolver.refresh_token(acc)

    acc.valid = is_valid
    await pool.save() # 直接保存，不调用 mark_invalid 以免熔断影响正常测试

    return {"email": acc.email, "valid": is_valid}

@router.delete("/accounts/chats", dependencies=[Depends(verify_admin)])
async def clear_all_upstream_chats(payload: ChatClearBatchRequest, request: Request):
    from backend.services.qwen_client import QwenClient

    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    requested_emails = _normalize_selected_emails(getattr(payload, "emails", []))
    if not requested_emails:
        raise HTTPException(status_code=400, detail="emails are required")

    accounts_by_email = {acc.email: acc for acc in pool.accounts}
    results: list[dict] = []

    for email in requested_emails:
        acc = accounts_by_email.get(email)
        if not acc:
            results.append({"email": email, "status": "skipped", "reason": "missing_account"})
            continue

        try:
            results.append(await _clear_single_account_chats(client, acc))
        except Exception as exc:
            logger.exception("Failed to clear upstream chats in batch", extra={"email": email})
            results.append({"email": email, "status": "failed", "error": str(exc)})

    return {"ok": True, "summary": _summarize_clear_results(results), "results": results}


@router.delete("/accounts/{email}/chats", dependencies=[Depends(verify_admin)])
async def clear_upstream_chats_for_account(email: str, request: Request):
    from backend.services.qwen_client import QwenClient

    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    acc = _get_account_by_email(pool, email)
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")

    return await _clear_single_account_chats(client, acc)


@router.delete("/accounts/{email}", dependencies=[Depends(verify_admin)])
async def delete_account(email: str, request: Request):
    from backend.core.account_pool import AccountPool
    pool: AccountPool = request.app.state.account_pool
    await pool.remove(email)
    return {"ok": True}

@router.get("/settings", dependencies=[Depends(verify_admin)])
async def get_settings():
    from backend.core.config import MODEL_MAP
    # 从 settings.py 所在的同级导入 VERSION，避免循环导入或未定义报错
    from backend.core.config import settings as backend_settings

    # 强制将 dict 转换，确保能被 JSON 序列化
    safe_map = {k: v for k, v in MODEL_MAP.items()}
    return {
        "version": "2.0.0",
        "max_inflight_per_account": backend_settings.MAX_INFLIGHT_PER_ACCOUNT,
        "model_aliases": safe_map
    }

@router.put("/settings", dependencies=[Depends(verify_admin)])
async def update_settings(data: dict):
    from backend.core.config import MODEL_MAP
    if "max_inflight_per_account" in data:
        settings.MAX_INFLIGHT_PER_ACCOUNT = data["max_inflight_per_account"]
    if "model_aliases" in data:
        MODEL_MAP.clear()
        MODEL_MAP.update(data["model_aliases"])
    return {"ok": True}

@router.get("/keys", dependencies=[Depends(verify_admin)])
async def get_keys():
    from backend.core.config import API_KEYS
    return {"keys": list(API_KEYS)}

@router.post("/keys", dependencies=[Depends(verify_admin)])
async def create_key():
    from backend.core.config import API_KEYS, save_api_keys

    new_key = f"sk-{secrets.token_hex(24)}"
    API_KEYS.add(new_key)
    save_api_keys(API_KEYS)
    return {"ok": True, "key": new_key}

@router.delete("/keys/{key}", dependencies=[Depends(verify_admin)])
async def delete_key(key: str):
    from backend.core.config import API_KEYS, save_api_keys

    if key in API_KEYS:
        API_KEYS.remove(key)
        save_api_keys(API_KEYS)
    return {"ok": True}
