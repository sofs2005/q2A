import json
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


class AccountStatusBatchRequest(BaseModel):
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


_PERSONALIZATION_MEMORY_KEYS = ("enable_memory", "enable_history_memory")


def _payload_get_value(payload, key: str, default=None):
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _normalize_personalization_update_payload(payload) -> dict:
    memory_payload = _payload_get_value(payload, "memory")
    tools_payload = _payload_get_value(payload, "tools_enabled")

    if not isinstance(memory_payload, dict):
        raise HTTPException(status_code=400, detail="memory block is required")
    if not isinstance(tools_payload, dict):
        raise HTTPException(status_code=400, detail="tools_enabled block is required")

    normalized_memory = {
        "enable_memory": bool(memory_payload.get("enable_memory", False)),
        "enable_history_memory": bool(memory_payload.get("enable_history_memory", False)),
    }
    normalized_tools = {str(key): bool(value) for key, value in tools_payload.items() if str(key)}
    if len(normalized_tools) != 9:
        raise HTTPException(status_code=400, detail="tools_enabled must contain 9 flags")

    return {"memory": normalized_memory, "tools_enabled": normalized_tools}


def _extract_requested_emails(payload) -> list[str]:
    emails = _payload_get_value(payload, "emails", [])
    if not isinstance(emails, list):
        raise HTTPException(status_code=400, detail="emails must be a list")
    return _normalize_selected_emails(emails)


def _extract_personalization_view(body_text: str) -> dict:
    try:
        parsed = json.loads(body_text or "{}")
    except Exception:
        parsed = {}

    source = parsed.get("data") if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict) else parsed
    if not isinstance(source, dict):
        source = {}

    memory_source = source.get("memory") if isinstance(source.get("memory"), dict) else {}
    tools_source = source.get("tools_enabled") if isinstance(source.get("tools_enabled"), dict) else {}

    return {
        "memory": {
            "enable_memory": bool(memory_source.get("enable_memory", False)),
            "enable_history_memory": bool(memory_source.get("enable_history_memory", False)),
        },
        "tools_enabled": {str(key): bool(value) for key, value in tools_source.items() if str(key)},
    }


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

    log = logging.getLogger("backend.api.admin")
    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON body")

    token = data.get("token", "")
    email = data.get("email", "")
    password = data.get("password", "")

    # 支持仅填邮箱密码自动登录获取 token
    if not token:
        if not email or not password:
            raise HTTPException(400, detail="需要提供 token，或者同时提供 email + password")
        log.info(f"[AddAccount] 未提供 token，尝试用 {email} 自动登录获取...")
        ok, new_token, err_detail = await client.auth_resolver.login(email, password)
        if not ok or not new_token:
            return {"ok": False, "error": f"自动登录失败: {err_detail or '未知错误'}"}
        token = new_token
        log.info(f"[AddAccount] {email} 自动登录成功，已获取 token")

    acc = Account(
        email=email or f"manual_{int(time.time())}@qwen",
        password=password,
        token=token,
        cookies="",
        username=data.get("username", "")
    )

    is_valid = await client.verify_token(token, acc)
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
        is_valid = await client.verify_token(acc.token, acc)
        if not is_valid and acc.password:
            log.info(f"[校验] {acc.email} token失效，尝试自动刷新...")
            is_valid = await client.auth_resolver.refresh_token(acc)

        acc.valid = is_valid
        results.append({"email": acc.email, "valid": is_valid, "refreshed": not is_valid})

    await pool.save() # 直接保存全部状态，不调用 mark_invalid 以免熔断影响测试
    return {"ok": True, "results": results}

@router.post("/accounts/disable", dependencies=[Depends(verify_admin)])
async def disable_accounts(payload: AccountStatusBatchRequest, request: Request):
    pool: AccountPool = request.app.state.account_pool
    requested_emails = _normalize_selected_emails(getattr(payload, "emails", []))
    if not requested_emails:
        raise HTTPException(status_code=400, detail="emails are required")

    results = await pool.disable_accounts(requested_emails)
    return {"ok": True, "summary": _summarize_clear_results(results), "results": results}


@router.post("/accounts/enable", dependencies=[Depends(verify_admin)])
async def enable_accounts(payload: AccountStatusBatchRequest, request: Request):
    pool: AccountPool = request.app.state.account_pool
    requested_emails = _normalize_selected_emails(getattr(payload, "emails", []))
    if not requested_emails:
        raise HTTPException(status_code=400, detail="emails are required")

    results = await pool.enable_accounts(requested_emails)
    return {"ok": True, "summary": _summarize_clear_results(results), "results": results}


@router.post("/accounts/{email}/disable", dependencies=[Depends(verify_admin)])
async def disable_account(email: str, request: Request):
    pool: AccountPool = request.app.state.account_pool
    account = _get_account_by_email(pool, email)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    await pool.disable_accounts([email])
    return {"ok": True, "email": email, "status": "disabled"}


@router.post("/accounts/{email}/enable", dependencies=[Depends(verify_admin)])
async def enable_account(email: str, request: Request):
    pool: AccountPool = request.app.state.account_pool
    account = _get_account_by_email(pool, email)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    await pool.enable_accounts([email])
    return {"ok": True, "email": email, "status": "valid"}


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

    is_valid = await client.verify_token(acc.token, acc)
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


@router.get("/accounts/{email}/personalization", dependencies=[Depends(verify_admin)])
async def get_account_personalization(email: str, request: Request):
    from backend.services.qwen_client import QwenClient

    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    acc = _get_account_by_email(pool, email)
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")
    if not getattr(acc, "cookies", "") and not getattr(acc, "token", ""):
        raise HTTPException(status_code=400, detail="missing_credentials")

    upstream = await client.get_personalization_settings(acc)
    if upstream.get("status") == "skipped":
        raise HTTPException(status_code=400, detail=upstream.get("reason", "missing_credentials"))
    if upstream.get("status") != "success":
        return {
            "ok": False,
            "email": email,
            "status": upstream.get("status", "failed"),
            "error": upstream.get("error", "upstream request failed"),
        }

    view = _extract_personalization_view(upstream.get("body", ""))
    return {
        "ok": True,
        "email": email,
        "status": "success",
        "transport": upstream.get("transport"),
        "http_status": upstream.get("http_status"),
        **view,
    }


@router.put("/accounts/{email}/personalization", dependencies=[Depends(verify_admin)])
async def update_account_personalization(email: str, payload: dict, request: Request):
    from backend.services.qwen_client import QwenClient

    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    acc = _get_account_by_email(pool, email)
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")
    if not getattr(acc, "cookies", "") and not getattr(acc, "token", ""):
        raise HTTPException(status_code=400, detail="missing_credentials")

    normalized_payload = _normalize_personalization_update_payload(payload)
    upstream = await client.update_personalization_settings(acc, normalized_payload)
    if upstream.get("status") == "skipped":
        raise HTTPException(status_code=400, detail=upstream.get("reason", "missing_credentials"))
    if upstream.get("status") != "success":
        return {
            "ok": False,
            "email": email,
            "status": upstream.get("status", "failed"),
            "error": upstream.get("error", "upstream request failed"),
        }

    return {
        "ok": True,
        "email": email,
        "status": "success",
        "transport": upstream.get("transport"),
        "http_status": upstream.get("http_status"),
    }


@router.put("/accounts/personalization", dependencies=[Depends(verify_admin)])
async def update_accounts_personalization(payload: dict, request: Request):
    from backend.services.qwen_client import QwenClient

    pool: AccountPool = request.app.state.account_pool
    client: QwenClient = request.app.state.qwen_client

    requested_emails = _extract_requested_emails(payload)
    if not requested_emails:
        raise HTTPException(status_code=400, detail="emails are required")

    normalized_payload = _normalize_personalization_update_payload(payload)
    accounts_by_email = {acc.email: acc for acc in pool.accounts}
    results: list[dict] = []

    for email in requested_emails:
        acc = accounts_by_email.get(email)
        if not acc:
            results.append({"email": email, "status": "skipped", "reason": "missing_account"})
            continue
        if not getattr(acc, "cookies", "") and not getattr(acc, "token", ""):
            results.append({"email": email, "status": "skipped", "reason": "missing_credentials"})
            continue

        try:
            upstream = await client.update_personalization_settings(acc, normalized_payload)
            if upstream.get("status") == "success":
                results.append(
                    {
                        "email": email,
                        "status": "success",
                        "transport": upstream.get("transport"),
                        "http_status": upstream.get("http_status"),
                    }
                )
            elif upstream.get("status") == "skipped":
                results.append({"email": email, "status": "skipped", "reason": upstream.get("reason", "missing_credentials")})
            else:
                results.append(
                    {
                        "email": email,
                        "status": "failed",
                        "transport": upstream.get("transport"),
                        "http_status": upstream.get("http_status"),
                        "error": upstream.get("error", "upstream request failed"),
                    }
                )
        except Exception as exc:
            logger.exception("Failed to update upstream personalization in batch", extra={"email": email})
            results.append({"email": email, "status": "failed", "error": str(exc)})

    return {"ok": True, "summary": _summarize_clear_results(results), "results": results}


@router.delete("/accounts/{email}", dependencies=[Depends(verify_admin)])
async def delete_account(email: str, request: Request):
    from backend.core.account_pool import AccountPool
    pool: AccountPool = request.app.state.account_pool
    await pool.remove(email)
    return {"ok": True}

@router.get("/settings", dependencies=[Depends(verify_admin)])
async def get_settings():
    from backend.core.config import MODEL_MAP
    # 版本标签统一从 config 的单一来源导入，避免多处硬编码漂移
    from backend.core.config import settings as backend_settings, VERSION_LABEL

    # 强制将 dict 转换，确保能被 JSON 序列化
    safe_map = {k: v for k, v in MODEL_MAP.items()}
    return {
        "version": VERSION_LABEL,
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
