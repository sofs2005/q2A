import hashlib
import logging

from backend.core.account_pool import Account, AccountPool
from backend.core.browser_fingerprint import fingerprint_for_account, get_session

log = logging.getLogger(__name__)

BASE_URL = "https://chat.qwen.ai"


async def get_fresh_token(email: str, password: str) -> str:
    """No-browser builds do not support page-based token extraction."""
    del email, password
    raise NotImplementedError("No-browser build does not support page-based token extraction")


async def register_qwen_account():
    """Automatic Qwen registration requires browser automation and is disabled."""
    log.warning("[Register] 无浏览器模式不支持自动注册，请手动添加账号 token")
    return None


async def activate_account(acc: Account) -> bool:
    """Page-based account activation is disabled in no-browser builds."""
    log.warning(f"[Activate] 无浏览器模式不支持页面激活: {acc.email}")
    return False


class AuthResolver:
    """自动登录并提取 Token，在检测到 401 时自动自愈凭证"""

    def __init__(self, pool: AccountPool):
        self.pool = pool

    @staticmethod
    def _sha256_password(password: str) -> str:
        return hashlib.sha256((password or "").encode("utf-8")).hexdigest()

    async def auto_heal_account(self, acc: Account):
        """Background task to refresh token. If successful, marks account valid.
        If refresh fails or account is pending activation, tries to activate via email."""
        if getattr(acc, "healing", False):
            log.info(f"[BGRefresh] {acc.email} healing already in progress")
            return

        acc.healing = True
        try:
            ok = await self.refresh_token(acc)
            if ok:
                if not getattr(acc, 'activation_pending', False):
                    acc.valid = True
                    await self.pool.save()
                    log.info(f"[自愈] {acc.email} Token 刷新成功，已标记有效")
                    return
                log.info(f"[BGRefresh] {acc.email} token refreshed but account still needs activation")
            else:
                log.warning(f"[自愈] {acc.email} Token 刷新失败，无浏览器模式不支持自动激活")

            activated = await activate_account(acc)
            if activated:
                acc.activation_pending = False
                acc.valid = True
                await self.pool.save()
                log.info(f"[自愈] {acc.email} 激活成功，已保存")
            else:
                log.warning(f"[自愈] {acc.email} 激活失败")
        except Exception as e:
            log.warning(f"[BGRefresh] {acc.email} auto heal failed: {e}")
        finally:
            acc.healing = False

    async def refresh_token(self, acc: Account) -> bool:
        """Re-login with email+password via curl_cffi to get a fresh token."""
        if not acc.email or not acc.password:
            log.warning(f"[Refresh] 账号 {acc.email} 无密码，无法刷新")
            return False

        log.info(f"[Refresh] 正在为 {acc.email} 刷新 token...")
        payload = {
            "email": acc.email,
            "password": self._sha256_password(acc.password),
        }

        try:
            fingerprint = fingerprint_for_account(acc)
            session = await get_session(fingerprint)
            resp = await session.post(
                f"{BASE_URL}/api/v1/auths/signin",
                json=payload,
                headers=fingerprint.build_headers(content_type="application/json"),
            )
        except Exception as e:
            log.warning(f"[Refresh] {acc.email} curl_cffi 登录异常: {e}")
            return False

        if resp.status_code != 200:
            log.warning(f"[Refresh] {acc.email} HTTP {resp.status_code}，登录失败")
            return False

        try:
            data = resp.json()
        except Exception:
            log.warning(f"[Refresh] {acc.email} 登录响应不是 JSON")
            return False

        new_token = str(data.get("token", "") or "").strip()
        if not new_token:
            log.warning(f"[Refresh] {acc.email} 登录响应缺少 token 字段")
            return False

        acc.token = new_token
        acc.valid = True
        acc.activation_pending = False
        acc.status_code = "valid"
        acc.last_error = ""
        await self.pool.save()
        log.info(f"[Refresh] {acc.email} token 已更新")
        return True
