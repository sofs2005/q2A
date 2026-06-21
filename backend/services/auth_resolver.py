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
        If refresh fails or account is pending activation, tries to activate via email.

        WAF/rate_limit 冷却期间跳过自愈：token 刷新无法解除 IP/行为级封锁，
        且刷新过程中的种子请求可能进一步触发 WAF。
        """
        if getattr(acc, "healing", False):
            log.info(f"[BGRefresh] {acc.email} healing already in progress")
            return

        # WAF/rate_limit 冷却中不触发自愈，避免种子请求加重封锁
        import time as _time
        if getattr(acc, "rate_limited_until", 0) > _time.time():
            remaining = int(acc.rate_limited_until - _time.time())
            log.info(f"[自愈] {acc.email} 仍在冷却中(剩余{remaining}s)，跳过 token 刷新")
            return

        acc.healing = True
        try:
            ok = await self.refresh_token(acc)
            if ok:
                # refresh_token 内部已处理状态标记（保留 rate_limited），
                # 此处仅在非冷却状态下额外确认 valid
                if not getattr(acc, 'activation_pending', False):
                    if acc.status_code != "rate_limited":
                        acc.valid = True
                        await self.pool.save()
                        log.info(f"[自愈] {acc.email} Token 刷新成功，已标记有效")
                    else:
                        await self.pool.save()
                        log.info(f"[自愈] {acc.email} Token 刷新成功，冷却中保留 rate_limited 状态")
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
        acc.activation_pending = False
        # Token 刷新成功只恢复凭证有效性，不覆盖 WAF/rate_limit 冷却状态
        # WAF punish 是 IP/行为级封锁，与 Token 有效性无关
        if acc.status_code not in ("rate_limited",):
            acc.valid = True
            acc.status_code = "valid"
            acc.last_error = ""
        else:
            # 冷却中：仅更新 token，保留 rate_limited 状态和冷却时间
            log.info(f"[Refresh] {acc.email} token 已刷新，但账号仍在冷却中，保留 rate_limited 状态")

        # 自动捕获 acw_tc WAF cookie（登录响应 Set-Cookie）
        import time as _time
        acw_tc = resp.cookies.get("acw_tc", "")

        # 登录接口通常不返回 acw_tc，需额外发一次轻量请求触发 WAF 下发
        if not acw_tc:
            try:
                seed_headers = fingerprint.build_headers(
                    token=new_token,
                    content_type="application/json",
                )
                seed_resp = await session.post(
                    f"{BASE_URL}/api/v2/chats/new",
                    json={"model": "qwen-max", "chat_type": "t2t"},
                    headers=seed_headers,
                )
                acw_tc = seed_resp.cookies.get("acw_tc", "")
                if acw_tc:
                    log.info(f"[Refresh] {acc.email} acw_tc 通过 chats/new 种子请求获取")
            except Exception as seed_err:
                log.warning(f"[Refresh] {acc.email} 种子请求异常: {seed_err}")

        if acw_tc:
            acc.waf_cookies = f"acw_tc={acw_tc}"
            acc.waf_cookies_expires_at = _time.time() + 1500
            log.info(f"[Refresh] {acc.email} acw_tc 已同步刷新")
        else:
            # 登录和种子请求都没返回 acw_tc，标记过期让下次 create_chat 收割
            acc.waf_cookies_expires_at = 0
            log.warning(f"[Refresh] {acc.email} 未能获取 acw_tc，将在下次 create_chat 时收割")

        await self.pool.save()
        log.info(f"[Refresh] {acc.email} token 已更新")
        return True
