"""
WAF Cookie Manager — 通过轻量 HTTP GET 获取 acw_tc cookie，用于阿里云 WAF 会话跟踪。

说明：
- 首页 GET 只返回 acw_tc 单个 cookie（已由下游源码证实）。
- cna/cnaui/tfstk/isg 等是浏览器端 JS 生成的，GET 首页拿不到。
- acw_tc 被 WAF 拉黑后封禁 1800 秒，触发后由 stream_chat_once 调用
  account_pool.mark_rate_limited(cooldown=1800) 进入冷却，不再立即重试。
"""

import logging
import time

from curl_cffi.requests import AsyncSession

log = logging.getLogger("qwen2api.waf_cookies")

WAF_COOKIE_TTL = 1500  # 25 分钟（acw_tc 的 Max-Age=1800，提前刷新）


class WafCookieManager:
    """管理每账号的 WAF cookie（acw_tc），通过轻量 GET 请求获取。"""

    _instance = None

    def __init__(self):
        pass

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_cookies(self, account) -> str:
        """获取账号的 WAF cookie，过期时自动刷新。"""
        now = time.time()
        if not account.waf_cookies or now > account.waf_cookies_expires_at:
            await self.refresh_account_cookies(account)
        return account.waf_cookies

    async def refresh_account_cookies(self, account):
        """通过 curl_cffi GET 请求获取 acw_tc cookie。"""
        email = account.email
        try:
            log.info(f"[WafCookie] Refreshing acw_tc for {email}...")
            async with AsyncSession(impersonate="chrome", timeout=15) as s:
                await s.get("https://chat.qwen.ai", allow_redirects=True)
                acw_tc = s.cookies.get("acw_tc", "")
                if acw_tc:
                    account.waf_cookies = f"acw_tc={acw_tc}"
                    account.waf_cookies_expires_at = time.time() + WAF_COOKIE_TTL
                    log.info(f"[WafCookie] Refreshed acw_tc for {email}: {acw_tc[:20]}...")
                else:
                    log.warning(f"[WafCookie] No acw_tc returned for {email}")
        except Exception as e:
            log.error(f"[WafCookie] Failed to refresh for {email}: {e}")
            raise

    def mark_expired(self, account):
        """标记 cookie 过期（检测到 x5sec 时调用）。"""
        log.warning(f"[WafCookie] Marking acw_tc expired for {account.email}")
        account.waf_cookies_expires_at = 0

    def update_cookies(self, account, new_cookies: dict[str, str]):
        """用滑块突破后的 cookie 更新账号 cookie 存储。

        将新 cookie（x5sec / acw_sc__v3 / acw_tc 等）合并到现有 waf_cookies，
        放行 cookie 有效期内复用（对齐阿里云 WAF 1800s 放行窗口）。

        Args:
            account: 账号对象
            new_cookies: {"acw_tc": "...", "x5sec": "...", ...}
        """
        if not new_cookies:
            return

        # 解析现有 cookie 为 dict
        existing: dict[str, str] = {}
        if account.waf_cookies:
            for pair in account.waf_cookies.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    existing[k.strip()] = v.strip()

        # 合并新 cookie（覆盖同名）
        existing.update(new_cookies)

        # 重建 cookie 字符串
        account.waf_cookies = "; ".join(f"{k}={v}" for k, v in existing.items() if k and v)
        account.waf_cookies_expires_at = time.time() + WAF_COOKIE_TTL
        log.info(
            "[WafCookie] Updated cookies for %s: keys=%s, expires_in=%ss",
            account.email,
            list(existing.keys()),
            WAF_COOKIE_TTL,
        )
