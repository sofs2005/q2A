"""
WAF Cookie Manager — 收集阿里系风控 cookie，用于阿里云 WAF 会话跟踪。

说明：
- 主链路是 acw_tc（首页 GET 即可拿到），其余风控 cookie 随响应下发时一并收割。
- tfstk / isg / ssxmod_itna* / atpsida / cna / cnaui / aui / sca 多为浏览器端 JS 生成，
  GET 首页未必拿得到；此处只做「拿得到就存」，不伪造、不合成 —— 伪造加密串比不发
  更容易触发风险识别。
- 同理，bx-ua / bx-umidtoken 等 bx-* 指纹头保持不发送：2026-09 抓包显示服务端并未
  强校验（实测值为 SDK 加载失败的占位串仍正常放行）。
- acw_tc 被 WAF 拉黑后封禁 WAF_PUNISH_COOLDOWN 秒，触发后由 stream_chat_once 调用
  account_pool.mark_rate_limited(cooldown=...) 进入冷却，不再立即重试。
- 必须走 browser_fingerprint.new_session：注入 UPSTREAM_PROXY + 账号 TLS 指纹，
  避免 cookie 与主请求出口 IP / 指纹不一致。
"""

import logging
import time

from backend.core.browser_fingerprint import fingerprint_for_account, new_session

log = logging.getLogger("qwen2api.waf_cookies")

WAF_COOKIE_TTL = 1500  # 25 分钟（acw_tc 的 Max-Age=1800，提前刷新）

# 需要按账号保存并回放的风控 cookie 白名单。
# acw_tc 之外的都是「能收割到就存」，不强求。
WAF_COOKIE_NAMES: tuple[str, ...] = (
    "acw_tc",
    "tfstk",
    "isg",
    "ssxmod_itna",
    "ssxmod_itna2",
    "atpsida",
    "cna",
    "cnaui",
    "aui",
    "sca",
)


def collect_waf_cookies(source) -> dict[str, str]:
    """从 cookie jar / dict 里按白名单挑出风控 cookie。

    source 可以是 curl_cffi 的 Cookies 对象，也可以是普通 dict。
    """
    getter = getattr(source, "get", None)
    if getter is None:
        return {}
    collected: dict[str, str] = {}
    for name in WAF_COOKIE_NAMES:
        try:
            value = getter(name, "")
        except Exception:
            continue
        value = str(value or "").strip()
        if value:
            collected[name] = value
    return collected


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
        """通过 curl_cffi GET 请求收割风控 cookie（与主链路同代理/指纹）。"""
        email = account.email
        try:
            log.info(f"[WafCookie] Refreshing acw_tc for {email}...")
            fingerprint = fingerprint_for_account(account)
            async with new_session(fingerprint, timeout=15) as s:
                await s.get("https://chat.qwen.ai", allow_redirects=True)
                collected = collect_waf_cookies(s.cookies)
                if collected:
                    # 合并而非覆盖：避免冲掉滑块放行后拿到的 x5sec / acw_sc__v3
                    self.update_cookies(account, collected)
                    log.info(f"[WafCookie] Refreshed for {email}: keys={sorted(collected)}")
                else:
                    log.warning(f"[WafCookie] No waf cookie returned for {email}")
        except Exception as e:
            log.error(f"[WafCookie] Failed to refresh for {email}: {e}")
            raise

    def invalidate(self, account):
        """真正作废账号的 WAF cookie：清空值并把过期时间归零。

        只置 expires_at=0 而不清值是历史遗留缺陷 —— 读取侧曾把 0 当作
        「未设置过期」从而继续注入已失效的 cookie。
        """
        log.warning(f"[WafCookie] Invalidating waf cookies for {account.email}")
        account.waf_cookies = ""
        account.waf_cookies_expires_at = 0

    def mark_expired(self, account):
        """标记 cookie 过期（检测到 x5sec 时调用）。"""
        self.invalidate(account)

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
