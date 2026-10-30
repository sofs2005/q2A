"""短效 access_token 提供者：用 headless chromium 登录官方站签发 token。

## 为什么需要它（2026-10-30 实测）

上游 `/api/v2/chat/completions` 现在只接受 `type=access_token` 的短效 token
（`exp-iat=900s`）。账号池里那批长期 token（209 字符、无 `type` 声明）在
`Version: 0.3.27` 下会被直接判 401。

而 token 只能由真浏览器签发：

- `POST {BASE_URL}/api/v2/auths/signin` 只发长期 token；
- `POST https://auth.qwen.ai/api/v2/auths/signin` 走 AWSC 风控，纯 HTTP 客户端
  拿到的是挑战页 HTML（`aliyun_waf_aa` / `aliyun_waf_bb`），浏览器靠 JS 才过。

所以本模块只做一件事：**开浏览器登录、把短效 token 取出来**。业务请求仍然走
httpx/curl_cffi，不经过浏览器。

## 已知的坑

- 容器里 chromium 必须带 `--disable-quic`，否则访问 chat.qwen.ai 直接
  `net::ERR_TIMED_OUT`（容器 DNS 给的两条 A 记录里会挑到 QUIC 不通的那条）。
- 登录页 `wait_until="domcontentloaded"` 会挂住（子资源依赖 auth 域），
  必须用 `wait_until="commit"` 再自己等选择器。
- 密码登录要先把「使用验证码登录」切到「使用密码登录」，密码框才出现。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
from typing import Any

from backend.core.config import settings

log = logging.getLogger("qwen2api.token_provider")

BASE_URL = "https://chat.qwen.ai"

# token 剩余寿命低于该秒数就提前续期（access_token 只有 900s）
REFRESH_MARGIN_SECONDS = 120

# chromium 启动参数：--disable-quic 是容器内的硬要求，见模块 docstring
_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-quic",
]


# 登录页滑块会检测自动化特征（webdriver / 缺 chrome 对象），必须先注入伪装脚本
_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => false});
try { delete navigator.webdriver; } catch (e) {}
window.chrome = window.chrome || {
  runtime: {connect: function(){}, sendMessage: function(){}},
  loadTimes: function(){ return {}; },
  csi: function(){ return {}; },
  app: {isInstalled: false, InstallState: {DISABLED: 'disabled'}, RunningState: {RUNNING: 'running'}},
};
Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
"""

try:  # pragma: no cover - 可选依赖，运行时降级
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover
    async_playwright = None


def parse_jwt_claims(token: str) -> dict[str, Any]:
    """解析 JWT 的 payload 段（不校验签名），失败返回空 dict。"""
    raw = str(token or "").strip()
    parts = raw.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def token_expires_at(token: str) -> float:
    """取 token 的过期时间戳；解析不出返回 0。"""
    exp = parse_jwt_claims(token).get("exp")
    try:
        return float(exp or 0)
    except (TypeError, ValueError):
        return 0.0


def token_kind(token: str) -> str:
    """返回 token 类型：`access`（短效）/ `legacy`（长期）/ `unknown`。"""
    claims = parse_jwt_claims(token)
    if not claims:
        return "unknown"
    return "access" if claims.get("type") == "access_token" else "legacy"


def is_token_usable(token: str, *, margin: float = REFRESH_MARGIN_SECONDS) -> bool:
    """token 是否还够用（有 exp 就按 exp 判，没 exp 视为不可用）。"""
    if not token:
        return False
    exp = token_expires_at(token)
    if exp <= 0:
        return False
    return exp - time.time() > margin


class BrowserTokenProvider:
    """按账号签发短效 access_token，带进程内缓存与并发保护。"""

    _instance: "BrowserTokenProvider | None" = None

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._contexts: dict[str, Any] = {}
        self._cache: dict[str, tuple[str, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._browser_lock = asyncio.Lock()
        # 登录是重操作且上游有风控，全局串行化，避免多账号并发登录互相挤
        self._login_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "BrowserTokenProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---- 浏览器生命周期 -------------------------------------------------

    async def _ensure_browser(self):
        if async_playwright is None:
            raise RuntimeError("playwright 未安装，无法签发短效 token")
        async with self._browser_lock:
            if self._browser is not None:
                return self._browser
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True, args=_LAUNCH_ARGS
            )
            log.info("[TokenProvider] chromium 已启动")
            return self._browser

    async def _context_for(self, account) -> Any:
        """按账号复用 BrowserContext（cookie 与登录态保留）。"""
        email = str(getattr(account, "email", "") or "")
        ctx = self._contexts.get(email)
        if ctx is not None:
            return ctx
        browser = await self._ensure_browser()
        ctx = await browser.new_context(locale="zh-CN", viewport={"width": 1280, "height": 800})
        try:
            await ctx.add_init_script(_STEALTH_SCRIPT)
        except Exception:
            pass
        self._contexts[email] = ctx
        return ctx

    async def close(self) -> None:
        """关闭所有 context 与浏览器，静默处理异常。"""
        for ctx in list(self._contexts.values()):
            try:
                await ctx.close()
            except Exception:
                pass
        self._contexts.clear()
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    # ---- 登录 -----------------------------------------------------------

    async def _drag_login_slider(self, page) -> bool:
        """拖动登录页的阿里云「访问验证」滑块（aliyunCaptcha 新版组件）。

        与 punish 页的 nc_1_n1z 不是同一套 DOM，故单独实现：
        滑块 #aliyunCaptcha-sliding-slider（40px），轨道宽 300px，需拖 260px。
        """
        slider = await page.query_selector("#aliyunCaptcha-sliding-slider")
        if slider is None:
            slider = await page.query_selector(".aliyunCaptcha-sliding-slider")
        if slider is None:
            return False
        box = await slider.bounding_box()
        if not box:
            return False

        # 轨道宽度优先从容器量，量不到用默认 300-40
        distance = 260.0
        for sel in ("#aliyunCaptcha-sliding-text-box", "#aliyunCaptcha-sliding"):
            node = await page.query_selector(sel)
            if node is not None:
                nbox = await node.bounding_box()
                if nbox and nbox["width"] > box["width"]:
                    distance = nbox["width"] - box["width"]
                    break

        start_x = box["x"] + box["width"] / 2
        start_y = box["y"] + box["height"] / 2
        try:
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()
            # 先小幅移动再加速，末段减速，模拟人手（匀速直线是滑块风控最容易识别的特征）
            steps = 22
            for i in range(1, steps + 1):
                ratio = i / steps
                eased = 1 - (1 - ratio) ** 2  # ease-out
                await page.mouse.move(
                    start_x + distance * eased,
                    start_y + random.uniform(-1.2, 1.2),
                )
                await page.wait_for_timeout(random.randint(8, 22))
            await page.mouse.move(start_x + distance, start_y)
            await page.wait_for_timeout(random.randint(60, 160))
            await page.mouse.up()
        except Exception as exc:
            log.warning("[TokenProvider] 滑块拖动异常: %s", str(exc)[:120])
            return False
        await page.wait_for_timeout(2500)
        return True

    async def _challenge_present(self, page) -> bool:
        """判断登录页是否弹出了「访问验证」滑块。

        优先看滑块元素本身（比文本可靠：挑战渲染期间 inner_text 可能取不到，
        甚至抛异常），其次才看文本。
        """
        try:
            if await page.query_selector("#aliyunCaptcha-sliding-slider") is not None:
                return True
        except Exception:
            pass
        try:
            text = await page.inner_text("body")
        except Exception:
            return False
        return ("访问验证" in text) or ("验证您是真人" in text)

    async def _solve_login_challenge(self, page) -> None:
        """拖动登录页滑块，并在通过后重新提交表单。"""
        log.info("[TokenProvider] 登录页触发滑块验证，尝试拖动")
        dragged = False
        deadline = time.time() + 12
        while time.time() < deadline:
            try:
                if await self._drag_login_slider(page):
                    dragged = True
                    break
            except Exception as exc:
                log.warning("[TokenProvider] 新版滑块拖动异常: %s", str(exc)[:120])
            await page.wait_for_timeout(700)
        if not dragged:
            try:
                from backend.services.captcha_solver import CaptchaSolver

                await CaptchaSolver.get_instance()._drag_slider(page, max_wait_ms=8000)
            except Exception as exc:
                log.warning("[TokenProvider] 通用滑块拖动失败: %s", str(exc)[:120])
        await page.wait_for_timeout(2500)

        # 挑战可能发生在提交之前，通过后需要再提交一次
        try:
            if "/auth" in page.url and await page.locator("input[name=password]").count() > 0:
                await page.locator("input[name=password]").press("Enter")
                await page.wait_for_timeout(2000)
        except Exception:
            pass

    async def _login_once(self, account) -> str:
        """在浏览器里用邮箱+密码登录，返回 localStorage 里的短效 token。"""
        email = str(getattr(account, "email", "") or "")
        password = str(getattr(account, "password", "") or "")
        if not email or not password:
            log.warning("[TokenProvider] 账号 %s 缺邮箱或密码，无法登录", email)
            return ""

        ctx = await self._context_for(account)
        page = await ctx.new_page()
        try:
            # domcontentloaded 会因 auth 子资源挂住，必须用 commit
            await page.goto(f"{BASE_URL}/auth", wait_until="commit", timeout=60000)
            await page.wait_for_selector("input[name=email]", timeout=40000)
            await page.wait_for_timeout(3000)

            # 切到密码登录（默认是验证码登录，没有密码框）
            for _ in range(10):
                if await page.locator("input[name=password]").count() > 0:
                    break
                for label in ("使用密码登录", "Log in with a password"):
                    try:
                        await page.get_by_text(label, exact=False).first.click(timeout=3000)
                        break
                    except Exception:
                        continue
                await page.wait_for_timeout(1200)

            await page.wait_for_selector("input[name=password]", timeout=20000)
            await page.fill("input[name=email]", email)
            await page.fill("input[name=password]", password)
            await page.locator("input[name=password]").press("Enter")

            # 提交后轮询：一旦出现滑块就过，直到跳出 /auth（登录成功的标志）
            deadline = time.time() + 45
            tick = 0
            while time.time() < deadline:
                tick += 1
                if "/auth" not in page.url:
                    break
                present = await self._challenge_present(page)
                if present:
                    log.info("[TokenProvider] %s 检测到登录滑块(t=%ds)", email, tick)
                    await self._solve_login_challenge(page)
                    continue
                if tick % 10 == 0:
                    log.info("[TokenProvider] %s 等待登录完成 t=%ds url=%s", email, tick, page.url)
                await page.wait_for_timeout(800)
            await page.wait_for_timeout(2500)

            token = await page.evaluate("() => localStorage.getItem('token') || ''")
            token = str(token or "").strip()
            if not token:
                snippet = ""
                try:
                    snippet = (await page.inner_text("body"))[:160].replace("\n", " | ")
                except Exception:
                    pass
                log.warning(
                    "[TokenProvider] %s 登录后未取到 token（url=%s）page=%r",
                    email, page.url, snippet,
                )
                return ""
            return token
        except Exception as exc:
            log.warning("[TokenProvider] %s 浏览器登录失败: %s", email, str(exc)[:160])
            return ""
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def get_token(self, account, *, force: bool = False) -> str:
        """取账号的短效 token；缓存够用直接返回，否则重新登录签发。"""
        email = str(getattr(account, "email", "") or "")
        if not email:
            return ""

        if not force:
            cached, exp = self._cache.get(email, ("", 0.0))
            if cached and exp - time.time() > REFRESH_MARGIN_SECONDS:
                return cached

        lock = self._locks.setdefault(email, asyncio.Lock())
        async with lock:
            if not force:
                cached, exp = self._cache.get(email, ("", 0.0))
                if cached and exp - time.time() > REFRESH_MARGIN_SECONDS:
                    return cached

            async with self._login_lock:
                token = await self._login_once(account)
            if not token:
                # 登录失败时丢弃缓存的 context，下次重建（避免坏 cookie 卡死）
                ctx = self._contexts.pop(email, None)
                if ctx is not None:
                    try:
                        await ctx.close()
                    except Exception:
                        pass
                return ""

            exp = token_expires_at(token)
            if exp <= 0:
                # 解析不出 exp 时给一个保守 TTL，避免每请求都重登
                exp = time.time() + 600
            self._cache[email] = (token, exp)
            log.info(
                "[TokenProvider] %s 签发短效 token ok kind=%s ttl=%.0fs",
                email, token_kind(token), exp - time.time(),
            )
            return token


async def ensure_fresh_token(account, *, force: bool = False) -> str:
    """便捷入口：取账号可用的短效 token，失败返回空串。"""
    if not getattr(settings, "QWEN_BROWSER_TOKEN_ENABLED", True):
        return ""
    return await BrowserTokenProvider.get_instance().get_token(account, force=force)
