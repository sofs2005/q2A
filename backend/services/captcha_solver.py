"""
Captcha Solver — 用 playwright 无头浏览器完成阿里云 x5sec punish 滑块挑战。

工作原理:
1. 检测到 punish 拦截后,从 punish 页 window._config_.url 提取挑战 URL
2. 启动/复用 playwright chromium,先注入账号 token cookie 模拟登录态
3. 导航到 punish URL,阿里云滑块在 headless 下通常自动验证(无需拖拽)
4. 等待页面跳转离开 punish 路径(验证通过的标志)
5. 提取全部 cookie(acw_tc / x5sec / acw_sc__v3 等)返回给调用方

注意:
- x5secdata cookie 只有 20 秒有效期,需快速完成挑战
- headless 可能被检测 webdriver 标志,通过 stealth 参数规避
- 挑战失败时返回空 dict,调用方降级为 1800s 冷却
"""

import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

log = logging.getLogger("qwen2api.captcha_solver")

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent

# punish 页中 window._config_ = {"action":"captcha","url":"..."} 的 URL 提取正则
# 支持双引号和转义引号两种格式
_PUNISH_URL_RE = re.compile(r'window\._config_\s*=\s*\{[^}]*"url"\s*:\s*"([^"]+)"', re.DOTALL)
_PUNISH_URL_RE_ESCAPED = re.compile(r'window\._config_\s*=\s*\{[^}]*\\"url\\"\s*:\s*\\"([^\\]+)\\"', re.DOTALL)

# 备选:直接从 window.location.replace() 提取
_LOCATION_REPLACE_RE = re.compile(r'window\.location\.replace\s*\(\s*"([^"]*_____tmd_____[^"]*)"', re.DOTALL)
_LOCATION_REPLACE_RE_ESCAPED = re.compile(r'window\.location\.replace\s*\(\s*\\"([^\\]*_____tmd_____[^\\]*)\\"', re.DOTALL)

# 放行 cookie 名称(滑块通过后下发的)
PASS_COOKIES = {"x5sec", "acw_sc__v3", "acw_tc", "x5secdata"}

# 浏览器空闲超时(秒),超过后自动关闭以释放内存
_BROWSER_IDLE_TIMEOUT_SECONDS = 30.0

# 滑块元素备选 selectors
_SLIDER_SELECTORS = [
    "#nc_1_n1z",
    "[id*='_n1z']",
    ".nc_scale span.btn_slide",
    ".nc_wrapper .nc_scale span",
    ".btn_slide",
]

try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
        TimeoutError as PlaywrightTimeout,
    )
except ImportError:  # pragma: no cover - 可选依赖,运行时降级
    async_playwright = None
    Browser = object
    BrowserContext = object
    Page = object
    Playwright = object
    PlaywrightTimeout = TimeoutError


def extract_punish_url(punish_body: str) -> Optional[str]:
    """从 punish 响应中提取挑战 URL。

    支持两种响应格式:
    1. HTML 格式: <script>window._config_ = {"url":"..."};</script>
    2. JSON 格式: {"ret":["FAIL_SYS_USER_VALIDATE",...],"data":{"url":"..."}}

    Args:
        punish_body: punish 拦截响应 body

    Returns:
        punish 挑战 URL,提取失败返回 None
    """
    if not punish_body:
        return None

    # 尝试 JSON 格式
    try:
        data = json.loads(punish_body)
        if isinstance(data, dict):
            url = data.get("data", {}).get("url") if isinstance(data.get("data"), dict) else None
            if url and "_____tmd_____" in url:
                log.debug("[CaptchaSolver] extract_punish_url via json data.url: %s...", url[:80])
                return url.replace("\\/", "/")
    except Exception:
        pass

    # 尝试 window._config_ (标准引号)
    m = _PUNISH_URL_RE.search(punish_body)
    if m:
        url = m.group(1).replace("\\/", "/").replace('\\"', '"')
        log.debug("[CaptchaSolver] extract_punish_url via _config_ std: %s...", url[:80])
        return url

    # 尝试 window._config_ (转义引号)
    m = _PUNISH_URL_RE_ESCAPED.search(punish_body)
    if m:
        url = m.group(1).replace("\\/", "/").replace('\\"', '"')
        log.debug("[CaptchaSolver] extract_punish_url via _config_ escaped: %s...", url[:80])
        return url

    # 尝试 window.location.replace() (标准引号)
    m = _LOCATION_REPLACE_RE.search(punish_body)
    if m:
        url = m.group(1).replace("\\/", "/").replace('\\"', '"')
        log.debug("[CaptchaSolver] extract_punish_url via location.replace std: %s...", url[:80])
        return url

    # 尝试 window.location.replace() (转义引号)
    m = _LOCATION_REPLACE_RE_ESCAPED.search(punish_body)
    if m:
        url = m.group(1).replace("\\/", "/").replace('\\"', '"')
        log.debug("[CaptchaSolver] extract_punish_url via location.replace escaped: %s...", url[:80])
        return url

    log.warning("[CaptchaSolver] extract_punish_url: 无法匹配 punish URL, body=%s", punish_body[:300])
    return None


def _cookie_domain_for_base_url(base_url: str) -> str:
    """从 base_url 推导 cookie domain,失败时回退到 .qwen.ai。"""
    try:
        netloc = urlparse(base_url).netloc
        if netloc:
            # 去掉端口,取最后两段作为 domain
            host = netloc.split(":")[0]
            parts = host.split(".")
            if len(parts) >= 2:
                return "." + ".".join(parts[-2:])
    except Exception:
        pass
    return ".qwen.ai"


class CaptchaSolver:
    """用 playwright 完成 x5sec 滑块挑战,提取放行 cookie。

    单例模式管理一个可复用的 Browser 实例,跨调用只创建/关闭 BrowserContext
    和 Page,并在空闲一段时间后自动关闭浏览器以释放内存。
    """

    _instance: Optional["CaptchaSolver"] = None

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None
        self._idle_token = 0
        self._last_used = 0.0
        self._idle_timeout = _BROWSER_IDLE_TIMEOUT_SECONDS

    @classmethod
    def get_instance(cls) -> "CaptchaSolver":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _close_browser(self) -> None:
        """关闭浏览器并停止 playwright,静默处理异常。"""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as e:
                log.debug("[CaptchaSolver] 关闭 browser 异常: %s", str(e)[:120])
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as e:
                log.debug("[CaptchaSolver] 停止 playwright 异常: %s", str(e)[:120])
            self._playwright = None
        log.info("[CaptchaSolver] 浏览器已关闭,释放内存")

    async def _idle_close_browser(self, token: int) -> None:
        """空闲超时后关闭浏览器。"""
        await asyncio.sleep(self._idle_timeout)
        async with self._lock:
            # 若期间有新调用,idle_token 会被递增,直接放弃
            if self._idle_token != token:
                return
            now = asyncio.get_event_loop().time()
            if now - self._last_used < self._idle_timeout:
                return
            await self._close_browser()

    async def _touch(self) -> None:
        """记录一次使用,重置空闲关闭计时器。"""
        async with self._lock:
            self._last_used = asyncio.get_event_loop().time()
            if self._idle_task is not None and not self._idle_task.done():
                self._idle_task.cancel()
            self._idle_token += 1
            self._idle_task = asyncio.create_task(
                self._idle_close_browser(self._idle_token)
            )

    async def _ensure_browser(self) -> Browser:
        """确保复用的 Browser 实例已启动。"""
        async with self._lock:
            if self._browser is not None:
                try:
                    # 简单探测 browser 是否仍连接
                    if self._browser.is_connected():
                        return self._browser
                except Exception:
                    pass
                await self._close_browser()

            if async_playwright is None:
                raise ImportError("playwright 未安装")

            log.info("[CaptchaSolver] 启动 Chromium 浏览器")
            try:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins",
                        "--disable-features=site-per-process",
                        "--disable-client-side-phishing-detection",
                        "--disable-hang-monitor",
                        "--disable-popup-blocking",
                        "--disable-prompt-on-repost",
                        "--disable-sync",
                        "--disable-translate",
                        "--metrics-recording-only",
                        "--no-first-run",
                        "--safebrowsing-disable-auto-update",
                    ],
                )
            except Exception:
                # launch 失败时清理已创建的 playwright,避免进程泄漏
                await self._close_browser()
                raise

            self._last_used = asyncio.get_event_loop().time()
            log.info("[CaptchaSolver] Chromium 启动完成")
            return self._browser

    async def solve_punish(
        self,
        punish_url: str,
        account_token: str,
        base_url: str = "https://chat.qwen.ai",
        timeout_ms: int = 15000,
        debug: bool = False,
    ) -> dict[str, str]:
        """完成 x5sec 滑块挑战,返回放行 cookie 字典。

        Args:
            punish_url: punish 挑战 URL(从 punish 页 window._config_.url 提取)
            account_token: 账号 Bearer token,用于注入登录态
            base_url: chat.qwen.ai 基础 URL,用于推导 cookie domain
            timeout_ms: 挑战超时时间(毫秒),默认 15s 以避免 x5secdata 过期
            debug: 是否保存截图/日志供分析,生产环境建议 False

        Returns:
            cookie 字典 {"acw_tc": "...", "x5sec": "...", ...},失败返回 {}
        """
        if async_playwright is None:
            log.warning("[CaptchaSolver] playwright 未安装,无法完成滑块挑战")
            return {}

        if not punish_url:
            log.warning("[CaptchaSolver] punish_url 为空,跳过滑块挑战")
            return {}

        log.info("[CaptchaSolver] 启动滑块挑战 punish_url=%s...", punish_url[:80])

        debug_dir = ROOT / ".snow" / "log" / "captcha_solver"
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        await self._touch()
        browser = await self._ensure_browser()
        await self._touch()

        cookie_domain = _cookie_domain_for_base_url(base_url)
        result: dict[str, str] = {}
        passed = False

        try:
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
                extra_http_headers={
                    "Accept-Language": "zh-CN,zh;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                },
            )
            try:
                # 注入账号 token cookie 模拟登录态
                if account_token:
                    await context.add_cookies([
                        {
                            "name": "login_tongyi_ticket",
                            "value": account_token,
                            "domain": cookie_domain,
                            "path": "/",
                        },
                    ])

                page = await context.new_page()
                try:
                    console_logs: list[str] = []
                    if debug:
                        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
                        page.on("pageerror", lambda err: console_logs.append(f"[pageerror] {err}"))

                    # 隐藏 webdriver 标志和自动化检测特征
                    await page.add_init_script("""
// 隐藏 navigator.webdriver
Object.defineProperty(navigator, 'webdriver', {get: () => false});
try { delete navigator.webdriver; } catch (e) {}

// 模拟真实浏览器 chrome 对象
window.chrome = {
    runtime: {
        OnInstalledReason: {CHROME_UPDATE: 'chrome_update', SHARED_MODULE_UPDATE: 'shared_module_update'},
        PlatformArch: {ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64'},
        PlatformNaclArch: {ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64'},
        RequestUpdateCheckStatus: {THROTTLED: 'throttled', NO_UPDATE: 'no_update', UPDATE_AVAILABLE: 'update_available'},
        connect: function() {},
        sendMessage: function() {}
    },
    loadTimes: function() {
        return {
            requestTime: Date.now() / 1000,
            startLoadTime: Date.now() / 1000,
            commitLoadTime: Date.now() / 1000,
            finishDocumentLoadTime: Date.now() / 1000,
            finishLoadTime: Date.now() / 1000,
            firstPaintTime: Date.now() / 1000,
            firstPaintAfterLoadTime: 0,
            navigationType: 'Other',
            wasFetchedViaSpdy: true,
            wasNpnNegotiated: true,
            npnNegotiatedProtocol: 'h2',
            wasAlternateProtocolAvailable: false,
            connectionInfo: 'h2'
        };
    },
    csi: function() { return {startE: Date.now(), onloadT: Date.now()}; },
    app: {isInstalled: false, InstallState: {DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed'}, RunningState: {CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running'}}
};

// 模拟 plugins (真实 Chrome 有 3-5 个 PDF/Chrome PDF 等)
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            {name: 'PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1},
            {name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1},
            {name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1},
            {name: 'Microsoft Edge PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1},
            {name: 'WebKit built-in PDF', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1}
        ];
        plugins.item = function(i) { return plugins[i] || null; };
        plugins.namedItem = function(name) { return plugins.find(p => p.name === name) || null; };
        plugins.refresh = function() {};
        return plugins;
    }
});

// 模拟 mimeTypes
Object.defineProperty(navigator, 'mimeTypes', {
    get: () => {
        const mimes = [
            {type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format', enabledPlugin: {name: 'PDF Viewer'}}
        ];
        mimes.item = function(i) { return mimes[i] || null; };
        mimes.namedItem = function(name) { return mimes.find(m => m.type === name) || null; };
        return mimes;
    }
});

// 模拟 languages
Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en-US', 'en']});

// 模拟 permissions (拒绝 'notifications' 和 'midi' 更像真实浏览器)
const originalQuery = navigator.permissions && navigator.permissions.query;
Object.defineProperty(navigator, 'permissions', {
    get: () => ({
        query: (params) => {
            if (params.name === 'notifications') return Promise.resolve({state: Notification.permission});
            if (params.name === 'midi') return Promise.resolve({state: 'prompt'});
            return Promise.resolve({state: 'granted'});
        }
    })
});

// 修复 window.outerWidth/outerHeight (headless 通常为 0)
if (window.outerWidth === 0) Object.defineProperty(window, 'outerWidth', {get: () => 1280});
if (window.outerHeight === 0) Object.defineProperty(window, 'outerHeight', {get: () => 720});

// 模拟 ConnectionInfo
try {
    Object.defineProperty(navigator, 'connection', {
        get: () => ({
            effectiveType: '4g',
            downlink: 10,
            rtt: 50,
            saveData: false,
            addEventListener: function() {},
            removeEventListener: function() {}
        })
    });
} catch (e) {}

// 隐藏 automation 相关属性
try { delete Object.getPrototypeOf(navigator).webdriver; } catch (e) {}

// 隐藏 webdriver 属性
try {
    Object.defineProperty(HTMLElement.prototype, 'webdriver', {get: () => false, configurable: true});
} catch (e) {}

// 移除 document 上的 automation 标志
try {
    window.document.documentElement.removeAttribute('webdriver');
    window.document.documentElement.removeAttribute('driver');
} catch (e) {}

// 拦截 toString 检测 (有些检测调用 toString)
const origToString = Function.prototype.toString;
Function.prototype.toString = function() {
    if (this === Function.prototype.toString) return origToString.call(origToString);
    if (this === navigator.permissions.query) return 'function query() { [native code] }';
    return origToString.call(this);
};
""")

                    # 导航到 punish URL
                    try:
                        await page.goto(punish_url, wait_until="domcontentloaded", timeout=timeout_ms)
                    except Exception as e:
                        log.warning("[CaptchaSolver] 导航 punish URL 异常: %s", str(e)[:200])

                    # ── 自动验证等待期 ──
                    # headless 模式下 x5sec punish 页面可能自动放行（无需拖动）
                    # 先等 3s 轮询 cookie / URL 变化，若已放行则跳过拖动
                    AUTO_VERIFY_WAIT = 3.0
                    auto_verified = False
                    auto_deadline = asyncio.get_event_loop().time() + AUTO_VERIFY_WAIT
                    while asyncio.get_event_loop().time() < auto_deadline:
                        cur_url = page.url or ""
                        if cur_url and "punish" not in cur_url and "_____tmd_____" not in cur_url:
                            log.info("[CaptchaSolver] 自动验证通过（无需拖动），URL=%s", cur_url[:80])
                            auto_verified = True
                            passed = True
                            break
                        # 检查是否已下发 x5sec cookie
                        try:
                            cookies = await context.cookies()
                            cookie_names = {c["name"] for c in cookies}
                            if "x5sec" in cookie_names or "acw_sc__v3" in cookie_names:
                                log.info("[CaptchaSolver] 自动验证通过（检测到放行 cookie）")
                                auto_verified = True
                                passed = True
                                break
                        except Exception:
                            pass
                        await page.wait_for_timeout(300)

                    if not auto_verified:
                        # 自动验证未通过，需要主动拖动滑块
                        log.info("[CaptchaSolver] 自动验证未通过，开始主动拖动滑块")
                        await self._drag_slider(page)

                    # 等待滑块验证完成:页面 URL 不再含 punish/_____tmd_____
                    last_url = ""
                    last_title = ""
                    check_count = 0
                    deadline = asyncio.get_event_loop().time() + timeout_ms / 1000

                    while asyncio.get_event_loop().time() < deadline:
                        try:
                            current_url = page.url or ""
                            last_url = current_url
                            last_title = await page.title()
                            check_count += 1

                            # 每 2 秒输出一次调试信息
                            if check_count % 4 == 1:
                                log.info(
                                    "[CaptchaSolver] 验证检查 #%d: URL=%s, title=%s",
                                    check_count, current_url[:80], last_title,
                                )
                        except Exception:
                            current_url = ""

                        # 验证通过的标志:URL 不再含 punish
                        if current_url and "punish" not in current_url and "_____tmd_____" not in current_url:
                            passed = True
                            break

                        # 检查页面是否还有 captcha 脚本
                        try:
                            has_captcha = await page.evaluate("""
() => {
    const scripts = document.querySelectorAll('script');
    for (const s of scripts) {
        if (s.textContent && s.textContent.includes('captcha')) return true;
    }
    return false;
}
""")
                            if not has_captcha:
                                passed = True
                                break
                        except Exception:
                            pass

                        await asyncio.sleep(0.5)

                    # 保存调试信息(仅视口截图 + 控制台日志,避免大文件)
                    if debug:
                        ts = int(asyncio.get_event_loop().time())
                        try:
                            await page.screenshot(path=str(debug_dir / f"screenshot_{ts}.png"), full_page=False)
                            (debug_dir / f"console_{ts}.log").write_text(
                                "\n".join(console_logs), encoding="utf-8"
                            )
                            log.info("[CaptchaSolver] 调试信息已保存到 %s", debug_dir)
                        except Exception as e:
                            log.warning("[CaptchaSolver] 保存调试信息失败: %s", str(e)[:120])

                    log.info("[CaptchaSolver] 最终 URL=%s title=%s passed=%s", last_url, last_title, passed)

                    if not passed:
                        log.warning("[CaptchaSolver] 滑块挑战超时未通过")
                        return {}

                    # 提取全部 cookie
                    cookies = await context.cookies()
                    for c in cookies:
                        name = c.get("name", "")
                        value = c.get("value", "")
                        if name and value and name in PASS_COOKIES:
                            result[name] = value

                    if result:
                        log.info(
                            "[CaptchaSolver] 滑块挑战完成 passed=%s cookies=%s",
                            passed,
                            list(result.keys()),
                        )
                    else:
                        log.warning("[CaptchaSolver] 滑块挑战完成但未提取到放行 cookie")

                finally:
                    await page.close()
            finally:
                await context.close()

        except Exception as e:
            log.error("[CaptchaSolver] 滑块挑战异常: %s", str(e)[:300])
            return {}

        await self._touch()
        return result

    async def _human_drag(
        self,
        page: Page,
        start_x: float,
        start_y: float,
        end_x: float,
    ) -> dict[str, int]:
        """类人化拖动：稳定悬停、按下晃动、ease-in-out 轨迹、随机微暂停。

        返回实际使用的 {target_time_ms, total_steps} 供日志/调试使用。
        """
        target_time_ms = random.randint(1000, 1800)
        total_steps = random.randint(25, 35)
        avg_delay = target_time_ms / total_steps
        overshoot = random.randint(1, 4)

        # 1. 稳定悬停，让组件触发 mouseenter / pointerenter
        await page.mouse.move(start_x, start_y, steps=random.randint(5, 10))
        await page.wait_for_timeout(random.randint(300, 600))

        # 2. 按下并小幅晃动，确保句柄被"咬住"
        await page.mouse.down()
        await page.wait_for_timeout(random.randint(100, 250))
        await page.mouse.move(
            start_x + random.randint(1, 2),
            start_y + random.randint(-1, 1),
        )
        await page.wait_for_timeout(random.randint(80, 150))

        # 3. ease-in-out cubic 分段拖动
        for i in range(1, total_steps + 1):
            t = i / total_steps
            progress = t * t * (3 - 2 * t)
            current_end = end_x + overshoot
            x = start_x + (current_end - start_x) * progress
            y = start_y + random.randint(-1, 1)

            if random.random() < 0.15:
                x += random.choice([-1, 0, 1])

            await page.mouse.move(x, y)

            if t < 0.15 or t > 0.85:
                speed_factor = random.uniform(1.3, 1.8)
            elif t < 0.5:
                speed_factor = random.uniform(0.6, 0.9)
            else:
                speed_factor = random.uniform(0.7, 1.1)

            delay = int(avg_delay * speed_factor + random.randint(-3, 8))
            delay = max(10, delay)
            await page.wait_for_timeout(delay)

            # 随机微暂停，模拟人类犹豫
            if random.random() < 0.08:
                await page.wait_for_timeout(random.randint(40, 90))

        # 4. 小幅回退修正 overshoot
        if random.random() < 0.35:
            await page.mouse.move(
                end_x + random.randint(-2, 2),
                start_y + random.randint(-1, 1),
            )
            await page.wait_for_timeout(random.randint(80, 180))

        # 5. 释放
        await page.mouse.up()
        await page.wait_for_timeout(random.randint(120, 250))

        return {"target_time_ms": target_time_ms, "total_steps": total_steps}

    async def _drag_slider(self, page: Page, max_wait_ms: int = 10000) -> None:
        """尝试定位并拖动阿里云滑块验证码。

        支持顶层 DOM、iframe 以及 shadow DOM 查找,失败不影响主流程。
        """
        slider = await self._find_slider(page, max_wait_ms)
        if slider is None:
            log.warning("[CaptchaSolver] 未找到滑块元素,跳过主动拖动")
            return

        try:
            # 等待滑块完全渲染，最多重试3次获取 bounding_box
            box = None
            for attempt in range(5):
                await page.wait_for_timeout(200)
                box = await slider.bounding_box()
                if box:
                    break
                log.warning("[CaptchaSolver] 滑块元素 bounding box 为空, 重试第%d次", attempt + 1)
                await page.wait_for_timeout(300)

            if not box:
                log.warning("[CaptchaSolver] 滑块元素 bounding box 始终为空")
                try:
                    html = await page.content()
                    log.info("[CaptchaSolver] 页面 HTML 预览(前2000字符): %s", html[:2000])
                except Exception:
                    pass
                return

            # 记录滑块初始位置
            initial_x = box["x"]
            log.info("[CaptchaSolver] 滑块初始位置 x=%.1f", initial_x)

            track_info = await slider.evaluate("""
(el) => {
    // 多级轨道探测：.nc_scale → .nc_wrapper → 父容器 → 视口估算
    const candidates = ['.nc_scale', '.nc_wrapper', '[class*="scale"]', '[class*="track"]'];
    for (const sel of candidates) {
        const track = el.closest ? el.closest(sel) : null;
        if (track) {
            const rect = track.getBoundingClientRect();
            // 合理性校验：宽度必须在 [100, 800] 范围内
            if (rect.width >= 100 && rect.width <= 800 && rect.left >= 0) {
                return {width: rect.width, left: rect.left, found: true, selector: sel};
            }
        }
    }
    // 所有候选都无效，返回标记
    const elRect = el.getBoundingClientRect();
    return {width: 0, left: elRect.left, found: false, selector: null};
}
""")
            track_width = track_info["width"]
            track_left = track_info["left"]
            track_found = track_info.get("found", False)

            start_x = box["x"] + box["width"] / 2
            start_y = box["y"] + box["height"] / 2

            if track_found:
                end_x = track_left + track_width - box["width"] / 2 - 2
                log.info(
                    "[CaptchaSolver] 轨道探测成功 selector=%s width=%.1f left=%.1f",
                    track_info.get("selector"), track_width, track_left,
                )
            else:
                # 多级 fallback：视口比例估算
                viewport_width = 1280  # 与 BrowserContext viewport 一致
                estimated_track_width = 300  # 阿里云滑块轨道通常 ~300px
                # 尝试从页面获取实际视口宽度
                try:
                    vw = await page.evaluate("() => window.innerWidth")
                    if isinstance(vw, (int, float)) and 320 <= vw <= 3840:
                        viewport_width = vw
                        # 轨道通常是视口宽度的 20%-30%
                        estimated_track_width = max(200, min(350, int(viewport_width * 0.25)))
                except Exception:
                    pass
                log.warning(
                    "[CaptchaSolver] 未找到有效轨道，使用视口估算 viewport=%d track_est=%d",
                    viewport_width, estimated_track_width,
                )
                end_x = start_x + estimated_track_width

            distance = end_x - start_x

            # 安全钳位：距离必须在 [180, 350] 合理区间内
            MIN_DISTANCE = 180
            MAX_DISTANCE = 350
            if distance < MIN_DISTANCE or distance > MAX_DISTANCE:
                log.warning(
                    "[CaptchaSolver] 距离异常 distance=%.1f，钳位到 [%d, %d]",
                    distance, MIN_DISTANCE, MAX_DISTANCE,
                )
                distance = max(MIN_DISTANCE, min(MAX_DISTANCE, distance))
                end_x = start_x + distance

            log.info(
                "[CaptchaSolver] 开始拖动滑块 from=(%.1f,%.1f) to=(%.1f,%.1f) distance=%.1f",
                start_x, start_y, end_x, start_y, distance,
            )

            drag_info = await self._human_drag(page, start_x, start_y, end_x)
            target_time_ms = drag_info["target_time_ms"]
            total_steps = drag_info["total_steps"]
            log.info("[CaptchaSolver] 滑块拖动完成 (目标时间=%dms, 步数=%d)", target_time_ms, total_steps)

            # 6. 检查是否有验证结果提示
            await page.wait_for_timeout(300)

            # 检查滑块位置是否改变
            moved_successfully = False
            try:
                new_box = await slider.bounding_box()
                if new_box:
                    new_x = new_box["x"]
                    moved = new_x - initial_x
                    log.info("[CaptchaSolver] 滑块拖动后位置 x=%.1f, 移动距离=%.1f", new_x, moved)
                    if abs(moved) >= distance * 0.8:
                        moved_successfully = True
                    else:
                        log.warning("[CaptchaSolver] 滑块移动距离过小，可能拖动失败")
            except Exception as e:
                log.warning("[CaptchaSolver] 检查拖动结果异常: %s", str(e)[:100])

            # 如果拖动失败，最多重试1次（x5secdata 仅 20s 有效，不宜浪费时间在多次重试上）
            max_retries = 1
            retry_count = 0
            while not moved_successfully and retry_count < max_retries:
                retry_count += 1
                log.warning("[CaptchaSolver] 第%d次拖动失败，尝试第%d次重试", retry_count, retry_count + 1)
                await page.wait_for_timeout(random.randint(800, 1500))

                try:
                    retry_box = await slider.bounding_box()
                    if not retry_box:
                        continue

                    retry_start_x = retry_box["x"] + retry_box["width"] / 2
                    retry_start_y = retry_box["y"] + retry_box["height"] / 2
                    # 重试时始终回到轨道终点作为目标，避免越界
                    retry_end_x = end_x

                    log.info(
                        "[CaptchaSolver] 重试%d: 拖动滑块 from=(%.1f,%.1f) to=(%.1f,%.1f)",
                        retry_count, retry_start_x, retry_start_y, retry_end_x, retry_start_y,
                    )

                    await self._human_drag(page, retry_start_x, retry_start_y, retry_end_x)
                    log.info("[CaptchaSolver] 重试%d 拖动完成", retry_count)

                    check_box = await slider.bounding_box()
                    if check_box:
                        retry_moved = check_box["x"] - retry_box["x"]
                        log.info("[CaptchaSolver] 重试%d 后移动距离=%.1f", retry_count, retry_moved)
                        if abs(retry_moved) >= distance * 0.8:
                            moved_successfully = True

                except Exception as e:
                    log.warning("[CaptchaSolver] 重试%d 拖动异常: %s", retry_count, str(e)[:100])

                try:
                    # 检查是否有验证失败提示
                    error_text = await page.evaluate("""
() => {
    // 检查常见的错误提示元素
    const selectors = ['.nc_iconfont.btn_ok', '.nc_wrapper .nc_iconfont', '.errloading', '.nc_error'];
    for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el) return el.className + ': ' + (el.textContent || '').trim();
    }
    return null;
}
""")
                    if error_text:
                        log.info("[CaptchaSolver] 检测到验证状态: %s", error_text)
                except Exception:
                    pass

        except PlaywrightTimeout:
            log.warning("[CaptchaSolver] 查找滑块元素超时")
        except Exception as e:
            log.warning("[CaptchaSolver] 拖动滑块异常: %s", str(e)[:200])

    async def _find_slider(self, page: Page, max_wait_ms: int) -> Optional[Any]:
        """在顶层 DOM、iframe 和 shadow DOM 中查找可见的滑块元素。

        Returns:
            滑块 ElementHandle,未找到返回 None
        """
        deadline = asyncio.get_event_loop().time() + max_wait_ms / 1000

        while asyncio.get_event_loop().time() < deadline:
            # 1. 顶层 DOM
            for sel in _SLIDER_SELECTORS:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        log.debug("[CaptchaSolver] 顶层找到滑块 selector=%s", sel)
                        return el
                except Exception:
                    pass

            # 2. 跨 iframe 查找
            for frame in page.frames:
                if frame is page.main_frame:
                    continue
                for sel in _SLIDER_SELECTORS:
                    try:
                        el = await frame.query_selector(sel)
                        if el and await el.is_visible():
                            log.debug("[CaptchaSolver] iframe 找到滑块 selector=%s", sel)
                            return el
                    except Exception:
                        pass

            # 3. shadow DOM 查找
            try:
                js_handle = await page.evaluate_handle(
                    """(selectors) => {
    function queryShadow(root, selectors) {
        for (const sel of selectors) {
            const found = root.querySelector(sel);
            if (found && found.offsetParent !== null) return found;
        }
        const shadows = root.querySelectorAll('*');
        for (const node of shadows) {
            if (node.shadowRoot) {
                const found = queryShadow(node.shadowRoot, selectors);
                if (found) return found;
            }
        }
        return null;
    }
    return queryShadow(document, selectors);
}""",
                    _SLIDER_SELECTORS,
                )
                el = js_handle.as_element()
                if el and await el.is_visible():
                    log.debug("[CaptchaSolver] shadow DOM 找到滑块")
                    return el
            except Exception:
                pass

            await asyncio.sleep(0.3)

        return None

    def cookies_dict_to_string(self, cookies: dict[str, str]) -> str:
        """将 cookie 字典转为 HTTP Cookie 头字符串。

        Args:
            cookies: {"acw_tc": "...", "x5sec": "..."}

        Returns:
            "acw_tc=...; x5sec=..."
        """
        return "; ".join(f"{k}={v}" for k, v in cookies.items() if k and v)
