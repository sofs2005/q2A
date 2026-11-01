"""浏览器上传代理单测（不连真浏览器）。

用 fake page / fake CDP session 覆盖：

- WAF → 解验证码 → 重试 的循环，以及轮次上限；
- 账号级熔断（连续失败后 10 分钟不再尝试）；
- 标定缓存读写 + `linear` / `quadratic` / `auto` 三种映射的选择；
- 拖动轨迹的统计特征（时长 / 事件数 / 停顿 / y 抖动）；
- 请求头 / cookie / JWT 过期时间等纯函数。

真实浏览器只由 `backend/services/browser_upload_proxy.py` 在运行时连，单测一律 fake。
"""

import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import browser_upload_proxy as bux
from backend.services.browser_upload_proxy import (
    BUILTIN_QUADRATIC_A,
    BUILTIN_QUADRATIC_B,
    BrowserUploadError,
    BrowserUploadProxy,
    BrowserUploadUnavailable,
    DragCalibration,
    build_drag_trajectory,
    build_request_headers,
    cookies_for_context,
    extract_verify_code,
    fit_calibration,
    linear_mouse_dx,
    load_calibration,
    mouse_dx_for_piece,
    normalize_drag_mapping,
    parse_cookie_header,
    quadratic_mouse_dx,
    sanitize_fetch_headers,
    save_calibration,
    token_expire_ms,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "captcha"

#: 真实抓到的阿里云 WAF 页面特征片段（含 aliyun_waf / sceneId）。
WAF_HTML = (
    "<!doctype html><html><head><script>"
    "window.aliyun_waf_aa=1;var sceneId='19x5u7lo';var region='sgp';"
    "</script></head><body>aliyunCaptcha</body></html>"
)

#: 真实 getstsToken 的 data（字段取自 findings §9 的实测抓包）。
STS_DATA = {
    "file_id": "079054e8-4e5e-433d-a94a-9a41f146cba2",
    "file_path": "18639e0d-1111-4222-8333-444455556666/079054e8-4e5e-433d-a94a-9a41f146cba2_test-2m.mp4",
    "file_url": "https://cdn.qwenlm.ai/18639e0d-1111-4222-8333-444455556666/"
    "079054e8-4e5e-433d-a94a-9a41f146cba2_test-2m.mp4?key=eyJhbGciOiJIUzI1NiJ9.payload.sig",
    "bucketname": "qwen-webui-prod",
    "endpoint": "qwen-webui-prod.oss-accelerate.aliyuncs.com",
    "region": "oss-ap-southeast-1",
    "access_key_id": "STS.NTestAccessKeyId",
    "access_key_secret": "test-access-key-secret",
    "security_token": "test-security-token",
}

VIDEO_SIZE = 2058934


def _captcha_assets(capture: str = "cap1") -> tuple[str, str, dict[str, bytes]]:
    """构造一对素材 URL + 字节（用真实 fixture，让定位算法真的跑一遍）。"""
    shadow = (FIXTURES / capture / "shadow.png").read_bytes()
    back = (FIXTURES / capture / "back.png").read_bytes()
    base = f"https://static-captcha-sgp.aliyuncs.com/qst/PUZZLE/online/903/uuid-{capture}"
    shadow_url = f"{base}/shadow.png"
    back_url = f"{base}/back.png"
    return shadow_url, back_url, {shadow_url: shadow, back_url: back}


def _account(email: str = "alice@example.com") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        email=email,
        token="eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE4OTM0NTYwMDB9.signature",
        cookies="sid=abc; uid=42",
        waf_cookies="acw_tc=waf-token",
        waf_cookies_expires_at=4102444800.0,
    )


def _settings(**overrides) -> types.SimpleNamespace:
    base = {
        "UPLOAD_CDP_ENDPOINT": "http://10.6.6.4:3000",
        "UPLOAD_VIA_BROWSER": "auto",
        "UPLOAD_CAPTCHA_MAX_ROUNDS": 3,
        "UPLOAD_CAPTCHA_TIMEOUT_SECONDS": 5,
        "UPLOAD_DRAG_MAPPING": "linear",
        "UPLOAD_BROWSER_TIMEOUT_SECONDS": 30,
        "UPLOAD_DRAG_CALIBRATION_FILE": "",
        "QWEN_WEB_VERSION": "0.3.28",
        "QWEN_BX_VERSION": "2.5.37",
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


class FakeResponse:
    """`page.request.get()` 的返回值。"""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def body(self) -> bytes:
        return self._body


class FakeRequestContext:
    """`page.request`：按 URL 下发素材字节。"""

    def __init__(self, assets: dict[str, bytes]) -> None:
        self.assets = dict(assets)
        self.requested: list[str] = []

    async def get(self, url: str) -> FakeResponse:
        self.requested.append(url)
        if url not in self.assets:
            raise RuntimeError(f"unexpected asset url: {url}")
        return FakeResponse(self.assets[url])


class FakeElement:
    def __init__(self, visible: bool = True) -> None:
        self._visible = visible

    async def is_visible(self) -> bool:
        return self._visible


class FakeCdpSession:
    """记录 `Input.dispatchMouseEvent` 调用。"""

    def __init__(self, on_release=None) -> None:
        self.events: list[tuple[str, dict]] = []
        self._on_release = on_release

    async def send(self, method: str, params: dict) -> None:
        self.events.append((method, dict(params)))
        if params.get("type") == "mouseReleased" and self._on_release is not None:
            self._on_release()

    @property
    def event_types(self) -> list[str]:
        return [params.get("type", "") for _method, params in self.events]


class FakePage:
    """最小 playwright Page 替身，按 JS 内容分发 evaluate。"""

    def __init__(
        self,
        *,
        fetch_results: list,
        assets: dict[str, bytes],
        display_width: float = 300.0,
        slider_visible: bool = True,
        slider_visible_after_drag: bool = False,
        puzzle_offset: float | None = None,
    ) -> None:
        self.fetch_results = list(fetch_results)
        self.request = FakeRequestContext(assets)
        self.display_width = display_width
        self.slider_visible = slider_visible
        self.slider_visible_after_drag = slider_visible_after_drag
        self.puzzle_offset = puzzle_offset
        self.fetch_calls: list[list] = []
        self.waits: list[tuple[str, int | None]] = []
        self._assets_by_kind = {
            kind: url
            for url in assets
            for kind in ("shadow", "back")
            if url.endswith(f"{kind}.png")
        }

    # -- Page API ---------------------------------------------------------

    async def evaluate(self, script: str, arg=None):
        text = str(script)
        if "fetch(" in text:
            self.fetch_calls.append(arg)
            return self._next_fetch()
        if "getEntriesByType" in text:
            return {
                "shadow": self._assets_by_kind.get("shadow", ""),
                "back": self._assets_by_kind.get("back", ""),
            }
        if "aliyunCaptcha-sliding-slider" in text:
            return {"x": 100.0, "y": 500.0, "width": 40.0, "height": 40.0}
        if "aliyunCaptcha-img" in text:
            return {"width": self.display_width, "height": 200.0}
        if "aliyunCaptcha-puzzle" in text:
            return self.puzzle_offset
        return None

    async def wait_for_selector(self, selector: str, timeout: int | None = None, state=None):
        del state
        self.waits.append((selector, timeout))
        if not self.slider_visible:
            raise TimeoutError("slider never appeared")
        return FakeElement(self.slider_visible)

    async def query_selector(self, selector: str):
        del selector
        return FakeElement(self.slider_visible) if self.slider_visible else None

    def mark_drag_released(self) -> None:
        self.slider_visible = self.slider_visible_after_drag

    # -- 内部 -------------------------------------------------------------

    def _next_fetch(self) -> dict:
        if not self.fetch_results:
            return {"status": 403, "body": WAF_HTML}
        item = self.fetch_results.pop(0)
        if callable(item):
            item = item(self)
        kind, payload = item
        if kind == "waf":
            return {"status": 403, "body": payload}
        if kind == "raw":
            return {"status": 200, "body": payload}
        return {"status": 200, "body": json.dumps(payload)}


class ProxyTestBase(unittest.IsolatedAsyncioTestCase):
    """公共装配：fake page + fake cdp + 零耗时 sleep。"""

    def build_proxy(self, settings=None, calibration_path=None, account=None):
        del account
        if calibration_path is None:
            # 单测绝不写仓库里的 data/drag_calibration.json（会污染其他用例）。
            calibration_path = Path(tempfile.mkdtemp(prefix="q2a-cal-")) / "drag_calibration.json"
        proxy = BrowserUploadProxy(
            settings or _settings(),
            calibration_path=calibration_path,
            clock=lambda: 1_000_000.0,
            rng=random.Random(20261031),
        )

        async def _no_sleep(_seconds: float) -> None:
            return None

        proxy._sleep = _no_sleep  # type: ignore[assignment]
        return proxy

    def build_page(self, *, fetch_results, capture="cap1", **kwargs) -> tuple[FakePage, FakeCdpSession]:
        shadow_url, back_url, assets = _captcha_assets(capture)
        del shadow_url, back_url
        page = FakePage(fetch_results=fetch_results, assets=assets, **kwargs)
        cdp = FakeCdpSession(on_release=page.mark_drag_released)
        return page, cdp

    @staticmethod
    def patch_page(proxy: BrowserUploadProxy, page: FakePage, cdp: FakeCdpSession):
        async def _ensure_page(_account):
            return page, cdp

        return patch.object(proxy, "_ensure_page", _ensure_page)


class FetchStsLoopTests(ProxyTestBase):
    """WAF → 求解 → 重试的循环与轮次上限。"""

    async def test_waf_then_json_solves_captcha_and_returns_sts(self) -> None:
        page, cdp = self.build_page(fetch_results=[("waf", WAF_HTML), ("json", {"data": STS_DATA})])
        proxy = self.build_proxy()
        solves: list[int] = []
        original = proxy.solve_puzzle

        async def counting(page_arg, cdp_arg):
            solves.append(1)
            return await original(page_arg, cdp_arg)

        proxy.solve_puzzle = counting  # type: ignore[assignment]

        with self.patch_page(proxy, page, cdp):
            result = await proxy.fetch_sts(_account(), "test-2m.mp4", "video/mp4", VIDEO_SIZE)

        self.assertIsNotNone(result)
        self.assertEqual(result["file_id"], STS_DATA["file_id"])
        # 一轮 WAF → 一轮求解 → 第二次请求直接拿到 JSON
        self.assertEqual(len(solves), 1)
        self.assertEqual(len(page.fetch_calls), 2)
        # 请求体与网页端抓包一致（filesize 是字符串）
        url, _headers, body = page.fetch_calls[0]
        self.assertEqual(url, "https://chat.qwen.ai/api/v2/files/getstsToken")
        self.assertEqual(body["filename"], "test-2m.mp4")
        self.assertEqual(body["filesize"], str(VIDEO_SIZE))
        self.assertEqual(body["filetype"], "video")
        # 拖动走 CDP 合成事件，且成对出现 press/release
        self.assertIn("mousePressed", cdp.event_types)
        self.assertIn("mouseReleased", cdp.event_types)
        self.assertEqual(cdp.event_types[0], "mouseMoved")  # 起手前先 hover
        self.assertTrue(proxy.recent_captcha_pass())
        self.assertEqual(proxy.failure_streak("alice@example.com"), 0)

    async def test_two_waf_rounds_then_json(self) -> None:
        page, cdp = self.build_page(
            fetch_results=[("waf", WAF_HTML), ("waf", WAF_HTML), ("json", {"data": STS_DATA})]
        )
        proxy = self.build_proxy(_settings(UPLOAD_CAPTCHA_MAX_ROUNDS=3))
        solves: list[int] = []
        original = proxy.solve_puzzle

        async def counting(page_arg, cdp_arg):
            solves.append(1)
            return await original(page_arg, cdp_arg)

        proxy.solve_puzzle = counting  # type: ignore[assignment]

        with self.patch_page(proxy, page, cdp):
            result = await proxy.fetch_sts(_account(), "test-2m.mp4", "video/mp4", VIDEO_SIZE)

        self.assertEqual(result["file_id"], STS_DATA["file_id"])
        self.assertEqual(len(solves), 2)
        self.assertEqual(len(page.fetch_calls), 3)

    async def test_rounds_limit_raises_and_opens_circuit_breaker(self) -> None:
        # 滑块一直不消失 → 每轮都判失败
        page, cdp = self.build_page(
            fetch_results=[("waf", WAF_HTML) for _ in range(10)],
            slider_visible_after_drag=True,
        )
        proxy = self.build_proxy(_settings(UPLOAD_CAPTCHA_MAX_ROUNDS=2))
        solves: list[int] = []
        original = proxy.solve_puzzle

        async def counting(page_arg, cdp_arg):
            solves.append(1)
            return await original(page_arg, cdp_arg)

        proxy.solve_puzzle = counting  # type: ignore[assignment]

        with self.patch_page(proxy, page, cdp):
            with self.assertRaises(BrowserUploadError):
                await proxy.fetch_sts(_account(), "test-2m.mp4", "video/mp4", VIDEO_SIZE)

        self.assertEqual(len(solves), 2)
        self.assertEqual(len(page.fetch_calls), 3)  # 1 次初始 + 每轮解完再试 2 次
        self.assertTrue(proxy.is_cooled_down("alice@example.com"))

        # 熔断后不再连浏览器：第二次调用直接 BrowserUploadUnavailable
        with self.patch_page(proxy, page, cdp):
            with self.assertRaises(BrowserUploadUnavailable):
                await proxy.fetch_sts(_account(), "test-2m.mp4", "video/mp4", VIDEO_SIZE)
        self.assertEqual(len(page.fetch_calls), 3)

    async def test_circuit_breaker_is_per_account(self) -> None:
        page, cdp = self.build_page(
            fetch_results=[("waf", WAF_HTML) for _ in range(10)],
            slider_visible_after_drag=True,
        )
        proxy = self.build_proxy(_settings(UPLOAD_CAPTCHA_MAX_ROUNDS=1))
        with self.patch_page(proxy, page, cdp):
            with self.assertRaises(BrowserUploadError):
                await proxy.fetch_sts(_account("alice@example.com"), "a.mp4", "video/mp4", 10)
        self.assertTrue(proxy.is_cooled_down("alice@example.com"))
        self.assertFalse(proxy.is_cooled_down("bob@example.com"))

    async def test_success_resets_failure_streak(self) -> None:
        page, cdp = self.build_page(
            fetch_results=[("waf", WAF_HTML), ("json", {"data": STS_DATA})],
        )
        proxy = self.build_proxy(_settings(UPLOAD_CAPTCHA_MAX_ROUNDS=3))
        proxy._register_failure("alice@example.com")
        self.assertEqual(proxy.failure_streak("alice@example.com"), 1)
        with self.patch_page(proxy, page, cdp):
            await proxy.fetch_sts(_account(), "test-2m.mp4", "video/mp4", VIDEO_SIZE)
        self.assertEqual(proxy.failure_streak("alice@example.com"), 0)
        self.assertFalse(proxy.is_cooled_down("alice@example.com"))

    async def test_disabled_when_endpoint_missing_returns_none(self) -> None:
        proxy = self.build_proxy(_settings(UPLOAD_CDP_ENDPOINT=""))
        self.assertIsNone(await proxy.fetch_sts(_account(), "a.mp4", "video/mp4", 10))

    async def test_disabled_when_mode_never_returns_none(self) -> None:
        proxy = self.build_proxy(_settings(UPLOAD_VIA_BROWSER="never"))
        self.assertIsNone(await proxy.fetch_sts(_account(), "a.mp4", "video/mp4", 10))

    async def test_non_json_body_raises_browser_upload_error(self) -> None:
        page, cdp = self.build_page(fetch_results=[("raw", "<html>502 Bad Gateway</html>")])
        proxy = self.build_proxy()
        with self.patch_page(proxy, page, cdp):
            with self.assertRaises(BrowserUploadError):
                await proxy.fetch_sts(_account(), "a.mp4", "video/mp4", 10)

    async def test_missing_file_id_raises_browser_upload_error(self) -> None:
        page, cdp = self.build_page(fetch_results=[("json", {"data": {"code": "Bad_Request"}})])
        proxy = self.build_proxy()
        with self.patch_page(proxy, page, cdp):
            with self.assertRaises(BrowserUploadError):
                await proxy.fetch_sts(_account(), "a.mp4", "video/mp4", 10)

    async def test_rate_limited_maps_to_dedicated_error(self) -> None:
        page, cdp = self.build_page(
            fetch_results=[
                (
                    "json",
                    {"data": {"code": "RateLimited", "num": 19, "details": "daily limit reached"}},
                )
            ]
        )
        proxy = self.build_proxy()
        with self.patch_page(proxy, page, cdp):
            with self.assertRaises(bux.BrowserUploadRateLimited) as ctx:
                await proxy.fetch_sts(_account(), "test-2m.mp4", "video/mp4", VIDEO_SIZE)
        self.assertAlmostEqual(ctx.exception.retry_after_seconds, 19 * 3600.0)
        # 配额类失败也要计入熔断（避免继续打风控）
        self.assertEqual(proxy.failure_streak("alice@example.com"), 1)

    async def test_solver_returns_none_when_assets_missing(self) -> None:
        page = FakePage(fetch_results=[], assets={})
        cdp = FakeCdpSession()
        proxy = self.build_proxy()
        self.assertIsNone(await proxy.solve_puzzle(page, cdp))

    async def test_solver_returns_none_when_slider_never_appears(self) -> None:
        page, cdp = self.build_page(fetch_results=[], slider_visible=False)
        proxy = self.build_proxy()
        self.assertIsNone(await proxy.solve_puzzle(page, cdp))
        self.assertEqual(page.waits[0][0], bux.SLIDER_SELECTOR)


class DragMappingTests(ProxyTestBase):
    """三种映射的选择 + 标定缓存读写。"""

    def test_linear_mapping_is_one_to_one(self) -> None:
        self.assertEqual(mouse_dx_for_piece(157.0, mapping="linear"), 157.0)
        self.assertEqual(linear_mouse_dx(157.0), 157.0)

    def test_quadratic_mapping_matches_measured_curve(self) -> None:
        # 实测：拼图要动 157px，鼠标要拖 199.7px
        self.assertAlmostEqual(quadratic_mouse_dx(157.0), 199.7, delta=0.5)
        self.assertAlmostEqual(
            quadratic_mouse_dx(157.0, BUILTIN_QUADRATIC_A, BUILTIN_QUADRATIC_B),
            199.7,
            delta=0.5,
        )
        self.assertAlmostEqual(mouse_dx_for_piece(157.0, mapping="quadratic"), 199.7, delta=0.5)

    def test_quadratic_inverts_forward_mapping(self) -> None:
        calibration = DragCalibration(a=BUILTIN_QUADRATIC_A, b=BUILTIN_QUADRATIC_B)
        for piece_dx in (10.0, 55.0, 157.0, 260.0):
            mouse_dx = calibration.mouse_dx(piece_dx)
            self.assertAlmostEqual(calibration.piece_dx(mouse_dx), piece_dx, places=6)

    def test_auto_without_calibration_falls_back_to_quadratic(self) -> None:
        self.assertAlmostEqual(mouse_dx_for_piece(157.0, mapping="auto"), 199.7, delta=0.5)

    def test_auto_uses_calibration_when_present(self) -> None:
        calibration = DragCalibration(a=0.001, b=0.5)
        # piece = 0.001*m² + 0.5*m；piece=157 → 解出 mouse ≈ 173.x
        expected = quadratic_mouse_dx(157.0, 0.001, 0.5)
        self.assertAlmostEqual(
            mouse_dx_for_piece(157.0, mapping="auto", calibration=calibration),
            expected,
            places=6,
        )

    def test_linear_mode_ignores_cached_calibration(self) -> None:
        calibration = DragCalibration(a=0.003555, b=0.075899)
        self.assertEqual(
            mouse_dx_for_piece(157.0, mapping="linear", calibration=calibration),
            157.0,
        )

    def test_normalize_drag_mapping_rejects_unknown(self) -> None:
        self.assertEqual(normalize_drag_mapping("LINEAR"), "linear")
        self.assertEqual(normalize_drag_mapping("quadratic"), "quadratic")
        self.assertEqual(normalize_drag_mapping("nonsense"), "auto")
        self.assertEqual(normalize_drag_mapping(""), "auto")

    def test_fit_calibration_recovers_quadratic_from_two_points(self) -> None:
        samples = [
            (40.0, BUILTIN_QUADRATIC_A * 40.0**2 + BUILTIN_QUADRATIC_B * 40.0),
            (80.0, BUILTIN_QUADRATIC_A * 80.0**2 + BUILTIN_QUADRATIC_B * 80.0),
        ]
        calibration = fit_calibration(samples, now=123.0)
        self.assertIsNotNone(calibration)
        self.assertAlmostEqual(calibration.a, BUILTIN_QUADRATIC_A, places=6)
        self.assertAlmostEqual(calibration.b, BUILTIN_QUADRATIC_B, places=5)
        self.assertEqual(calibration.measured_at, 123.0)
        self.assertEqual(len(calibration.samples), 2)

    def test_fit_calibration_degrades_to_linear(self) -> None:
        # 两点几乎共线（piece == 0.5 * mouse）→ 二次项≈0，退化成线性
        calibration = fit_calibration([(40.0, 20.0), (80.0, 40.0)])
        self.assertIsNotNone(calibration)
        self.assertTrue(calibration.is_linear)
        self.assertAlmostEqual(calibration.mouse_dx(100.0), 200.0, places=6)

    def test_fit_calibration_rejects_insufficient_samples(self) -> None:
        self.assertIsNone(fit_calibration([(40.0, 8.7)]))
        self.assertIsNone(fit_calibration([]))
        self.assertIsNone(fit_calibration([(40.0, 0.0), (80.0, 0.0)]))

    def test_calibration_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drag_calibration.json"
            calibration = DragCalibration(
                a=BUILTIN_QUADRATIC_A,
                b=BUILTIN_QUADRATIC_B,
                measured_at=1234.5,
                samples=[{"mouse_dx": 40.0, "piece_dx": 8.7}],
                source="measured",
            )
            save_calibration(path, calibration)
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["a"], BUILTIN_QUADRATIC_A, places=6)
            self.assertAlmostEqual(payload["b"], BUILTIN_QUADRATIC_B, places=6)
            self.assertEqual(payload["measured_at"], 1234.5)

            reloaded = load_calibration(path)
            self.assertIsNotNone(reloaded)
            self.assertAlmostEqual(reloaded.a, BUILTIN_QUADRATIC_A, places=6)
            self.assertAlmostEqual(reloaded.b, BUILTIN_QUADRATIC_B, places=6)

    def test_load_calibration_tolerates_broken_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drag_calibration.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(load_calibration(path))
            path.write_text(json.dumps({"a": "x", "b": 1.0}), encoding="utf-8")
            self.assertIsNone(load_calibration(path))
            self.assertIsNone(load_calibration(Path(tmpdir) / "missing.json"))

    async def test_auto_mode_reads_cache_instead_of_calibrating(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drag_calibration.json"
            save_calibration(
                path,
                DragCalibration(a=0.003555, b=0.075899, measured_at=1.0, source="measured"),
            )
            proxy = self.build_proxy(
                _settings(UPLOAD_DRAG_MAPPING="auto", UPLOAD_DRAG_CALIBRATION_FILE=str(path)),
                calibration_path=path,
            )
            page, cdp = self.build_page(fetch_results=[])
            resolution = await proxy._resolve_mapping(page, cdp, mode="auto")
            self.assertFalse(resolution.calibrated_now)
            self.assertIsNotNone(resolution.calibration)
            self.assertAlmostEqual(resolution.mouse_dx(157.0), 199.7, delta=0.5)

    async def test_auto_mode_calibrates_two_points_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drag_calibration.json"
            proxy = self.build_proxy(
                _settings(UPLOAD_DRAG_MAPPING="auto", UPLOAD_DRAG_CALIBRATION_FILE=str(path)),
                calibration_path=path,
            )
            probes: list[float] = []

            async def fake_probe(_page, _cdp, mouse_dx: float) -> float:
                probes.append(mouse_dx)
                # 用内置二次常数造一个「真值」，标定应能反解回来
                return BUILTIN_QUADRATIC_A * mouse_dx * mouse_dx + BUILTIN_QUADRATIC_B * mouse_dx

            proxy._probe_hold_offset = fake_probe  # type: ignore[assignment]
            page, cdp = self.build_page(fetch_results=[])

            calibration = await proxy.calibrate_drag(page, cdp)

            self.assertIsNotNone(calibration)
            self.assertEqual(probes, [40.0, 80.0])
            self.assertAlmostEqual(calibration.a, BUILTIN_QUADRATIC_A, places=6)
            self.assertAlmostEqual(calibration.b, BUILTIN_QUADRATIC_B, places=5)
            self.assertTrue(path.exists())
            reloaded = load_calibration(path)
            self.assertIsNotNone(reloaded)
            self.assertAlmostEqual(reloaded.a, calibration.a, places=6)
            # 标定成功后 auto 模式立刻复用内存里的标定，不再重复标定
            resolution = await proxy._resolve_mapping(page, cdp, mode="auto")
            self.assertFalse(resolution.calibrated_now)
            self.assertAlmostEqual(resolution.mouse_dx(157.0), 199.7, delta=0.5)

    async def test_calibration_failure_falls_back_to_builtin_constants(self) -> None:
        proxy = self.build_proxy(_settings(UPLOAD_DRAG_MAPPING="auto"))

        async def failing_probe(_page, _cdp, _mouse_dx: float):
            return None

        proxy._probe_hold_offset = failing_probe  # type: ignore[assignment]
        page, cdp = self.build_page(fetch_results=[])

        self.assertIsNone(await proxy.calibrate_drag(page, cdp))
        resolution = await proxy._resolve_mapping(page, cdp, mode="auto")
        self.assertFalse(resolution.calibrated_now)
        self.assertIsNone(resolution.calibration)
        self.assertAlmostEqual(resolution.mouse_dx(157.0), 199.7, delta=0.5)

    async def test_solve_puzzle_retries_after_calibration_spoils_puzzle(self) -> None:
        """auto 模式标定会把当前拼图拖废 → 必须用新拼图再拖一次。"""
        page, cdp = self.build_page(fetch_results=[], capture="cap2")
        proxy = self.build_proxy(_settings(UPLOAD_DRAG_MAPPING="auto"))

        async def fake_probe(_page, _cdp, mouse_dx: float) -> float:
            return BUILTIN_QUADRATIC_A * mouse_dx * mouse_dx + BUILTIN_QUADRATIC_B * mouse_dx

        proxy._probe_hold_offset = fake_probe  # type: ignore[assignment]
        verify = await proxy.solve_puzzle(page, cdp)

        self.assertEqual(verify, "T001")
        # 标定（fake，不发 CDP 事件）把当前拼图拖废 → 重新取素材（shadow + back）
        # 后再真正拖一次，因此素材只下载一轮、CDP 只有一对 press/release。
        self.assertEqual(len(page.request.requested), 2)
        self.assertEqual(cdp.event_types.count("mousePressed"), 1)
        self.assertEqual(cdp.event_types.count("mouseReleased"), 1)


class TrajectoryTests(unittest.TestCase):
    """拖动轨迹的统计特征（拟人度的下限约束）。"""

    def test_trajectory_constraints(self) -> None:
        for seed in range(8):
            moves = build_drag_trajectory(120.0, 480.0, 199.7, rng=random.Random(seed))
            total_ms = sum(move.delay_ms for move in moves)
            self.assertGreaterEqual(len(moves), 250)
            self.assertLessEqual(len(moves), 320)
            self.assertGreaterEqual(total_ms, 2590.0)
            self.assertLessEqual(total_ms, 3410.0)
            pauses = [move for move in moves if move.pause]
            self.assertGreaterEqual(len(pauses), 2)
            self.assertLessEqual(len(pauses), 3)
            for pause in pauses:
                self.assertGreaterEqual(pause.delay_ms, 300.0)
                self.assertLessEqual(pause.delay_ms, 700.0 + 20.0)
            for move in moves:
                self.assertGreaterEqual(move.delay_ms, 1.0)
                self.assertLessEqual(abs(move.y - 480.0), 5.0)
            # 终点必须精确落在目标位置，否则缺口永远对不上
            self.assertAlmostEqual(moves[-1].x, 120.0 + 199.7, places=3)

    def test_trajectory_x_is_monotonic(self) -> None:
        moves = build_drag_trajectory(0.0, 100.0, 150.0, rng=random.Random(7))
        xs = [move.x for move in moves]
        self.assertEqual(xs, sorted(xs))
        self.assertGreaterEqual(xs[0], 0.0)
        self.assertLessEqual(xs[-1], 150.0)


class HelperTests(unittest.TestCase):
    """请求头 / cookie / verify 码 / JWT 过期时间等纯函数。"""

    def test_sanitize_fetch_headers_drops_forbidden_names(self) -> None:
        headers = build_request_headers(_account())
        self.assertIn("Authorization", headers)
        sanitized = sanitize_fetch_headers(headers)
        self.assertIn("Authorization", sanitized)
        self.assertIn("X-Request-Id", sanitized)
        for name in ("cookie", "user-agent", "referer", "origin"):
            self.assertNotIn(name, {key.lower() for key in sanitized})

    def test_parse_cookie_header_and_context_cookies(self) -> None:
        parsed = parse_cookie_header("acw_tc=1; sid=abc; broken; =empty")
        self.assertEqual(parsed["acw_tc"], "1")
        self.assertEqual(parsed["sid"], "abc")
        self.assertNotIn("broken", parsed)
        cookies = cookies_for_context("acw_tc=1; sid=abc")
        self.assertEqual(len(cookies), 2)
        self.assertEqual(cookies[0]["domain"], ".qwen.ai")
        self.assertEqual(cookies[0]["path"], "/")

    def test_extract_verify_code(self) -> None:
        self.assertEqual(extract_verify_code('{"code":"T001"}'), "T001")
        self.assertEqual(extract_verify_code('{"code":"F015"}'), "F015")
        self.assertEqual(extract_verify_code(""), "")

    def test_token_expire_ms_reads_jwt_exp(self) -> None:
        token = "header.eyJleHAiOjE4OTM0NTYwMDB9.signature"
        self.assertEqual(token_expire_ms(token, now=0.0), 1893456000000)
        # 解不出来时退回 now + 默认 TTL
        self.assertEqual(token_expire_ms("garbage", now=1000.0, default_ttl_seconds=60.0), 1_060_000)

    def test_probe_reports_missing_endpoint(self) -> None:
        import asyncio

        proxy = BrowserUploadProxy(_settings(UPLOAD_CDP_ENDPOINT=""), clock=lambda: 5.0)
        result = asyncio.run(proxy.probe())
        self.assertFalse(result["connected"])
        self.assertEqual(result["browser_version"], "")
        self.assertIn("error", result)


class ManagerWiringTests(unittest.TestCase):
    """附件链路的开关判定（`context_attachment_manager._browser_upload_enabled`）。"""

    def test_requires_proxy_and_endpoint_and_not_never(self) -> None:
        from backend.services.context_attachment_manager import _browser_upload_enabled

        proxy = object()
        self.assertTrue(
            _browser_upload_enabled(
                types.SimpleNamespace(UPLOAD_VIA_BROWSER="auto", UPLOAD_CDP_ENDPOINT="http://10.6.6.4:3000"),
                proxy,
            )
        )
        self.assertTrue(
            _browser_upload_enabled(
                types.SimpleNamespace(UPLOAD_VIA_BROWSER="always", UPLOAD_CDP_ENDPOINT="http://10.6.6.4:3000"),
                proxy,
            )
        )
        # 没有代理实例（旧 app.state）→ 关闭
        self.assertFalse(
            _browser_upload_enabled(
                types.SimpleNamespace(UPLOAD_VIA_BROWSER="auto", UPLOAD_CDP_ENDPOINT="http://10.6.6.4:3000"),
                None,
            )
        )
        # endpoint 没配 → 关闭
        self.assertFalse(
            _browser_upload_enabled(
                types.SimpleNamespace(UPLOAD_VIA_BROWSER="auto", UPLOAD_CDP_ENDPOINT="   "),
                proxy,
            )
        )
        # 显式 never → 关闭
        self.assertFalse(
            _browser_upload_enabled(
                types.SimpleNamespace(UPLOAD_VIA_BROWSER="never", UPLOAD_CDP_ENDPOINT="http://10.6.6.4:3000"),
                proxy,
            )
        )


if __name__ == "__main__":
    unittest.main()
