"""出口池（`UPLOAD_EXIT_POOL`）单测：不打真网络，全部用假代理层。

覆盖：
1. 出口 id 解析 / 配置解析 / 开关模式（不泄漏凭据）；
2. `ExitPool` 轮询、冷却跳过、冷却到期复位、成功清零、快照、探测；
3. `UpstreamFileUploader` 取 STS 的出口池路径：第一个出口 WAF → 换第二个成功、
   全部失败 → `UpstreamWafChallengeError`、池关闭 / 为空 → 走原直连路径；
4. `context_attachment_manager` 的降级顺序：出口池 → 浏览器 → 直连 HTTP。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

if "oss2" not in sys.modules:
    fake_oss2 = types.ModuleType("oss2")

    class StsAuth:
        pass

    class Bucket:
        pass

    fake_oss2.StsAuth = StsAuth
    fake_oss2.Bucket = Bucket
    sys.modules["oss2"] = fake_oss2

from backend.core.account_pool import Account
from backend.runtime.attachment_types import NormalizedAttachment
from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import prepare_context_attachments
from backend.services.exit_pool import (
    STATE_COOLING,
    STATE_READY,
    ExitPool,
    exit_id_from_url,
    parse_exit_list,
    resolve_exit_mode,
)
from backend.services.upstream_file_uploader import (
    UpstreamFileUploader,
    UpstreamWafChallengeError,
)
from backend.toolcore.context_offload import ContextOffloader

G_EXIT = "socks5h://Default.g@10.6.6.114:2260"
J_EXIT = "socks5h://Default.j@10.6.6.114:2260"
C_EXIT = "socks5h://Default.c@10.6.6.114:2260"

WAF_HTML = (
    "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body>"
    "<script>window.__aliyun_waf_aa=1;</script>"
    "<div id=\"aliyunCaptcha\"></div>"
    "<script>var sceneId='19x5u7lo';</script></body></html>"
)

STS_DATA = {
    "file_id": "file_ok",
    "file_path": "user/object.bin",
    "bucketname": "qwen-webui-prod",
    "endpoint": "qwen-webui-prod.oss-accelerate.aliyuncs.com",
    "region": "oss-cn-hangzhou",
    "access_key_id": "ak",
    "access_key_secret": "sk",
    "security_token": "st",
    "file_url": "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/user/object.bin?x-oss-expires=300&x-oss-signature=sig",
}


def _sts_ok_body() -> str:
    return json.dumps({"success": True, "data": STS_DATA})


# ----------------------------------------------------------------------
# 假 curl_cffi 会话层（"假代理层"）
# ----------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeSession:
    """模拟一条「经该出口发请求」的会话；行为由 script 决定。"""

    def __init__(self, proxy_url: str, script, calls: list[str]) -> None:
        self.proxy_url = proxy_url
        self.script = script
        self.calls = calls
        self.closed = False

    async def post(self, url, headers=None, json=None, timeout=None):
        del url, headers, json, timeout
        self.calls.append(self.proxy_url)
        outcome = self.script(self.proxy_url)
        if isinstance(outcome, Exception):
            raise outcome
        status, text = outcome
        return _FakeResponse(status, text)

    async def close(self) -> None:
        self.closed = True


class _FakeExitNetwork:
    """按出口 URL 编排响应：`{exit_url: (status, body) | Exception}`。"""

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = dict(responses)
        self.calls: list[str] = []
        self.sessions: list[_FakeSession] = []

    def script(self, proxy_url: str):
        if proxy_url not in self.responses:
            raise AssertionError(f"unexpected exit: {proxy_url}")
        return self.responses[proxy_url]

    async def factory(self, proxy_url: str) -> _FakeSession:
        session = _FakeSession(proxy_url, self.script, self.calls)
        self.sessions.append(session)
        return session


class _FakeClient:
    """假 `QwenClient`：直连路径只走 `_request_json`。"""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses = dict(responses or {})
        self.requests: list[str] = []

    async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
        del method, token, body, timeout, account
        self.requests.append(path)
        if path not in self.responses:
            raise AssertionError(f"unexpected path: {path}")
        return self.responses[path]

    def _build_headers(self, *, account=None, token=None, **_kwargs):
        del account
        headers = {"Accept": "application/json, text/plain, */*"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS": 1,
        "UPLOAD_EXIT_POOL": "",
        "UPLOAD_VIA_EXIT": "auto",
        "UPLOAD_EXIT_COOLDOWN_SECONDS": 600,
        "UPLOAD_EXIT_MAX_FAILURES": 2,
        "UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS": 10,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _account() -> Account:
    return Account(email="alice@example.com", token="tok")


class ExitIdTests(unittest.TestCase):
    def test_exit_id_is_suffix_only(self) -> None:
        self.assertEqual(exit_id_from_url("socks5h://Default.g@10.6.6.114:2260"), "Default.g")
        # 带密码时也只取用户名部分：日志里绝不出现凭据 / host。
        self.assertEqual(exit_id_from_url("socks5h://Default.j:secret@10.6.6.114:2260"), "Default.j")
        # 无凭据的普通代理退回 host:port
        self.assertEqual(exit_id_from_url("socks5h://10.6.6.114:2260"), "10.6.6.114:2260")
        self.assertEqual(exit_id_from_url(""), "")

    def test_parse_exit_list_dedupes_and_trims(self) -> None:
        parsed = parse_exit_list(f" {G_EXIT} , {J_EXIT} ,, {G_EXIT} ")
        self.assertEqual(parsed, [G_EXIT, J_EXIT])

    def test_resolve_exit_mode_falls_back_to_auto(self) -> None:
        self.assertEqual(resolve_exit_mode(_settings(UPLOAD_VIA_EXIT="NEVER")), "never")
        self.assertEqual(resolve_exit_mode(_settings(UPLOAD_VIA_EXIT="oops")), "auto")
        self.assertEqual(resolve_exit_mode(SimpleNamespace()), "auto")


class ExitPoolSelectionTests(unittest.TestCase):
    def test_round_robin_skips_nothing_when_healthy(self) -> None:
        pool = ExitPool([G_EXIT, J_EXIT, C_EXIT])
        self.assertEqual(
            [pool.pick(), pool.pick(), pool.pick(), pool.pick()],
            [G_EXIT, J_EXIT, C_EXIT, G_EXIT],
        )

    def test_cooling_exit_is_skipped(self) -> None:
        pool = ExitPool([G_EXIT, J_EXIT], max_failures=1, cooldown_seconds=600)
        pool.mark_failure("Default.g", "waf")
        # 冷却中的 g 被跳过：连续两次都只能拿到 j
        self.assertEqual(pool.pick(), J_EXIT)
        self.assertEqual(pool.pick(), J_EXIT)
        snapshot = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["state"], STATE_COOLING)
        self.assertGreater(snapshot["Default.g"]["cooldown_remaining"], 0)
        self.assertEqual(snapshot["Default.j"]["state"], STATE_READY)

    def test_exclude_skips_exit_within_one_round(self) -> None:
        pool = ExitPool([G_EXIT, J_EXIT])
        self.assertIsNone(pool.pick(exclude={"Default.g", "Default.j"}))

    def test_cooldown_expiry_resets_failures(self) -> None:
        now = [1000.0]
        pool = ExitPool(
            [G_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            clock=lambda: now[0],
        )
        pool.mark_failure("Default.g", "waf")
        self.assertIsNone(pool.pick())
        self.assertEqual(pool.snapshot()[0]["failures"], 1)
        now[0] += 601
        # 冷却到期：重新可用，且失败计数清零（给满新一轮机会）
        self.assertEqual(pool.pick(), G_EXIT)
        self.assertEqual(pool.snapshot()[0]["failures"], 0)
        self.assertEqual(pool.snapshot()[0]["state"], STATE_READY)

    def test_mark_success_clears_failures_and_cooldown(self) -> None:
        pool = ExitPool([G_EXIT], max_failures=1, cooldown_seconds=600)
        pool.mark_failure("Default.g", "waf")
        self.assertIsNone(pool.pick())
        pool.mark_success("Default.g")
        self.assertEqual(pool.pick(), G_EXIT)
        row = pool.snapshot()[0]
        self.assertEqual((row["failures"], row["successes"]), (0, 1))

    def test_repeated_failures_below_threshold_keep_exit_usable(self) -> None:
        pool = ExitPool([G_EXIT], max_failures=2, cooldown_seconds=600)
        pool.mark_failure("Default.g", "connect_failed:OSError")
        self.assertEqual(pool.pick(), G_EXIT)
        pool.mark_failure("Default.g", "waf")
        self.assertIsNone(pool.pick())

    def test_from_settings_respects_never_and_empty_pool(self) -> None:
        self.assertIsNone(ExitPool.from_settings(_settings(UPLOAD_EXIT_POOL="")))
        self.assertIsNone(ExitPool.from_settings(_settings(UPLOAD_EXIT_POOL=G_EXIT, UPLOAD_VIA_EXIT="never")))
        pool = ExitPool.from_settings(_settings(UPLOAD_EXIT_POOL=f"{G_EXIT},{J_EXIT}"))
        self.assertIsNotNone(pool)
        self.assertTrue(pool.enabled)
        self.assertEqual(pool.exit_ids, ["Default.g", "Default.j"])


class ExitPoolProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_marks_success_and_failure(self) -> None:
        seen: list[tuple[str, float]] = []

        async def fake_fetch(proxy_url: str, timeout: float) -> bool:
            seen.append((proxy_url, timeout))
            return proxy_url == G_EXIT

        pool = ExitPool([G_EXIT, J_EXIT], max_failures=1, probe_fetch=fake_fetch)
        self.assertTrue(await pool.probe("Default.g"))
        self.assertFalse(await pool.probe("Default.j"))
        self.assertEqual([item[0] for item in seen], [G_EXIT, J_EXIT])
        snapshot = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["successes"], 1)
        self.assertEqual(snapshot["Default.j"]["state"], STATE_COOLING)

    async def test_probe_exception_is_folded_into_failure(self) -> None:
        async def explode(_proxy_url: str, _timeout: float) -> bool:
            raise OSError("socks5 connect failed")

        pool = ExitPool([G_EXIT], max_failures=1, probe_fetch=explode)
        self.assertFalse(await pool.probe("Default.g"))
        self.assertEqual(pool.snapshot()[0]["state"], STATE_COOLING)

    async def test_probe_unknown_exit_is_false(self) -> None:
        pool = ExitPool([G_EXIT])
        self.assertFalse(await pool.probe("Default.zzz"))


class StsViaExitPoolTests(unittest.IsolatedAsyncioTestCase):
    """`getstsToken` 走出口池的核心行为（顺序 / 记账 / 全失败）。"""

    def _uploader(self, network: _FakeExitNetwork, pool: ExitPool, **settings_overrides):
        return UpstreamFileUploader(
            _FakeClient(),
            _settings(**settings_overrides),
            exit_pool=pool,
            exit_session_factory=network.factory,
        )

    async def test_first_exit_waf_retries_next_exit(self) -> None:
        network = _FakeExitNetwork({
            G_EXIT: (200, WAF_HTML),
            J_EXIT: (200, _sts_ok_body()),
        })
        pool = ExitPool([G_EXIT, J_EXIT], max_failures=5)
        uploader = self._uploader(network, pool)

        sts = await uploader.request_sts(_account(), "clip.mp4", b"x" * 32, "video/mp4")

        self.assertEqual(sts["file_id"], "file_ok")
        # 顺序：先 g（被 WAF 拦）→ 再 j（成功）
        self.assertEqual(network.calls, [G_EXIT, J_EXIT])
        snapshot = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["failures"], 1)
        self.assertEqual(snapshot["Default.g"]["last_error"], "waf")
        self.assertEqual(snapshot["Default.g"]["successes"], 0)
        self.assertEqual(snapshot["Default.j"]["successes"], 1)
        self.assertEqual(snapshot["Default.j"]["failures"], 0)
        # 会话必须被关闭：出口不稳，长连接会泄漏。
        self.assertTrue(all(session.closed for session in network.sessions))

    async def test_connect_failure_marks_failure_and_moves_on(self) -> None:
        network = _FakeExitNetwork({
            G_EXIT: OSError("Failed to connect to 10.6.6.114 port 2260"),
            J_EXIT: (200, _sts_ok_body()),
        })
        pool = ExitPool([G_EXIT, J_EXIT], max_failures=5)
        uploader = self._uploader(network, pool)

        sts = await uploader.request_sts(_account(), "clip.mp4", b"x" * 8, "video/mp4")

        self.assertEqual(sts["file_id"], "file_ok")
        self.assertEqual(network.calls, [G_EXIT, J_EXIT])
        row = {item["id"]: item for item in pool.snapshot()}["Default.g"]
        self.assertEqual(row["failures"], 1)
        self.assertTrue(row["last_error"].startswith("connect_failed:"))

    async def test_all_exits_failing_raises_waf_challenge(self) -> None:
        network = _FakeExitNetwork({
            G_EXIT: (200, WAF_HTML),
            J_EXIT: (403, "forbidden"),
        })
        pool = ExitPool([G_EXIT, J_EXIT], max_failures=1, cooldown_seconds=600)
        uploader = self._uploader(network, pool)

        with self.assertRaises(UpstreamWafChallengeError):
            await uploader.request_sts(_account(), "doc.pdf", b"x" * 8, "application/pdf")

        # 一轮遍历：每个出口只试一次，不无限重试
        self.assertEqual(network.calls, [G_EXIT, J_EXIT])
        snapshot = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["state"], STATE_COOLING)
        self.assertEqual(snapshot["Default.j"]["state"], STATE_COOLING)

    async def test_cooldown_exit_not_retried_on_next_upload(self) -> None:
        network = _FakeExitNetwork({
            G_EXIT: (200, WAF_HTML),
            J_EXIT: (200, _sts_ok_body()),
        })
        pool = ExitPool([G_EXIT, J_EXIT], max_failures=1, cooldown_seconds=600)
        uploader = self._uploader(network, pool)

        await uploader.request_sts(_account(), "a.bin", b"x" * 4, "application/octet-stream")
        self.assertEqual(network.calls, [G_EXIT, J_EXIT])

        network.calls.clear()
        await uploader.request_sts(_account(), "b.bin", b"x" * 4, "application/octet-stream")
        # g 进入冷却：第二次上传直接命中 j，不再浪费一次 WAF 尝试
        self.assertEqual(network.calls, [J_EXIT])

    async def test_rate_limited_via_exit_keeps_exit_healthy(self) -> None:
        rate_limited = json.dumps({"success": True, "data": {"code": "RateLimited", "num": 2}})
        network = _FakeExitNetwork({G_EXIT: (200, rate_limited)})
        pool = ExitPool([G_EXIT], max_failures=1)
        uploader = self._uploader(network, pool)

        from backend.services.upstream_file_uploader import FileUploadRateLimitedError

        with self.assertRaises(FileUploadRateLimitedError) as ctx:
            await uploader.request_sts(_account(), "a.bin", b"x" * 4, "application/octet-stream")
        self.assertEqual(ctx.exception.retry_after_seconds, 7200.0)
        row = pool.snapshot()[0]
        self.assertEqual((row["state"], row["successes"], row["failures"]), (STATE_READY, 1, 0))

    async def test_pool_disabled_falls_back_to_direct_request(self) -> None:
        """`UPLOAD_VIA_EXIT=never` / 池为空：走原直连路径，一行代理都不碰。"""
        client = _FakeClient({
            "/api/v2/files/getstsToken": {"status": 200, "body": _sts_ok_body()},
        })

        async def forbidden_factory(_proxy_url):
            raise AssertionError("pool disabled: must not touch any proxy")

        uploader = UpstreamFileUploader(
            client,
            _settings(UPLOAD_EXIT_POOL=G_EXIT, UPLOAD_VIA_EXIT="never"),
            exit_session_factory=forbidden_factory,
        )
        self.assertIsNone(uploader.exit_pool)
        sts = await uploader.request_sts(_account(), "a.bin", b"x" * 4, "application/octet-stream")
        self.assertEqual(sts["file_id"], "file_ok")
        self.assertEqual(client.requests, ["/api/v2/files/getstsToken"])

    async def test_direct_upload_skips_pool_even_when_enabled(self) -> None:
        """降级最后一档 `upload_local_file_direct` 必须真的不挂代理。"""
        client = _FakeClient({
            "/api/v2/files/getstsToken": {"status": 200, "body": _sts_ok_body()},
        })

        async def forbidden_factory(_proxy_url):
            raise AssertionError("direct upload must not touch any proxy")

        pool = ExitPool([G_EXIT])
        uploader = UpstreamFileUploader(
            client,
            _settings(UPLOAD_EXIT_POOL=G_EXIT),
            exit_pool=pool,
            exit_session_factory=forbidden_factory,
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
            handle.write(b"x" * 16)
            local_path = handle.name
        try:
            with patch(
                "backend.services.upstream_file_uploader.oss2.StsAuth", return_value=object()
            ), patch(
                "backend.services.upstream_file_uploader.oss2.Bucket", _FakeBucket
            ):
                result = await uploader.upload_local_file_direct(
                    _account(),
                    {"filename": "notes.txt", "path": local_path, "content_type": "video/mp4"},
                )
        finally:
            Path(local_path).unlink(missing_ok=True)
        self.assertEqual(result["remote_file_id"], "file_ok")
        self.assertEqual(client.requests, ["/api/v2/files/getstsToken"])


class _FakeBucket:
    def __init__(self, auth, endpoint, bucketname, region=None):
        del auth, endpoint, bucketname, region

    def put_object(self, key, raw, headers=None):
        del key, raw, headers
        return types.SimpleNamespace(status=200)


# ----------------------------------------------------------------------
# 降级顺序：出口池 → 浏览器 → 直连 HTTP
# ----------------------------------------------------------------------
class _OrderedBrowserProxy:
    """浏览器代理替身：调用即抛「不可用」，并记录自己被调用的顺序。"""

    def __init__(self, order: list[str], *, error: Exception) -> None:
        self.order = order
        self.error = error
        self.fetch_calls: list[str] = []

    async def fetch_sts(self, account, filename, content_type, size):
        del account, content_type, size
        self.order.append("browser")
        self.fetch_calls.append(filename)
        raise self.error

    def note_hard_timeout(self, email: str) -> None:
        del email


class DegradeOrderTests(unittest.IsolatedAsyncioTestCase):
    """`context_attachment_manager` 的上传降级顺序。"""

    def _app(self, *, uploader, browser_proxy, settings_overrides: dict | None = None):
        settings_kwargs = {
            "CONTEXT_INLINE_MAX_CHARS": 80,
            "CONTEXT_FORCE_FILE_MAX_CHARS": 160,
            "CONTEXT_ATTACHMENT_TTL_SECONDS": 600,
            "INLINE_ATTACHMENT_ENABLED": True,
            "INLINE_ATTACHMENT_MAX_BYTES": 10 * 1024 * 1024,
            "UPLOAD_VIA_BROWSER": "auto",
            "UPLOAD_CDP_ENDPOINT": "http://10.6.6.4:3000",
            "UPLOAD_BROWSER_MODE": "auto",
            "UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS": 45,
        }
        settings_kwargs.update(settings_overrides or {})
        return SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(**settings_kwargs)),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(save_text=AsyncMock(), delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=uploader,
            browser_upload_proxy=browser_proxy,
        ))

    def _attachment(self) -> tuple[NormalizedAttachment, str]:
        """文档附件：图片 / 音频 / 视频默认内联，只有文档才走上传降级链。"""
        # .pdf：纯文本类附件（.txt）走网关本地解码注入，不经过上传降级链。
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        handle.write(b"user supplied document bytes")
        handle.close()
        attachment = NormalizedAttachment(
            file_id="local-doc",
            filename="notes.pdf",
            content_type="application/pdf",
            source="manual-upload",
            local_path=handle.name,
            sha256="sha-doc",
            purpose="user-upload",
        )
        return attachment, handle.name

    async def test_order_is_exit_pool_then_browser_then_direct(self) -> None:
        from backend.services.browser_upload_proxy import BrowserUploadUnavailable

        order: list[str] = []

        async def pool_upload(_acc, _local_meta):
            order.append("exit_pool")
            raise UpstreamWafChallengeError("waf_challenge: all upload exits failed")

        async def direct_upload(_acc, _local_meta):
            order.append("direct")
            return {"remote_ref": {"file_id": "direct-file", "filename": "notes.txt"}}

        uploader = SimpleNamespace(
            exit_pool=ExitPool([G_EXIT]),
            upload_local_file=pool_upload,
            upload_local_file_direct=direct_upload,
            put_bytes=AsyncMock(),
        )
        browser_proxy = _OrderedBrowserProxy(order, error=BrowserUploadUnavailable("no browser"))
        app = self._app(uploader=uploader, browser_proxy=browser_proxy)
        attachment, local_path = self._attachment()
        try:
            result = await prepare_context_attachments(
                app=app,
                payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "look"}]},
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:exit-order",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(order, ["exit_pool", "browser", "direct"])
        self.assertEqual(result["upstream_files"][0]["file_id"], "direct-file")
        uploader.put_bytes.assert_not_awaited()

    async def test_browser_never_reached_when_exit_pool_succeeds(self) -> None:
        order: list[str] = []

        async def pool_upload(_acc, _local_meta):
            order.append("exit_pool")
            return {"remote_ref": {"file_id": "pool-file", "filename": "notes.txt"}}

        async def direct_upload(_acc, _local_meta):
            order.append("direct")
            return {"remote_ref": {"file_id": "direct-file"}}

        uploader = SimpleNamespace(
            exit_pool=ExitPool([G_EXIT]),
            upload_local_file=pool_upload,
            upload_local_file_direct=direct_upload,
            put_bytes=AsyncMock(),
        )
        browser_proxy = _OrderedBrowserProxy(order, error=RuntimeError("should not be used"))
        app = self._app(uploader=uploader, browser_proxy=browser_proxy)
        attachment, local_path = self._attachment()
        try:
            result = await prepare_context_attachments(
                app=app,
                payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "look"}]},
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:exit-hit",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(order, ["exit_pool"])
        self.assertEqual(browser_proxy.fetch_calls, [])
        self.assertEqual(result["upstream_files"][0]["file_id"], "pool-file")

    async def test_pool_disabled_keeps_browser_then_direct_order(self) -> None:
        """池为空 / never：跳档到「浏览器 → 直连」，与历史行为一致。"""
        from backend.services.browser_upload_proxy import BrowserUploadUnavailable

        order: list[str] = []

        async def pool_upload(_acc, _local_meta):
            order.append("exit_pool")
            raise AssertionError("pool disabled: upload_local_file must not be used first")

        async def direct_upload(_acc, _local_meta):
            order.append("direct")
            return {"remote_ref": {"file_id": "direct-file"}}

        uploader = SimpleNamespace(
            upload_local_file=pool_upload,
            upload_local_file_direct=direct_upload,
        )
        browser_proxy = _OrderedBrowserProxy(order, error=BrowserUploadUnavailable("no browser"))
        app = self._app(uploader=uploader, browser_proxy=browser_proxy)
        attachment, local_path = self._attachment()
        try:
            result = await prepare_context_attachments(
                app=app,
                payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "look"}]},
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:pool-off",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(order, ["browser", "direct"])
        self.assertEqual(result["upstream_files"][0]["file_id"], "direct-file")


class ProbePayloadTests(unittest.TestCase):
    """`/admin/upload/probe` 的 `exit_pool` 字段。"""

    def test_probe_exposes_exit_pool_snapshot(self) -> None:
        from backend.api.probes import _exit_pool_probe

        pool = ExitPool([G_EXIT, J_EXIT], max_failures=1)
        pool.mark_failure("Default.g", "waf")
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(upload_exit_pool=pool)))

        payload = _exit_pool_probe(request)
        self.assertTrue(payload["enabled"])
        rows = {row["id"]: row for row in payload["exits"]}
        self.assertEqual(rows["Default.g"]["state"], STATE_COOLING)
        self.assertEqual(rows["Default.j"]["state"], STATE_READY)
        # 绝不泄漏代理凭据 / host
        self.assertNotIn("10.6.6.114", json.dumps(payload))

    def test_probe_without_pool(self) -> None:
        from backend.api.probes import _exit_pool_probe

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        self.assertEqual(_exit_pool_probe(request), {"enabled": False, "exits": []})


if __name__ == "__main__":
    unittest.main()
