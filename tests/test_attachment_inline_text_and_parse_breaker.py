"""纯文本附件本地注入 + 文档 parse 熔断 + 失败兜底全文注入。

覆盖三件事（对应上游 `/api/v2/files/parse` 被阿里云 WAF 拦死之后的三条止血）：

1. 纯文本类附件（txt/md/csv/json/...）由网关本地解码后注入 prompt，**零上游请求**；
2. `files/parse` 熔断：撞 WAF / 连续失败后，冷却期内直接跳过「上传 + parse」；
3. 生成文件上传失败时的兜底注入**不截断**（5 万字符原样进去）。
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.runtime.attachment_types import NormalizedAttachment
from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import prepare_context_attachments
from backend.services.document_parse_breaker import (
    DocumentParseBreaker,
    get_document_parse_breaker,
    reset_document_parse_breaker,
)
from backend.services.token_calc import count_tokens
from backend.services.upstream_file_uploader import (
    UpstreamFileUploader,
    UpstreamWafChallengeError,
)
from backend.toolcore.context_offload import ContextOffloader

WAF_HTML = (
    "<!DOCTYPE html><html><body><script>window.__aliyun_waf_aa=1;</script>"
    "<div id='aliyunCaptcha'></div><script>var sceneId='19x5u7lo';</script></body></html>"
)


def _settings(**overrides):
    base = dict(
        CONTEXT_INLINE_MAX_CHARS=4000,
        CONTEXT_FORCE_FILE_MAX_CHARS=10000,
        CONTEXT_ATTACHMENT_TTL_SECONDS=600,
        INLINE_ATTACHMENT_ENABLED=True,
        INLINE_ATTACHMENT_MAX_BYTES=10 * 1024 * 1024,
        ATTACHMENT_INLINE_TEXT_ENABLED=True,
        ATTACHMENT_INLINE_TEXT_MAX_BYTES=131072,
        DOCUMENT_PARSE_MAX_FAILURES=2,
        DOCUMENT_PARSE_COOLDOWN_SECONDS=600,
        CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _RecordingUploader:
    """记录上传调用；可注入异常模拟 WAF / 网络失败。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []
        self.exit_pool = None

    async def upload_local_file(self, _acc, local_meta):
        self.calls.append(local_meta["filename"])
        if self.error is not None:
            raise self.error
        return {"remote_ref": {"file_id": f"remote-{local_meta['filename']}", "filename": local_meta["filename"]}}

    async def upload_local_file_direct(self, _acc, local_meta):
        return await self.upload_local_file(_acc, local_meta)


_ACCOUNT = SimpleNamespace(email="bot@example.com", token="tok")


def _build_app(*, settings=None, uploader=None, account=_ACCOUNT):
    settings = settings or _settings()
    uploader = uploader if uploader is not None else _RecordingUploader()
    app = SimpleNamespace(
        state=SimpleNamespace(
            context_offloader=ContextOffloader(settings),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=account),
                acquire_wait_preferred=AsyncMock(return_value=account),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(
                save_text=AsyncMock(return_value={
                    "id": "ctx",
                    "path": "/tmp/ctx.txt",
                    "filename": "ctx.txt",
                    "content_type": "text/plain",
                    "sha256": "sha-ctx",
                    "created_at": 1,
                }),
                delete_path=AsyncMock(),
            ),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=uploader,
            browser_upload_proxy=None,
        )
    )
    return app, uploader


def _attachment(filename: str, content_type: str, raw: bytes, *, file_id: str = "local-1"):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
    handle.write(raw)
    handle.close()
    attachment = NormalizedAttachment(
        file_id=file_id,
        filename=filename,
        content_type=content_type,
        source="manual-upload",
        local_path=handle.name,
        sha256="sha-1",
        purpose="user-upload",
    )
    return attachment, handle.name


def _payload(text: str = "summarize this file") -> dict:
    return {"model": "qwen3.8-max", "messages": [{"role": "user", "content": text}]}


class InlineTextAttachmentTests(unittest.IsolatedAsyncioTestCase):
    """纯文本类附件：网关本地解码注入，一次上游请求都不发。"""

    def setUp(self) -> None:
        reset_document_parse_breaker()

    async def _run(self, attachments, *, settings=None, payload=None):
        app, uploader = _build_app(settings=settings, uploader=_RecordingUploader())
        result = await prepare_context_attachments(
            app=app,
            payload=payload or _payload(),
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:inline-text",
            existing_attachments=list(attachments),
        )
        return result, app, uploader

    async def test_txt_attachment_is_inlined_with_zero_upstream_calls(self) -> None:
        body = "region,amount\nhangzhou,42\nshanghai,7"
        attachment, path = _attachment("report.csv", "text/csv", body.encode("utf-8"))
        try:
            result, app, uploader = await self._run([attachment])
        finally:
            Path(path).unlink(missing_ok=True)

        content = result["payload"]["messages"][-1]["content"]
        self.assertIn("[附件: report.csv]", content)
        self.assertIn(body, content)
        # 零上游调用：不取 STS、不 PUT、不 parse，连账号都不挑。
        self.assertEqual(uploader.calls, [])
        app.state.account_pool.acquire_wait.assert_not_awaited()
        app.state.account_pool.acquire_wait_preferred.assert_not_awaited()
        self.assertEqual(result["upstream_files"], [])
        self.assertIsNone(result["bound_account"])
        self.assertFalse(result["attachment_fallback"])
        # token 计数：注入块按 count_tokens 计入。
        self.assertGreater(result["context_attachment_tokens"], 0)
        self.assertEqual(
            result["context_attachment_tokens"],
            count_tokens("[附件: report.csv]\n" + body),
        )

    async def test_markdown_and_json_attachments_are_inlined(self) -> None:
        md, md_path = _attachment("notes.md", "text/markdown", "# 标题\n正文".encode("utf-8"), file_id="a")
        js, js_path = _attachment("cfg.json", "application/json", '{"k": "v"}'.encode("utf-8"), file_id="b")
        try:
            result, _app, uploader = await self._run([md, js])
        finally:
            Path(md_path).unlink(missing_ok=True)
            Path(js_path).unlink(missing_ok=True)

        content = result["payload"]["messages"][-1]["content"]
        self.assertIn("[附件: notes.md]", content)
        self.assertIn("# 标题", content)
        self.assertIn("[附件: cfg.json]", content)
        self.assertIn('{"k": "v"}', content)
        self.assertEqual(uploader.calls, [])

    async def test_extension_only_attachment_with_octet_stream_is_inlined(self) -> None:
        """content_type 给成 octet-stream 时按扩展名兜底。"""
        attachment, path = _attachment("run.log", "application/octet-stream", b"line-1\nline-2\n")
        try:
            result, _app, uploader = await self._run([attachment])
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertIn("line-1", result["payload"]["messages"][-1]["content"])
        self.assertEqual(uploader.calls, [])

    async def test_inline_text_is_appended_to_current_user_message(self) -> None:
        """注入块不能变成新的 user 消息（否则会被当成 CURRENT TASK）。"""
        attachment, path = _attachment("a.txt", "text/plain", b"payload body")
        try:
            result, _app, _uploader = await self._run([attachment])
        finally:
            Path(path).unlink(missing_ok=True)

        messages = result["payload"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith("summarize this file"))

    async def test_oversized_text_attachment_is_truncated_with_notice(self) -> None:
        settings = _settings(ATTACHMENT_INLINE_TEXT_MAX_BYTES=64)
        body = "x" * 5000
        attachment, path = _attachment("big.txt", "text/plain", body.encode("utf-8"))
        try:
            result, _app, uploader = await self._run([attachment], settings=settings)
        finally:
            Path(path).unlink(missing_ok=True)

        content = result["payload"]["messages"][-1]["content"]
        self.assertIn("…（已截断，原文件 5 KB）", content)
        # 正文只保留前 64 字节，没有把 5000 字符全塞进去。
        self.assertLess(content.count("x"), 200)
        self.assertEqual(uploader.calls, [])

    async def test_binary_payload_with_text_content_type_falls_back_to_upload(self) -> None:
        """前 4KB 有 NUL 字节 → 不当文本注入，回落到原上传路径。"""
        attachment, path = _attachment("fake.txt", "text/plain", b"abc\x00\x01\x02def")
        try:
            result, _app, uploader = await self._run(
                [attachment],
                payload=_payload(),
            )
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertEqual(uploader.calls, ["fake.txt"])
        self.assertNotIn("[附件: fake.txt]", str(result["payload"]["messages"]))
        self.assertEqual(result["upstream_files"][0]["file_id"], "remote-fake.txt")

    async def test_disabled_switch_keeps_upload_path(self) -> None:
        settings = _settings(ATTACHMENT_INLINE_TEXT_ENABLED=False)
        attachment, path = _attachment("a.txt", "text/plain", b"hello")
        try:
            _result, _app, uploader = await self._run([attachment], settings=settings)
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertEqual(uploader.calls, ["a.txt"])


class ParseCooldownGateTests(unittest.IsolatedAsyncioTestCase):
    """熔断期间：文档类附件不再尝试「上传 + parse」。"""

    def setUp(self) -> None:
        reset_document_parse_breaker()

    def tearDown(self) -> None:
        reset_document_parse_breaker()

    async def test_document_attachment_skips_upload_while_cooldown_active(self) -> None:
        settings = _settings()
        breaker = get_document_parse_breaker(settings)
        breaker.record_failure("waf", immediate=True)
        self.assertTrue(breaker.blocked())
        self.assertGreater(breaker.remaining_seconds(), 0)

        attachment, path = _attachment("report.pdf", "application/pdf", b"%PDF-1.4 fake")
        try:
            app, uploader = _build_app(settings=settings, uploader=_RecordingUploader())
            with self.assertLogs("qwen2api.context_attachment_manager", level="INFO") as logs:
                result = await prepare_context_attachments(
                    app=app,
                    payload=_payload(),
                    surface="openai",
                    auth_token="tok",
                    client_profile=OPENCLAW_OPENAI_PROFILE,
                    session_key="openai:cooldown",
                    existing_attachments=[attachment],
                )
        finally:
            Path(path).unlink(missing_ok=True)

        self.assertEqual(uploader.calls, [])
        app.state.account_pool.acquire_wait.assert_not_awaited()
        self.assertTrue(result["attachment_fallback"])
        messages = [record.getMessage() for record in logs.records]
        blocked_lines = [line for line in messages if "document parse blocked (cooldown)" in line]
        self.assertEqual(len(blocked_lines), 1, messages)
        self.assertIn("file=report.pdf", blocked_lines[0])
        # 附件名仍进入兜底消息（不静默丢附件）。
        joined = "\n".join(str(message.get("content", "")) for message in result["payload"]["messages"])
        self.assertIn("report.pdf", joined)

    async def test_generated_context_only_request_skips_upload_while_cooldown_active(self) -> None:
        settings = _settings(CONTEXT_INLINE_MAX_CHARS=200, CONTEXT_FORCE_FILE_MAX_CHARS=120)
        get_document_parse_breaker(settings).record_failure("waf", immediate=True)

        payload = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "history line\n" * 200}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {}}}],
        }
        app, uploader = _build_app(settings=settings, uploader=_RecordingUploader())
        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:cooldown-generated",
        )

        self.assertEqual(uploader.calls, [])
        self.assertTrue(result["attachment_fallback"])
        self.assertEqual(result["context_mode"], "inline")

    async def test_cooldown_expired_allows_one_probe_then_reopens(self) -> None:
        now = [1000.0]
        breaker = DocumentParseBreaker(
            max_failures=2,
            cooldown_seconds=600,
            clock=lambda: now[0],
        )
        breaker.record_failure("waf", immediate=True)
        self.assertTrue(breaker.blocked())
        now[0] += 601
        self.assertFalse(breaker.blocked())  # 冷却到期 → 放行探针
        self.assertTrue(breaker.is_half_open)
        breaker.record_failure("waf")
        self.assertTrue(breaker.blocked())  # 探针失败 → 立刻重新熔断
        now[0] += 601
        self.assertFalse(breaker.blocked())
        breaker.record_success()
        self.assertFalse(breaker.is_half_open)
        self.assertEqual(breaker.failure_count, 0)

    async def test_consecutive_failures_open_after_threshold(self) -> None:
        now = [0.0]
        breaker = DocumentParseBreaker(max_failures=2, cooldown_seconds=600, clock=lambda: now[0])
        self.assertFalse(breaker.record_failure("RuntimeError"))
        self.assertFalse(breaker.blocked())
        self.assertTrue(breaker.record_failure("RuntimeError"))
        self.assertTrue(breaker.blocked())

    async def test_parse_waf_opens_breaker_via_uploader(self) -> None:
        """`put_bytes` 的 parse 阶段撞 WAF → 立刻熔断（不等阈值）。"""

        class _WafClient:
            def __init__(self) -> None:
                self.requests: list[str] = []

            async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
                del method, token, body, timeout, account
                self.requests.append(path)
                return {"status": 200, "body": WAF_HTML}

        settings = _settings()
        client = _WafClient()
        uploader = UpstreamFileUploader(client, settings)
        account = SimpleNamespace(email="a@example.com", token="tok")

        with self.assertRaises(UpstreamWafChallengeError):
            await uploader._parse_document(account, "file_1")

        self.assertEqual(client.requests, ["/api/v2/files/parse"])
        self.assertTrue(get_document_parse_breaker(settings).blocked())

    async def test_parse_failure_twice_opens_breaker_via_uploader(self) -> None:
        class _FailingClient:
            async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
                del method, token, body, timeout, account
                return {"status": 500, "body": "{}"}

        settings = _settings()
        uploader = UpstreamFileUploader(_FailingClient(), settings)
        account = SimpleNamespace(email="a@example.com", token="tok")
        breaker = get_document_parse_breaker(settings)

        with self.assertRaises(RuntimeError):
            await uploader._parse_document(account, "file_1")
        self.assertFalse(breaker.blocked())
        with self.assertRaises(RuntimeError):
            await uploader._parse_document(account, "file_2")
        self.assertTrue(breaker.blocked())

    async def test_parse_success_closes_breaker(self) -> None:
        class _OkClient:
            async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
                del method, token, body, timeout, account
                if path.endswith("/status"):
                    return {"status": 200, "body": '{"data": [{"status": "success"}]}'}
                return {"status": 200, "body": "{}"}

        settings = _settings()
        breaker = get_document_parse_breaker(settings)
        breaker.record_failure("waf", immediate=True)
        uploader = UpstreamFileUploader(_OkClient(), settings)
        account = SimpleNamespace(email="a@example.com", token="tok")

        status = await uploader._parse_document(account, "file_1")

        self.assertEqual(status, "success")
        self.assertFalse(breaker.blocked())
        self.assertEqual(breaker.failure_count, 0)


class FullHistoryFallbackTests(unittest.IsolatedAsyncioTestCase):
    """生成文件上传失败 → 兜底注入必须保留**完整**历史正文（不截断）。"""

    def setUp(self) -> None:
        reset_document_parse_breaker()

    def tearDown(self) -> None:
        reset_document_parse_breaker()

    async def test_fallback_injects_full_50k_history(self) -> None:
        settings = _settings()
        history_line = "0123456789abcdef" * 4  # 64 字符一行
        history = "\n".join(history_line for _ in range(781))  # ≈50_000 字符
        self.assertGreaterEqual(len(history), 50_000)
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": history},
                {"role": "assistant", "content": "TAIL_MARKER_KEEP_ME"},
                {"role": "user", "content": "latest task"},
            ],
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {}}}],
        }
        app, uploader = _build_app(
            settings=settings,
            uploader=_RecordingUploader(error=UpstreamWafChallengeError("waf_challenge: /api/v2/files/parse")),
            account=SimpleNamespace(email="bot@example.com"),
        )

        with self.assertLogs("qwen2api.context_attachment_manager", level="INFO") as logs:
            result = await prepare_context_attachments(
                app=app,
                payload=payload,
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:full-history",
            )

        self.assertTrue(result["attachment_fallback"])
        self.assertEqual(result["context_mode"], "inline")
        joined = "\n".join(str(message.get("content", "")) for message in result["payload"]["messages"])
        self.assertIn("TAIL_MARKER_KEEP_ME", joined)
        self.assertIn("latest task", joined)
        # 完整历史：50k 字符一行不少（兜底注入不截断）。
        self.assertGreaterEqual(joined.count(history_line), 781)
        fallback_lines = [
            record.getMessage() for record in logs.records if "inline fallback chars=" in record.getMessage()
        ]
        self.assertEqual(len(fallback_lines), 1)
        chars = int(fallback_lines[0].split("inline fallback chars=")[1].split()[0])
        self.assertGreaterEqual(chars, 50_000)


if __name__ == "__main__":
    unittest.main()
