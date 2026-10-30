"""上游确定性错误（风控软拦 200 + 裸 JSON）的识别与「不重试」行为测试。

背景：上游 AWSC 风控生效后，/api/v2/chat/completions 会返回 HTTP 200 +
一个非 SSE 的裸 JSON 错误信封。旧实现只解析 `data:` 行，导致 0 事件、
错误被静默吞掉，调用方干等 idle 超时并按瞬时错误重试，最终把账号池打成
disabled。本文件锁定修复后的行为。
"""

import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

from backend.core.account_pool import Account
from backend.upstream.qwen_executor import (
    DeterministicUpstreamError,
    QwenExecutor,
    _is_deterministic_upstream_error,
)
from backend.upstream.sse_consumer import parse_sse_chunk

# 上游风控软拦的真实响应体（非 SSE、无 data: 前缀、HTTP 200）
BARE_JSON_ENVELOPE = (
    '{"success":false,"request_id":"req-abc123",'
    '"data":{"code":"Bad_Request","details":"Internal error, please try again later."}}'
)


class BareJsonErrorParsingTests(unittest.TestCase):
    def test_parses_bare_json_envelope_as_error_event(self) -> None:
        with self.assertLogs("qwen2api.sse", level="WARNING") as logs:
            parsed = parse_sse_chunk(BARE_JSON_ENVELOPE)

        self.assertEqual(len(parsed), 1)
        event = parsed[0]
        self.assertEqual(event["type"], "error")
        self.assertEqual(event["phase"], "error")
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["code"], "Bad_Request")
        self.assertIn("Internal error", event["details"])
        self.assertEqual(event["response_id"], "req-abc123")
        # 运维一眼可见：日志里必须带 code 和截断预览
        joined = "\n".join(logs.output)
        self.assertIn("non-sse json error body", joined)
        self.assertIn("Bad_Request", joined)

    def test_parses_bare_json_with_surrounding_whitespace(self) -> None:
        parsed = parse_sse_chunk(f"\n\n  {BARE_JSON_ENVELOPE}\n")

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["code"], "Bad_Request")

    def test_parses_bare_json_with_top_level_error_object(self) -> None:
        chunk = '{"error":{"code":"invalid_argument","details":"bad payload"}}'

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["type"], "error")
        self.assertEqual(parsed[0]["code"], "invalid_argument")
        self.assertEqual(parsed[0]["details"], "bad payload")

    def test_ignores_bare_json_that_is_not_an_error(self) -> None:
        """success=true 的裸 JSON 不是错误体，不应产出错误事件。"""
        with self.assertNoLogs("qwen2api.sse", level="WARNING"):
            parsed = parse_sse_chunk('{"success":true,"data":{"id":"chat-1"}}')

        self.assertEqual(parsed, [])

    def test_ignores_bare_plain_text(self) -> None:
        with self.assertNoLogs("qwen2api.sse", level="WARNING"):
            parsed = parse_sse_chunk("upstream plain text response")

        self.assertEqual(parsed, [])

    def test_normal_sse_delta_is_unaffected(self) -> None:
        with self.assertNoLogs("qwen2api.sse", level="WARNING"):
            parsed = parse_sse_chunk('data: {"content": "hello"}\n\n')

        self.assertEqual(
            parsed,
            [{"type": "delta", "phase": "answer", "content": "hello", "status": "", "extra": {}}],
        )

    def test_data_line_takes_precedence_over_bare_json_detection(self) -> None:
        """含 data: 行时走原路径，绝不触发裸 JSON 分支（行为与改动前一致）。"""
        chunk = 'data: {"success": false, "msg": "blocked", "code": 400}\n\n'

        with self.assertLogs("qwen2api.sse", level="WARNING") as logs:
            parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed, [])
        self.assertIn("unparsed json event", "\n".join(logs.output))


class DeterministicErrorClassificationTests(unittest.TestCase):
    def test_deterministic_codes_are_recognized_case_insensitively(self) -> None:
        for code in ("Bad_Request", "bad_request", "UNAUTHORIZED", "Forbidden", "invalid_argument", "InvalidParameter"):
            with self.subTest(code=code):
                self.assertTrue(_is_deterministic_upstream_error(code, ""))

    def test_internal_error_details_is_recognized(self) -> None:
        self.assertTrue(_is_deterministic_upstream_error("", "Internal error, please try again later."))
        self.assertTrue(_is_deterministic_upstream_error("some_code", "internal error"))

    def test_transient_errors_are_not_deterministic(self) -> None:
        self.assertFalse(_is_deterministic_upstream_error("quota_limit", "目前服务访问量较大，请稍后再试。"))
        self.assertFalse(_is_deterministic_upstream_error("429", "Too Many Requests"))
        self.assertFalse(_is_deterministic_upstream_error("", ""))
        self.assertFalse(_is_deterministic_upstream_error(None, None))


class _FakeEngine:
    """最小 stream 传输桩：按序吐 chunk。

    hang=True 时吐完不结束（模拟风控软拦后上游既不出流也不关连接），用于验证
    「立即判错」而不是干等到 idle 超时。
    """

    def __init__(self, chunks: list[str], hang: bool = True) -> None:
        self._chunks = chunks
        self._hang = hang

    async def stream_chat_once(self, token, chat_id, payload, account=None):
        del token, chat_id, payload, account
        for chunk in self._chunks:
            yield {"status": "streamed", "chunk": chunk}
        if self._hang:
            await asyncio.Event().wait()


class _RotatingPool:
    """每次 acquire 都给一个新账号，便于断言「尝试了几次」。"""

    def __init__(self) -> None:
        self.acquired: list = []
        self.released: list = []
        self.invalidated: list = []
        self.rate_limited: list = []
        self.succeeded: list = []

    async def acquire_wait(self, timeout=60, exclude=None):
        del timeout, exclude
        account = Account(email=f"acc{len(self.acquired)}@example.com", token=f"tok-{len(self.acquired)}")
        self.acquired.append(account)
        return account

    def mark_success(self, acc) -> None:
        self.succeeded.append(acc)

    def mark_invalid(self, acc) -> None:
        self.invalidated.append(acc)

    def mark_rate_limited(self, acc) -> None:
        self.rate_limited.append(acc)

    def release(self, acc) -> None:
        self.released.append(acc)


class _FakeChatPool:
    """chat_id 池桩：总是命中，避免重试测试走到真实 create_chat（需要 HTTP 传输）。"""

    async def remember_model(self, model, chat_type="t2t") -> None:
        del model, chat_type

    async def take(self, email, model, chat_type="t2t"):
        del email, model, chat_type
        return "chat-1", True


class ExecutorStreamDeterministicErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_raises_immediately_on_bare_json_envelope(self) -> None:
        """裸 JSON 错误体必须立刻判错，而不是等 idle 超时。"""
        executor = QwenExecutor(_FakeEngine([f"{BARE_JSON_ENVELOPE}\n\n"]), account_pool=None)
        account = SimpleNamespace(email="acc@example.com", token="tok")

        with self.assertRaises(DeterministicUpstreamError) as ctx:
            async def _drain():
                async for _evt in executor.stream(account, "chat-1", "qwen3.8-max", "hello"):
                    pass

            await asyncio.wait_for(_drain(), timeout=5)

        self.assertEqual(ctx.exception.code, "Bad_Request")
        self.assertIn("Internal error", ctx.exception.details)

    async def test_stream_yields_error_event_for_transient_error(self) -> None:
        """非确定性错误仍然作为事件下发，交由上层决定重试（行为不变）。"""
        chunk = 'data: {"error":{"code":"quota_limit","details":"请稍后再试。"}}\n\n'
        executor = QwenExecutor(_FakeEngine([chunk], hang=False), account_pool=None)
        account = SimpleNamespace(email="acc@example.com", token="tok")

        events = []
        async for evt in executor.stream(account, "chat-1", "qwen3.8-max", "hello"):
            events.append(evt)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["code"], "quota_limit")


class ExecutorRetryDeterministicErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_error_is_not_retried(self) -> None:
        pool = _RotatingPool()
        executor = QwenExecutor(SimpleNamespace(chat_id_pool=_FakeChatPool()), pool)
        attempts: list = []

        async def fake_stream(acc, chat_id, model, content, has_custom_tools=False, files=None, chat_type="t2t", media_options=None):
            del chat_id, model, content, has_custom_tools, files, chat_type, media_options
            attempts.append(acc)
            raise DeterministicUpstreamError(code="Bad_Request", details="Internal error, please try again later.")
            yield  # pragma: no cover - 仅为把函数标记为 async generator

        executor.stream = fake_stream

        with self.assertRaises(DeterministicUpstreamError):
            async for _event in executor.chat_stream_events_with_retry("qwen3.8-max", "hello"):
                pass

        self.assertEqual(len(attempts), 1, "确定性错误不应重试")
        self.assertEqual(len(pool.acquired), 1)
        self.assertEqual(pool.released, pool.acquired, "账号必须被释放，否则会被永久占住")
        self.assertEqual(pool.invalidated, [], "确定性错误不应把账号标成 invalid")
        self.assertEqual(pool.rate_limited, [])

    async def test_transient_error_still_retries_until_max_retries(self) -> None:
        pool = _RotatingPool()
        executor = QwenExecutor(SimpleNamespace(chat_id_pool=_FakeChatPool()), pool)
        attempts: list = []

        async def fake_stream(acc, chat_id, model, content, has_custom_tools=False, files=None, chat_type="t2t", media_options=None):
            del chat_id, model, content, has_custom_tools, files, chat_type, media_options
            attempts.append(acc)
            raise TimeoutError("upstream stream idle timeout after 20s")
            yield  # pragma: no cover

        executor.stream = fake_stream

        with patch("backend.upstream.qwen_executor.settings.MAX_RETRIES", 3):
            with patch("backend.upstream.qwen_executor._retry_backoff_seconds", return_value=0.0):
                with self.assertRaisesRegex(Exception, "All 3 attempts failed"):
                    async for _event in executor.chat_stream_events_with_retry("qwen3.8-max", "hello"):
                        pass

        self.assertEqual(len(attempts), 3, "瞬时错误应保持原有重试逻辑")
        self.assertEqual(len(pool.released), 3)


if __name__ == "__main__":
    unittest.main()
