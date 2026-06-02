import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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

from backend.adapter.standard_request import StandardRequest
from backend.core import browser_fingerprint
from backend.core.config import settings
from backend.core.browser_fingerprint import fingerprint_for_email
from backend.runtime.execution import collect_completion_run
from backend.services.qwen_client import QwenClient
from backend.upstream.qwen_executor import QwenExecutor, _has_textual_tool_contract_marker


class UpstreamTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_contract_marker_detection_accepts_dsml_and_legacy_formats(self) -> None:
        self.assertTrue(_has_textual_tool_contract_marker("<|DSML|tool_calls></|DSML|tool_calls>"))
        self.assertTrue(_has_textual_tool_contract_marker("##TOOL_CALL##\n{}\n##END_CALL##"))
        self.assertFalse(_has_textual_tool_contract_marker("plain prompt"))

    async def asyncSetUp(self) -> None:
        self.original_request_timeout = getattr(
            settings,
            "QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS",
            None,
        )
        self.original_stream_timeout = getattr(
            settings,
            "QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS",
            None,
        )
        browser_fingerprint._sessions.clear()
        browser_fingerprint._session_lock = None

    async def asyncTearDown(self) -> None:
        if self.original_request_timeout is not None:
            settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS = self.original_request_timeout
        if self.original_stream_timeout is not None:
            settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS = self.original_stream_timeout
        browser_fingerprint._sessions.clear()
        browser_fingerprint._session_lock = None

    async def test_create_chat_uses_configured_upstream_request_timeout(self) -> None:
        captured = {}

        class FakeEngine:
            async def _request_json(self, method, path, token, body, timeout, account=None):
                del method, path, token, body, account
                captured["timeout"] = timeout
                return {"status": 200, "body": '{"success": true, "data": {"id": "chat-1"}}'}

        settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS = 75.0
        executor = QwenExecutor(FakeEngine(), account_pool=None)

        chat_id = await executor.create_chat("tok", "qwen3.6-plus")

        self.assertEqual(chat_id, "chat-1")
        self.assertEqual(captured["timeout"], 75.0)

    async def test_fingerprint_session_uses_configured_upstream_stream_timeout(self) -> None:
        captured = {}

        class FakeSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS = 300.0
        fingerprint = fingerprint_for_email("user@example.com")

        with patch("backend.core.browser_fingerprint.AsyncSession", FakeSession):
            session = await browser_fingerprint.get_session(fingerprint)

        self.assertIsInstance(session, FakeSession)
        self.assertEqual(captured["impersonate"], fingerprint.impersonate)
        self.assertEqual(captured["timeout"], 300.0)

    async def test_qwen_client_request_json_uses_configured_default_timeout(self) -> None:
        captured = {}

        class FakeSession:
            async def request(self, *_args, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(status_code=200, text="{}")

        settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS = 88.0
        get_session = AsyncMock(return_value=FakeSession())

        with patch("backend.services.qwen_client.get_session", get_session):
            result = await QwenClient(account_pool=None)._request_json(
                "GET",
                "/api/test",
                "tok",
            )

        self.assertEqual(result, {"status": 200, "body": "{}"})
        self.assertEqual(captured["timeout"], 88.0)

    async def test_executor_releases_account_when_stream_is_closed_after_acquire(self) -> None:
        acc = SimpleNamespace(email="acc@example.com", token="tok")

        class FakePool:
            def __init__(self):
                self.released = 0

            async def acquire_wait(self, *, timeout, exclude):
                del timeout, exclude
                return acc

            def release(self, released_acc):
                if released_acc is not acc:
                    raise AssertionError("released unexpected account")
                self.released += 1

        class FakeExecutor(QwenExecutor):
            async def create_chat(self, account, model):
                del account, model
                return "chat-1"

            async def stream(self, account, chat_id, model, content, has_custom_tools, files=None):
                del account, chat_id, model, content, has_custom_tools, files
                yield {"type": "delta", "phase": "answer", "content": "hello"}

        pool = FakePool()
        executor = FakeExecutor(engine=None, account_pool=pool)
        stream = executor.chat_stream_events_with_retry("qwen3.6-plus", "hello")

        first = await anext(stream)
        await stream.aclose()

        self.assertEqual(first["type"], "meta")
        self.assertEqual(pool.released, 1)

    async def test_executor_releases_account_when_waiting_stream_is_cancelled(self) -> None:
        acc = SimpleNamespace(email="acc@example.com", token="tok")

        class FakePool:
            def __init__(self):
                self.released = 0

            async def acquire_wait(self, *, timeout, exclude):
                del timeout, exclude
                return acc

            def release(self, released_acc):
                if released_acc is not acc:
                    raise AssertionError("released unexpected account")
                self.released += 1

        class FakeExecutor(QwenExecutor):
            async def create_chat(self, account, model):
                del account, model
                return "chat-1"

            async def stream(self, account, chat_id, model, content, has_custom_tools, files=None):
                del account, chat_id, model, content, has_custom_tools, files
                await asyncio.Event().wait()
                yield {"type": "delta", "phase": "answer", "content": "never"}

        pool = FakePool()
        executor = FakeExecutor(engine=None, account_pool=pool)
        stream = executor.chat_stream_events_with_retry("qwen3.6-plus", "hello")

        first = await anext(stream)
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(anext(stream), timeout=0.01)

        self.assertEqual(first["type"], "meta")
        self.assertEqual(pool.released, 1)

    async def test_collect_completion_run_cleans_up_when_stream_fails_after_meta(self) -> None:
        acc = SimpleNamespace(email="acc@example.com", token="tok")

        class FakeClient:
            async def chat_stream_events_with_retry(self, *_args, **_kwargs):
                yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
                raise RuntimeError("stream failed")

        request = StandardRequest(
            prompt="hello",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )
        client = FakeClient()

        with patch("backend.runtime.execution.cleanup_runtime_resources", new=AsyncMock()) as cleanup:
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                await collect_completion_run(client, request, "hello")

        cleanup.assert_awaited_once_with(client, acc, "chat-1")

    async def test_qwen_client_stream_uses_configured_read_timeout(self) -> None:
        captured = {}

        class FakeResponse:
            status_code = 500

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def aiter_content(self):
                yield b"upstream error"

        class FakeSession:
            def stream(self, *_args, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS = 420.0
        get_session = AsyncMock(return_value=FakeSession())

        with patch("backend.services.qwen_client.get_session", get_session):
            events = [
                event
                async for event in QwenClient(account_pool=None).stream_chat_once(
                    "tok",
                    "chat-1",
                    {},
                )
            ]

        self.assertEqual(events, [{"status": 500, "body": "upstream error"}])
        self.assertEqual(captured["timeout"], 420.0)


if __name__ == "__main__":
    unittest.main()
