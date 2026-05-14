import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.core import httpx_engine
from backend.core.config import settings
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
        httpx_engine._global_session = None
        httpx_engine._session_lock = None

    async def asyncTearDown(self) -> None:
        if self.original_request_timeout is not None:
            settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS = self.original_request_timeout
        if self.original_stream_timeout is not None:
            settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS = self.original_stream_timeout
        httpx_engine._global_session = None
        httpx_engine._session_lock = None

    async def test_create_chat_uses_configured_upstream_request_timeout(self) -> None:
        captured = {}

        class FakeEngine:
            async def _request_json(self, method, path, token, body, timeout):
                captured["timeout"] = timeout
                return {"status": 200, "body": '{"success": true, "data": {"id": "chat-1"}}'}

        settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS = 75.0
        executor = QwenExecutor(FakeEngine(), account_pool=None)

        chat_id = await executor.create_chat("tok", "qwen3.6-plus")

        self.assertEqual(chat_id, "chat-1")
        self.assertEqual(captured["timeout"], 75.0)

    async def test_curl_stream_session_uses_configured_upstream_stream_timeout(self) -> None:
        captured = {}

        class FakeSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS = 300.0

        with patch("backend.core.httpx_engine.AsyncSession", FakeSession):
            session = await httpx_engine._get_global_session()

        self.assertIsInstance(session, FakeSession)
        self.assertEqual(captured["timeout"], 300.0)

    async def test_qwen_client_request_json_uses_configured_default_timeout(self) -> None:
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def request(self, *_args, **_kwargs):
                return SimpleNamespace(status_code=200, text="{}")

        settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS = 88.0

        with patch("backend.services.qwen_client.httpx.AsyncClient", FakeClient):
            result = await QwenClient(account_pool=None)._request_json(
                "GET",
                "/api/test",
                "tok",
            )

        self.assertEqual(result, {"status": 200, "body": "{}"})
        self.assertEqual(captured["timeout"], 88.0)

    async def test_qwen_client_stream_uses_configured_read_timeout(self) -> None:
        captured = {}

        class FakeResponse:
            status_code = 500

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def aread(self):
                return b"upstream error"

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def stream(self, *_args, **_kwargs):
                return FakeResponse()

        settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS = 420.0

        with patch("backend.services.qwen_client.httpx.AsyncClient", FakeClient):
            events = [
                event
                async for event in QwenClient(account_pool=None).stream_chat_once(
                    "tok",
                    "chat-1",
                    {},
                )
            ]

        self.assertEqual(events, [{"status": 500, "body": b"upstream error"}])
        self.assertEqual(captured["timeout"].read, 420.0)


if __name__ == "__main__":
    unittest.main()
