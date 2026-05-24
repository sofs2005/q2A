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

if "httpx" not in sys.modules:
    fake_httpx = types.ModuleType("httpx")

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Timeout:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_httpx.AsyncClient = AsyncClient
    fake_httpx.Timeout = Timeout
    sys.modules["httpx"] = fake_httpx

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

from backend.core.account_pool import Account
from backend.services.qwen_client import QwenClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses: list[_FakeResponse], calls: list[dict], **kwargs):
        self.responses = responses
        self.calls = calls
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb

    async def request(self, method, url, headers=None, json=None):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json})
        return self.responses.pop(0)


class QwenClientChatClearTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_all_chats_uses_cookie_then_falls_back_to_token(self) -> None:
        account = Account(email="user@example.com", token="token-1", cookies="aui=1; cna=2")
        client = QwenClient(SimpleNamespace())
        calls: list[dict] = []
        responses = [
            _FakeResponse(403, {"detail": "forbidden"}, "forbidden"),
            _FakeResponse(200, {"success": True, "data": {"status": True}}, '{"success": true, "data": {"status": true}}'),
        ]

        with patch(
            "backend.services.qwen_client.httpx.AsyncClient",
            lambda **kwargs: _FakeAsyncClient(responses, calls, **kwargs),
        ):
            result = await client.clear_all_chats(account)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["transport"], "token")
        self.assertEqual(calls[0]["method"], "DELETE")
        self.assertEqual(calls[0]["url"], "https://chat.qwen.ai/api/v2/chats/")
        self.assertEqual(calls[0]["headers"]["Cookie"], "aui=1; cna=2")
        self.assertNotIn("Authorization", calls[0]["headers"])
        self.assertEqual(calls[1]["headers"]["Authorization"], "Bearer token-1")

    async def test_clear_all_chats_skips_without_credentials(self) -> None:
        account = Account(email="user@example.com")
        client = QwenClient(SimpleNamespace())
        calls: list[dict] = []

        with patch("backend.services.qwen_client.httpx.AsyncClient", lambda **kwargs: _FakeAsyncClient([], calls, **kwargs)):
            result = await client.clear_all_chats(account)

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "missing_credentials")
        self.assertEqual(calls, [])

    async def test_clear_all_chats_reports_failure_when_both_attempts_fail(self) -> None:
        account = Account(email="user@example.com", token="token-1", cookies="aui=1; cna=2")
        client = QwenClient(SimpleNamespace())
        calls: list[dict] = []
        responses = [
            _FakeResponse(403, {"detail": "forbidden"}, "forbidden"),
            _FakeResponse(403, {"detail": "forbidden"}, "forbidden"),
        ]

        with patch(
            "backend.services.qwen_client.httpx.AsyncClient",
            lambda **kwargs: _FakeAsyncClient(responses, calls, **kwargs),
        ):
            result = await client.clear_all_chats(account)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["transport"], "token")
        self.assertTrue(result["error"].startswith("HTTP 403"))
        self.assertEqual(len(calls), 2)

    async def test_clear_all_chats_does_not_fallback_on_non_auth_cookie_failure(self) -> None:
        account = Account(email="user@example.com", token="token-1", cookies="aui=1; cna=2")
        client = QwenClient(SimpleNamespace())
        calls: list[dict] = []
        responses = [
            _FakeResponse(500, {"detail": "server error"}, "server error"),
        ]

        with patch(
            "backend.services.qwen_client.httpx.AsyncClient",
            lambda **kwargs: _FakeAsyncClient(responses, calls, **kwargs),
        ):
            result = await client.clear_all_chats(account)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["transport"], "cookie")
        self.assertTrue(result["error"].startswith("HTTP 500"))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
