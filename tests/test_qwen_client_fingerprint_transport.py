import json
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

from backend.core.account_pool import Account
from backend.core.browser_fingerprint import fingerprint_for_account
from backend.services.qwen_client import QwenClient


class _FakeResponse:
    status_code = 200
    text = '{"ok": true}'


class _FakeSession:
    def __init__(self) -> None:
        self.calls = []

    async def request(self, method, url, headers=None, json=None, data=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json, "data": data, "kwargs": kwargs})
        return _FakeResponse()


class _FakeStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        yield b"data: {\"text\":\"ok\"}\n\n"


class _FakeAsyncClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        _FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json, "timeout": timeout})
        return _FakeStreamResponse()


class QwenClientFingerprintTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_personalization_uses_account_fingerprint_session_and_headers(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        account.fingerprint_id = "firefox147_windows"
        client = QwenClient(SimpleNamespace())
        session = _FakeSession()
        get_session = AsyncMock(return_value=session)

        with patch("backend.services.qwen_client.get_session", get_session):
            result = await client.update_personalization_settings(account, {"memory": {"enable_memory": True}})

        fingerprint = fingerprint_for_account(account)
        self.assertEqual(result["status"], "success")
        self.assertEqual(get_session.await_args.args[0], fingerprint)
        self.assertEqual(session.calls[0]["headers"]["User-Agent"], fingerprint.user_agent)
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer token-1")
        self.assertEqual(session.calls[0]["json"], {"memory": {"enable_memory": True}})

    def test_build_headers_uses_account_cookies_without_dropping_authorization(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="waf=ok; session=browser")
        account.fingerprint_id = "chrome136_windows"

        headers = QwenClient._build_headers(account=account, token=account.token)

        fingerprint = fingerprint_for_account(account)
        self.assertEqual(headers["User-Agent"], fingerprint.user_agent)
        self.assertEqual(headers["Cookie"], "waf=ok; session=browser")
        self.assertEqual(headers["Authorization"], "Bearer token-1")

    def test_build_chat_transport_headers_default_to_token_only(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="waf=ok; session=browser")
        account.fingerprint_id = "chrome136_windows"

        headers = QwenClient._build_chat_transport_headers(account=account, token=account.token)

        self.assertNotIn("Cookie", headers)
        self.assertEqual(headers["Authorization"], "Bearer token-1")

    def test_build_chat_transport_headers_use_go_like_static_browser_headers(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="waf=ok; session=browser")
        account.fingerprint_id = "chrome136_windows"

        headers = QwenClient._build_chat_transport_headers(account=account, token=account.token, accept="text/event-stream")

        self.assertEqual(headers["Accept"], "text/event-stream")
        self.assertEqual(headers["User-Agent"], "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        self.assertEqual(headers["sec-ch-ua"], '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"')
        # Chat transport now includes web client headers for WAF bypass
        self.assertIn("Version", headers)
        self.assertIn("source", headers)
        self.assertIn("Timezone", headers)
        self.assertEqual(headers["X-Accel-Buffering"], "no")

    async def test_request_json_sends_web_client_headers_for_chat_requests(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        client = QwenClient(SimpleNamespace())
        session = _FakeSession()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)):
            await client._request_json("POST", "/api/v2/chats/new", account.token, {}, account=account)

        headers = session.calls[0]["headers"]
        self.assertEqual(headers["Version"], "0.2.57")
        self.assertEqual(headers["source"], "web")
        self.assertIn("X-Request-Id", headers)
        self.assertIn("Timezone", headers)

    async def test_request_json_omits_cookie_for_chat_transport(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="waf=ok; session=browser")
        client = QwenClient(SimpleNamespace())
        session = _FakeSession()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)):
            await client._request_json("POST", "/api/v2/chats/new", account.token, {}, account=account, chat_transport=True)

        headers = session.calls[0]["headers"]
        self.assertNotIn("Cookie", headers)
        self.assertEqual(headers["Authorization"], "Bearer token-1")

    async def test_stream_chat_once_uses_go_like_http_transport_by_default(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="waf=ok; session=browser")
        client = QwenClient(SimpleNamespace())
        _FakeAsyncClient.instances = []

        with patch("backend.services.qwen_client.httpx.AsyncClient", _FakeAsyncClient), patch("backend.services.qwen_client.new_session") as new_session:
            chunks = [item async for item in client.stream_chat_once(account.token, "chat-1", {"stream": True}, account=account)]

        self.assertFalse(new_session.called)
        self.assertEqual(chunks[-1], {"status": "streamed"})
        self.assertEqual(_FakeAsyncClient.instances[0].kwargs["http2"], False)
        call = _FakeAsyncClient.instances[0].calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertIn("/api/v2/chat/completions?chat_id=chat-1", call["url"])
        self.assertNotIn("Cookie", call["headers"])

    async def test_refresh_token_captures_acw_tc_cookie(self) -> None:
        """auth_resolver.refresh_token should capture acw_tc from login response Set-Cookie."""
        from backend.services.auth_resolver import AuthResolver
        from backend.core.account_pool import Account, AccountPool

        account = Account(email="test@example.com", password="pass123", token="old-token")
        pool = SimpleNamespace(save=AsyncMock())

        resolver = AuthResolver(pool)

        fake_resp = SimpleNamespace(
            status_code=200,
            cookies={"acw_tc": "captured-acw-tc-value"},
            json=lambda: {"token": "new-token-xyz"},
        )

        fake_session = AsyncMock()
        fake_session.post = AsyncMock(return_value=fake_resp)

        with patch("backend.services.auth_resolver.get_session", AsyncMock(return_value=fake_session)):
            result = await resolver.refresh_token(account)

        self.assertTrue(result)
        self.assertEqual(account.token, "new-token-xyz")
        self.assertIn("acw_tc=captured-acw-tc-value", getattr(account, "waf_cookies", ""))
        self.assertGreater(getattr(account, "waf_cookies_expires_at", 0), 0)

    def test_header_diagnostics_are_redacted(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="waf=ok; session=browser")
        account.fingerprint_id = "chrome136_windows"

        headers = QwenClient._build_headers(account=account, token=account.token)
        diagnostics = QwenClient._header_diagnostics(account=account, headers=headers)

        self.assertTrue(diagnostics["has_cookie"])
        self.assertTrue(diagnostics["has_authorization"])
        self.assertEqual(diagnostics["cookie_names"], ["waf", "session"])
        self.assertEqual(diagnostics["fingerprint_id"], "chrome136_windows")
        self.assertNotIn("token-1", str(diagnostics))
        self.assertNotIn("browser", str(diagnostics))

    def test_web_client_headers_include_x_accel_buffering(self) -> None:
        headers = QwenClient._web_client_headers()
        self.assertEqual(headers["X-Accel-Buffering"], "no")
        self.assertIn("Version", headers)
        self.assertIn("source", headers)
        self.assertIn("X-Request-Id", headers)
        self.assertIn("Timezone", headers)

    def test_build_chat_transport_headers_include_web_client_headers(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        headers = QwenClient._build_chat_transport_headers(account=account, token=account.token)
        self.assertEqual(headers["X-Accel-Buffering"], "no")
        self.assertIn("Version", headers)
        self.assertEqual(headers["source"], "web")
        self.assertIn("X-Request-Id", headers)
        self.assertIn("Timezone", headers)


if __name__ == "__main__":
    unittest.main()
