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


if __name__ == "__main__":
    unittest.main()
