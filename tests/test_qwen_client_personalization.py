import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
from backend.services.qwen_client import QwenClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse], calls: list[dict]):
        self.responses = responses
        self.calls = calls

    async def request(self, method, url, headers=None, json=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json, "kwargs": kwargs})
        return self.responses.pop(0)


class QwenClientPersonalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_personalization_settings_uses_cookie_then_token_fallback(self) -> None:
        account = Account(email="user@example.com", token="token-1", cookies="aui=1; cna=2")
        client = QwenClient(SimpleNamespace())
        calls: list[dict] = []
        upstream_body = {
            "data": {
                "memory": {"enable_memory": True, "enable_history_memory": False},
                "tools_enabled": {f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
            }
        }
        responses = [
            _FakeResponse(403, {"detail": "forbidden"}, "forbidden"),
            _FakeResponse(200, upstream_body, json.dumps(upstream_body)),
        ]
        get_session = AsyncMock(return_value=_FakeSession(responses, calls))

        with unittest.mock.patch("backend.services.qwen_client.get_session", get_session):
            result = await client.get_personalization_settings(account)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["transport"], "token")
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["url"], "https://chat.qwen.ai/api/v2/configs/setting-config")
        self.assertEqual(calls[0]["headers"]["Cookie"], "aui=1; cna=2")
        self.assertNotIn("Authorization", calls[0]["headers"])
        self.assertEqual(calls[1]["headers"]["Authorization"], "Bearer token-1")
        self.assertNotIn("Cookie", calls[1]["headers"])
        self.assertEqual(get_session.await_count, 2)

    async def test_update_personalization_settings_posts_payload_with_token_headers(self) -> None:
        account = Account(email="user@example.com", token="token-1")
        client = QwenClient(SimpleNamespace())
        calls: list[dict] = []
        payload = {
            "memory": {"enable_memory": True, "enable_history_memory": False},
            "tools_enabled": {f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
        }
        responses = [_FakeResponse(200, {"ok": True}, '{"ok": true}')]
        get_session = AsyncMock(return_value=_FakeSession(responses, calls))

        with unittest.mock.patch("backend.services.qwen_client.get_session", get_session):
            result = await client.update_personalization_settings(account, payload)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["transport"], "token")
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(calls[0]["url"], "https://chat.qwen.ai/api/v2/users/user/settings/update")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer token-1")
        self.assertEqual(calls[0]["json"], payload)
        self.assertEqual(get_session.await_count, 1)

    async def test_get_personalization_settings_skips_without_credentials(self) -> None:
        account = Account(email="user@example.com")
        client = QwenClient(SimpleNamespace())
        get_session = AsyncMock()

        with unittest.mock.patch("backend.services.qwen_client.get_session", get_session):
            result = await client.get_personalization_settings(account)

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "missing_credentials")
        self.assertEqual(get_session.await_count, 0)


if __name__ == "__main__":
    unittest.main()
