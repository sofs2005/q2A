import sys
import types
import unittest
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
from backend.services.auth_resolver import AuthResolver, activate_account, register_qwen_account, BASE_URL


class _DummyPool:
    def __init__(self) -> None:
        self.saved = 0

    async def save(self):
        self.saved += 1


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse, calls: list[tuple[str, dict, dict]]):
        self._response = response
        self._calls = calls

    async def post(self, url, json=None, headers=None):
        self._calls.append((url, json or {}, headers or {}))
        return self._response


class AuthResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_token_uses_curl_cffi_signin(self) -> None:
        pool = _DummyPool()
        resolver = AuthResolver(pool)
        account = Account(email="user@example.com", password="secret", token="old-token")
        account.fingerprint_id = "chrome146_windows"
        calls: list[tuple[str, dict, dict]] = []
        fake_response = _FakeResponse(200, {"token": "new-token"})
        get_session = AsyncMock(return_value=_FakeSession(fake_response, calls))

        with patch("backend.services.auth_resolver.get_session", get_session):
            ok = await resolver.refresh_token(account)

        fingerprint = fingerprint_for_account(account)
        self.assertTrue(ok)
        self.assertEqual(account.token, "new-token")
        self.assertEqual(account.status_code, "valid")
        self.assertEqual(account.last_error, "")
        self.assertFalse(account.activation_pending)
        self.assertEqual(pool.saved, 1)
        self.assertEqual(calls[0][0], f"{BASE_URL}/api/v1/auths/signin")
        self.assertEqual(calls[0][1]["email"], "user@example.com")
        self.assertNotEqual(calls[0][1]["password"], "secret")
        self.assertEqual(calls[0][2]["User-Agent"], fingerprint.user_agent)

    async def test_refresh_token_returns_false_on_non_200(self) -> None:
        pool = _DummyPool()
        resolver = AuthResolver(pool)
        account = Account(email="user@example.com", password="secret", token="old-token")
        calls: list[tuple[str, dict, dict]] = []
        fake_response = _FakeResponse(401, {"detail": "bad credentials"})
        get_session = AsyncMock(return_value=_FakeSession(fake_response, calls))

        with patch("backend.services.auth_resolver.get_session", get_session):
            ok = await resolver.refresh_token(account)

        self.assertFalse(ok)
        self.assertEqual(account.token, "old-token")
        self.assertEqual(pool.saved, 0)
        self.assertEqual(len(calls), 1)

    async def test_refresh_token_returns_false_on_non_json(self) -> None:
        pool = _DummyPool()
        resolver = AuthResolver(pool)
        account = Account(email="user@example.com", password="secret", token="old-token")
        calls: list[tuple[str, dict, dict]] = []
        fake_response = _FakeResponse(200, None, "<html>blocked</html>")
        get_session = AsyncMock(return_value=_FakeSession(fake_response, calls))

        with patch("backend.services.auth_resolver.get_session", get_session):
            ok = await resolver.refresh_token(account)

        self.assertFalse(ok)
        self.assertEqual(account.token, "old-token")
        self.assertEqual(pool.saved, 0)

    async def test_refresh_token_returns_false_when_token_missing(self) -> None:
        pool = _DummyPool()
        resolver = AuthResolver(pool)
        account = Account(email="user@example.com", password="secret", token="old-token")
        calls: list[tuple[str, dict, dict]] = []
        fake_response = _FakeResponse(200, {"token": ""})
        get_session = AsyncMock(return_value=_FakeSession(fake_response, calls))

        with patch("backend.services.auth_resolver.get_session", get_session):
            ok = await resolver.refresh_token(account)

        self.assertFalse(ok)
        self.assertEqual(account.token, "old-token")
        self.assertEqual(pool.saved, 0)

    async def test_register_qwen_account_is_disabled_without_browser(self) -> None:
        self.assertIsNone(await register_qwen_account())

    async def test_activate_account_is_disabled_without_browser(self) -> None:
        account = Account(email="user@example.com", password="secret", token="old-token")
        self.assertFalse(await activate_account(account))


if __name__ == "__main__":
    unittest.main()
