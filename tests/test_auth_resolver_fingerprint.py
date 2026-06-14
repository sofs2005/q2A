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
from backend.services.auth_resolver import AuthResolver


class _DummyPool:
    async def save(self):
        return None


class _FakeResponse:
    def __init__(self):
        self.status_code = 200
        self.cookies = {}

    def json(self):
        return {"token": "new-token"}


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse()


class AuthResolverFingerprintTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_token_uses_account_fingerprint_session(self) -> None:
        account = Account(email="alice@example.com", password="secret", token="old-token")
        account.fingerprint_id = "chrome146_windows"
        resolver = AuthResolver(_DummyPool())
        session = _FakeSession()
        get_session = AsyncMock(return_value=session)

        with patch("backend.services.auth_resolver.get_session", get_session):
            ok = await resolver.refresh_token(account)

        self.assertTrue(ok)
        self.assertEqual(get_session.await_count, 1)
        fingerprint = fingerprint_for_account(account)
        self.assertEqual(session.calls[0]["headers"]["User-Agent"], fingerprint.user_agent)
        self.assertEqual(session.calls[0]["headers"]["sec-ch-ua-platform"], fingerprint.platform)


if __name__ == "__main__":
    unittest.main()
