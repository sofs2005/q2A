"""WAF 首页预热必须走与主链路一致的指纹 Session（含 UPSTREAM_PROXY）。"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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
from backend.services.waf_cookie_manager import WafCookieManager


class _FakeSession:
    def __init__(self) -> None:
        self.get = AsyncMock()
        self.cookies = {"acw_tc": "proxy-acw-tc-value"}
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False


class WafCookieManagerProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_account_cookies_uses_fingerprint_new_session(self) -> None:
        """首页 GET acw_tc 必须经 new_session，才能注入 UPSTREAM_PROXY 与账号 TLS 指纹。"""
        account = Account(email="alice@example.com", token="token-1")
        account.fingerprint_id = "chrome146_windows"
        account.waf_cookies = ""
        account.waf_cookies_expires_at = 0

        fake_session = _FakeSession()
        fingerprint = fingerprint_for_account(account)
        new_session = MagicMock(return_value=fake_session)

        with patch("backend.services.waf_cookie_manager.new_session", new_session):
            mgr = WafCookieManager()
            await mgr.refresh_account_cookies(account)

        new_session.assert_called_once()
        called_fp = new_session.call_args.args[0]
        self.assertEqual(called_fp.impersonate, fingerprint.impersonate)
        fake_session.get.assert_awaited()
        get_url = fake_session.get.await_args.args[0]
        self.assertEqual(get_url, "https://chat.qwen.ai")
        self.assertEqual(account.waf_cookies, "acw_tc=proxy-acw-tc-value")
        self.assertGreater(account.waf_cookies_expires_at, 0)


if __name__ == "__main__":
    unittest.main()
