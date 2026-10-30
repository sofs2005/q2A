"""短效 access_token 提供者的纯逻辑测试（不启动浏览器）。

覆盖：JWT claims 解析、token 类型判定、可用性判定、以及 AuthResolver 在
"浏览器签发成功" 时直接采用短效 token 的行为。
"""

import base64
import json
import sys
import time
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
from backend.services.auth_resolver import AuthResolver
from backend.services.token_provider import (
    is_token_usable,
    parse_jwt_claims,
    token_expires_at,
    token_kind,
)


def make_jwt(claims: dict) -> str:
    """造一个只用于解析的假 JWT（不校验签名）。"""
    def seg(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'HS256', 'typ': 'JWT'})}.{seg(claims)}.sig"


class JwtParsingTests(unittest.TestCase):
    def test_parse_claims_roundtrip(self) -> None:
        token = make_jwt({"id": "abc", "type": "access_token", "exp": 1793351217, "iat": 1793350317})
        claims = parse_jwt_claims(token)
        self.assertEqual(claims["type"], "access_token")
        self.assertEqual(claims["exp"], 1793351217)

    def test_parse_claims_tolerates_garbage(self) -> None:
        for bad in ("", "not-a-jwt", "a.b", "a.b.c.d"):
            with self.subTest(bad=bad):
                self.assertIsInstance(parse_jwt_claims(bad), dict)

    def test_token_kind(self) -> None:
        access = make_jwt({"type": "access_token", "exp": 1})
        legacy = make_jwt({"id": "x", "exp": 1})
        self.assertEqual(token_kind(access), "access")
        self.assertEqual(token_kind(legacy), "legacy")
        self.assertEqual(token_kind("garbage"), "unknown")

    def test_token_expires_at(self) -> None:
        token = make_jwt({"exp": 1793351217})
        self.assertEqual(token_expires_at(token), 1793351217.0)
        self.assertEqual(token_expires_at("garbage"), 0.0)

    def test_is_token_usable(self) -> None:
        now = time.time()
        fresh = make_jwt({"exp": now + 600})
        expiring = make_jwt({"exp": now + 30})
        expired = make_jwt({"exp": now - 10})
        self.assertTrue(is_token_usable(fresh))
        self.assertFalse(is_token_usable(expiring))  # 落在 120s 续期窗口内
        self.assertFalse(is_token_usable(expired))
        # 解析不出 exp 的长期 token 视为不可用：上游只认带 exp 的短效 token
        self.assertFalse(is_token_usable(make_jwt({"id": "x"})))
        self.assertFalse(is_token_usable(""))


class _DummyPool:
    def __init__(self) -> None:
        self.saved = 0

    async def save(self) -> None:
        self.saved += 1


class RefreshPrefersBrowserTokenTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_token_short_circuits_http_signin(self) -> None:
        pool = _DummyPool()
        resolver = AuthResolver(pool)
        account = Account(email="user@example.com", password="secret", token="old-token")
        access = make_jwt({"type": "access_token", "exp": time.time() + 600})

        http_session = AsyncMock()
        with patch(
            "backend.services.auth_resolver.ensure_fresh_token",
            AsyncMock(return_value=access),
        ), patch("backend.services.auth_resolver.get_session", http_session):
            ok = await resolver.refresh_token(account)

        self.assertTrue(ok)
        self.assertEqual(account.token, access)
        self.assertTrue(account.valid)
        self.assertEqual(pool.saved, 1)
        # 浏览器签发成功时不应再走 HTTP signin
        http_session.assert_not_called()

    async def test_falls_back_to_http_when_browser_fails(self) -> None:
        pool = _DummyPool()
        resolver = AuthResolver(pool)
        account = Account(email="user@example.com", password="secret", token="old-token")
        account.fingerprint_id = "chrome146_windows"

        class _Resp:
            status_code = 200
            cookies = {}

            def json(self):
                return {"success": True, "data": {"token": "legacy-token"}}

            @property
            def text(self):
                return ""

        class _Session:
            async def post(self, url, json=None, headers=None):
                return _Resp()

        with patch(
            "backend.services.auth_resolver.ensure_fresh_token",
            AsyncMock(return_value=""),
        ), patch("backend.services.auth_resolver.get_session", AsyncMock(return_value=_Session())):
            ok = await resolver.refresh_token(account)

        self.assertTrue(ok)
        self.assertEqual(account.token, "legacy-token")


if __name__ == "__main__":
    unittest.main()
