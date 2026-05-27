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

if "pydantic" not in sys.modules:
    fake_pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

    fake_pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = fake_pydantic

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

if "fastapi" not in sys.modules:
    fake_fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class APIRouter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def _decorator(self, *args, **kwargs):
            def wrapper(func):
                return func

            return wrapper

        get = post = put = delete = _decorator

    class Depends:
        def __init__(self, dependency):
            self.dependency = dependency

    def Header(default=None):
        return default

    class Request:
        pass

    fake_fastapi.APIRouter = APIRouter
    fake_fastapi.Depends = Depends
    fake_fastapi.HTTPException = HTTPException
    fake_fastapi.Header = Header
    fake_fastapi.Request = Request
    sys.modules["fastapi"] = fake_fastapi

from fastapi import HTTPException

from backend.api import admin
from backend.core.account_pool import Account


def _make_request(account_pool, qwen_client):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(account_pool=account_pool, qwen_client=qwen_client)))


class AdminChatClearTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_clear_uses_only_selected_emails(self) -> None:
        available_cookie = Account(email="cookie@example.com", token="token-1", cookies="aui=1; cna=2")
        available_token = Account(email="token@example.com", token="token-2")
        unselected = Account(email="other@example.com", token="token-3")

        pool = SimpleNamespace(accounts=[available_cookie, available_token, unselected])
        qwen_client = SimpleNamespace(
            clear_all_chats=AsyncMock(
                side_effect=[
                    {"email": "cookie@example.com", "status": "success", "transport": "cookie"},
                    {"email": "token@example.com", "status": "failed", "transport": "token", "error": "HTTP 403: forbidden"},
                ]
            )
        )
        payload = SimpleNamespace(emails=["cookie@example.com", "token@example.com"])
        request = _make_request(pool, qwen_client)

        result = await admin.clear_all_upstream_chats(payload, request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], {"success": 1, "failed": 1, "skipped": 0})
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(
            [call.args[0].email for call in qwen_client.clear_all_chats.await_args_list],
            ["cookie@example.com", "token@example.com"],
        )

    async def test_batch_clear_normalizes_selection_and_skips_missing_accounts(self) -> None:
        available_cookie = Account(email="cookie@example.com", token="token-1", cookies="aui=1; cna=2")
        unselected = Account(email="other@example.com", token="token-3")
        pool = SimpleNamespace(accounts=[available_cookie, unselected])
        qwen_client = SimpleNamespace(
            clear_all_chats=AsyncMock(
                return_value={"email": "cookie@example.com", "status": "success", "transport": "cookie"}
            )
        )
        payload = SimpleNamespace(
            emails=[" cookie@example.com ", "missing@example.com", "cookie@example.com", "", "   "]
        )
        request = _make_request(pool, qwen_client)

        result = await admin.clear_all_upstream_chats(payload, request)

        self.assertEqual(result["summary"], {"success": 1, "failed": 0, "skipped": 1})
        self.assertEqual(
            [item["email"] for item in result["results"]],
            ["cookie@example.com", "missing@example.com"],
        )
        self.assertEqual(
            [call.args[0].email for call in qwen_client.clear_all_chats.await_args_list],
            ["cookie@example.com"],
        )

    async def test_batch_clear_rejects_empty_selection(self) -> None:
        pool = SimpleNamespace(accounts=[])
        qwen_client = SimpleNamespace(clear_all_chats=AsyncMock())
        payload = SimpleNamespace(emails=[])
        request = _make_request(pool, qwen_client)

        with self.assertRaises(HTTPException) as ctx:
            await admin.clear_all_upstream_chats(payload, request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(qwen_client.clear_all_chats.await_count, 0)

    async def test_batch_clear_rejects_whitespace_only_selection(self) -> None:
        pool = SimpleNamespace(accounts=[])
        qwen_client = SimpleNamespace(clear_all_chats=AsyncMock())
        payload = SimpleNamespace(emails=["", "   "])
        request = _make_request(pool, qwen_client)

        with self.assertRaises(HTTPException) as ctx:
            await admin.clear_all_upstream_chats(payload, request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(qwen_client.clear_all_chats.await_count, 0)

    async def test_single_account_clear_succeeds_for_existing_account_with_credentials(self) -> None:
        account = Account(email="cookie@example.com", token="token-1", cookies="aui=1; cna=2")
        pool = SimpleNamespace(accounts=[account])
        qwen_client = SimpleNamespace(
            clear_all_chats=AsyncMock(return_value={"email": account.email, "status": "success", "transport": "cookie"})
        )
        request = _make_request(pool, qwen_client)

        result = await admin.clear_upstream_chats_for_account("cookie@example.com", request)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["email"], "cookie@example.com")
        self.assertEqual(qwen_client.clear_all_chats.await_count, 1)
        self.assertEqual(qwen_client.clear_all_chats.await_args.args[0].email, "cookie@example.com")

    async def test_single_account_clear_returns_skipped_when_account_has_no_credentials(self) -> None:
        account = Account(email="nocreds@example.com")
        pool = SimpleNamespace(accounts=[account])
        qwen_client = SimpleNamespace(clear_all_chats=AsyncMock())
        request = _make_request(pool, qwen_client)

        result = await admin.clear_upstream_chats_for_account("nocreds@example.com", request)

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "missing_credentials")
        self.assertEqual(qwen_client.clear_all_chats.await_count, 0)

    async def test_single_account_clear_returns_404_when_account_does_not_exist(self) -> None:
        pool = SimpleNamespace(accounts=[])
        qwen_client = SimpleNamespace(clear_all_chats=AsyncMock())
        request = _make_request(pool, qwen_client)

        with self.assertRaises(HTTPException) as ctx:
            await admin.clear_upstream_chats_for_account("missing@example.com", request)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(qwen_client.clear_all_chats.await_count, 0)


if __name__ == "__main__":
    unittest.main()
