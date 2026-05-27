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


class AdminPersonalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_get_returns_modal_ready_data(self) -> None:
        account = Account(email="alice@example.com", token="token-1", cookies="aui=1; cna=2")
        pool = SimpleNamespace(accounts=[account])
        upstream_body = {
            "data": {
                "memory": {"enable_memory": True, "enable_history_memory": False},
                "tools_enabled": {f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
            }
        }
        qwen_client = SimpleNamespace(
            get_personalization_settings=AsyncMock(
                return_value={
                    "email": account.email,
                    "status": "success",
                    "transport": "cookie",
                    "http_status": 200,
                    "body": json.dumps(upstream_body),
                }
            )
        )
        request = _make_request(pool, qwen_client)

        result = await admin.get_account_personalization("alice@example.com", request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["email"], "alice@example.com")
        self.assertEqual(result["memory"], {"enable_memory": True, "enable_history_memory": False})
        self.assertEqual(len(result["tools_enabled"]), 9)
        self.assertEqual(result["transport"], "cookie")
        self.assertEqual(qwen_client.get_personalization_settings.await_count, 1)
        self.assertEqual(qwen_client.get_personalization_settings.await_args.args[0].email, "alice@example.com")

    async def test_single_put_sanitizes_payload_before_upstream_update(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = SimpleNamespace(accounts=[account])
        qwen_client = SimpleNamespace(
            update_personalization_settings=AsyncMock(return_value={"email": account.email, "status": "success", "transport": "token", "http_status": 200})
        )
        request = _make_request(pool, qwen_client)
        payload = SimpleNamespace(
            memory={"enable_memory": 1, "enable_history_memory": 0, "ignored": True},
            tools_enabled={f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
            emails=["ignored@example.com"],
            extra_block={"ignored": True},
        )

        result = await admin.update_account_personalization("alice@example.com", payload, request)

        self.assertTrue(result["ok"])
        called_payload = qwen_client.update_personalization_settings.await_args.args[1]
        self.assertEqual(called_payload["memory"], {"enable_memory": True, "enable_history_memory": False})
        self.assertEqual(len(called_payload["tools_enabled"]), 9)
        self.assertNotIn("emails", called_payload)
        self.assertNotIn("extra_block", called_payload)
        self.assertEqual(result["transport"], "token")

    async def test_single_get_returns_404_when_account_is_missing(self) -> None:
        pool = SimpleNamespace(accounts=[])
        qwen_client = SimpleNamespace(get_personalization_settings=AsyncMock())
        request = _make_request(pool, qwen_client)

        with self.assertRaises(HTTPException) as ctx:
            await admin.get_account_personalization("missing@example.com", request)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(qwen_client.get_personalization_settings.await_count, 0)

    async def test_single_put_rejects_missing_credentials(self) -> None:
        account = Account(email="alice@example.com")
        pool = SimpleNamespace(accounts=[account])
        qwen_client = SimpleNamespace(update_personalization_settings=AsyncMock())
        request = _make_request(pool, qwen_client)
        payload = SimpleNamespace(
            memory={"enable_memory": True, "enable_history_memory": False},
            tools_enabled={f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
        )

        with self.assertRaises(HTTPException) as ctx:
            await admin.update_account_personalization("alice@example.com", payload, request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "missing_credentials")
        self.assertEqual(qwen_client.update_personalization_settings.await_count, 0)

    async def test_batch_put_normalizes_emails_and_skips_missing_accounts(self) -> None:
        available = Account(email="alice@example.com", token="token-1")
        no_creds = Account(email="nocreds@example.com")
        pool = SimpleNamespace(accounts=[available, no_creds])
        qwen_client = SimpleNamespace(
            update_personalization_settings=AsyncMock(
                return_value={"email": available.email, "status": "success", "transport": "token", "http_status": 200}
            )
        )
        request = _make_request(pool, qwen_client)
        payload = SimpleNamespace(
            emails=[" alice@example.com ", "missing@example.com", "alice@example.com", "", "   ", "nocreds@example.com"],
            memory={"enable_memory": True, "enable_history_memory": False},
            tools_enabled={f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
        )

        result = await admin.update_accounts_personalization(payload, request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], {"success": 1, "failed": 0, "skipped": 2})
        self.assertEqual(
            [item["email"] for item in result["results"]],
            ["alice@example.com", "missing@example.com", "nocreds@example.com"],
        )
        self.assertEqual(
            [call.args[0].email for call in qwen_client.update_personalization_settings.await_args_list],
            ["alice@example.com"],
        )
        self.assertEqual(len(qwen_client.update_personalization_settings.await_args.args[1]["tools_enabled"]), 9)

    async def test_batch_put_rejects_empty_selection(self) -> None:
        pool = SimpleNamespace(accounts=[])
        qwen_client = SimpleNamespace(update_personalization_settings=AsyncMock())
        request = _make_request(pool, qwen_client)
        payload = SimpleNamespace(emails=[], memory={"enable_memory": True, "enable_history_memory": False}, tools_enabled={f"tool_{index}": True for index in range(1, 10)})

        with self.assertRaises(HTTPException) as ctx:
            await admin.update_accounts_personalization(payload, request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(qwen_client.update_personalization_settings.await_count, 0)


if __name__ == "__main__":
    unittest.main()
