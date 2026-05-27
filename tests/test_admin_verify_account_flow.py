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
            pass

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

from backend.api import admin
from backend.core.account_pool import Account


class AdminVerifyAccountFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_verify_passes_account_to_upstream_token_check(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = SimpleNamespace(accounts=[account], save=AsyncMock())
        qwen_client = SimpleNamespace(verify_token=AsyncMock(return_value=True), auth_resolver=SimpleNamespace(refresh_token=AsyncMock()))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(account_pool=pool, qwen_client=qwen_client)))

        result = await admin.verify_account("alice@example.com", request)

        self.assertTrue(result["valid"])
        self.assertEqual(qwen_client.verify_token.await_count, 1)
        self.assertEqual(qwen_client.verify_token.await_args.args[0], "token-1")
        self.assertEqual(qwen_client.verify_token.await_args.args[1], account)


if __name__ == "__main__":
    unittest.main()
