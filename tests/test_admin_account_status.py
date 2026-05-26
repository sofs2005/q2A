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
from backend.core.account_pool import Account, AccountPool


def _make_pool(*accounts: Account) -> AccountPool:
    pool = AccountPool(SimpleNamespace(save=AsyncMock()))
    pool.accounts = list(accounts)
    return pool


def _make_request(pool: AccountPool):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(account_pool=pool)))


class AdminAccountStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_disable_marks_account_disabled(self) -> None:
        account = Account(email="alice@example.com")
        pool = _make_pool(account)
        request = _make_request(pool)

        result = await admin.disable_account("alice@example.com", request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["email"], "alice@example.com")
        self.assertEqual(result["status"], "disabled")
        self.assertFalse(account.valid)
        self.assertEqual(account.status_code, "disabled")
        self.assertEqual(pool.db.save.await_count, 1)

    async def test_single_enable_marks_account_valid(self) -> None:
        account = Account(email="alice@example.com", activation_pending=True)
        account.valid = False
        account.status_code = "disabled"
        pool = _make_pool(account)
        request = _make_request(pool)

        result = await admin.enable_account("alice@example.com", request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["email"], "alice@example.com")
        self.assertEqual(result["status"], "valid")
        self.assertTrue(account.valid)
        self.assertFalse(account.activation_pending)
        self.assertEqual(account.status_code, "valid")
        self.assertEqual(pool.db.save.await_count, 1)

    async def test_batch_disable_normalizes_emails_and_skips_missing_accounts(self) -> None:
        alice = Account(email="alice@example.com")
        bob = Account(email="bob@example.com")
        bob.valid = False
        bob.status_code = "disabled"
        pool = _make_pool(alice, bob)
        request = _make_request(pool)
        payload = SimpleNamespace(emails=[" alice@example.com ", "missing@example.com", "alice@example.com", "", "bob@example.com"])

        result = await admin.disable_accounts(payload, request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], {"success": 1, "failed": 0, "skipped": 2})
        self.assertEqual([item["email"] for item in result["results"]], ["alice@example.com", "missing@example.com", "bob@example.com"])
        self.assertEqual(alice.status_code, "disabled")
        self.assertEqual(pool.db.save.await_count, 1)

    async def test_batch_enable_normalizes_emails_and_skips_missing_accounts(self) -> None:
        alice = Account(email="alice@example.com")
        alice.valid = False
        alice.status_code = "disabled"
        bob = Account(email="bob@example.com")
        pool = _make_pool(alice, bob)
        request = _make_request(pool)
        payload = SimpleNamespace(emails=["alice@example.com", "missing@example.com", "alice@example.com", "bob@example.com"])

        result = await admin.enable_accounts(payload, request)

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], {"success": 1, "failed": 0, "skipped": 2})
        self.assertEqual([item["email"] for item in result["results"]], ["alice@example.com", "missing@example.com", "bob@example.com"])
        self.assertEqual(alice.status_code, "valid")
        self.assertTrue(alice.valid)
        self.assertEqual(pool.db.save.await_count, 1)

    async def test_single_disable_returns_404_when_account_is_missing(self) -> None:
        pool = _make_pool()
        request = _make_request(pool)

        with self.assertRaises(HTTPException) as ctx:
            await admin.disable_account("missing@example.com", request)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(pool.db.save.await_count, 0)

    async def test_batch_enable_rejects_empty_selection(self) -> None:
        pool = _make_pool()
        request = _make_request(pool)
        payload = SimpleNamespace(emails=["", "   "])

        with self.assertRaises(HTTPException) as ctx:
            await admin.enable_accounts(payload, request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(pool.db.save.await_count, 0)


if __name__ == "__main__":
    unittest.main()
