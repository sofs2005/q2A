import sys
import types
import unittest

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

from backend.core.account_pool import Account, AccountPool
from backend.core.browser_fingerprint import fingerprint_id_for_email


class _MemoryDB:
    def __init__(self, data):
        self.data = data
        self.saved = None

    async def load(self):
        return self.data

    async def save(self, data):
        self.saved = data
        self.data = data


class AccountFingerprintTests(unittest.IsolatedAsyncioTestCase):
    def test_account_serializes_fingerprint_id(self) -> None:
        account = Account(email="alice@example.com", fingerprint_id="chrome146_windows")

        self.assertEqual(account.to_dict()["fingerprint_id"], "chrome146_windows")

    async def test_load_assigns_missing_fingerprint_id_and_persists(self) -> None:
        db = _MemoryDB([{"email": "alice@example.com", "token": "tok"}])
        pool = AccountPool(db)

        await pool.load()

        expected = fingerprint_id_for_email("alice@example.com")
        self.assertEqual(pool.accounts[0].fingerprint_id, expected)
        self.assertIsNotNone(db.saved)
        self.assertEqual(db.saved[0]["fingerprint_id"], expected)


if __name__ == "__main__":
    unittest.main()
