import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

FASTAPI_RUNTIME_AVAILABLE = True
FASTAPI_RUNTIME_IMPORT_ERROR = ""

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import admin
    from backend.core.account_pool import Account
    from backend.core.config import settings
except (ModuleNotFoundError, ImportError) as exc:
    FASTAPI_RUNTIME_AVAILABLE = False
    FASTAPI_RUNTIME_IMPORT_ERROR = str(exc)
    FastAPI = None
    TestClient = None
    admin = None
    Account = None
    settings = None


@unittest.skipUnless(
    FASTAPI_RUNTIME_AVAILABLE,
    f"FastAPI runtime dependencies unavailable: {FASTAPI_RUNTIME_IMPORT_ERROR}",
)
class AdminPersonalizationHttpTests(unittest.TestCase):
    def _make_client(self, get_result, update_result, accounts):
        app = FastAPI()
        app.include_router(admin.router, prefix="/api/admin")
        app.state.account_pool = SimpleNamespace(accounts=accounts)
        app.state.qwen_client = SimpleNamespace(
            get_personalization_settings=AsyncMock(return_value=get_result),
            update_personalization_settings=AsyncMock(return_value=update_result),
        )
        return TestClient(app)

    def test_get_account_personalization_returns_modal_data(self) -> None:
        client = self._make_client(
            {
                "email": "alice@example.com",
                "status": "success",
                "transport": "cookie",
                "http_status": 200,
                "body": '{"data":{"memory":{"enable_memory":true,"enable_history_memory":false},"tools_enabled":{"tool_1":true,"tool_2":false,"tool_3":true,"tool_4":false,"tool_5":true,"tool_6":false,"tool_7":true,"tool_8":false,"tool_9":true}}}',
            },
            {"email": "alice@example.com", "status": "success", "transport": "token", "http_status": 200},
            [Account(email="alice@example.com", token="token-1", cookies="aui=1; cna=2")],
        )

        response = client.get(
            "/api/admin/accounts/alice@example.com/personalization",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["memory"], {"enable_memory": True, "enable_history_memory": False})
        self.assertEqual(len(payload["tools_enabled"]), 9)

    def test_batch_put_account_personalization_accepts_json_body(self) -> None:
        client = self._make_client(
            {"email": "alice@example.com", "status": "success", "transport": "cookie", "http_status": 200},
            {"email": "alice@example.com", "status": "success", "transport": "token", "http_status": 200},
            [
                Account(email="alice@example.com", token="token-1"),
                Account(email="bob@example.com"),
            ],
        )

        response = client.request(
            "PUT",
            "/api/admin/accounts/personalization",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={
                "emails": ["alice@example.com", "bob@example.com", "alice@example.com"],
                "memory": {"enable_memory": True, "enable_history_memory": False},
                "tools_enabled": {f"tool_{index}": index % 2 == 0 for index in range(1, 10)},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"], {"success": 1, "failed": 0, "skipped": 1})
        self.assertEqual(
            [item["email"] for item in payload["results"]],
            ["alice@example.com", "bob@example.com"],
        )
        self.assertEqual(
            [call.args[0].email for call in client.app.state.qwen_client.update_personalization_settings.await_args_list],
            ["alice@example.com"],
        )


if __name__ == "__main__":
    unittest.main()
