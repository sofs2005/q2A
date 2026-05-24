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
class AdminChatClearHttpTests(unittest.TestCase):
    def _make_client(self, clear_results):
        app = FastAPI()
        app.include_router(admin.router, prefix="/api/admin")

        app.state.account_pool = SimpleNamespace(
            accounts=[
                Account(email="cookie@example.com", token="token-1", cookies="aui=1; cna=2"),
                Account(email="token@example.com", token="token-2"),
                Account(email="other@example.com", token="token-3"),
            ]
        )
        app.state.qwen_client = SimpleNamespace(clear_all_chats=AsyncMock(side_effect=clear_results))
        return TestClient(app)

    def test_delete_accounts_chats_accepts_json_body_and_clears_selected_only(self) -> None:
        client = self._make_client(
            [
                {"email": "cookie@example.com", "status": "success", "transport": "cookie"},
                {
                    "email": "token@example.com",
                    "status": "failed",
                    "transport": "token",
                    "error": "HTTP 403: forbidden",
                },
            ]
        )
        headers = {"Authorization": f"Bearer {settings.ADMIN_KEY}"}

        response = client.request(
            "DELETE",
            "/api/admin/accounts/chats",
            headers=headers,
            json={"emails": ["cookie@example.com", "token@example.com"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"], {"success": 1, "failed": 1, "skipped": 0})
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(
            [call.args[0].email for call in client.app.state.qwen_client.clear_all_chats.await_args_list],
            ["cookie@example.com", "token@example.com"],
        )

    def test_delete_accounts_chats_rejects_empty_json_selection(self) -> None:
        client = self._make_client([])
        headers = {"Authorization": f"Bearer {settings.ADMIN_KEY}"}

        response = client.request(
            "DELETE",
            "/api/admin/accounts/chats",
            headers=headers,
            json={"emails": []},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(client.app.state.qwen_client.clear_all_chats.await_count, 0)

    def test_delete_accounts_chats_rejects_whitespace_json_selection(self) -> None:
        client = self._make_client([])
        headers = {"Authorization": f"Bearer {settings.ADMIN_KEY}"}

        response = client.request(
            "DELETE",
            "/api/admin/accounts/chats",
            headers=headers,
            json={"emails": ["", "   "]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(client.app.state.qwen_client.clear_all_chats.await_count, 0)


if __name__ == "__main__":
    unittest.main()
