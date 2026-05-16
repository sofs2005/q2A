import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.admin import router
from backend.core.config import settings


class _DummyPool:
    def status(self):
        return {"total": 0}


class AdminNoBrowserTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        app.state.account_pool = _DummyPool()
        self.client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {settings.ADMIN_KEY}"}

    def test_status_reports_browser_automation_disabled(self) -> None:
        response = self.client.get("/status", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["browser_automation"]["mode"], "disabled")
        self.assertFalse(data["browser_automation"]["available"])

    def test_register_endpoint_returns_no_browser_error(self) -> None:
        response = self.client.post("/accounts/register", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["ok"])
        self.assertIn("无浏览器", data["error"])
        self.assertIn("手动添加", data["error"])

    def test_activate_endpoint_returns_no_browser_error(self) -> None:
        response = self.client.post("/accounts/user@example.com/activate", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["ok"])
        self.assertIn("无浏览器", data["error"])


if __name__ == "__main__":
    unittest.main()
