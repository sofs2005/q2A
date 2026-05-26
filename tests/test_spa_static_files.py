import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.core.spa_static_files import SPAStaticFiles


class SPAStaticFilesTests(unittest.TestCase):
    def _create_client(self) -> TestClient:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        dist_dir = Path(self.tmpdir.name)
        (dist_dir / "index.html").write_text("<div id=\"root\"></div>", encoding="utf-8")

        app = FastAPI()

        @app.get("/api/status")
        async def api_status() -> dict[str, str]:
            return {"status": "ok"}

        app.mount("/", SPAStaticFiles(directory=dist_dir, html=True), name="frontend")
        return TestClient(app)

    def test_frontend_route_falls_back_to_index_html(self) -> None:
        client = self._create_client()

        response = client.get("/accounts/123")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(response.text, "<div id=\"root\"></div>")

    def test_missing_api_paths_still_return_404(self) -> None:
        client = self._create_client()

        for path in ("/api/missing", "/v1/missing"):
            with self.subTest(path=path):
                response = client.get(path)

                self.assertEqual(response.status_code, 404)

    def test_existing_api_route_mounted_before_frontend_still_wins(self) -> None:
        client = self._create_client()

        response = client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_missing_asset_and_dotted_file_still_return_404(self) -> None:
        client = self._create_client()

        for path in ("/assets/missing.js", "/favicon.ico", "/accounts/123.json"):
            with self.subTest(path=path):
                response = client.get(path)

                self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
