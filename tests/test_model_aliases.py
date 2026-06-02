import json
import unittest
from types import SimpleNamespace

from backend.api.models import _build_model_list_payload, list_models
from backend.core.config import resolve_model, settings


class ModelAliasTests(unittest.TestCase):
    def test_qwen37_plus_preview_resolves_to_invite_beta_upstream_name(self) -> None:
        self.assertEqual(
            resolve_model("qwen3.7-plus-preview"),
            "qwen-latest-series-invite-beta-v16",
        )

    def test_model_list_fallback_includes_qwen37_plus_preview_alias(self) -> None:
        payload = _build_model_list_payload()
        model_ids = {item["id"] for item in payload["data"]}

        self.assertIn("qwen3.7-plus-preview", model_ids)


class _FakeUsersDB:
    async def get(self):
        return [{"id": "test-key", "quota": 100, "used_tokens": 0}]


class _FakeQwenClient:
    def __init__(self):
        self.called = False

    async def list_models(self, token: str) -> list[dict]:
        self.called = True
        return [
            {"id": "qwen3.6-plus"},
            {"id": "qwen3.6-max-preview"},
            {"id": "qwen3.7-max"},
        ]


class ModelListEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_models_use_upstream = getattr(settings, "MODELS_USE_UPSTREAM", None)

    async def asyncTearDown(self) -> None:
        if self.original_models_use_upstream is None:
            if hasattr(settings, "MODELS_USE_UPSTREAM"):
                delattr(settings, "MODELS_USE_UPSTREAM")
        else:
            settings.MODELS_USE_UPSTREAM = self.original_models_use_upstream

    async def test_model_list_defaults_to_local_models_without_upstream_call(self) -> None:
        qwen_client = _FakeQwenClient()
        settings.MODELS_USE_UPSTREAM = False
        request = SimpleNamespace(
            headers={"Authorization": "Bearer test-key"},
            query_params={},
            app=SimpleNamespace(
                state=SimpleNamespace(
                    users_db=_FakeUsersDB(),
                    qwen_client=qwen_client,
                )
            ),
        )

        response = await list_models(request)
        payload = json.loads(response.body.decode("utf-8"))
        model_ids = {item["id"] for item in payload["data"]}

        self.assertFalse(qwen_client.called)
        self.assertIn("qwen3.7-plus-preview", model_ids)

    async def test_model_list_can_merge_upstream_models_when_enabled(self) -> None:
        qwen_client = _FakeQwenClient()
        settings.MODELS_USE_UPSTREAM = True
        request = SimpleNamespace(
            headers={"Authorization": "Bearer test-key"},
            query_params={},
            app=SimpleNamespace(
                state=SimpleNamespace(
                    users_db=_FakeUsersDB(),
                    qwen_client=qwen_client,
                )
            ),
        )

        response = await list_models(request)
        payload = json.loads(response.body.decode("utf-8"))
        model_ids = {item["id"] for item in payload["data"]}

        self.assertTrue(qwen_client.called)
        self.assertIn("qwen3.6-plus", model_ids)
        self.assertIn("qwen3.6-max-preview", model_ids)
        self.assertIn("qwen3.7-max", model_ids)
        self.assertIn("qwen3.7-plus-preview", model_ids)


if __name__ == "__main__":
    unittest.main()
