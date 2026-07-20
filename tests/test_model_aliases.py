import json
import unittest
from types import SimpleNamespace

from backend.api.models import _build_model_list_payload, list_models
from backend.core.config import MODEL_MAP, resolve_model, settings


class ModelAliasTests(unittest.TestCase):
    def test_model_map_only_keeps_required_qwen_short_aliases(self) -> None:
        self.assertEqual(
            MODEL_MAP,
            {
                "qwen-max": "qwen3.8-max-preview",
                "qwen-plus": "qwen3.7-plus",
            },
        )

    def test_qwen_short_aliases_resolve_to_current_upstream_names(self) -> None:
        self.assertEqual(resolve_model("qwen-max"), "qwen3.8-max-preview")
        self.assertEqual(resolve_model("qwen-plus"), "qwen3.7-plus")

    def test_model_list_fallback_only_includes_required_qwen_short_aliases(self) -> None:
        payload = _build_model_list_payload()
        model_ids = [item["id"] for item in payload["data"]]

        self.assertEqual(model_ids, ["qwen-max", "qwen-plus"])


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
            {"id": "qwen3.8-max-preview"},
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

    async def test_model_list_defaults_to_upstream_models_and_required_short_aliases(self) -> None:
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
        model_ids = [item["id"] for item in payload["data"]]

        self.assertTrue(qwen_client.called)
        self.assertEqual(
            model_ids,
            ["qwen3.6-plus", "qwen3.6-max-preview", "qwen3.8-max-preview", "qwen-max", "qwen-plus"],
        )

    async def test_model_list_can_disable_upstream_models(self) -> None:
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
        self.assertEqual(model_ids, {"qwen-max", "qwen-plus"})


if __name__ == "__main__":
    unittest.main()
