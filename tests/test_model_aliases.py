import unittest

from backend.api.models import _build_model_list_payload
from backend.core.config import resolve_model


class ModelAliasTests(unittest.TestCase):
    def test_qwen37_max_resolves_to_raw_upstream_name(self) -> None:
        self.assertEqual(resolve_model("qwen3.7-max"), "qwen3.7-max")

    def test_qwen37_plus_preview_resolves_to_invite_beta_upstream_name(self) -> None:
        self.assertEqual(
            resolve_model("qwen3.7-plus-preview"),
            "qwen-latest-series-invite-beta-v16",
        )

    def test_model_list_fallback_includes_qwen37_downstream_names(self) -> None:
        payload = _build_model_list_payload()
        model_ids = {item["id"] for item in payload["data"]}

        self.assertIn("qwen3.7-max", model_ids)
        self.assertIn("qwen3.7-plus-preview", model_ids)


if __name__ == "__main__":
    unittest.main()
