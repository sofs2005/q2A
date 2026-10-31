import os
import unittest
from unittest.mock import patch

from backend.services import model_modalities
from backend.services.model_modalities import (
    ModalityNotSupportedError,
    ensure_modalities,
    modalities_for,
    unsupported_modalities,
)


class ModalitiesForTests(unittest.TestCase):
    def test_omni_models_support_audio(self) -> None:
        for model in ("qwen3.8-omni-flash", "qwen3.5-omni-plus"):
            supported = modalities_for(model)
            self.assertIn("audio", supported, model)
            self.assertIn("image", supported, model)
            self.assertIn("video", supported, model)

    def test_plus_and_max_support_image_and_video_only(self) -> None:
        for model in ("qwen3.8-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus"):
            supported = modalities_for(model)
            self.assertEqual(supported, {"text", "image", "video"}, model)

    def test_text_only_model(self) -> None:
        self.assertEqual(modalities_for("qwen3.7-max"), {"text"})

    def test_alias_is_resolved(self) -> None:
        self.assertEqual(modalities_for("qwen-max"), {"text", "image", "video"})
        self.assertEqual(modalities_for("QWEN-OMNI"), {"text", "image", "video", "audio"})

    def test_unknown_model_uses_default(self) -> None:
        self.assertEqual(modalities_for("some-future-model"), set(model_modalities.DEFAULT_MODALITIES))
        self.assertEqual(modalities_for(""), set(model_modalities.DEFAULT_MODALITIES))

    def test_override_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"MODEL_MODALITIES_OVERRIDE": '{"some-future-model": ["text", "audio"]}'},
        ):
            self.assertEqual(modalities_for("some-future-model"), {"text", "audio"})

    def test_invalid_override_is_ignored(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODALITIES_OVERRIDE": "not-json"}):
            self.assertEqual(modalities_for("some-future-model"), set(model_modalities.DEFAULT_MODALITIES))

    def test_override_accepts_comma_string(self) -> None:
        with patch.dict(os.environ, {"MODEL_MODALITIES_OVERRIDE": '{"m1": "text,image,audio"}'}):
            self.assertEqual(modalities_for("m1"), {"text", "image", "audio"})


class UnsupportedModalitiesTests(unittest.TestCase):
    def test_reports_missing_kinds(self) -> None:
        self.assertEqual(unsupported_modalities("qwen3.7-max", {"image", "audio"}), {"image", "audio"})
        self.assertEqual(unsupported_modalities("qwen3.8-omni-flash", {"audio"}), set())

    def test_empty_request_is_empty(self) -> None:
        self.assertEqual(unsupported_modalities("qwen3.7-max", []), set())


class EnsureModalitiesTests(unittest.TestCase):
    def test_strict_mode_raises_with_actionable_message(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            with self.assertRaises(ModalityNotSupportedError) as ctx:
                ensure_modalities("qwen3.7-max", {"audio"})

        message = str(ctx.exception)
        self.assertIn("qwen3.7-max", message)
        self.assertIn("does not support", message)
        self.assertIn("audio", message)
        self.assertIn("qwen3.8-omni-flash", message)
        # 继承 ValueError，便于上层按 400 处理
        self.assertIsInstance(ctx.exception, ValueError)

    def test_strict_mode_allows_supported_kinds(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            self.assertEqual(ensure_modalities("qwen3.8-omni-flash", {"image", "audio", "video"}), set())

    def test_non_strict_mode_returns_dropped_kinds(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_STRICT": "false"}):
            with self.assertLogs("qwen2api.model_modalities", level="WARNING") as logs:
                dropped = ensure_modalities("qwen3.7-max", {"image", "audio"})

        self.assertEqual(dropped, {"image", "audio"})
        self.assertTrue(any("does not support" in line for line in logs.output))

    def test_document_is_not_gated(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            self.assertEqual(ensure_modalities("qwen3.7-max", {"document"}), set())

    def test_unknown_kinds_ignored(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            self.assertEqual(ensure_modalities("qwen3.7-max", {"", "unknown"}), set())

    def test_multiple_kinds_listed_in_message(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            with self.assertRaises(ModalityNotSupportedError) as ctx:
                ensure_modalities("qwen3.7-max", {"image", "audio"})

        message = str(ctx.exception)
        self.assertIn("image input", message)
        self.assertIn("audio input", message)


if __name__ == "__main__":
    unittest.main()
