import re
import unittest
from unittest.mock import patch

from backend.upstream.payload_builder import build_chat_payload


class UpstreamPayloadBuilderTests(unittest.TestCase):
    def test_chat_payload_uses_web_style_uuid_message_ids(self) -> None:
        """官网 completions 抓包：fid / childrenIds 为带横线 UUID。"""
        payload = build_chat_payload("chat-1", "qwen3.7-plus", "hello")
        message = payload["messages"][0]
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.I,
        )

        self.assertRegex(message["fid"], uuid_re)
        self.assertRegex(message["childrenIds"][0], uuid_re)
        self.assertIsNone(message.get("id"))
        self.assertEqual(message.get("model"), "")

    def test_chat_payload_timestamp_is_seconds(self) -> None:
        """completions 官网抓包用秒级 timestamp（约 10 位），不是毫秒。"""
        fixed = 1718000000.5  # 2024 秒级 + 小数

        with patch("backend.upstream.payload_builder.time.time", return_value=fixed):
            payload = build_chat_payload("chat-1", "qwen3.7-plus", "hello")

        expected_s = int(fixed)
        self.assertEqual(payload["timestamp"], expected_s)
        self.assertEqual(payload["messages"][0]["timestamp"], expected_s)
        # 秒级约 10 位，毫秒约 13 位
        self.assertLess(payload["timestamp"], 1_000_000_000_000)
        self.assertGreaterEqual(payload["timestamp"], 1_000_000_000)

    def test_feature_config_with_files_matches_official_web_subset(self) -> None:
        """带附件时 feature_config 必须收敛到官网子集，禁止 enable_tools/tool_choice 等键。"""
        files = [{"type": "file", "id": "f1", "status": "uploaded"}]
        payload = build_chat_payload(
            "chat-1",
            "qwen3.8-max-preview",
            "hello",
            has_custom_tools=True,
            files=files,
        )
        fc = payload["messages"][0]["feature_config"]

        self.assertEqual(fc.get("thinking_mode"), "Thinking")
        self.assertEqual(fc.get("auto_thinking"), False)
        self.assertEqual(fc.get("auto_search"), True)
        self.assertEqual(fc.get("thinking_format"), "summary")
        self.assertEqual(fc.get("function_calling"), False)
        for banned in ("enable_tools", "enable_function_call", "tool_choice", "code_interpreter", "plugins_enabled"):
            self.assertNotIn(banned, fc)

    def test_feature_config_without_files_strips_enable_tool_keys(self) -> None:
        """无附件时也不再塞 enable_tools / tool_choice。"""
        payload = build_chat_payload("chat-1", "qwen3.8-max-preview", "hello", has_custom_tools=True)
        fc = payload["messages"][0]["feature_config"]
        self.assertEqual(fc.get("function_calling"), False)
        for banned in ("enable_tools", "enable_function_call", "tool_choice"):
            self.assertNotIn(banned, fc)


if __name__ == "__main__":
    unittest.main()
