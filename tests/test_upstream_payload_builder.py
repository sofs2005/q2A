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


if __name__ == "__main__":
    unittest.main()
