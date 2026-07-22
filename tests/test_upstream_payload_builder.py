import time
import unittest
from unittest.mock import patch

from backend.upstream.payload_builder import build_chat_payload


class UpstreamPayloadBuilderTests(unittest.TestCase):
    def test_chat_payload_uses_go_style_hex_message_ids(self) -> None:
        payload = build_chat_payload("chat-1", "qwen3.7-plus", "hello")
        message = payload["messages"][0]

        self.assertRegex(message["fid"], r"^[0-9a-f]{32}$")
        self.assertRegex(message["childrenIds"][0], r"^[0-9a-f]{32}$")
        self.assertNotIn("-", message["fid"])
        self.assertNotIn("-", message["childrenIds"][0])

    def test_chat_payload_timestamp_is_milliseconds(self) -> None:
        """与 create_chat / 浏览器 softs 对齐：timestamp 必须是毫秒，不能是秒。"""
        fixed = 1718000000.5  # 2024 秒级 + 小数

        with patch("backend.upstream.payload_builder.time.time", return_value=fixed):
            payload = build_chat_payload("chat-1", "qwen3.7-plus", "hello")

        expected_ms = int(fixed * 1000)
        self.assertEqual(payload["timestamp"], expected_ms)
        self.assertEqual(payload["messages"][0]["timestamp"], expected_ms)
        # 秒级时间戳约 10 位，毫秒约 13 位
        self.assertGreaterEqual(payload["timestamp"], 1_000_000_000_000)


if __name__ == "__main__":
    unittest.main()
