import re
import unittest

from backend.upstream.payload_builder import build_chat_payload


class UpstreamPayloadBuilderTests(unittest.TestCase):
    def test_chat_payload_uses_go_style_hex_message_ids(self) -> None:
        payload = build_chat_payload("chat-1", "qwen3.7-plus", "hello")
        message = payload["messages"][0]

        self.assertRegex(message["fid"], r"^[0-9a-f]{32}$")
        self.assertRegex(message["childrenIds"][0], r"^[0-9a-f]{32}$")
        self.assertNotIn("-", message["fid"])
        self.assertNotIn("-", message["childrenIds"][0])


if __name__ == "__main__":
    unittest.main()
