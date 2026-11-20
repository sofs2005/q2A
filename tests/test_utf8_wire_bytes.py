"""真实传输层验证：curl_cffi 实际写到线上的请求体是不是原始 UTF-8。

其它测试都用 fake session，只能证明「传给 curl_cffi 的参数是对的」。这里用真实的
`curl_cffi.requests.AsyncSession(impersonate="chrome")` 向本机回环上的一次性
HTTP 服务器发请求，由服务器记录收到的原始字节。只连 127.0.0.1，不出网。
"""

import importlib
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.services.qwen_client import encode_json_body

_PAYLOAD = {
    "stream": True,
    "messages": [{"role": "user", "content": "你好，世界！请总结这段中文。" * 20}],
}
_HEADERS = {"Content-Type": "application/json"}


def _load_real_async_session():
    """取真实的 curl_cffi AsyncSession。

    别的测试文件在 curl_cffi 尚未导入时会往 `sys.modules` 塞一个空壳桩模块；
    这里临时挪开桩、导入真实包，再把桩放回去，不影响其它测试。
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "curl_cffi" or name.startswith("curl_cffi.")
    }
    stubbed = any(getattr(module, "__file__", None) is None for module in saved.values())
    if not stubbed:
        return importlib.import_module("curl_cffi.requests").AsyncSession
    for name in saved:
        del sys.modules[name]
    try:
        return importlib.import_module("curl_cffi.requests").AsyncSession
    finally:
        for name in [n for n in sys.modules if n == "curl_cffi" or n.startswith("curl_cffi.")]:
            del sys.modules[name]
        sys.modules.update(saved)


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 (http.server 约定的方法名)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.received.append(
            {
                "path": self.path,
                "body": body,
                "content_type": self.headers.get("Content-Type"),
                "content_length": self.headers.get("Content-Length"),
                "transfer_encoding": self.headers.get("Transfer-Encoding"),
            }
        )
        reply = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args) -> None:
        del args


class Utf8WireBytesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        try:
            self.async_session_cls = _load_real_async_session()
        except ImportError as exc:
            self.skipTest(f"真实 curl_cffi 不可用: {exc}")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self.server.received = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    async def _post(self, path: str, **body_kwargs) -> dict:
        session = self.async_session_cls(impersonate="chrome")
        try:
            resp = await session.post(
                f"{self.base_url}{path}", headers=dict(_HEADERS), timeout=10, **body_kwargs
            )
        finally:
            await session.close()
        self.assertEqual(resp.status_code, 200)
        matches = [r for r in self.server.received if r["path"] == path]
        self.assertEqual(len(matches), 1)
        return matches[0]

    async def test_data_bytes_reach_the_wire_as_raw_utf8(self) -> None:
        expected = encode_json_body(_PAYLOAD)

        received = await self._post("/data", data=expected)

        self.assertEqual(received["body"], expected)
        self.assertNotIn(b"\\u", received["body"])
        self.assertIn("你好，世界".encode("utf-8"), received["body"])
        self.assertEqual(json.loads(received["body"].decode("utf-8")), _PAYLOAD)
        self.assertEqual(received["content_length"], str(len(expected)))
        self.assertIsNone(received["transfer_encoding"])
        self.assertEqual(received["content_type"], "application/json")

    async def test_json_kwarg_escapes_and_inflates_the_body(self) -> None:
        """对照组：证明本测试确实能区分 `data=` 与 `json=` 两种行为。"""
        expected = encode_json_body(_PAYLOAD)

        via_data = await self._post("/data", data=expected)
        via_json = await self._post("/json", json=_PAYLOAD)

        print(
            f"\n[wire-bytes] data= {len(via_data['body'])} bytes; "
            f"json= {len(via_json['body'])} bytes"
        )
        self.assertIn(b"\\u", via_json["body"])
        self.assertNotEqual(via_json["body"], expected)
        self.assertEqual(json.loads(via_json["body"]), _PAYLOAD)
        self.assertGreater(len(via_json["body"]), len(via_data["body"]) * 1.5)

    async def test_large_body_near_default_limit_is_not_altered(self) -> None:
        payload = {"stream": True, "messages": [{"role": "user", "content": "长" * 25000}]}
        expected = encode_json_body(payload)
        self.assertGreater(len(expected), 75000)

        received = await self._post("/large", data=expected)

        self.assertEqual(received["body"], expected)
        self.assertEqual(received["content_length"], str(len(expected)))


if __name__ == "__main__":
    unittest.main()
