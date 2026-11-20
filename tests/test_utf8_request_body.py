"""请求体发原始 UTF-8（不走 `json=` 的 \\uXXXX 转义）与按字节的本地拦截。

全部离线：上游会话用 fake session 顶替，不发任何真实请求。
"""

import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

try:
    import oss2  # noqa: F401
except ImportError:
    fake_oss2 = types.ModuleType("oss2")

    class StsAuth:
        pass

    class Bucket:
        pass

    fake_oss2.StsAuth = StsAuth
    fake_oss2.Bucket = Bucket
    sys.modules["oss2"] = fake_oss2

from backend.core.account_pool import Account
from backend.services.exit_pool import ExitPool
from backend.services.qwen_client import QwenClient, encode_json_body
from backend.services.upstream_file_uploader import UpstreamFileUploader

_SETTINGS = "backend.services.qwen_client.settings."
_CHINESE_PAYLOAD = {
    "stream": True,
    "messages": [{"role": "user", "content": "你好，世界！请总结这段中文。"}],
}


class _FakeResponse:
    status_code = 200
    text = '{"ok": true}'


class _FakeRequestSession:
    """记录 `request()` 收到的全部关键字参数（用于断言有没有带请求体）。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        return _FakeResponse()


class _FakeCurlStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_content(self):
        yield b"data: {\"text\":\"ok\"}\n\n"


class _FakeCurlStreamSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False

    def stream(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        return _FakeCurlStreamResponse()

    async def close(self):
        self.closed = True


class _FakeHttpxStreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        yield b"data: {\"text\":\"ok\"}\n\n"


class _FakeAsyncClient:
    instances: list["_FakeAsyncClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict] = []
        _FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        return _FakeHttpxStreamResponse()


def _account() -> Account:
    return Account(email="alice@example.com", token="token-1")


async def _run_stream(payload: dict, session: _FakeCurlStreamSession, **limits: int):
    """跑一次 curl_cffi 主流式分支，返回 `(items, new_session_mock)`。"""
    account = _account()
    client = QwenClient(SimpleNamespace())
    patches = [patch(_SETTINGS + name, value) for name, value in limits.items()]
    for p in patches:
        p.start()
    try:
        with (
            patch(_SETTINGS + "QWEN_CHAT_TRANSPORT_GO_LIKE_HTTP", False),
            patch("backend.services.qwen_client.httpx.AsyncClient", _FakeAsyncClient),
            patch("backend.services.qwen_client.new_session", return_value=session) as new_session,
            patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)) as get_session,
            patch("backend.services.qwen_client.WafCookieManager") as waf_cls,
        ):
            waf_cls.get_instance.return_value.get_cookies = AsyncMock(return_value="")
            items = [item async for item in client.stream_chat_once(account.token, "chat-1", payload, account=account)]
    finally:
        for p in patches:
            p.stop()
    return items, new_session, get_session


class EncodeJsonBodyTests(unittest.TestCase):
    def test_none_stays_none(self) -> None:
        self.assertIsNone(encode_json_body(None))

    def test_chinese_is_raw_utf8_and_compact(self) -> None:
        encoded = encode_json_body({"a": "中文", "b": [1, 2]})
        self.assertEqual(encoded, '{"a":"中文","b":[1,2]}'.encode("utf-8"))
        self.assertNotIn(b"\\u", encoded)

    def test_empty_dict_is_still_a_body(self) -> None:
        self.assertEqual(encode_json_body({}), b"{}")


class Utf8RequestBodyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeAsyncClient.instances = []

    # (a) 含中文 payload：发出的字节不含 \u 转义，解码后等于原 payload
    async def test_curl_stream_sends_raw_utf8_bytes(self) -> None:
        session = _FakeCurlStreamSession()
        items, _new_session, _get_session = await _run_stream(_CHINESE_PAYLOAD, session)

        self.assertEqual(items[-1], {"status": "streamed"})
        self.assertEqual(len(session.calls), 1)
        kwargs = session.calls[0]["kwargs"]
        self.assertNotIn("json", kwargs)
        body = kwargs["data"]
        self.assertIsInstance(body, bytes)
        self.assertNotIn(b"\\u", body)
        self.assertIn("你好，世界".encode("utf-8"), body)
        self.assertEqual(json.loads(body.decode("utf-8")), _CHINESE_PAYLOAD)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    async def test_httpx_stream_sends_raw_utf8_bytes_via_content(self) -> None:
        account = _account()
        client = QwenClient(SimpleNamespace())
        with (
            patch(_SETTINGS + "QWEN_CHAT_TRANSPORT_GO_LIKE_HTTP", True),
            patch("backend.services.qwen_client.httpx.AsyncClient", _FakeAsyncClient),
            patch("backend.services.qwen_client.new_session") as new_session,
            patch("backend.services.qwen_client.WafCookieManager") as waf_cls,
        ):
            waf_cls.get_instance.return_value.get_cookies = AsyncMock(return_value="")
            items = [
                item
                async for item in client.stream_chat_once(account.token, "chat-1", _CHINESE_PAYLOAD, account=account)
            ]

        self.assertFalse(new_session.called)
        self.assertEqual(items[-1], {"status": "streamed"})
        kwargs = _FakeAsyncClient.instances[0].calls[0]["kwargs"]
        self.assertNotIn("json", kwargs)
        self.assertNotIn("data", kwargs)
        body = kwargs["content"]
        self.assertIsInstance(body, bytes)
        self.assertNotIn(b"\\u", body)
        self.assertEqual(json.loads(body.decode("utf-8")), _CHINESE_PAYLOAD)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    async def test_request_json_sends_raw_utf8_bytes(self) -> None:
        client = QwenClient(SimpleNamespace())
        session = _FakeRequestSession()
        body = {"title": "中文标题"}
        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)):
            result = await client._request_json("POST", "/api/v2/chats/new", "token-1", body, account=_account())

        self.assertEqual(result["status"], 200)
        kwargs = session.calls[0]["kwargs"]
        self.assertNotIn("json", kwargs)
        self.assertNotIn(b"\\u", kwargs["data"])
        self.assertEqual(json.loads(kwargs["data"].decode("utf-8")), body)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    # (b) 字节超限：413，且没有发起上游请求
    async def test_body_over_byte_limit_is_rejected_without_upstream_call(self) -> None:
        session = _FakeCurlStreamSession()
        size = len(encode_json_body(_CHINESE_PAYLOAD))
        items, new_session, get_session = await _run_stream(
            _CHINESE_PAYLOAD, session, WAF_MAX_INLINE_BODY_BYTES=size - 1, WAF_MAX_INLINE_PROMPT_CHARS=0
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], 413)
        error = json.loads(items[0]["body"])["error"]
        self.assertEqual(error["code"], "prompt_too_large")
        self.assertTrue(error["type"])
        self.assertIn("WAF_MAX_INLINE_BODY_BYTES", error["message"])
        self.assertIn(f"{size} bytes", error["message"])
        self.assertIn(str(size - 1), error["message"])
        self.assertEqual(session.calls, [])
        self.assertFalse(new_session.called)
        self.assertFalse(get_session.called)
        self.assertEqual(_FakeAsyncClient.instances, [])

    async def test_body_exactly_at_byte_limit_passes(self) -> None:
        session = _FakeCurlStreamSession()
        size = len(encode_json_body(_CHINESE_PAYLOAD))
        items, _new_session, _get_session = await _run_stream(
            _CHINESE_PAYLOAD, session, WAF_MAX_INLINE_BODY_BYTES=size, WAF_MAX_INLINE_PROMPT_CHARS=0
        )

        self.assertEqual(items[-1], {"status": "streamed"})
        self.assertEqual(len(session.calls[0]["kwargs"]["data"]), size)

    # 边界（QA 补充）：用默认上限 76000 的真实量级，恰好等于放行、多 1 字节拦截
    @staticmethod
    def _payload_of_exact_bytes(target: int) -> dict:
        payload = {"stream": True, "messages": [{"role": "user", "content": "中" * 20000}]}
        pad = target - len(encode_json_body(payload))
        payload["messages"][0]["content"] += "a" * pad
        assert len(encode_json_body(payload)) == target
        return payload

    async def test_body_of_exactly_76000_bytes_passes_default_sized_limit(self) -> None:
        session = _FakeCurlStreamSession()
        payload = self._payload_of_exact_bytes(76000)
        items, _new_session, _get_session = await _run_stream(
            payload, session, WAF_MAX_INLINE_BODY_BYTES=76000, WAF_MAX_INLINE_PROMPT_CHARS=0
        )

        self.assertEqual(items[-1], {"status": "streamed"})
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0]["kwargs"]["data"], encode_json_body(payload))

    async def test_body_one_byte_over_limit_is_rejected(self) -> None:
        session = _FakeCurlStreamSession()
        payload = self._payload_of_exact_bytes(76001)
        items, new_session, get_session = await _run_stream(
            payload, session, WAF_MAX_INLINE_BODY_BYTES=76000, WAF_MAX_INLINE_PROMPT_CHARS=0
        )

        self.assertEqual([item["status"] for item in items], [413])
        error = json.loads(items[0]["body"])["error"]
        self.assertEqual(error["code"], "prompt_too_large")
        self.assertIn("76001 bytes > WAF_MAX_INLINE_BODY_BYTES=76000", error["message"])
        self.assertEqual(session.calls, [])
        self.assertFalse(new_session.called)
        self.assertFalse(get_session.called)

    async def test_limit_counts_utf8_bytes_not_characters(self) -> None:
        """同样字符数，中文按 3 字节/字计：上限卡在字符数与字节数之间时必须拦截。"""
        session = _FakeCurlStreamSession()
        chars = len(json.dumps(_CHINESE_PAYLOAD, ensure_ascii=False, separators=(",", ":")))
        size = len(encode_json_body(_CHINESE_PAYLOAD))
        self.assertLess(chars, size)
        items, _new_session, _get_session = await _run_stream(
            _CHINESE_PAYLOAD, session, WAF_MAX_INLINE_BODY_BYTES=chars, WAF_MAX_INLINE_PROMPT_CHARS=0
        )

        self.assertEqual([item["status"] for item in items], [413])
        self.assertEqual(session.calls, [])

    async def test_prompt_char_limit_at_exact_length_passes(self) -> None:
        session = _FakeCurlStreamSession()
        chars = len(_CHINESE_PAYLOAD["messages"][0]["content"])
        items, _new_session, _get_session = await _run_stream(
            _CHINESE_PAYLOAD, session, WAF_MAX_INLINE_BODY_BYTES=0, WAF_MAX_INLINE_PROMPT_CHARS=chars
        )

        self.assertEqual(items[-1], {"status": "streamed"})
        self.assertEqual(len(session.calls), 1)

    # (c) WAF_MAX_INLINE_BODY_BYTES=0：不拦截
    async def test_zero_byte_limit_disables_the_check(self) -> None:
        session = _FakeCurlStreamSession()
        payload = {"stream": True, "messages": [{"role": "user", "content": "长" * 60000}]}
        self.assertGreater(len(encode_json_body(payload)), 76000)
        items, _new_session, _get_session = await _run_stream(
            payload, session, WAF_MAX_INLINE_BODY_BYTES=0, WAF_MAX_INLINE_PROMPT_CHARS=0
        )

        self.assertEqual(items[-1], {"status": "streamed"})
        self.assertEqual(json.loads(session.calls[0]["kwargs"]["data"].decode("utf-8")), payload)

    # (d) 显式设置 WAF_MAX_INLINE_PROMPT_CHARS 仍然生效
    async def test_explicit_prompt_char_limit_still_rejects(self) -> None:
        session = _FakeCurlStreamSession()
        chars = len(_CHINESE_PAYLOAD["messages"][0]["content"])
        items, new_session, _get_session = await _run_stream(
            _CHINESE_PAYLOAD, session, WAF_MAX_INLINE_BODY_BYTES=0, WAF_MAX_INLINE_PROMPT_CHARS=chars - 1
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], 413)
        error = json.loads(items[0]["body"])["error"]
        self.assertEqual(error["code"], "prompt_too_large")
        self.assertIn("WAF_MAX_INLINE_PROMPT_CHARS", error["message"])
        self.assertIn(f"{chars} chars", error["message"])
        self.assertIn(str(chars - 1), error["message"])
        self.assertEqual(session.calls, [])
        self.assertFalse(new_session.called)

    async def test_default_prompt_char_limit_is_disabled(self) -> None:
        from backend.core.config import settings

        self.assertEqual(int(settings.WAF_MAX_INLINE_PROMPT_CHARS), 0)
        self.assertEqual(int(settings.WAF_MAX_INLINE_BODY_BYTES), 76000)

    # (e) body 为 None：不带请求体（不得发出 `null`）
    async def test_request_json_without_body_sends_no_body(self) -> None:
        client = QwenClient(SimpleNamespace())
        session = _FakeRequestSession()
        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)):
            result = await client._request_json("GET", "/api/v2/chats/", "token-1", account=_account())

        self.assertEqual(result["status"], 200)
        kwargs = session.calls[0]["kwargs"]
        self.assertNotIn("data", kwargs)
        self.assertNotIn("json", kwargs)
        self.assertNotIn("content", kwargs)

    async def test_request_raw_json_without_body_sends_no_body(self) -> None:
        client = QwenClient(SimpleNamespace())
        session = _FakeRequestSession()
        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)):
            result = await client._request_raw_json("DELETE", "/api/v2/chats/", {"Accept": "*/*"}, account=_account())

        self.assertEqual(result["status"], 200)
        kwargs = session.calls[0]["kwargs"]
        self.assertNotIn("data", kwargs)
        self.assertNotIn("json", kwargs)


class _FakePostSession:
    """记录 `post()` 收到的全部关键字参数（出口池取 STS 用）。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []
        self.closed = False

    async def post(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        return SimpleNamespace(status_code=200, text=self.text)

    async def close(self) -> None:
        self.closed = True


class Utf8SecondaryPathTests(unittest.IsolatedAsyncioTestCase):
    """QA 补充：工程师改了但原先没有任何断言覆盖的两处发送点。"""

    # qwen_client.py `_handle_punish_and_retry`：解完滑块后的重试请求
    async def test_punish_retry_sends_raw_utf8_bytes(self) -> None:
        account = _account()
        client = QwenClient(SimpleNamespace())
        session = _FakeCurlStreamSession()
        with (
            patch("backend.services.qwen_client.new_session", return_value=session) as new_session,
            patch("backend.services.qwen_client.CaptchaSolver") as solver_cls,
            patch("backend.services.qwen_client.WafCookieManager"),
        ):
            solver_cls.get_instance.return_value.solve_punish = AsyncMock(return_value={"x5sec": "pass"})
            items = [
                item
                async for item in client._handle_punish_and_retry(
                    account.token, "chat-1", _CHINESE_PAYLOAD, account, "<html>punish</html>", 30.0
                )
            ]

        self.assertTrue(new_session.called)
        self.assertEqual(items[-1], {"status": "streamed"})
        self.assertEqual(len(session.calls), 1)
        kwargs = session.calls[0]["kwargs"]
        self.assertNotIn("json", kwargs)
        self.assertEqual(kwargs["data"], encode_json_body(_CHINESE_PAYLOAD))
        self.assertNotIn(b"\\u", kwargs["data"])
        self.assertEqual(json.loads(kwargs["data"].decode("utf-8")), _CHINESE_PAYLOAD)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertTrue(session.closed)

    # upstream_file_uploader.py：经出口取 STS，请求体带（可能是中文的）文件名
    async def test_sts_via_exit_sends_raw_utf8_filename(self) -> None:
        sts_data = {
            "file_id": "file_ok",
            "file_path": "user/object.bin",
            "bucketname": "qwen-webui-prod",
            "endpoint": "qwen-webui-prod.oss-accelerate.aliyuncs.com",
            "region": "oss-cn-hangzhou",
            "access_key_id": "ak",
            "access_key_secret": "sk",
            "security_token": "st",
            "file_url": "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/user/object.bin"
            "?x-oss-expires=300&x-oss-signature=sig",
        }
        session = _FakePostSession(json.dumps({"success": True, "data": sts_data}))
        exit_url = "socks5h://Default.g@10.6.6.114:2260"

        async def factory(proxy_url: str) -> _FakePostSession:
            self.assertEqual(proxy_url, exit_url)
            return session

        uploader = UpstreamFileUploader(
            QwenClient(SimpleNamespace()),
            SimpleNamespace(
                CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1,
                UPLOAD_EXIT_POOL="",
                UPLOAD_VIA_EXIT="auto",
                UPLOAD_EXIT_COOLDOWN_SECONDS=600,
                UPLOAD_EXIT_MAX_FAILURES=2,
                UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS=10,
            ),
            exit_pool=ExitPool([exit_url], max_failures=5),
            exit_session_factory=factory,
        )

        sts = await uploader.request_sts(_account(), "季度报告（终稿）.mp4", b"x" * 32, "video/mp4")

        self.assertEqual(sts["file_id"], "file_ok")
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0]["url"].endswith("/api/v2/files/getstsToken"))
        kwargs = session.calls[0]["kwargs"]
        self.assertNotIn("json", kwargs)
        body = kwargs["data"]
        self.assertIsInstance(body, bytes)
        self.assertNotIn(b"\\u", body)
        self.assertIn("季度报告（终稿）".encode("utf-8"), body)
        sent = json.loads(body.decode("utf-8"))
        self.assertEqual(sent["filename"], "季度报告（终稿）.mp4")
        self.assertEqual(sent["filesize"], "32")
        self.assertEqual(body, encode_json_body(sent))
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")


if __name__ == "__main__":
    unittest.main()
