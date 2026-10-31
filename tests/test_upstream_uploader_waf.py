import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

if "oss2" not in sys.modules:
    fake_oss2 = types.ModuleType("oss2")

    class StsAuth:
        pass

    class Bucket:
        pass

    fake_oss2.StsAuth = StsAuth
    fake_oss2.Bucket = Bucket
    sys.modules["oss2"] = fake_oss2

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

from backend.core.account_pool import Account
from backend.services.upstream_file_uploader import (
    UpstreamFileUploader,
    UpstreamWafChallengeError,
    looks_like_waf_challenge,
)

# 真实抓到的 WAF 拼图验证码页面特征片段
WAF_HTML = (
    "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body>"
    "<script>window.__aliyun_waf_aa=1;</script>"
    "<div id=\"aliyunCaptcha\"></div>"
    "<script>var sceneId='19x5u7lo'; var region='sgp';</script>"
    "</body></html>"
)

_STS_DATA = {
    "file_id": "file_1",
    "file_path": "user/object.bin",
    "bucketname": "qwen-webui-prod",
    "endpoint": "qwen-webui-prod.oss-accelerate.aliyuncs.com",
    "region": "oss-cn-hangzhou",
    "access_key_id": "ak",
    "access_key_secret": "sk",
    "security_token": "st",
    "file_url": "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/user/object.bin?x-oss-expires=300&x-oss-signature=sig",
}


class _ScriptedClient:
    """按路径返回预设响应的假上游客户端。"""

    def __init__(self, responses: dict[str, dict]) -> None:
        self.responses = responses
        self.requests: list[str] = []

    async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
        del method, token, body, timeout, account
        self.requests.append(path)
        if path not in self.responses:
            raise AssertionError(f"unexpected path: {path}")
        return self.responses[path]


class _FakeBucket:
    def __init__(self, auth, endpoint, bucketname, region=None):
        del auth, endpoint, bucketname, region

    def put_object(self, key, raw, headers=None):
        del key, raw, headers
        return types.SimpleNamespace(status=200)


def _settings() -> types.SimpleNamespace:
    return types.SimpleNamespace(CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1)


class WafDetectionTests(unittest.TestCase):
    def test_detects_markers(self) -> None:
        self.assertTrue(looks_like_waf_challenge(WAF_HTML))
        self.assertTrue(looks_like_waf_challenge("<html>aliyun_waf_aa</html>"))
        self.assertTrue(looks_like_waf_challenge("<script>aliyunCaptcha.init()</script>"))
        self.assertTrue(looks_like_waf_challenge("<!doctype html><html></html>"))
        self.assertFalse(looks_like_waf_challenge('{"success": true}'))
        self.assertFalse(looks_like_waf_challenge(""))


class GetStsWafTests(unittest.IsolatedAsyncioTestCase):
    async def _upload(self, client, filename: str, content_type: str) -> None:
        uploader = UpstreamFileUploader(client, _settings())
        account = Account(email="alice@example.com", token="tok")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as handle:
            handle.write(b"bytes")
            local_path = handle.name
        local_meta = {
            "filename": filename,
            "path": local_path,
            "content_type": content_type,
            "sha256": "abc",
        }
        try:
            with patch("backend.services.upstream_file_uploader.oss2.StsAuth", return_value=object()), \
                 patch("backend.services.upstream_file_uploader.oss2.Bucket", _FakeBucket):
                await uploader.upload_local_file(account, local_meta)
        finally:
            Path(local_path).unlink(missing_ok=True)

    async def test_waf_html_raises_waf_challenge_error(self) -> None:
        client = _ScriptedClient({"/api/v2/files/getstsToken": {"status": 200, "body": WAF_HTML}})

        with self.assertRaises(UpstreamWafChallengeError) as ctx:
            await self._upload(client, "a.png", "image/png")

        self.assertIn("waf_challenge", str(ctx.exception))

    async def test_waf_html_with_error_status_raises_waf_challenge_error(self) -> None:
        client = _ScriptedClient({"/api/v2/files/getstsToken": {"status": 403, "body": WAF_HTML}})

        with self.assertRaises(UpstreamWafChallengeError) as ctx:
            await self._upload(client, "a.png", "image/png")

        self.assertIn("waf_challenge", str(ctx.exception))

    async def test_non_json_body_raises_runtime_error_not_json_decode_error(self) -> None:
        client = _ScriptedClient({"/api/v2/files/getstsToken": {"status": 200, "body": "<html>gateway</html>"}})

        with self.assertRaises(RuntimeError) as ctx:
            await self._upload(client, "a.png", "image/png")

        self.assertNotIsInstance(ctx.exception, UpstreamWafChallengeError)
        self.assertNotIn("JSONDecodeError", type(ctx.exception).__name__)
        self.assertIn("non-JSON body", str(ctx.exception))


class RefTypeAndParseGatingTests(unittest.IsolatedAsyncioTestCase):
    async def _upload(self, client, filename: str, content_type: str) -> dict:
        uploader = UpstreamFileUploader(client, _settings())
        account = Account(email="alice@example.com", token="tok")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as handle:
            handle.write(b"bytes")
            local_path = handle.name
        local_meta = {
            "filename": filename,
            "path": local_path,
            "content_type": content_type,
            "sha256": "abc",
        }
        try:
            with patch("backend.services.upstream_file_uploader.oss2.StsAuth", return_value=object()), \
                 patch("backend.services.upstream_file_uploader.oss2.Bucket", _FakeBucket):
                return await uploader.upload_local_file(account, local_meta)
        finally:
            Path(local_path).unlink(missing_ok=True)

    def _sts_ok(self) -> dict:
        return {"status": 200, "body": json.dumps({"data": _STS_DATA})}

    async def test_audio_ref_type_and_no_parse(self) -> None:
        client = _ScriptedClient({"/api/v2/files/getstsToken": self._sts_ok()})

        result = await self._upload(client, "voice.mp3", "audio/mpeg")

        self.assertEqual(client.requests, ["/api/v2/files/getstsToken"])
        self.assertEqual(result["remote_ref"]["type"], "audio")
        self.assertEqual(result["remote_ref"]["showType"], "audio")
        self.assertEqual(result["remote_ref"]["file_class"], "audio")

    async def test_video_ref_type_and_no_parse(self) -> None:
        client = _ScriptedClient({"/api/v2/files/getstsToken": self._sts_ok()})

        result = await self._upload(client, "clip.mp4", "video/mp4")

        self.assertEqual(client.requests, ["/api/v2/files/getstsToken"])
        self.assertEqual(result["remote_ref"]["type"], "video")
        self.assertEqual(result["remote_ref"]["showType"], "video")
        self.assertEqual(result["remote_ref"]["file_class"], "video")

    async def test_image_ref_type_and_no_parse(self) -> None:
        client = _ScriptedClient({"/api/v2/files/getstsToken": self._sts_ok()})

        result = await self._upload(client, "a.png", "image/png")

        self.assertEqual(client.requests, ["/api/v2/files/getstsToken"])
        self.assertEqual(result["remote_ref"]["type"], "image")
        self.assertEqual(result["remote_ref"]["showType"], "image")
        self.assertEqual(result["remote_ref"]["file_class"], "vision")
        self.assertEqual(result["remote_ref"]["file"]["meta"]["parse_meta"]["parse_status"], "success")

    async def test_document_is_parsed(self) -> None:
        client = _ScriptedClient({
            "/api/v2/files/getstsToken": self._sts_ok(),
            "/api/v2/files/parse": {"status": 200, "body": json.dumps({"ok": True})},
            "/api/v2/files/parse/status": {"status": 200, "body": json.dumps({"data": [{"status": "success"}]})},
        })

        result = await self._upload(client, "notes.pdf", "application/pdf")

        self.assertEqual(
            client.requests,
            ["/api/v2/files/getstsToken", "/api/v2/files/parse", "/api/v2/files/parse/status"],
        )
        self.assertEqual(result["remote_ref"]["type"], "file")
        self.assertEqual(result["remote_ref"]["showType"], "file")
        self.assertEqual(result["remote_ref"]["file_class"], "document")
        self.assertEqual(result["parse_status"], "success")

    async def test_parse_waf_page_raises_waf_challenge_error(self) -> None:
        client = _ScriptedClient({
            "/api/v2/files/getstsToken": self._sts_ok(),
            "/api/v2/files/parse": {"status": 200, "body": WAF_HTML},
        })

        with self.assertRaises(UpstreamWafChallengeError) as ctx:
            await self._upload(client, "notes.pdf", "application/pdf")

        self.assertIn("waf_challenge", str(ctx.exception))

    async def test_parse_status_waf_page_raises_waf_challenge_error(self) -> None:
        client = _ScriptedClient({
            "/api/v2/files/getstsToken": self._sts_ok(),
            "/api/v2/files/parse": {"status": 200, "body": json.dumps({"ok": True})},
            "/api/v2/files/parse/status": {"status": 200, "body": WAF_HTML},
        })

        with self.assertRaises(UpstreamWafChallengeError):
            await self._upload(client, "notes.pdf", "application/pdf")


if __name__ == "__main__":
    unittest.main()
