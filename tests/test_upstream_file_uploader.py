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
from backend.services.upstream_file_uploader import UpstreamFileUploader, FileUploadRateLimitedError


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict | None, Account | None]] = []

    async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
        del method, token, timeout
        self.requests.append((path, body, account))
        if path == "/api/v2/files/getstsToken":
            return {
                "status": 200,
                "body": json.dumps(
                    {
                        "data": {
                            "file_id": "file_1",
                            "file_path": "user/object_inline-image.png",
                            "bucketname": "qwen-webui-prod",
                            "endpoint": "qwen-webui-prod.oss-accelerate.aliyuncs.com",
                            "region": "oss-cn-hangzhou",
                            "access_key_id": "ak",
                            "access_key_secret": "sk",
                            "security_token": "st",
                            "file_url": "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/user/object_inline-image.png?x-oss-signature=test-signature",
                        }
                    }
                ),
            }
        if path == "/api/v2/files/parse":
            return {"status": 200, "body": json.dumps({"ok": True})}
        if path == "/api/v2/files/parse/status":
            return {"status": 200, "body": json.dumps({"data": [{"status": "success"}]})}
        raise AssertionError(f"unexpected path: {path}")


class _FakeBucket:
    endpoints = []

    def __init__(self, auth, endpoint, bucketname, region=None):
        del auth, bucketname, region
        self.endpoint = endpoint
        self.__class__.endpoints.append(endpoint)

    def put_object(self, key, raw, headers=None):
        del key, raw, headers
        if "oss-accelerate.aliyuncs.com" in self.endpoint:
            raise RuntimeError("NameResolutionError: Failed to resolve oss accelerate host")
        return types.SimpleNamespace(status=200)


class _RateLimitedStsClient:
    """getstsToken 返回每日上传配额超限（RateLimited）的上游响应。"""

    def __init__(self, *, num: int = 19, include_num: bool = True) -> None:
        self.num = num
        self.include_num = include_num
        self.requests: list[tuple[str, dict | None, Account | None]] = []

    async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
        del method, token, timeout
        self.requests.append((path, body, account))
        if path == "/api/v2/files/getstsToken":
            data = {
                "code": "RateLimited",
                "details": "You've reached the upper limit for today's usage.",
            }
            if self.include_num:
                data["num"] = self.num
            return {"status": 200, "body": json.dumps({"success": False, "data": data})}
        raise AssertionError(f"unexpected path: {path}")


class UpstreamFileUploaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_from_accelerate_to_regional_endpoint_on_dns_failure(self) -> None:
        settings = types.SimpleNamespace(CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1)
        fake_client = _FakeClient()
        uploader = UpstreamFileUploader(fake_client, settings)
        local_file = None
        _FakeBucket.endpoints = []
        account = Account(email="alice@example.com", token="tok")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
            handle.write(b"png-bytes")
            local_file = handle.name

        local_meta = {
            "filename": "inline-image.png",
            "path": local_file,
            "content_type": "image/png",
            "sha256": "abc",
        }

        with patch("backend.services.upstream_file_uploader.oss2.StsAuth", return_value=object()), \
             patch("backend.services.upstream_file_uploader.oss2.Bucket", _FakeBucket):
            result = await uploader.upload_local_file(account, local_meta)

        self.assertEqual(
            _FakeBucket.endpoints,
            [
                "https://qwen-webui-prod.oss-accelerate.aliyuncs.com",
                "https://qwen-webui-prod.oss-cn-hangzhou.aliyuncs.com",
            ],
        )
        self.assertEqual(
            result["remote_ref"]["url"],
            "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/user/object_inline-image.png?x-oss-signature=test-signature",
        )
        self.assertEqual(fake_client.requests[0][0], "/api/v2/files/getstsToken")
        self.assertEqual(fake_client.requests[0][1]["filetype"], "image")
        self.assertIs(fake_client.requests[0][2], account)
        self.assertEqual([path for path, _, _ in fake_client.requests], ["/api/v2/files/getstsToken"])
        self.assertEqual(result["remote_ref"]["type"], "image")
        self.assertEqual(result["remote_ref"]["showType"], "image")
        self.assertEqual(result["remote_ref"]["file_class"], "vision")

        Path(local_file).unlink(missing_ok=True)

    async def test_getsts_ratelimited_raises_with_retry_after_from_num(self) -> None:
        """getstsToken 命中每日上传配额时，抛 FileUploadRateLimitedError 且 retry_after = num 小时。"""
        settings = types.SimpleNamespace(CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1)
        fake_client = _RateLimitedStsClient(num=19)
        uploader = UpstreamFileUploader(fake_client, settings)
        account = Account(email="alice@example.com", token="tok")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as handle:
            handle.write(b"context-bytes")
            local_file = handle.name

        local_meta = {
            "filename": "context.txt",
            "path": local_file,
            "content_type": "text/plain",
            "sha256": "abc",
        }

        try:
            with self.assertRaises(FileUploadRateLimitedError) as ctx:
                await uploader.upload_local_file(account, local_meta)
            self.assertEqual(ctx.exception.retry_after_seconds, 19 * 3600.0)
        finally:
            Path(local_file).unlink(missing_ok=True)

    async def test_getsts_ratelimited_without_num_uses_conservative_retry(self) -> None:
        """RateLimited 但缺 num 时，仍给保守的正数封禁窗口，避免立刻重撞额度。"""
        settings = types.SimpleNamespace(CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1)
        fake_client = _RateLimitedStsClient(include_num=False)
        uploader = UpstreamFileUploader(fake_client, settings)
        account = Account(email="alice@example.com", token="tok")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as handle:
            handle.write(b"context-bytes")
            local_file = handle.name

        local_meta = {
            "filename": "context.txt",
            "path": local_file,
            "content_type": "text/plain",
            "sha256": "abc",
        }

        try:
            with self.assertRaises(FileUploadRateLimitedError) as ctx:
                await uploader.upload_local_file(account, local_meta)
            self.assertGreater(ctx.exception.retry_after_seconds, 0)
        finally:
            Path(local_file).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
