"""远程 ref 构造单测：对齐 findings §9 的真实上传抓包。

用一份真实 `getstsToken` 的 `data`（字段取自 §9 的 2MB 视频实测）构造 ref，断言：

- 顶层 **16** 个字段、`file` 子对象 **13** 个字段，与真实 ref 完全一致；
- 视频的 `greenNet` 是 `greening`（不是 `success`）；
- `url` 用上游下发的 CDN 短链，`id` / `file.id` 用 `file_id`；
- `type` / `showType` / `file_class` 按模态映射（图片 vision、音频 audio、视频 video、文档 file/document）。

另外验证 `UpstreamFileUploader.put_bytes()` 拆分后可以只吃外部提供的 STS
（不再自己调 `getstsToken`）—— 这是浏览器上传代理复用的关键前提。
"""

import sys
import types
import unittest
from unittest.mock import patch

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

from backend.services.sts_ref_builder import (  # noqa: E402
    REMOTE_REF_FILE_FIELDS,
    REMOTE_REF_TOP_LEVEL_FIELDS,
    build_remote_ref,
    normalize_sign_region,
    remote_file_class_from_content_type,
    remote_ref_type_from_content_type,
    sts_file_url,
    sts_user_id,
    upload_filetype_from_content_type,
)


#: 真实 `getstsToken` 的 data（§9 实测：真人过验证码后上传 2MB 视频拿到的 STS）。
REAL_STS = {
    "file_id": "079054e8-4e5e-433d-a94a-9a41f146cba2",
    "file_path": "18639e0d-5b7f-4f3c-9b8e-2f0d6a1c7e55/079054e8-4e5e-433d-a94a-9a41f146cba2_test-2m.mp4",
    "file_url": (
        "https://cdn.qwenlm.ai/18639e0d-5b7f-4f3c-9b8e-2f0d6a1c7e55/"
        "079054e8-4e5e-433d-a94a-9a41f146cba2_test-2m.mp4?key=eyJhbGciOiJIUzI1NiJ9.e30.sig"
    ),
    "bucketname": "qwen-webui-prod",
    "endpoint": "qwen-webui-prod.oss-accelerate.aliyuncs.com",
    "region": "oss-ap-southeast-1",
    "access_key_id": "STS.NTestAccessKeyId",
    "access_key_secret": "test-access-key-secret",
    "security_token": "test-security-token",
}

#: §9 从 `GET /api/v2/chats/{id}` 取回的真实 ref（字段集合与关键值）。
REAL_REF = {
    "id": "079054e8-4e5e-433d-a94a-9a41f146cba2",
    "name": "test-2m.mp4",
    "file_type": "video/mp4",
    "type": "video",
    "file_class": "video",
    "size": 2058934,
    "url": REAL_STS["file_url"],
    "file": {
        "created_at": 1793509335860,
        "data": {},
        "filename": "test-2m.mp4",
        "hash": None,
        "id": "079054e8-4e5e-433d-a94a-9a41f146cba2",
        "user_id": "18639e0d-5b7f-4f3c-9b8e-2f0d6a1c7e55",
        "meta": {"name": "test-2m.mp4", "size": 2058934, "content_type": "video/mp4"},
        "update_at": 1793509335860,
        "lastModified": 1793509207490,
        "name": "test-2m.mp4",
        "webkitRelativePath": "",
        "size": 2058934,
        "type": "video/mp4",
    },
    "collection_name": "",
    "progress": 0,
    "status": "uploaded",
    "greenNet": "greening",
    "error": "",
    "itemId": "0f2f9a7e-1111-4222-8333-444455556666",
    "showType": "video",
    "uploadTaskId": "7c1d3e5f-7777-4888-9999-aaaabbbbcccc",
}


class RemoteRefShapeTests(unittest.TestCase):
    """ref 结构必须与真实抓包逐字段一致。"""

    def test_top_level_and_file_field_counts_match_real_ref(self) -> None:
        self.assertEqual(len(REAL_REF), 16)
        self.assertEqual(len(REAL_REF["file"]), 13)
        self.assertEqual(set(REAL_REF) - {"file"}, set(REMOTE_REF_TOP_LEVEL_FIELDS) - {"file"})
        self.assertEqual(set(REAL_REF["file"]), set(REMOTE_REF_FILE_FIELDS))

    def test_video_ref_matches_real_ref(self) -> None:
        ref = build_remote_ref(
            sts=REAL_STS,
            filename="test-2m.mp4",
            content_type="video/mp4",
            size=2058934,
            now=1793509335.86,
        )
        self.assertEqual(set(ref), set(REAL_REF))
        self.assertEqual(set(ref["file"]), set(REAL_REF["file"]))
        # 关键值逐字段核对（时间戳/uuid 类字段不比较）
        for key in ("id", "name", "file_type", "type", "file_class", "size", "url", "greenNet", "showType"):
            self.assertEqual(ref[key], REAL_REF[key], msg=key)
        self.assertEqual(ref["file"]["id"], REAL_REF["file"]["id"])
        self.assertEqual(ref["file"]["name"], REAL_REF["file"]["name"])
        self.assertEqual(ref["file"]["filename"], REAL_REF["file"]["filename"])
        self.assertEqual(ref["file"]["user_id"], REAL_REF["file"]["user_id"])
        self.assertEqual(ref["file"]["size"], REAL_REF["file"]["size"])
        self.assertEqual(ref["file"]["type"], REAL_REF["file"]["type"])
        self.assertEqual(ref["file"]["webkitRelativePath"], "")
        self.assertEqual(ref["file"]["meta"]["name"], REAL_REF["file"]["meta"]["name"])
        self.assertEqual(ref["file"]["meta"]["size"], REAL_REF["file"]["meta"]["size"])
        self.assertEqual(ref["file"]["meta"]["content_type"], REAL_REF["file"]["meta"]["content_type"])
        # 视频 greenNet 必须是 greening（实测），不是 success
        self.assertEqual(ref["greenNet"], "greening")

    def test_created_at_and_update_at_use_now(self) -> None:
        ref = build_remote_ref(sts=REAL_STS, filename="a.mp4", content_type="video/mp4", size=1, now=1000.5)
        self.assertEqual(ref["file"]["created_at"], 1000500)
        self.assertEqual(ref["file"]["update_at"], 1000500)
        self.assertEqual(ref["file"]["lastModified"], 1000500)

    def test_item_id_and_upload_task_id_are_unique_uuids(self) -> None:
        first = build_remote_ref(sts=REAL_STS, filename="a.mp4", content_type="video/mp4", size=1)
        second = build_remote_ref(sts=REAL_STS, filename="a.mp4", content_type="video/mp4", size=1)
        self.assertNotEqual(first["itemId"], second["itemId"])
        self.assertNotEqual(first["uploadTaskId"], second["uploadTaskId"])
        self.assertEqual(len(first["itemId"]), 36)
        self.assertEqual(len(first["uploadTaskId"]), 36)


class RemoteRefModalityTests(unittest.TestCase):
    """模态映射：`file_class → type / showType`。"""

    def test_image_maps_to_vision(self) -> None:
        ref = build_remote_ref(sts=REAL_STS, filename="a.png", content_type="image/png", size=800)
        self.assertEqual(ref["type"], "image")
        self.assertEqual(ref["showType"], "image")
        self.assertEqual(ref["file_class"], "vision")
        self.assertEqual(ref["greenNet"], "success")

    def test_audio_maps_to_audio(self) -> None:
        ref = build_remote_ref(sts=REAL_STS, filename="a.mp3", content_type="audio/mpeg", size=1200)
        self.assertEqual(ref["type"], "audio")
        self.assertEqual(ref["showType"], "audio")
        self.assertEqual(ref["file_class"], "audio")
        self.assertEqual(ref["greenNet"], "success")

    def test_document_maps_to_file_document(self) -> None:
        ref = build_remote_ref(
            sts=REAL_STS,
            filename="a.pdf",
            content_type="application/pdf",
            size=4096,
            parse_status="success",
        )
        self.assertEqual(ref["type"], "file")
        self.assertEqual(ref["showType"], "file")
        self.assertEqual(ref["file_class"], "document")
        self.assertEqual(ref["file"]["meta"]["parse_meta"]["parse_status"], "success")

    def test_content_type_is_guessed_from_filename(self) -> None:
        ref = build_remote_ref(sts=REAL_STS, filename="clip.mp4", content_type="", size=10)
        self.assertEqual(ref["file_type"], "video/mp4")
        self.assertEqual(ref["type"], "video")

    def test_upload_filetype_mapping(self) -> None:
        self.assertEqual(upload_filetype_from_content_type("image/png"), "image")
        self.assertEqual(upload_filetype_from_content_type("audio/mpeg"), "audio")
        self.assertEqual(upload_filetype_from_content_type("video/mp4"), "video")
        self.assertEqual(upload_filetype_from_content_type("application/pdf"), "file")
        self.assertEqual(upload_filetype_from_content_type(""), "file")

    def test_ref_type_and_file_class_mapping(self) -> None:
        self.assertEqual(remote_ref_type_from_content_type("image/webp"), "image")
        self.assertEqual(remote_file_class_from_content_type("image/webp"), "vision")
        self.assertEqual(remote_file_class_from_content_type("application/zip"), "document")


class RemoteRefUrlTests(unittest.TestCase):
    """url / user_id / region 等字段的推导。"""

    def test_url_prefers_upstream_file_url(self) -> None:
        self.assertEqual(sts_file_url(REAL_STS), REAL_STS["file_url"])

    def test_url_falls_back_to_endpoint_plus_object_key(self) -> None:
        sts = dict(REAL_STS)
        sts.pop("file_url")
        self.assertEqual(
            sts_file_url(sts),
            f"https://{REAL_STS['endpoint']}/{REAL_STS['file_path']}",
        )

    def test_url_uses_fallback_endpoint_when_given(self) -> None:
        sts = dict(REAL_STS)
        sts.pop("file_url")
        self.assertEqual(
            sts_file_url(sts, endpoint="qwen-webui-prod.oss-ap-southeast-1.aliyuncs.com"),
            f"https://qwen-webui-prod.oss-ap-southeast-1.aliyuncs.com/{REAL_STS['file_path']}",
        )

    def test_user_id_comes_from_object_key_prefix(self) -> None:
        self.assertEqual(sts_user_id(REAL_STS), "18639e0d-5b7f-4f3c-9b8e-2f0d6a1c7e55")
        self.assertEqual(sts_user_id({"file_path": "no-slash-key"}), "")

    def test_normalize_sign_region_strips_oss_prefix(self) -> None:
        self.assertEqual(normalize_sign_region("oss-cn-hangzhou"), "cn-hangzhou")
        self.assertEqual(normalize_sign_region("ap-southeast-1"), "ap-southeast-1")
        self.assertEqual(normalize_sign_region(""), "")

    def test_missing_file_id_raises(self) -> None:
        sts = dict(REAL_STS)
        sts.pop("file_id")
        with self.assertRaises(ValueError):
            build_remote_ref(sts=sts, filename="a.mp4", content_type="video/mp4", size=1)

    def test_missing_file_path_raises(self) -> None:
        sts = dict(REAL_STS)
        sts.pop("file_path")
        with self.assertRaises(ValueError):
            build_remote_ref(sts=sts, filename="a.mp4", content_type="video/mp4", size=1)

    def test_size_defaults_to_sts_filesize(self) -> None:
        sts = dict(REAL_STS)
        sts["filesize"] = 4321
        ref = build_remote_ref(sts=sts, filename="a.mp4", content_type="video/mp4")
        self.assertEqual(ref["size"], 4321)
        self.assertEqual(ref["file"]["size"], 4321)


class _FakeBucket:
    """OSS Bucket 替身：只记录 put_object 的参数。"""

    puts: list[tuple[str, int, dict]] = []

    def __init__(self, auth, endpoint, bucketname, region=None):
        del auth
        self.endpoint = endpoint
        self.bucketname = bucketname
        self.region = region

    def put_object(self, key, raw, headers=None):
        self.__class__.puts.append((key, len(raw), dict(headers or {})))
        return types.SimpleNamespace(status=200)


class _FakeClient:
    """QwenClient 替身：`put_bytes` 不允许再调 getstsToken。"""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def _request_json(self, method, path, token, body=None, timeout=30.0, account=None):
        del method, token, body, timeout, account
        self.requests.append(path)
        if path == "/api/v2/files/parse":
            return {"status": 200, "body": "{}"}
        if path == "/api/v2/files/parse/status":
            return {"status": 200, "body": '{"data": [{"status": "success"}]}'}
        raise AssertionError(f"unexpected path: {path}")


class PutBytesTests(unittest.IsolatedAsyncioTestCase):
    """`put_bytes` 必须能只吃外部 STS（浏览器上传代理的复用点）。"""

    async def test_put_bytes_uploads_and_builds_ref_without_getsts_token(self) -> None:
        from backend.services.upstream_file_uploader import UpstreamFileUploader

        _FakeBucket.puts = []
        client = _FakeClient()
        uploader = UpstreamFileUploader(client, types.SimpleNamespace(CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1))
        account = types.SimpleNamespace(email="alice@example.com", token="tok")

        with patch("backend.services.upstream_file_uploader.oss2.StsAuth", return_value=object()), \
             patch("backend.services.upstream_file_uploader.oss2.Bucket", _FakeBucket):
            result = await uploader.put_bytes(account, REAL_STS, b"x" * 2058934, "test-2m.mp4", "video/mp4")

        self.assertEqual(client.requests, [])  # 没再调 getstsToken / parse
        self.assertEqual(len(_FakeBucket.puts), 1)
        key, size, headers = _FakeBucket.puts[0]
        self.assertEqual(key, REAL_STS["file_path"])
        self.assertEqual(size, 2058934)
        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertEqual(result["remote_file_id"], REAL_STS["file_id"])
        self.assertEqual(result["remote_object_key"], REAL_STS["file_path"])
        self.assertEqual(result["parse_status"], "success")
        self.assertEqual(result["remote_ref"]["greenNet"], "greening")
        self.assertEqual(result["remote_ref"]["url"], REAL_STS["file_url"])
        self.assertEqual(set(result["remote_ref"]), set(REAL_REF))
        self.assertEqual(set(result["remote_ref"]["file"]), set(REAL_REF["file"]))
        # CDN 短链里没有 x-oss-expires，因此解析不出签名过期时间（缓存按会话 TTL 处理）
        self.assertIsNone(result["url_expires_at"])

    async def test_put_bytes_rejects_incomplete_sts(self) -> None:
        from backend.services.upstream_file_uploader import UpstreamFileUploader

        uploader = UpstreamFileUploader(_FakeClient(), types.SimpleNamespace(CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=1))
        account = types.SimpleNamespace(email="alice@example.com", token="tok")
        with self.assertRaises(RuntimeError):
            await uploader.put_bytes(account, {"file_id": "x"}, b"data", "a.mp4", "video/mp4")


if __name__ == "__main__":
    unittest.main()
