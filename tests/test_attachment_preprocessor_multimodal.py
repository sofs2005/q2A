import base64
import hashlib
import os
import time
import types
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.services import attachment_preprocessor
from backend.services.attachment_preprocessor import preprocess_attachments
from backend.services.model_modalities import ModalityNotSupportedError

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-bytes"
MP3_BYTES = b"fake-mp3-bytes"
WAV_BYTES = b"RIFFfake-wav-bytes"
MP4_BYTES = b"fake-mp4-bytes"
PDF_BYTES = b"%PDF-1.4 fake-pdf-bytes"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _data_uri(content_type: str, raw: bytes) -> str:
    return f"data:{content_type};base64,{_b64(raw)}"


class _FakeFileStore:
    """内存版 file_store，落盘到临时目录以便断言字节真实存在。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, dict] = {}

    async def save_bytes(self, filename, content_type, raw, purpose, owner_token=None):
        file_id = uuid.uuid4().hex
        target = self.root / purpose / f"{file_id}_{filename}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        meta = {
            "id": file_id,
            "path": str(target),
            "filename": filename,
            "content_type": content_type,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "created_at": time.time(),
            "owner_token": owner_token or "",
            "purpose": purpose,
        }
        self.records[file_id] = meta
        return meta

    async def get(self, file_id):
        return self.records.get(file_id)

    async def delete_path(self, path):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


class _FakeResponse:
    def __init__(self, status_code: int, content_type: str, chunks: list[bytes]) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeAsyncClient:
    """假的 httpx.AsyncClient：行为由类属性配置。"""

    response: _FakeResponse | None = None
    raise_error: Exception | None = None
    requested: list[tuple[str, str]] = []
    init_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = dict(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    def stream(self, method: str, url: str):
        type(self).requested.append((method, url))
        if type(self).raise_error is not None:
            raise type(self).raise_error
        return _FakeStreamContext(type(self).response or _FakeResponse(200, "application/octet-stream", []))


def _fake_httpx() -> types.SimpleNamespace:
    return types.SimpleNamespace(AsyncClient=_FakeAsyncClient)


class _PreprocessorTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.store = _FakeFileStore(Path(self._tmp.name))
        _FakeAsyncClient.response = None
        _FakeAsyncClient.raise_error = None
        _FakeAsyncClient.requested = []
        _FakeAsyncClient.init_kwargs = {}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def config_patch(self, **overrides):
        values = {
            "INLINE_ATTACHMENT_MAX_BYTES": 10 * 1024 * 1024,
            "INLINE_ATTACHMENT_REMOTE_FETCH": True,
            "INLINE_ATTACHMENT_ENABLED": True,
        }
        values.update(overrides)
        return patch.object(
            attachment_preprocessor,
            "_config_value",
            side_effect=lambda name, default: values.get(name, default),
        )


class OpenAIChatSurfaceTests(_PreprocessorTestBase):
    async def test_image_data_uri_is_normalized_and_persisted(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": _data_uri("image/png", PNG_BYTES)}},
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][1]
        self.assertEqual(block["type"], "input_image")
        self.assertEqual(block["mime_type"], "image/png")
        self.assertEqual(len(result.attachments), 1)
        attachment = result.attachments[0]
        self.assertEqual(attachment.content_type, "image/png")
        self.assertEqual(attachment.source, "inline-image")
        self.assertEqual(attachment.purpose, "user-upload")
        self.assertEqual(Path(attachment.local_path).read_bytes(), PNG_BYTES)
        self.assertEqual(result.uploaded_file_ids, [attachment.file_id])

    async def test_input_audio_base64_is_normalized(self) -> None:
        payload = {
            "model": "qwen3.8-omni-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": _b64(WAV_BYTES), "format": "wav"}},
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_audio")
        self.assertEqual(block["mime_type"], "audio/wav")
        self.assertEqual(result.attachments[0].content_type, "audio/wav")
        self.assertEqual(result.attachments[0].source, "inline-audio")

    async def test_input_audio_mp3_format_maps_to_audio_mpeg(self) -> None:
        """OpenAI `input_audio.format="mp3"` 必须归一化成 `audio/mpeg` 且文件名带 .mp3。

        下游 `context_attachment_manager` 靠 filename/mime 判定「可内联音频」并生成
        `type/showType=audio, file_class=audio` 的 ref，这里断了整条链路就退化上传。
        """
        payload = {
            "model": "qwen3.8-omni-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "listen"},
                        {"type": "input_audio", "input_audio": {"data": _b64(MP3_BYTES), "format": "mp3"}},
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][1]
        self.assertEqual(block["type"], "input_audio")
        self.assertEqual(block["mime_type"], "audio/mpeg")
        attachment = result.attachments[0]
        self.assertEqual(attachment.content_type, "audio/mpeg")
        self.assertTrue(attachment.filename.endswith(".mp3"), attachment.filename)
        self.assertEqual(Path(attachment.local_path).read_bytes(), MP3_BYTES)

    async def test_audio_url_data_uri(self) -> None:
        payload = {
            "model": "qwen3.8-omni-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": _data_uri("audio/mpeg", MP3_BYTES)}}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_audio")
        self.assertEqual(result.attachments[0].content_type, "audio/mpeg")
        self.assertEqual(Path(result.attachments[0].local_path).read_bytes(), MP3_BYTES)

    async def test_video_url_data_uri(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video_url", "video_url": {"url": _data_uri("video/mp4", MP4_BYTES)}}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_video")
        self.assertEqual(block["mime_type"], "video/mp4")
        self.assertEqual(result.attachments[0].source, "inline-video")

    async def test_file_block_still_produces_input_file(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "filename": "notes.txt", "mime_type": "text/plain", "text": "hello"},
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_file")
        self.assertEqual(block["filename"], "notes.txt")
        self.assertEqual(result.attachments[0].source, "manual-upload")


class RemoteUrlTests(_PreprocessorTestBase):
    async def test_remote_image_is_downloaded(self) -> None:
        _FakeAsyncClient.response = _FakeResponse(200, "image/png", [PNG_BYTES[:8], PNG_BYTES[8:]])
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://example.com/logo.png"}}],
                }
            ],
        }

        with self.config_patch(), patch.object(attachment_preprocessor, "httpx", _fake_httpx()):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_image")
        self.assertEqual(result.attachments[0].source, "remote-url")
        self.assertEqual(Path(result.attachments[0].local_path).read_bytes(), PNG_BYTES)
        self.assertEqual(_FakeAsyncClient.requested, [("GET", "https://example.com/logo.png")])
        # 超时 20s / 手动跟随重定向（每一跳都要复检主机，故关掉 httpx 自动跟随）
        self.assertEqual(_FakeAsyncClient.init_kwargs.get("timeout"), 20.0)
        self.assertFalse(_FakeAsyncClient.init_kwargs.get("follow_redirects"))
        self.assertEqual(_FakeAsyncClient.init_kwargs.get("max_redirects"), 3)

    async def test_remote_fetch_failure_skips_attachment_without_raising(self) -> None:
        _FakeAsyncClient.raise_error = RuntimeError("connect timeout")
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/missing.png"}},
                    ],
                }
            ],
        }

        with self.config_patch(), patch.object(attachment_preprocessor, "httpx", _fake_httpx()):
            with self.assertLogs("qwen2api.attachment_preprocessor", level="WARNING") as logs:
                result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.attachments, [])
        skipped = result.payload["messages"][0]["content"][1]
        self.assertEqual(skipped["type"], "text")
        self.assertIn("Attachment skipped", skipped["text"])
        self.assertTrue(any("remote fetch error" in line for line in logs.output))

    async def test_remote_fetch_disabled(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://example.com/logo.png"}}],
                }
            ],
        }

        with self.config_patch(INLINE_ATTACHMENT_REMOTE_FETCH=False), patch.object(
            attachment_preprocessor, "httpx", _fake_httpx()
        ):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.attachments, [])
        self.assertEqual(_FakeAsyncClient.requested, [])
        self.assertEqual(result.payload["messages"][0]["content"][0]["type"], "text")

    async def test_remote_payload_over_limit_is_skipped(self) -> None:
        _FakeAsyncClient.response = _FakeResponse(200, "video/mp4", [b"x" * 64, b"y" * 64])
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video_url", "video_url": {"url": "https://example.com/big.mp4"}}],
                }
            ],
        }

        with self.config_patch(INLINE_ATTACHMENT_MAX_BYTES=100), patch.object(
            attachment_preprocessor, "httpx", _fake_httpx()
        ):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.attachments, [])

    async def test_non_http_scheme_is_rejected(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "ftp://example.com/a.png"}}],
                }
            ],
        }

        with self.config_patch(), patch.object(attachment_preprocessor, "httpx", _fake_httpx()):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.attachments, [])
        self.assertEqual(_FakeAsyncClient.requested, [])

    async def test_private_network_remote_urls_are_not_fetched(self) -> None:
        """内网 / 回环 / 云元数据地址必须拒绝，否则网关会变成 SSRF 跳板。

        调用方可以任意填写 `image_url.url`，这些地址一旦被下载：
        1) 内网服务可被探测（127.0.0.1、10/8、192.168/16、172.16/12）；
        2) 云厂商元数据服务（169.254.169.254）可能泄露账号凭证。
        """
        urls = [
            "http://127.0.0.1:8080/admin.png",
            "http://localhost/a.png",
            "http://10.0.0.5/a.png",
            "http://192.168.1.1/a.png",
            "http://169.254.169.254/latest/meta-data/a.png",
            "http://[::1]/a.png",
        ]

        for url in urls:
            with self.subTest(url=url):
                _FakeAsyncClient.requested = []
                _FakeAsyncClient.response = _FakeResponse(200, "image/png", [PNG_BYTES])
                payload = {
                    "model": "qwen3.8-max",
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
                    ],
                }
                with self.config_patch(), patch.object(attachment_preprocessor, "httpx", _fake_httpx()):
                    result = await preprocess_attachments(payload, self.store, owner_token="tok")

                self.assertEqual(_FakeAsyncClient.requested, [], f"private url was fetched: {url}")
                self.assertEqual(result.attachments, [])


class ResponsesSurfaceTests(_PreprocessorTestBase):
    async def test_input_image_with_image_url(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": _data_uri("image/png", PNG_BYTES)}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_image")
        self.assertEqual(len(result.attachments), 1)

    async def test_input_image_with_file_id_reuses_local_file(self) -> None:
        saved = await self.store.save_bytes("photo.png", "image/png", PNG_BYTES, "vision", owner_token="tok")
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": saved["id"]}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_image")
        self.assertEqual(block["file_id"], saved["id"])
        self.assertEqual(result.uploaded_file_ids, [])
        self.assertEqual(result.attachments[0].source, "manual-upload-ref")
        self.assertEqual(result.attachments[0].local_path, saved["path"])

    async def test_unknown_file_id_block_is_left_untouched(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "does-not-exist"}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.attachments, [])
        self.assertEqual(result.payload["messages"][0]["content"][0]["file_id"], "does-not-exist")

    async def test_unknown_file_id_falls_back_to_inline_payload(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": "does-not-exist",
                            "filename": "notes.txt",
                            "mime_type": "text/plain",
                            "text": "inline fallback",
                        }
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_file")
        self.assertNotEqual(block["file_id"], "does-not-exist")
        self.assertEqual(Path(result.attachments[0].local_path).read_bytes(), b"inline fallback")

    async def test_file_ref_with_image_mime_becomes_input_image(self) -> None:
        saved = await self.store.save_bytes("photo.png", "image/png", PNG_BYTES, "upload", owner_token="tok")
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": saved["id"]}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_image")
        self.assertEqual(block["mime_type"], "image/png")

    async def test_input_file_with_file_data(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "notes.pdf",
                            "file_data": _data_uri("application/pdf", PDF_BYTES),
                        }
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_file")
        self.assertEqual(block["mime_type"], "application/pdf")
        self.assertEqual(Path(result.attachments[0].local_path).read_bytes(), PDF_BYTES)

    async def test_input_video_url(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_video", "video_url": _data_uri("video/mp4", MP4_BYTES)}],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.payload["messages"][0]["content"][0]["type"], "input_video")


class AnthropicSurfaceTests(_PreprocessorTestBase):
    async def test_image_base64_source(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": _b64(PNG_BYTES)},
                        },
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][1]
        self.assertEqual(block["type"], "input_image")
        self.assertEqual(block["mime_type"], "image/png")
        self.assertEqual(result.attachments[0].content_type, "image/png")

    async def test_image_url_source(self) -> None:
        _FakeAsyncClient.response = _FakeResponse(200, "image/png", [PNG_BYTES])
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                    ],
                }
            ],
        }

        with self.config_patch(), patch.object(attachment_preprocessor, "httpx", _fake_httpx()):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.attachments[0].source, "remote-url")

    async def test_document_base64_source(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": _b64(PDF_BYTES)},
                        }
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        block = result.payload["messages"][0]["content"][0]
        self.assertEqual(block["type"], "input_file")
        self.assertEqual(block["mime_type"], "application/pdf")
        self.assertEqual(result.attachments[0].content_type, "application/pdf")

    async def test_document_text_source(self) -> None:
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "text", "media_type": "text/plain", "data": "inline text"},
                        }
                    ],
                }
            ],
        }

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(result.payload["messages"][0]["content"][0]["type"], "input_file")
        self.assertEqual(Path(result.attachments[0].local_path).read_bytes(), b"inline text")


class ModalityGatingTests(_PreprocessorTestBase):
    def _audio_payload(self) -> dict:
        return {
            "model": "qwen3.7-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "listen"},
                        {"type": "input_audio", "input_audio": {"data": _b64(WAV_BYTES), "format": "wav"}},
                    ],
                }
            ],
        }

    async def test_strict_mode_rejects_audio_for_text_only_model(self) -> None:
        with self.config_patch(), patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            with self.assertRaises(ModalityNotSupportedError) as ctx:
                await preprocess_attachments(self._audio_payload(), self.store, owner_token="tok")

        self.assertIn("does not support", str(ctx.exception))
        self.assertIn("audio", str(ctx.exception))
        self.assertEqual(self.store.records, {})

    async def test_non_strict_mode_drops_audio_attachment(self) -> None:
        with self.config_patch(), patch.dict(os.environ, {"MULTIMODAL_STRICT": "false"}):
            result = await preprocess_attachments(self._audio_payload(), self.store, owner_token="tok")

        self.assertEqual(result.attachments, [])
        dropped = result.payload["messages"][0]["content"][1]
        self.assertEqual(dropped["type"], "text")
        self.assertIn("unsupported by selected model", dropped["text"])

    async def test_document_not_gated_for_text_only_model(self) -> None:
        payload = {
            "model": "qwen3.7-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "filename": "notes.txt", "mime_type": "text/plain", "text": "hi"},
                    ],
                }
            ],
        }

        with self.config_patch(), patch.dict(os.environ, {"MULTIMODAL_STRICT": "true"}):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(len(result.attachments), 1)

    async def test_payload_without_media_is_untouched(self) -> None:
        payload = {"model": "qwen3.8-max", "messages": [{"role": "user", "content": "hello"}]}

        with self.config_patch():
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertIs(result.payload, payload)
        self.assertEqual(result.attachments, [])


if __name__ == "__main__":
    unittest.main()
