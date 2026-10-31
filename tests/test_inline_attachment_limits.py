"""附件内联上限 / 图片压缩 / SSRF / WAF 日志级别的回归测试。

覆盖 QA 提出的阻塞级问题：

1. url 字符数门控（上游 `files[].url` 是 `max_length=8192` 的字符串）；
2. 图片超限自动压缩，压不进就抛 `InlineMediaTooLargeError`（上层 400）；
3. 音频 / 视频超限给出可执行文案；
4. 远程 URL 附件的 SSRF 防护（含每一跳重定向复检）；
5. WAF 拦截上传是预期路径，只打一行 warning、不打整栈 ERROR。
"""

from __future__ import annotations

import base64
import io
import json
import random
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.runtime.attachment_types import NormalizedAttachment
from backend.services import attachment_preprocessor
from backend.services.attachment_errors import (
    INLINE_URL_HARD_CAP,
    InlineMediaTooLargeError,
    attachment_rejection_from_upstream_error,
)
from backend.services.attachment_preprocessor import _remote_url_is_safe, preprocess_attachments
from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import prepare_context_attachments
from backend.services.inline_media_ref import (
    build_inline_ref,
    compress_image_to_fit,
    resolve_url_chars_limit,
    to_data_uri,
)
from backend.services.upstream_file_uploader import UpstreamWafChallengeError
from backend.toolcore.context_offload import ContextOffloader


def _is_fake_module(module: object) -> bool:
    """判断 `sys.modules` 里的条目是不是别的测试手搓的假模块。

    真包一定有 `__file__`（普通模块）或 `__path__`（namespace package）；
    `types.ModuleType("fastapi")` 直接塞进去的假模块两者都没有。
    """
    return module is not None and getattr(module, "__file__", None) is None and not hasattr(module, "__path__")


#: 历史测试会在 import 期往 `sys.modules` 里塞这几个包的假实现（且从不还原）。
_POISONED_ROOTS: tuple = ("fastapi", "pydantic", "pydantic_settings")


def _poisoned_names() -> list:
    """列出 `sys.modules` 里被替换成假实现的包及其子模块。"""
    names = []
    for name, module in list(sys.modules.items()):
        root = name.split(".", 1)[0]
        if root in _POISONED_ROOTS and _is_fake_module(module):
            names.append(name)
    return names


def _import_attachment_handlers() -> tuple:
    """导入真实的附件异常处理器，不受其它测试注入的假 `fastapi` / `pydantic` 影响。

    污染源：`tests/test_admin_account_status.py`（以及 test_admin_chat_clear /
    test_admin_personalization / test_admin_verify_account_flow）在 **import 期**
    执行 `sys.modules["fastapi"] = types.ModuleType("fastapi")` 等，且从不还原。
    pytest 全量跑时这些假模块会一直留在 `sys.modules` 里，于是任何后续
    `from fastapi.responses import JSONResponse` 都会抛
    `No module named 'fastapi.responses'; 'fastapi' is not a package`
    —— 单独跑本文件时没有污染，所以之前是「单跑绿、全量红」。

    这里临时把假模块从 `sys.modules` 摘掉、导入真实包，导入完成后把
    `sys.modules` 原样还原：既不依赖全局状态，也不改变其它测试看到的环境。

    Returns:
        `(inline_media_too_large_handler, upstream_attachment_rejected_handler)`。
    """
    fake_names = _poisoned_names()
    if not fake_names:  # 没有污染：直接走正常导入
        from backend.api.attachment_error_handlers import (
            inline_media_too_large_handler,
            upstream_attachment_rejected_handler,
        )

        return inline_media_too_large_handler, upstream_attachment_rejected_handler

    saved = {name: sys.modules[name] for name in fake_names}
    for name in fake_names:
        del sys.modules[name]
    try:
        from backend.api.attachment_error_handlers import (
            inline_media_too_large_handler,
            upstream_attachment_rejected_handler,
        )

        return inline_media_too_large_handler, upstream_attachment_rejected_handler
    finally:
        # 还原现场：清掉本次导入的真实模块（含子模块），再把假模块放回去。
        for name in [
            n for n in list(sys.modules) if n.split(".", 1)[0] in _POISONED_ROOTS
        ]:
            del sys.modules[name]
        sys.modules.update(saved)



inline_media_too_large_handler, upstream_attachment_rejected_handler = _import_attachment_handlers()


try:  # 图片压缩依赖 pillow；缺失时跳过相关用例（requirements.txt 已声明）
    import PIL  # noqa: F401

    HAS_PILLOW = True
except Exception:  # pragma: no cover - 环境未装 pillow
    HAS_PILLOW = False


def _bytes_for_url_length(mime: str, total_chars: int) -> int:
    """算出能让 `to_data_uri(mime, b'x' * n)` 恰好等于 total_chars 的 n。"""
    prefix = len(f"data:{mime};base64,")
    groups = (total_chars - prefix) // 4
    # 4*ceil(n/3) == 4*groups 且 n 不是 3 的倍数
    return max(0, groups * 3 - 2)


def _noise_png(width: int = 640, height: int = 480, seed: int = 7) -> bytes:
    """生成一张纯噪点 PNG（最坏压缩场景：几乎不可压缩）。"""
    from PIL import Image

    rnd = random.Random(seed)
    image = Image.new("RGB", (width, height))
    image.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(width * height)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _photo_like_png(width: int = 640, height: int = 480, seed: int = 11) -> bytes:
    """生成一张「类照片」PNG（平滑渐变 + 轻微噪点）。"""
    from PIL import Image

    rnd = random.Random(seed)
    image = Image.new("RGB", (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            base_r = (x * 255) // width
            base_g = (y * 255) // height
            base_b = ((x + y) * 255) // (width + height)
            pixels.append(
                (
                    max(0, min(255, base_r + rnd.randrange(-6, 7))),
                    max(0, min(255, base_g + rnd.randrange(-6, 7))),
                    max(0, min(255, base_b + rnd.randrange(-6, 7))),
                )
            )
    image.putdata(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class UrlLengthGateTests(unittest.TestCase):
    """url 字符数门控：边界、文案、以及「绝不静默放行超长 url」。"""

    def test_effective_limit_is_below_upstream_hard_cap(self) -> None:
        limit = resolve_url_chars_limit()

        self.assertLess(limit, INLINE_URL_HARD_CAP, "must keep a margin below max_length=8192")
        self.assertGreater(limit, 7000)

    def test_url_exactly_at_limit_is_accepted(self) -> None:
        limit = resolve_url_chars_limit()
        size = _bytes_for_url_length("audio/mpeg", limit)

        uri = to_data_uri("audio/mpeg", b"x" * size)
        self.assertLessEqual(len(uri), limit)
        ref = build_inline_ref("a.mp3", "audio/mpeg", b"x" * size)
        self.assertLessEqual(len(ref["url"]), limit)

    def test_url_just_over_limit_is_refused(self) -> None:
        limit = resolve_url_chars_limit()
        size = _bytes_for_url_length("audio/mpeg", limit) + 3

        with self.assertRaises(InlineMediaTooLargeError) as ctx:
            to_data_uri("audio/mpeg", b"x" * size)
        self.assertGreater(ctx.exception.url_chars, limit)
        self.assertEqual(ctx.exception.limit, limit)

    def test_measured_8182_boundary_is_refused_locally(self) -> None:
        """实测上游 8182 能过、8198 被拒；本地阈值更保守（留边距），两者都拒绝。

        宁可提前 400，也不要把一个「刚好能过」的 url 发出去赌上游行为。
        """
        for total in (8182, 8198):
            with self.subTest(total=total):
                size = _bytes_for_url_length("image/png", total)
                uri = to_data_uri("image/png", b"x" * size) if total <= resolve_url_chars_limit() else None
                if uri is not None:  # pragma: no cover - 阈值内不会走到
                    self.assertLessEqual(len(uri), total)
                with self.assertRaises(InlineMediaTooLargeError):
                    build_inline_ref("a.png", "image/png", b"x" * size)

    def test_audio_message_is_actionable(self) -> None:
        with self.assertRaises(InlineMediaTooLargeError) as ctx:
            build_inline_ref("voice.mp3", "audio/mpeg", b"x" * 87_800)

        message = str(ctx.exception)
        self.assertIn("audio attachment 'voice.mp3'", message)
        self.assertIn("KB", message)
        self.assertIn("url<=", message)
        self.assertIn("@32kbps", message)
        self.assertIn("upload via the file API", message)

    def test_video_message_says_inline_is_not_viable(self) -> None:
        with self.assertRaises(InlineMediaTooLargeError) as ctx:
            build_inline_ref("clip.mp4", "video/mp4", b"x" * 200_000)

        message = str(ctx.exception)
        self.assertIn("video attachment 'clip.mp4'", message)
        self.assertIn("not viable", message)
        self.assertIn("upload via the file api", message.lower())

    def test_document_message_points_at_upload_api(self) -> None:
        with self.assertRaises(InlineMediaTooLargeError) as ctx:
            build_inline_ref("notes.pdf", "application/pdf", b"%PDF" + b"x" * 20_000)

        self.assertIn("Documents need the file upload + parse API", str(ctx.exception))


class ImageCompressionTests(unittest.TestCase):
    """图片超限时的自动降采样 / 重编码。"""

    def setUp(self) -> None:
        if not HAS_PILLOW:  # pragma: no cover - 环境未装 pillow
            self.skipTest("pillow not installed")

    def test_noise_png_is_compressed_below_limit(self) -> None:
        raw = _noise_png()
        limit = resolve_url_chars_limit()
        self.assertGreater(len(raw), 200 * 1024, "fixture should be a genuinely large PNG")

        ref = build_inline_ref("noise.png", "image/png", raw)

        self.assertLessEqual(len(ref["url"]), limit)
        self.assertLess(ref["size"], len(raw))
        self.assertIn(ref["file_type"], {"image/jpeg", "image/webp"})
        self.assertTrue(ref["url"].startswith(f"data:{ref['file_type']};base64,"))
        # 压缩后文件名后缀要与 mime 一致（避免上游按后缀误判）
        self.assertTrue(ref["name"].endswith((".jpg", ".webp")), ref["name"])

    def test_photo_like_png_is_compressed_below_limit(self) -> None:
        raw = _photo_like_png()
        limit = resolve_url_chars_limit()

        compressed = compress_image_to_fit(raw, filename="photo.png")

        self.assertIsNotNone(compressed)
        payload, mime = compressed
        self.assertLessEqual(len(f"data:{mime};base64,") + 4 * ((len(payload) + 2) // 3), limit)
        self.assertLess(len(payload), len(raw))

    def test_transparent_png_prefers_webp(self) -> None:
        from PIL import Image

        image = Image.new("RGBA", (400, 300), (255, 0, 0, 0))
        for x in range(0, 400, 4):
            for y in range(0, 300, 4):
                image.putpixel((x, y), (0, 128, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()

        compressed = compress_image_to_fit(raw, filename="logo.png")

        self.assertIsNotNone(compressed)
        _payload, mime = compressed
        self.assertEqual(mime, "image/webp")

    def test_undecodable_image_is_refused_not_truncated(self) -> None:
        with self.assertRaises(InlineMediaTooLargeError):
            build_inline_ref("broken.png", "image/png", b"x" * 100_000)

    def test_compression_can_be_disabled(self) -> None:
        raw = _noise_png()
        with self.assertRaises(InlineMediaTooLargeError):
            build_inline_ref("noise.png", "image/png", raw, allow_compression=False)

    def test_small_image_is_not_recompressed(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (8, 8), (1, 2, 3))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()

        ref = build_inline_ref("tiny.png", "image/png", raw)

        self.assertEqual(ref["file_type"], "image/png")
        self.assertEqual(ref["name"], "tiny.png")
        self.assertEqual(ref["size"], len(raw))


class RemoteUrlSsrfTests(unittest.IsolatedAsyncioTestCase):
    """远程 URL 附件的 SSRF 防护。"""

    PRIVATE_URLS = (
        "http://127.0.0.1/a.png",
        "http://127.1.2.3/a.png",
        "http://10.0.0.5/a.png",
        "http://172.16.3.4/a.png",
        "http://172.31.255.255/a.png",
        "http://192.168.1.1/a.png",
        "http://169.254.169.254/latest/meta-data/a.png",
        "http://0.0.0.0/a.png",
        "http://[::1]/a.png",
        "http://[fc00::1]/a.png",
        "http://[fe80::1]/a.png",
        "http://[::ffff:127.0.0.1]/a.png",
        "http://localhost/a.png",
        "http://LOCALHOST/a.png",
        "ftp://example.com/a.png",
        "file:///etc/passwd",
    )

    def test_private_and_reserved_addresses_are_rejected(self) -> None:
        for url in self.PRIVATE_URLS:
            with self.subTest(url=url):
                allowed, reason = _remote_url_is_safe(url)
                self.assertFalse(allowed, f"{url} should be rejected ({reason})")
                self.assertTrue(reason)

    def test_public_addresses_are_allowed(self) -> None:
        allowed, reason = _remote_url_is_safe("https://example.com/logo.png")
        self.assertTrue(allowed, reason)

    async def test_redirect_to_internal_address_is_not_followed(self) -> None:
        calls: list[str] = []

        class _Response:
            def __init__(self, status: int, location: str = "", chunks: list[bytes] | None = None) -> None:
                self.status_code = status
                self.headers = {"location": location, "content-type": "image/png"}
                self._chunks = chunks or []

            async def aiter_bytes(self):
                for chunk in self._chunks:
                    yield chunk

        class _Client:
            def __init__(self, **_kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info) -> bool:
                return False

            def stream(self, _method: str, url: str):
                calls.append(url)
                response = _Response(302, location="http://169.254.169.254/latest/meta-data/iam")
                return _Ctx(response)

        class _Ctx:
            def __init__(self, response) -> None:
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, *exc_info) -> bool:
                return False

        with patch.object(attachment_preprocessor, "httpx", SimpleNamespace(AsyncClient=_Client)):
            fetched = await attachment_preprocessor._download_remote_media(
                "https://example.com/redirect.png",
                max_bytes=1024,
            )

        self.assertIsNone(fetched)
        self.assertEqual(calls, ["https://example.com/redirect.png"])

    async def test_private_urls_never_reach_the_http_client(self) -> None:
        calls: list[str] = []

        class _Client:
            def __init__(self, **_kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info) -> bool:
                return False

            def stream(self, _method: str, url: str):  # pragma: no cover - 不应被调用
                calls.append(url)
                raise AssertionError(f"private url was fetched: {url}")

        with patch.object(attachment_preprocessor, "httpx", SimpleNamespace(AsyncClient=_Client)):
            for url in self.PRIVATE_URLS:
                if not url.startswith("http"):
                    continue
                with self.subTest(url=url):
                    self.assertIsNone(
                        await attachment_preprocessor._download_remote_media(url, max_bytes=1024)
                    )

        self.assertEqual(calls, [])


class _FakeFileStore:
    """内存 file_store（落盘到临时目录，便于断言字节真的存在）。"""

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
            "sha256": "deadbeef",
            "created_at": 1.0,
            "owner_token": owner_token or "",
            "purpose": purpose,
        }
        self.records[file_id] = meta
        return meta

    async def get(self, file_id):
        return self.records.get(file_id)


class OversizedAttachmentTests(unittest.IsolatedAsyncioTestCase):
    """超限附件必须真的落盘（oversized=True），不能凭空消失。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _FakeFileStore(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_oversized_inline_media_is_still_persisted(self) -> None:
        payload = {
            "model": "qwen3.8-omni-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"x" * 4096).decode("ascii"),
                                "format": "mp3",
                            },
                        }
                    ],
                }
            ],
        }
        values = {
            "INLINE_ATTACHMENT_MAX_BYTES": 1024,
            "INLINE_ATTACHMENT_REMOTE_FETCH": False,
        }

        with patch.object(
            attachment_preprocessor,
            "_config_value",
            side_effect=lambda name, default: values.get(name, default),
        ):
            result = await preprocess_attachments(payload, self.store, owner_token="tok")

        self.assertEqual(len(result.attachments), 1)
        attachment = result.attachments[0]
        self.assertTrue(attachment.oversized)
        self.assertEqual(Path(attachment.local_path).read_bytes(), b"x" * 4096)
        # block 已改写成本地 file_id，交给下游决定压缩 / 上传回退 / 4xx
        self.assertEqual(result.payload["messages"][0]["content"][0]["type"], "input_audio")
        self.assertEqual(result.payload["messages"][0]["content"][0]["file_id"], attachment.file_id)


class _WafUploader:
    async def upload_local_file(self, _acc, _local_meta):
        raise UpstreamWafChallengeError("aliyun waf puzzle challenge (sceneId=19x5u7lo)")


class WafLogLevelTests(unittest.IsolatedAsyncioTestCase):
    """WAF 拦截是预期路径：一行 warning，不打整栈 ERROR。"""

    def _app(self) -> tuple[SimpleNamespace, str]:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        handle.write(b"%PDF-1.4 doc")
        handle.close()
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=4000,
                CONTEXT_FORCE_FILE_MAX_CHARS=10000,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
                INLINE_ATTACHMENT_ENABLED=True,
                INLINE_ATTACHMENT_MAX_BYTES=10 * 1024 * 1024,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=_WafUploader(),
        ))
        return app, handle.name

    async def test_waf_challenge_logs_warning_without_traceback(self) -> None:
        app, path = self._app()
        attachment = NormalizedAttachment(
            file_id="local-1",
            filename="notes.pdf",
            content_type="application/pdf",
            source="manual-upload",
            local_path=path,
            sha256="abc",
            purpose="user-upload",
        )
        try:
            with self.assertLogs("qwen2api.context_attachment_manager", level="WARNING") as logs:
                result = await prepare_context_attachments(
                    app=app,
                    payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "read"}]},
                    surface="openai",
                    auth_token="tok",
                    client_profile=OPENCLAW_OPENAI_PROFILE,
                    session_key="openai:waf",
                    existing_attachments=[attachment],
                )
        finally:
            Path(path).unlink(missing_ok=True)

        waf_records = [r for r in logs.records if "blocked by WAF" in r.getMessage()]
        self.assertEqual(len(waf_records), 1, [r.getMessage() for r in logs.records])
        self.assertEqual(waf_records[0].levelname, "WARNING")
        self.assertIsNone(waf_records[0].exc_info, "WAF 是预期路径，不应带 traceback")
        self.assertTrue(result["attachment_fallback"])


class UpstreamAttachmentRejectionTests(unittest.TestCase):
    """上游把校验错误塞进 200 + content 时，必须翻译成 4xx。"""

    def test_url_too_long_with_files_is_translated(self) -> None:
        error = attachment_rejection_from_upstream_error(
            code="RequestValidationError",
            details="[\"...files -> 0 -> url': Value error, url too long\"]",
            has_files=True,
        )

        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error.code, "upstream_attachment_rejected")
        self.assertIn("url too long", str(error))

    def test_plain_text_request_keeps_legacy_behaviour(self) -> None:
        self.assertIsNone(
            attachment_rejection_from_upstream_error(
                code="RequestValidationError",
                details="url too long",
                has_files=False,
            )
        )

    def test_unrelated_upstream_error_is_not_translated(self) -> None:
        self.assertIsNone(
            attachment_rejection_from_upstream_error(
                code="quota_limit",
                details="daily quota exceeded",
                has_files=True,
            )
        )


class AttachmentErrorHttpMappingTests(unittest.IsolatedAsyncioTestCase):
    """`InlineMediaTooLargeError` / `UpstreamAttachmentRejectedError` 必须映射成 400。"""

    async def test_too_large_is_rendered_as_400_attachment_too_large(self) -> None:
        exc = InlineMediaTooLargeError(
            "audio attachment 'a.mp3' is 85.7KB; inline limit is ~5.9KB (url<=8100 chars).",
            filename="a.mp3",
            url_chars=11_400,
            limit=8100,
            size=87_800,
            kind="audio",
        )
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"))

        response = await inline_media_too_large_handler(request, exc)

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["error"]["code"], "attachment_too_large")
        self.assertIn("a.mp3", body["error"]["message"])
        self.assertEqual(body["error"]["limit"], 8100)

    async def test_upstream_rejection_is_rendered_as_400(self) -> None:
        exc = attachment_rejection_from_upstream_error(
            code="RequestValidationError",
            details="url too long",
            has_files=True,
        )
        assert exc is not None
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"))

        response = await upstream_attachment_rejected_handler(request, exc)

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["error"]["code"], "upstream_attachment_rejected")
        self.assertEqual(body["error"]["upstream_code"], "RequestValidationError")


class InlineCompressionThroughManagerTests(unittest.IsolatedAsyncioTestCase):
    """大图经 context_attachment_manager 会被压缩内联，不触发上传。"""

    def setUp(self) -> None:
        if not HAS_PILLOW:  # pragma: no cover - 环境未装 pillow
            self.skipTest("pillow not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_large_png_is_compressed_and_inlined_without_upload(self) -> None:
        uploader = SimpleNamespace(upload_local_file=AsyncMock())
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=4000,
                CONTEXT_FORCE_FILE_MAX_CHARS=10000,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
                INLINE_ATTACHMENT_ENABLED=True,
                INLINE_ATTACHMENT_MAX_BYTES=10 * 1024 * 1024,
                INLINE_ATTACHMENT_MAX_URL_CHARS=8192,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=uploader,
        ))
        raw = _photo_like_png()
        path = self.root / "photo.png"
        path.write_bytes(raw)
        attachment = NormalizedAttachment(
            file_id="local-big",
            filename="photo.png",
            content_type="image/png",
            source="inline-image",
            local_path=str(path),
            sha256="abc",
            purpose="user-upload",
        )

        result = await prepare_context_attachments(
            app=app,
            payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "what is this"}]},
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:compress",
            existing_attachments=[attachment],
        )

        refs = result["upstream_files"]
        self.assertEqual(len(refs), 1)
        self.assertLessEqual(len(refs[0]["url"]), resolve_url_chars_limit())
        self.assertEqual(refs[0]["type"], "image")
        self.assertLess(refs[0]["size"], len(raw))
        uploader.upload_local_file.assert_not_awaited()
        app.state.account_pool.acquire_wait.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
