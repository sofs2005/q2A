import base64
import unittest

from backend.services.inline_media_ref import (
    KIND_AUDIO,
    KIND_DOCUMENT,
    KIND_IMAGE,
    KIND_VIDEO,
    InlineMediaTooLargeError,
    build_inline_ref,
    classify_media,
    normalize_mime_type,
    resolve_url_chars_limit,
    to_data_uri,
)


class ClassifyMediaTests(unittest.TestCase):
    def test_classifies_by_mime(self) -> None:
        self.assertEqual(classify_media("a.png", "image/png"), KIND_IMAGE)
        self.assertEqual(classify_media("a.bin", "audio/wav"), KIND_AUDIO)
        self.assertEqual(classify_media("a.bin", "video/mp4"), KIND_VIDEO)
        self.assertEqual(classify_media("a.bin", "application/pdf"), KIND_DOCUMENT)

    def test_classifies_by_extension_when_mime_missing(self) -> None:
        for name in ("a.png", "a.jpg", "a.jpeg", "a.gif", "a.webp", "a.bmp"):
            self.assertEqual(classify_media(name, ""), KIND_IMAGE, name)
        for name in ("a.mp3", "a.wav", "a.m4a", "a.aac", "a.amr"):
            self.assertEqual(classify_media(name, ""), KIND_AUDIO, name)
        for name in ("a.mp4", "a.mov", "a.webm", "a.avi"):
            self.assertEqual(classify_media(name, ""), KIND_VIDEO, name)
        self.assertEqual(classify_media("a.pdf", ""), KIND_DOCUMENT)

    def test_octet_stream_falls_back_to_extension(self) -> None:
        self.assertEqual(classify_media("voice.mp3", "application/octet-stream"), KIND_AUDIO)
        self.assertEqual(classify_media("clip.mov", "application/octet-stream"), KIND_VIDEO)

    def test_unknown_falls_back_to_document(self) -> None:
        self.assertEqual(classify_media("notes", ""), KIND_DOCUMENT)


class NormalizeMimeTests(unittest.TestCase):
    def test_mime_aliases(self) -> None:
        self.assertEqual(normalize_mime_type("audio/x-wav"), "audio/wav")
        self.assertEqual(normalize_mime_type("audio/x-m4a"), "audio/mp4")
        self.assertEqual(normalize_mime_type("image/jpg"), "image/jpeg")

    def test_strips_parameters_and_lowercases(self) -> None:
        self.assertEqual(normalize_mime_type("AUDIO/WAV; codecs=1"), "audio/wav")
        self.assertEqual(normalize_mime_type(" image/PNG "), "image/png")

    def test_missing_mime_guessed_from_filename(self) -> None:
        self.assertEqual(normalize_mime_type("", "a.mp3"), "audio/mpeg")
        self.assertEqual(normalize_mime_type("", "a.png"), "image/png")

    def test_unknown_mime_kept(self) -> None:
        self.assertEqual(normalize_mime_type("application/x-weird", "a.weird"), "application/x-weird")


class DataUriTests(unittest.TestCase):
    def test_to_data_uri_roundtrip(self) -> None:
        raw = b"hello-audio"
        uri = to_data_uri("audio/x-wav", raw)
        self.assertTrue(uri.startswith("data:audio/wav;base64,"))
        header, encoded = uri.split(",", 1)
        self.assertEqual(header, "data:audio/wav;base64")
        self.assertEqual(base64.b64decode(encoded), raw)

    def test_to_data_uri_uses_filename_fallback_mime(self) -> None:
        uri = to_data_uri("", b"png-bytes", "a.png")
        self.assertTrue(uri.startswith("data:image/png;base64,"))


class BuildInlineRefTests(unittest.TestCase):
    def test_image_ref_fields(self) -> None:
        ref = build_inline_ref("a.png", "image/png", b"png-bytes")

        self.assertEqual(ref["type"], "image")
        self.assertEqual(ref["showType"], "image")
        self.assertEqual(ref["file_class"], "vision")
        self.assertNotIn("context", ref)
        self.assertEqual(ref["status"], "uploaded")
        self.assertEqual(ref["greenNet"], "success")
        self.assertEqual(ref["id"], "")
        self.assertEqual(ref["size"], len(b"png-bytes"))
        self.assertEqual(ref["file"]["meta"]["size"], len(b"png-bytes"))
        self.assertEqual(ref["file"]["size"], len(b"png-bytes"))
        self.assertEqual(ref["file"]["type"], "image/png")
        self.assertEqual(ref["file_type"], "image/png")
        self.assertEqual(ref["name"], "a.png")
        self.assertTrue(ref["url"].startswith("data:image/png;base64,"))
        self.assertTrue(ref["itemId"])
        self.assertTrue(ref["uploadTaskId"])
        self.assertNotEqual(ref["itemId"], ref["uploadTaskId"])

    def test_audio_ref_fields(self) -> None:
        ref = build_inline_ref("voice.mp3", "audio/mpeg", b"mp3-bytes")

        self.assertEqual(ref["type"], "audio")
        self.assertEqual(ref["showType"], "audio")
        self.assertEqual(ref["file_class"], "audio")
        self.assertNotIn("context", ref)
        self.assertTrue(ref["url"].startswith("data:audio/mpeg;base64,"))
        self.assertEqual(ref["size"], len(b"mp3-bytes"))

    def test_video_ref_fields(self) -> None:
        ref = build_inline_ref("clip.mp4", "video/mp4", b"mp4-bytes")

        self.assertEqual(ref["type"], "video")
        self.assertEqual(ref["showType"], "video")
        self.assertEqual(ref["file_class"], "video")
        self.assertNotIn("context", ref)
        self.assertTrue(ref["url"].startswith("data:video/mp4;base64,"))

    def test_document_ref_fields(self) -> None:
        ref = build_inline_ref("notes.pdf", "application/pdf", b"%PDF-1.4")

        self.assertEqual(ref["type"], "file")
        self.assertEqual(ref["showType"], "file")
        self.assertEqual(ref["file_class"], "document")
        # 文档需要上游按全文解析
        self.assertEqual(ref["context"], "full")
        self.assertTrue(ref["url"].startswith("data:application/pdf;base64,"))

    def test_ids_are_unique_per_ref(self) -> None:
        first = build_inline_ref("a.png", "image/png", b"x")
        second = build_inline_ref("a.png", "image/png", b"x")

        self.assertNotEqual(first["itemId"], second["itemId"])
        self.assertNotEqual(first["uploadTaskId"], second["uploadTaskId"])

    def test_mime_is_normalized_in_ref(self) -> None:
        ref = build_inline_ref("voice", "audio/x-wav", b"wav")

        self.assertEqual(ref["file_type"], "audio/wav")
        self.assertEqual(ref["file"]["meta"]["content_type"], "audio/wav")
        self.assertTrue(ref["url"].startswith("data:audio/wav;base64,"))

    def test_empty_filename_gets_placeholder(self) -> None:
        ref = build_inline_ref("", "image/jpeg", b"jpg")

        self.assertTrue(ref["name"])
        self.assertIn(".", ref["name"])


class UpstreamUrlLengthContractTests(unittest.TestCase):
    """上游 `files[].url` 有硬长度上限（实测 2026-10-31，q2a-mm-test + 真实账号）。

    实测边界（同一 ref 结构，只改 url 长度）：

    - `data:image/png;base64,...` url_len=8182 → 200，模型正常描述图片
    - `data:image/png;base64,...` url_len=8198 → `RequestValidationError: url too long`
    - `data:audio/mpeg;base64,...` url_len=7810 → 200；url_len=8194 → `url too long`

    即上游对 url 的校验是 `max_length=8192`（字符数，与模态无关）。内联 ref 的
    url 一旦超过该值，整条请求会被上游拒绝，因此构造 ref 时必须按 url 长度而不是
    按原始字节数做上限判断。

    注：上游把这个校验错误**塞在 HTTP 200 的 content 里**返回，早期只看响应前
    200 字符的探测因此误判成"大图也能过"。修正后的结论是：超过 ~6KB 二进制就
    必须拒绝（图片先自动压缩），绝不把超长 url 发出去。
    """

    #: 实测上游 url 字段上限（字符数）。
    UPSTREAM_URL_MAX_LENGTH = 8192

    def test_small_media_url_fits_upstream_limit(self) -> None:
        ref = build_inline_ref("a.png", "image/png", b"x" * 100)

        self.assertLessEqual(len(ref["url"]), self.UPSTREAM_URL_MAX_LENGTH)

    def test_large_media_url_must_not_exceed_upstream_limit(self) -> None:
        # 100KB 非图片字节 → base64 约 136KB，既装不进 url 上限，也无法压缩，
        # 因此正确行为是**拒绝**（抛 InlineMediaTooLargeError → 上层 400），
        # 而不是把超长 url 发出去换一条 200 + "url too long"。
        with self.assertRaises(InlineMediaTooLargeError) as ctx:
            build_inline_ref("big.png", "image/png", b"x" * (100 * 1024))

        error = ctx.exception
        self.assertGreater(error.url_chars, self.UPSTREAM_URL_MAX_LENGTH)
        self.assertLessEqual(error.limit, self.UPSTREAM_URL_MAX_LENGTH)
        self.assertEqual(error.kind, KIND_IMAGE)
        self.assertEqual(error.filename, "big.png")

    def test_audio_media_url_must_not_exceed_upstream_limit(self) -> None:
        # 8.5KB mp3（2s / 32kbps）→ base64 约 11.4KB，实测已被上游拒绝。
        with self.assertRaises(InlineMediaTooLargeError) as ctx:
            build_inline_ref("voice.mp3", "audio/mpeg", b"x" * 8577)

        error = ctx.exception
        self.assertGreater(error.url_chars, self.UPSTREAM_URL_MAX_LENGTH)
        self.assertLessEqual(error.limit, self.UPSTREAM_URL_MAX_LENGTH)
        self.assertIn("audio attachment 'voice.mp3'", str(error))
        self.assertIn("upload via the file API", str(error))

    def test_effective_limit_keeps_margin_below_upstream_cap(self) -> None:
        limit = resolve_url_chars_limit()

        self.assertLessEqual(limit, self.UPSTREAM_URL_MAX_LENGTH)
        self.assertGreater(limit, 4096, "margin should not eat most of the budget")
        # 数据 URI 前缀也要计入长度：上限内的小图必须能构造成功。
        self.assertLessEqual(len(to_data_uri("image/png", b"y" * 100)), limit)


if __name__ == "__main__":
    unittest.main()
