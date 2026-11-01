"""内联媒体 ref 构造（图片 / 音频 / 视频 / 文档）。

背景（实测结论见 docs/superpowers/specs/2026-10-31-multimodal-attachments-findings.md）：

- 上游 `chat/completions` 的 `messages[0].files[i].url` 直接放
  `data:<mime>;base64,<...>` 就能让模型真正看到 / 听到内容，`id` 传空串也 OK。
- 因此默认不再走 OSS 上传（`/api/v2/files/getstsToken` 现已被阿里云 WAF
  人机验证拦住），改成「本地字节 → data URI 内联 ref」。

本模块只负责「字节 → ref 字典」的纯函数构造，不做任何 I/O，便于单测。
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import time
import uuid
from typing import Any, Iterator

from backend.services.attachment_errors import (
    INLINE_URL_CHARS_MARGIN,
    INLINE_URL_HARD_CAP,
    InlineMediaTooLargeError,
    build_inline_too_large_message,
    data_uri_length,
)

log = logging.getLogger("qwen2api.inline_media_ref")

#: 上游 `files[].url` 上限（字符数，实测 Pydantic max_length=8192）。
DEFAULT_INLINE_URL_CHARS_LIMIT = INLINE_URL_HARD_CAP

#: 图片压缩策略：长边逐级缩小，配合质量阶梯，取第一个塞得进上限的结果。
_LONG_EDGE_STEPS: tuple[int, ...] = (768, 512, 384, 256, 192, 128)
_QUALITY_STEPS: tuple[int, ...] = (85, 60, 40)

KIND_IMAGE = "image"
KIND_AUDIO = "audio"
KIND_VIDEO = "video"
KIND_DOCUMENT = "document"

# 媒体种类 → 上游 ref 字段映射（文档第 2 节实测表）：
#   image    → type/showType=image,    file_class=vision
#   audio    → type/showType=audio,    file_class=audio
#   video    → type/showType=video,    file_class=video
#   document → type/showType=file,     file_class=document, 且额外带 context="full"
_KIND_TO_REF_TYPE: dict[str, str] = {
    KIND_IMAGE: "image",
    KIND_AUDIO: "audio",
    KIND_VIDEO: "video",
    KIND_DOCUMENT: "file",
}

_KIND_TO_FILE_CLASS: dict[str, str] = {
    KIND_IMAGE: "vision",
    KIND_AUDIO: "audio",
    KIND_VIDEO: "video",
    KIND_DOCUMENT: "document",
}

# 常见 mime 归一化：上游 / 浏览器给出的别名统一成标准 mime。
_MIME_ALIASES: dict[str, str] = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/mp4a-latm": "audio/mp4",
    "audio/mp3": "audio/mpeg",
    "audio/mpeg3": "audio/mpeg",
    "audio/x-mp3": "audio/mpeg",
    "audio/x-aac": "audio/aac",
    "video/x-m4v": "video/x-m4v",
    "video/quicktime": "video/quicktime",
}

# 扩展名兜底：mime 缺失或为 application/octet-stream 时按后缀判定种类。
_EXTENSION_KIND_MAP: dict[str, str] = {
    # 图片
    "png": KIND_IMAGE,
    "jpg": KIND_IMAGE,
    "jpeg": KIND_IMAGE,
    "gif": KIND_IMAGE,
    "webp": KIND_IMAGE,
    "bmp": KIND_IMAGE,
    "tif": KIND_IMAGE,
    "tiff": KIND_IMAGE,
    "svg": KIND_IMAGE,
    # 音频
    "mp3": KIND_AUDIO,
    "wav": KIND_AUDIO,
    "m4a": KIND_AUDIO,
    "aac": KIND_AUDIO,
    "amr": KIND_AUDIO,
    "ogg": KIND_AUDIO,
    "flac": KIND_AUDIO,
    "opus": KIND_AUDIO,
    # 视频
    "mp4": KIND_VIDEO,
    "mov": KIND_VIDEO,
    "webm": KIND_VIDEO,
    "avi": KIND_VIDEO,
    "mkv": KIND_VIDEO,
    "m4v": KIND_VIDEO,
    "mpeg": KIND_VIDEO,
    "mpg": KIND_VIDEO,
}

# 扩展名 → 默认 mime（mime 缺失时补齐，避免上游按 octet-stream 处理）。
_EXTENSION_MIME_MAP: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "amr": "audio/amr",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "m4v": "video/x-m4v",
}


def _extension_of(filename: str) -> str:
    """取小写扩展名（不含点）；无扩展名返回空串。"""
    name = str(filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].strip().lower()


def _base_mime(content_type: str) -> str:
    """剥离 mime 参数（如 `audio/wav; codecs=1`）并小写化。"""
    text = str(content_type or "").strip().lower()
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    return text


def normalize_mime_type(content_type: str, filename: str = "") -> str:
    """归一化 mime：去参数、别名修正、缺失时按扩展名兜底。

    - `audio/x-wav` → `audio/wav`
    - `audio/x-m4a` → `audio/mp4`
    - `image/jpg`   → `image/jpeg`
    - 空 / `application/octet-stream` → 按文件名后缀推断
    """
    mime = _base_mime(content_type)
    if mime in _MIME_ALIASES:
        mime = _MIME_ALIASES[mime]
    if mime and mime != "application/octet-stream":
        return mime
    ext = _extension_of(filename)
    if ext and ext in _EXTENSION_MIME_MAP:
        return _EXTENSION_MIME_MAP[ext]
    guessed = mimetypes.guess_type(str(filename or "").strip())[0]
    if guessed:
        return _base_mime(guessed)
    return mime or "application/octet-stream"


def classify_media(filename: str = "", content_type: str = "") -> str:
    """判定附件种类：`image` / `audio` / `video` / `document`。

    优先用 mime 判定，mime 不可用时回落到扩展名；两者都判定不了时按文档处理
    （文档是唯一需要上游 parse 的种类，保守归入可保证走完整链路）。
    """
    mime = normalize_mime_type(content_type, filename)
    if mime.startswith("image/"):
        return KIND_IMAGE
    if mime.startswith("audio/"):
        return KIND_AUDIO
    if mime.startswith("video/"):
        return KIND_VIDEO
    ext = _extension_of(filename)
    if ext in _EXTENSION_KIND_MAP:
        return _EXTENSION_KIND_MAP[ext]
    return KIND_DOCUMENT


def default_filename(kind: str, content_type: str = "") -> str:
    """无文件名时给一个带正确后缀的占位名，便于下游按后缀兜底判定。"""
    mime = normalize_mime_type(content_type)
    ext = (mimetypes.guess_extension(mime) or "").lstrip(".")
    if not ext:
        ext = {
            KIND_IMAGE: "png",
            KIND_AUDIO: "mp3",
            KIND_VIDEO: "mp4",
            KIND_DOCUMENT: "txt",
        }.get(kind, "bin")
    # mimetypes 对 jpeg 会给出 .jpg，对 mpeg 音频会给出 .mp3，这里都可用。
    return f"attachment.{ext}"


def resolve_url_chars_limit(explicit: int | None = None) -> int:
    """解析实际生效的 url 字符上限（已扣掉边距，且绝不超过上游硬上限）。

    Args:
        explicit: 调用方显式给出的上限；None 时读 `INLINE_ATTACHMENT_MAX_URL_CHARS`。

    Returns:
        实际用于门控的 url 字符数上限（默认 8192 - 92 = 8100）。
    """
    if explicit is None:
        limit = DEFAULT_INLINE_URL_CHARS_LIMIT
        try:
            from backend.core.config import settings as runtime_settings

            limit = int(getattr(runtime_settings, "INLINE_ATTACHMENT_MAX_URL_CHARS", limit) or limit)
        except Exception:  # pragma: no cover - 配置模块缺失时回落到默认值
            limit = DEFAULT_INLINE_URL_CHARS_LIMIT
    else:
        limit = int(explicit or 0) or DEFAULT_INLINE_URL_CHARS_LIMIT
    # 上游是硬上限，配置写大了也不能突破；再留出 mime 前缀长度差异的边距。
    limit = min(limit, DEFAULT_INLINE_URL_CHARS_LIMIT)
    return max(1, limit - INLINE_URL_CHARS_MARGIN)


def _load_pillow():
    """惰性导入 Pillow；未安装时返回 None（由调用方降级为「拒绝」）。"""
    try:
        from PIL import Image  # type: ignore import-not-found
    except Exception:  # pragma: no cover - pillow 未安装
        return None
    return Image


def _image_has_alpha(image: Any) -> bool:
    """判断图片是否带透明通道（决定优先编码成 WebP 还是 JPEG）。"""
    try:
        mode = str(getattr(image, "mode", "") or "").upper()
        if mode in {"RGBA", "LA", "PA", "RGBa", "La", "BGRA"}:
            return True
        info = getattr(image, "info", {}) or {}
        return "transparency" in info
    except Exception:
        return False


def _resize_to_long_edge(image: Any, long_edge: int) -> Any:
    """按长边等比缩放（不放大）。"""
    width, height = image.size
    longest = max(int(width or 0), int(height or 0))
    if longest <= 0 or longest <= long_edge:
        return image
    scale = float(long_edge) / float(longest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        resample = _pillow_resample_lanczos()
        return image.resize(target, resample)
    except Exception:
        return image.resize(target)


def _pillow_resample_lanczos() -> Any:
    """取 Pillow 的 LANCZOS 常量（不同版本命名不同）。"""
    from PIL import Image  # type: ignore import-not-found

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _encode_image(image: Any, fmt: str, *, quality: int | None = None, lossless: bool = False) -> bytes | None:
    """把 PIL 图像编码成字节；失败返回 None。"""
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {}
    try:
        if fmt == "JPEG":
            kwargs.update({"quality": int(quality or 85), "optimize": True, "progressive": False})
        elif fmt == "WEBP":
            if lossless:
                kwargs.update({"lossless": True, "quality": 100, "method": 6})
            else:
                kwargs.update({"quality": int(quality or 85), "method": 4})
        image.save(buffer, format=fmt, **kwargs)
    except Exception:
        return None
    data = buffer.getvalue()
    return data or None


def _image_candidates(image: Any, *, alpha: bool) -> Iterator[tuple[str, bytes]]:
    """按「长边阶梯 × 质量阶梯」产出候选编码（先无损/高质量，再逐步降质）。"""
    for long_edge in _LONG_EDGE_STEPS:
        resized = _resize_to_long_edge(image, long_edge)
        if alpha:
            # PNG 带透明：优先 WebP 无损（小图标常能直接过），再退化到有损 WebP。
            lossless = _encode_image(resized, "WEBP", lossless=True)
            if lossless:
                yield "image/webp", lossless
            for quality in _QUALITY_STEPS:
                payload = _encode_image(resized, "WEBP", quality=quality)
                if payload:
                    yield "image/webp", payload
            continue
        rgb = resized
        try:
            if str(getattr(resized, "mode", "")) != "RGB":
                rgb = resized.convert("RGB")
        except Exception:
            rgb = resized
        for quality in _QUALITY_STEPS:
            payload = _encode_image(rgb, "JPEG", quality=quality)
            if payload:
                yield "image/jpeg", payload
        for quality in _QUALITY_STEPS:
            payload = _encode_image(rgb, "WEBP", quality=quality)
            if payload:
                yield "image/webp", payload


def compress_image_to_fit(
    raw: bytes,
    *,
    filename: str = "",
    max_url_chars: int | None = None,
) -> tuple[bytes, str] | None:
    """把图片压缩到「data URI 字符数 <= 上限」以内。

    策略：长边逐级缩小（768→512→384→256→192→128）+ 质量阶梯（85→60→40），
    取第一个满足上限的编码结果。带透明的图优先 WebP 无损，再退化到有损 WebP；
    不带透明的图优先 JPEG，其次 WebP。

    Args:
        raw: 原始图片字节。
        filename: 附件名（仅用于日志与错误文案）。
        max_url_chars: 显式 url 字符上限；None 时读配置。

    Returns:
        `(compressed_bytes, mime)`；压不进上限 / 不是可解码图片 / Pillow 缺失时返回 None。
    """
    data = bytes(raw or b"")
    if not data:
        return None
    Image = _load_pillow()
    if Image is None:  # pragma: no cover - pillow 未安装
        log.warning("[InlineMedia] pillow not available; cannot compress filename=%s", filename)
        return None
    limit = resolve_url_chars_limit(max_url_chars)
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            alpha = _image_has_alpha(opened)
            base = opened.copy()
    except Exception as exc:
        log.warning("[InlineMedia] image decode failed filename=%s error=%s", filename, exc)
        return None

    for mime, payload in _image_candidates(base, alpha=alpha):
        if data_uri_length(mime, len(payload)) <= limit:
            log.info(
                "[InlineMedia] image compressed filename=%s original_bytes=%s compressed_bytes=%s url_chars=%s limit=%s mime=%s",
                filename,
                len(data),
                len(payload),
                data_uri_length(mime, len(payload)),
                limit,
                mime,
            )
            return payload, mime
    log.warning(
        "[InlineMedia] image cannot be compressed under limit filename=%s original_bytes=%s limit=%s",
        filename,
        len(data),
        limit,
    )
    return None


def to_data_uri(
    content_type: str,
    raw: bytes,
    filename: str = "",
    *,
    max_url_chars: int | None = None,
) -> str:
    """把字节编码成 `data:<mime>;base64,<payload>`，并校验 url 字符数。

    Args:
        content_type: 附件 mime（会被归一化）。
        raw: 原始字节。
        filename: 附件名（mime 缺失时按后缀兜底，并写进错误文案）。
        max_url_chars: 显式 url 字符上限；None 时读配置。

    Returns:
        data URI 字符串。

    Raises:
        InlineMediaTooLargeError: 最终 url 字符数超过上限。
    """
    mime = normalize_mime_type(content_type, filename)
    data = bytes(raw or b"")
    payload = base64.b64encode(data).decode("ascii")
    uri = f"data:{mime};base64,{payload}"
    limit = resolve_url_chars_limit(max_url_chars)
    if len(uri) > limit:
        kind = classify_media(filename, mime)
        name = str(filename or "").strip() or default_filename(kind, mime)
        message = build_inline_too_large_message(
            kind=kind,
            filename=name,
            size=len(data),
            url_chars=len(uri),
            limit=limit,
            mime=mime,
        )
        raise InlineMediaTooLargeError(
            message,
            filename=name,
            url_chars=len(uri),
            limit=limit,
            size=len(data),
            kind=kind,
        )
    return uri


def _mime_extension(mime: str, fallback: str) -> str:
    """按 mime 取扩展名（压缩后需要让文件名与 mime 保持一致）。"""
    guessed = (mimetypes.guess_extension(str(mime or "")) or "").lstrip(".")
    if guessed == "jpe":
        guessed = "jpg"
    return guessed or fallback


def _renamed_for_mime(name: str, mime: str, fallback_ext: str) -> str:
    """把文件名后缀换成与 mime 一致的扩展名（`a.png` + `image/webp` → `a.webp`）。"""
    stem = str(name or "").strip() or "attachment"
    if "." in stem:
        stem = stem.rsplit(".", 1)[0] or "attachment"
    return f"{stem}.{_mime_extension(mime, fallback_ext)}"


def build_inline_ref(
    filename: str,
    content_type: str,
    raw: bytes,
    *,
    max_url_chars: int | None = None,
    allow_compression: bool = True,
) -> dict[str, Any]:
    """构造上游 `messages[0].files[i]` 可用的内联 ref。

    `url` 放 data URI（实测模型可真正看到 / 听到内容），`id` 传空串即可；
    `itemId` / `uploadTaskId` 用 uuid4 保证每次请求唯一。

    url 字符数超过上限时：

    - **图片**：用 Pillow 自动降采样 / 重编码，压进上限就用压缩后的字节；
    - **音频 / 视频 / 文档**：无法在 ~6KB 内承载，直接抛
      :class:`InlineMediaTooLargeError`（上层映射成 HTTP 400）。

    Args:
        filename: 附件名。
        content_type: 附件 mime。
        raw: 原始字节。
        max_url_chars: 显式 url 字符上限；None 时读配置。
        allow_compression: 是否允许对图片做自动压缩。

    Returns:
        上游可直接使用的 ref 字典。

    Raises:
        InlineMediaTooLargeError: 无法在 url 上限内表达该附件。
    """
    data = bytes(raw or b"")
    kind = classify_media(filename, content_type)
    mime = normalize_mime_type(content_type, filename)
    name = str(filename or "").strip() or default_filename(kind, mime)

    try:
        url = to_data_uri(mime, data, name, max_url_chars=max_url_chars)
    except InlineMediaTooLargeError:
        if kind != KIND_IMAGE or not allow_compression:
            raise
        compressed = compress_image_to_fit(data, filename=name, max_url_chars=max_url_chars)
        if compressed is None:
            raise
        data, mime = compressed
        name = _renamed_for_mime(name, mime, "webp")
        url = to_data_uri(mime, data, name, max_url_chars=max_url_chars)
    ref_type = _KIND_TO_REF_TYPE.get(kind, "file")
    file_class = _KIND_TO_FILE_CLASS.get(kind, "document")
    now_ms = int(time.time() * 1000)
    size = len(data)

    ref: dict[str, Any] = {
        "type": ref_type,
        # 浏览器 File 对象镜像字段，与官网抓包结构保持一致。
        "file": {
            "created_at": now_ms,
            "data": {},
            "filename": name,
            "hash": None,
            "id": "",
            "user_id": "",
            "meta": {
                "name": name,
                "size": size,
                "content_type": mime,
            },
            "update_at": now_ms,
            "lastModified": now_ms,
            "name": name,
            "webkitRelativePath": "",
            "size": size,
            "type": mime,
        },
        "id": "",
        "url": url,
        "name": name,
        "collection_name": "",
        "progress": 0,
        "status": "uploaded",
        # 官网抓包：视频在绿网校验完成前是 "greening"，图片/音频/文档是 "success"
        "greenNet": "greening" if kind == KIND_VIDEO else "success",
        "size": size,
        "error": "",
        "itemId": str(uuid.uuid4()),
        "file_type": mime,
        "showType": ref_type,
        "file_class": file_class,
        "uploadTaskId": str(uuid.uuid4()),
    }
    # 文档需要上游按全文解析（parsedFileTypes=[FILE]）；图片/音频/视频不加该键。
    if kind == KIND_DOCUMENT:
        ref["context"] = "full"
    return ref
