"""附件链路的统一错误类型与文案构造。

背景（实测结论见 docs/superpowers/specs/2026-10-31-multimodal-attachments-findings.md §3）：

上游 `chat/completions` 的 `messages[0].files[i].url` 是 Pydantic `max_length=8192`
的**字符串**字段（实测 url_len=8182 通过、8198 被拒，错误体 `url too long`）。
data URI 前缀 `data:<mime>;base64,` 也计入长度，因此内联只够装约 6KB 二进制
（base64 膨胀 1.33 倍）。超出后上游返回 `RequestValidationError`，且**以 HTTP 200
+ content 的形式**回包，网关若不识别就会把校验错误当成 assistant 正文返回。

本模块只放「错误类型 + 文案」，不依赖任何其它服务模块，避免循环导入：

- :class:`InlineMediaTooLargeError` —— 本地字节无法塞进 url 长度上限。
- :class:`UpstreamAttachmentRejectedError` —— 上游明确拒绝了带附件的请求。
"""

from __future__ import annotations

from typing import Final

#: 上游 `files[].url` 的硬上限（字符数，与模态无关）。
INLINE_URL_HARD_CAP: Final[int] = 8192

#: 实现里预留的边距：上游按字符数校验，不同 mime 前缀长度不同，留 92 字符余量。
INLINE_URL_CHARS_MARGIN: Final[int] = 92

#: 音频压缩到 32kbps 时的估算码率（bit/s），仅用于生成可执行的提示文案。
_AUDIO_HINT_BITRATE: Final[int] = 32_000


def human_bytes(size: int) -> str:
    """把字节数格式化成人类可读的字符串（如 `85.7KB`）。"""
    value = float(max(0, int(size or 0)))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{value:.0f}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}GB"


def data_uri_prefix(mime: str) -> str:
    """返回 data URI 的头（含结尾逗号），长度会计入 url 字符数。"""
    return f"data:{mime or 'application/octet-stream'};base64,"


def data_uri_length(mime: str, size: int) -> int:
    """计算 `size` 字节二进制编码成 data URI 后的 url 字符数。"""
    payload_chars = 4 * ((max(0, int(size or 0)) + 2) // 3)
    return len(data_uri_prefix(mime)) + payload_chars


def inline_byte_capacity(max_url_chars: int, mime: str = "") -> int:
    """在给定 url 字符上限下，最多能内联多少字节的二进制。"""
    usable = int(max_url_chars or 0) - len(data_uri_prefix(mime))
    if usable <= 0:
        return 0
    return (usable * 3) // 4


def build_inline_too_large_message(
    *,
    kind: str,
    filename: str,
    size: int,
    url_chars: int,
    limit: int,
    mime: str = "",
) -> str:
    """生成可执行的超限提示（告诉调用方到底该改什么）。

    Args:
        kind: 附件种类（image / audio / video / document）。
        filename: 附件名，用于让用户定位是哪一份。
        size: 原始字节数。
        url_chars: 该字节编码成 data URI 后的 url 字符数。
        limit: 实际生效的 url 字符上限（已含边距）。
        mime: 归一化后的 mime。

    Returns:
        一行可直接展示给调用方的错误文案。
    """
    capacity = inline_byte_capacity(limit, mime)
    capacity_text = human_bytes(capacity)
    limit_text = f"url<={limit} chars"
    head = f"{kind} attachment '{filename}' is {human_bytes(size)}"
    if kind == "audio":
        max_seconds = (capacity * 8) / _AUDIO_HINT_BITRATE if capacity > 0 else 0.0
        return (
            f"{head}; inline limit is ~{capacity_text} ({limit_text}). "
            f"Shorten to <={max_seconds:.1f}s @32kbps, or upload via the file API."
        )
    if kind == "video":
        return (
            f"{head}; inline limit is ~{capacity_text} ({limit_text}) — inline video is not "
            f"viable. Upload via the file API."
        )
    if kind == "image":
        return (
            f"{head} (url would be {url_chars} chars); inline limit is ~{capacity_text} "
            f"({limit_text}). The image could not be compressed below the limit; "
            f"resize it, or upload via the file API."
        )
    return (
        f"{head} (url would be {url_chars} chars); inline limit is ~{capacity_text} "
        f"({limit_text}). Documents need the file upload + parse API."
    )


class InlineMediaTooLargeError(ValueError):
    """本地字节编码成 data URI 后超过上游 url 长度上限。

    上层必须把它映射成 **HTTP 4xx**（`code=attachment_too_large`），
    绝不允许把超长 url 发给上游 —— 那只会换来一条
    `RequestValidationError: url too long`，且上游以 200 + content 返回。
    """

    code = "attachment_too_large"

    def __init__(
        self,
        message: str,
        *,
        filename: str = "",
        url_chars: int = 0,
        limit: int = 0,
        size: int = 0,
        kind: str = "",
    ) -> None:
        super().__init__(str(message))
        self.message = str(message)
        self.filename = str(filename or "")
        self.url_chars = int(url_chars or 0)
        self.limit = int(limit or 0)
        self.size = int(size or 0)
        self.kind = str(kind or "")

    def __str__(self) -> str:
        return self.message


#: 上游把「附件校验失败」塞进 200 + content 时的错误码（小写比较）。
_ATTACHMENT_REJECTION_CODES: Final[frozenset[str]] = frozenset(
    {
        "requestvalidationerror",
        "request_validation_error",
        "validation_error",
        "validationerror",
        "invalid_input",
        "invalidinput",
        "invalidparameter",
        "invalid_argument",
        "bad_request",
        "url_too_long",
    }
)

#: details 里出现这些片段即判定为「附件被拒」。
_ATTACHMENT_REJECTION_MARKERS: Final[tuple[str, ...]] = (
    "url too long",
    "files ->",
    "files.0.url",
    "files[0].url",
    "too long",
    "invalid url",
    "file validation",
)


def looks_like_attachment_rejection(code: str, details: str) -> bool:
    """判断上游错误码 / 详情是否属于「附件被上游拒绝」。"""
    code_text = str(code or "").strip().lower()
    details_text = str(details or "").strip().lower()
    if code_text in _ATTACHMENT_REJECTION_CODES:
        return True
    return any(marker in details_text for marker in _ATTACHMENT_REJECTION_MARKERS)


class UpstreamAttachmentRejectedError(Exception):
    """上游明确拒绝了带附件的请求（如 `url too long` / `invalid_input`）。

    该错误必须以 4xx（`code=upstream_attachment_rejected`）回给调用方，
    **不能**降级成 `200 + content="[upstream_error] ..."` —— 那会把上游的
    参数校验失败伪装成模型回答，调用方完全看不出附件根本没送到。
    """

    code = "upstream_attachment_rejected"

    def __init__(self, message: str, *, upstream_code: str = "", details: str = "") -> None:
        super().__init__(str(message))
        self.message = str(message)
        self.upstream_code = str(upstream_code or "")
        self.details = str(details or "")

    def __str__(self) -> str:
        return self.message


def attachment_rejection_from_upstream_error(
    *,
    code: str,
    details: str,
    has_files: bool,
) -> UpstreamAttachmentRejectedError | None:
    """把上游错误事件翻译成 :class:`UpstreamAttachmentRejectedError`。

    仅当「请求确实带了附件」且「错误形态像附件校验失败」时才翻译；
    纯文本请求的上游错误保持既有的 `[upstream_error]` 行为不变。

    Args:
        code: 上游错误码（如 `RequestValidationError`）。
        details: 上游错误详情。
        has_files: 本次请求是否带了 `files`（upstream_files 非空）。

    Returns:
        翻译后的异常；不该翻译时返回 None。
    """
    if not has_files:
        return None
    if not looks_like_attachment_rejection(code, details):
        return None
    detail_text = str(details or "").strip()
    message = (
        f"upstream rejected the attachment payload (code={str(code or 'unknown')}"
        + (f": {detail_text}" if detail_text else "")
        + "). The attachment could not be delivered; retry with a smaller attachment "
        "(images are auto-compressed; audio/video must go through the file API)."
    )
    return UpstreamAttachmentRejectedError(message, upstream_code=str(code or ""), details=detail_text)


__all__ = [
    "INLINE_URL_CHARS_MARGIN",
    "INLINE_URL_HARD_CAP",
    "InlineMediaTooLargeError",
    "UpstreamAttachmentRejectedError",
    "attachment_rejection_from_upstream_error",
    "build_inline_too_large_message",
    "data_uri_length",
    "data_uri_prefix",
    "human_bytes",
    "inline_byte_capacity",
    "looks_like_attachment_rejection",
]
