"""附件预处理：把三种 surface 的多模态附件归一化成内部统一形态。

支持面：

- **OpenAI chat**：`image_url`（data URI 与 http(s) URL）、`input_audio`、
  `audio_url`、`video_url`、`input_video`、`file` / `input_file`
- **Responses API**：`input_image`（`image_url` 或 `file_id`）、`input_audio`、
  `input_video` / `video_url`、`input_file`
- **Anthropic messages**：`{"type":"image","source":{...}}`、
  `{"type":"document","source":{...}}`（`base64` / `url` / `text` 三种 source）

统一后的 block 形态（供 `messages_to_prompt` 与附件上传链路消费）：

```python
{"type": "input_image", "file_id": "...", "mime_type": "image/png"}
{"type": "input_audio", "file_id": "...", "mime_type": "audio/wav"}
{"type": "input_video", "file_id": "...", "mime_type": "video/mp4"}
{"type": "input_file",  "file_id": "...", "filename": "...", "mime_type": "..."}
```

下载 / 解码后的字节统一落本地 `file_store`，再由
`context_attachment_manager` 决定「内联 data URI」还是「OSS 上传 + parse」。
"""

from __future__ import annotations

import base64
import binascii
import copy
import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote_to_bytes, urljoin, urlparse

import httpx

from backend.runtime.attachment_types import NormalizedAttachment
from backend.services.inline_media_ref import (
    KIND_AUDIO,
    KIND_DOCUMENT,
    KIND_IMAGE,
    KIND_VIDEO,
    classify_media,
    default_filename,
    normalize_mime_type,
)
from backend.services.model_modalities import ensure_modalities

log = logging.getLogger("qwen2api.attachment_preprocessor")

# 单个附件原始字节护栏：真正能否内联由 url 字符数决定（见 inline_media_ref），
# 这里只是防止调用方塞一个巨大的 base64 把进程内存打满。
_DEFAULT_INLINE_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
_REMOTE_FETCH_TIMEOUT_SECONDS = 20.0
_REMOTE_FETCH_MAX_REDIRECTS = 3
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})

# 明确拒绝的主机名（除 IP 判定外的兜底）。
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.tencentyun.com",
    }
)

# 音频 format 短名 → mime
_AUDIO_FORMAT_MIME: dict[str, str] = {
    "wav": "audio/wav",
    "wave": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "aac": "audio/aac",
    "amr": "audio/amr",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "webm": "audio/webm",
}

# 视频 format 短名 → mime
_VIDEO_FORMAT_MIME: dict[str, str] = {
    "mp4": "video/mp4",
    "m4v": "video/x-m4v",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
}


@dataclass(slots=True)
class PreprocessedAttachments:
    payload: dict[str, Any]
    attachments: list[NormalizedAttachment] = field(default_factory=list)
    uploaded_file_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _MediaSpec:
    """一个待处理的附件：要么自带字节，要么有远程 URL，要么引用已有 file_id。"""

    kind: str
    filename: str = ""
    content_type: str = ""
    data: bytes | None = None
    remote_url: str = ""
    file_id: str = ""
    origin: str = "inline"  # inline | remote | ref
    # file_id 解析不到本地文件时的兜底 spec（兼容同时带内联字节的 file 块）
    fallback: "_MediaSpec | None" = None


def _config_value(name: str, default: Any) -> Any:
    """惰性读取运行时配置，避免模块导入期强依赖 pydantic-settings。"""
    try:
        from backend.core.config import settings as runtime_settings
    except Exception:  # pragma: no cover - 配置模块缺失时回落到默认值
        return default
    return getattr(runtime_settings, name, default)


def _decode_data_uri(url: str) -> tuple[str, bytes]:
    """解析 `data:<mime>[;base64],<payload>`，返回 (mime, bytes)。"""
    header, separator, encoded = str(url or "").partition(",")
    if not separator:
        raise ValueError("malformed data URI")
    meta = header[len("data:"):] if header.startswith("data:") else header
    parts = [part.strip() for part in meta.split(";") if part.strip()]
    content_type = parts[0] if parts and "/" in parts[0] else ""
    is_base64 = any(part.lower() == "base64" for part in parts)
    if is_base64:
        raw = base64.b64decode(encoded, validate=False)
    else:
        raw = unquote_to_bytes(encoded)
    return normalize_mime_type(content_type), raw


def _b64decode(value: Any) -> bytes | None:
    """解码 base64 字符串（兼容直接给 data URI 的写法）。"""
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("data:"):
        try:
            return _decode_data_uri(text)[1]
        except Exception:
            return None
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError):
        return None


def _is_http_url(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _filename_from_url(url: str, kind: str) -> str:
    """从 URL 路径取文件名；取不到时给带正确后缀的占位名。"""
    try:
        name = urlparse(url).path.rsplit("/", 1)[-1].strip()
    except Exception:
        name = ""
    if name and "." in name:
        return name
    return default_filename(kind)


def _as_url(value: Any) -> str:
    """从 str / dict 里提取 URL 字段。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("url", "image_url", "audio_url", "video_url", "file_url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _format_mime(fmt: str, mapping: dict[str, str], default: str) -> str:
    """`wav` / `audio/wav` 两种写法都接受。"""
    text = str(fmt or "").strip().lower()
    if not text:
        return default
    if "/" in text:
        return normalize_mime_type(text)
    return mapping.get(text, default)


def _spec_from_url(kind_hint: str, url: str, filename: str = "", content_type: str = "") -> _MediaSpec | None:
    """data URI → 自带字节；http(s) → 远程下载；其它 scheme 忽略。"""
    text = str(url or "").strip()
    if not text:
        return None
    if text.startswith("data:"):
        try:
            mime, raw = _decode_data_uri(text)
        except Exception:
            log.warning("[Attachment] malformed data URI for kind=%s", kind_hint)
            return None
        kind = classify_media(filename, mime)
        if kind == KIND_DOCUMENT and kind_hint:
            kind = kind_hint
        name = str(filename or "").strip() or default_filename(kind, mime)
        return _MediaSpec(
            kind=kind,
            filename=name,
            content_type=normalize_mime_type(mime, name),
            data=raw,
            origin="inline",
        )
    if _is_http_url(text):
        kind = classify_media(filename, content_type)
        if kind == KIND_DOCUMENT and kind_hint:
            kind = kind_hint
        name = str(filename or "").strip() or _filename_from_url(text, kind)
        return _MediaSpec(
            kind=kind,
            filename=name,
            content_type=normalize_mime_type(content_type, name),
            remote_url=text,
            origin="remote",
        )
    log.warning("[Attachment] unsupported attachment url scheme for kind=%s", kind_hint)
    return None


def _spec_from_base64(
    kind: str,
    encoded: Any,
    content_type: str,
    filename: str = "",
) -> _MediaSpec | None:
    """base64 载荷（OpenAI `input_audio.data` 等）→ 自带字节的 spec。"""
    raw = _b64decode(encoded)
    if not raw:
        return None
    name = str(filename or "").strip() or default_filename(kind, content_type)
    return _MediaSpec(
        kind=kind,
        filename=name,
        content_type=normalize_mime_type(content_type, name),
        data=raw,
        origin="inline",
    )


def _extract_inline_file_payload(block: dict[str, Any]) -> tuple[str, str, bytes] | None:
    """解析 `file` / `input_file` 块里的内联字节。"""
    filename = str(block.get("filename") or block.get("name") or "attachment.txt")
    content_type = str(block.get("mime_type") or block.get("content_type") or "text/plain")

    # OpenAI Responses 的 `file_data`：data URI 或裸 base64
    if isinstance(block.get("file_data"), str):
        value = block["file_data"].strip()
        if value.startswith("data:"):
            try:
                ct, raw = _decode_data_uri(value)
            except Exception:
                return None
            return filename, ct, raw
        raw = _b64decode(value)
        return (filename, content_type, raw) if raw is not None else None
    if isinstance(block.get("text"), str):
        return filename, content_type, block["text"].encode("utf-8")
    if isinstance(block.get("content"), str) and not str(block.get("content", "")).startswith("data:"):
        return filename, content_type, block["content"].encode("utf-8")
    if isinstance(block.get("data_base64"), str):
        raw = _b64decode(block["data_base64"])
        return (filename, content_type, raw) if raw is not None else None
    if isinstance(block.get("data"), str):
        raw = _b64decode(block["data"])
        return (filename, content_type, raw) if raw is not None else None
    if isinstance(block.get("content"), str) and str(block.get("content", "")).startswith("data:"):
        try:
            ct, raw = _decode_data_uri(block["content"])
        except Exception:
            return None
        return filename, ct, raw
    return None


def _classify_anthropic_source(kind_hint: str, block: dict[str, Any]) -> _MediaSpec | None:
    """Anthropic `{"type":"image|document","source":{...}}` 归一化。"""
    source = block.get("source")
    filename = str(block.get("filename") or block.get("title") or "").strip()
    if isinstance(source, str):
        # 少数客户端直接给 data URI 字符串
        return _spec_from_url(kind_hint, source, filename=filename)
    if not isinstance(source, dict):
        return _spec_from_url(kind_hint, _as_url(block), filename=filename)

    source_type = str(source.get("type") or "").strip().lower()
    media_type = str(source.get("media_type") or source.get("mime_type") or "").strip()
    if source_type == "base64":
        content_type = normalize_mime_type(media_type, filename) if media_type else ""
        if kind_hint == KIND_DOCUMENT and not content_type:
            content_type = "text/plain"
        if kind_hint == KIND_IMAGE and not content_type:
            content_type = "image/png"
        return _spec_from_base64(kind_hint, source.get("data"), content_type, filename=filename)
    if source_type in {"url", "image_url"}:
        return _spec_from_url(kind_hint, str(source.get("url") or ""), filename=filename, content_type=media_type)
    if source_type == "text":
        text = source.get("data")
        if not isinstance(text, str) or not text:
            return None
        name = filename or "document.txt"
        return _MediaSpec(
            kind=KIND_DOCUMENT,
            filename=name,
            content_type=normalize_mime_type(media_type or "text/plain", name),
            data=text.encode("utf-8"),
            origin="inline",
        )
    return None


def _classify_block(block: dict[str, Any]) -> _MediaSpec | None:
    """把单个 content block 归一化成 `_MediaSpec`（不做任何 I/O）。"""
    block_type = str(block.get("type") or "").strip().lower()

    # ---- OpenAI chat ----
    if block_type == "image_url":
        return _spec_from_url(KIND_IMAGE, _as_url(block.get("image_url")))
    if block_type in {"input_audio", "audio_url"}:
        url = _as_url(block.get("audio_url")) or _as_url(block.get("url"))
        if not url and isinstance(block.get("input_audio"), str):
            url = block["input_audio"].strip()
        if url:
            return _spec_from_url(KIND_AUDIO, url)
        payload = block.get("input_audio") if isinstance(block.get("input_audio"), dict) else block
        content_type = _format_mime(
            str(payload.get("format") or payload.get("mime_type") or payload.get("content_type") or ""),
            _AUDIO_FORMAT_MIME,
            "audio/wav",
        )
        encoded = payload.get("data") or payload.get("base64") or payload.get("audio")
        return _spec_from_base64(KIND_AUDIO, encoded, content_type, filename=str(payload.get("filename") or ""))
    if block_type in {"video_url", "input_video"}:
        url = _as_url(block.get("video_url")) or _as_url(block.get("url"))
        if url:
            return _spec_from_url(KIND_VIDEO, url)
        payload = block.get("input_video") if isinstance(block.get("input_video"), dict) else block
        content_type = _format_mime(
            str(payload.get("format") or payload.get("mime_type") or payload.get("content_type") or ""),
            _VIDEO_FORMAT_MIME,
            "video/mp4",
        )
        encoded = payload.get("data") or payload.get("base64")
        return _spec_from_base64(KIND_VIDEO, encoded, content_type, filename=str(payload.get("filename") or ""))

    # ---- Responses API ----
    if block_type == "input_image":
        url = _as_url(block.get("image_url")) or _as_url(block.get("url"))
        if url:
            return _spec_from_url(KIND_IMAGE, url, filename=str(block.get("filename") or ""))
        file_id = str(block.get("file_id") or "").strip()
        if file_id:
            return _MediaSpec(
                kind=KIND_IMAGE,
                filename=str(block.get("filename") or ""),
                content_type=normalize_mime_type(str(block.get("mime_type") or "")),
                file_id=file_id,
                origin="ref",
            )
        return None

    # ---- 文档 / 通用文件 ----
    if block_type in {"file", "input_file"}:
        file_id = str(block.get("file_id") or "").strip()
        extracted = _extract_inline_file_payload(block)
        inline_spec: _MediaSpec | None = None
        if extracted is not None:
            filename, content_type, raw = extracted
            kind = classify_media(filename, content_type)
            inline_spec = _MediaSpec(
                kind=kind,
                filename=filename,
                content_type=normalize_mime_type(content_type, filename),
                data=raw,
                origin="inline",
            )
        if file_id:
            # 旧行为优先用 file_id 指向的本地文件；解析失败时回退内联字节。
            return _MediaSpec(
                kind=KIND_DOCUMENT,
                filename=str(block.get("filename") or ""),
                content_type=normalize_mime_type(str(block.get("mime_type") or "")),
                file_id=file_id,
                origin="ref",
                fallback=inline_spec,
            )
        return inline_spec

    # ---- Anthropic ----
    if block_type == "image":
        return _classify_anthropic_source(KIND_IMAGE, block)
    if block_type == "document":
        return _classify_anthropic_source(KIND_DOCUMENT, block)
    return None


def _block_needs_processing(block: dict[str, Any]) -> bool:
    """廉价预判：该 block 是否属于本模块负责的媒体类型。"""
    block_type = str(block.get("type") or "").strip().lower()
    return block_type in {
        "image_url",
        "input_image",
        "input_audio",
        "audio_url",
        "video_url",
        "input_video",
        "file",
        "input_file",
        "image",
        "document",
    }


def _rewritten_block(kind: str, file_id: str, filename: str, content_type: str) -> dict[str, Any]:
    """内部统一 block 形态。"""
    if kind == KIND_AUDIO:
        return {"type": "input_audio", "file_id": file_id, "mime_type": content_type}
    if kind == KIND_VIDEO:
        return {"type": "input_video", "file_id": file_id, "mime_type": content_type}
    if kind == KIND_IMAGE:
        return {"type": "input_image", "file_id": file_id, "mime_type": content_type}
    return {
        "type": "input_file",
        "file_id": file_id,
        "filename": filename,
        "mime_type": content_type,
    }


def _notice_block(spec: _MediaSpec, reason: str) -> dict[str, Any]:
    """附件被跳过时留一条文本说明，避免上游因缺内容直接 invalid_input。"""
    name = spec.filename or spec.kind
    return {"type": "text", "text": f"[Attachment skipped: {name} ({reason})]"}


def _source_for(spec: _MediaSpec) -> str:
    """标注附件来源，便于日志与排查。"""
    if spec.origin == "remote":
        return "remote-url"
    if spec.origin == "ref":
        return "manual-upload-ref"
    if spec.kind == KIND_DOCUMENT:
        return "manual-upload"
    return f"inline-{spec.kind}"


def _is_disallowed_ip(value: str) -> bool:
    """该 IP 字面量是否属于必须拒绝的网段（回环 / 私网 / 链路本地 / 保留 / 组播 / 未指定）。

    覆盖 `127.0.0.0/8`、`10/8`、`172.16/12`、`192.168/16`、`169.254/16`（云元数据）、
    `::1`、`fc00::/7`、`fe80::/10`，以及 IPv4-mapped IPv6（`::ffff:127.0.0.1`）。
    """
    try:
        address = ipaddress.ip_address(str(value or "").strip())
    except ValueError:
        return True
    candidates: list[Any] = [address]
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        candidates.append(mapped)
    for candidate in candidates:
        if (
            candidate.is_loopback
            or candidate.is_private
            or candidate.is_link_local
            or candidate.is_reserved
            or candidate.is_multicast
            or candidate.is_unspecified
            or getattr(candidate, "is_site_local", False)
        ):
            return True
    return False


def _resolve_host_addresses(host: str, port: int) -> list[str]:
    """解析主机名到全部 IP（任一落在禁用网段即整体拒绝）。"""
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    addresses: list[str] = []
    for info in infos or []:
        sockaddr = info[4] if len(info) > 4 else None
        if sockaddr:
            addresses.append(str(sockaddr[0]))
    return addresses


def _remote_url_is_safe(url: str) -> tuple[bool, str]:
    """校验远程附件 URL 是否允许出网抓取。

    Args:
        url: 待校验的 http(s) URL。

    Returns:
        `(allowed, reason)`；`allowed=False` 时 `reason` 说明拒绝原因（写日志用）。
    """
    if not _is_http_url(url):
        return False, "non-http(s) scheme"
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False, "unparsable url"
    host = str(parsed.hostname or "").strip()
    if not host:
        return False, "missing host"
    if host.lower() in _BLOCKED_HOSTNAMES:
        return False, f"blocked hostname {host}"
    try:
        port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    except (TypeError, ValueError):
        return False, "invalid port"
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_disallowed_ip(str(literal)):
            return False, f"private/reserved address {literal}"
        return True, ""
    try:
        addresses = _resolve_host_addresses(host, port)
    except (socket.gaierror, OSError, UnicodeError) as exc:
        # 解析不出来就无法证明它指向公网 → 保守拒绝（真正连接时也会失败）。
        return False, f"dns resolution failed: {exc}"
    if not addresses:
        return False, "dns resolution returned no address"
    for address in addresses:
        if _is_disallowed_ip(address):
            return False, f"resolves to private/reserved address {address}"
    return True, ""


async def _download_remote_media(url: str, *, max_bytes: int) -> tuple[str, bytes] | None:
    """下载远程附件：仅 http(s)、每跳重定向复检主机、20s 超时、上限字节数。

    SSRF 防护：调用方可以任意填写 `image_url.url`，因此每个请求（含每一跳
    重定向）都先解析主机并拒绝 loopback / private / link-local / reserved /
    multicast / unspecified 地址（含云元数据 169.254.169.254）。校验与真正连接
    之间的窗口由 httpx 在连接时再解析一次 DNS 收窄，且默认
    `INLINE_ATTACHMENT_REMOTE_FETCH=false` 时根本不会走到这里。
    """
    current = str(url or "").strip()
    if not _is_http_url(current):
        log.warning("[Attachment] reject non-http(s) remote url=%s", current[:120])
        return None
    try:
        async with httpx.AsyncClient(
            # 关闭自动跟随：每一跳都要自己复检主机，否则重定向到内网就绕过校验。
            follow_redirects=False,
            max_redirects=_REMOTE_FETCH_MAX_REDIRECTS,
            timeout=_REMOTE_FETCH_TIMEOUT_SECONDS,
        ) as client:
            for _hop in range(_REMOTE_FETCH_MAX_REDIRECTS + 1):
                allowed, reason = _remote_url_is_safe(current)
                if not allowed:
                    log.warning(
                        "[Attachment] blocked remote attachment url=%s reason=%s",
                        current[:120],
                        reason,
                    )
                    return None
                async with client.stream("GET", current) as response:
                    status = int(getattr(response, "status_code", 0) or 0)
                    headers = getattr(response, "headers", {}) or {}
                    if status in _REDIRECT_STATUS_CODES:
                        location = ""
                        try:
                            location = str(headers.get("location", "") or "").strip()
                        except Exception:
                            location = ""
                        if not location:
                            log.warning(
                                "[Attachment] redirect without location status=%s url=%s",
                                status,
                                current[:120],
                            )
                            return None
                        current = urljoin(current, location)
                        continue
                    if status >= 400:
                        log.warning("[Attachment] remote fetch failed status=%s url=%s", status, current[:120])
                        return None
                    content_type = ""
                    try:
                        content_type = str(headers.get("content-type", "") or "")
                    except Exception:
                        content_type = ""
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        total += len(chunk)
                        if max_bytes > 0 and total > max_bytes:
                            log.warning(
                                "[Attachment] remote media exceeds limit url=%s limit=%s",
                                current[:120],
                                max_bytes,
                            )
                            return None
                        chunks.append(bytes(chunk))
                    if total <= 0:
                        log.warning("[Attachment] remote media is empty url=%s", current[:120])
                        return None
                    return normalize_mime_type(content_type), b"".join(chunks)
            log.warning(
                "[Attachment] too many redirects url=%s limit=%s",
                url[:120],
                _REMOTE_FETCH_MAX_REDIRECTS,
            )
            return None
    except Exception as exc:  # 网络异常不能让整条请求 500
        log.warning("[Attachment] remote fetch error url=%s error=%s", url[:120], exc)
        return None


@dataclass(slots=True)
class _Materialized:
    """`_materialize_bytes` 的产物：字节 + 是否超过原始字节护栏。"""

    content_type: str
    raw: bytes
    oversized: bool = False


async def _materialize_bytes(
    spec: _MediaSpec,
    *,
    max_bytes: int,
    allow_remote: bool,
) -> _Materialized | None:
    """拿到 spec 的 (content_type, bytes)；拿不到返回 None。

    超过 `max_bytes` 时**仍然返回字节**并标记 `oversized=True`：落盘交给下游
    决定（图片压缩 / 上传回退 / 明确 4xx），不能在这里静默丢弃 —— 丢掉后
    `context_attachment_manager` 的上传回退分支永远走不到，附件会凭空消失。
    """
    if spec.data is not None:
        raw = spec.data
        if not raw:
            return None
        oversized = bool(max_bytes > 0 and len(raw) > max_bytes)
        if oversized:
            log.warning(
                "[Attachment] inline media exceeds byte guard filename=%s size=%s limit=%s; keeping for downstream fallback",
                spec.filename,
                len(raw),
                max_bytes,
            )
        return _Materialized(
            content_type=normalize_mime_type(spec.content_type, spec.filename),
            raw=raw,
            oversized=oversized,
        )

    if spec.remote_url:
        if not allow_remote:
            log.warning("[Attachment] remote attachment fetching disabled; skipped url=%s", spec.remote_url[:120])
            return None
        fetched = await _download_remote_media(spec.remote_url, max_bytes=max_bytes)
        if fetched is None:
            return None
        content_type, raw = fetched
        if not content_type:
            content_type = normalize_mime_type(spec.content_type, spec.filename)
        return _Materialized(content_type=content_type, raw=raw, oversized=False)

    return None


def _payload_has_media(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and _block_needs_processing(block):
                return True
    return False


async def preprocess_attachments(
    payload: dict[str, Any],
    file_store,
    owner_token: str | None = None,
) -> PreprocessedAttachments:
    """把 payload 里的多模态附件归一化 + 落本地，返回改写后的 payload。"""
    if not _payload_has_media(payload.get("messages")):
        return PreprocessedAttachments(payload=payload)

    max_bytes = int(_config_value("INLINE_ATTACHMENT_MAX_BYTES", _DEFAULT_INLINE_ATTACHMENT_MAX_BYTES) or 0)
    # 远程附件默认不抓取：网关代下会把调用方可控的 URL 变成 SSRF 跳板。
    allow_remote = bool(_config_value("INLINE_ATTACHMENT_REMOTE_FETCH", False))

    rewritten = copy.deepcopy(payload)
    messages = rewritten.get("messages") or []

    # 第一遍：只做判定（无 I/O），用于模态门控。
    locations: list[tuple[int, int, _MediaSpec]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            spec = _classify_block(block)
            if spec is not None:
                locations.append((message_index, block_index, spec))

    if not locations:
        return PreprocessedAttachments(payload=payload)

    # 模态门控：strict 模式抛 ModalityNotSupportedError，非 strict 返回需丢弃的模态。
    requested_kinds = {spec.kind for _, _, spec in locations}
    dropped_kinds = ensure_modalities(str(payload.get("model") or ""), requested_kinds)

    attachments: list[NormalizedAttachment] = []
    uploaded_file_ids: list[str] = []

    for message_index, block_index, spec in locations:
        content = messages[message_index]["content"]

        if spec.kind in dropped_kinds:
            content[block_index] = _notice_block(spec, "unsupported by selected model")
            continue

        # 引用已有本地文件（file_id 形态）：沿用原文件，不重复落盘。
        target = spec
        if target.file_id:
            existing = await file_store.get(target.file_id)
            same_owner = (
                not existing
                or not existing.get("owner_token")
                or existing.get("owner_token") == (owner_token or "")
            )
            if not existing or not same_owner:
                if target.fallback is not None:
                    # 同时带了内联字节：回退到内联字节重新落盘。
                    target = target.fallback
                else:
                    # 无法确认归属时保留原 block，交由后续链路按原样处理。
                    log.warning(
                        "[Attachment] file_id not found locally; keeping block as-is file_id=%s",
                        target.file_id,
                    )
                    continue
            else:
                existing_type = str(existing.get("content_type") or "application/octet-stream")
                existing_name = str(existing.get("filename") or target.filename or "attachment")
                kind = target.kind
                if kind == KIND_DOCUMENT:
                    kind = classify_media(existing_name, existing_type)
                attachments.append(NormalizedAttachment(
                    file_id=existing["id"],
                    filename=existing_name,
                    content_type=existing_type,
                    source=_source_for(target),
                    local_path=existing.get("path", ""),
                    sha256=existing.get("sha256", ""),
                    purpose="user-upload",
                ))
                content[block_index] = _rewritten_block(kind, existing["id"], existing_name, existing_type)
                continue

        resolved = await _materialize_bytes(target, max_bytes=max_bytes, allow_remote=allow_remote)
        if resolved is None:
            content[block_index] = _notice_block(target, "content could not be loaded")
            continue

        content_type, raw, oversized = resolved.content_type, resolved.raw, resolved.oversized
        kind = target.kind
        if kind == KIND_DOCUMENT:
            # mime 可能把种类判得更准（例如 .png 走了 file 块）
            kind = classify_media(target.filename, content_type)
        filename = target.filename or default_filename(kind, content_type)
        purpose = "vision" if kind == KIND_IMAGE else "upload"
        try:
            result = await file_store.save_bytes(filename, content_type, raw, purpose, owner_token=owner_token)
        except Exception as exc:
            log.warning("[Attachment] failed to persist attachment filename=%s error=%s", filename, exc)
            content[block_index] = _notice_block(target, "storage failed")
            continue

        uploaded_file_ids.append(result["id"])
        attachments.append(NormalizedAttachment(
            file_id=result["id"],
            filename=result["filename"],
            content_type=content_type,
            source=_source_for(target),
            local_path=result["path"],
            sha256=result["sha256"],
            purpose="user-upload",
            oversized=oversized,
        ))
        content[block_index] = _rewritten_block(kind, result["id"], result["filename"], content_type)

    return PreprocessedAttachments(
        payload=rewritten,
        attachments=attachments,
        uploaded_file_ids=uploaded_file_ids,
    )
