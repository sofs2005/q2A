from __future__ import annotations

import asyncio
import codecs
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from backend.core.config import resolve_model
from backend.core.upstream_file_cache import UpstreamFileCacheEntry
from backend.services.attachment_errors import InlineMediaTooLargeError
from backend.services.client_profiles import user_role_system_text
from backend.services.context_summarizer import count_history_messages, get_context_summarizer
from backend.services.document_parse_breaker import get_document_parse_breaker
from backend.services.inline_media_ref import (
    KIND_AUDIO,
    KIND_IMAGE,
    KIND_VIDEO,
    build_inline_ref,
    classify_media,
)
from backend.services.token_calc import count_tokens
from backend.services.standard_request_builder import _excluded_command_like_tool_names
from backend.services.upstream_file_uploader import (
    FileUploadRateLimitedError,
    UpstreamWafChallengeError,
    safe_signed_url_cache_expires_at,
)
from backend.toolcore.context_offload import SYSTEM_CONTEXT_FILE_PREFIX, SYSTEM_CONTEXT_PROMPT_NOTE
from backend.toolcore.request_normalizer import normalize_chat_request, to_prompt_payload


log = logging.getLogger("qwen2api.context_attachment_manager")

# 无法解析签名过期时间时，对带 x-oss-signature 的 URL 使用保守复用窗口（官方常见 300s - 30s）。
_DEFAULT_SIGNED_URL_CACHE_TTL_SECONDS = 270.0

# 浏览器上传路径的硬超时兜底值（秒）。比 `UPLOAD_BROWSER_TIMEOUT_SECONDS`（120s，单次
# 页面操作超时）短得多：浏览器一旦卡在启动 / 登录态注入 / 验证码上，整条请求就被拖死，
# 宁可降级走 HTTP 上传（或内联）也不要等满两分钟。
_DEFAULT_BROWSER_HARD_TIMEOUT_SECONDS = 45.0


def _remote_url_from_meta(remote_meta: dict[str, Any] | None) -> str:
    if not isinstance(remote_meta, dict):
        return ""
    remote_ref = remote_meta.get("remote_ref")
    if isinstance(remote_ref, dict):
        return str(remote_ref.get("url") or "")
    return str(remote_meta.get("url") or "")


def _resolve_attachment_cache_expires_at(
    remote_meta: dict[str, Any],
    *,
    created_at: float,
    session_ttl_seconds: float,
) -> float:
    """缓存过期取 min(会话 TTL, 签名 URL 安全过期)。

    官方 file_url 通常 x-oss-expires=300；超过后复用会触发 invalid_input。
    """
    session_expires = float(created_at) + max(60.0, float(session_ttl_seconds or 0.0))
    candidates: list[float] = [session_expires]

    url = _remote_url_from_meta(remote_meta)
    safe = safe_signed_url_cache_expires_at(url, now=created_at)
    if safe is not None:
        candidates.append(safe)
    else:
        raw_url_expires = remote_meta.get("url_expires_at")
        if raw_url_expires is not None:
            try:
                candidates.append(float(raw_url_expires) - 30.0)
            except (TypeError, ValueError):
                pass
        elif "x-oss-signature" in url.lower() or "x-oss-expires" in url.lower():
            candidates.append(float(created_at) + _DEFAULT_SIGNED_URL_CACHE_TTL_SECONDS)

    return min(candidates)


def _is_retryable_attachment_upload_error(exc: Exception) -> bool:
    lowered = str(exc or "").lower()
    markers = (
        "temporary failure in name resolution",
        "failed to resolve",
        "nameresolutionerror",
        "connecterror",
        "connectionerror",
        "max retries exceeded",
        "newconnectionerror",
        "timed out",
        "timeout",
    )
    return any(marker in lowered for marker in markers)


def _upload_account_excludes(account_pool) -> set:
    """已知文件上传超限的账号邮箱集合；旧账号池无此方法时返回空集。"""
    getter = getattr(account_pool, "file_upload_blocked_emails", None)
    if callable(getter):
        return set(getter() or ())
    return set()


def _upload_exit_pool_enabled(uploader: Any) -> bool:
    """出口池取 STS 这条链路是否启用。

    判定口径：上传器带 `exit_pool` 且池非空（`UPLOAD_EXIT_POOL` 配了出口）。

    - 池为空 / `UPLOAD_VIA_EXIT=never` → `ExitPool.from_settings()` 返回 None，
      上传器的 `exit_pool` 属性就是 None → 本函数 False，降级顺序退化为
      「浏览器 → 直连 HTTP」（与历史行为一致，一行代理都不碰）。
    - 单测替身可能只给一个裸 `exit_pool` 对象（无 `enabled` 属性）：视为启用。
    """
    pool = getattr(uploader, "exit_pool", None)
    if pool is None:
        return False
    enabled = getattr(pool, "enabled", None)
    if enabled is None:
        return True
    return bool(enabled)


def _browser_upload_enabled(settings_obj: Any, browser_proxy: Any) -> bool:
    """浏览器上传链路是否可用。

    两个条件同时满足才启用：代理实例存在（`app.state.browser_upload_proxy`）、
    解析出的模式不是 `off`（`UPLOAD_VIA_BROWSER=never`，或 `UPLOAD_BROWSER_MODE=remote`
    但没配 `UPLOAD_CDP_ENDPOINT`）。`local` 模式不需要 endpoint，因此 endpoint 为空
    也可能启用（用进程内置 Chromium）。

    注意 `UPLOAD_VIA_BROWSER` 的默认值现在是 `never`：该通道通过率尚未验证，而失败
    代价是「把请求拖慢几十秒到几分钟」，必须由部署显式打开。
    """
    if browser_proxy is None:
        return False
    # 延迟导入：playwright 是可选依赖，多数单测不打桩浏览器代理。
    from backend.services.browser_upload_proxy import BROWSER_MODE_OFF, resolve_browser_mode

    return resolve_browser_mode(settings_obj) != BROWSER_MODE_OFF


def _browser_hard_timeout_seconds(settings_obj: Any) -> float:
    """浏览器上传路径的独立硬超时（秒）。

    读 `UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS`，缺失 / 非法 / 非正值都退回
    :data:`_DEFAULT_BROWSER_HARD_TIMEOUT_SECONDS`（45s）。这个超时**独立于**
    `UPLOAD_BROWSER_TIMEOUT_SECONDS`：后者是单次页面操作（如 `goto`）的超时，
    整条链路加起来可能远超它，不能拿来做请求级护栏。
    """
    raw = getattr(settings_obj, "UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS", _DEFAULT_BROWSER_HARD_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BROWSER_HARD_TIMEOUT_SECONDS
    return value if value > 0 else _DEFAULT_BROWSER_HARD_TIMEOUT_SECONDS


def _text_from_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return str(content or "")


# 图片 / 音频 / 视频走「本地字节 → data URI 内联 ref」：不调上传、不调 parse、
# 不换号、不绑账号（实测上游 `files[i].url` 直接吃 data URI）。
_INLINE_MEDIA_KINDS = frozenset({KIND_IMAGE, KIND_AUDIO, KIND_VIDEO})
_DEFAULT_INLINE_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024


def _is_inline_media_attachment(attachment: Any) -> bool:
    """该附件是否可以直接内联（图片 / 音频 / 视频，且本地有字节）。"""
    if getattr(attachment, "remote_ref", None):
        return False
    if not getattr(attachment, "local_path", ""):
        return False
    kind = classify_media(
        str(getattr(attachment, "filename", "") or ""),
        str(getattr(attachment, "content_type", "") or ""),
    )
    return kind in _INLINE_MEDIA_KINDS


async def _inline_ref_for_attachment(
    attachment: Any,
    *,
    settings_obj: Any,
) -> tuple[dict[str, Any] | None, InlineMediaTooLargeError | None]:
    """读本地字节并构造内联 ref。

    返回 `(ref, too_large_error)`：

    - 成功：`(ref, None)`
    - 读不到 / 开关关闭 / 超过字节护栏：`(None, None)`，调用方回退上传
    - url 字符数超限且压不进上限：`(None, InlineMediaTooLargeError)`，
      调用方回退上传；若上传也不可用（WAF / 无账号）则把它抛出去换 4xx，
      绝不静默丢附件。
    """
    if not bool(getattr(settings_obj, "INLINE_ATTACHMENT_ENABLED", True)):
        return None, None
    max_bytes = int(getattr(settings_obj, "INLINE_ATTACHMENT_MAX_BYTES", _DEFAULT_INLINE_ATTACHMENT_MAX_BYTES) or 0)
    max_url_chars = getattr(settings_obj, "INLINE_ATTACHMENT_MAX_URL_CHARS", None)
    path = str(getattr(attachment, "local_path", "") or "")
    if not path:
        return None, None
    try:
        raw = await asyncio.to_thread(Path(path).read_bytes)
    except OSError as exc:
        log.warning(
            "[ContextAttachment] inline media read failed filename=%s error=%s",
            getattr(attachment, "filename", ""),
            exc,
        )
        return None, None
    if not raw:
        log.warning("[ContextAttachment] inline media is empty filename=%s", getattr(attachment, "filename", ""))
        return None, None
    if max_bytes > 0 and len(raw) > max_bytes:
        log.warning(
            "[ContextAttachment] inline media exceeds limit filename=%s size=%s limit=%s; falling back to upstream upload",
            getattr(attachment, "filename", ""),
            len(raw),
            max_bytes,
        )
        return None, None
    filename = str(getattr(attachment, "filename", "") or "")
    content_type = str(getattr(attachment, "content_type", "") or "")
    try:
        # 图片压缩是 CPU 活（Pillow 解码 + 多轮重编码），放到线程里跑。
        ref = await asyncio.to_thread(
            build_inline_ref,
            filename,
            content_type,
            raw,
            max_url_chars=(int(max_url_chars) if max_url_chars else None),
        )
    except InlineMediaTooLargeError as exc:
        log.warning(
            "[ContextAttachment] inline media too large filename=%s url_chars=%s limit=%s size=%s kind=%s; "
            "falling back to upstream upload",
            filename,
            exc.url_chars,
            exc.limit,
            exc.size,
            exc.kind,
        )
        return None, exc
    return ref, None


#: 纯文本类附件的 content_type 白名单（带 charset 参数时只看主类型）。
_INLINE_TEXT_CONTENT_TYPES = frozenset({
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/csv",
    "application/json",
    "application/x-ndjson",
    "application/jsonlines",
    "application/jsonl",
})
#: 纯文本类附件的扩展名白名单（content_type 缺失 / 给成 octet-stream 时兜底）。
_INLINE_TEXT_EXTENSIONS = frozenset({
    "txt", "md", "markdown", "csv", "json", "jsonl", "ndjson",
    "log", "yaml", "yml", "xml", "ini", "conf",
})
#: 解码顺序：utf-8 → utf-8-sig → gb18030 → latin-1（latin-1 永不失败，兜底）。
_TEXT_DECODE_ORDER = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
#: 二进制 sniff 只看前 4KB：出现 NUL 字节就按二进制处理，不当文本注入。
_BINARY_SNIFF_BYTES = 4096
#: 单文件注入字节上限默认值（128KB）。
_DEFAULT_INLINE_TEXT_MAX_BYTES = 128 * 1024
#: 内联上下文总字符预算默认值（8K 字符）。上游对超大内联 prompt 直接返回 x5sec
#: punish 页（实测 8k 通过 / 12k 必挂），所以内联正文必须有体积上限。
_DEFAULT_INLINE_TEXT_MAX_CHARS = 8000

#: 历史 blob 里每条消息的分块头（`## Message 12 [user]`），行首对齐。
#: 只认「行首 + 数字序号 + [role]」的形式，避免把正文里出现的 `## Message` 误当分块头。
_HISTORY_BLOCK_HEADER_RE = re.compile(r"^## Message \d+ \[[^\]]*\]\s*$", re.MULTILINE)
#: 判断某一行是否「像一个分块头」（用于把 header 与正文拆开）。
_HISTORY_BLOCK_HEADER_PREFIX_RE = re.compile(r"^## Message \d+ \[")
#: 历史 blob 被按整条丢弃时的提示行（放在最前面）。
_HISTORY_TRUNCATION_NOTE = "…（已省略前 {dropped} 条历史消息，原始 {original} 字符，保留最近 {kept} 字符）"
#: 只有一条消息且它本身就超预算时（退化成头部截断）的提示行。
_HISTORY_HEAD_TRUNCATION_NOTE = "…（已截断，原始 {original} 字符，保留开头 {kept} 字符）"
#: 单条消息/单个注入块自身被截断时，追加在**结尾**的提示（保留的是开头）。
_BLOCK_TRUNCATION_NOTE = "…（已截断，原消息 {original} 字符）"
#: 纯文本附件注入块被字符预算截断时的提示（保留的是开头，与文件开头更关键一致）。
_FILE_TRUNCATION_NOTE = "…（已截断，原文件 {original} 字符）"

#: 两段式总结：历史超过多少字符才值得多花一次上游请求（与内联硬预算同源）。
_DEFAULT_SUMMARY_THRESHOLD_CHARS = 8000
#: 摘要失败回退截断时，临时借号的等待上限（秒）。比主请求的 60s 短：
#: 摘要只是优化路径，不能为了它把已经降级的请求再拖一分钟。
_SUMMARY_ACCOUNT_TIMEOUT_SECONDS = 15.0
#: 摘要 blob 的头行（下游 prompt 里一眼能看出「这是摘要，不是原文」）。
_SUMMARY_BLOB_HEADER = "【历史摘要】"
#: 摘要 blob 的尾行（原始规模，让模型知道上下文被压缩过）。
_SUMMARY_BLOB_NOTE = "（原始 {original} 字符 / {messages} 条消息）"
#: 历史条数太少就不摘要：1-2 条历史压出来的摘要几乎等于原文，白花一次上游请求。
#: 切掉「当前任务」之后，短会话（只有一轮问答）会退化成 0-1 条历史 → 走原逻辑。
_MIN_SUMMARY_HISTORY_MESSAGES = 3

#: 「预算 <= 0 → 不限制」的 WARNING 只打一次，避免每个请求刷屏。
_inline_budget_warned = False


def _is_inline_text_attachment(attachment: Any) -> bool:
    """该附件是否可以「网关本地解码 → 直接注入 prompt」。

    条件：还没有远程 ref（没上传过）、本地有字节、mime 或扩展名落在文本白名单里。
    图片 / 音频 / 视频由 `_is_inline_media_attachment` 先行处理，不会走到这里。
    """
    if getattr(attachment, "remote_ref", None):
        return False
    if not getattr(attachment, "local_path", ""):
        return False
    content_type = str(getattr(attachment, "content_type", "") or "").strip().lower()
    main_type = content_type.split(";", 1)[0].strip()
    if main_type in _INLINE_TEXT_CONTENT_TYPES:
        return True
    ext = Path(str(getattr(attachment, "filename", "") or "")).suffix.lstrip(".").lower()
    return ext in _INLINE_TEXT_EXTENSIONS


def _looks_binary(head: bytes) -> bool:
    """前 4KB 里出现 NUL 字节即认为二进制。"""
    return b"\x00" in head[:_BINARY_SNIFF_BYTES]


def _decode_text_head(raw: bytes, codec: str) -> str | None:
    """按指定编码解码；截断点切在多字节字符中间时，丢尾部残缺字节再试一次。

    Returns:
        解码后的字符串；该编码不适用时返回 None（交给下一个候选编码）。
    """
    try:
        return raw.decode(codec)
    except LookupError:  # pragma: no cover - 运行环境缺该编码（极少见）
        return None
    except UnicodeDecodeError as exc:
        # 只有「错在尾部」才可能是我们自己截断造成的，丢掉残字节重试。
        if exc.start <= 0 or exc.start < len(raw) - 4:
            return None
        try:
            return raw[: exc.start].decode(codec)
        except UnicodeDecodeError:
            return None


def _decode_text_bytes(raw: bytes) -> tuple[str, str]:
    """按 `utf-8 → utf-8-sig → gb18030 → latin-1` 顺序解码。

    Returns:
        `(text, codec)`；全部失败返回 `("", "")`（latin-1 兜底后基本不会发生）。
    """
    candidates: list[str] = []
    if raw.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
    elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidates.append("utf-16")
    for codec in dict.fromkeys([*candidates, *_TEXT_DECODE_ORDER]):
        text = _decode_text_head(raw, codec)
        if text is not None:
            return text, codec
    return "", ""


def _format_inline_text_block(
    filename: str,
    text: str,
    *,
    truncated: bool,
    original_bytes: int,
) -> str:
    """拼装注入块：`[附件: name]` + 正文（截断时附一行提示）。"""
    body = text
    if truncated:
        original_kb = max(1, (int(original_bytes) + 1023) // 1024)
        body = f"{body}\n\n…（已截断，原文件 {original_kb} KB）"
    return f"[附件: {filename}]\n{body}"


def _read_text_head(path: str, max_bytes: int) -> tuple[bytes, int]:
    """读文件头部与真实大小（不把超大文件整读进内存）。

    Args:
        path: 本地路径。
        max_bytes: 头部字节上限；<=0 表示不限（整读）。

    Returns:
        `(head_bytes, total_bytes)`。
    """
    total = Path(path).stat().st_size
    with open(path, "rb") as handle:
        head = handle.read(max_bytes) if max_bytes > 0 else handle.read()
    return head, total


async def _inline_text_block_for_attachment(
    attachment: Any,
    *,
    settings_obj: Any,
    session_key: str,
) -> str:
    """把纯文本类附件解码成可直接拼进 prompt 的注入块。

    Returns:
        注入块文本；不满足条件（开关关闭 / 读不到 / 空文件 / 二进制 sniff 命中）
        返回空串，调用方回落到原上传路径。
    """
    if not bool(getattr(settings_obj, "ATTACHMENT_INLINE_TEXT_ENABLED", True)):
        return ""
    max_bytes = int(getattr(settings_obj, "ATTACHMENT_INLINE_TEXT_MAX_BYTES", _DEFAULT_INLINE_TEXT_MAX_BYTES) or 0)
    filename = str(getattr(attachment, "filename", "") or "")
    path = str(getattr(attachment, "local_path", "") or "")
    if not path:
        return ""
    try:
        raw, total_bytes = await asyncio.to_thread(_read_text_head, path, max_bytes)
    except OSError as exc:
        log.warning(
            "[ContextAttachment] inline text read failed session_key=%s filename=%s error=%s",
            session_key,
            filename,
            exc,
        )
        return ""
    if not raw:
        log.info(
            "[ContextAttachment] inline text skipped (empty) session_key=%s filename=%s",
            session_key,
            filename,
        )
        return ""
    if _looks_binary(raw):
        # 声明成 text/* 但实际是二进制：不当文本注入，回落到原上传路径。
        log.info(
            "[ContextAttachment] inline text skipped (binary sniff) session_key=%s filename=%s",
            session_key,
            filename,
        )
        return ""
    text, codec = _decode_text_bytes(raw)
    if not text.strip():
        log.info(
            "[ContextAttachment] inline text skipped (empty after decode) session_key=%s filename=%s",
            session_key,
            filename,
        )
        return ""
    truncated = max_bytes > 0 and total_bytes > max_bytes
    block = _format_inline_text_block(
        filename,
        text,
        truncated=truncated,
        original_bytes=total_bytes,
    )
    log.info(
        "[ContextAttachment] inline text injected session_key=%s filename=%s codec=%s chars=%s bytes=%s truncated=%s",
        session_key,
        filename,
        codec,
        len(text),
        total_bytes,
        truncated,
    )
    return block


async def _collect_inline_text_blocks(
    attachments: list[Any],
    *,
    file_store: Any,
    settings_obj: Any,
    session_key: str,
) -> tuple[list[str], int, list[Any]]:
    """批量处理纯文本类附件。

    Returns:
        `(blocks, tokens, leftovers)`：注入块列表、注入块的 token 数、
        需要回落到原上传路径的附件。
    """
    blocks: list[str] = []
    tokens = 0
    leftovers: list[Any] = []
    for attachment in attachments:
        block = await _inline_text_block_for_attachment(
            attachment,
            settings_obj=settings_obj,
            session_key=session_key,
        )
        if not block:
            leftovers.append(attachment)
            continue
        blocks.append(block)
        tokens += count_tokens(block)
        # 正文已进 payload，本地副本不再需要（与内联媒体一致）。
        delete_path = getattr(file_store, "delete_path", None)
        if callable(delete_path):
            try:
                await delete_path(str(getattr(attachment, "local_path", "") or ""))
            except Exception:  # pragma: no cover - 清理失败不影响注入结果
                log.debug("[ContextAttachment] inline text cleanup failed", exc_info=True)
    return blocks, tokens, leftovers


def _inline_char_budget(settings_obj: Any) -> int | None:
    """读取内联上下文总字符预算。

    Returns:
        正数预算；`None` 表示不限制（配置为 0 / 负数，或读不到时按默认值处理）。
    """
    global _inline_budget_warned
    raw = getattr(settings_obj, "ATTACHMENT_INLINE_TEXT_MAX_CHARS", _DEFAULT_INLINE_TEXT_MAX_CHARS)
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        budget = _DEFAULT_INLINE_TEXT_MAX_CHARS
    if budget <= 0:
        if not _inline_budget_warned:
            _inline_budget_warned = True
            log.warning(
                "[ContextAttachment] ATTACHMENT_INLINE_TEXT_MAX_CHARS=%s <= 0 → 内联上下文不限制体积"
                "（旧行为；上游对超大内联 prompt 会返回 x5sec punish 页）",
                raw,
            )
        return None
    return budget


def _remaining_budget(budget: int | None, used: int) -> int | None:
    """扣掉已占用字符后的剩余预算；预算不限制时继续返回 `None`。"""
    if budget is None:
        return None
    return max(0, budget - max(0, int(used)))


def _summary_enabled(settings_obj: Any) -> bool:
    """两段式总结总开关（`CONTEXT_SUMMARY_ENABLED`）。"""
    raw = getattr(settings_obj, "CONTEXT_SUMMARY_ENABLED", True)
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _summary_threshold(settings_obj: Any) -> int:
    """历史超过多少字符才触发摘要（`CONTEXT_SUMMARY_THRESHOLD_CHARS`）。"""
    try:
        return int(getattr(settings_obj, "CONTEXT_SUMMARY_THRESHOLD_CHARS", _DEFAULT_SUMMARY_THRESHOLD_CHARS))
    except (TypeError, ValueError):
        return _DEFAULT_SUMMARY_THRESHOLD_CHARS


def _summary_model_for(payload: Any, settings_obj: Any) -> str:
    """决定摘要请求用哪个上游模型。

    `CONTEXT_SUMMARY_MODEL` 为空时表示「跟主请求同模型」：这里取调用方请求的模型
    并过一遍别名解析（`resolve_model`）。解析结果不是 qwen 家族（比如调用方用
    `gpt-4.1` 这种网关别名）时，退回 `QWEN_PLUS_MODEL` —— 摘要只需要「便宜 + 会
    归纳」，用最便宜的 plus 比把未知模型名丢给上游直接 400 更实用。
    """
    explicit = str(getattr(settings_obj, "CONTEXT_SUMMARY_MODEL", "") or "").strip()
    if explicit:
        return explicit
    requested = str((payload or {}).get("model") or "") if isinstance(payload, dict) else ""
    resolved = resolve_model(requested)
    if resolved.lower().startswith("qwen"):
        return resolved
    fallback = str(getattr(settings_obj, "QWEN_PLUS_MODEL", "") or "").strip()
    return fallback or resolved or "qwen-plus"


async def _summarize_history_blob(
    *,
    blob: str,
    settings_obj: Any,
    app: Any = None,
    payload: Any = None,
    session_key: str = "",
    summary_account: Any = None,
) -> str | None:
    """把完整历史正文换成摘要 blob（两段式总结的第一段）。

    `blob` 必须是**历史前缀**（不含本次提问及其后的工具续跑尾部）：调用方要么用
    `_split_current_task()` 过滤后的消息序列化，要么用
    `_trim_current_task_from_blob()` 从 `plan.summary_text` 里切掉当前任务。

    Args:
        blob: 历史前缀正文（`## Message N [role]` 格式）。
        settings_obj: 配置对象（`context_offloader.settings`）。
        app: FastAPI app；用来取 `qwen_client` 与（必要时）借一个上游账号。
        payload: 原始请求 payload（取 model 决定摘要模型）。
        session_key: 仅用于日志定位。
        summary_account: 调用方**已经持有**的上游账号（有就复用，避免多借一次号）。

    Returns:
        `【历史摘要】\\n<摘要>\\n（原始 N 字符 / M 条消息）`；未启用 / 未超阈值 /
        历史条数太少 / 没有可用账号 / 摘要失败时返回 `None`（调用方回退到硬截断）。
    """
    if not blob or not _summary_enabled(settings_obj):
        return None
    message_count = count_history_messages(blob)
    if message_count < _MIN_SUMMARY_HISTORY_MESSAGES:
        # 切掉当前任务后历史太短（单轮问答就是这种情况）：摘要没有意义。
        log.info(
            "[ContextSummary] skipped reason=history_too_short messages=%d chars=%d session_key=%s",
            message_count,
            len(blob),
            session_key,
        )
        return None
    threshold = _summary_threshold(settings_obj)
    if len(blob) <= threshold:
        return None

    state = getattr(app, "state", None) if app is not None else None
    client = getattr(state, "qwen_client", None) if state is not None else None
    if client is None:
        # 精简 app / 单测替身没有客户端：直接走截断，不制造新的失败点。
        log.info("[ContextSummary] skipped reason=no_client session_key=%s", session_key)
        return None

    summarizer = get_context_summarizer(settings_obj)
    account = summary_account
    pool = getattr(state, "account_pool", None) if state is not None else None
    borrowed = False
    if account is None and pool is not None:
        try:
            account = await pool.acquire_wait(timeout=_SUMMARY_ACCOUNT_TIMEOUT_SECONDS)
            borrowed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 借号失败等价于「没账号」
            log.info(
                "[ContextSummary] skipped reason=no_account error=%s session_key=%s",
                f"{type(exc).__name__}: {str(exc)[:120]}",
                session_key,
            )
            account = None
    if account is None:
        log.info("[ContextSummary] skipped reason=no_account session_key=%s", session_key)
        return None

    try:
        summary = await summarizer.summarize(
            text=blob,
            model=_summary_model_for(payload, settings_obj),
            account=account,
            client=client,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - 摘要永不 500
        log.warning(
            "[ContextSummary] failed reason=%s → fallback truncate session_key=%s",
            f"{type(exc).__name__}: {str(exc)[:160]}",
            session_key,
        )
        summary = None
    finally:
        if borrowed and pool is not None:
            try:
                pool.release(account)
            except Exception:  # pragma: no cover - 归还失败不影响主流程
                log.debug("[ContextSummary] release borrowed account failed", exc_info=True)

    if not summary:
        return None
    return (
        f"{_SUMMARY_BLOB_HEADER}\n{summary}\n"
        f"{_SUMMARY_BLOB_NOTE.format(original=len(blob), messages=count_history_messages(blob))}"
    )


def _latest_user_text(messages: list[Any] | None) -> str:
    """取最后一条 user 消息的纯文本（截断预算要给它让路，永不截断）。"""
    for message in reversed(list(messages or [])):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _text_from_content(message.get("content", ""))
        if text:
            return text
    return ""


def _latest_user_len(payload: dict[str, Any]) -> int:
    """payload 里最后一条 user 消息的字符数（预算要给它让路，附件不能把它挤掉）。"""
    return len(_latest_user_text(payload.get("messages")))


def _ensure_latest_user_intact(
    messages: list[Any],
    latest_text: str,
    *,
    session_key: str,
) -> list[Any]:
    """不变式：最新的用户消息**永远完整**保留在 payload 里。

    截断只允许动「历史 blob」和「附件正文」，最新用户消息是真正的问题本身，丢了
    就等于答非所问。这里用「检查 + 兜底补回」而不是 `assert`：真出现回归时宁可
    多带一条消息，也不能让整条请求 500（500 正是本次要消灭的故障形态）。
    """
    if not latest_text:
        return messages
    for message in messages:
        if not isinstance(message, dict):
            continue
        if latest_text in _text_from_content(message.get("content", "")):
            return messages
    log.error(
        "[ContextAttachment] latest user message lost after inline truncation; re-appending session_key=%s chars=%s",
        session_key,
        len(latest_text),
    )
    return [*messages, {"role": "user", "content": latest_text}]


def _split_history_blocks(text: str) -> list[str]:
    """把历史 blob 按 `## Message N [role]` **整条**切块。

    切出来的块首尾相接（`"".join(blocks) == text`），所以拼接后逐字节等于原文，
    不做任何重编号 / 改内容。没有分块头时整体当作一块返回。
    """
    starts = [match.start() for match in _HISTORY_BLOCK_HEADER_RE.finditer(text)]
    if not starts:
        return [text]
    bounds: list[int] = [0]
    bounds.extend(start for start in starts if start > 0)
    blocks: list[str] = []
    for index, start in enumerate(bounds):
        end = bounds[index + 1] if index + 1 < len(bounds) else len(text)
        blocks.append(text[start:end])
    return blocks


def _split_block_header(block: str) -> tuple[str, str]:
    """拆出「分块头」与「正文」；不像分块头时返回 `("", block)`。"""
    newline = block.find("\n")
    if newline < 0:
        return "", block
    header = block[: newline + 1]
    if not _HISTORY_BLOCK_HEADER_PREFIX_RE.match(header):
        return "", block
    return header, block[newline + 1 :]


def _block_role(block: str) -> str | None:
    """取分块头里的 role（`## Message 12 [user]` → `user`）；不是分块头返回 `None`。"""
    header, _body = _split_block_header(block)
    if not header:
        return None
    start = header.find("[")
    end = header.find("]", start + 1)
    if start < 0 or end < 0:
        return None
    return header[start + 1 : end].strip().lower()


def _trim_current_task_from_blob(blob: str) -> str:
    """按 `## Message N [role]` 结构切掉「最后一条 user 消息及其后的所有块」。

    `ContextOffloader.plan()` 的 `summary_text` 是**全量**消息序列化，包含本次提问。
    直接拿它做摘要有两个后果：

    1. 换个问题 → blob 变 → 缓存 key（sha256(text+model+target)）变 → 永远 miss；
    2. 把当前任务也拿去摘要，模型可能把「正在做的事」提前总结掉。

    所以这里按分块头反向找最后一个 `[... user]` 块，从它开始整块丢弃（同时丢掉它
    之后的 assistant / tool 续跑尾部），只留历史前缀。

    Args:
        blob: `plan.summary_text` 形态的历史序列化文本。

    Returns:
        历史前缀文本（已 strip）。切不出干净的历史（没有分块头 / 找不到 user 块）
        时原样返回，由 `_summarize_history_blob()` 的「条数太少不摘要」兜底。
    """
    text = str(blob or "")
    if not text.strip():
        return ""
    blocks = _split_history_blocks(text)
    if len(blocks) <= 1:
        # 没有分块头（或只有一块）：无法按消息切分，保持原样。
        return text
    for index in range(len(blocks) - 1, -1, -1):
        if _block_role(blocks[index]) != "user":
            continue
        return "".join(blocks[:index]).strip()
    return text


def _history_note(dropped: int, original: int, kept: int) -> str:
    """历史 blob 的截断提示行。"""
    if dropped > 0:
        return _HISTORY_TRUNCATION_NOTE.format(dropped=dropped, original=original, kept=kept)
    return _HISTORY_HEAD_TRUNCATION_NOTE.format(original=original, kept=kept)


def _truncate_block_head(block: str, *, budget: int, original: int) -> str:
    """单块超预算：保留**开头**（分块头完整）+ 结尾提示。

    分块头（`## Message N [role]`）一定整体保留：宁可略微超出预算，也绝不输出半
    个分块头 —— 半个头会让下游分块解析出错。
    """
    header, body = _split_block_header(block)
    marker = _BLOCK_TRUNCATION_NOTE.format(original=original)
    available = int(budget) - len(header) - len(marker) - 1
    if available <= 0:
        return f"{header}{marker}"
    return f"{header}{body[:available]}\n{marker}"


def _truncate_history_blob(text: str, budget: int | None) -> tuple[str, int, int, int]:
    """把历史序列化文本压到字符预算内（**最新内容在最后面，所以截最早的部分**）。

    策略：按 `## Message N [role]` 整条切块 → 从**头部**逐条丢弃，直到
    「保留正文 + 提示行」≤ 预算；只丢整条，不做字符位置切片。最新那几条消息
    因此在结果里完整保留。

    Args:
        text: 完整历史序列化文本（`plan.summary_text`）。
        budget: 字符预算上限；`None` 表示不限制（配置 0 / 负数时由
            `_inline_char_budget` 归一成 `None`）。`0` 是「一点都放不下」的硬上限，
            此时只保留提示行。

    Returns:
        `(结果文本, 原始字符数, 保留字符数, 丢弃的消息条数)`。未截断时结果与入参
        逐字节相同，且不加任何提示行。
    """
    original = len(text or "")
    if not text or budget is None or original <= int(budget):
        return text, original, original, 0

    blocks = _split_history_blocks(text)
    dropped = 0
    while True:
        kept_text = "".join(blocks)
        note = _history_note(dropped, original, len(kept_text))
        if len(kept_text) + len(note) + 1 <= int(budget) or len(blocks) <= 1:
            break
        blocks.pop(0)
        dropped += 1

    kept_text = "".join(blocks)
    note = _history_note(dropped, original, len(kept_text))
    if len(kept_text) + len(note) + 1 > int(budget):
        # 只剩一条（或首块）仍然超预算 → 退化成「保留开头」的头部截断。
        # 内层预算要先扣掉提示行本身（-4 是提示行里数字位数变化的安全余量）。
        inner_budget = max(0, int(budget) - len(note) - 1 - 4)
        kept_text = _truncate_block_head(kept_text, budget=inner_budget, original=original)
        note = _history_note(dropped, original, len(kept_text))
    return f"{note}\n{kept_text}", original, len(kept_text), dropped


def _truncate_inline_text_blocks(
    blocks: list[str],
    budget: int | None,
) -> tuple[list[str], int]:
    """把多个注入块的总字符压到预算内（**保留每块开头**，文件开头通常更关键）。

    预算按「剩余块数」均分，保证多个附件都还能被看到一部分，而不是前面的大文件
    把预算吃光、后面的附件彻底消失。

    Args:
        blocks: 注入块列表（`[附件: name]\\n正文`）。
        budget: 字符预算（含块间 `\\n\\n` 分隔符）；`None` 表示不限制。

    Returns:
        `(块列表, 被截断的块数)`。未超预算时原样返回（逐字节不变、无提示）。
    """
    items = list(blocks or [])
    if not items or budget is None:
        return items, 0
    separators = 2 * max(0, len(items) - 1)
    if sum(len(block) for block in items) + separators <= int(budget):
        return items, 0

    remaining = max(0, int(budget) - separators)
    kept: list[str] = []
    truncated = 0
    total = len(items)
    for index, block in enumerate(items):
        share = max(1, remaining // max(1, total - index))
        if len(block) <= share:
            kept.append(block)
            remaining = max(0, remaining - len(block))
            continue
        marker = _FILE_TRUNCATION_NOTE.format(original=len(block))
        if share <= len(marker):
            # 预算连标记都放不下：只留标记（让模型知道这里被截断过）。
            kept.append(marker)
            remaining = 0
            truncated += 1
            continue
        keep_chars = max(1, share - len(marker))
        kept.append(f"{block[:keep_chars]}{marker}")
        remaining = max(0, remaining - keep_chars - len(marker))
        truncated += 1
    return kept, truncated


def _inject_inline_text_blocks(
    payload: dict[str, Any],
    blocks: list[str],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    """把注入块拼进最后一条 user 消息（保持角色交替，不新增 role）。

    为什么要拼进「当前用户消息」而不是追加新消息：prompt 构建会把**最后一条
    user 消息**渲染成 `Human (CURRENT TASK - TOP PRIORITY)`，追加一条用户消息会
    把附件正文顶成「当前任务」，真实问题反而被降级。

    Args:
        payload: 原始 payload。
        blocks: 注入块列表。
        budget: 注入块总字符预算；`None` 表示不限制。超预算时每块保留开头
            并在结尾标注「已截断，原文件 N 字符」。
    """
    if not blocks:
        return payload
    blocks, truncated_blocks = _truncate_inline_text_blocks(blocks, budget)
    if truncated_blocks:
        log.info(
            "[ContextAttachment] inline text truncated blocks=%s kept_chars=%s budget=%s",
            truncated_blocks,
            sum(len(block) for block in blocks),
            budget,
        )
    text = "\n\n".join(blocks)
    rewritten = dict(payload)
    messages: list[Any] = []
    for message in payload.get("messages", []) or []:
        messages.append(dict(message) if isinstance(message, dict) else message)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [*content, {"type": "text", "text": text}]
        elif isinstance(content, str) and content.strip():
            message["content"] = f"{content}\n\n{text}"
        else:
            message["content"] = text
        rewritten["messages"] = messages
        return rewritten
    messages.append({"role": "user", "content": text})
    rewritten["messages"] = messages
    return rewritten


def build_request_session_key(surface: str, request_id: str) -> str:
    return f"{surface}:{request_id}"


def derive_session_key(surface: str, auth_token: str, payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    explicit = payload.get("session_key") or payload.get("conversation_id") or metadata.get("conversation_id")
    messages = payload.get("messages", []) or []
    first_user = next((m for m in messages if m.get("role") == "user"), {})
    first_text = _text_from_content(first_user.get("content", ""))
    system_text = _text_from_content(payload.get("system", ""))
    developer_parts = [_text_from_content(payload.get("developer", ""))]
    developer_parts.extend(
        _text_from_content(message.get("content", ""))
        for message in messages
        if message.get("role") in {"system", "developer"}
    )
    developer_parts.extend(
        user_system_text
        for message in messages
        if message.get("role") == "user"
        for user_system_text in [user_role_system_text(_text_from_content(message.get("content", "")))]
        if user_system_text
    )
    developer_text = "\n".join(part for part in developer_parts if part)
    instructions_text = _text_from_content(payload.get("instructions", ""))
    persona_basis = f"{system_text[:400]}::{developer_text[:400]}::{instructions_text[:400]}"
    explicit_text = str(explicit or "")
    user_basis = "" if explicit else f"::{first_text[:400]}"
    basis = f"{surface}::{auth_token}::{payload.get('model', '')}::{explicit_text}::{persona_basis}{user_basis}"
    return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _context_messages_for_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    context_messages: list[dict[str, Any]] = []
    for field_name, role in (("system", "system"), ("developer", "developer"), ("instructions", "system")):
        text = _text_from_content(payload.get(field_name, ""))
        if text.strip():
            context_messages.append({"role": role, "content": text})
    context_messages.extend(payload.get("messages", []) or [])
    return context_messages


def _generated_context_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext}"


def _split_current_task(messages: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把消息切成「历史前缀」与「当前任务」两段（**全模块唯一口径**）。

    切分规则：**最后一条 user 消息及其之后的消息都属于「当前任务」**。

    - 最后一条 user 消息 = 本次提问本身；
    - 它之后的 assistant / tool 消息 = 工具续跑尾部（多轮工具调用），同样属于当前
      任务，不能进摘要 —— 否则摘要会把「当前正在做的事」提前总结掉，多轮工具会断链。

    摘要路径与 `_payload_with_history_summary()` 必须共用这一个函数，避免两处规则
    漂移（漂移的直接后果：摘要 blob 里混进当前提问 → 缓存 key 每次都变 → 永远 miss）。

    Args:
        messages: 消息列表（元素可以是任意对象，非 `dict` 一律忽略）。

    Returns:
        `(history_messages, current_task_messages)`。没有 user 消息时返回
        `(全部消息, [])`，即「整段都算历史」，与旧行为一致。
    """
    items = [message for message in (messages or []) if isinstance(message, dict)]
    last_user_index = -1
    for index, message in enumerate(items):
        if message.get("role") == "user":
            last_user_index = index
    if last_user_index < 0:
        return items, []
    return items[:last_user_index], items[last_user_index:]


def _serialize_history_messages(messages: list[Any]) -> str:
    """把消息序列化成与 `ContextOffloader.plan()` 同构的历史 blob。

    只在「不走上下文卸载」的内联路径上用（那里 `plan.summary_text` 是空的），
    格式必须和卸载路径一致，摘要器才能按 `## Message N [role]` 数消息条数、
    按整条对齐裁剪。
    """
    parts: list[str] = []
    for idx, message in enumerate(messages or [], 1):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        text = _text_from_content(message.get("content", ""))
        if not text.strip():
            continue
        parts.append(f"## Message {idx} [{role}]\n{text.strip()}\n")
    return "\n".join(parts).strip()


def _payload_with_history_summary(payload: dict[str, Any], summary_blob: str) -> dict[str, Any]:
    """用摘要 blob 替换 payload 里的历史，**保留最后一条 user 消息及其之后的消息**。

    最后一条 user 消息是「当前任务」，工具续跑的尾部（tool_result 等）也必须跟在
    它后面，否则多轮工具调用会断链。摘要放在它们**前面**，避免被 prompt 构建当成
    `CURRENT TASK - TOP PRIORITY`。

    切分走 `_split_current_task()`（与摘要路径同一口径）。
    """
    messages = [message for message in (payload.get("messages") or []) if isinstance(message, dict)]
    _history_messages, current_task_messages = _split_current_task(messages)
    rewritten = dict(payload)
    rewritten["messages"] = [{"role": "user", "content": summary_blob}, *current_task_messages]
    return rewritten


async def _fallback_context_attachment_result(
    *,
    payload: dict[str, Any],
    session_key: str,
    plan,
    use_generated_context_files: bool,
    manual_attachments: list[Any],
    fallback_message: str,
    inline_text_blocks: list[str] | None = None,
    inline_text_tokens: int = 0,
    upstream_files: list[dict[str, Any]] | None = None,
    settings_obj: Any = None,
    app: Any = None,
    summary_account: Any = None,
) -> dict[str, Any]:
    """上传 / parse 不可用时的兜底：把历史正文当成一条 user 消息内联注入。

    `plan.summary_text` 就是完整历史正文（不是摘要），注入它意味着「附件上传失败
    也不丢上下文」。但上游对超大内联 prompt **直接拒绝**（x5sec punish 页 → 500），
    所以这里按 `ATTACHMENT_INLINE_TEXT_MAX_CHARS` 做**体积裁剪**：

    0. **两段式总结优先**：历史超 `CONTEXT_SUMMARY_THRESHOLD_CHARS` 时先把它压成
       要点摘要（摘要 blob 远小于原文，天然落在预算内）；摘要失败（超时 / 额度 /
       无账号 / 无客户端）则原样往下走截断 —— 摘要永远不能变成 500；
    1. 历史 blob：按 `## Message N` 整条从**最早**开始丢弃（最新内容在最后面）；
    2. 附件文本：每块保留开头 + 结尾标注；
    3. 优先级：最新的用户消息（永不截断） > 用户附件文本 > 历史 blob。

    Args:
        app: FastAPI app（取 `qwen_client` / 需要时借号做摘要）；`None` 表示不做摘要。
        summary_account: 调用方已持有的上游账号（避免为摘要多借一次号）。
    """
    budget = _inline_char_budget(settings_obj)
    fallback_payload = dict(payload)
    summary_parts: list[str] = []

    raw_history = str(plan.summary_text or "") if use_generated_context_files else ""
    original_history_chars = len(raw_history)
    history_text = raw_history
    summarized = False
    if history_text:
        # 摘要只吃「历史前缀」：`plan.summary_text` 含本次提问（及其后的工具续跑
        # 尾部），原样送进去会让缓存 key 随问题变化 → 永远 miss，而且把当前任务
        # 也摘要掉。这里按分块头切掉「最后一条 user 及其后」，切完太短就不摘要。
        summary_source = _trim_current_task_from_blob(history_text)
        summary_blob = await _summarize_history_blob(
            blob=summary_source,
            settings_obj=settings_obj,
            app=app,
            payload=payload,
            session_key=session_key,
            summary_account=summary_account,
        )
        if summary_blob:
            history_text = summary_blob
            summarized = True
    # 截断前的长度（摘要生效时是摘要 blob 的长度，不是原始历史）。
    blob_before_truncate = len(history_text)
    kept_history_chars = blob_before_truncate
    dropped_messages = 0

    latest_user_text = _latest_user_text(
        plan.inline_messages if use_generated_context_files and getattr(plan, "inline_messages", None) else payload.get("messages")
    )
    # 附件文本优先于历史 blob：先按「预算 - 最新用户消息」裁附件，再把剩下的给历史。
    attachment_budget = _remaining_budget(budget, len(latest_user_text))
    text_blocks, truncated_blocks = _truncate_inline_text_blocks(list(inline_text_blocks or []), attachment_budget)
    blocks_chars = sum(len(block) for block in text_blocks) + 2 * max(0, len(text_blocks) - 1)
    history_budget = _remaining_budget(attachment_budget, blocks_chars)
    truncated_history = False
    if history_text:
        history_text, _blob_chars, kept_history_chars, dropped_messages = _truncate_history_blob(
            history_text, history_budget
        )
        truncated_history = kept_history_chars < blob_before_truncate

    if history_text:
        summary_parts.append(history_text)
    if manual_attachments:
        names = ", ".join(att.filename for att in manual_attachments[:4])
        summary_parts.append(
            f"User attachments were provided but attachment upload failed. Attachment names: {names}"
        )
    fallback_messages: list[dict[str, Any]] = []
    if summary_parts:
        fallback_messages.append({
            "role": "user",
            "content": f"{summary_parts[0]}\n\n{SYSTEM_CONTEXT_PROMPT_NOTE}",
        })
    elif fallback_message:
        fallback_messages.append({"role": "user", "content": fallback_message})
    if use_generated_context_files and getattr(plan, "inline_messages", None):
        fallback_messages.extend(plan.inline_messages)
    else:
        fallback_messages.extend(payload.get("messages", []) or [])
    fallback_payload["messages"] = fallback_messages
    fallback_payload = _inject_inline_text_blocks(fallback_payload, text_blocks, budget=attachment_budget)
    # 不变式：最新用户消息在截断后必须完整存在（丢了就补回来，绝不让请求 500）。
    fallback_payload["messages"] = _ensure_latest_user_intact(
        list(fallback_payload.get("messages") or []),
        latest_user_text,
        session_key=session_key,
    )
    truncated = bool(truncated_history or truncated_blocks)
    log.info(
        "[ContextAttachment] inline fallback chars=%s messages=%s session_key=%s text_attachment_tokens=%s truncated=%s",
        kept_history_chars,
        len(fallback_payload.get("messages") or []),
        session_key,
        inline_text_tokens,
        str(truncated).lower(),
    )
    if truncated:
        log.info(
            "[ContextAttachment] inline fallback truncated original=%d kept=%d dropped_messages=%d",
            original_history_chars,
            kept_history_chars,
            dropped_messages,
        )
    if summarized:
        log.info(
            "[ContextAttachment] inline fallback summarized original=%d blob=%d session_key=%s",
            original_history_chars,
            kept_history_chars,
            session_key,
        )
    return {
        "payload": fallback_payload,
        "session_key": session_key,
        "context_mode": "inline",
        "upstream_files": list(upstream_files if upstream_files is not None else (payload.get("upstream_files", []) or [])),
        "bound_account": None,
        "bound_account_email": None,
        "generated_local_files": [],
        "context_attachment_tokens": inline_text_tokens,
        "attachment_fallback": True,
    }


def _context_tools_for_payload(payload: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    tools = payload.get("tools", []) or []
    if surface not in {"openai", "responses"}:
        return tools
    try:
        normalized = normalize_chat_request(payload, excluded_tool_names=_excluded_command_like_tool_names(payload))
        prompt_payload = to_prompt_payload(
            normalized,
            model=str(payload.get("model") or ""),
            stream=bool(payload.get("stream", False)),
        )
        return prompt_payload.get("tools", []) or []
    except Exception:
        return tools


async def prepare_context_attachments(*, app, payload: dict[str, Any], surface: str, auth_token: str, client_profile: str, session_key: str, existing_attachments=None) -> dict[str, Any]:
    context_offloader = app.state.context_offloader
    account_pool = app.state.account_pool
    file_store = app.state.file_store
    affinity = app.state.session_affinity
    cache = app.state.upstream_file_cache
    uploader = app.state.upstream_file_uploader
    # 浏览器上传代理（可选）：内联装不下时，用远程 CDP 浏览器过阿里云拼图验证码换
    # getstsToken 的 STS，再复用 uploader.put_bytes() 把字节 PUT 到 OSS。
    browser_proxy = getattr(app.state, "browser_upload_proxy", None)
    browser_upload_enabled = _browser_upload_enabled(context_offloader.settings, browser_proxy)

    tools = _context_tools_for_payload(payload, surface)
    messages = payload.get("messages", []) or []
    context_messages = _context_messages_for_payload(payload)
    manual_attachments = list(existing_attachments or [])
    plan = context_offloader.plan(context_messages, tools=tools, client_profile=client_profile)
    # 内联正文总字符预算（None = 不限制）。超预算的截断由下面各条路径统一执行。
    inline_char_budget = _inline_char_budget(context_offloader.settings)
    use_generated_context_files = bool(plan.generated_files)
    if not use_generated_context_files and not manual_attachments:
        # 正常内联路径：历史原样留在 payload 里。默认配置下这里的历史 ≤ 8k（卸载
        # 阈值），无需处理；但把 `CONTEXT_INLINE_MAX_CHARS` 调大之后，这里会直接
        # 内联十几 k 的原文 —— 正好踩上游 punish 阈值，所以同样过一遍摘要。
        # 摘要只吃「历史前缀」（`_split_current_task` 与 `_payload_with_history_summary`
        # 同一口径）：本次提问及其后的工具续跑尾部不进摘要，缓存 key 才对同一段历史
        # 稳定。
        history_messages, _current_task_messages = _split_current_task(context_messages)
        history_blob = _serialize_history_messages(history_messages)
        summary_blob = await _summarize_history_blob(
            blob=history_blob,
            settings_obj=context_offloader.settings,
            app=app,
            payload=payload,
            session_key=session_key,
        )
        if summary_blob:
            log.info(
                "[ContextSummary] injected inline payload session_key=%s original=%d blob=%d",
                session_key,
                len(history_blob),
                len(summary_blob),
            )
            return {
                "payload": _payload_with_history_summary(payload, summary_blob),
                "session_key": session_key,
                "context_mode": "summary",
                "upstream_files": list(payload.get("upstream_files", []) or []),
                "bound_account": None,
                "bound_account_email": None,
                "generated_local_files": [],
                "context_attachment_tokens": 0,
                "attachment_fallback": False,
            }
        return {
            "payload": payload,
            "session_key": session_key,
            "context_mode": "inline",
            "upstream_files": list(payload.get("upstream_files", []) or []),
            "bound_account": None,
            "bound_account_email": None,
            "generated_local_files": [],
            "context_attachment_tokens": 0,
            "attachment_fallback": False,
        }

    # 图片 / 音频 / 视频不需要上游上传与解析，直接内联；纯文本类附件在网关本地
    # 解码后注入 prompt（也不发上游请求）；只有文档（pdf/docx/...）仍需 PUT + parse。
    inline_media_attachments: list[Any] = []
    upload_candidates: list[Any] = []
    for attachment in manual_attachments:
        if _is_inline_media_attachment(attachment):
            inline_media_attachments.append(attachment)
        else:
            upload_candidates.append(attachment)

    text_candidates: list[Any] = []
    upload_attachments: list[Any] = []
    for attachment in upload_candidates:
        if _is_inline_text_attachment(attachment):
            text_candidates.append(attachment)
        else:
            upload_attachments.append(attachment)

    # 纯文本类：本地读字节 → 解码 → 注入块。**零上游请求**（不调 getstsToken /
    # 不 PUT / 不 parse）。二进制 sniff 命中的会退回 upload_attachments。
    inline_text_blocks: list[str] = []
    inline_text_tokens = 0
    if text_candidates:
        inline_text_blocks, inline_text_tokens, text_leftovers = await _collect_inline_text_blocks(
            text_candidates,
            file_store=file_store,
            settings_obj=context_offloader.settings,
            session_key=session_key,
        )
        upload_attachments.extend(text_leftovers)

    # 内联媒体不依赖上游账号：先处理，失败（读取失败 / 超限 / 开关关闭）才回退上传。
    upstream_files = list(payload.get("upstream_files", []) or [])
    fallback_uploads: list[Any] = []
    inline_too_large_errors: list[InlineMediaTooLargeError] = []
    inline_ok = 0
    for attachment in inline_media_attachments:
        inline_ref, too_large = await _inline_ref_for_attachment(
            attachment,
            settings_obj=context_offloader.settings,
        )
        if inline_ref is None:
            log.warning(
                "[ContextAttachment] inline media unavailable session_key=%s surface=%s filename=%s content_type=%s too_large=%s",
                session_key,
                surface,
                getattr(attachment, "filename", ""),
                getattr(attachment, "content_type", ""),
                too_large is not None,
            )
            if too_large is not None:
                inline_too_large_errors.append(too_large)
            fallback_uploads.append(attachment)
            continue
        upstream_files.append(inline_ref)
        inline_ok += 1
        # 已内联进 payload，本地副本不再需要
        await file_store.delete_path(attachment.local_path)

    # 纯内联（媒体 + 文本）：不挑号、不绑号、不上传、不解析。
    if not upload_attachments and not fallback_uploads and not use_generated_context_files and not payload.get("upstream_files"):
        log.info(
            "[ContextAttachment] inline only session_key=%s surface=%s inline_media=%s inline_text=%s total=%s",
            session_key,
            surface,
            inline_ok,
            len(inline_text_blocks),
            len(manual_attachments),
        )
        return {
            "payload": _inject_inline_text_blocks(
                payload,
                inline_text_blocks,
                budget=_remaining_budget(inline_char_budget, _latest_user_len(payload)),
            ),
            "session_key": session_key,
            "context_mode": "inline",
            "upstream_files": upstream_files,
            "bound_account": None,
            "bound_account_email": None,
            "generated_local_files": [],
            "context_attachment_tokens": inline_text_tokens,
            "attachment_fallback": bool(inline_media_attachments) and inline_ok == 0,
        }

    # 文档 parse 熔断：上游 `/files/parse` 已被阿里云 WAF 拦死（换出口也过不去），
    # 冷却期内**直接跳过「上传 + parse」**，省掉一次注定失败的 10~20s。
    if upload_attachments or use_generated_context_files:
        breaker = get_document_parse_breaker(context_offloader.settings)
        if breaker.blocked():
            pending_files = [str(getattr(att, "filename", "") or "") for att in upload_attachments]
            pending_files.extend(
                f"{SYSTEM_CONTEXT_FILE_PREFIX}_{getattr(item, 'ext', 'txt')}"
                for item in getattr(plan, "generated_files", []) or []
            )
            log.info(
                "[ContextAttachment] document parse blocked (cooldown) → inline text fallback file=%s "
                "session_key=%s surface=%s remaining=%.0fs",
                ",".join(name for name in pending_files if name) or "-",
                session_key,
                surface,
                breaker.remaining_seconds(),
            )
            return await _fallback_context_attachment_result(
                payload=payload,
                session_key=session_key,
                plan=plan,
                use_generated_context_files=use_generated_context_files,
                manual_attachments=manual_attachments,
                fallback_message=(
                    "Document parsing is temporarily unavailable upstream. "
                    "Continue with the available inline context only."
                ),
                inline_text_blocks=inline_text_blocks,
                inline_text_tokens=inline_text_tokens,
                upstream_files=upstream_files,
                settings_obj=context_offloader.settings,
                app=app,
            )

    requires_sticky_account = bool(manual_attachments or payload.get("upstream_files"))
    record = await affinity.get(session_key) if requires_sticky_account else None
    preferred_email = record.account_email if record else None
    upload_excludes = _upload_account_excludes(account_pool)
    if preferred_email:
        preferred_kwargs: dict[str, Any] = {"timeout": 60}
        if upload_excludes:
            preferred_kwargs["exclude"] = set(upload_excludes)
        acc = await account_pool.acquire_wait_preferred(preferred_email, **preferred_kwargs)
    else:
        acquire_kwargs: dict[str, Any] = {"timeout": 60}
        if upload_excludes:
            acquire_kwargs["exclude"] = set(upload_excludes)
        acc = await account_pool.acquire_wait(**acquire_kwargs)
    if not acc:
        log.warning(
            "[ContextAttachment] no upstream account available; falling back inline session_key=%s surface=%s manual_attachments=%s generated_files=%s",
            session_key,
            surface,
            [getattr(att, "filename", "") for att in manual_attachments],
            [getattr(item, "ext", "") for item in getattr(plan, "generated_files", [])],
        )
        if inline_too_large_errors:
            # 内联放不下 + 没有账号可上传 → 明确 4xx，绝不静默丢附件。
            raise inline_too_large_errors[0]
        return await _fallback_context_attachment_result(
            payload=payload,
            session_key=session_key,
            plan=plan,
            use_generated_context_files=use_generated_context_files,
            manual_attachments=manual_attachments,
            fallback_message=(
                "No upstream account was available for attachment upload. "
                "Continue with the available inline context only."
            ),
            inline_text_blocks=inline_text_blocks,
            inline_text_tokens=inline_text_tokens,
            upstream_files=upstream_files,
            settings_obj=context_offloader.settings,
        )
    await affinity.bind_account(session_key, surface, acc.email, context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS)

    local_file_records: list[dict[str, Any]] = []
    context_attachment_tokens = 0
    attachment_ttl = context_offloader.settings.CONTEXT_ATTACHMENT_TTL_SECONDS

    async def _switch_account_on_retry(current_acc, upload_exc: Exception):
        if not _is_retryable_attachment_upload_error(upload_exc):
            raise upload_exc

        account_pool.release(current_acc)
        exclude = set(upload_excludes)
        current_email = getattr(current_acc, "email", None)
        if current_email:
            exclude.add(current_email)
        next_acc = await account_pool.acquire_wait(timeout=10, exclude=exclude)
        if not next_acc:
            raise upload_exc
        await affinity.bind_account(session_key, surface, next_acc.email, attachment_ttl)
        log.warning(
            "[ContextAttachment] retrying upload with alternate account session_key=%s surface=%s previous_account=%s new_account=%s error=%s",
            session_key,
            surface,
            current_email,
            getattr(next_acc, "email", None),
            upload_exc,
        )
        return next_acc

    async def _switch_account_on_upload_limit(current_acc, rate_exc: FileUploadRateLimitedError):
        account_pool.mark_file_upload_limited(current_acc, rate_exc.retry_after_seconds, str(rate_exc))
        await account_pool.save()
        current_email = getattr(current_acc, "email", None)
        if current_email:
            upload_excludes.add(current_email)
        account_pool.release(current_acc)
        next_acc = await account_pool.acquire_wait(timeout=10, exclude=set(upload_excludes))
        if not next_acc:
            raise rate_exc
        await affinity.bind_account(session_key, surface, next_acc.email, attachment_ttl)
        log.warning(
            "[ContextAttachment] upload quota exceeded; switching account session_key=%s surface=%s previous_account=%s new_account=%s retry_after=%ss",
            session_key,
            surface,
            current_email,
            getattr(next_acc, "email", None),
            int(getattr(rate_exc, "retry_after_seconds", 0) or 0),
        )
        return next_acc

    async def _try_browser_upload(current_acc, local_meta, *, allow: bool) -> dict[str, Any] | None:
        """浏览器上传：过验证码拿 STS → 复用 `put_bytes()` PUT 到 OSS。

        Args:
            current_acc: 当前绑定的上游账号（浏览器要用它的 token / cookies 种登录态）。
            local_meta: 本地文件元数据（`path` / `filename` / `content_type` / ...）。
            allow: 是否允许走浏览器上传。**只有调用方真实提供的附件才能为 True**；
                内部生成的 context 文件必须为 False（原因见
                :func:`_upload_with_failover` 的说明）。

        Returns:
            与 `uploader.upload_local_file()` 同结构的结果字典；
            未启用 / 超时 / 任何浏览器侧异常都返回 None（调用方继续走原 HTTP 上传链路）。
        """
        if not allow or not browser_upload_enabled:
            return None
        # 延迟导入：playwright 是可选依赖，且多数单测不打桩浏览器代理。
        from backend.services.browser_upload_proxy import (
            BrowserUploadError,
            BrowserUploadRateLimited,
            BrowserUploadUnavailable,
        )

        filename = str(local_meta.get("filename") or "")
        content_type = str(local_meta.get("content_type") or "")
        hard_timeout = _browser_hard_timeout_seconds(context_offloader.settings)
        uploaded_size = 0

        async def _browser_sts_and_put() -> dict[str, Any] | None:
            """真正会「慢」的那一段：读字节 → 浏览器换 STS → OSS PUT。"""
            nonlocal uploaded_size
            raw = await asyncio.to_thread(Path(local_meta["path"]).read_bytes)
            if not raw:
                return None
            uploaded_size = len(raw)
            sts = await browser_proxy.fetch_sts(current_acc, filename, content_type, len(raw))
            if not sts:
                return None
            return await uploader.put_bytes(current_acc, sts, raw, filename, content_type)

        try:
            # 硬超时护栏：`wait_for` 会取消整段浏览器流程（含 `_session_lock` 的持有），
            # 保证浏览器卡死时请求仍能按降级路径返回，而不是等满
            # `UPLOAD_BROWSER_TIMEOUT_SECONDS`（120s）× 若干轮。
            remote = await asyncio.wait_for(_browser_sts_and_put(), timeout=hard_timeout)
        except asyncio.TimeoutError:
            # 超时等价于「浏览器链路不可用」：按 BrowserUploadUnavailable 语义降级，
            # 同时立刻打开该账号的熔断，避免下一个请求继续为同一条坏链路付硬超时。
            log.warning(
                "[ContextAttachment] browser upload hard timeout session_key=%s surface=%s account=%s filename=%s timeout=%ss",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                hard_timeout,
            )
            note_timeout = getattr(browser_proxy, "note_hard_timeout", None)
            if callable(note_timeout):
                try:
                    note_timeout(str(getattr(current_acc, "email", "") or ""))
                except Exception:  # pragma: no cover - 熔断记账失败不影响降级
                    log.debug("[ContextAttachment] note_hard_timeout failed", exc_info=True)
            return None
        except BrowserUploadRateLimited as exc:
            # 上游每日上传配额用尽：翻译成既有异常，复用「标记超限号 → 换号」逻辑。
            raise FileUploadRateLimitedError(
                retry_after_seconds=exc.retry_after_seconds,
                message=str(exc),
            ) from exc
        except BrowserUploadUnavailable as exc:
            log.warning(
                "[ContextAttachment] browser upload unavailable session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                str(exc)[:200],
            )
            return None
        except BrowserUploadError as exc:
            log.warning(
                "[ContextAttachment] browser upload failed session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                str(exc)[:200],
            )
            return None
        except Exception as exc:
            log.warning(
                "[ContextAttachment] browser upload error session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                filename,
                str(exc)[:200],
            )
            return None
        if remote is None:
            return None
        log.info(
            "[ContextAttachment] browser upload ok session_key=%s surface=%s account=%s filename=%s size=%s",
            session_key,
            surface,
            getattr(current_acc, "email", None),
            filename,
            uploaded_size,
        )
        return remote

    async def _try_exit_pool_upload(current_acc, local_meta) -> dict[str, Any] | None:
        """出口池取 STS + OSS 直传：最快（~6s）且不烧风控分，**降级链第一档**。

        `uploader.upload_local_file()` 内部会在池里轮询出口，拿到 STS 后 OSS PUT
        直连。池里所有出口都被 WAF 拦 / 连不上时抛 `UpstreamWafChallengeError`，
        这里吞掉返回 None，把机会交给浏览器代理。

        Returns:
            与 `uploader.upload_local_file()` 同结构的结果字典，或 None。
        """
        if not _upload_exit_pool_enabled(uploader):
            return None
        try:
            return await uploader.upload_local_file(current_acc, local_meta)
        except UpstreamWafChallengeError as exc:
            log.warning(
                "[ContextAttachment] exit pool upload unavailable session_key=%s surface=%s account=%s filename=%s error=%s",
                session_key,
                surface,
                getattr(current_acc, "email", None),
                str(local_meta.get("filename") or ""),
                str(exc)[:200],
            )
            return None

    async def _upload_direct_http(current_acc, local_meta) -> dict[str, Any]:
        """直连 HTTP（现状）：不挂代理取 STS → OSS PUT。

        没有出口池的老部署、或出口池已全部失败时的最后一条 HTTP 路径。
        上传器若没实现 `upload_local_file_direct`（老实现 / 单测替身），
        退回 `upload_local_file`。
        """
        direct = getattr(uploader, "upload_local_file_direct", None)
        if callable(direct):
            return await direct(current_acc, local_meta)
        return await uploader.upload_local_file(current_acc, local_meta)

    async def _upload_with_failover(current_acc, local_meta, ext, *, allow_browser: bool):
        """上传单文件并处理换号回退，返回 (remote, acc, uploaded)。

        降级顺序（快 / 便宜 → 慢 / 贵）：

        1. **出口池取 STS + 直传 OSS**（~6s，不烧风控分）：池启用时才试；
        2. **浏览器代理**（过验证码换 STS，45s 硬超时）：`allow_browser=True` 时；
        3. **直连 HTTP**（现状）：`upload_local_file_direct`，撞 WAF 就抛；
        4. 内联 / 400：由外层统一处理。

        另：
        - 命中缓存：直接返回 (remote, acc, False)
        - RateLimited（每日上传配额）：标记超限号→落盘→排除→换号循环，无号可用则抛出（外层回退内联）
        - 网络类错误：换号重试一次，仍失败则抛出

        **为什么 `allow_browser` 必须按来源区分（不要"顺手"统一打开）：**

        浏览器上传的代价是「起一个 Chromium + 注入登录态 + 解拼图」，单次几百毫秒到
        几十秒，还会消耗账号的风控分。它只该服务于**调用方真实提供的附件**——那种场景
        下没有替代路径，慢一点也比丢附件好。

        而 `plan.generated_files`（上下文卸载生成的 context 文件）是**网关自己造的**：
        它每个「带 tools / 大上下文」的请求都会产生，且**永远有替代路径**（出口池 /
        直连 OSS 上传，撞 WAF 就按既有逻辑降级为内联文本）。一旦让内部生成文件也走
        浏览器上传，就等于「每个普通请求都去开一次浏览器解一次拼图」：请求被拖死、
        风控分被打高，且收益为零（内联文本本来就是可接受的降级）。2026-11 的生产
        事故就是这么来的：一个 3s 的带 tools 请求被拖成 120s 超时。

        出口池**不受这条限制**：它只是换出口 IP 直连上游，几秒内结束、不碰风控分，
        所以生成文件同样走第一档。
        """
        network_retry_used = False
        while True:
            cache_entry = await cache.get(session_key, current_acc.email, local_meta["sha256"], ext)
            if cache_entry is not None:
                return cache_entry.remote_file_meta, current_acc, False
            try:
                # 1) 出口池取 STS + 直传 OSS（最快，且不烧风控分）
                pool_remote = await _try_exit_pool_upload(current_acc, local_meta)
                if pool_remote is not None:
                    return pool_remote, current_acc, True
                # 2) 浏览器代理（过验证码换 STS），失败/未启用再往下走
                browser_remote = await _try_browser_upload(current_acc, local_meta, allow=allow_browser)
                if browser_remote is not None:
                    return browser_remote, current_acc, True
                # 3) 直连 HTTP（现状）
                remote = await _upload_direct_http(current_acc, local_meta)
                return remote, current_acc, True
            except FileUploadRateLimitedError as rate_exc:
                current_acc = await _switch_account_on_upload_limit(current_acc, rate_exc)
            except Exception as upload_exc:
                if network_retry_used:
                    raise
                network_retry_used = True
                current_acc = await _switch_account_on_retry(current_acc, upload_exc)

    try:
        # 内联失败（读取失败 / 超限）的媒体在这里补走一次上传链路。
        for attachment in list(upload_attachments) + list(fallback_uploads):
            if getattr(attachment, "remote_ref", None):
                upstream_files.append(attachment.remote_ref)
                continue
            if not attachment.local_path:
                continue
            local_meta = {
                "id": attachment.file_id,
                "path": attachment.local_path,
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "sha256": attachment.sha256,
                "created_at": __import__("time").time(),
            }
            ext = Path(attachment.filename).suffix.lstrip(".").lower()
            # 调用方真实提供的附件（inline-* / remote-url / manual-upload）：允许浏览器上传。
            remote, acc, uploaded = await _upload_with_failover(
                acc, local_meta, ext, allow_browser=True
            )
            if uploaded:
                cache_expires = _resolve_attachment_cache_expires_at(
                    remote,
                    created_at=local_meta["created_at"],
                    session_ttl_seconds=attachment_ttl,
                )
                await cache.set(UpstreamFileCacheEntry(
                    session_key=session_key,
                    account_email=acc.email,
                    sha256=local_meta["sha256"],
                    ext=ext,
                    filename=attachment.filename,
                    remote_file_meta=remote,
                    created_at=local_meta["created_at"],
                    expires_at=cache_expires,
                ))
            upstream_files.append(remote["remote_ref"])
            await affinity.add_uploaded_file(session_key, remote)
            await file_store.delete_path(attachment.local_path)

        if use_generated_context_files:
            for generated in plan.generated_files:
                filename = _generated_context_filename(generated.ext)
                local_meta = await file_store.save_text(filename, generated.text, generated.content_type, purpose="context")
                # 内部生成的 context 文件**绝不走浏览器上传**（allow_browser=False）：
                # 每个带 tools / 大上下文的请求都会生成这类文件，让它走浏览器等于给
                # 普通请求挂上一次「起 Chromium + 解拼图」，会把请求拖死并烧风控分；
                # 而这条链路本来就有兜底（HTTP OSS 上传，撞 WAF 降级为内联文本）。
                remote, acc, uploaded = await _upload_with_failover(
                    acc, local_meta, generated.ext, allow_browser=False
                )
                if uploaded:
                    cache_expires = _resolve_attachment_cache_expires_at(
                        remote,
                        created_at=local_meta["created_at"],
                        session_ttl_seconds=attachment_ttl,
                    )
                    await cache.set(UpstreamFileCacheEntry(
                        session_key=session_key,
                        account_email=acc.email,
                        sha256=local_meta["sha256"],
                        ext=generated.ext,
                        filename=filename,
                        remote_file_meta=remote,
                        created_at=local_meta["created_at"],
                        expires_at=cache_expires,
                    ))
                upstream_files.append(remote["remote_ref"])
                context_attachment_tokens += count_tokens(generated.text)
                await affinity.add_uploaded_file(session_key, remote)
                await file_store.delete_path(local_meta["path"])
                local_file_records.append(local_meta)
    except Exception as exc:
        # 上游上传被 WAF 人机验证拦住是**预期路径**（getstsToken 现在必被拦），
        # 只打一行 warning、不打整栈 ERROR；其它异常保留 exception 便于排查。
        if isinstance(exc, UpstreamWafChallengeError):
            log.warning(
                "[ContextAttachment] upstream upload blocked by WAF session_key=%s surface=%s account=%s error=%s",
                session_key,
                surface,
                getattr(acc, "email", None),
                str(exc)[:200],
            )
        else:
            log.exception(
                "[ContextAttachment] upload failed session_key=%s surface=%s account=%s manual_attachments=%s generated_files=%s error=%s",
                session_key,
                surface,
                getattr(acc, "email", None),
                [getattr(att, "filename", "") for att in manual_attachments],
                [getattr(item, "ext", "") for item in getattr(plan, "generated_files", [])],
                exc,
            )
        if inline_too_large_errors:
            # 内联放不下 + 上传回退不可用 → 明确 4xx，绝不静默丢附件。
            account_pool.release(acc)
            raise inline_too_large_errors[0] from exc
        # 摘要请求复用**仍然持有**的账号（acc 还没归还）：这里必须等兜底结果算完
        # 才能 release，否则账号会被别的请求并发占用。
        try:
            return await _fallback_context_attachment_result(
                payload=payload,
                session_key=session_key,
                plan=plan,
                use_generated_context_files=use_generated_context_files,
                manual_attachments=manual_attachments,
                fallback_message="Attachment upload failed. Continue with the available inline context only.",
                inline_text_blocks=inline_text_blocks,
                inline_text_tokens=inline_text_tokens,
                upstream_files=upstream_files,
                settings_obj=context_offloader.settings,
                app=app,
                summary_account=acc,
            )
        finally:
            account_pool.release(acc)

    rewritten = dict(payload)
    rewritten["messages"] = plan.inline_messages if use_generated_context_files else messages
    # 纯文本类附件：正文拼进当前用户消息（已上传 / 已生成的文档不走这里）。
    rewritten = _inject_inline_text_blocks(
        rewritten,
        inline_text_blocks,
        budget=_remaining_budget(inline_char_budget, _latest_user_len(rewritten)),
    )
    return {
        "payload": rewritten,
        "session_key": session_key,
        "context_mode": plan.mode if use_generated_context_files else "inline",
        "upstream_files": upstream_files,
        "bound_account": acc,
        "bound_account_email": acc.email,
        "generated_local_files": local_file_records,
        "context_attachment_tokens": context_attachment_tokens + inline_text_tokens,
        "attachment_fallback": False,
    }
