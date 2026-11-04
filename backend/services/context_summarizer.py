"""上下文摘要器（两段式总结的第二段入口）。

## 为什么需要它

上游对**大段内联 prompt** 不是「变慢」而是**直接拒绝**：实测

```
短文本 200/6.5s | 6k 200/10.3s | 8k 200/12.2s | 12k ❌ punish | 15k ❌ punish
```

所以在历史超过阈值时，硬截断（`ATTACHMENT_INLINE_TEXT_MAX_CHARS`，按整条丢最早
的消息）虽然能保证请求不被 punish，但**会丢上下文**。两段式总结把这条降级路径
变成「不丢要点」：

```
历史超阈值
  ├─ 第一段请求：历史正文 + 「请总结以上对话要点」  → 摘要
  └─ 第二段请求：摘要 + 本次用户消息（+工具上下文） → 最终回答
```

本模块只负责**第一段**（把历史正文压成要点摘要），第二段由调用方照常发起。

## 设计约束

1. **绝不抛异常给调用方**：超时 / 额度 / 上游异常一律返回 `None`，由调用方回退到
   现有的硬截断。摘要是一条**优化**路径，不能成为新的 500 来源。
2. **缓存必须真的生效**：同一 session 连续对话时历史不变 → 第二次直接命中内存
   缓存（日志 `cached=true`），不再打上游。key 里带 model（不同模型摘要质量不同）。
3. **摘要请求天然不会二次触发上下文卸载 / 附件预处理**：它走
   `QwenClient.stream_chat_once()`（`chat_transport=True`、`accept: application/json`），
   payload 是单条纯文本 user 消息、`files=[]`，不经过 `prepare_context_attachments`
   管线。代码里对这两点做了**保护性校验**（`_assert_isolated_payload`），一旦将来
   有人把这条请求接回主管线会立刻失败而不是静默递归。
4. **摘要请求必须是流式**：t2t 对话上游**不支持非流式** —— `stream=False` 会得到
   HTTP 200 + `{"success":false,"data":{"code":"Bad_Request"}}`（生产实测 b430d19
   每次都失败，日志只有 `empty body`）。所以 payload 保持
   `stream=True` / `incremental_output=True`，按 SSE 逐帧取
   `choices[].delta.content`，并且**只取非思考** phase 的正文。
5. **摘要太短 = 失败**：源文本 > `MIN_SOURCE_CHARS_FOR_LENGTH_CHECK` 却只摘要出
   < `MIN_SUMMARY_CHARS` 字符（生产实测出现过 `summary=2`），一律判失败回退截断 ——
   把垃圾摘要塞进 prompt 等于把上下文清空。
6. **摘要只吃「历史前缀」**：本模块只负责「把传进来的文本压成摘要」，切分口径由
   调用方（`context_attachment_manager._split_current_task()` /
   `_trim_current_task_from_blob()`）保证 —— 传进来的绝不能含本次提问，否则缓存
   key 会随问题变化而永远 miss。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from backend.core.config import settings as default_settings
from backend.upstream.payload_builder import build_chat_payload
from backend.upstream.sse_consumer import parse_sse_chunk

log = logging.getLogger("qwen2api.context_summary")

#: 摘要 prompt（与用户确认过的措辞，别随意改：它决定了摘要的覆盖维度）。
SUMMARY_PROMPT_TEMPLATE = (
    "请把下面这段对话历史压缩成要点摘要，保留：任务目标、已达成的结论、未完成事项、\n"
    "关键数据与约束。不要遗漏用户明确提出的要求。输出纯文本，不超过 {target} 字。\n"
    "\n"
    "{body}"
)

#: 历史被裁到 `CONTEXT_SUMMARY_MAX_SOURCE_CHARS` 以内时的标注（会同时进摘要 prompt
#: 和最终注入的 blob，让模型知道「更早的内容不是没有，是被省略了」）。
OMITTED_NOTE_TEMPLATE = "（更早的 {omitted} 条消息已省略）"

#: 历史 blob 里每条消息的分块头（与 `ContextOffloader.plan()` 的序列化格式一致）。
_BLOCK_HEADER_MARKER = "## Message "

#: 摘要结果超长时的截断后缀。
_TRIM_SUFFIX = "…"

#: 摘要正文的**最短**合理长度。生产实测出现过 `summary=2`（只有 2 个字符）——
#: 那不是摘要，是上游回了个噪声（或解析只捡到一个残片）。这种 blob 塞进 prompt
#: 等于把上下文清空，必须判失败并回退硬截断。
MIN_SUMMARY_CHARS = 20

#: 「太短」判定只在**源文本足够长**时生效：短历史本来就该产出短摘要，用同一个
#: 下限去卡会误杀正常结果。
MIN_SOURCE_CHARS_FOR_LENGTH_CHECK = 1000

#: 日志里摘要预览的字符数（`summary_preview='...'`）。
SUMMARY_PREVIEW_CHARS = 80

#: 从上游响应体里取正文时优先尝试的路径（Qwen 非流式 completions 是
#: `data.messages[0].content`；OpenAI 兼容体是 `choices[0].message.content`）。
_TEXT_PATHS: tuple[tuple[Any, ...], ...] = (
    ("data", "messages", 0, "content"),
    ("choices", 0, "message", "content"),
    ("choices", 0, "text"),
    ("messages", 0, "content"),
    ("data", "content"),
    ("content",),
)

#: 递归兜底时认的「正文键」（顺序即优先级）。
_TEXT_KEYS: tuple[str, ...] = ("content", "answer", "text", "output_text", "response")

#: 上游「思考」阶段的 phase 名（多版本拼写）。这些帧的正文**不进摘要**：
#: 思考过程会把摘要预算吃光，而且模型自己也不会需要它。
_THINK_PHASES: frozenset[str] = frozenset({"think", "thinking", "thinking_summary"})

#: 默认值（配置缺失时的兜底，与 `config.py` 里的默认保持一致）。
_DEFAULTS: dict[str, Any] = {
    "CONTEXT_SUMMARY_ENABLED": True,
    "CONTEXT_SUMMARY_THRESHOLD_CHARS": 8000,
    "CONTEXT_SUMMARY_MAX_SOURCE_CHARS": 24000,
    "CONTEXT_SUMMARY_TARGET_CHARS": 2000,
    "CONTEXT_SUMMARY_TIMEOUT_SECONDS": 60.0,
    "CONTEXT_SUMMARY_CACHE_TTL_SECONDS": 1800.0,
    "CONTEXT_SUMMARY_CACHE_MAX_ENTRIES": 256,
}


def _as_bool(value: Any, default: bool) -> bool:
    """把配置值归一成 bool（兼容 `"true"/"0"/"on"` 这类 env 字符串）。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int:
    """把配置值归一成 int；非法值回落默认。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """把配置值归一成 float；非法值回落默认。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dig(obj: Any, path: tuple[Any, ...]) -> Any:
    """按路径逐层取值；任一层缺失/类型不符返回 None。"""
    node = obj
    for step in path:
        if isinstance(step, int):
            if not isinstance(node, list) or len(node) <= step:
                return None
            node = node[step]
            continue
        if not isinstance(node, dict):
            return None
        node = node.get(step)
    return node


def extract_summary_text(body: str) -> str:
    """从上游**非流式**响应体里取出正文。

    摘要主路径已改成流式（见 :class:`SummaryStreamCollector`），本函数保留为
    「少数部署仍回非流式 JSON 体」时的兜底解析器，也被单测直接覆盖。

    Args:
        body: `QwenClient.complete_chat_once()` 返回的 `body`（JSON 文本，少数
            部署下可能是裸文本）。

    Returns:
        正文（已 strip）；取不到返回空串。
    """
    text = str(body or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        # 不是 JSON：上游偶尔直接回纯文本，按正文处理。
        return text

    for path in _TEXT_PATHS:
        value = _dig(parsed, path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # 递归兜底：Qwen 的响应结构会随版本漂移（extra / 嵌套 messages），
    # 与其逐版本适配，不如按「正文键」广度优先找第一个非空字符串。
    queue: list[Any] = [parsed]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for key in _TEXT_KEYS:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return ""


def describe_upstream_error(event: dict) -> str:
    """把 `parse_sse_chunk()` 的错误事件翻译成可诊断的一行消息。

    上游风控软拦的形态是 HTTP 200 + `{"success":false,"data":{"code":"Bad_Request",
    "details":"..."}}`；不把 `code` / `details` 带进异常消息，日志里就只剩
    `empty body`，等于把原因扔掉了。

    Args:
        event: `parse_sse_chunk()` 产出的 `type == "error"` 事件。

    Returns:
        形如 `upstream error code=Bad_Request details=Internal error ...` 的消息。
    """
    code = str(event.get("code") or "").strip() or "unknown"
    details = str(event.get("details") or "").strip()
    message = f"upstream error code={code}"
    if details:
        message = f"{message} details={details[:200]}"
    return message


class SummaryStreamCollector:
    """增量解析摘要请求的 SSE 流，按 phase 分桶收集正文。

    两个必须处理的现实：

    1. **分片边界与行边界无关**：`stream_chat_once()` 按 TCP 到达分片 yield，
       一行 `data: {...}` 可能被切成两半，所以先按 `\\n` 缓冲、凑齐一行才解析；
    2. **上游会混入思考帧**：`phase` 为 `think` / `thinking` / `thinking_summary`
       的正文若也拼进摘要，摘要预算会被思考过程吃光。

    因此正文优先取「非思考」帧；全是思考、没有 answer 时才退回思考正文兜底
    （总比空摘要 → 回退截断好）。
    """

    def __init__(self) -> None:
        """初始化收集器（每个请求一个实例，不要复用）。"""
        self.answer: list[str] = []
        self.think: list[str] = []
        self.error: str = ""
        self.saw_data_line: bool = False
        self._raw: list[str] = []
        self._buffer: str = ""

    def feed(self, chunk: Any) -> None:
        """喂入一个流分片（已解码文本），按整行解析。"""
        text = str(chunk or "")
        if not text:
            return
        self._raw.append(text)
        self._buffer += text
        while True:
            newline = self._buffer.find("\n")
            if newline < 0:
                return
            line = self._buffer[:newline]
            self._buffer = self._buffer[newline + 1 :]
            self._handle_line(line)

    def flush(self) -> None:
        """处理最后一个没有换行结尾的残行（流结束时调用一次）。"""
        remainder = self._buffer
        self._buffer = ""
        if remainder.strip():
            self._handle_line(remainder)

    def body(self) -> str:
        """拼出摘要正文：优先非思考内容，全是思考时用思考正文兜底。"""
        answer = "".join(self.answer).strip()
        if answer:
            return answer
        return "".join(self.think).strip()

    def raw_text(self) -> str:
        """原始流文本（诊断 / 非流式兜底解析用）。"""
        return "".join(self._raw)

    def _handle_line(self, line: str) -> None:
        """解析单行：`data: {...}` 走 SSE，裸 JSON 错误体走 sse_consumer 的兜底。"""
        stripped = str(line or "").strip()
        if not stripped:
            return
        if stripped.startswith("data:"):
            self.saw_data_line = True
        try:
            events = parse_sse_chunk(stripped)
        except Exception:  # noqa: BLE001 - 单行解析失败不该中断整个摘要
            return
        if not events and stripped.startswith("data:"):
            # 错误信封也可能被包在 `data:` 里（{"success":false,...}）：把 JSON 本体
            # 去掉前缀再走一次，命中 sse_consumer 的裸 JSON 错误识别。
            events = self._parse_wrapped_error(stripped[5:].strip())
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                if not self.error:
                    self.error = describe_upstream_error(event)
                continue
            content = event.get("content")
            if not isinstance(content, str) or not content:
                continue
            phase = str(event.get("phase") or "answer")
            if phase in _THINK_PHASES:
                self.think.append(content)
            else:
                self.answer.append(content)

    @staticmethod
    def _parse_wrapped_error(payload_text: str) -> list[dict]:
        """把 `data:` 前缀后的 JSON 当裸错误体重解析一次（非错误体返回空列表）。"""
        if not payload_text.startswith("{"):
            return []
        try:
            payload = json.loads(payload_text)
        except (TypeError, ValueError):
            return []
        if not isinstance(payload, dict) or payload.get("success") is not False:
            return []
        try:
            return parse_sse_chunk(payload_text)
        except Exception:  # noqa: BLE001 - 解析失败按「无事件」处理
            return []


def clip_history_source(text: str, max_chars: int) -> tuple[str, int]:
    """把历史正文裁到最近 `max_chars` 字符（**保留最新**，丢弃最早）。

    只把最近的一段送去总结：更早的内容对「当前任务」价值最低，而总结请求本身
    也受同一条 punish 阈值约束（24k 摘要输入 ≈ 一次 24k 内联请求，仍在安全区）。

    Args:
        text: 完整历史序列化文本（`## Message N [role]` 格式）。
        max_chars: 保留上限；`<= 0` 表示不裁剪。

    Returns:
        `(裁剪后文本, 被省略的消息条数)`。未裁剪时第二项为 0，且第一项与入参
        逐字节相同。
    """
    source = str(text or "")
    limit = _as_int(max_chars, 0)
    if limit <= 0 or len(source) <= limit:
        return source, 0

    head = source[: len(source) - limit]
    tail = source[-limit:]
    # 对齐到整条消息边界：tail 开头若是「半条消息」，从下一个分块头开始，
    # 绝不把半个 `## Message N [user]` 头送进摘要请求。
    boundary = tail.find("\n" + _BLOCK_HEADER_MARKER)
    if boundary >= 0:
        head = head + tail[: boundary + 1]
        tail = tail[boundary + 1 :]
    omitted = head.count(_BLOCK_HEADER_MARKER)
    return tail, omitted


def count_history_messages(text: str) -> int:
    """数历史 blob 里有多少条 `## Message N [role]` 消息。"""
    return str(text or "").count(_BLOCK_HEADER_MARKER)


class ContextSummarizer:
    """把大段对话历史压成要点摘要（带内存 LRU + TTL 缓存）。

    实例应**复用**（缓存挂在实例上）：同一个 `settings` 对象对应同一个实例，
    见 :func:`get_context_summarizer`。
    """

    def __init__(self, settings_obj: Any = None) -> None:
        """初始化摘要器。

        Args:
            settings_obj: 配置对象（`backend.core.config.settings` 或测试替身）。
                缺省用全局 `settings`。
        """
        self.settings = settings_obj if settings_obj is not None else default_settings
        # key -> (summary, created_at_monotonic)；OrderedDict 用作 LRU。
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.stats: dict[str, int] = {"hits": 0, "misses": 0, "sets": 0, "failures": 0}

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------
    def _cfg(self, name: str) -> Any:
        """读配置项；缺失时回落到模块默认值。"""
        return getattr(self.settings, name, _DEFAULTS.get(name))

    def enabled(self) -> bool:
        """摘要总开关（`CONTEXT_SUMMARY_ENABLED`）。"""
        return _as_bool(self._cfg("CONTEXT_SUMMARY_ENABLED"), bool(_DEFAULTS["CONTEXT_SUMMARY_ENABLED"]))

    def threshold_chars(self) -> int:
        """历史超过多少字符才触发摘要（`CONTEXT_SUMMARY_THRESHOLD_CHARS`）。"""
        return _as_int(self._cfg("CONTEXT_SUMMARY_THRESHOLD_CHARS"), int(_DEFAULTS["CONTEXT_SUMMARY_THRESHOLD_CHARS"]))

    def max_source_chars(self) -> int:
        """送去总结的历史正文上限（`CONTEXT_SUMMARY_MAX_SOURCE_CHARS`）。"""
        return _as_int(self._cfg("CONTEXT_SUMMARY_MAX_SOURCE_CHARS"), int(_DEFAULTS["CONTEXT_SUMMARY_MAX_SOURCE_CHARS"]))

    def target_chars(self) -> int:
        """摘要目标字数（`CONTEXT_SUMMARY_TARGET_CHARS`）。"""
        return _as_int(self._cfg("CONTEXT_SUMMARY_TARGET_CHARS"), int(_DEFAULTS["CONTEXT_SUMMARY_TARGET_CHARS"]))

    def timeout_seconds(self) -> float:
        """摘要请求超时（`CONTEXT_SUMMARY_TIMEOUT_SECONDS`）。"""
        return max(0.1, _as_float(self._cfg("CONTEXT_SUMMARY_TIMEOUT_SECONDS"), float(_DEFAULTS["CONTEXT_SUMMARY_TIMEOUT_SECONDS"])))

    def cache_ttl_seconds(self) -> float:
        """缓存 TTL；`<= 0` 表示不使用缓存。"""
        return _as_float(self._cfg("CONTEXT_SUMMARY_CACHE_TTL_SECONDS"), float(_DEFAULTS["CONTEXT_SUMMARY_CACHE_TTL_SECONDS"]))

    def cache_max_entries(self) -> int:
        """缓存条目上限（LRU 淘汰）；`<= 0` 表示不使用缓存。"""
        return _as_int(self._cfg("CONTEXT_SUMMARY_CACHE_MAX_ENTRIES"), int(_DEFAULTS["CONTEXT_SUMMARY_CACHE_MAX_ENTRIES"]))

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------
    def _cache_key(self, text: str, model: str, target_chars: int) -> str:
        """缓存 key = `sha256(text + model + target_chars)`。

        用 `\\x00` 分隔三个字段，避免「正文结尾 + 模型名前缀」错位拼出同一个
        hash（`text="ab", model="c"` 与 `text="a", model="bc"` 必须不同）。
        """
        basis = "\x00".join((str(text or ""), str(model or ""), str(int(target_chars))))
        return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        """取缓存；未命中 / 已过期返回 None。命中时刷新 LRU 位置。"""
        entry = self._cache.get(key)
        if entry is None:
            self.stats["misses"] += 1
            return None
        summary, created_at = entry
        ttl = self.cache_ttl_seconds()
        if ttl <= 0 or (time.monotonic() - created_at) > ttl:
            self._cache.pop(key, None)
            self.stats["misses"] += 1
            return None
        self._cache.move_to_end(key)
        self.stats["hits"] += 1
        return summary

    def _cache_set(self, key: str, summary: str) -> None:
        """写缓存并按 LRU 淘汰超出上限的最旧条目。"""
        text = str(summary or "")
        if not text:
            return
        max_entries = self.cache_max_entries()
        if max_entries <= 0 or self.cache_ttl_seconds() <= 0:
            return
        self._cache[key] = (text, time.monotonic())
        self._cache.move_to_end(key)
        while len(self._cache) > max_entries:
            self._cache.popitem(last=False)
        self.stats["sets"] += 1

    def clear_cache(self) -> None:
        """清空缓存（管理接口 / 单测用）。"""
        self._cache.clear()

    def cache_size(self) -> int:
        """当前缓存条目数。"""
        return len(self._cache)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    async def summarize(
        self,
        *,
        text: str,
        model: str,
        account: Any,
        client: Any,
        target_chars: int | None = None,
        max_source_chars: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str | None:
        """把历史正文压成要点摘要。

        Args:
            text: 完整历史序列化文本（`## Message N [role]` 格式）。
            model: 上游模型名；空串表示跟主请求同模型（由调用方决定）。
            account: 用于发请求的上游账号（需带 `token`）。
            client: `QwenClient`（或其测试替身），需实现 `stream_chat_once`
                （异步生成器，逐片 yield `{"chunk": ...}` / `{"status": ...}`）。
            target_chars: 覆盖配置里的摘要目标字数。
            max_source_chars: 覆盖配置里的「送去总结的历史上限」。
            timeout_seconds: 覆盖配置里的超时。

        Returns:
            摘要文本（已 trim 到 `target_chars` 以内，历史被裁过时带
            「更早的 N 条消息已省略」标注）；任何失败返回 `None`，调用方回退截断。
        """
        source_text = str(text or "")
        if not source_text.strip():
            return None
        if not self.enabled():
            log.info("[ContextSummary] skipped reason=disabled")
            return None

        target = self.target_chars() if target_chars is None else _as_int(target_chars, self.target_chars())
        if target <= 0:
            log.info("[ContextSummary] skipped reason=target_chars<=0")
            return None
        limit = self.max_source_chars() if max_source_chars is None else _as_int(max_source_chars, self.max_source_chars())
        timeout = self.timeout_seconds() if timeout_seconds is None else max(0.1, _as_float(timeout_seconds, self.timeout_seconds()))
        model_name = str(model or "")

        source, omitted = clip_history_source(source_text, limit)
        key = self._cache_key(source, model_name, target)
        cached = self._cache_get(key)
        if cached is not None:
            log.info(
                "[ContextSummary] ok original=%d summary=%d model=%s cached=true elapsed=%.1fs summary_preview=%s",
                len(source_text),
                len(cached),
                model_name or "-",
                0.0,
                _preview(cached),
            )
            return cached

        prompt = build_summary_prompt(source, target=target, omitted_messages=omitted)
        started = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                self._request_summary(
                    prompt=prompt,
                    model=model_name,
                    account=account,
                    client=client,
                ),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            # 调用方取消（客户端断连）不是「摘要失败」，必须原样上抛。
            raise
        except asyncio.TimeoutError:
            self.stats["failures"] += 1
            log.warning(
                "[ContextSummary] failed reason=timeout(%.0fs) → fallback truncate original=%d",
                timeout,
                len(source_text),
            )
            return None
        except Exception as exc:  # noqa: BLE001 - 摘要失败一律降级，绝不冒泡
            self.stats["failures"] += 1
            log.warning(
                "[ContextSummary] failed reason=%s → fallback truncate original=%d",
                f"{type(exc).__name__}: {str(exc)[:160]}",
                len(source_text),
            )
            return None

        summary = trim_summary(raw, target)
        if not summary:
            self.stats["failures"] += 1
            log.warning(
                "[ContextSummary] failed reason=empty_response → fallback truncate original=%d",
                len(source_text),
            )
            return None

        if len(summary) < MIN_SUMMARY_CHARS and len(source_text) > MIN_SOURCE_CHARS_FOR_LENGTH_CHECK:
            # 源文本上千字符却只摘要出十几个字符：这是上游噪声，不是摘要。
            # 塞进 prompt 等于把上下文清空 —— 宁可用硬截断的原文。
            self.stats["failures"] += 1
            log.warning(
                "[ContextSummary] failed reason=summary too short summary=%d min=%d original=%d summary_preview=%s → fallback truncate",
                len(summary),
                MIN_SUMMARY_CHARS,
                len(source_text),
                _preview(summary),
            )
            return None

        if omitted > 0:
            summary = f"{OMITTED_NOTE_TEMPLATE.format(omitted=omitted)}\n{summary}"
        self._cache_set(key, summary)
        log.info(
            "[ContextSummary] ok original=%d summary=%d model=%s cached=false elapsed=%.1fs summary_preview=%s",
            len(source_text),
            len(summary),
            model_name or "-",
            time.monotonic() - started,
            _preview(summary),
        )
        return summary

    async def _request_summary(
        self,
        *,
        prompt: str,
        model: str,
        account: Any,
        client: Any,
    ) -> str:
        """发一次**流式** completions，把 SSE 正文拼成摘要。

        为什么必须流式：t2t 对话上游**不支持非流式** —— `stream=False` 会拿到
        HTTP 200 + `{"success":false,"data":{"code":"Bad_Request"}}`（只有视频 t2v
        才需要非流式，因为它要的是 body 里的 `task_id`）。走流式的另一个好处是
        能按 `phase` 过滤掉思考帧。

        超时由调用方的 `asyncio.wait_for` 统一负责（`stream_chat_once` 不接受
        `timeout` 参数，底层超时取 `QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS`）。

        Raises:
            RuntimeError: 传输不可用 / 上游非 200 / 上游错误信封 / 取不到正文。
                错误消息里必须带 `code` / `details`，否则日志只剩 `empty body`。
        """
        stream = getattr(client, "stream_chat_once", None)
        if not callable(stream):
            raise RuntimeError("stream transport unavailable")
        if account is None:
            raise RuntimeError("account unavailable")

        chat_id = await self._resolve_chat_id(client, account, model)
        payload = build_chat_payload(
            chat_id,
            model,
            prompt,
            has_custom_tools=False,
            files=None,
            chat_type="t2t",
        )
        # 保持 build_chat_payload 的 t2t 默认（stream / incremental_output 均为 True），
        # 绝不覆盖成 False —— 非流式上游必然 Bad_Request。
        payload["stream"] = True
        payload["incremental_output"] = True
        _assert_isolated_payload(payload)

        token = str(getattr(account, "token", "") or "")
        collector = SummaryStreamCollector()
        async for item in stream(token, chat_id, payload, account=account):
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if isinstance(status, int) and not isinstance(status, bool) and status != 200:
                preview = " ".join(str(item.get("body") or "").split())[:200]
                raise RuntimeError(f"http {status}: {preview}")
            chunk = item.get("chunk")
            if chunk:
                collector.feed(chunk)
        collector.flush()

        text = collector.body()
        if not text and not collector.saw_data_line:
            # 少数部署仍回非流式 JSON 体（choices / messages / content）：复用旧解析。
            text = extract_summary_text(collector.raw_text())
        if not text and collector.error:
            raise RuntimeError(collector.error)
        if not text:
            raw_text = collector.raw_text()
            preview = " ".join(raw_text.split())[:200]
            raise RuntimeError(f"empty summary body raw_len={len(raw_text)} preview={preview!r}")
        if collector.error:
            log.warning(
                "[ContextSummary] stream had upstream error frame but produced text: %s",
                collector.error,
            )
        return text

    async def _resolve_chat_id(self, client: Any, account: Any, model: str) -> str:
        """取一个 chat_id 发摘要请求（复用执行器的建会话链路）。

        摘要请求必须落在某个上游 chat 里（completions 的 URL 带 `chat_id`）。
        这里复用 `QwenExecutor.get_chat_id()`：它会走 chat 预热池或新建会话，
        与主请求同一条链路，不额外发明协议。测试替身没有 `executor` 时返回空串
        （`build_chat_payload` 允许 `chat_id=""`）。
        """
        executor = getattr(client, "executor", None)
        get_chat_id = getattr(executor, "get_chat_id", None)
        if not callable(get_chat_id):
            return ""
        chat_id, _reused = await get_chat_id(account, model)
        return str(chat_id or "")


def _assert_isolated_payload(payload: dict[str, Any]) -> None:
    """保护性校验：摘要请求必须是「单条纯文本、无附件」的独立请求。

    为什么要在代码里断言而不是只写注释：这条请求一旦被接回主请求管线（比如有人
    复用了带上下文卸载的 payload 构造），就会**递归触发**摘要 / 附件上传 —— 请求
    自己把自己拖死，且现象极难定位。这里直接失败，让回归在测试里就暴露。
    """
    messages = payload.get("messages") or []
    if len(messages) != 1:
        raise RuntimeError(f"summary payload must have exactly 1 message, got {len(messages)}")
    message = messages[0] if isinstance(messages[0], dict) else {}
    if message.get("files"):
        raise RuntimeError("summary payload must not carry attachments")


def build_summary_prompt(text: str, *, target: int, omitted_messages: int = 0) -> str:
    """拼摘要 prompt。

    Args:
        text: 送去总结的历史正文。
        target: 目标字数（写进 prompt，约束输出长度）。
        omitted_messages: 更早被省略的消息条数（>0 时在正文前加一句说明，避免
            模型把「更早的内容不存在」误判成「任务刚开始」）。

    Returns:
        完整的单条 prompt 文本。
    """
    body = str(text or "")
    if omitted_messages > 0:
        body = f"{OMITTED_NOTE_TEMPLATE.format(omitted=omitted_messages)}\n{body}"
    return SUMMARY_PROMPT_TEMPLATE.format(target=int(target), body=body)


def preview_text(text: str, limit: int = SUMMARY_PREVIEW_CHARS) -> str:
    """把文本压成一行预览串（`repr` 形态，日志里带引号、看得见换行）。

    Args:
        text: 原始文本。
        limit: 最多取多少字符。

    Returns:
        形如 `'要点 1…'` 的单行预览；空文本返回 `''`。
    """
    flat = " ".join(str(text or "").split())
    return repr(flat[: max(0, int(limit))])


#: 内部别名（日志里写起来短一点）。
_preview = preview_text


def trim_summary(text: str, target_chars: int) -> str:
    """把摘要 trim 到 `target_chars` 以内（超了按字符截断 + `…`）。"""
    summary = str(text or "").strip()
    if not summary:
        return ""
    limit = _as_int(target_chars, 0)
    if limit <= 0 or len(summary) <= limit:
        return summary
    keep = max(0, limit - len(_TRIM_SUFFIX))
    return f"{summary[:keep].rstrip()}{_TRIM_SUFFIX}"


# ----------------------------------------------------------------------
# 进程级实例（缓存挂在实例上，所以必须复用同一个对象）
# ----------------------------------------------------------------------
#: `id(settings) -> ContextSummarizer`。生产环境只有一个 settings，故只有一项；
#: 单测会造多个 settings 替身，超过 `_MAX_SUMMARIZERS` 就整表重置，避免泄漏。
_SUMMARIZERS: dict[int, ContextSummarizer] = {}
_MAX_SUMMARIZERS = 8


def get_context_summarizer(settings_obj: Any = None) -> ContextSummarizer:
    """取（或创建）与 `settings_obj` 绑定的摘要器实例。

    缓存必须跨请求存活，所以**不能**每次调用都 new 一个。
    """
    key = id(settings_obj) if settings_obj is not None else 0
    summarizer = _SUMMARIZERS.get(key)
    if summarizer is not None:
        return summarizer
    if len(_SUMMARIZERS) >= _MAX_SUMMARIZERS:
        _SUMMARIZERS.clear()
    summarizer = ContextSummarizer(settings_obj)
    _SUMMARIZERS[key] = summarizer
    return summarizer


def reset_context_summarizers() -> None:
    """清空进程级实例表（单测 / 配置热重载用）。"""
    _SUMMARIZERS.clear()
