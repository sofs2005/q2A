"""两段式总结（`ContextSummarizer`）与「摘要优先、失败回退截断」的端到端行为。

上游对**大段内联 prompt** 直接返回 x5sec punish（实测 8k 过 / 12k 必挂），所以历史
超阈值时不能原样内联。两段式总结是硬截断的**升级**：

```
历史超阈值
  ├─ 第一段：历史正文 + 「请总结以上对话要点」 → 摘要（进缓存）
  └─ 第二段：摘要 + 本次用户消息 → 最终回答
```

本文件锁死六件事：

1. 摘要成功时，最终 prompt 里是**摘要**而不是原文，且长度回到阈值内；
2. **缓存真的生效**：同一输入第二次不再打上游（fake client 只被调用 1 次）；
3. 历史变化 → key 变化 → miss；
4. 摘要超时 / 抛异常 / 关闭开关 → 回退硬截断，结果仍 ≤ 预算、上下文不清空、不 500；
5. **摘要只吃「历史前缀」**：最后一条 user 消息（本次提问）及其后的工具续跑尾部
   不进摘要 —— 否则「同一历史 + 不同问题」会让 key 每次都变、缓存永远 miss；
6. **摘要太短 = 失败**：源文本 > 1000 字符却只摘要出 < 20 字符（生产实测出现过
   `summary=2`）→ 判失败、回退截断，绝不把垃圾摘要塞进 prompt。
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import (
    _split_current_task,
    _trim_current_task_from_blob,
    prepare_context_attachments,
)
from backend.services.context_summarizer import (
    ContextSummarizer,
    OMITTED_NOTE_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    clip_history_source,
    count_history_messages,
    extract_summary_text,
    get_context_summarizer,
    reset_context_summarizers,
    trim_summary,
)
from backend.services.document_parse_breaker import (
    get_document_parse_breaker,
    reset_document_parse_breaker,
)
from backend.toolcore.context_offload import ContextOffloader


LATEST_MARKER = "LATEST_TASK_MARKER_keep_me_please"
#: 预算只约束「历史 blob + 附件正文」；最新用户消息与提示行**永不截断**，
#: 所以拼接后的总长度会比预算多出这一小段。
BUDGET_SLACK = 1000


def _settings(**overrides):
    """构造与 `config.py` 默认一致的最小配置替身。"""
    base = dict(
        CONTEXT_INLINE_MAX_CHARS=4000,
        CONTEXT_FORCE_FILE_MAX_CHARS=10000,
        CONTEXT_ATTACHMENT_TTL_SECONDS=600,
        INLINE_ATTACHMENT_ENABLED=True,
        INLINE_ATTACHMENT_MAX_BYTES=10 * 1024 * 1024,
        ATTACHMENT_INLINE_TEXT_ENABLED=True,
        ATTACHMENT_INLINE_TEXT_MAX_BYTES=131072,
        ATTACHMENT_INLINE_TEXT_MAX_CHARS=8000,
        DOCUMENT_PARSE_MAX_FAILURES=2,
        DOCUMENT_PARSE_COOLDOWN_SECONDS=600,
        CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS=5,
        CONTEXT_SUMMARY_ENABLED=True,
        CONTEXT_SUMMARY_THRESHOLD_CHARS=8000,
        CONTEXT_SUMMARY_MAX_SOURCE_CHARS=24000,
        CONTEXT_SUMMARY_TARGET_CHARS=2000,
        CONTEXT_SUMMARY_TIMEOUT_SECONDS=60,
        CONTEXT_SUMMARY_MODEL="",
        CONTEXT_SUMMARY_CACHE_TTL_SECONDS=1800,
        CONTEXT_SUMMARY_CACHE_MAX_ENTRIES=256,
        QWEN_PLUS_MODEL="qwen-plus-latest",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _history_text(count: int = 20, body_chars: int = 750) -> str:
    """构造与 `ContextOffloader.plan()` 同构的历史序列化文本（最早在前）。"""
    parts: list[str] = []
    for idx in range(1, count + 1):
        role = "user" if idx % 2 else "assistant"
        prefix = f"MSG{idx:02d}"
        body = prefix + "x" * max(0, body_chars - len(prefix))
        parts.append(f"## Message {idx} [{role}]\n{body}\n")
    return "".join(parts)


def _sse_delta(content: str, phase: str = "answer") -> str:
    """拼一帧上游 SSE（与 /api/v2/chat/completions 实测形状一致）。"""
    return (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": content, "phase": phase}}]}, ensure_ascii=False)
        + "\n\n"
    )


#: 上游风控软拦的裸 JSON 错误信封（生产实测：stream=False 必回这个）。
BAD_REQUEST_BODY = (
    '{"success":false,"request_id":"req-1",'
    '"data":{"code":"Bad_Request","details":"Internal error, please try again"}}'
)


class _FakeClient:
    """假 QwenClient：记录调用，按脚本产出 SSE 分片 / 抛异常 / 挂住。

    摘要请求走**流式**（t2t 上游不支持非流式），所以替身必须实现
    `stream_chat_once` 异步生成器，逐片 yield `{"chunk": ...}`。
    """

    def __init__(
        self,
        *,
        reply: str = "SUMMARY_TEXT_LONG_ENOUGH_FOR_GUARD",
        error: Exception | None = None,
        delay: float = 0.0,
        stream_body: str | None = None,
        pieces: list[str] | None = None,
    ):
        self.reply = reply
        self.error = error
        self.delay = delay
        self.stream_body = stream_body
        self._pieces = pieces
        self.calls: list[dict] = []
        self.prompts: list[str] = []

    def _build_body(self) -> str:
        if self.stream_body is not None:
            return self.stream_body
        return _sse_delta(self.reply, "answer")

    def _chunks(self) -> list[str]:
        body = self._build_body()
        if self._pieces is not None:
            return list(self._pieces)
        # 默认按 2 片切（顺带覆盖「一帧被切成两半」的缓冲逻辑只有 1 片时不会触发）。
        middle = max(1, len(body) // 2)
        return [body[:middle], body[middle:]]

    async def stream_chat_once(self, token, chat_id, payload, account=None):
        self.calls.append({"token": token, "chat_id": chat_id, "payload": payload})
        message = (payload.get("messages") or [{}])[0]
        self.prompts.append(str(message.get("content") or ""))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        for piece in self._chunks():
            if piece:
                yield {"chunk": piece}
        yield {"status": "streamed"}


def _account(email: str = "bot@example.com") -> SimpleNamespace:
    return SimpleNamespace(email=email, token="tok")


def _build_app(settings, *, client=None, account=None):
    """构造 `prepare_context_attachments` 需要的最小 app。"""
    account = account or _account()
    return SimpleNamespace(
        state=SimpleNamespace(
            context_offloader=ContextOffloader(settings),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=account),
                acquire_wait_preferred=AsyncMock(return_value=account),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(save_text=AsyncMock(), delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(upload_local_file=AsyncMock(), exit_pool=None),
            browser_upload_proxy=None,
            qwen_client=client,
        )
    )


def _payload(messages: list[dict] | None = None) -> dict:
    """20 条 ≈15k 历史 + 一条最新用户消息 + 一个工具（触发上下文卸载）。"""
    if messages is None:
        messages = []
        for idx in range(1, 21):
            role = "user" if idx % 2 else "assistant"
            messages.append({"role": role, "content": f"MSG{idx:02d}" + "x" * 745})
        messages.append({"role": "user", "content": LATEST_MARKER})
    return {
        "model": "gpt-4.1",
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {}}}],
    }


def _joined(result: dict) -> str:
    """把结果 payload 的所有消息正文拼起来（断言用）。"""
    chunks: list[str] = []
    for message in result["payload"]["messages"]:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "\n".join(chunks)


class SummarizerUnitTests(unittest.IsolatedAsyncioTestCase):
    """摘要器本身的缓存 / 裁剪 / 解析行为。"""

    def setUp(self) -> None:
        reset_context_summarizers()

    async def test_summary_is_trimmed_to_target(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_TARGET_CHARS=50))
        client = _FakeClient(reply="S" * 500)

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertIsNotNone(summary)
        self.assertLessEqual(len(summary), 50)
        self.assertTrue(summary.endswith("…"))
        self.assertEqual(len(client.calls), 1)

    async def test_second_identical_call_hits_cache(self) -> None:
        settings = _settings()
        summarizer = ContextSummarizer(settings)
        client = _FakeClient(reply="CACHED_SUMMARY_LONG_ENOUGH_OK")
        text = _history_text(20, 750)

        first = await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)
        second = await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        self.assertEqual(first, second)
        self.assertEqual(len(client.calls), 1, "第二次必须命中缓存，不能再打上游")
        self.assertEqual(summarizer.stats["hits"], 1)
        self.assertEqual(summarizer.cache_size(), 1)

    async def test_cache_key_includes_model(self) -> None:
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient()
        text = _history_text(20, 750)

        await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)
        await summarizer.summarize(text=text, model="qwen-max", account=_account(), client=client)

        self.assertEqual(len(client.calls), 2, "不同模型摘要质量不同，必须各自缓存")

    async def test_history_change_misses_cache(self) -> None:
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient()
        base = _history_text(20, 750)
        changed = base.replace("MSG20", "MSG20-CHANGED")

        await summarizer.summarize(text=base, model="qwen-plus", account=_account(), client=client)
        await summarizer.summarize(text=changed, model="qwen-plus", account=_account(), client=client)

        self.assertEqual(len(client.calls), 2)

    async def test_cache_expires_by_ttl(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_CACHE_TTL_SECONDS=0.01))
        client = _FakeClient()
        text = _history_text(20, 750)

        await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)
        await asyncio.sleep(0.05)
        await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        self.assertEqual(len(client.calls), 2)

    async def test_cache_lru_evicts_oldest(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_CACHE_MAX_ENTRIES=2))
        client = _FakeClient()
        for idx in (1, 2, 3):
            await summarizer.summarize(
                text=_history_text(20, 750) + f"## Message 9{idx} [user]\nzzz\n",
                model="qwen-plus",
                account=_account(),
                client=client,
            )
        self.assertEqual(summarizer.cache_size(), 2)

    async def test_large_history_is_clipped_to_recent_window(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_MAX_SOURCE_CHARS=24000))
        client = _FakeClient(reply="CLIPPED_WINDOW_SUMMARY_LONG_ENOUGH")
        # 40k 历史：只把最近 24k 送去总结，并标注省略条数。
        text = _history_text(50, 800)
        self.assertGreater(len(text), 40000)

        summary = await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        self.assertIsNotNone(summary)
        prompt = client.prompts[0]
        body = prompt.split("\n\n", 1)[1]
        self.assertLessEqual(len(body), 24000 + len(OMITTED_NOTE_TEMPLATE.format(omitted=99)) + 2)
        self.assertIn("更早的", body)
        self.assertIn("条消息已省略", summary)
        self.assertNotIn("MSG01", body)

    async def test_timeout_returns_none(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_TIMEOUT_SECONDS=0.05))
        client = _FakeClient(delay=1.0)

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertIsNone(summary)
        self.assertEqual(summarizer.stats["failures"], 1)

    async def test_upstream_exception_returns_none(self) -> None:
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(error=RuntimeError("quota exceeded"))

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertIsNone(summary)

    async def test_non_200_returns_none(self) -> None:
        summarizer = ContextSummarizer(_settings())

        class _Boom(_FakeClient):
            async def stream_chat_once(self, token, chat_id, payload, account=None):
                self.calls.append({"payload": payload})
                yield {"status": 500, "body": "boom"}

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=_Boom())
        self.assertIsNone(summary)

    async def test_disabled_switch_skips_upstream(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_ENABLED=False))
        client = _FakeClient()

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertIsNone(summary)
        self.assertEqual(client.calls, [])

    async def test_prompt_shape(self) -> None:
        summarizer = ContextSummarizer(_settings(CONTEXT_SUMMARY_TARGET_CHARS=1234))
        client = _FakeClient()
        text = _history_text(20, 750)

        await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        prompt = client.prompts[0]
        self.assertTrue(prompt.startswith("请把下面这段对话历史压缩成要点摘要"))
        self.assertIn("不超过 1234 字", prompt)
        self.assertIn("MSG20", prompt)
        # 单条纯文本 + 无附件：摘要请求天然不会二次触发卸载 / 附件预处理。
        payload = client.calls[0]["payload"]
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["files"], [])
        # t2t 上游不支持非流式：stream / incremental_output 必须是 True。
        self.assertTrue(payload["stream"], "摘要请求必须走流式，非流式上游回 Bad_Request")
        self.assertTrue(payload["incremental_output"])

    async def test_stream_parses_only_answer_phase(self) -> None:
        """SSE 多帧含 think + answer：只拼 answer，思考不污染摘要。"""
        summarizer = ContextSummarizer(_settings())
        body = (
            'data: {"response.created":{"response_id":"r1"}}\n\n'
            + _sse_delta("让我先想想…", "think")
            + _sse_delta("思考摘要片段", "thinking_summary")
            + _sse_delta("ANSWER_PART_1 ", "answer")
            + _sse_delta("ANSWER_PART_2", "answer")
        )
        client = _FakeClient(stream_body=body)

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertEqual(summary, "ANSWER_PART_1 ANSWER_PART_2")
        self.assertNotIn("让我先想想", summary or "")
        self.assertNotIn("思考摘要", summary or "")

    async def test_stream_falls_back_to_think_when_no_answer(self) -> None:
        """只有思考帧、没有 answer 时用思考正文兜底（总比空摘要→截断好）。"""
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(stream_body=_sse_delta("ONLY_THINKING_PHASE_LONG_ENOUGH", "think"))

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertEqual(summary, "ONLY_THINKING_PHASE_LONG_ENOUGH")

    async def test_stream_frame_split_across_chunks_is_reassembled(self) -> None:
        """一帧被 TCP 切成两半（甚至切在 `data:` 中间）也要拼回来。"""
        summarizer = ContextSummarizer(_settings())
        frame = _sse_delta("SPLIT_OK_REASSEMBLED_LONG_ENOUGH", "answer")
        pieces = [frame[:7], frame[7:25], frame[25:]]
        client = _FakeClient(pieces=pieces)

        summary = await summarizer.summarize(text=_history_text(20, 750), model="qwen-plus", account=_account(), client=client)

        self.assertEqual(summary, "SPLIT_OK_REASSEMBLED_LONG_ENOUGH")

    async def test_upstream_error_envelope_is_diagnosable(self) -> None:
        """`{"success":false,"data":{"code":"Bad_Request"}}` → 日志里必须出现 Bad_Request。"""
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(stream_body=BAD_REQUEST_BODY)

        with self.assertLogs("qwen2api.context_summary", level="WARNING") as logs:
            summary = await summarizer.summarize(
                text=_history_text(20, 750),
                model="qwen-plus",
                account=_account(),
                client=client,
            )

        self.assertIsNone(summary)
        messages = [record.getMessage() for record in logs.records]
        self.assertTrue(any("Bad_Request" in line for line in messages), messages)
        self.assertFalse(any("empty body" == line.rstrip() for line in messages), messages)

    async def test_upstream_error_envelope_wrapped_in_data_line(self) -> None:
        """错误信封也可能被包在 `data:` 里：同样要留下 Bad_Request。"""
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(stream_body=f"data: {BAD_REQUEST_BODY}\n\n")

        with self.assertLogs("qwen2api.context_summary", level="WARNING") as logs:
            summary = await summarizer.summarize(
                text=_history_text(20, 750),
                model="qwen-plus",
                account=_account(),
                client=client,
            )

        self.assertIsNone(summary)
        messages = [record.getMessage() for record in logs.records]
        self.assertTrue(any("Bad_Request" in line for line in messages), messages)

    async def test_upstream_error_envelope_without_choices_raises_code(self) -> None:
        """错误信封走 `_request_summary` 时异常消息里也要带 code / details。"""
        summarizer = ContextSummarizer(_settings())

        with self.assertRaises(RuntimeError) as ctx:
            await summarizer._request_summary(
                prompt="P",
                model="qwen-plus",
                account=_account(),
                client=_FakeClient(stream_body=BAD_REQUEST_BODY),
            )
        message = str(ctx.exception)
        self.assertIn("Bad_Request", message)
        self.assertIn("Internal error", message)

    async def test_too_short_summary_is_rejected_when_source_is_long(self) -> None:
        """源文本 > 1000 字符却只摘要出 < 20 字符 → 判失败、回退截断、不进缓存。

        生产实测出现过 `summary=2`：那不是摘要，是上游噪声。把它塞进 prompt 等于
        把上下文清空，宁可用硬截断的原文。
        """
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(reply="OK")
        text = _history_text(20, 750)
        self.assertGreater(len(text), 1000)

        with self.assertLogs("qwen2api.context_summary", level="WARNING") as logs:
            summary = await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        self.assertIsNone(summary)
        messages = [record.getMessage() for record in logs.records]
        self.assertTrue(any("reason=summary too short" in line for line in messages), messages)
        self.assertTrue(any("summary_preview=" in line for line in messages), messages)
        self.assertEqual(summarizer.cache_size(), 0, "垃圾摘要绝不能进缓存")
        self.assertEqual(summarizer.stats["failures"], 1)

    async def test_short_summary_is_accepted_when_source_is_short(self) -> None:
        """源文本本身就短（≤ 1000 字符）时，短摘要不算失败（否则误杀正常结果）。"""
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(reply="OK")
        text = _history_text(2, 100)
        self.assertLessEqual(len(text), 1000)

        summary = await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        self.assertEqual(summary, "OK")

    async def test_ok_log_lines_carry_summary_preview(self) -> None:
        """`ok` 日志（miss + hit 两行）都要带 `summary_preview='...'`。"""
        summarizer = ContextSummarizer(_settings())
        client = _FakeClient(reply="PREVIEWED_SUMMARY_LONG_ENOUGH")
        text = _history_text(20, 750)

        with self.assertLogs("qwen2api.context_summary", level="INFO") as logs:
            await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)
            await summarizer.summarize(text=text, model="qwen-plus", account=_account(), client=client)

        ok_lines = [record.getMessage() for record in logs.records if "[ContextSummary] ok" in record.getMessage()]
        self.assertEqual(len(ok_lines), 2, ok_lines)
        self.assertIn("cached=false", ok_lines[0])
        self.assertIn("cached=true", ok_lines[1])
        for line in ok_lines:
            self.assertIn("summary_preview='PREVIEWED_SUMMARY_LONG_ENOUGH'", line)

    def test_clip_history_source_keeps_tail_and_counts_omitted(self) -> None:
        text = _history_text(50, 800)
        clipped, omitted = clip_history_source(text, 24000)

        self.assertLessEqual(len(clipped), 24000)
        self.assertGreater(omitted, 0)
        self.assertEqual(omitted + count_history_messages(clipped), count_history_messages(text))
        self.assertTrue(clipped.startswith("## Message "))

    def test_clip_is_noop_when_within_limit(self) -> None:
        text = _history_text(5, 100)
        clipped, omitted = clip_history_source(text, 24000)
        self.assertEqual(clipped, text)
        self.assertEqual(omitted, 0)

    def test_trim_summary_bounds(self) -> None:
        self.assertEqual(trim_summary("abc", 10), "abc")
        self.assertEqual(len(trim_summary("a" * 100, 10)), 10)
        self.assertEqual(trim_summary("", 10), "")

    def test_extract_summary_text_handles_shapes(self) -> None:
        self.assertEqual(extract_summary_text('{"data": {"messages": [{"content": "hi"}]}}'), "hi")
        self.assertEqual(extract_summary_text('{"choices": [{"message": {"content": "yo"}}]}'), "yo")
        self.assertEqual(extract_summary_text("plain text"), "plain text")
        self.assertEqual(extract_summary_text("{}"), "")

    def test_get_context_summarizer_is_shared_per_settings(self) -> None:
        settings = _settings()
        self.assertIs(get_context_summarizer(settings), get_context_summarizer(settings))
        self.assertIsNot(get_context_summarizer(settings), get_context_summarizer(_settings()))

    def test_summary_prompt_template_mentions_required_dimensions(self) -> None:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(target=10, body="BODY")
        for keyword in ("任务目标", "已达成的结论", "未完成事项", "关键数据与约束"):
            self.assertIn(keyword, prompt)


class CurrentTaskSplitTests(unittest.TestCase):
    """切分口径：最后一条 user 消息及其后属于「当前任务」，绝不进摘要。

    这条口径一旦漂移，现象就是「同一段历史 + 不同问题 → 缓存永远 miss」：
    当前提问混进摘要 blob → key 变 → 每次都打上游，而且模型还会把当前任务
    提前总结掉。
    """

    def test_split_puts_last_user_and_tool_tail_into_current_task(self) -> None:
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": LATEST_MARKER},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "tool output"},
        ]

        history, current_task = _split_current_task(messages)

        self.assertEqual([m["content"] for m in history], ["SYS", "Q1", "A1"])
        self.assertEqual([m["role"] for m in current_task], ["user", "assistant", "tool"])
        self.assertIs(current_task[0], messages[3])

    def test_split_without_user_keeps_everything_as_history(self) -> None:
        messages = [{"role": "system", "content": "S"}, {"role": "assistant", "content": "A"}]

        history, current_task = _split_current_task(messages)

        self.assertEqual(history, messages)
        self.assertEqual(current_task, [])

    def test_split_ignores_non_dict_messages(self) -> None:
        history, current_task = _split_current_task(
            ["not-a-message", {"role": "user", "content": "Q1"}, {"role": "user", "content": LATEST_MARKER}]
        )

        self.assertEqual([m["content"] for m in history], ["Q1"])
        self.assertEqual([m["content"] for m in current_task], [LATEST_MARKER])

    def test_trim_blob_drops_last_user_block(self) -> None:
        blob = _history_text(20, 100) + f"\n## Message 21 [user]\n{LATEST_MARKER}\n"

        trimmed = _trim_current_task_from_blob(blob)

        self.assertNotIn(LATEST_MARKER, trimmed)
        self.assertNotIn("## Message 21", trimmed)
        self.assertIn("## Message 20 [assistant]", trimmed)
        self.assertEqual(count_history_messages(trimmed), 20)

    def test_trim_blob_drops_continuation_tail_after_last_user(self) -> None:
        blob = (
            _history_text(4, 100)
            + f"\n## Message 5 [user]\n{LATEST_MARKER}\n"
            + "\n## Message 6 [assistant]\nTOOL CALL\n"
            + "\n## Message 7 [tool]\nTOOL RESULT\n"
        )

        trimmed = _trim_current_task_from_blob(blob)

        self.assertNotIn(LATEST_MARKER, trimmed)
        self.assertNotIn("TOOL RESULT", trimmed)
        self.assertEqual(count_history_messages(trimmed), 4)

    def test_trim_blob_without_headers_is_noop(self) -> None:
        self.assertEqual(_trim_current_task_from_blob("plain text body"), "plain text body")
        self.assertEqual(_trim_current_task_from_blob(""), "")


class SummarizerPipelineTests(unittest.IsolatedAsyncioTestCase):
    """端到端：`prepare_context_attachments` 的摘要优先 / 回退截断。"""

    def setUp(self) -> None:
        reset_document_parse_breaker()
        reset_context_summarizers()

    def tearDown(self) -> None:
        reset_document_parse_breaker()

    async def _prepare(self, settings, client, payload=None):
        """打开 parse 熔断，强制走「内联兜底」路径（生产最常见的那条）。"""
        get_document_parse_breaker(settings).record_failure("waf", immediate=True)
        app = _build_app(settings, client=client)
        return await prepare_context_attachments(
            app=app,
            payload=payload if payload is not None else _payload(),
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:summary",
        )

    async def test_15k_history_becomes_summary_not_raw_text(self) -> None:
        settings = _settings()
        client = _FakeClient(reply="SUMMARY_OF_THE_HISTORY")
        result = await self._prepare(settings, client)

        joined = _joined(result)
        # 原文不在 prompt 里（被摘要替换），摘要 + 最新用户消息在。
        self.assertNotIn("MSG01" + "x" * 745, joined)
        self.assertIn("SUMMARY_OF_THE_HISTORY", joined)
        self.assertIn("【历史摘要】", joined)
        self.assertIn(LATEST_MARKER, joined)
        self.assertLessEqual(len(joined), 8000 + BUDGET_SLACK)
        self.assertEqual(len(client.calls), 1)
        # 摘要请求发的是「历史正文 + 总结指令」，不是用户消息。
        self.assertIn("## Message 1 [user]", client.prompts[0])

    async def test_second_identical_request_hits_cache(self) -> None:
        settings = _settings()
        client = _FakeClient(reply="CACHED_SUMMARY_LONG_ENOUGH_OK")
        await self._prepare(settings, client)
        with self.assertLogs("qwen2api.context_summary", level="INFO") as logs:
            second = await self._prepare(settings, client)

        self.assertEqual(len(client.calls), 1, "历史没变 → 第二次必须命中缓存")
        self.assertIn("CACHED_SUMMARY", _joined(second))
        hit_lines = [record.getMessage() for record in logs.records if "[ContextSummary] ok" in record.getMessage()]
        self.assertEqual(len(hit_lines), 1, hit_lines)
        self.assertIn("cached=true", hit_lines[0])

    async def test_history_change_triggers_second_request(self) -> None:
        settings = _settings()
        client = _FakeClient(reply="HISTORY_CHANGED_SUMMARY_LONG_ENOUGH")
        await self._prepare(settings, client)

        changed = _payload()
        changed["messages"][19]["content"] = "MSG20" + "y" * 745
        await self._prepare(settings, client, payload=changed)

        self.assertEqual(len(client.calls), 2)

    async def test_summary_failure_falls_back_to_truncation(self) -> None:
        settings = _settings()
        client = _FakeClient(error=RuntimeError("quota exceeded"))
        with self.assertLogs("qwen2api.context_attachment_manager", level="INFO") as logs:
            result = await self._prepare(settings, client)

        joined = _joined(result)
        # 回退到硬截断：仍是历史原文（不是摘要），且被压到预算内、上下文没被清空。
        self.assertIn("## Message", joined)
        self.assertIn(LATEST_MARKER, joined)
        self.assertNotIn("【历史摘要】", joined)
        self.assertLessEqual(len(joined), 8000 + BUDGET_SLACK)
        lines = [record.getMessage() for record in logs.records]
        self.assertTrue(any("inline fallback truncated" in line for line in lines), lines)

    async def test_summary_timeout_falls_back_to_truncation(self) -> None:
        settings = _settings(CONTEXT_SUMMARY_TIMEOUT_SECONDS=0.05)
        client = _FakeClient(delay=1.0)
        result = await self._prepare(settings, client)

        joined = _joined(result)
        self.assertIn("## Message", joined)
        self.assertIn(LATEST_MARKER, joined)
        self.assertLessEqual(len(joined), 8000 + BUDGET_SLACK)

    async def test_40k_history_summarizes_only_recent_window(self) -> None:
        settings = _settings()
        client = _FakeClient(reply="RECENT_ONLY_SUMMARY_LONG_ENOUGH")
        messages: list[dict] = []
        for idx in range(1, 51):
            role = "user" if idx % 2 else "assistant"
            messages.append({"role": role, "content": f"MSG{idx:02d}" + "x" * 795})
        messages.append({"role": "user", "content": LATEST_MARKER})

        result = await self._prepare(settings, client, payload=_payload(messages))

        prompt = client.prompts[0]
        self.assertNotIn("MSG01", prompt)
        self.assertIn("MSG50", prompt)
        joined = _joined(result)
        self.assertIn("RECENT_ONLY", joined)
        self.assertIn("更早的", joined)
        self.assertIn("条消息已省略", joined)

    async def test_disabled_switch_uses_plain_truncation(self) -> None:
        settings = _settings(CONTEXT_SUMMARY_ENABLED=False)
        client = _FakeClient(reply="SHOULD_NOT_BE_USED")
        result = await self._prepare(settings, client)

        self.assertEqual(client.calls, [])
        joined = _joined(result)
        self.assertNotIn("SHOULD_NOT_BE_USED", joined)
        self.assertIn("## Message", joined)
        self.assertLessEqual(len(joined), 8000 + BUDGET_SLACK)

    async def test_short_history_never_calls_summarizer(self) -> None:
        """历史 ≤ 阈值：走原来的内联路径，零额外上游请求。"""
        settings = _settings(CONTEXT_SUMMARY_THRESHOLD_CHARS=100000)
        client = _FakeClient()
        result = await self._prepare(settings, client)

        self.assertEqual(client.calls, [], "历史 ≤ 阈值时不该有任何摘要请求")
        joined = _joined(result)
        self.assertNotIn("【历史摘要】", joined)
        self.assertIn("## Message", joined)

    async def test_no_qwen_client_falls_back_silently(self) -> None:
        """精简 app（没有 qwen_client）：直接截断，不制造新的失败点。"""
        settings = _settings()
        get_document_parse_breaker(settings).record_failure("waf", immediate=True)
        app = _build_app(settings, client=None)
        result = await prepare_context_attachments(
            app=app,
            payload=_payload(),
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:no-client",
        )
        joined = _joined(result)
        self.assertIn("## Message", joined)
        self.assertLessEqual(len(joined), 8000 + BUDGET_SLACK)

    # ------------------------------------------------------------------
    # 「摘要只吃历史前缀」：缓存 key 稳定 + 当前提问不进摘要
    # ------------------------------------------------------------------
    async def test_different_question_same_history_hits_cache(self) -> None:
        """同一段历史 + 两个不同的问题 → 摘要只打 1 次上游，第二次 `cached=true`。

        这是生产故障的最小复现：修复前当前提问会被序列化进摘要 blob，换个问题
        blob 就变 → sha256 key 变 → 永远 miss（日志里两次都是 `cached=false`）。
        """
        settings = _settings()
        client = _FakeClient(reply="STABLE_HISTORY_SUMMARY_LONG_ENOUGH")
        await self._prepare(settings, client)

        other = _payload()
        other["messages"][-1]["content"] = "QUESTION_B_MARKER"
        with self.assertLogs("qwen2api.context_summary", level="INFO") as logs:
            second = await self._prepare(settings, client, payload=other)

        self.assertEqual(len(client.calls), 1, "历史没变、只是换了问题 → 第二次必须命中缓存")
        second_joined = _joined(second)
        self.assertIn("STABLE_HISTORY_SUMMARY_LONG_ENOUGH", second_joined)
        self.assertIn("QUESTION_B_MARKER", second_joined, "新问题本体照常进第二段请求")
        ok_lines = [record.getMessage() for record in logs.records if "[ContextSummary] ok" in record.getMessage()]
        self.assertEqual(len(ok_lines), 1, ok_lines)
        self.assertIn("cached=true", ok_lines[0])

    async def test_summary_prompt_excludes_current_question(self) -> None:
        """送进摘要的文本不含当前提问，也不含「最后一条 user」那个 `## Message N` 块。"""
        settings = _settings()
        client = _FakeClient(reply="HISTORY_ONLY_SUMMARY_LONG_ENOUGH")
        result = await self._prepare(settings, client)

        prompt = client.prompts[0]
        self.assertNotIn(LATEST_MARKER, prompt, "当前提问不能进摘要")
        self.assertNotIn("## Message 21 [user]", prompt, "最后一条 user 消息整块不能进摘要")
        self.assertIn("## Message 20 [assistant]", prompt, "它之前的历史必须完整进摘要")

        joined = _joined(result)
        self.assertIn("HISTORY_ONLY_SUMMARY_LONG_ENOUGH", joined)
        self.assertIn(LATEST_MARKER, joined, "当前任务原样保留在摘要之后")

    async def test_inline_path_summarizes_history_prefix_only(self) -> None:
        """不触发上下文卸载的内联路径：同样只摘要历史前缀，换问题命中缓存。"""
        settings = _settings(CONTEXT_INLINE_MAX_CHARS=200000, CONTEXT_SUMMARY_THRESHOLD_CHARS=2000)
        client = _FakeClient(reply="INLINE_PREFIX_SUMMARY_LONG_ENOUGH")
        payload = _payload()
        payload.pop("tools", None)

        first = await prepare_context_attachments(
            app=_build_app(settings, client=client),
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:inline-summary",
        )
        self.assertEqual(first["context_mode"], "summary", "内联路径也被摘要替换")
        self.assertIn("INLINE_PREFIX_SUMMARY_LONG_ENOUGH", _joined(first))
        self.assertNotIn(LATEST_MARKER, client.prompts[0], "当前提问不能进摘要")

        other = dict(payload)
        other["messages"] = [*payload["messages"][:-1], {"role": "user", "content": "QUESTION_B_MARKER"}]
        with self.assertLogs("qwen2api.context_summary", level="INFO") as logs:
            second = await prepare_context_attachments(
                app=_build_app(settings, client=client),
                payload=other,
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:inline-summary",
            )

        self.assertEqual(len(client.calls), 1, "同一历史 + 不同问题 → 第二次必须命中缓存")
        self.assertIn("QUESTION_B_MARKER", _joined(second))
        ok_lines = [record.getMessage() for record in logs.records if "[ContextSummary] ok" in record.getMessage()]
        self.assertEqual(len(ok_lines), 1, ok_lines)
        self.assertIn("cached=true", ok_lines[0])

    async def test_too_short_summary_falls_back_to_truncation(self) -> None:
        """摘要 < 20 字符且源 > 1000 字符 → 判失败，回退截断（垃圾摘要不进 prompt）。"""
        settings = _settings()
        client = _FakeClient(reply="OK")

        with self.assertLogs("qwen2api.context_summary", level="WARNING") as logs:
            result = await self._prepare(settings, client)

        joined = _joined(result)
        self.assertNotIn("【历史摘要】", joined)
        self.assertIn("## Message", joined, "回退到硬截断的历史原文")
        self.assertIn(LATEST_MARKER, joined)
        self.assertLessEqual(len(joined), 8000 + BUDGET_SLACK)
        messages = [record.getMessage() for record in logs.records]
        self.assertTrue(any("reason=summary too short" in line for line in messages), messages)

    async def test_single_message_history_never_calls_summarizer(self) -> None:
        """历史只有 1 条消息（= 当前任务本身）→ 不做摘要，零上游请求。"""
        settings = _settings()
        client = _FakeClient(reply="SHOULD_NOT_BE_USED_AT_ALL")
        payload = _payload([{"role": "user", "content": "x" * 12000}])

        result = await self._prepare(settings, client, payload=payload)

        self.assertEqual(client.calls, [], "切完历史为空时不该有任何摘要请求")
        self.assertNotIn("【历史摘要】", _joined(result))

    async def test_inline_path_single_message_history_skips_summary(self) -> None:
        """内联路径同理：单条 user 消息全算「当前任务」，没有历史可摘要。"""
        settings = _settings(CONTEXT_INLINE_MAX_CHARS=200000, CONTEXT_SUMMARY_THRESHOLD_CHARS=100)
        client = _FakeClient(reply="SHOULD_NOT_BE_USED_AT_ALL")
        payload = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "x" * 9000}]}

        result = await prepare_context_attachments(
            app=_build_app(settings, client=client),
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:inline-single",
        )

        self.assertEqual(client.calls, [], "历史前缀为空时不该有任何摘要请求")
        self.assertEqual(result["context_mode"], "inline")
        self.assertNotIn("【历史摘要】", _joined(result))
