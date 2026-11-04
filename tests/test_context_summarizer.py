"""两段式总结（`ContextSummarizer`）与「摘要优先、失败回退截断」的端到端行为。

上游对**大段内联 prompt** 直接返回 x5sec punish（实测 8k 过 / 12k 必挂），所以历史
超阈值时不能原样内联。两段式总结是硬截断的**升级**：

```
历史超阈值
  ├─ 第一段：历史正文 + 「请总结以上对话要点」 → 摘要（进缓存）
  └─ 第二段：摘要 + 本次用户消息 → 最终回答
```

本文件锁死四件事：

1. 摘要成功时，最终 prompt 里是**摘要**而不是原文，且长度回到阈值内；
2. **缓存真的生效**：同一输入第二次不再打上游（fake client 只被调用 1 次）；
3. 历史变化 → key 变化 → miss；
4. 摘要超时 / 抛异常 / 关闭开关 → 回退硬截断，结果仍 ≤ 预算、上下文不清空、不 500。
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import prepare_context_attachments
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


class _FakeClient:
    """假 QwenClient：记录调用次数，按脚本返回 / 抛异常 / 挂住。"""

    def __init__(self, *, reply: str = "SUMMARY_TEXT", error: Exception | None = None, delay: float = 0.0):
        self.reply = reply
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []
        self.prompts: list[str] = []

    async def complete_chat_once(self, token, chat_id, payload, account=None, timeout=None):
        self.calls.append({"token": token, "chat_id": chat_id, "payload": payload, "timeout": timeout})
        message = (payload.get("messages") or [{}])[0]
        self.prompts.append(str(message.get("content") or ""))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {"status": 200, "body": f'{{"data": {{"messages": [{{"content": "{self.reply}"}}]}}}}'}


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
        client = _FakeClient(reply="CACHED_SUMMARY")
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
        client = _FakeClient(reply="OK")
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
            async def complete_chat_once(self, token, chat_id, payload, account=None, timeout=None):
                self.calls.append({"payload": payload})
                return {"status": 500, "body": "boom"}

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
        self.assertFalse(payload["stream"])

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
        client = _FakeClient(reply="CACHED_SUMMARY")
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
        client = _FakeClient(reply="S")
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
        client = _FakeClient(reply="RECENT_ONLY")
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
