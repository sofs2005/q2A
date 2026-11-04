"""内联上下文字符预算（`ATTACHMENT_INLINE_TEXT_MAX_CHARS`）的截断策略。

上游对**大段内联 prompt** 不是变慢而是**直接拒绝**：实测 8k 通过 / 12k 返回
x5sec punish 页 + idle timeout → 500。所以内联正文必须有体积上限，且截断方向
必须正确：

- 历史 blob 是「最早在前、最新在后」的 `## Message N [role]` 序列 → **丢最早**的；
- 只能**按整条消息**丢，不能按字符位置切（不能切出半个 `## Message N` 头）；
- 最新的用户消息（真正的问题）永不截断。

覆盖：历史 blob 截断、单条超大消息退化、预算内逐字节不变、预算 0 = 不限制、
单文件注入块共享预算、以及走 `prepare_context_attachments` 的端到端日志。
"""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import (
    _BLOCK_TRUNCATION_NOTE,
    _FILE_TRUNCATION_NOTE,
    _inline_char_budget,
    _inject_inline_text_blocks,
    _remaining_budget,
    _truncate_history_blob,
    _truncate_inline_text_blocks,
    prepare_context_attachments,
)
from backend.services.document_parse_breaker import (
    get_document_parse_breaker,
    reset_document_parse_breaker,
)
from backend.toolcore.context_offload import ContextOffloader


BLOCK_HEADER_RE = re.compile(r"^## Message \d+ \[[a-z_]+\]$")


def _settings(**overrides):
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


def _block_bodies(text: str) -> dict[str, str]:
    """把结果文本拆成 `{分块头: 正文}`，顺带校验没有半个分块头。"""
    bodies: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## Message"):
            if not BLOCK_HEADER_RE.match(line):
                raise AssertionError(f"half/`broken block header: {line!r}")
            current = line
            bodies[current] = ""
            continue
        if current is not None:
            bodies[current] += line + "\n"
    return bodies


class HistoryBlobTruncationTests(unittest.TestCase):
    """历史 blob：按整条丢**最早**的消息，保留尾部。"""

    def setUp(self) -> None:
        self.budget = _inline_char_budget(_settings())

    def test_15k_history_keeps_tail_within_budget(self) -> None:
        original = _history_text(20, 750)
        self.assertGreaterEqual(len(original), 15_000)
        self.assertIsNotNone(self.budget)

        result, original_chars, kept_chars, dropped = _truncate_history_blob(original, self.budget)

        self.assertEqual(original_chars, len(original))
        self.assertLessEqual(len(result), self.budget)
        self.assertEqual(kept_chars, len(result.split("\n", 1)[1]))
        # 保留的是**尾部**：最早的 Message 1 消失，最后一条完整保留。
        self.assertNotIn("## Message 1 [user]", result)
        self.assertIn("## Message 20 [assistant]", result)
        self.assertIn("MSG20" + "x" * 745, result)
        self.assertGreater(dropped, 0)
        self.assertLess(dropped, 20)

    def test_dropped_count_and_note_text(self) -> None:
        original = _history_text(20, 750)
        result, original_chars, kept_chars, dropped = _truncate_history_blob(original, self.budget)

        self.assertGreater(dropped, 0)
        note = result.split("\n", 1)[0]
        self.assertTrue(note.startswith("…（已省略前 "), note)
        self.assertIn(f"已省略前 {dropped} 条历史消息", note)
        self.assertIn(f"原始 {original_chars} 字符", note)
        self.assertIn(f"保留最近 {kept_chars} 字符", note)
        # 丢掉的消息确实不在结果里，保留的消息确实在。
        for idx in range(1, dropped + 1):
            role = "user" if idx % 2 else "assistant"
            self.assertNotIn(f"## Message {idx} [{role}]", result)
        for idx in range(dropped + 1, 21):
            role = "user" if idx % 2 else "assistant"
            self.assertIn(f"## Message {idx} [{role}]", result)

    def test_drops_whole_messages_only(self) -> None:
        """按整条丢：结果里不会出现半个 `## Message N` 头，且保留块逐字节不变。"""
        original = _history_text(20, 750)
        result, _original_chars, _kept_chars, dropped = _truncate_history_blob(original, self.budget)

        bodies = _block_bodies(result)  # 内含「半个分块头」断言
        self.assertEqual(len(bodies), 20 - dropped)
        # 保留部分必须是原文的**后缀**（不改内容、不重编号）。
        tail = result.split("\n", 1)[1]
        self.assertTrue(original.endswith(tail), "kept blocks must be a byte-identical suffix")
        # 每条保留消息的正文与原文完全一致。
        for header, body in bodies.items():
            self.assertIn(f"{header}\n{body}", original)

    def test_within_budget_is_byte_identical(self) -> None:
        text = _history_text(3, 100)
        result, original_chars, kept_chars, dropped = _truncate_history_blob(text, self.budget)

        self.assertEqual(result, text)
        self.assertEqual(original_chars, kept_chars)
        self.assertEqual(dropped, 0)
        self.assertNotIn("已省略", result)
        self.assertNotIn("已截断", result)

    def test_single_20k_message_keeps_head_with_marker(self) -> None:
        """单条消息本身超预算 → 保留**开头** + 标记，不能整条塞进去。"""
        head_marker = "HEAD_MARKER_"
        body = head_marker + "y" * 19_990
        text = f"## Message 1 [user]\n{body}\n"
        self.assertGreater(len(text), 19_000)

        result, original_chars, kept_chars, dropped = _truncate_history_blob(text, self.budget)

        self.assertEqual(original_chars, len(text))
        self.assertEqual(dropped, 0)
        self.assertLessEqual(len(result), self.budget)
        self.assertIn("## Message 1 [user]", result)  # 分块头完整
        self.assertIn(head_marker, result)  # 保留的是开头
        self.assertIn(_BLOCK_TRUNCATION_NOTE.format(original=original_chars), result)
        self.assertIn("…（已截断", result)
        self.assertLess(kept_chars, original_chars)
        # 结尾没被保留（不是保留尾部）。
        self.assertNotIn("y" * 19_000, result)

    def test_zero_or_negative_config_disables_truncation(self) -> None:
        """预算配 0 / 负数 → 视为不限制（旧行为），并打一次 WARNING。"""
        with self.assertLogs("qwen2api.context_attachment_manager", level="WARNING") as logs:
            budget = _inline_char_budget(_settings(ATTACHMENT_INLINE_TEXT_MAX_CHARS=0))
        self.assertIsNone(budget)
        self.assertTrue(any("ATTACHMENT_INLINE_TEXT_MAX_CHARS" in record.getMessage() for record in logs.records))

        self.assertIsNone(_inline_char_budget(_settings(ATTACHMENT_INLINE_TEXT_MAX_CHARS=-1)))

        text = _history_text(20, 750)
        result, original_chars, kept_chars, dropped = _truncate_history_blob(text, None)
        self.assertEqual(result, text)
        self.assertEqual(kept_chars, original_chars)
        self.assertEqual(dropped, 0)

    def test_remaining_budget_math(self) -> None:
        self.assertIsNone(_remaining_budget(None, 123))
        self.assertEqual(_remaining_budget(8000, 0), 8000)
        self.assertEqual(_remaining_budget(8000, 1000), 7000)
        self.assertEqual(_remaining_budget(8000, 9000), 0)


class InlineTextBlockBudgetTests(unittest.TestCase):
    """单文件注入块共享同一预算：保留开头 + 结尾标注。"""

    def test_blocks_share_budget_and_keep_head(self) -> None:
        first = "[附件: a.txt]\n" + "A" * 6000
        second = "[附件: b.txt]\n" + "B" * 6000
        kept, truncated = _truncate_inline_text_blocks([first, second], 8000)

        self.assertEqual(truncated, 2)
        self.assertLessEqual(sum(len(block) for block in kept) + 2, 8000)
        self.assertIn("[附件: a.txt]", kept[0])
        self.assertIn("[附件: b.txt]", kept[1])
        self.assertIn(_FILE_TRUNCATION_NOTE.format(original=len(first)), kept[0])
        self.assertIn(_FILE_TRUNCATION_NOTE.format(original=len(second)), kept[1])
        # 两个附件都还能看到开头，而不是被第一个吃光。
        self.assertTrue(kept[0].startswith("[附件: a.txt]\nAAAA"))
        self.assertTrue(kept[1].startswith("[附件: b.txt]\nBBBB"))

    def test_blocks_within_budget_are_untouched(self) -> None:
        blocks = ["[附件: a.txt]\nsmall", "[附件: b.txt]\nalso small"]
        kept, truncated = _truncate_inline_text_blocks(blocks, 8000)
        self.assertEqual(kept, blocks)
        self.assertEqual(truncated, 0)

    def test_inject_respects_budget_kwarg(self) -> None:
        payload = {"messages": [{"role": "user", "content": "do the thing"}]}
        blocks = ["[附件: a.txt]\n" + "A" * 5000]
        result = _inject_inline_text_blocks(payload, blocks, budget=1000)

        content = result["messages"][-1]["content"]
        self.assertTrue(content.startswith("do the thing"))
        self.assertIn("已截断", content)
        self.assertLess(content.count("A"), 1100)

    def test_inject_without_budget_is_unchanged(self) -> None:
        payload = {"messages": [{"role": "user", "content": "do the thing"}]}
        blocks = ["[附件: a.txt]\n" + "A" * 5000]
        result = _inject_inline_text_blocks(payload, blocks)
        self.assertEqual(result["messages"][-1]["content"], f"do the thing\n\n{blocks[0]}")


class _NeverUsedUploader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.exit_pool = None

    async def upload_local_file(self, _acc, local_meta):
        self.calls.append(local_meta["filename"])
        raise AssertionError("upload must not be attempted while parse breaker is open")


def _build_app(settings):
    account = SimpleNamespace(email="bot@example.com", token="tok")
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
            file_store=SimpleNamespace(
                save_text=AsyncMock(return_value={
                    "id": "ctx",
                    "path": "/tmp/ctx.txt",
                    "filename": "ctx.txt",
                    "content_type": "text/plain",
                    "sha256": "sha-ctx",
                    "created_at": 1,
                }),
                delete_path=AsyncMock(),
            ),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=_NeverUsedUploader(),
            browser_upload_proxy=None,
        )
    )


class InlineFallbackBudgetIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """端到端：15k 历史走兜底内联 → 截断到预算内 + 日志带 original/kept/dropped。"""

    LATEST = "LATEST_TASK_MARKER_do_not_drop_me_please"

    def setUp(self) -> None:
        reset_document_parse_breaker()

    def tearDown(self) -> None:
        reset_document_parse_breaker()

    def _payload(self) -> dict:
        messages: list[dict] = []
        for idx in range(1, 21):
            role = "user" if idx % 2 else "assistant"
            prefix = f"MSG{idx:02d}"
            messages.append({"role": role, "content": prefix + "x" * 745})
        messages.append({"role": "user", "content": self.LATEST})
        return {
            "model": "gpt-4.1",
            "messages": messages,
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {}}}],
        }

    async def test_15k_history_is_truncated_to_budget_keeping_tail(self) -> None:
        settings = _settings(CONTEXT_INLINE_MAX_CHARS=200, CONTEXT_FORCE_FILE_MAX_CHARS=120)
        get_document_parse_breaker(settings).record_failure("waf", immediate=True)
        app = _build_app(settings)
        payload = self._payload()

        with self.assertLogs("qwen2api.context_attachment_manager", level="INFO") as logs:
            result = await prepare_context_attachments(
                app=app,
                payload=payload,
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:inline-budget",
            )

        self.assertTrue(result["attachment_fallback"])
        joined = "\n".join(str(message.get("content", "")) for message in result["payload"]["messages"])

        # 尾部保留、头部丢弃。
        self.assertNotIn("## Message 1 [user]", joined)
        self.assertIn("MSG20" + "x" * 745, joined)
        self.assertNotIn("MSG01" + "x" * 745, joined)
        # 最新用户消息 100% 保留（它是真正的问题）。
        self.assertIn(self.LATEST, joined)
        # 没有半个分块头。
        for line in joined.splitlines():
            if line.startswith("## Message"):
                self.assertTrue(BLOCK_HEADER_RE.match(line), line)

        truncated_lines = [
            record.getMessage()
            for record in logs.records
            if "inline fallback truncated" in record.getMessage()
        ]
        self.assertEqual(len(truncated_lines), 1, truncated_lines)
        line = truncated_lines[0]
        self.assertIn("inline fallback truncated original=", line)
        original = int(line.split("original=")[1].split()[0])
        kept = int(line.split("kept=")[1].split()[0])
        dropped = int(line.split("dropped_messages=")[1].split()[0])
        self.assertGreaterEqual(original, 15_000)
        self.assertLessEqual(kept, 8000)
        self.assertGreater(dropped, 0)
        self.assertEqual(original - kept > 0, True)

        fallback_lines = [
            record.getMessage() for record in logs.records if "inline fallback chars=" in record.getMessage()
        ]
        self.assertEqual(len(fallback_lines), 1, fallback_lines)
        self.assertIn("truncated=true", fallback_lines[0])

    async def test_history_within_budget_is_not_truncated_and_has_no_marker(self) -> None:
        """预算内（8.5k 历史 / 20k 预算）→ 不加标记、内容原样。"""
        settings = _settings(ATTACHMENT_INLINE_TEXT_MAX_CHARS=20000)
        get_document_parse_breaker(settings).record_failure("waf", immediate=True)
        app = _build_app(settings)
        messages: list[dict] = []
        for idx in range(1, 10):
            role = "user" if idx % 2 else "assistant"
            body = f"H{idx:02d}" + "h" * 997
            messages.append({"role": role, "content": body})
        messages.append({"role": "user", "content": self.LATEST})
        payload = {
            "model": "gpt-4.1",
            "messages": messages,
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {}}}],
        }

        with self.assertLogs("qwen2api.context_attachment_manager", level="INFO") as logs:
            result = await prepare_context_attachments(
                app=app,
                payload=payload,
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:inline-budget-short",
            )

        joined = "\n".join(str(message.get("content", "")) for message in result["payload"]["messages"])
        self.assertIn("## Message 1 [user]\nH01" + "h" * 997, joined)
        self.assertIn(self.LATEST, joined)
        self.assertNotIn("已省略", joined)
        self.assertNotIn("已截断", joined)
        messages_logged = [record.getMessage() for record in logs.records]
        self.assertFalse(any("inline fallback truncated" in line for line in messages_logged), messages_logged)
        fallback_lines = [line for line in messages_logged if "inline fallback chars=" in line]
        self.assertEqual(len(fallback_lines), 1, fallback_lines)
        self.assertIn("truncated=false", fallback_lines[0])


if __name__ == "__main__":
    unittest.main()
