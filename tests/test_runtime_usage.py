import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.adapter.standard_request import StandardRequest
import backend.runtime.execution as runtime_execution
from backend.runtime.execution import RuntimeRetryDirective, RuntimeToolDirective, build_usage_delta_factory, collect_completion_run
from backend.services import completion_bridge
from backend.services.token_calc import count_tokens


class RuntimeUsageTests(unittest.TestCase):
    def test_usage_delta_factory_counts_tool_calls_as_completion_usage(self) -> None:
        prompt = "prompt"
        execution = SimpleNamespace(state=SimpleNamespace(
            answer_text="",
            tool_calls=[{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        ))

        usage_delta = build_usage_delta_factory(prompt)(execution)

        self.assertGreater(usage_delta, count_tokens(prompt))

    def test_usage_delta_factory_includes_context_attachment_tokens(self) -> None:
        prompt = "prompt"
        answer_text = "answer"
        execution = SimpleNamespace(state=SimpleNamespace(answer_text=answer_text))
        attachment_tokens = 17

        usage_delta = build_usage_delta_factory(prompt, extra_prompt_tokens=attachment_tokens)(execution)

        self.assertEqual(usage_delta, count_tokens(prompt) + count_tokens(answer_text) + attachment_tokens)

    def test_usage_delta_factory_uses_token_counts(self) -> None:
        prompt = "hello world hello world hello world"
        answer_text = "I will keep this concise."
        execution = SimpleNamespace(state=SimpleNamespace(answer_text=answer_text))

        usage_delta = build_usage_delta_factory(prompt)(execution)

        self.assertEqual(usage_delta, count_tokens(prompt) + count_tokens(answer_text))
        self.assertNotEqual(usage_delta, len(prompt) + len(answer_text))


class CollectCompletionRunStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_on_delta_receives_final_safe_text_tail(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Read", "parameters": {}}],
            tool_names=["Read"],
            tool_enabled=True,
        )
        answer = (
            "自定义 Skill 应放置在以下目录中：\n\n"
            "C:/Users/daife/.agents/skills\n\n"
            "需要我帮你创建一个自定义 Skill 的模板吗？"
        )
        client = _FakeStreamClient(
            [
                {"type": "meta", "chat_id": "chat_1", "acc": None},
                {"type": "event", "event": {"type": "delta", "phase": "answer", "content": answer}},
            ]
        )
        deltas: list[str] = []

        async def on_delta(_evt, text_chunk, _tool_calls):
            if text_chunk is not None:
                deltas.append(text_chunk)

        with patch.object(runtime_execution.settings, "TOOLCORE_V2_ENABLED", True):
            result = await collect_completion_run(
                client,
                request,
                request.prompt,
                capture_events=False,
                on_delta=on_delta,
            )

        self.assertEqual(result.state.answer_text, answer)
        self.assertEqual("".join(deltas), answer)

    async def test_slow_on_delta_warning_is_rate_limited(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[],
            tool_names=[],
            tool_enabled=False,
        )
        client = _FakeStreamClient(
            [{"type": "meta", "chat_id": "chat_1", "acc": None}]
            + [
                {"type": "event", "event": {"type": "delta", "phase": "answer", "content": "chunk"}}
                for _ in range(20)
            ]
        )

        async def on_delta(_evt, _text_chunk, _tool_calls):
            return None

        with patch.object(runtime_execution.settings, "DIAGNOSTIC_SLOW_STEP_SECONDS", 0.0):
            with self.assertLogs("qwen2api.runtime", level="WARNING") as captured_logs:
                await collect_completion_run(
                    client,
                    request,
                    request.prompt,
                    capture_events=False,
                    on_delta=on_delta,
                )

        slow_on_delta_logs = [line for line in captured_logs.output if "slow on_delta" in line]
        self.assertLessEqual(len(slow_on_delta_logs), 3)

    async def test_streaming_on_delta_receives_safe_text_not_raw_dsml(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Read", "parameters": {}}],
            tool_names=["Read"],
            tool_enabled=True,
        )
        client = _FakeStreamClient(
            [
                {"type": "meta", "chat_id": "chat_1", "acc": None},
                {"type": "event", "event": {"type": "delta", "phase": "answer", "content": "prefix <|DSML|tool_calls>\n"}},
                {
                    "type": "event",
                    "event": {
                        "type": "delta",
                        "phase": "answer",
                        "content": (
                            '  <|DSML|invoke name="Read">\n'
                            '    <|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>\n'
                            '  </|DSML|invoke>\n'
                            '</|DSML|tool_calls> suffix'
                        ),
                    },
                },
            ]
        )
        deltas: list[str] = []
        safe_flags: list[bool] = []

        async def on_delta(evt, text_chunk, _tool_calls):
            if text_chunk is not None:
                deltas.append(text_chunk)
                safe_flags.append(bool(evt.get("_qwen2api_safe_text")))

        with patch.object(runtime_execution.settings, "TOOLCORE_V2_ENABLED", True):
            result = await collect_completion_run(
                client,
                request,
                request.prompt,
                capture_events=False,
                on_delta=on_delta,
            )

        self.assertEqual(deltas, ["prefix ", " suffix"])
        self.assertEqual(safe_flags, [True, True])
        self.assertNotIn("<|DSML|", "".join(deltas))
        self.assertEqual(result.state.tool_calls[0]["name"], "Read")


class _FakeStreamClient:
    def __init__(self, items):
        self._items = items

    async def chat_stream_events_with_retry(self, *args, **kwargs):
        for item in self._items:
            yield item


class CompletionBridgeUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_retryable_bridge_counts_tool_calls_in_returned_usage(self) -> None:
        prompt = "prompt"
        execution = SimpleNamespace(
            state=SimpleNamespace(
                answer_text="",
                tool_calls=[{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
            ),
            acc=None,
            chat_id=None,
        )
        standard_request = StandardRequest(
            prompt=prompt,
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )

        with patch.object(completion_bridge, "collect_completion_run", AsyncMock(return_value=execution)), \
             patch.object(completion_bridge, "evaluate_retry_directive", return_value=RuntimeRetryDirective(retry=False, next_prompt="")), \
             patch.object(completion_bridge, "build_tool_directive", return_value=RuntimeToolDirective(stop_reason="tool_use")), \
             patch.object(completion_bridge, "_apply_terminal_tool_guard", return_value=(execution, RuntimeToolDirective(stop_reason="tool_use"))), \
             patch.object(completion_bridge, "add_used_tokens", AsyncMock()), \
             patch.object(completion_bridge, "cleanup_runtime_resources", AsyncMock()):
            result = await completion_bridge.run_retryable_completion_bridge(
                client=object(),
                standard_request=standard_request,
                prompt=prompt,
                users_db=object(),
                token="tok",
                history_messages=[],
                max_attempts=1,
            )

        self.assertGreater(result.usage["completion_tokens"], 0)
        self.assertEqual(result.usage["total_tokens"], result.usage["prompt_tokens"] + result.usage["completion_tokens"])

    async def test_retryable_bridge_includes_context_attachment_tokens_in_usage(self) -> None:
        prompt = "prompt"
        attachment_tokens = 23
        execution = SimpleNamespace(
            state=SimpleNamespace(answer_text="answer", tool_calls=[]),
            acc=None,
            chat_id=None,
        )
        standard_request = StandardRequest(
            prompt=prompt,
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            context_attachment_tokens=attachment_tokens,
        )
        users_db = object()

        with patch.object(completion_bridge, "collect_completion_run", AsyncMock(return_value=execution)), \
             patch.object(completion_bridge, "evaluate_retry_directive", return_value=RuntimeRetryDirective(retry=False, next_prompt="")), \
             patch.object(completion_bridge, "build_tool_directive", return_value=RuntimeToolDirective(stop_reason="end_turn")), \
             patch.object(completion_bridge, "_apply_terminal_tool_guard", return_value=(execution, RuntimeToolDirective(stop_reason="end_turn"))), \
             patch.object(completion_bridge, "add_used_tokens", AsyncMock()) as add_tokens_mock, \
             patch.object(completion_bridge, "cleanup_runtime_resources", AsyncMock()):
            result = await completion_bridge.run_retryable_completion_bridge(
                client=object(),
                standard_request=standard_request,
                prompt=prompt,
                users_db=users_db,
                token="tok",
                history_messages=[],
                max_attempts=1,
                usage_delta_factory=build_usage_delta_factory(prompt, extra_prompt_tokens=attachment_tokens),
            )

        self.assertEqual(result.usage["prompt_tokens"], count_tokens(prompt) + attachment_tokens)
        self.assertEqual(result.usage["total_tokens"], result.usage["prompt_tokens"] + result.usage["completion_tokens"])
        add_tokens_mock.assert_awaited_once_with(users_db, "tok", result.usage["total_tokens"])

    async def test_retryable_bridge_replaces_final_blocked_tool_error(self) -> None:
        prompt = "prompt"
        execution = SimpleNamespace(
            state=SimpleNamespace(
                answer_text="Tool exec does not exists.",
                reasoning_text="",
                tool_calls=[],
                blocked_tool_names=["exec"],
                finish_reason="stop",
            ),
            acc=None,
            chat_id=None,
        )
        standard_request = StandardRequest(
            prompt=prompt,
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "bridge-3", "parameters": {}}],
            tool_names=["bridge-3"],
            tool_enabled=True,
        )

        with patch.object(completion_bridge, "collect_completion_run", AsyncMock(return_value=execution)), \
             patch.object(completion_bridge, "add_used_tokens", AsyncMock()), \
             patch.object(completion_bridge, "cleanup_runtime_resources", AsyncMock()):
            result = await completion_bridge.run_retryable_completion_bridge(
                client=object(),
                standard_request=standard_request,
                prompt=prompt,
                users_db=object(),
                token="tok",
                history_messages=[],
                max_attempts=1,
            )

        self.assertNotEqual(result.execution.state.answer_text, "Tool exec does not exists.")
        self.assertIn("upstream model", result.execution.state.answer_text)
        self.assertEqual(result.execution.state.blocked_tool_names, [])
        self.assertEqual(result.execution.state.tool_calls, [])

    async def test_retryable_bridge_fails_when_attachment_account_cannot_be_reacquired(self) -> None:
        prompt = "prompt"
        execution = SimpleNamespace(
            state=SimpleNamespace(answer_text="", tool_calls=[]),
            acc=SimpleNamespace(email="acc@example.com"),
            chat_id="chat_1",
        )
        standard_request = StandardRequest(
            prompt=prompt,
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            bound_account_email="acc@example.com",
            upstream_files=[{"file_id": "file-history"}],
        )
        client = SimpleNamespace(account_pool=SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)))

        with patch.object(completion_bridge, "collect_completion_run", AsyncMock(return_value=execution)), \
             patch.object(completion_bridge, "evaluate_retry_directive", return_value=RuntimeRetryDirective(retry=True, next_prompt="retry")), \
             patch.object(completion_bridge, "cleanup_runtime_resources", AsyncMock()):
            with self.assertRaises(RuntimeError):
                await completion_bridge.run_retryable_completion_bridge(
                    client=client,
                    standard_request=standard_request,
                    prompt=prompt,
                    users_db=object(),
                    token="tok",
                    history_messages=[],
                    max_attempts=2,
                )

        client.account_pool.acquire_wait_preferred.assert_awaited_once_with("acc@example.com", timeout=60)

    async def test_retryable_bridge_releases_account_when_post_collection_step_fails(self) -> None:
        prompt = "prompt"
        execution = SimpleNamespace(
            state=SimpleNamespace(answer_text="hello", tool_calls=[]),
            acc=SimpleNamespace(email="acc@example.com"),
            chat_id="chat_1",
        )
        standard_request = StandardRequest(
            prompt=prompt,
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )

        client = object()

        with patch.object(completion_bridge, "collect_completion_run", AsyncMock(return_value=execution)), \
             patch.object(completion_bridge, "evaluate_retry_directive", return_value=RuntimeRetryDirective(retry=False, next_prompt="")), \
             patch.object(completion_bridge, "build_tool_directive", return_value=RuntimeToolDirective(stop_reason="end_turn")), \
             patch.object(completion_bridge, "_apply_terminal_tool_guard", return_value=(execution, RuntimeToolDirective(stop_reason="end_turn"))), \
             patch.object(completion_bridge, "add_used_tokens", AsyncMock(side_effect=RuntimeError("quota write failed"))), \
             patch.object(completion_bridge, "cleanup_runtime_resources", AsyncMock()) as cleanup_mock:
            with self.assertRaises(RuntimeError):
                await completion_bridge.run_retryable_completion_bridge(
                    client=client,
                    standard_request=standard_request,
                    prompt=prompt,
                    users_db=object(),
                    token="tok",
                    history_messages=[],
                    max_attempts=1,
                )

        cleanup_mock.assert_awaited_once_with(
            client,
            execution.acc,
            execution.chat_id,
            preserve_chat=False,
        )

    async def test_completion_bridge_releases_account_when_usage_write_fails(self) -> None:
        prompt = "prompt"
        execution = SimpleNamespace(
            state=SimpleNamespace(answer_text="hello", tool_calls=[]),
            acc=SimpleNamespace(email="acc@example.com"),
            chat_id="chat_1",
        )
        standard_request = StandardRequest(
            prompt=prompt,
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )
        client = object()

        with patch.object(completion_bridge, "collect_completion_run", AsyncMock(return_value=execution)), \
             patch.object(completion_bridge, "add_used_tokens", AsyncMock(side_effect=RuntimeError("quota write failed"))), \
             patch.object(completion_bridge, "cleanup_runtime_resources", AsyncMock()) as cleanup_mock:
            with self.assertRaises(RuntimeError):
                await completion_bridge.run_completion_bridge(
                    client=client,
                    standard_request=standard_request,
                    prompt=prompt,
                    users_db=object(),
                    token="tok",
                )

        cleanup_mock.assert_awaited_once_with(
            client,
            execution.acc,
            execution.chat_id,
            preserve_chat=False,
        )


if __name__ == "__main__":
    unittest.main()
