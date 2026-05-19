import unittest
from unittest.mock import patch

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import RuntimeAttemptState, collect_completion_run, extract_blocked_tool_names, parse_tool_directive_once
from backend.toolcore.request_normalizer import normalize_chat_request


class ToolCoreSingleTrackNamingTests(unittest.TestCase):
    def test_blocked_tool_names_preserve_raw_upstream_name(self) -> None:
        blocked = extract_blocked_tool_names("Tool exec does not exists.", ["Bash"])

        self.assertEqual(blocked, ["exec"])

    def test_textual_tool_call_must_match_declared_name_exactly(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Bash", "parameters": {}}],
            tool_names=["Bash"],
            tool_enabled=True,
        )

        directive = parse_tool_directive_once(
            request,
            RuntimeAttemptState(
                answer_text='##TOOL_CALL##\n{"name": "exec", "input": {"command": "echo hi"}}\n##END_CALL##'
            ),
        )

        self.assertEqual(directive.stop_reason, "end_turn")
        self.assertFalse(any(block.get("type") == "tool_use" for block in directive.tool_blocks))

    def test_history_tool_calls_drop_undeclared_alias_names(self) -> None:
        request = normalize_chat_request(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "exec",
                                    "arguments": '{"command": "echo hi"}',
                                },
                            }
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "Bash", "parameters": {}},
                    }
                ],
            }
        )

        self.assertEqual(request.tool_calls, [])


class BlockedToolGuardStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_text_and_dsml_tool_call_strips_visible_text(self) -> None:
        class FakeClient:
            async def chat_stream_events_with_retry(self, *args, **kwargs):
                yield {"type": "meta", "chat_id": "chat-1", "acc": None}
                yield {
                    "type": "event",
                    "event": {
                        "type": "delta",
                        "phase": "answer",
                        "content": (
                            "检查 gateway.platforms.api_server 进程。\n"
                            "<|DSML|tool_calls>\n"
                            "<|DSML|invoke name=\"bridge-22\">\n"
                            "<|DSML|parameter name=\"command\"><![CDATA[pgrep -f \"gateway.platforms.api_server\"]]></|DSML|parameter>\n"
                            "</|DSML|invoke>\n"
                            "</|DSML|tool_calls>"
                        ),
                    },
                }

        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "bridge-22", "parameters": {}}],
            tool_names=["bridge-22"],
            tool_enabled=True,
        )

        result = await collect_completion_run(FakeClient(), request, "prompt")

        self.assertEqual(result.state.finish_reason, "tool_calls")
        self.assertEqual(result.state.answer_text, "")
        self.assertEqual(result.state.tool_calls[0]["name"], "bridge-22")
        self.assertEqual(result.state.tool_calls[0]["input"], {"command": 'pgrep -f "gateway.platforms.api_server"'})

    async def test_plain_chunks_do_not_trigger_periodic_full_answer_blocked_scan(self) -> None:
        class FakeClient:
            async def chat_stream_events_with_retry(self, *args, **kwargs):
                yield {"type": "meta", "chat_id": "chat-1", "acc": None}
                for _ in range(9):
                    yield {"type": "event", "event": {"type": "delta", "phase": "answer", "content": "plain output "}}

        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Read", "parameters": {}}],
            tool_names=["Read"],
            tool_enabled=True,
        )

        with patch("backend.runtime.execution.extract_blocked_tool_names", wraps=extract_blocked_tool_names) as wrapped:
            result = await collect_completion_run(FakeClient(), request, "prompt")

        self.assertEqual(result.state.finish_reason, "stop")
        self.assertEqual(wrapped.call_count, 1)


if __name__ == "__main__":
    unittest.main()
