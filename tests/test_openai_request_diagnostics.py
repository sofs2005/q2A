import unittest

from backend.api.v1_chat import (
    _RepeatedToolRequestGuard,
    _build_openai_request_diagnostics,
    _build_repeated_tool_request_notice,
    _record_repeated_tool_guard,
)


class OpenAIRequestDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_detects_plain_repeated_request_without_tool_results(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "检查 gateway.platforms.api_server 进程"},
            ],
            "tools": [{"name": "execute_code"}],
        }

        diagnostics = _build_openai_request_diagnostics(req_data, prompt="same prompt")

        self.assertEqual(diagnostics["message_count"], 1)
        self.assertEqual(diagnostics["role_counts"], {"user": 1})
        self.assertFalse(diagnostics["has_assistant_tool_calls"])
        self.assertFalse(diagnostics["has_tool_results"])
        self.assertEqual(diagnostics["tool_result_count"], 0)
        self.assertEqual(diagnostics["prompt_hash"], _build_openai_request_diagnostics(req_data, prompt="same prompt")["prompt_hash"])
        self.assertEqual(len(diagnostics["latest_user_hash"]), 16)

    def test_diagnostics_detects_tool_continuation_messages(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "检查 gateway.platforms.api_server 进程"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "execute_code", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "process is running"},
            ],
        }

        diagnostics = _build_openai_request_diagnostics(req_data, prompt="prompt with tool result")

        self.assertEqual(diagnostics["message_count"], 3)
        self.assertEqual(diagnostics["role_counts"], {"assistant": 1, "tool": 1, "user": 1})
        self.assertTrue(diagnostics["has_assistant_tool_calls"])
        self.assertTrue(diagnostics["has_tool_results"])
        self.assertEqual(diagnostics["assistant_tool_call_count"], 1)
        self.assertEqual(diagnostics["tool_result_count"], 1)

    def test_repeated_tool_request_guard_detects_user_only_replay(self) -> None:
        guard = _RepeatedToolRequestGuard(ttl_seconds=60.0, now=lambda: 100.0)
        req_data = {"messages": [{"role": "user", "content": "检查 gateway 进程"}]}
        diagnostics = _build_openai_request_diagnostics(req_data, prompt="same prompt")

        guard.record_tool_response(
            session_key="session-1",
            prompt_hash=diagnostics["prompt_hash"],
            latest_user_hash=diagnostics["latest_user_hash"],
            tool_names=["execute_code"],
        )

        repeated = guard.repeated_user_only_tool_request("session-1", diagnostics)

        self.assertEqual(repeated, ["execute_code"])

    def test_repeated_tool_request_guard_detects_replay_when_prompt_hash_drifts(self) -> None:
        guard = _RepeatedToolRequestGuard(ttl_seconds=60.0, now=lambda: 100.0)
        req_data = {"messages": [{"role": "user", "content": "检查 gateway 进程"}]}
        original = _build_openai_request_diagnostics(req_data, prompt="prompt before context upload")
        replay = _build_openai_request_diagnostics(req_data, prompt="prompt after regenerated context upload")
        guard.record_tool_response(
            session_key="session-1",
            prompt_hash=original["prompt_hash"],
            latest_user_hash=original["latest_user_hash"],
            tool_names=["execute_code"],
        )

        repeated = guard.repeated_user_only_tool_request("session-1", replay)

        self.assertEqual(repeated, ["execute_code"])

    def test_record_repeated_tool_guard_stores_final_context_diagnostics(self) -> None:
        guard = _RepeatedToolRequestGuard(ttl_seconds=60.0, now=lambda: 100.0)
        req_data = {"messages": [{"role": "user", "content": "生成一张图"}]}
        early = _build_openai_request_diagnostics(req_data, prompt="early prompt")
        final = _build_openai_request_diagnostics(req_data, prompt="final prompt with context attachment")

        _record_repeated_tool_guard(
            guard=guard,
            session_key="session-1",
            diagnostics=early,
            final_diagnostics=final,
            tool_names=["image_generate"],
            finish_reason="tool_calls",
        )

        repeated = guard.repeated_user_only_tool_request("session-1", final)

        self.assertEqual(repeated, ["image_generate"])

    def test_repeated_tool_request_guard_ignores_tool_continuation(self) -> None:
        guard = _RepeatedToolRequestGuard(ttl_seconds=60.0, now=lambda: 100.0)
        original = _build_openai_request_diagnostics(
            {"messages": [{"role": "user", "content": "检查 gateway 进程"}]},
            prompt="same prompt",
        )
        continuation = _build_openai_request_diagnostics(
            {
                "messages": [
                    {"role": "user", "content": "检查 gateway 进程"},
                    {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "execute_code"}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                ]
            },
            prompt="same prompt",
        )
        guard.record_tool_response(
            session_key="session-1",
            prompt_hash=original["prompt_hash"],
            latest_user_hash=original["latest_user_hash"],
            tool_names=["execute_code"],
        )

        repeated = guard.repeated_user_only_tool_request("session-1", continuation)

        self.assertIsNone(repeated)

    def test_repeated_tool_request_notice_is_not_user_visible(self) -> None:
        notice = _build_repeated_tool_request_notice(["exec"])

        self.assertEqual(notice, "")
        self.assertNotIn("上一轮已经返回工具调用", notice)
        self.assertNotIn("请让客户端执行工具", notice)

    def test_repeated_tool_request_notice_uses_silent_when_prompt_requests_it(self) -> None:
        notice = _build_repeated_tool_request_notice(["terminal"], prompt='respond with exactly "[SILENT]"')

        self.assertEqual(notice, "[SILENT]")

    def test_repeated_tool_request_notice_uses_openclaw_no_reply_when_prompt_requests_it(self) -> None:
        notice = _build_repeated_tool_request_notice(["image_generate"], prompt="After sending with the message tool, reply with ONLY: NO_REPLY.")

        self.assertEqual(notice, "NO_REPLY")


if __name__ == "__main__":
    unittest.main()
