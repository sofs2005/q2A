import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.api import anthropic
from backend.services.response_formatters import build_anthropic_message_payload
from backend.services.token_calc import count_tokens


class AnthropicToolCoreIntegrationTests(unittest.TestCase):
    def test_build_standard_request_preserves_tool_choice_fields(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read file",
                        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            }
        )

        self.assertEqual(request.tool_choice_mode, "required")
        self.assertEqual(request.required_tool_name, "bridge-0")
        self.assertEqual(request.tool_choice_raw, {"type": "function", "function": {"name": "bridge-0"}})

    def test_build_standard_request_rejects_undeclared_forced_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            anthropic._build_standard_request(
                {
                    "model": "claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "read the file"}],
                    "tools": [{"name": "Read", "description": "Read file", "input_schema": {}}],
                    "tool_choice": {"type": "function", "function": {"name": "WebSearch"}},
                }
            )

    def test_build_standard_request_preserves_top_level_system_prompt(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "system": "Always answer as a pirate captain.",
                "messages": [{"role": "user", "content": "Who are you?"}],
            }
        )

        self.assertIn("Always answer as a pirate captain.", request.prompt)
        self.assertIn("<system>\n", request.prompt)

    def test_build_standard_request_preserves_top_level_developer_and_instructions(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "developer": "Always answer as a pirate captain.",
                "instructions": "Never claim to be a robot.",
                "messages": [{"role": "user", "content": "Who are you?"}],
            }
        )

        self.assertIn("Always answer as a pirate captain.", request.prompt)
        self.assertIn("Never claim to be a robot.", request.prompt)

    def test_build_standard_request_normalizes_anthropic_tools(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read file",
                        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    }
                ],
            }
        )

        self.assertEqual(request.tool_names, ["bridge-0"])
        self.assertIn("Bridge-call slots available: bridge-0", request.prompt)
        self.assertNotIn("Bridge-call slots available: Read", request.prompt)
        self.assertEqual(request.tools[0]["parameters"], {"type": "object", "properties": {"file_path": {"type": "string"}}})

    def test_anthropic_stream_usage_uses_token_counts(self) -> None:
        prompt = "hello world hello world hello world"
        answer_text = "I will keep this concise."

        usage = anthropic._anthropic_usage(prompt, answer_text)

        self.assertEqual(usage["input_tokens"], count_tokens(prompt))
        self.assertEqual(usage["output_tokens"], count_tokens(answer_text))
        self.assertNotEqual(usage["input_tokens"], len(prompt))

    def test_visible_answer_tokens_survive_stream_buffer_flush(self) -> None:
        answer_text = "I will keep this concise."
        stream_state = anthropic._AnthropicStreamState(msg_id="msg_1", model_name="claude-3-5-sonnet", prompt="prompt")
        stream_state.buffer_answer_text(answer_text)
        stream_state.flush_answer_text()
        execution = SimpleNamespace(state=SimpleNamespace(answer_text=answer_text))

        output_tokens = anthropic._visible_answer_text_length(
            directive=SimpleNamespace(stop_reason="end_turn"),
            execution=execution,
            stream_state=stream_state,
        )

        self.assertEqual(output_tokens, count_tokens(answer_text))

    def test_anthropic_message_payload_formatter_matches_tool_directive(self) -> None:
        request = anthropic._build_standard_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {}}],
            }
        )
        execution = SimpleNamespace(state=SimpleNamespace(answer_text="", reasoning_text="", tool_calls=[{"id": "call_123", "name": "bridge-0", "input": {"file_path": "README.md"}}]))

        payload = build_anthropic_message_payload(
            msg_id="msg_123",
            model_name="claude-3-5-sonnet",
            prompt="prompt",
            execution=execution,
            standard_request=request,
        )

        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0]["type"], "tool_use")
        self.assertEqual(payload["content"][0]["name"], "Read")


class AnthropicBridgeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_stream_path_uses_retryable_completion_bridge(self) -> None:
        app = SimpleNamespace(
            state=SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=SimpleNamespace(hold=lambda _key: _DummyLock()),
                account_pool=SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        request = _FakeRequest(
            app,
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {}}],
            },
        )
        standard_request = anthropic._build_standard_request(request._payload)
        bridge_result = SimpleNamespace(
            execution=SimpleNamespace(state=SimpleNamespace(answer_text="", reasoning_text="", tool_calls=[{"id": "call_123", "name": "Read", "input": {"file_path": "README.md"}}]), acc=None, chat_id=None),
            prompt="prompt",
            directive=None,
        )

        with patch.object(anthropic, "resolve_auth_context", AsyncMock(return_value=SimpleNamespace(token="tok"))), \
             patch.object(anthropic, "build_request_session_key", return_value="session"), \
             patch.object(anthropic, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(anthropic, "preprocess_attachments", AsyncMock(side_effect=lambda payload, *_args, **_kwargs: SimpleNamespace(payload=payload, attachments=[], uploaded_file_ids=[]))), \
             patch.object(anthropic, "run_retryable_completion_bridge", AsyncMock(return_value=bridge_result)) as bridge_mock, \
             patch.object(anthropic, "update_request_context"):
            response = await anthropic.anthropic_messages(request)

        self.assertEqual(response.status_code, 200)
        bridge_mock.assert_awaited_once()


class _DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb


class _FakeRequest:
    def __init__(self, app, payload):
        self.app = app
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
