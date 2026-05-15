import json
import types
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.responses import StreamingResponse

from backend.adapter.standard_request import StandardRequest
from backend.api import v1_chat
from backend.runtime.execution import normalize_streamed_tool_calls
from backend.services.openai_stream_translator import OpenAIStreamTranslator


class _DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb


class _DummyLocks:
    def hold(self, session_key):
        del session_key
        return _DummyLock()


class _FakeTranslator:
    def __init__(self, **kwargs):
        del kwargs
        self.pending_chunks: list[str] = []

    def on_delta(self, evt, text_chunk, tool_calls):
        del evt, tool_calls
        if text_chunk:
            self.pending_chunks.append(f"data: {text_chunk}\n\n")

    def finalize(self, finish_reason, *, usage=None):
        return [f"data: FINAL-{finish_reason}-{usage['total_tokens'] if usage else 0}\n\n"]


class _FakeRequest:
    def __init__(self, app, payload):
        self.app = app
        self._payload = payload
        self.headers = {}
        self.client = types.SimpleNamespace(host="127.0.0.1")

    async def json(self):
        return self._payload


class V1ChatStreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        guard_patch = patch.object(v1_chat, "_repeated_tool_request_guard", v1_chat._RepeatedToolRequestGuard(now=lambda: 100.0))
        guard_patch.start()
        self.addCleanup(guard_patch.stop)

    async def test_repeated_user_only_request_short_circuits_before_context_upload(self) -> None:
        app = types.SimpleNamespace(
            state=types.SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=_DummyLocks(),
                account_pool=types.SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        payload = {"messages": [{"role": "user", "content": "check process"}], "stream": True}
        request = _FakeRequest(app, payload)
        standard_request = StandardRequest(
            prompt="raw prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            stream=True,
            client_profile="openclaw_openai",
            tool_names=["exec"],
            tools=[{"name": "exec", "parameters": {}}],
            tool_enabled=True,
        )
        guard = v1_chat._RepeatedToolRequestGuard(now=lambda: 100.0)
        diagnostics = v1_chat._build_openai_request_diagnostics(payload, prompt="raw prompt")
        guard.record_tool_response(
            session_key="session",
            prompt_hash=diagnostics["prompt_hash"],
            latest_user_hash=diagnostics["latest_user_hash"],
            tool_names=["exec"],
        )
        prepare_context_attachments = AsyncMock()

        with patch.object(v1_chat, "resolve_auth_context", AsyncMock(return_value=types.SimpleNamespace(token="tok"))), \
             patch.object(v1_chat, "derive_session_key", return_value="session"), \
             patch.object(v1_chat, "_build_standard_request", return_value=standard_request), \
             patch.object(v1_chat, "_repeated_tool_request_guard", guard), \
             patch.object(v1_chat, "prepare_context_attachments", prepare_context_attachments), \
             patch.object(v1_chat, "run_retryable_completion_bridge", AsyncMock()):
            response = await v1_chat.chat_completions(request)
            self.assertIsInstance(response, StreamingResponse)
            chunks = [chunk async for chunk in response.body_iterator]

        prepare_context_attachments.assert_not_awaited()
        self.assertTrue(chunks)

    async def test_streaming_response_yields_delta_before_finalize(self) -> None:
        app = types.SimpleNamespace(
            state=types.SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=_DummyLocks(),
                account_pool=types.SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        request = _FakeRequest(app, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        standard_request = StandardRequest(
            prompt="hi",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            stream=True,
            client_profile="openclaw_openai",
            tool_names=[],
            tools=[],
        )

        async def fake_bridge(**kwargs):
            on_delta = kwargs["on_delta"]
            await on_delta({"phase": "answer"}, "chunk-1", None)
            return types.SimpleNamespace(
                execution=types.SimpleNamespace(chat_id="chat_1", state=types.SimpleNamespace(finish_reason="stop", answer_text="")),
                directive=None,
                usage={"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
            )

        chunks = []
        with patch.object(v1_chat, "resolve_auth_context", AsyncMock(return_value=types.SimpleNamespace(token="tok"))), \
             patch.object(v1_chat, "derive_session_key", return_value="session"), \
             patch.object(v1_chat, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(v1_chat, "_build_standard_request", return_value=standard_request), \
             patch.object(v1_chat, "plan_persistent_session_turn", AsyncMock(return_value=types.SimpleNamespace(enabled=False))), \
             patch.object(v1_chat, "OpenAIStreamTranslator", _FakeTranslator), \
             patch.object(v1_chat, "run_retryable_completion_bridge", new=fake_bridge), \
             patch.object(v1_chat, "build_tool_directive", return_value=types.SimpleNamespace(stop_reason="end_turn", tool_blocks=[])), \
             patch.object(v1_chat, "build_openai_assistant_history_message", return_value={"role": "assistant", "content": "done"}), \
             patch.object(v1_chat, "persist_session_turn", AsyncMock()), \
             patch.object(v1_chat, "clear_invalidated_session_chat", AsyncMock()), \
             patch.object(v1_chat, "update_request_context"):
            response = await v1_chat.chat_completions(request)
            self.assertIsInstance(response, StreamingResponse)
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        self.assertEqual(chunks[0], "data: chunk-1\n\n")
        self.assertEqual(chunks[-1], "data: FINAL-stop-7\n\n")

    async def test_final_tool_call_discards_staged_content_chunks(self) -> None:
        app = types.SimpleNamespace(
            state=types.SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=_DummyLocks(),
                account_pool=types.SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        request = _FakeRequest(app, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        standard_request = StandardRequest(
            prompt="hi",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            stream=True,
            client_profile="openclaw_openai",
            tool_names=["image_generate"],
            tools=[{"name": "image_generate", "parameters": {}}],
            tool_enabled=True,
        )
        directive = types.SimpleNamespace(
            stop_reason="tool_use",
            tool_blocks=[{"type": "tool_use", "id": "call_1", "name": "image_generate", "input": {"prompt": "cat"}}],
        )

        async def fake_bridge(**kwargs):
            on_attempt_start = kwargs["on_attempt_start"]
            on_delta = kwargs["on_delta"]
            await on_attempt_start(0, "prompt")
            await on_delta({"phase": "answer"}, "temporary answer", None)
            return types.SimpleNamespace(
                execution=types.SimpleNamespace(chat_id="chat_1", state=types.SimpleNamespace(finish_reason="stop", answer_text="", reasoning_text="", tool_calls=[])),
                directive=directive,
            )

        chunks = []
        with patch.object(v1_chat, "resolve_auth_context", AsyncMock(return_value=types.SimpleNamespace(token="tok"))), \
             patch.object(v1_chat, "derive_session_key", return_value="session"), \
             patch.object(v1_chat, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(v1_chat, "_build_standard_request", return_value=standard_request), \
             patch.object(v1_chat, "plan_persistent_session_turn", AsyncMock(return_value=types.SimpleNamespace(enabled=False))), \
             patch.object(v1_chat, "run_retryable_completion_bridge", new=fake_bridge), \
             patch.object(v1_chat, "build_tool_directive", return_value=directive), \
             patch.object(v1_chat, "build_openai_assistant_history_message", return_value={"role": "assistant", "content": None, "tool_calls": []}), \
             patch.object(v1_chat, "persist_session_turn", AsyncMock()), \
             patch.object(v1_chat, "clear_invalidated_session_chat", AsyncMock()), \
             patch.object(v1_chat, "update_request_context"):
            response = await v1_chat.chat_completions(request)
            self.assertIsInstance(response, StreamingResponse)
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        payloads = [
            json.loads(chunk[6:].strip())
            for chunk in chunks
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"
        ]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload.get("choices") and payload["choices"][0]["delta"].get("content")
        )
        joined = "".join(chunks)
        self.assertEqual(content_text, "")
        self.assertIn('"tool_calls"', joined)
        self.assertIn('"finish_reason": "tool_calls"', joined)

    async def test_streaming_response_does_not_leak_cross_chunk_tool_prefix(self) -> None:
        app = types.SimpleNamespace(
            state=types.SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=_DummyLocks(),
                account_pool=types.SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        request = _FakeRequest(app, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        standard_request = StandardRequest(
            prompt="hi",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            stream=True,
            client_profile="openclaw_openai",
            tool_names=["Read"],
            tools=[{"name": "Read", "parameters": {}}],
            tool_enabled=True,
        )
        directive = types.SimpleNamespace(
            stop_reason="tool_use",
            tool_blocks=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
        )

        async def fake_bridge(**kwargs):
            on_attempt_start = kwargs["on_attempt_start"]
            on_delta = kwargs["on_delta"]
            await on_attempt_start(0, "prompt")
            await on_delta({"phase": "answer"}, "##TOOL_C", None)
            await on_delta({"phase": "answer"}, 'ALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##', None)
            return types.SimpleNamespace(
                execution=types.SimpleNamespace(chat_id="chat_1", state=types.SimpleNamespace(finish_reason="stop", answer_text="")),
                directive=directive,
            )

        chunks = []
        with patch.object(v1_chat, "resolve_auth_context", AsyncMock(return_value=types.SimpleNamespace(token="tok"))), \
             patch.object(v1_chat, "derive_session_key", return_value="session"), \
             patch.object(v1_chat, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(v1_chat, "_build_standard_request", return_value=standard_request), \
             patch.object(v1_chat, "plan_persistent_session_turn", AsyncMock(return_value=types.SimpleNamespace(enabled=False))), \
             patch.object(v1_chat, "run_retryable_completion_bridge", new=fake_bridge), \
             patch.object(v1_chat, "build_tool_directive", return_value=directive), \
             patch.object(v1_chat, "build_openai_assistant_history_message", return_value={"role": "assistant", "content": None, "tool_calls": []}), \
             patch.object(v1_chat, "persist_session_turn", AsyncMock()), \
             patch.object(v1_chat, "clear_invalidated_session_chat", AsyncMock()), \
             patch.object(v1_chat, "update_request_context"):
            response = await v1_chat.chat_completions(request)
            self.assertIsInstance(response, StreamingResponse)
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        joined = "".join(chunks)
        self.assertNotIn("##TOOL_C", joined)
        self.assertIn('"tool_calls"', joined)

    async def test_streaming_tool_call_alias_is_normalized_before_emission(self) -> None:
        app = types.SimpleNamespace(
            state=types.SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=_DummyLocks(),
                account_pool=types.SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        payload = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_prev",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{\"command\": \"echo hi\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_prev", "name": "exec", "content": "tool output ok"},
                {"role": "user", "content": "continue"},
            ],
            "stream": True,
        }
        request = _FakeRequest(app, payload)
        standard_request = StandardRequest(
            prompt="hi",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            stream=True,
            client_profile="openclaw_openai",
            tool_names=["Bash"],
            tools=[{"name": "Bash", "parameters": {}}],
            tool_enabled=True,
        )
        directive = types.SimpleNamespace(stop_reason="tool_use", tool_blocks=[{"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "echo hi"}}])

        async def fake_bridge(**kwargs):
            on_attempt_start = kwargs["on_attempt_start"]
            on_delta = kwargs["on_delta"]
            await on_attempt_start(0, "prompt")
            normalized_calls = normalize_streamed_tool_calls(
                [{"id": "call_1", "name": "exec", "input": {"command": "echo hi"}}],
                standard_request.tool_names,
            )
            await on_delta({"phase": "tool_call"}, None, normalized_calls)
            return types.SimpleNamespace(
                execution=types.SimpleNamespace(chat_id="chat_1", state=types.SimpleNamespace(finish_reason="tool_calls", answer_text="", reasoning_text="", tool_calls=[{"id": "call_1", "name": "exec", "input": {"command": "echo hi"}}])),
                directive=directive,
            )

        chunks = []
        with self.assertLogs("qwen2api.chat", level="INFO") as captured_logs, \
             patch.object(v1_chat, "resolve_auth_context", AsyncMock(return_value=types.SimpleNamespace(token="tok"))), \
             patch.object(v1_chat, "derive_session_key", return_value="session"), \
             patch.object(v1_chat, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(v1_chat, "_build_standard_request", return_value=standard_request), \
             patch.object(v1_chat, "plan_persistent_session_turn", AsyncMock(return_value=types.SimpleNamespace(enabled=False))), \
             patch.object(v1_chat, "OpenAIStreamTranslator", OpenAIStreamTranslator), \
             patch.object(v1_chat, "run_retryable_completion_bridge", new=fake_bridge), \
             patch.object(v1_chat, "build_tool_directive", return_value=directive), \
             patch.object(v1_chat, "build_openai_assistant_history_message", return_value={"role": "assistant", "content": None, "tool_calls": []}), \
             patch.object(v1_chat, "persist_session_turn", AsyncMock()), \
             patch.object(v1_chat, "clear_invalidated_session_chat", AsyncMock()), \
             patch.object(v1_chat, "update_request_context"):
            response = await v1_chat.chat_completions(request)
            self.assertIsInstance(response, StreamingResponse)
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        joined = "".join(chunks)
        log_output = "\n".join(captured_logs.output)
        self.assertIn('"name": "exec"', joined)
        self.assertNotIn('"name": "Bash"', joined)
        self.assertIn("[OAI] stream_sse_chunk", log_output)
        self.assertIn("has_tool_calls=True", log_output)
        self.assertIn("finish_reason=tool_calls", log_output)
        self.assertIn("tool_details=", log_output)
        self.assertIn("'index': 0", log_output)
        self.assertIn("'id': 'call_1'", log_output)
        self.assertIn("'type': 'function'", log_output)
        self.assertIn("'name': 'exec'", log_output)
        self.assertIn("'arguments_chars': 22", log_output)
        self.assertIn("[OAI] tool_name_map", log_output)
        self.assertIn("[OAI] stream_request_options", log_output)
        self.assertIn("include_usage_requested=False", log_output)
        self.assertIn("[OAI] stream_finalize_options", log_output)
        self.assertIn("usage_will_be_sent=True", log_output)
        self.assertIn("[OAI] inbound_tool_result", log_output)
        self.assertIn("tool_call_id=call_prev", log_output)
        self.assertIn("name=exec", log_output)
        self.assertIn("content_chars=14", log_output)
        self.assertIn("has_error_signal=False", log_output)
        self.assertIn("content_preview='tool output ok'", log_output)
        self.assertIn("[OAI] outbound_tool_call", log_output)
        self.assertIn("client_name=exec", log_output)
        self.assertIn("arguments_json_valid=True", log_output)
        self.assertIn("input_keys=['command']", log_output)

    async def test_streaming_retry_does_not_leak_failed_attempt_text(self) -> None:
        app = types.SimpleNamespace(
            state=types.SimpleNamespace(
                users_db=object(),
                qwen_client=object(),
                file_store=None,
                session_locks=_DummyLocks(),
                account_pool=types.SimpleNamespace(acquire_wait_preferred=AsyncMock(return_value=None)),
            )
        )
        request = _FakeRequest(app, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        standard_request = StandardRequest(
            prompt="hi",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            stream=True,
            client_profile="openclaw_openai",
            tool_names=["exec"],
            tools=[{"name": "exec", "parameters": {}}],
            tool_enabled=True,
        )
        directive = types.SimpleNamespace(stop_reason="tool_use", tool_blocks=[{"type": "tool_use", "id": "call_1", "name": "exec", "input": {"command": "echo hi"}}])

        async def fake_bridge(**kwargs):
            on_attempt_start = kwargs["on_attempt_start"]
            on_delta = kwargs["on_delta"]
            on_retry = kwargs["on_retry"]

            await on_attempt_start(0, "prompt")
            await on_delta({"phase": "answer"}, "Tool exec does not exists.", None)
            await on_retry(0, types.SimpleNamespace(reason="blocked_tool_name:exec"), types.SimpleNamespace())

            await on_attempt_start(1, "prompt")
            await on_delta({"phase": "tool_call"}, None, [{"id": "call_1", "name": "exec", "input": {"command": "echo hi"}}])
            return types.SimpleNamespace(
                execution=types.SimpleNamespace(chat_id="chat_1", state=types.SimpleNamespace(finish_reason="tool_calls", answer_text="", reasoning_text="", tool_calls=[{"id": "call_1", "name": "exec", "input": {"command": "echo hi"}}])),
                directive=directive,
            )

        chunks = []
        with patch.object(v1_chat, "resolve_auth_context", AsyncMock(return_value=types.SimpleNamespace(token="tok"))), \
             patch.object(v1_chat, "derive_session_key", return_value="session"), \
             patch.object(v1_chat, "prepare_context_attachments", AsyncMock(return_value={"payload": request._payload, "upstream_files": [], "session_key": "session", "context_mode": "inline", "bound_account_email": None, "bound_account": None})), \
             patch.object(v1_chat, "_build_standard_request", return_value=standard_request), \
             patch.object(v1_chat, "plan_persistent_session_turn", AsyncMock(return_value=types.SimpleNamespace(enabled=False))), \
             patch.object(v1_chat, "run_retryable_completion_bridge", new=fake_bridge), \
             patch.object(v1_chat, "build_tool_directive", return_value=directive), \
             patch.object(v1_chat, "build_openai_assistant_history_message", return_value={"role": "assistant", "content": None, "tool_calls": []}), \
             patch.object(v1_chat, "persist_session_turn", AsyncMock()), \
             patch.object(v1_chat, "clear_invalidated_session_chat", AsyncMock()), \
             patch.object(v1_chat, "update_request_context"):
            response = await v1_chat.chat_completions(request)
            self.assertIsInstance(response, StreamingResponse)
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        joined = "".join(chunks)
        self.assertNotIn("Tool exec does not exists.", joined)
        self.assertIn('"tool_calls"', joined)


if __name__ == "__main__":
    unittest.main()
