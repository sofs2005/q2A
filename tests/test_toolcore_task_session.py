import json
import unittest
from types import SimpleNamespace

from backend.adapter.standard_request import StandardRequest
from backend.toolcore.task_session import build_openai_assistant_history_message
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcore.types import ToolDefinition


class ToolCoreTaskSessionTests(unittest.TestCase):
    def test_standard_request_has_no_persistent_session_fields(self) -> None:
        request = StandardRequest(
            prompt="hello",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )
        for attr in (
            "persistent_session",
            "full_prompt",
            "upstream_chat_id",
            "session_message_hashes",
            "session_chat_invalidated",
        ):
            self.assertFalse(hasattr(request, attr))

    def test_build_openai_assistant_history_message_emits_tool_calls(self) -> None:
        execution = SimpleNamespace(state=SimpleNamespace(answer_text="ignored"))
        directive = SimpleNamespace(
            stop_reason="tool_use",
            tool_blocks=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
        )
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )

        message = build_openai_assistant_history_message(
            execution=execution,
            request=request,
            directive=directive,
        )

        self.assertEqual(message["role"], "assistant")
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "Read")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"file_path": "README.md"})

    def test_build_openai_assistant_history_message_maps_model_tool_name_to_client_name(self) -> None:
        execution = SimpleNamespace(state=SimpleNamespace(answer_text="ignored"))
        directive = SimpleNamespace(
            stop_reason="tool_use",
            tool_blocks=[{"type": "tool_use", "id": "call_1", "name": "bridge-0", "input": {}}],
        )
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tool_catalog=ToolCatalog([ToolDefinition(name="Read", client_name="Read", model_name="bridge-0")]),
        )

        message = build_openai_assistant_history_message(
            execution=execution,
            request=request,
            directive=directive,
        )

        self.assertEqual(message["tool_calls"][0]["function"]["name"], "Read")

    def test_build_openai_assistant_history_message_emits_text_for_end_turn(self) -> None:
        execution = SimpleNamespace(state=SimpleNamespace(answer_text="done"))
        directive = SimpleNamespace(stop_reason="end_turn", tool_blocks=[])
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )

        message = build_openai_assistant_history_message(
            execution=execution,
            request=request,
            directive=directive,
        )

        self.assertEqual(message, {"role": "assistant", "content": "done"})


if __name__ == "__main__":
    unittest.main()
