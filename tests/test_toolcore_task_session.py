import unittest
from types import SimpleNamespace

from backend.adapter.standard_request import StandardRequest
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcore.types import ToolDefinition
from backend.toolcore.task_session import (
    SessionHistoryEntry,
    build_continuation_prompt,
    build_openai_assistant_history_message,
    build_retry_rebase_prompt,
    extract_session_history_entries,
    render_session_message,
)


class ToolCoreTaskSessionTests(unittest.TestCase):
    def test_render_session_message_formats_tool_result_only_user_message(self) -> None:
        message = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "done"}]}
            ],
        }

        rendered = render_session_message(message, client_profile="openclaw_openai", tools_enabled=True)

        self.assertEqual(rendered, "[Tool Result for call call_1]\ndone\n[/Tool Result]")

    def test_render_session_message_maps_assistant_tool_call_to_model_name(self) -> None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "README.md"}'}}
            ],
        }
        catalog = ToolCatalog([ToolDefinition(name="Read", client_name="Read", model_name="bridge-0")])

        rendered = render_session_message(
            message,
            client_profile="openclaw_openai",
            tools_enabled=True,
            tool_catalog=catalog,
        )

        self.assertIn('<|DSML|invoke name="bridge-0">', rendered)
        self.assertNotIn('<|DSML|invoke name="Read">', rendered)

    def test_extract_session_history_entries_is_stable(self) -> None:
        messages = [{"role": "assistant", "content": "hello"}]

        first = extract_session_history_entries(messages, client_profile="openclaw_openai", tools_enabled=False)
        second = extract_session_history_entries(messages, client_profile="openclaw_openai", tools_enabled=False)

        self.assertEqual(first[0].digest, second[0].digest)
        self.assertEqual(first[0].rendered, "Assistant: hello")

    def test_build_continuation_prompt_uses_new_entries(self) -> None:
        entries = [SessionHistoryEntry(rendered="Human: inspect file", digest="abc")]

        prompt = build_continuation_prompt(entries, tool_names=["Read"], tools=[{"name": "Read", "input_schema": {"properties": {"file_path": {"type": "string"}}}}])

        self.assertIn("=== SAME TASK SESSION CONTINUATION ===", prompt)
        self.assertIn("- Read: file_path", prompt)
        self.assertIn("Human: inspect file", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_build_retry_rebase_prompt_keeps_generic_search_wording(self) -> None:
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tool_names=["web_fetch"],
            tools=[{"name": "web_fetch", "parameters": {}}],
            tool_enabled=True,
        )

        prompt = build_retry_rebase_prompt(request, reason="search_no_results")

        self.assertIn("last search tool returned no results", prompt)
        self.assertNotIn("WebSearch", prompt)

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


if __name__ == "__main__":
    unittest.main()
