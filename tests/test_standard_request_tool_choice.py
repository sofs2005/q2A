import unittest

from backend.services.standard_request_builder import build_chat_standard_request
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcore.types import ToolDefinition


class StandardRequestToolChoiceTests(unittest.TestCase):
    def test_named_function_tool_choice_is_preserved(self) -> None:
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            },
            default_model="gpt-4.1",
            surface="openai",
        )

        self.assertEqual(request.tool_choice_mode, "required")
        self.assertEqual(request.required_tool_name, "bridge-0")
        self.assertEqual(request.tool_choice_raw, {"type": "function", "function": {"name": "Read"}})
        self.assertEqual(request.tool_catalog.get_client_name("Read"), "Read")

    def test_preserves_upstream_tool_catalog_when_present(self) -> None:
        upstream_catalog = ToolCatalog(
            [
                ToolDefinition(
                    name="Read",
                    description="Read file",
                    parameters={"type": "object"},
                    client_name="Read",
                    model_name="bridge-9",
                )
            ]
        )
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
                "_tool_catalog": upstream_catalog,
            },
            default_model="gpt-4.1",
            surface="gemini",
        )

        self.assertIs(request.tool_catalog, upstream_catalog)

    def test_required_tool_choice_adds_prompt_constraint(self) -> None:
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "must use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "required",
            },
            default_model="gpt-4.1",
            surface="openai",
        )

        self.assertEqual(request.tool_choice_mode, "required")
        self.assertIn("MUST include at least one tool call", request.prompt)

    def test_named_tool_choice_is_canonicalized_to_declared_tool_name(self) -> None:
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "parameters": {}}],
                "tool_choice": {"type": "function", "function": {"name": "read"}},
            },
            default_model="gpt-4.1",
            surface="openai",
        )

        self.assertEqual(request.required_tool_name, "bridge-0")

    def test_undeclared_forced_tool_choice_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            build_chat_standard_request(
                {
                    "model": "gpt-4.1",
                    "messages": [{"role": "user", "content": "read the file"}],
                    "tools": [{"name": "Read", "parameters": {}}],
                    "tool_choice": {"type": "function", "function": {"name": "WebSearch"}},
                },
                default_model="gpt-4.1",
                surface="openai",
            )

    def test_subagents_command_alias_is_not_exposed_when_agent_tools_exist(self) -> None:
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "list available subagents"}],
                "tools": [
                    {"type": "function", "function": {"name": "subagents", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "agents_list", "parameters": {"type": "object"}}},
                    {
                        "type": "function",
                        "function": {
                            "name": "sessions_spawn",
                            "parameters": {"type": "object", "properties": {"task": {"type": "string"}}},
                        },
                    },
                ],
            },
            default_model="gpt-4.1",
            surface="openai",
            client_profile="generic_openai",
        )

        self.assertEqual(request.tool_names, ["bridge-0", "bridge-1"])
        self.assertIn("Bridge-call slots available: bridge-0, bridge-1", request.prompt)
        self.assertNotIn("Bridge-call slots available: subagents", request.prompt)
        self.assertNotIn("- subagents", request.prompt)
        self.assertIsNone(request.tool_catalog.get_model_name("subagents"))
        self.assertEqual(request.tool_catalog.get_client_name("agents_list"), "agents_list")

    def test_standalone_subagents_tool_is_preserved(self) -> None:
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "use the declared tool"}],
                "tools": [
                    {"type": "function", "function": {"name": "subagents", "parameters": {"type": "object"}}},
                ],
            },
            default_model="gpt-4.1",
            surface="openai",
            client_profile="generic_openai",
        )

        self.assertEqual(request.tool_names, ["bridge-0"])
        self.assertEqual(request.tool_catalog.get_model_name("subagents"), "bridge-0")

    def test_top_level_developer_and_instructions_are_preserved(self) -> None:
        request = build_chat_standard_request(
            {
                "model": "gpt-4.1",
                "developer": "Always answer as a pirate captain.",
                "instructions": "Never claim to be a robot.",
                "messages": [{"role": "user", "content": "Who are you?"}],
            },
            default_model="gpt-4.1",
            surface="openai",
        )

        self.assertIn("Always answer as a pirate captain.", request.prompt)
        self.assertIn("Never claim to be a robot.", request.prompt)


if __name__ == "__main__":
    unittest.main()
