import unittest

from backend.services.standard_request_builder import build_chat_standard_request


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
