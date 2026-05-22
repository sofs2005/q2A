import unittest

from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.toolcore.directive_parser import parse_textual_tool_calls
from backend.toolcore.prompt_contract import build_tool_instruction_block, normalize_prompt_tools, render_history_tool_call


def _parse_history_input(rendered: str, name: str) -> dict:
    result = parse_textual_tool_calls(rendered, [{"name": name}])
    assert result.canonical_calls
    return result.canonical_calls[0].input


class PromptContractTests(unittest.TestCase):
    def test_same_tools_produce_same_contract_after_normalization(self) -> None:
        chat_tools = [{"type": "function", "function": {"name": "Read", "description": "Read file content", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}]
        responses_tools = [{"name": "Read", "description": "Read file content", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]

        chat_contract = build_tool_instruction_block(normalize_prompt_tools(chat_tools), OPENCLAW_OPENAI_PROFILE)
        responses_contract = build_tool_instruction_block(normalize_prompt_tools(responses_tools), OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(chat_contract, responses_contract)

    def test_required_tool_choice_adds_forced_tool_constraint(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([{"name": "Read", "description": "Read file", "parameters": {}}]),
            OPENCLAW_OPENAI_PROFILE,
            tool_choice_mode="required",
            required_tool_name="Read",
        )

        self.assertIn('MUST call the exact tool "Read"', contract)
        self.assertIn("##TOOL_CALL##", contract)
        self.assertNotIn("<|DSML|tool_calls>", contract)

    def test_tool_contract_distinguishes_bridge_tools_from_native_qwen_tools(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([{"name": "image_generate", "description": "Generate image", "parameters": {}}]),
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertIn("gateway bridge tools", contract)
        self.assertIn("not upstream/native Qwen tools", contract)
        self.assertIn("never answer with platform errors such as", contract)
        self.assertIn("Tool image_generate does not exists", contract)

    def test_none_tool_choice_suppresses_required_guidance(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([{"name": "Read", "description": "Read file", "parameters": {}}]),
            OPENCLAW_OPENAI_PROFILE,
            tool_choice_mode="none",
        )

        self.assertIn("do NOT call any tool", contract)
        self.assertNotIn("MUST include at least one tool call", contract)

    def test_tool_contract_does_not_override_client_persona_or_language(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([{"name": "Read", "description": "Read file", "parameters": {}}]),
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertIn("only defines how to serialize tool calls", contract)
        self.assertNotIn("IGNORE any previous output format instructions", contract)
        self.assertNotIn("用户输入什么语言", contract)

    def test_history_tool_call_uses_safe_json_wrapper_style(self) -> None:
        rendered = render_history_tool_call("Read", {"file_path": "README.md"}, CLAUDE_CODE_OPENAI_PROFILE)
        self.assertEqual(
            rendered,
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##',
        )
        self.assertNotIn("<|DSML|", rendered)

    def test_history_tool_call_renders_nested_values(self) -> None:
        rendered = render_history_tool_call(
            "Ask",
            {
                "questions": [{"question": "Proceed?", "multiSelect": False}],
                "count": 2,
                "empty": None,
            },
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertEqual(
            _parse_history_input(rendered, "Ask"),
            {
                "questions": [{"question": "Proceed?", "multiSelect": False}],
                "count": 2,
                "empty": None,
            },
        )
        self.assertNotIn("<|DSML|", rendered)

    def test_history_tool_call_preserves_json_special_strings(self) -> None:
        rendered = render_history_tool_call("Write", {"content": "a]]>b"}, OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(_parse_history_input(rendered, "Write"), {"content": "a]]>b"})
        self.assertNotIn("<![CDATA[", rendered)

    def test_history_tool_call_preserves_entity_literals(self) -> None:
        rendered = render_history_tool_call("Write", {"content": "a &amp; b &lt;tag&gt;"}, OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(
            _parse_history_input(rendered, "Write"),
            {"content": "a &amp; b &lt;tag&gt;"},
        )

    def test_history_tool_call_preserves_string_scalars(self) -> None:
        rendered = render_history_tool_call(
            "Write",
            {"truth": "true", "count": "123", "nothing": "null", "html": "<tag>a</tag>"},
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertEqual(
            _parse_history_input(rendered, "Write"),
            {"truth": "true", "count": "123", "nothing": "null", "html": "<tag>a</tag>"},
        )

    def test_history_tool_call_preserves_surrounding_whitespace(self) -> None:
        rendered = render_history_tool_call("Write", {"content": "  line\n"}, OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(_parse_history_input(rendered, "Write"), {"content": "  line\n"})

    def test_history_tool_call_roundtrips_invalid_nested_keys_as_json(self) -> None:
        rendered = render_history_tool_call(
            "Store",
            {"payload": {"bad key": "value", "1bad": True}},
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertEqual(_parse_history_input(rendered, "Store"), {"payload": {"bad key": "value", "1bad": True}})

    def test_history_tool_call_roundtrips_empty_containers(self) -> None:
        rendered = render_history_tool_call(
            "Store",
            {"items": [], "options": {}, "nested": {"items": [], "options": {}}},
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertEqual(
            _parse_history_input(rendered, "Store"),
            {"items": [], "options": {}, "nested": {"items": [], "options": {}}},
        )

    def test_large_tool_list_preserves_client_declared_order(self) -> None:
        tools = normalize_prompt_tools(
            [
                {"name": "tool_01", "description": "custom one", "parameters": {}},
                {"name": "tool_02", "description": "custom two", "parameters": {}},
                {"name": "tool_03", "description": "custom three", "parameters": {}},
                {"name": "tool_04", "description": "custom four", "parameters": {}},
                {"name": "tool_05", "description": "custom five", "parameters": {}},
                {"name": "tool_06", "description": "custom six", "parameters": {}},
                {"name": "tool_07", "description": "custom seven", "parameters": {}},
                {"name": "tool_08", "description": "custom eight", "parameters": {}},
                {"name": "tool_09", "description": "custom nine", "parameters": {}},
                {"name": "tool_10", "description": "custom ten", "parameters": {}},
                {"name": "tool_11", "description": "custom eleven", "parameters": {}},
                {"name": "tool_12", "description": "custom twelve", "parameters": {}},
                {"name": "tool_13", "description": "custom thirteen", "parameters": {}},
            ]
        )

        contract = build_tool_instruction_block(tools, OPENCLAW_OPENAI_PROFILE)

        self.assertIn("- tool_01: custom one", contract)
        self.assertIn("- tool_13: custom thirteen", contract)
        self.assertLess(contract.index("- tool_01: custom one"), contract.index("- tool_13: custom thirteen"))
        self.assertNotIn("Other available tools:", contract)


if __name__ == "__main__":
    unittest.main()
