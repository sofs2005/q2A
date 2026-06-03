import unittest

from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.toolcall.formats_dsml import parse_dsml_format
from backend.toolcore.prompt_contract import build_tool_instruction_block, normalize_prompt_tools, render_history_tool_call


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
        self.assertIn("<|DSML|tool_calls>", contract)
        self.assertNotIn("##TOOL_CALL##", contract)

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

    def test_tool_contract_warns_shell_commands_still_need_shell_quoting(self) -> None:
        contract = build_tool_instruction_block(
            normalize_prompt_tools([
                {
                    "name": "Bash",
                    "description": "Runs a shell command.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ]),
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertIn("CDATA preserves parameter text exactly", contract)
        self.assertIn("valid shell syntax", contract)
        self.assertIn("python -c", contract)
        self.assertIn("here-document", contract)

    def test_history_tool_call_uses_dsml_wrapper_style(self) -> None:
        rendered = render_history_tool_call("Read", {"file_path": "README.md"}, CLAUDE_CODE_OPENAI_PROFILE)
        self.assertEqual(
            rendered,
            '<|DSML|tool_calls>\n  <|DSML|invoke name="Read">\n    <|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>',
        )

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

        self.assertIn('<|DSML|parameter name="questions"><item><question><![CDATA[Proceed?]]></question><multiSelect>false</multiSelect></item></|DSML|parameter>', rendered)
        self.assertIn('<|DSML|parameter name="count">2</|DSML|parameter>', rendered)
        self.assertIn('<|DSML|parameter name="empty">null</|DSML|parameter>', rendered)

    def test_history_tool_call_splits_cdata_end_marker(self) -> None:
        rendered = render_history_tool_call("Write", {"content": "a]]>b"}, OPENCLAW_OPENAI_PROFILE)

        self.assertIn("<![CDATA[a]]]]><![CDATA[>b]]>", rendered)
        self.assertEqual(parse_dsml_format(rendered, {"Write"})[0]["input"], {"content": "a]]>b"})

    def test_history_tool_call_preserves_cdata_entity_literals(self) -> None:
        rendered = render_history_tool_call("Write", {"content": "a &amp; b &lt;tag&gt;"}, OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(
            parse_dsml_format(rendered, {"Write"})[0]["input"],
            {"content": "a &amp; b &lt;tag&gt;"},
        )

    def test_history_tool_call_preserves_cdata_string_scalars(self) -> None:
        rendered = render_history_tool_call(
            "Write",
            {"truth": "true", "count": "123", "nothing": "null", "html": "<tag>a</tag>"},
            OPENCLAW_OPENAI_PROFILE,
        )

        self.assertEqual(
            parse_dsml_format(rendered, {"Write"})[0]["input"],
            {"truth": "true", "count": "123", "nothing": "null", "html": "<tag>a</tag>"},
        )

    def test_history_tool_call_preserves_cdata_surrounding_whitespace(self) -> None:
        rendered = render_history_tool_call("Write", {"content": "  line\n"}, OPENCLAW_OPENAI_PROFILE)

        self.assertEqual(parse_dsml_format(rendered, {"Write"})[0]["input"], {"content": "  line\n"})

    def test_history_tool_call_roundtrips_invalid_nested_keys_as_json(self) -> None:
        rendered = render_history_tool_call(
            "Store",
            {"payload": {"bad key": "value", "1bad": True}},
            OPENCLAW_OPENAI_PROFILE,
        )

        calls = parse_dsml_format(rendered, {"Store"})

        self.assertEqual(calls, [{"name": "Store", "input": {"payload": {"bad key": "value", "1bad": True}}}])

    def test_history_tool_call_roundtrips_empty_containers(self) -> None:
        rendered = render_history_tool_call(
            "Store",
            {"items": [], "options": {}, "nested": {"items": [], "options": {}}},
            OPENCLAW_OPENAI_PROFILE,
        )

        calls = parse_dsml_format(rendered, {"Store"})

        self.assertEqual(
            calls,
            [{"name": "Store", "input": {"items": [], "options": {}, "nested": {"items": [], "options": {}}}}],
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
