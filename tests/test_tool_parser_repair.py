import unittest

from backend.services.tool_parser import parse_tool_calls_silent


class ToolParserRepairTests(unittest.TestCase):
    def test_parse_tool_calls_silent_prefers_dsml_protocol(self) -> None:
        blocks, stop_reason = parse_tool_calls_silent(
            '<|DSML|tool_calls><|DSML|invoke name="Read"><|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
            [{"name": "Read", "parameters": {}}],
        )

        self.assertEqual(stop_reason, "tool_use")
        tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
        self.assertEqual(tool_blocks[0]["name"], "Read")
        self.assertEqual(tool_blocks[0]["input"], {"file_path": "README.md"})

    def test_legacy_hash_wrapper_still_parses_after_dsml_integration(self) -> None:
        blocks, stop_reason = parse_tool_calls_silent(
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "legacy.md"}}\n##END_CALL##',
            [{"name": "Read", "parameters": {}}],
        )

        self.assertEqual(stop_reason, "tool_use")
        tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
        self.assertEqual(tool_blocks[0]["input"], {"file_path": "legacy.md"})

    def test_image_generate_filters_drift_fields_to_declared_schema(self) -> None:
        blocks, stop_reason = parse_tool_calls_silent(
            '##TOOL_CALL##\n{"name": "image_generate", "input": {"action": "generate", "model": "qwen-image", "prompt": "cat", "aspectRatio": "16:9"}}\n##END_CALL##',
            [{
                "name": "image_generate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "aspect_ratio": {"type": "string"},
                    },
                },
            }],
        )

        self.assertEqual(stop_reason, "tool_use")
        tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
        self.assertEqual(tool_blocks[0]["input"], {"prompt": "cat", "aspect_ratio": "16:9"})

    def test_image_generate_without_schema_keeps_prompt_only(self) -> None:
        blocks, stop_reason = parse_tool_calls_silent(
            '##TOOL_CALL##\n{"name": "image_generate", "input": {"action": "generate", "model": "qwen-image", "prompt": "cat"}}\n##END_CALL##',
            [{"name": "image_generate", "parameters": {}}],
        )

        self.assertEqual(stop_reason, "tool_use")
        tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
        self.assertEqual(tool_blocks[0]["input"], {"prompt": "cat"})


if __name__ == "__main__":
    unittest.main()
