import unittest

from backend.toolcall.markup_scan import (
    contains_tool_markup_syntax_outside_ignored,
    find_matching_tool_markup_close,
    find_partial_tool_markup_start,
    find_tool_markup_tag_outside_ignored,
)


class DSMLMarkupScannerTests(unittest.TestCase):
    def test_finds_dsml_tag_outside_markdown_fence(self) -> None:
        text = "```xml\n<|DSML|tool_calls></|DSML|tool_calls>\n```\n<|DSML|tool_calls>"

        tag = find_tool_markup_tag_outside_ignored(text, 0)

        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")
        self.assertEqual(tag.start, text.rindex("<|DSML|tool_calls>"))

    def test_finds_fullwidth_and_cjk_drift_tag(self) -> None:
        text = '前缀 〈！DSML！invoke name=“Bash”〉正文'

        tag = find_tool_markup_tag_outside_ignored(text, 0)

        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "invoke")
        self.assertFalse(tag.closing)

    def test_matches_nested_wrapper_close(self) -> None:
        text = "<|DSML|tool_calls><|DSML|tool_calls></|DSML|tool_calls></|DSML|tool_calls>"
        open_tag = find_tool_markup_tag_outside_ignored(text, 0)

        close_tag = find_matching_tool_markup_close(text, open_tag)

        self.assertIsNotNone(close_tag)
        self.assertTrue(close_tag.closing)
        self.assertEqual(close_tag.end, len(text))

    def test_partial_tag_start_is_held(self) -> None:
        self.assertEqual(find_partial_tool_markup_start("abc <|DSML|tool"), 4)
        self.assertEqual(find_partial_tool_markup_start("abc <not_a_tool"), -1)

    def test_ignores_inline_code_span(self) -> None:
        text = "Use `<|DSML|tool_calls>` as an example"

        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))


if __name__ == "__main__":
    unittest.main()
