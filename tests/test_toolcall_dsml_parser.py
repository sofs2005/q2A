import unittest

from backend.toolcall.markup_scan import (
    contains_tool_markup_syntax_outside_ignored,
    find_matching_tool_markup_close,
    find_partial_tool_markup_start,
    find_tool_markup_tag_outside_ignored,
)


class DSMLMarkupScannerTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Original tests (kept)
    # ------------------------------------------------------------------

    def test_finds_dsml_tag_outside_markdown_fence(self) -> None:
        text = "```xml\n<|DSML|tool_calls></|DSML|tool_calls>\n```\n<|DSML|tool_calls>"

        tag = find_tool_markup_tag_outside_ignored(text, 0)

        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")
        self.assertEqual(tag.start, text.rindex("<|DSML|tool_calls>"))

    def test_finds_fullwidth_and_cjk_drift_tag(self) -> None:
        text = '前缀 〈！DSML！invoke name="Bash"〉正文'

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

    # ------------------------------------------------------------------
    # 1. Plain XML tool tag recognition
    # ------------------------------------------------------------------

    def test_plain_xml_tool_calls_opening(self) -> None:
        """<tool_calls> is recognised and normalised."""
        tag = find_tool_markup_tag_outside_ignored("<tool_calls>")
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")
        self.assertFalse(tag.closing)

    def test_plain_xml_tool_calls_closing(self) -> None:
        """</tool_calls> is recognised as a closing tag."""
        tag = find_tool_markup_tag_outside_ignored("</tool_calls>")
        self.assertIsNotNone(tag)
        self.assertTrue(tag.closing)
        self.assertEqual(tag.name, "tool_calls")

    def test_plain_xml_hyphenated_tool_calls(self) -> None:
        """<tool-calls> canonicalises to tool_calls."""
        tag = find_tool_markup_tag_outside_ignored("<tool-calls>")
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")
        self.assertEqual(tag.raw_name, "tool-calls")

    def test_plain_xml_toolcalls(self) -> None:
        """<toolcalls> canonicalises to tool_calls."""
        tag = find_tool_markup_tag_outside_ignored("<toolcalls>")
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")

    def test_plain_xml_invoke_with_attribute(self) -> None:
        """<invoke name="Bash"> is recognised."""
        tag = find_tool_markup_tag_outside_ignored('<invoke name="Bash">')
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "invoke")
        self.assertFalse(tag.closing)

    def test_plain_xml_parameter_with_attribute(self) -> None:
        """<parameter name="command"> is recognised."""
        tag = find_tool_markup_tag_outside_ignored('<parameter name="command">')
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "parameter")
        self.assertFalse(tag.closing)

    def test_plain_xml_closing_invoke(self) -> None:
        """</invoke> is recognised."""
        tag = find_tool_markup_tag_outside_ignored("</invoke>")
        self.assertIsNotNone(tag)
        self.assertTrue(tag.closing)
        self.assertEqual(tag.name, "invoke")

    def test_plain_xml_closing_parameter(self) -> None:
        """</parameter> is recognised."""
        tag = find_tool_markup_tag_outside_ignored("</parameter>")
        self.assertIsNotNone(tag)
        self.assertTrue(tag.closing)
        self.assertEqual(tag.name, "parameter")

    def test_plain_xml_nested_wrapper_close(self) -> None:
        """find_matching_tool_markup_close works for plain XML nested wrapper."""
        text = "<tool_calls><tool_calls></tool_calls></tool_calls>"
        open_tag = find_tool_markup_tag_outside_ignored(text, 0)
        self.assertIsNotNone(open_tag)
        self.assertEqual(open_tag.name, "tool_calls")

        close_tag = find_matching_tool_markup_close(text, open_tag)
        self.assertIsNotNone(close_tag)
        self.assertTrue(close_tag.closing)
        self.assertEqual(close_tag.end, len(text))

    # ------------------------------------------------------------------
    # 2. Partial-start detection (streaming hold)
    # ------------------------------------------------------------------

    def test_partial_bare_lt(self) -> None:
        """Bare '<' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<"), 0)

    def test_partial_bare_lt_slash(self) -> None:
        """'</' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("</"), 0)

    def test_partial_dsml_pipe_start(self) -> None:
        """'<|' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<|"), 0)

    def test_partial_dsml_pipe_close_start(self) -> None:
        """'</|' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("</|"), 0)

    def test_partial_dsml_D(self) -> None:
        """'<|D' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<|D"), 0)

    def test_partial_dsml_full_pipe_no_name(self) -> None:
        """'<|DSML|' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<|DSML|"), 0)

    def test_partial_dsml_close_pipe_no_name(self) -> None:
        """'</|DSML|' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("</|DSML|"), 0)

    def test_partial_plain_tool(self) -> None:
        """'<tool' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<tool"), 0)

    def test_partial_plain_tool_slash(self) -> None:
        """'</tool' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("</tool"), 0)

    def test_partial_plain_invoke(self) -> None:
        """'<invoke' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<invoke"), 0)

    def test_partial_plain_param(self) -> None:
        """'<param' at end triggers a hold."""
        self.assertEqual(find_partial_tool_markup_start("<param"), 0)

    def test_partial_fullwidth_prefix(self) -> None:
        """Fullwidth/CJK-folded partial prefixes are also held."""
        self.assertEqual(find_partial_tool_markup_start("〈！DSML！"), 0)
        self.assertEqual(find_partial_tool_markup_start("prefix 〈tool"), 7)

    def test_partial_mid_text(self) -> None:
        """Partial detection works when the prefix is preceded by other text."""
        self.assertEqual(find_partial_tool_markup_start("prefix <|DSML|too"), 7)
        self.assertEqual(find_partial_tool_markup_start("abc <invoke"), 4)

    def test_partial_non_tool_prefix_ignored(self) -> None:
        """'<div' is not a tool tag prefix — returns -1."""
        self.assertEqual(find_partial_tool_markup_start("<div"), -1)
        self.assertEqual(find_partial_tool_markup_start("</span"), -1)

    # ------------------------------------------------------------------
    # 3. Partial start avoided inside ignored regions
    # ------------------------------------------------------------------

    def test_partial_inside_inline_code_returns_neg1(self) -> None:
        """Partial start inside a `...` inline code span is ignored."""
        # The `<tool_ca` sits inside a backtick span.
        text = "prefix `code <tool_ca`"
        self.assertEqual(find_partial_tool_markup_start(text), -1)

    def test_partial_inside_fenced_code_returns_neg1(self) -> None:
        """Partial start inside a ``` fenced block is ignored."""
        text = "```\n<param\n"
        self.assertEqual(find_partial_tool_markup_start(text), -1)

    def test_partial_outside_ignored_still_works(self) -> None:
        """After a closed fence, a partial outside is still detected."""
        text = "```\ncode\n```\n<|DSML|"
        pos = find_partial_tool_markup_start(text)
        self.assertGreaterEqual(pos, 0)
        self.assertLess(pos, len(text))

    # ------------------------------------------------------------------
    # 4. ~~~ fenced code blocks
    # ------------------------------------------------------------------

    def test_tilde_fence_dsml_tag_ignored(self) -> None:
        """A DSML tag inside a ~~~ fenced code block is ignored."""
        text = "~~~\n<|DSML|tool_calls>\n~~~\n"
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))
        self.assertIsNone(find_tool_markup_tag_outside_ignored(text, 0))

    def test_tilde_fence_plain_xml_tag_ignored(self) -> None:
        """A plain XML tool tag inside a ~~~ fence is ignored."""
        text = "~~~xml\n<invoke name=\"Bash\">\n~~~\n"
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))

    def test_tag_after_tilde_fence_found(self) -> None:
        """Tag after a closed ~~~ fence is found."""
        text = "~~~\ncode\n~~~\n<tool_calls>"
        tag = find_tool_markup_tag_outside_ignored(text, 0)
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")

    # ------------------------------------------------------------------
    # 5. Unclosed fence to EOF
    # ------------------------------------------------------------------

    def test_unclosed_backtick_fence_ignores_to_eof(self) -> None:
        """An unclosed ``` fence causes everything after it to be ignored."""
        text = "```python\n<|DSML|tool_calls>\nthis is still inside the fence"
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))
        self.assertIsNone(find_tool_markup_tag_outside_ignored(text, 0))

    def test_unclosed_tilde_fence_ignores_to_eof(self) -> None:
        """An unclosed ~~~ fence causes everything after it to be ignored."""
        text = "~~~bash\n<tool_calls>\n</tool_calls>"
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))

    def test_closed_fence_before_unclosed_still_works(self) -> None:
        """A properly closed fence followed by an unclosed one ignores
        from the second fence onward, but content before the second
        fence (after the first closes) is still visible."""
        text = "```\nclosed\n```\nsafe <|DSML|tool_calls>\n```\nunclosed\n"
        tag = find_tool_markup_tag_outside_ignored(text, 0)
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")

    # ------------------------------------------------------------------
    # 6. Double-backtick inline code
    # ------------------------------------------------------------------

    def test_double_backtick_inline_code_ignored(self) -> None:
        """DSML tag inside ``...`` double-backtick span is ignored."""
        text = "Use ``<|DSML|tool_calls>`` as an example"
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))

    def test_double_backtick_with_single_inside_ignored(self) -> None:
        """Double-backtick span containing a single backtick and a DSML tag."""
        text = "Use ``code with ` and <tool_calls>`` here"
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))

    def test_single_vs_double_backtick(self) -> None:
        """Double backtick does not interfere with single backtick detection."""
        text = "``safe`` and `<|DSML|tool_calls>`"
        # The second one is inside single backticks, should be ignored
        self.assertFalse(contains_tool_markup_syntax_outside_ignored(text))

    # ------------------------------------------------------------------
    # 7. Regression: original DSML still works
    # ------------------------------------------------------------------

    def test_dsml_form_still_works(self) -> None:
        """All original DSML <|DSML|...> forms still parse correctly."""
        tag = find_tool_markup_tag_outside_ignored("<|DSML|tool_calls>")
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "tool_calls")
        self.assertFalse(tag.closing)

        tag = find_tool_markup_tag_outside_ignored("</|DSML|tool_calls>")
        self.assertIsNotNone(tag)
        self.assertTrue(tag.closing)
        self.assertEqual(tag.name, "tool_calls")

        tag = find_tool_markup_tag_outside_ignored('<|DSML|invoke name="Bash">')
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "invoke")

        tag = find_tool_markup_tag_outside_ignored('<|DSML|parameter name="cmd">')
        self.assertIsNotNone(tag)
        self.assertEqual(tag.name, "parameter")


if __name__ == "__main__":
    unittest.main()
