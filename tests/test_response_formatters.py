import unittest

from backend.services.response_formatters import sanitize_visible_answer_text


class ResponseFormatterTests(unittest.TestCase):
    def test_sanitize_visible_answer_text_strips_mixed_text_with_dsml_tool_markup(self) -> None:
        text = (
            "Here is the result.\n"
            '<|DSML|tool_calls><|DSML|invoke name="Read"><|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
        )

        self.assertEqual(sanitize_visible_answer_text(text, tool_use=True), "")

    def test_sanitize_visible_answer_text_ignores_dsml_inside_fenced_code(self) -> None:
        text = "```xml\n<|DSML|tool_calls></|DSML|tool_calls>\n```"

        self.assertEqual(sanitize_visible_answer_text(text, tool_use=True), text)

    def test_sanitize_visible_answer_text_preserves_text_without_tool_use(self) -> None:
        text = '<|DSML|tool_calls></|DSML|tool_calls>'

        self.assertEqual(sanitize_visible_answer_text(text, tool_use=False), text)


if __name__ == "__main__":
    unittest.main()
