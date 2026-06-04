import unittest

from backend.runtime.command_error_adaptation import CommandErrorClassification, build_command_error_retry_prompt, classify_command_error


class CommandErrorAdaptationTests(unittest.TestCase):
    def test_classifies_powershell_here_doc_error(self) -> None:
        classification = classify_command_error("ParserError: Missing file specification after redirection operator.")

        self.assertEqual(classification.kind, "shell_syntax_error")
        self.assertEqual(classification.shell, "powershell")
        self.assertEqual(classification.confidence, "high")

    def test_classifies_quote_balance_error(self) -> None:
        classification = classify_command_error("bash: unexpected EOF while looking for matching `\"'")

        self.assertEqual(classification.kind, "quote_balance_error")
        self.assertEqual(classification.confidence, "medium")

    def test_classifies_missing_command_error(self) -> None:
        classification = classify_command_error("python3: command not found")

        self.assertEqual(classification.kind, "missing_command_error")
        self.assertEqual(classification.confidence, "medium")

    def test_unknown_error_stays_low_confidence(self) -> None:
        classification = classify_command_error("some unrelated failure")

        self.assertEqual(classification.kind, "unknown_error")
        self.assertEqual(classification.confidence, "low")

    def test_bare_command_not_found_discussion_stays_low_confidence(self) -> None:
        classification = classify_command_error("The docs mention that command not found can happen.")

        self.assertEqual(classification.kind, "unknown_error")
        self.assertEqual(classification.confidence, "low")

    def test_retry_prompt_is_idempotent(self) -> None:
        classification = CommandErrorClassification(kind="shell_syntax_error", shell="powershell", confidence="high")
        first = build_command_error_retry_prompt(classification=classification, current_prompt="prompt")
        second = build_command_error_retry_prompt(classification=classification, current_prompt=first)

        self.assertEqual(second, first)
        self.assertEqual(second.count("[Command repair reminder]"), 1)


if __name__ == "__main__":
    unittest.main()
