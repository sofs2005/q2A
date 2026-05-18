import unittest
from types import SimpleNamespace

from backend.toolcore.context_offload import ContextOffloader, SYSTEM_CONTEXT_PROMPT_NOTE


class ToolCoreContextOffloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            CONTEXT_INLINE_MAX_CHARS=80,
            CONTEXT_FORCE_FILE_MAX_CHARS=160,
            CONTEXT_ATTACHMENT_TTL_SECONDS=600,
        )
        self.offloader = ContextOffloader(self.settings)

    def test_plan_keeps_small_prompt_inline(self) -> None:
        messages = [{"role": "user", "content": "short request"}]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertEqual(plan.mode, "inline")
        self.assertEqual(plan.inline_messages, messages)
        self.assertEqual(plan.generated_files, [])

    def test_plan_offloads_large_history_even_when_latest_user_is_small(self) -> None:
        messages = [
            {"role": "assistant", "content": "A" * 120},
            {"role": "tool", "content": "tool output\n" * 20},
            {"role": "user", "content": "latest task"},
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertEqual(plan.mode, "file")
        self.assertEqual(len(plan.generated_files), 1)
        self.assertIn("Message 1 [assistant]", plan.generated_files[0].text)
        self.assertIn("Message 2 [tool]", plan.generated_files[0].text)
        self.assertIn(SYSTEM_CONTEXT_PROMPT_NOTE, plan.inline_messages[0]["content"])

    def test_plan_creates_file_mode_for_large_latest_user_input(self) -> None:
        messages = [
            {"role": "assistant", "content": "A" * 40},
            {"role": "user", "content": "B" * 120},
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertEqual(plan.mode, "file")
        self.assertEqual(len(plan.generated_files), 1)
        self.assertIn("Message 1 [assistant]", plan.generated_files[0].text)
        self.assertTrue(plan.inline_messages[0]["content"].endswith(SYSTEM_CONTEXT_PROMPT_NOTE))

    def test_plan_rewrites_large_latest_user_message_with_note(self) -> None:
        messages = [
            {"role": "assistant", "content": "A" * 40},
            {"role": "user", "content": "latest task " * 12},
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertIn("latest task", plan.inline_messages[0]["content"])
        self.assertIn(SYSTEM_CONTEXT_PROMPT_NOTE, plan.inline_messages[0]["content"])

    def test_plan_history_file_contains_large_latest_user_message(self) -> None:
        messages = [
            {"role": "assistant", "content": "A" * 40},
            {"role": "user", "content": "latest task " * 12},
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertIn("Message 2 [user]", plan.generated_files[0].text)
        self.assertIn("latest task", plan.generated_files[0].text)

    def test_plan_adds_tools_context_file_when_large_input_has_tools(self) -> None:
        messages = [{"role": "user", "content": "latest task " * 12}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Skill",
                    "description": "Run an available slash command skill.",
                    "parameters": {"type": "object", "properties": {"skill": {"type": "string"}}},
                },
            }
        ]

        plan = self.offloader.plan(messages, tools=tools, client_profile="openclaw_openai")

        self.assertEqual(len(plan.generated_files), 2)
        tools_file = next(file for file in plan.generated_files if "Available tool descriptions" in file.text)
        self.assertIn("Tool: bridge-0", tools_file.text)
        self.assertNotIn("Tool: Skill", tools_file.text)
        self.assertIn("Run an available slash command skill.", tools_file.text)
        self.assertIn('"skill"', tools_file.text)

    def test_plan_adds_tools_context_file_even_when_latest_user_is_small(self) -> None:
        messages = [{"role": "user", "content": "latest task"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Skill",
                    "description": "Run an available slash command skill.",
                    "parameters": {"type": "object", "properties": {"skill": {"type": "string"}}},
                },
            }
        ]

        plan = self.offloader.plan(messages, tools=tools, client_profile="openclaw_openai")

        self.assertEqual(plan.mode, "hybrid")
        tools_file = next(file for file in plan.generated_files if "Available tool descriptions" in file.text)
        self.assertIn("Tool: bridge-0", tools_file.text)
        self.assertIn('"skill"', tools_file.text)


if __name__ == "__main__":
    unittest.main()
