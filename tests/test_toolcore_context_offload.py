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
        self.assertEqual(plan.inline_messages[0]["content"], SYSTEM_CONTEXT_PROMPT_NOTE)
        self.assertIn("B" * 20, plan.inline_messages[1]["content"])

    def test_plan_rewrites_large_latest_user_message_with_note(self) -> None:
        messages = [
            {"role": "assistant", "content": "A" * 40},
            {"role": "user", "content": "latest task " * 12},
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertEqual(plan.inline_messages[0]["content"], SYSTEM_CONTEXT_PROMPT_NOTE)
        self.assertIn("latest task", plan.inline_messages[1]["content"])

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

    def test_plan_uses_latest_trusted_user_for_inline_current_task(self) -> None:
        messages = [
            {"role": "user", "content": "请回复飞哥刚才问的灵性问题"},
            {"role": "assistant", "content": "好的。"},
            {
                "role": "user",
                "content": (
                    "System (untrusted): [2026-05-19 08:01:14 GMT+8] 本次贴吧心跳任务已执行完毕。\n"
                    "System (untrusted): 飞哥，你觉得这是不是灵性？\n\n"
                    "Conversation info (untrusted metadata):\n```json\n{\"chat_id\": \"wechat:telphy\"}\n```"
                ),
            },
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertEqual(plan.inline_messages[0]["content"], SYSTEM_CONTEXT_PROMPT_NOTE)
        self.assertEqual(plan.inline_messages[1]["content"], "请回复飞哥刚才问的灵性问题")

    def test_plan_preserves_latest_user_text_after_untrusted_prefix(self) -> None:
        messages = [
            {"role": "user", "content": "上一轮问题"},
            {"role": "assistant", "content": "上一轮回答"},
            {
                "role": "user",
                "content": (
                    "System (untrusted): [2026-05-19 08:01:14 GMT+8] 心跳信息。\n\n"
                    "当前轮问题：请只回答 banana。\n\n"
                    "Conversation info (untrusted metadata):\n```json\n{\"chat_id\": \"wechat:telphy\"}\n```"
                ),
            },
        ]

        plan = self.offloader.plan(messages, tools=[], client_profile="openclaw_openai")

        self.assertEqual(plan.inline_messages[0]["content"], SYSTEM_CONTEXT_PROMPT_NOTE)
        self.assertEqual(plan.inline_messages[1]["content"], "当前轮问题：请只回答 banana。")

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
