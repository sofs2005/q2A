import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.adapter.standard_request import StandardRequest
from backend.toolcore.task_session import build_retry_rebase_prompt, plan_persistent_session_turn


class TaskSessionRetryPromptTests(unittest.IsolatedAsyncioTestCase):
    def _tool_request(self) -> StandardRequest:
        return StandardRequest(
            prompt="Human: inspect file\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            session_key="session",
            tool_names=["read"],
            tools=[{"name": "read", "parameters": {}}],
            tool_enabled=True,
        )

    @staticmethod
    def _app_with_session_record(record):
        affinity = SimpleNamespace(get=AsyncMock(return_value=record))
        return SimpleNamespace(state=SimpleNamespace(session_affinity=affinity))

    async def _plan_hashes_for_payload(self, payload: dict) -> list[str]:
        request = StandardRequest(
            prompt="Human: Who are you?\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            session_key="session",
        )
        plan = await plan_persistent_session_turn(
            app=SimpleNamespace(),
            request=request,
            payload=payload,
            surface="openai",
        )
        return plan.current_hashes

    async def _plan_hashes_for_system(self, system_prompt: str) -> list[str]:
        return await self._plan_hashes_for_payload({
            "system": system_prompt,
            "messages": [{"role": "user", "content": "Who are you?"}],
        })

    async def test_plan_hashes_change_when_top_level_system_prompt_changes(self) -> None:
        self.assertNotEqual(
            await self._plan_hashes_for_system("Always answer as a pirate captain."),
            await self._plan_hashes_for_system("Always answer as a robot."),
        )

    async def test_plan_hashes_change_when_top_level_developer_changes(self) -> None:
        base_payload = {
            "developer": "Always answer as a pirate captain.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }
        changed_payload = {
            "developer": "Always answer as a robot.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }

        self.assertNotEqual(
            await self._plan_hashes_for_payload(base_payload),
            await self._plan_hashes_for_payload(changed_payload),
        )

    async def test_plan_hashes_change_when_top_level_instructions_change(self) -> None:
        base_payload = {
            "instructions": "Always answer as a pirate captain.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }
        changed_payload = {
            "instructions": "Always answer as a robot.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }

        self.assertNotEqual(
            await self._plan_hashes_for_payload(base_payload),
            await self._plan_hashes_for_payload(changed_payload),
        )

    async def test_plan_hashes_change_when_openclaw_user_system_block_changes(self) -> None:
        base_payload = {
            "messages": [
                {"role": "user", "content": "## Memory Recall\nBefore answering, run memory_search."},
                {"role": "user", "content": "System: Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
        }
        changed_payload = {
            "messages": [
                {"role": "user", "content": "## Memory Recall\nBefore answering, run memory_search."},
                {"role": "user", "content": "System: Always answer as a robot."},
                {"role": "user", "content": "Who are you?"},
            ],
        }

        self.assertNotEqual(
            await self._plan_hashes_for_payload(base_payload),
            await self._plan_hashes_for_payload(changed_payload),
        )

    async def test_plan_uses_fresh_upstream_chat_for_tool_requests(self) -> None:
        request = self._tool_request()
        app = self._app_with_session_record(None)

        plan = await plan_persistent_session_turn(
            app=app,
            request=request,
            payload={"messages": [{"role": "user", "content": "inspect file"}]},
            surface="openai",
        )

        self.assertFalse(plan.enabled)
        self.assertFalse(plan.reuse_chat)
        self.assertEqual(plan.reason, "upstream_session_reuse_disabled")
        self.assertEqual(plan.prompt, request.prompt)
        app.state.session_affinity.get.assert_not_awaited()

    async def test_plan_does_not_reuse_existing_tool_session_chat(self) -> None:
        request = self._tool_request()
        record = SimpleNamespace(
            message_hashes=["existing_hash"],
            chat_id="chat_1",
            account_email="bot@example.com",
        )
        reuse_app = self._app_with_session_record(record)

        plan = await plan_persistent_session_turn(
            app=reuse_app,
            request=request,
            payload={
                "messages": [
                    {"role": "user", "content": "inspect file"},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "summarize it"},
                ]
            },
            surface="openai",
        )

        self.assertFalse(plan.enabled)
        self.assertFalse(plan.reuse_chat)
        self.assertIsNone(plan.existing_chat_id)
        self.assertEqual(plan.reason, "upstream_session_reuse_disabled")
        self.assertEqual(plan.prompt, request.prompt)
        reuse_app.state.session_affinity.get.assert_not_awaited()

    def test_search_no_results_prompt_is_generic(self) -> None:
        request = StandardRequest(
            prompt="Human: do task\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tool_names=["web_fetch"],
            tools=[{"name": "web_fetch", "parameters": {}}],
            tool_enabled=True,
        )

        prompt = build_retry_rebase_prompt(request, reason="search_no_results")

        self.assertIn("last search tool returned no results", prompt)
        self.assertNotIn("WebSearch", prompt)

    def test_repeated_same_read_prompt_avoids_edit_bias(self) -> None:
        request = StandardRequest(
            prompt="Human: analyze this script\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tool_names=["read"],
            tools=[{"name": "read", "parameters": {}}],
            tool_enabled=True,
        )

        prompt = build_retry_rebase_prompt(request, reason="repeated_same_read:read")

        self.assertIn("Use the current file content", prompt)
        self.assertNotIn("edit, write, verify", prompt)


if __name__ == "__main__":
    unittest.main()
