import unittest

from backend.adapter.standard_request import StandardRequest


class TaskSessionRetryPromptTests(unittest.TestCase):
    def test_standard_request_has_no_session_reuse_fields_for_system_prompt_tracking(self) -> None:
        request = StandardRequest(
            prompt="System: Always answer as a pirate captain.\n\nHuman: Who are you?\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )

        for attr in (
            "persistent_session",
            "full_prompt",
            "upstream_chat_id",
            "session_message_hashes",
            "session_chat_invalidated",
        ):
            self.assertFalse(hasattr(request, attr))

    def test_standard_request_has_no_session_reuse_fields_for_developer_prompt_tracking(self) -> None:
        request = StandardRequest(
            prompt="System: Follow developer instructions.\n\nHuman: Who are you?\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )

        for attr in (
            "persistent_session",
            "full_prompt",
            "upstream_chat_id",
            "session_message_hashes",
            "session_chat_invalidated",
        ):
            self.assertFalse(hasattr(request, attr))


if __name__ == "__main__":
    unittest.main()
