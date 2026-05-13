import unittest

from backend.services.context_attachment_manager import derive_session_key


class ContextAttachmentManagerTests(unittest.TestCase):
    def test_derive_session_key_changes_when_system_prompt_changes(self) -> None:
        base_payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
        }
        changed_payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "Always answer as a robot."},
                {"role": "user", "content": "Who are you?"},
            ],
        }

        self.assertNotEqual(
            derive_session_key("openai", "tok", base_payload),
            derive_session_key("openai", "tok", changed_payload),
        )

    def test_explicit_session_key_is_scoped_by_system_prompt(self) -> None:
        base_payload = {
            "model": "gpt-4.1",
            "session_key": "conversation-1",
            "metadata": {},
            "messages": [
                {"role": "system", "content": "Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
        }
        changed_payload = {
            "model": "gpt-4.1",
            "session_key": "conversation-1",
            "metadata": {},
            "messages": [
                {"role": "system", "content": "Always answer as a robot."},
                {"role": "user", "content": "Who are you?"},
            ],
        }

        base_key = derive_session_key("openai", "tok", base_payload)
        changed_key = derive_session_key("openai", "tok", changed_payload)

        self.assertNotEqual(base_key, "conversation-1")
        self.assertNotEqual(base_key, changed_key)

    def test_derive_session_key_changes_when_developer_prompt_changes(self) -> None:
        base_payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "developer", "content": "Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
        }
        changed_payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "developer", "content": "Always answer as a robot."},
                {"role": "user", "content": "Who are you?"},
            ],
        }

        self.assertNotEqual(
            derive_session_key("openai", "tok", base_payload),
            derive_session_key("openai", "tok", changed_payload),
        )

    def test_derive_session_key_changes_when_top_level_developer_changes(self) -> None:
        base_payload = {
            "model": "gpt-4.1",
            "developer": "Always answer as a pirate captain.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }
        changed_payload = {
            "model": "gpt-4.1",
            "developer": "Always answer as a robot.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }

        self.assertNotEqual(
            derive_session_key("openai", "tok", base_payload),
            derive_session_key("openai", "tok", changed_payload),
        )

    def test_derive_session_key_changes_when_openclaw_user_system_block_changes(self) -> None:
        base_payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "## Memory Recall\nBefore answering, run memory_search."},
                {"role": "user", "content": "System: Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
        }
        changed_payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "## Memory Recall\nBefore answering, run memory_search."},
                {"role": "user", "content": "System: Always answer as a robot."},
                {"role": "user", "content": "Who are you?"},
            ],
        }

        self.assertNotEqual(
            derive_session_key("openai", "tok", base_payload),
            derive_session_key("openai", "tok", changed_payload),
        )


if __name__ == "__main__":
    unittest.main()
