import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.services.context_attachment_manager import derive_session_key, prepare_context_attachments
from backend.toolcore.context_offload import ContextOffloader


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


class ContextAttachmentPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_requests_can_upload_generated_history_context(self) -> None:
        uploaded = []
        saved_texts = []

        async def save_text(filename, text, content_type, purpose):
            saved_texts.append(text)
            return {
                "id": filename,
                "path": f"/tmp/{filename}",
                "filename": filename,
                "content_type": content_type,
                "sha256": "sha-history",
                "created_at": 1,
            }

        async def upload_local_file(_acc, local_meta):
            uploaded.append(local_meta)
            return {"remote_ref": {"file_id": "file-history", "filename": local_meta["filename"]}}

        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
            ),
            file_store=SimpleNamespace(
                save_text=save_text,
                delete_path=AsyncMock(),
            ),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(
                get=AsyncMock(return_value=None),
                set=AsyncMock(),
            ),
            upstream_file_uploader=SimpleNamespace(upload_local_file=upload_local_file),
        ))
        payload = {
            "model": "gpt-4.1",
            "system": "You are a personal assistant running inside OpenClaw.\n" + "runtime line\n" * 20 + "Always answer as a pirate captain.",
            "messages": [
                {"role": "user", "content": "Who are you?"},
            ],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
        )

        self.assertEqual(result["context_mode"], "file")
        self.assertEqual(len(result["upstream_files"]), 1)
        uploaded_filename = result["upstream_files"][0]["filename"]
        self.assertRegex(uploaded_filename, r"^[0-9a-f]{32}\.txt$")
        self.assertNotIn("qwen2api", uploaded_filename)
        self.assertNotIn("context", uploaded_filename)
        self.assertEqual(len(uploaded), 1)
        self.assertEqual(len(saved_texts), 1)
        self.assertIn("Always answer as a pirate captain.", saved_texts[0])
        self.assertIn("Who are you?", saved_texts[0])


if __name__ == "__main__":
    unittest.main()
