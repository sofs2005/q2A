import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.runtime.attachment_types import NormalizedAttachment
from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.services.context_attachment_manager import derive_session_key, prepare_context_attachments
from backend.services.upstream_file_uploader import FileUploadRateLimitedError
from backend.services.token_calc import count_tokens
from backend.toolcore.context_offload import ContextOffloader, SYSTEM_CONTEXT_PROMPT_NOTE
from backend.toolcore.prompt_builder import messages_to_prompt


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
    async def test_prepare_context_attachments_keeps_caller_session_key(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=1000,
                CONTEXT_FORCE_FILE_MAX_CHARS=2000,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(),
            file_store=SimpleNamespace(),
            session_affinity=SimpleNamespace(),
            upstream_file_cache=SimpleNamespace(),
            upstream_file_uploader=SimpleNamespace(),
        ))
        payload = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "hello"}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:req_123",
        )

        self.assertEqual(result["session_key"], "openai:req_123")

    async def test_tool_requests_can_upload_large_latest_user_context(self) -> None:
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
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
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
            "system": "Always answer as a pirate captain.",
            "messages": [
                {"role": "assistant", "content": "prior result " * 10},
                {"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read file contents",
                        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    },
                }
            ],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertEqual(result["context_mode"], "file")
        self.assertEqual(len(result["upstream_files"]), 2)
        uploaded_filename = result["upstream_files"][0]["filename"]
        self.assertRegex(uploaded_filename, r"^[0-9a-f]{32}\.txt$")
        self.assertNotIn("qwen2api", uploaded_filename)
        self.assertNotIn("context", uploaded_filename)
        self.assertEqual(len(uploaded), 2)
        self.assertEqual(len(saved_texts), 2)
        self.assertEqual(result["context_attachment_tokens"], sum(count_tokens(text) for text in saved_texts))
        history_text = next(text for text in saved_texts if "Please analyze this current input." in text)
        tools_text = next(text for text in saved_texts if "Available tool descriptions" in text)
        self.assertIn("Always answer as a pirate captain.", history_text)
        self.assertIn("Please analyze this current input.", history_text)
        self.assertIn("Tool: bridge-0", tools_text)
        self.assertNotIn("Tool: read", tools_text)
        self.assertIn("Read file contents", tools_text)

    async def test_generated_tools_context_matches_filtered_bridge_slots(self) -> None:
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

        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
            ),
            file_store=SimpleNamespace(save_text=save_text, delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(
                upload_local_file=AsyncMock(return_value={"remote_ref": {"file_id": "file-id", "filename": "ctx.txt"}})
            ),
        ))
        payload = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20}],
            "tools": [
                {"type": "function", "function": {"name": "subagents", "description": "Subagent alias", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "agents_list", "description": "List agents", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "sessions_spawn", "description": "Spawn session", "parameters": {"type": "object", "properties": {"task": {"type": "string"}}}}},
            ],
        }

        await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="generic_openai",
            session_key="openai:test",
        )

        tools_text = next(text for text in saved_texts if "Available tool descriptions" in text)
        self.assertIn("Tool: bridge-0", tools_text)
        self.assertIn("List agents", tools_text)
        self.assertIn("Tool: bridge-1", tools_text)
        self.assertIn("Spawn session", tools_text)
        self.assertNotIn("Subagent alias", tools_text)
        self.assertNotIn("Tool: bridge-2", tools_text)

    async def test_large_prior_history_with_small_latest_user_uploads_history_and_tools(self) -> None:
        saved_texts = []

        async def save_text(filename, text, content_type, purpose):
            saved_texts.append(text)
            return {
                "id": filename,
                "path": f"/tmp/{filename}",
                "filename": filename,
                "content_type": content_type,
                "sha256": f"sha-{len(saved_texts)}",
                "created_at": 1,
            }

        async def upload_local_file(_acc, local_meta):
            return {"remote_ref": {"file_id": local_meta["sha256"], "filename": local_meta["filename"]}}

        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
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
            "messages": [
                {"role": "assistant", "content": "prior result " * 20},
                {"role": "tool", "content": "tool output\n" * 20},
                {"role": "user", "content": "continue"},
            ],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertEqual(result["context_mode"], "file")
        self.assertEqual(len(result["upstream_files"]), 2)
        self.assertTrue(any("Message 1 [assistant]" in text for text in saved_texts))
        self.assertTrue(any("Available tool descriptions" in text for text in saved_texts))
        self.assertIn(SYSTEM_CONTEXT_PROMPT_NOTE, result["payload"]["messages"][0]["content"])

    async def test_generated_context_fallback_preserves_latest_user_task(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=None),
                acquire_wait_preferred=AsyncMock(return_value=None),
                release=lambda _acc: None,
            ),
            file_store=SimpleNamespace(save_text=AsyncMock(), delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(upload_local_file=AsyncMock()),
        ))
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "You are a personal assistant running inside OpenClaw.\n" + "tooling\n" * 500},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "现在检查当前项目为什么工具不可见"},
            ],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertTrue(result["attachment_fallback"])
        self.assertEqual(result["context_mode"], "inline")
        contents = [message.get("content", "") for message in result["payload"]["messages"]]
        self.assertTrue(any("现在检查当前项目为什么工具不可见" in content for content in contents))
        self.assertEqual(contents[-1], "现在检查当前项目为什么工具不可见")

        prompt = messages_to_prompt(result["payload"], client_profile=OPENCLAW_OPENAI_PROFILE).prompt
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 现在检查当前项目为什么工具不可见", prompt)
        self.assertNotIn(f"Human (CURRENT TASK - TOP PRIORITY): {SYSTEM_CONTEXT_PROMPT_NOTE}", prompt)

    async def test_generated_context_fallback_preserves_full_history_summary(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=None),
                acquire_wait_preferred=AsyncMock(return_value=None),
                release=lambda _acc: None,
            ),
            file_store=SimpleNamespace(save_text=AsyncMock(), delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(upload_local_file=AsyncMock()),
        ))
        tail_marker = "TAIL_MARKER_DO_NOT_DROP"
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "A" * 1500},
                {"role": "assistant", "content": tail_marker},
                {"role": "user", "content": "latest task"},
            ],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertTrue(result["attachment_fallback"])
        self.assertIn(tail_marker, result["payload"]["messages"][0]["content"])

    async def test_generated_context_ignores_existing_affinity_when_selecting_upload_account(self) -> None:
        async def save_text(filename, text, content_type, purpose):
            return {
                "id": filename,
                "path": f"/tmp/{filename}",
                "filename": filename,
                "content_type": content_type,
                "sha256": "sha-history",
                "created_at": 1,
            }

        async def upload_local_file(acc, local_meta):
            return {"remote_ref": {"file_id": f"file-{acc.email}", "filename": local_meta["filename"]}}

        selected_account = SimpleNamespace(email="round-robin@example.com")
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=selected_account),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="sticky@example.com")),
                release=lambda _acc: None,
            ),
            file_store=SimpleNamespace(
                save_text=save_text,
                delete_path=AsyncMock(),
            ),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=SimpleNamespace(account_email="sticky@example.com")),
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
            "system": "system context",
            "messages": [{"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20}],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        app.state.account_pool.acquire_wait.assert_awaited_once_with(timeout=60)
        app.state.account_pool.acquire_wait_preferred.assert_not_awaited()
        self.assertEqual(result["bound_account"].email, "round-robin@example.com")
        self.assertEqual(result["bound_account_email"], "round-robin@example.com")
        self.assertEqual(result["upstream_files"][0]["file_id"], "file-round-robin@example.com")

    async def test_generated_context_falls_back_inline_when_no_account_available(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=None),
                acquire_wait_preferred=AsyncMock(return_value=None),
                release=lambda _acc: None,
            ),
            file_store=SimpleNamespace(
                save_text=AsyncMock(),
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
            upstream_file_uploader=SimpleNamespace(upload_local_file=AsyncMock()),
        ))
        payload = {
            "model": "gpt-4.1",
            "system": "system context",
            "messages": [{"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20}],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
            "upstream_files": [{"file_id": "existing-file"}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertEqual(result["context_mode"], "inline")
        self.assertTrue(result["attachment_fallback"])
        self.assertEqual(result["upstream_files"], [{"file_id": "existing-file"}])
        self.assertIsNone(result["bound_account"])
        self.assertIn("system context", result["payload"]["messages"][0]["content"])
        app.state.session_affinity.bind_account.assert_not_awaited()
        app.state.file_store.save_text.assert_not_awaited()

    async def test_generated_context_switches_account_on_upload_rate_limit(self) -> None:
        """上传命中每日配额(RateLimited)时，标记超限号并换到未超限号继续上传。"""
        acc_limited = SimpleNamespace(email="up-limited@example.com")
        acc_fresh = SimpleNamespace(email="up-fresh@example.com")

        upload_calls: list[str] = []

        async def upload_local_file(acc, local_meta):
            upload_calls.append(acc.email)
            if len(upload_calls) == 1:
                raise FileUploadRateLimitedError(retry_after_seconds=19 * 3600)
            return {"remote_ref": {"file_id": f"file-{acc.email}", "filename": local_meta["filename"]}}

        async def save_text(filename, text, content_type, purpose):
            return {
                "id": filename,
                "path": f"/tmp/{filename}",
                "filename": filename,
                "content_type": content_type,
                "sha256": "sha-history",
                "created_at": 1,
            }

        mark_calls: list[tuple[str, float]] = []

        def mark_file_upload_limited(acc, retry_after_seconds, error_message=""):
            mark_calls.append((acc.email, retry_after_seconds))

        acquire_wait = AsyncMock(side_effect=[acc_limited, acc_fresh])
        save = AsyncMock()
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=acquire_wait,
                acquire_wait_preferred=AsyncMock(),
                release=lambda _acc: None,
                mark_file_upload_limited=mark_file_upload_limited,
                file_upload_blocked_emails=lambda: set(),
                save=save,
            ),
            file_store=SimpleNamespace(save_text=save_text, delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(upload_local_file=upload_local_file),
        ))
        payload = {
            "model": "gpt-4.1",
            "system": "Always answer as a pirate captain.",
            "messages": [
                {"role": "assistant", "content": "prior result " * 20},
                {"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20},
            ],
            "tools": [
                {"type": "function", "function": {"name": "read", "description": "Read file contents", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}
            ],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertFalse(result["attachment_fallback"])
        self.assertEqual(result["context_mode"], "file")
        self.assertEqual(result["bound_account"].email, "up-fresh@example.com")
        self.assertEqual(mark_calls, [("up-limited@example.com", 19 * 3600)])
        save.assert_awaited()
        self.assertGreaterEqual(acquire_wait.await_count, 2)
        switch_call = acquire_wait.await_args_list[1]
        self.assertIn("up-limited@example.com", switch_call.kwargs.get("exclude", set()))
        for ref in result["upstream_files"]:
            self.assertEqual(ref["file_id"], "file-up-fresh@example.com")

    async def test_generated_context_falls_back_inline_when_all_accounts_upload_limited(self) -> None:
        """所有账号都超出上传配额时，回退内联发送上下文。"""
        acc_limited = SimpleNamespace(email="up-limited@example.com")

        async def upload_local_file(acc, local_meta):
            raise FileUploadRateLimitedError(retry_after_seconds=19 * 3600)

        mark_calls: list[str] = []

        def mark_file_upload_limited(acc, retry_after_seconds, error_message=""):
            mark_calls.append(acc.email)

        acquire_wait = AsyncMock(side_effect=[acc_limited, None])
        save = AsyncMock()
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=acquire_wait,
                acquire_wait_preferred=AsyncMock(),
                release=lambda _acc: None,
                mark_file_upload_limited=mark_file_upload_limited,
                file_upload_blocked_emails=lambda: set(),
                save=save,
            ),
            file_store=SimpleNamespace(
                save_text=AsyncMock(return_value={
                    "id": "ctx.txt", "path": "/tmp/ctx.txt", "filename": "ctx.txt",
                    "content_type": "text/plain", "sha256": "sha-history", "created_at": 1,
                }),
                delete_path=AsyncMock(),
            ),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(upload_local_file=upload_local_file),
        ))
        payload = {
            "model": "gpt-4.1",
            "system": "system context",
            "messages": [{"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20}],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
        }

        result = await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        self.assertTrue(result["attachment_fallback"])
        self.assertEqual(result["context_mode"], "inline")
        self.assertEqual(mark_calls, ["up-limited@example.com"])
        save.assert_awaited()
        self.assertIn("system context", result["payload"]["messages"][0]["content"])

    async def test_initial_acquire_excludes_known_upload_limited_accounts(self) -> None:
        """初始挑号时排除已知上传超限账号；exclude 非空才传 exclude。"""
        async def upload_local_file(acc, local_meta):
            return {"remote_ref": {"file_id": f"file-{acc.email}", "filename": local_meta["filename"]}}

        async def save_text(filename, text, content_type, purpose):
            return {
                "id": filename,
                "path": f"/tmp/{filename}",
                "filename": filename,
                "content_type": content_type,
                "sha256": "sha-history",
                "created_at": 1,
            }

        acquire_wait = AsyncMock(return_value=SimpleNamespace(email="fresh@example.com"))
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=80,
                CONTEXT_FORCE_FILE_MAX_CHARS=160,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=acquire_wait,
                acquire_wait_preferred=AsyncMock(),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: {"blocked@example.com"},
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(save_text=save_text, delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=SimpleNamespace(upload_local_file=upload_local_file),
        ))
        payload = {
            "model": "gpt-4.1",
            "system": "system context",
            "messages": [{"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20}],
            "tools": [{"name": "read", "description": "Read file contents", "parameters": {}}],
        }

        await prepare_context_attachments(
            app=app,
            payload=payload,
            surface="openai",
            auth_token="tok",
            client_profile="openclaw_openai",
            session_key="openai:test",
        )

        acquire_wait.assert_awaited_once_with(timeout=60, exclude={"blocked@example.com"})


class InlineMediaAttachmentTests(unittest.IsolatedAsyncioTestCase):
    """图片 / 音频 / 视频走 data URI 内联；文档仍走 OSS 上传。"""

    def _build_app(self, *, uploader, inline_enabled: bool = True, max_bytes: int = 10 * 1024 * 1024):
        delete_path = AsyncMock()
        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(
                CONTEXT_INLINE_MAX_CHARS=4000,
                CONTEXT_FORCE_FILE_MAX_CHARS=10000,
                CONTEXT_ATTACHMENT_TTL_SECONDS=600,
                INLINE_ATTACHMENT_ENABLED=inline_enabled,
                INLINE_ATTACHMENT_MAX_BYTES=max_bytes,
            )),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(delete_path=delete_path),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=uploader,
        ))
        return app, delete_path

    def _payload(self) -> dict:
        return {"model": "qwen3.8-max", "messages": [{"role": "user", "content": "what is this"}]}

    def _attachment(self, filename: str, content_type: str, raw: bytes) -> tuple[NormalizedAttachment, str]:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
        handle.write(raw)
        handle.close()
        attachment = NormalizedAttachment(
            file_id="local-1",
            filename=filename,
            content_type=content_type,
            source="inline-image",
            local_path=handle.name,
            sha256="abc",
            purpose="user-upload",
        )
        return attachment, handle.name

    async def test_image_is_inlined_without_upload_or_account_binding(self) -> None:
        uploader = SimpleNamespace(upload_local_file=AsyncMock())
        app, delete_path = self._build_app(uploader=uploader)
        attachment, local_path = self._attachment("a.png", "image/png", b"png-bytes")
        try:
            result = await prepare_context_attachments(
                app=app,
                payload=self._payload(),
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:inline",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(len(result["upstream_files"]), 1)
        ref = result["upstream_files"][0]
        self.assertEqual(ref["type"], "image")
        self.assertEqual(ref["show_type"] if "show_type" in ref else ref["showType"], "image")
        self.assertEqual(ref["file_class"], "vision")
        self.assertEqual(ref["size"], len(b"png-bytes"))
        self.assertTrue(ref["url"].startswith("data:image/png;base64,"))
        self.assertEqual(ref["id"], "")
        self.assertIsNone(result["bound_account"])
        self.assertIsNone(result["bound_account_email"])
        uploader.upload_local_file.assert_not_awaited()
        app.state.account_pool.acquire_wait.assert_not_awaited()
        app.state.session_affinity.bind_account.assert_not_awaited()
        delete_path.assert_awaited()

    async def test_audio_and_video_are_inlined(self) -> None:
        uploader = SimpleNamespace(upload_local_file=AsyncMock())
        app, _ = self._build_app(uploader=uploader)
        audio, audio_path = self._attachment("voice.mp3", "audio/mpeg", b"mp3-bytes")
        video, video_path = self._attachment("clip.mp4", "video/mp4", b"mp4-bytes")
        try:
            result = await prepare_context_attachments(
                app=app,
                payload=self._payload(),
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:inline",
                existing_attachments=[audio, video],
            )
        finally:
            Path(audio_path).unlink(missing_ok=True)
            Path(video_path).unlink(missing_ok=True)

        refs = result["upstream_files"]
        self.assertEqual([ref["type"] for ref in refs], ["audio", "video"])
        self.assertEqual([ref["showType"] for ref in refs], ["audio", "video"])
        self.assertEqual([ref["file_class"] for ref in refs], ["audio", "video"])
        # 音频 / 视频不能带 context="full"（那是文档 parse 才需要的字段）
        self.assertNotIn("context", refs[0])
        self.assertNotIn("context", refs[1])
        self.assertTrue(refs[0]["url"].startswith("data:audio/mpeg;base64,"))
        self.assertTrue(refs[1]["url"].startswith("data:video/mp4;base64,"))
        uploader.upload_local_file.assert_not_awaited()

    async def test_document_still_uses_upstream_upload(self) -> None:
        uploaded: list[str] = []

        async def upload_local_file(acc, local_meta):
            uploaded.append(local_meta["filename"])
            return {"remote_ref": {"file_id": "remote-doc", "filename": local_meta["filename"]}}

        app, _ = self._build_app(uploader=SimpleNamespace(upload_local_file=upload_local_file))
        document, document_path = self._attachment("notes.pdf", "application/pdf", b"%PDF-1.4")
        image, image_path = self._attachment("a.png", "image/png", b"png-bytes")
        try:
            result = await prepare_context_attachments(
                app=app,
                payload=self._payload(),
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:mixed",
                existing_attachments=[document, image],
            )
        finally:
            Path(document_path).unlink(missing_ok=True)
            Path(image_path).unlink(missing_ok=True)

        self.assertEqual(uploaded, ["notes.pdf"])
        types = [ref.get("type") for ref in result["upstream_files"]]
        self.assertIn("image", types)
        self.assertIn("remote-doc", [ref.get("file_id") for ref in result["upstream_files"]])
        self.assertIsNotNone(result["bound_account"])

    async def test_inline_disabled_falls_back_to_upload(self) -> None:
        uploaded: list[str] = []

        async def upload_local_file(acc, local_meta):
            uploaded.append(local_meta["filename"])
            return {"remote_ref": {"file_id": "remote-img", "filename": local_meta["filename"]}}

        app, _ = self._build_app(
            uploader=SimpleNamespace(upload_local_file=upload_local_file),
            inline_enabled=False,
        )
        attachment, local_path = self._attachment("a.png", "image/png", b"png-bytes")
        try:
            result = await prepare_context_attachments(
                app=app,
                payload=self._payload(),
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:fallback",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(uploaded, ["a.png"])
        self.assertEqual(result["upstream_files"][0]["file_id"], "remote-img")

    async def test_oversized_inline_media_falls_back_to_upload(self) -> None:
        uploaded: list[str] = []

        async def upload_local_file(acc, local_meta):
            uploaded.append(local_meta["filename"])
            return {"remote_ref": {"file_id": "remote-img", "filename": local_meta["filename"]}}

        app, _ = self._build_app(
            uploader=SimpleNamespace(upload_local_file=upload_local_file),
            max_bytes=4,
        )
        attachment, local_path = self._attachment("a.png", "image/png", b"png-bytes")
        try:
            result = await prepare_context_attachments(
                app=app,
                payload=self._payload(),
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:oversize",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(uploaded, ["a.png"])
        self.assertEqual(result["upstream_files"][0]["file_id"], "remote-img")

    async def test_missing_local_file_falls_back_to_upload(self) -> None:
        uploaded: list[str] = []

        async def upload_local_file(acc, local_meta):
            uploaded.append(local_meta["filename"])
            return {"remote_ref": {"file_id": "remote-img", "filename": local_meta["filename"]}}

        app, _ = self._build_app(uploader=SimpleNamespace(upload_local_file=upload_local_file))
        attachment = NormalizedAttachment(
            file_id="local-2",
            filename="gone.png",
            content_type="image/png",
            source="inline-image",
            local_path="/nonexistent/gone.png",
            sha256="abc",
            purpose="user-upload",
        )

        result = await prepare_context_attachments(
            app=app,
            payload=self._payload(),
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:missing",
            existing_attachments=[attachment],
        )

        self.assertEqual(uploaded, ["gone.png"])
        self.assertEqual(result["upstream_files"][0]["file_id"], "remote-img")


class _FakeBrowserProxy:
    """浏览器上传代理替身：记录调用，可注入慢调用 / 异常（不启动真浏览器）。"""

    def __init__(self, *, sts=None, delay: float = 0.0, error: Exception | None = None) -> None:
        self.sts = dict(sts or {"file_id": "sts-1", "file_path": "ctx/file.txt"})
        self.delay = float(delay)
        self.error = error
        self.fetch_calls: list[tuple[str, str, str, int]] = []
        self.hard_timeout_notes: list[str] = []

    async def fetch_sts(self, account, filename, content_type, size):
        self.fetch_calls.append((getattr(account, "email", ""), filename, content_type, int(size)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return dict(self.sts)

    def note_hard_timeout(self, email: str) -> None:
        self.hard_timeout_notes.append(str(email or ""))


class BrowserUploadScopeTests(unittest.IsolatedAsyncioTestCase):
    """浏览器上传的作用域 + 硬超时。

    核心约束（2026-11 生产事故回归）：浏览器上传只服务**调用方真实附件**；
    上下文卸载生成的 `plan.generated_files` 永远走 HTTP 上传，不许起浏览器。
    """

    def _build_app(
        self,
        *,
        uploader,
        browser_proxy,
        settings_overrides: dict | None = None,
    ):
        settings_kwargs = {
            "CONTEXT_INLINE_MAX_CHARS": 80,
            "CONTEXT_FORCE_FILE_MAX_CHARS": 160,
            "CONTEXT_ATTACHMENT_TTL_SECONDS": 600,
            "INLINE_ATTACHMENT_ENABLED": True,
            "INLINE_ATTACHMENT_MAX_BYTES": 10 * 1024 * 1024,
            # 显式打开浏览器上传：这些用例验证的是「作用域」，不是「默认值」。
            "UPLOAD_VIA_BROWSER": "auto",
            "UPLOAD_CDP_ENDPOINT": "http://10.6.6.4:3000",
            "UPLOAD_BROWSER_MODE": "auto",
            "UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS": 45,
        }
        settings_kwargs.update(settings_overrides or {})
        saved_texts: list[str] = []

        async def save_text(filename, text, content_type, purpose):
            saved_texts.append(text)
            return {
                "id": filename,
                "path": f"/tmp/{filename}",
                "filename": filename,
                "content_type": content_type,
                "sha256": "sha-ctx",
                "created_at": 1,
            }

        app = SimpleNamespace(state=SimpleNamespace(
            context_offloader=ContextOffloader(SimpleNamespace(**settings_kwargs)),
            account_pool=SimpleNamespace(
                acquire_wait=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                acquire_wait_preferred=AsyncMock(return_value=SimpleNamespace(email="bot@example.com")),
                release=lambda _acc: None,
                mark_file_upload_limited=lambda *a, **k: None,
                file_upload_blocked_emails=lambda: set(),
                save=AsyncMock(),
            ),
            file_store=SimpleNamespace(save_text=save_text, delete_path=AsyncMock()),
            session_affinity=SimpleNamespace(
                get=AsyncMock(return_value=None),
                bind_account=AsyncMock(),
                add_uploaded_file=AsyncMock(),
            ),
            upstream_file_cache=SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock()),
            upstream_file_uploader=uploader,
            browser_upload_proxy=browser_proxy,
        ))
        return app, saved_texts

    def _generated_context_payload(self) -> dict:
        """会触发上下文卸载（`plan.generated_files` 非空）的带 tools 请求。"""
        return {
            "model": "qwen3.8-max",
            "system": "Always answer as a pirate captain.",
            "messages": [
                {"role": "assistant", "content": "prior result " * 10},
                {"role": "user", "content": "Please analyze this current input.\n" + "runtime line\n" * 20},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read file contents",
                        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    },
                }
            ],
        }

    def _document_attachment(self) -> tuple[NormalizedAttachment, str]:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        handle.write(b"user supplied document bytes")
        handle.close()
        attachment = NormalizedAttachment(
            file_id="local-doc",
            filename="notes.txt",
            content_type="text/plain",
            source="manual-upload",
            local_path=handle.name,
            sha256="sha-doc",
            purpose="user-upload",
        )
        return attachment, handle.name

    async def test_generated_context_files_never_use_browser_upload(self) -> None:
        uploaded: list[str] = []

        async def upload_local_file(_acc, local_meta):
            uploaded.append(local_meta["filename"])
            return {"remote_ref": {"file_id": "file-ctx", "filename": local_meta["filename"]}}

        uploader = SimpleNamespace(
            upload_local_file=upload_local_file,
            put_bytes=AsyncMock(return_value={"remote_ref": {"file_id": "browser-ctx"}}),
        )
        browser_proxy = _FakeBrowserProxy()
        app, saved_texts = self._build_app(uploader=uploader, browser_proxy=browser_proxy)

        result = await prepare_context_attachments(
            app=app,
            payload=self._generated_context_payload(),
            surface="openai",
            auth_token="tok",
            client_profile=OPENCLAW_OPENAI_PROFILE,
            session_key="openai:generated",
        )

        self.assertEqual(result["context_mode"], "file")
        # 生成文件数量取决于 offload plan（历史 / tools 各一份），这里只关心「走了哪条上传链路」。
        self.assertGreaterEqual(len(saved_texts), 1)
        self.assertEqual(len(uploaded), len(saved_texts))
        # 生成文件绝不能碰浏览器：既不取 STS，也不 PUT。
        self.assertEqual(browser_proxy.fetch_calls, [])
        uploader.put_bytes.assert_not_awaited()
        self.assertEqual(browser_proxy.hard_timeout_notes, [])
        self.assertEqual(len(result["upstream_files"]), len(saved_texts))

    async def test_user_attachment_uses_browser_upload_when_enabled(self) -> None:
        uploader = SimpleNamespace(
            upload_local_file=AsyncMock(return_value={"remote_ref": {"file_id": "http-doc"}}),
            put_bytes=AsyncMock(return_value={"remote_ref": {"file_id": "browser-doc", "filename": "notes.txt"}}),
        )
        browser_proxy = _FakeBrowserProxy()
        app, _ = self._build_app(uploader=uploader, browser_proxy=browser_proxy)
        attachment, local_path = self._document_attachment()
        try:
            result = await prepare_context_attachments(
                app=app,
                payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "read this"}]},
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:user-attachment",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        self.assertEqual(len(browser_proxy.fetch_calls), 1)
        self.assertEqual(browser_proxy.fetch_calls[0][1], "notes.txt")
        uploader.put_bytes.assert_awaited()
        uploader.upload_local_file.assert_not_awaited()
        self.assertEqual(result["upstream_files"][0]["file_id"], "browser-doc")

    async def test_hard_timeout_degrades_without_raising(self) -> None:
        """慢调用（0.5s）> 硬超时（0.05s）：降级到 HTTP 上传，绝不抛到上层。"""
        uploader = SimpleNamespace(
            upload_local_file=AsyncMock(return_value={"remote_ref": {"file_id": "http-doc"}}),
            put_bytes=AsyncMock(return_value={"remote_ref": {"file_id": "browser-doc"}}),
        )
        browser_proxy = _FakeBrowserProxy(delay=0.5)
        app, _ = self._build_app(
            uploader=uploader,
            browser_proxy=browser_proxy,
            settings_overrides={"UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS": 0.05},
        )
        attachment, local_path = self._document_attachment()
        started = time.monotonic()
        try:
            result = await prepare_context_attachments(
                app=app,
                payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "read this"}]},
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:hard-timeout",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)
        elapsed = time.monotonic() - started

        # 超时真的生效：整体耗时被硬超时卡住，而不是等满 0.5s 的慢调用。
        self.assertLess(elapsed, 0.45)
        self.assertEqual(len(browser_proxy.fetch_calls), 1)
        uploader.put_bytes.assert_not_awaited()
        uploader.upload_local_file.assert_awaited()
        # 超时按 BrowserUploadUnavailable 语义降级 + 立刻熔断该账号。
        self.assertEqual(browser_proxy.hard_timeout_notes, ["bot@example.com"])
        self.assertEqual(result["upstream_files"][0]["file_id"], "http-doc")
        self.assertFalse(result["attachment_fallback"])

    async def test_unavailable_degrades_without_raising(self) -> None:
        from backend.services.browser_upload_proxy import BrowserUploadUnavailable

        uploader = SimpleNamespace(
            upload_local_file=AsyncMock(return_value={"remote_ref": {"file_id": "http-doc"}}),
            put_bytes=AsyncMock(),
        )
        browser_proxy = _FakeBrowserProxy(error=BrowserUploadUnavailable("no browser endpoint"))
        app, _ = self._build_app(uploader=uploader, browser_proxy=browser_proxy)
        attachment, local_path = self._document_attachment()
        try:
            result = await prepare_context_attachments(
                app=app,
                payload={"model": "qwen3.8-max", "messages": [{"role": "user", "content": "read this"}]},
                surface="openai",
                auth_token="tok",
                client_profile=OPENCLAW_OPENAI_PROFILE,
                session_key="openai:unavailable",
                existing_attachments=[attachment],
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        uploader.put_bytes.assert_not_awaited()
        uploader.upload_local_file.assert_awaited()
        self.assertEqual(result["upstream_files"][0]["file_id"], "http-doc")


class UploadViaBrowserDefaultTests(unittest.TestCase):
    """`UPLOAD_VIA_BROWSER` 默认必须是 `never`（默认不启浏览器）。"""

    def test_config_default_is_never(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        script = (
            "import json;"
            "from backend.core.config import Settings;"
            "from backend.services.browser_upload_proxy import BrowserUploadProxy, resolve_browser_mode;"
            "s = Settings(_env_file=None);"
            "print(json.dumps({"
            "'config': s.UPLOAD_VIA_BROWSER,"
            "'mode': resolve_browser_mode(s),"
            "'hard_timeout': s.UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS,"
            "'enabled': BrowserUploadProxy(s, calibration_path='/tmp/_cal_default_test.json').enabled,"
            "}))"
        )
        # 清掉可能影响默认值的外部环境变量，并在临时目录里跑（避免读到仓库根的 .env）。
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"UPLOAD_VIA_BROWSER", "UPLOAD_CDP_ENDPOINT", "UPLOAD_BROWSER_MODE"}
        }
        env["PYTHONPATH"] = str(repo_root)
        with tempfile.TemporaryDirectory() as tmp_dir:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=tmp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["config"], "never")
        self.assertEqual(payload["mode"], "off")
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["hard_timeout"], 45)

    def test_missing_setting_falls_back_to_never(self) -> None:
        from backend.services.browser_upload_proxy import BROWSER_MODE_OFF, resolve_browser_mode
        from backend.services.context_attachment_manager import _browser_upload_enabled

        # 配置对象里没有 UPLOAD_VIA_BROWSER 这个属性时，也必须按「关闭」处理。
        bare = SimpleNamespace(UPLOAD_CDP_ENDPOINT="http://10.6.6.4:3000")
        self.assertEqual(resolve_browser_mode(bare), BROWSER_MODE_OFF)
        self.assertFalse(_browser_upload_enabled(bare, object()))

    def test_hard_timeout_defaults_to_45(self) -> None:
        from backend.services.context_attachment_manager import _browser_hard_timeout_seconds

        self.assertEqual(_browser_hard_timeout_seconds(SimpleNamespace()), 45.0)
        self.assertEqual(
            _browser_hard_timeout_seconds(SimpleNamespace(UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS=8)),
            8.0,
        )
        # 非法 / 非正值退回默认，绝不让「0 秒超时」把所有上传瞬间打死。
        self.assertEqual(
            _browser_hard_timeout_seconds(SimpleNamespace(UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS="oops")),
            45.0,
        )
        self.assertEqual(
            _browser_hard_timeout_seconds(SimpleNamespace(UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS=0)),
            45.0,
        )


if __name__ == "__main__":
    unittest.main()
