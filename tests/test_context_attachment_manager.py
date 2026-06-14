import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


if __name__ == "__main__":
    unittest.main()
