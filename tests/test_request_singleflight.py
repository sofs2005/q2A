import asyncio
import unittest
from types import SimpleNamespace

from backend.api import v1_chat
from backend.toolcore.request_singleflight import RequestSingleflight


class RequestSingleflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_joiner_waits_for_owner_result(self) -> None:
        singleflight = RequestSingleflight(now=lambda: 100.0)
        key = ("openai", "session", "model", "prompt")

        owner_entry, is_owner, cached = await singleflight.start_or_join(key, owner_id="req-1")
        join_entry, join_is_owner, join_cached = await singleflight.start_or_join(key, owner_id="req-2")

        self.assertTrue(is_owner)
        self.assertIsNone(cached)
        self.assertFalse(join_is_owner)
        self.assertIsNone(join_cached)
        self.assertIs(join_entry, owner_entry)

        waiter = asyncio.ensure_future(join_entry.future)
        await singleflight.complete(key, {"id": "chatcmpl-1"})

        self.assertEqual(await waiter, {"id": "chatcmpl-1"})

    async def test_different_key_gets_new_owner(self) -> None:
        singleflight = RequestSingleflight(now=lambda: 100.0)

        await singleflight.start_or_join(("session", "prompt-a"), owner_id="req-1")
        _entry, is_owner, cached = await singleflight.start_or_join(("session", "prompt-b"), owner_id="req-2")

        self.assertTrue(is_owner)
        self.assertIsNone(cached)

    async def test_recent_completed_result_is_reused(self) -> None:
        current_time = 100.0
        singleflight = RequestSingleflight(result_ttl_seconds=30.0, now=lambda: current_time)
        key = ("session", "prompt")

        await singleflight.start_or_join(key, owner_id="req-1")
        await singleflight.complete(key, {"id": "chatcmpl-1"})
        entry, is_owner, cached = await singleflight.start_or_join(key, owner_id="req-2")

        self.assertIsNone(entry)
        self.assertFalse(is_owner)
        self.assertEqual(cached, {"id": "chatcmpl-1"})

    def test_openai_context_fingerprint_empty_for_plain_inline_request(self) -> None:
        payload = {
            "model": "gpt-4.1",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        self.assertEqual(v1_chat._build_openai_context_fingerprint(req_data=payload), "")
        self.assertEqual(
            v1_chat._build_openai_context_fingerprint(
                req_data=payload,
                context_prepared={
                    "context_mode": "inline",
                    "upstream_files": [],
                    "generated_local_files": [],
                    "attachment_fallback": False,
                    "context_attachment_tokens": 0,
                },
            ),
            "",
        )

    def test_openai_json_singleflight_key_includes_context_fingerprint(self) -> None:
        request = SimpleNamespace(
            stream=False,
            client_profile="openclaw_openai",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            tool_names=["Read"],
            tool_choice_mode="auto",
            required_tool_name=None,
            tool_choice_raw=None,
            context_fingerprint="",
        )
        diagnostics_a = {
            "prompt_hash": "prompt-hash",
            "latest_user_hash": "user-hash",
            "context_fingerprint": "ctx-a",
        }
        diagnostics_b = {
            "prompt_hash": "prompt-hash",
            "latest_user_hash": "user-hash",
            "context_fingerprint": "ctx-b",
        }

        key_a = v1_chat._build_openai_json_singleflight_key(
            standard_request=request,
            diagnostics=diagnostics_a,
        )
        key_b = v1_chat._build_openai_json_singleflight_key(
            standard_request=request,
            diagnostics=diagnostics_b,
        )

        self.assertNotEqual(key_a, key_b)
        self.assertEqual(key_a[-1], "ctx-a")
        self.assertEqual(key_b[-1], "ctx-b")

    def test_openai_json_singleflight_key_does_not_depend_on_session_key(self) -> None:
        diagnostics = {
            "prompt_hash": "prompt-hash",
            "latest_user_hash": "user-hash",
            "context_fingerprint": "ctx-hash",
        }
        request_a = SimpleNamespace(
            stream=False,
            client_profile="openclaw_openai",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            tool_names=["Read"],
            tool_choice_mode="auto",
            required_tool_name=None,
            tool_choice_raw=None,
            context_fingerprint="ctx-hash",
            session_key="openai:req_a",
        )
        request_b = SimpleNamespace(
            stream=False,
            client_profile="openclaw_openai",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            tool_names=["Read"],
            tool_choice_mode="auto",
            required_tool_name=None,
            tool_choice_raw=None,
            context_fingerprint="ctx-hash",
            session_key="openai:req_b",
        )

        key_a = v1_chat._build_openai_json_singleflight_key(
            standard_request=request_a,
            diagnostics=diagnostics,
        )
        key_b = v1_chat._build_openai_json_singleflight_key(
            standard_request=request_b,
            diagnostics=diagnostics,
        )

        self.assertEqual(key_a, key_b)
        self.assertNotIn("openai:req_a", key_a)
        self.assertNotIn("openai:req_b", key_b)
        self.assertEqual(key_a[-1], "ctx-hash")


if __name__ == "__main__":
    unittest.main()
