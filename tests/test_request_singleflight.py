import asyncio
import unittest

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


if __name__ == "__main__":
    unittest.main()
