import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

from backend.core.account_pool import Account
from backend.services.chat_id_pool import ChatIDPool, WarmChat, warm_chat_key


class ChatIDPoolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.settings_patch = patch.multiple(
            "backend.services.chat_id_pool.settings",
            CHAT_ID_PREWARM_TARGET_PER_ACCOUNT=2,
            CHAT_ID_PREWARM_TTL_SECONDS=120,
            CHAT_ID_PREWARM_MAX_CONCURRENCY=2,
            CHAT_ID_PREWARM_MODELS="",
        )
        self.settings_patch.start()

    async def asyncTearDown(self) -> None:
        self.settings_patch.stop()

    async def test_fill_creates_missing_warm_chats_and_take_reuses_one(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        account.valid = True
        pool = SimpleNamespace(accounts=[account], get_by_email=lambda email: account)
        created = []

        async def fake_create_chat(acc, model, chat_type="t2t"):
            created.append((acc.email, model, chat_type))
            return f"chat-{len(created)}"

        client = SimpleNamespace(executor=SimpleNamespace(create_chat=fake_create_chat), delete_chat=AsyncMock())
        chat_pool = ChatIDPool(client, pool)

        await chat_pool.remember_model("qwen3.7-plus", "t2t")
        if chat_pool._fill_task is not None:
            await chat_pool._fill_task

        self.assertEqual(await chat_pool.count(account.email, "qwen3.7-plus", "t2t"), 2)
        chat_id, reused = await chat_pool.take(account.email, "qwen3.7-plus", "t2t")

        self.assertEqual(chat_id, "chat-1")
        self.assertTrue(reused)
        self.assertEqual(await chat_pool.count(account.email, "qwen3.7-plus", "t2t"), 1)

    async def test_cleanup_removes_expired_chats_and_deletes_upstream(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = SimpleNamespace(accounts=[account], get_by_email=lambda email: account)
        client = SimpleNamespace(executor=SimpleNamespace(), delete_chat=AsyncMock())
        chat_pool = ChatIDPool(client, pool)
        key = warm_chat_key(account.email, "qwen3.7-plus", "t2t")
        chat_pool._items[key] = [WarmChat(account.email, account.token, "qwen3.7-plus", "t2t", "chat-old", 1.0)]

        with patch("backend.services.chat_id_pool.time.time", return_value=1000.0):
            expired = await chat_pool.cleanup(delete_all=False)

        self.assertEqual([item.chat_id for item in expired], ["chat-old"])
        self.assertEqual(await chat_pool.count(account.email, "qwen3.7-plus", "t2t"), 0)

    async def test_take_reused_chat_does_not_trigger_immediate_refill(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = SimpleNamespace(accounts=[account], get_by_email=lambda email: account)
        client = SimpleNamespace(executor=SimpleNamespace(), delete_chat=AsyncMock())
        chat_pool = ChatIDPool(client, pool)
        key = warm_chat_key(account.email, "qwen3.7-plus", "t2t")
        chat_pool._items[key] = [WarmChat(account.email, account.token, "qwen3.7-plus", "t2t", "chat-1", 100.0)]

        with patch.object(chat_pool, "trigger_fill") as trigger_fill:
            with patch("backend.services.chat_id_pool.time.time", return_value=101.0):
                chat_id, reused = await chat_pool.take(account.email, "qwen3.7-plus", "t2t")

        self.assertEqual(chat_id, "chat-1")
        self.assertTrue(reused)
        trigger_fill.assert_not_called()

    async def test_fill_uses_configured_prewarm_models_before_first_request(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = SimpleNamespace(accounts=[account], get_by_email=lambda email: account)
        created = []

        async def fake_create_chat(acc, model, chat_type="t2t"):
            created.append((acc.email, model, chat_type))
            return f"chat-{len(created)}"

        client = SimpleNamespace(executor=SimpleNamespace(create_chat=fake_create_chat), delete_chat=AsyncMock())

        with patch("backend.services.chat_id_pool.settings.CHAT_ID_PREWARM_MODELS", "qwen3.7-plus,qwen3.6-plus"):
            chat_pool = ChatIDPool(client, pool)
            await chat_pool.fill()

        self.assertEqual(await chat_pool.count(account.email, "qwen3.7-plus", "t2t"), 2)
        self.assertEqual(await chat_pool.count(account.email, "qwen3.6-plus", "t2t"), 2)
        self.assertEqual(
            created,
            [
                (account.email, "qwen3.7-plus", "t2t"),
                (account.email, "qwen3.7-plus", "t2t"),
                (account.email, "qwen3.6-plus", "t2t"),
                (account.email, "qwen3.6-plus", "t2t"),
            ],
        )

    async def test_fill_skips_busy_accounts(self) -> None:
        busy = Account(email="busy@example.com", token="token-1")
        busy.inflight = 1
        idle = Account(email="idle@example.com", token="token-2")
        pool = SimpleNamespace(accounts=[busy, idle], max_inflight=1, get_by_email=lambda email: idle if email == idle.email else busy)
        created = []

        async def fake_create_chat(acc, model, chat_type="t2t"):
            created.append(acc.email)
            return f"chat-{len(created)}"

        client = SimpleNamespace(executor=SimpleNamespace(create_chat=fake_create_chat), delete_chat=AsyncMock())
        chat_pool = ChatIDPool(client, pool)
        await chat_pool.remember_model("qwen3.7-plus", "t2t")
        if chat_pool._fill_task is not None:
            await chat_pool._fill_task

        self.assertEqual(created, [idle.email, idle.email])


if __name__ == "__main__":
    unittest.main()
