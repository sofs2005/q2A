import asyncio
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

    async def test_prewarm_waf_blocked_marks_cookie_expired_and_cools_account(self) -> None:
        """预热撞 aliyun_waf 时必须作废 acw_tc 并短冷却，避免下一轮继续砸同一坏 cookie。"""
        account = Account(email="alice@example.com", token="token-1")
        account.waf_cookies = "acw_tc=stale"
        account.waf_cookies_expires_at = 9999999999.0
        account.valid = True

        rate_limited = []

        def mark_rate_limited(acc, cooldown=None, error_message=""):
            rate_limited.append({"acc": acc, "cooldown": cooldown, "error": error_message})
            acc.rate_limited_until = 1000.0

        pool = SimpleNamespace(
            accounts=[account],
            get_by_email=lambda email: account,
            max_inflight=1,
            mark_rate_limited=mark_rate_limited,
        )

        async def fake_create_chat(acc, model, chat_type="t2t"):
            raise Exception("waf_blocked: create_chat returned WAF page: aliyun_waf_aa")

        client = SimpleNamespace(
            executor=SimpleNamespace(create_chat=fake_create_chat, auth_resolver=None),
            delete_chat=AsyncMock(),
        )
        chat_pool = ChatIDPool(client, pool)

        with (
            patch("backend.services.chat_id_pool.settings.WAF_RETRY_EXTRA_COOLDOWN_SECONDS", 5),
            patch("backend.services.chat_id_pool.asyncio.sleep", new=AsyncMock()),
        ):
            await chat_pool._create_warm_chat(asyncio.Semaphore(1), account, "qwen3.8-max-preview", "t2t")

        self.assertEqual(account.waf_cookies_expires_at, 0)
        self.assertEqual(account.waf_cookies, "")
        self.assertEqual(len(rate_limited), 1)
        self.assertIs(rate_limited[0]["acc"], account)
        self.assertEqual(rate_limited[0]["cooldown"], 5)
        self.assertIn("waf", rate_limited[0]["error"].lower())

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
