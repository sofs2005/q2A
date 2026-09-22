import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

from backend.core.config import settings
from backend.services.chat_id_pool import ChatIDPool


class _FakeAccount:
    def __init__(self, email: str, token: str = "tok") -> None:
        self.email = email
        self.token = token

    def is_available(self) -> bool:
        return True

    def next_available_at(self) -> float:
        return 0.0


class ChatIDPoolJitterTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_warm_chat_sleeps_bounded_jitter_before_request(self) -> None:
        """_create_warm_chat 应在发请求前产生落在 JITTER_SECONDS 内的抖动延迟。

        抖动刻意参与时间熵（不再是纯 email+model 哈希）：纯确定性哈希意味着同一
        账号同一模型每次预热的等待时间完全一致，长期看没有随机性可言。
        """
        client = MagicMock()
        client.executor = MagicMock()
        client.executor.create_chat = AsyncMock(return_value="chat-jitter-test")

        account_pool = MagicMock()
        account_pool.max_inflight = 1
        account_pool.accounts = []

        pool = ChatIDPool(client, account_pool)

        acc = _FakeAccount(email="jitter@example.com")
        semaphore = asyncio.Semaphore(1)

        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def capture_sleep(duration: float) -> None:
            sleep_calls.append(duration)
            # 不真正等待，避免测试变慢
            await original_sleep(0)

        jitter_max = float(getattr(settings, "CHAT_ID_PREWARM_JITTER_SECONDS", 1.5) or 0)
        with patch("backend.services.chat_id_pool.asyncio.sleep", side_effect=capture_sleep):
            await pool._create_warm_chat(semaphore, acc, "qwen3.7-plus", "t2t")

        # 必须至少有一次 sleep 调用
        self.assertTrue(len(sleep_calls) >= 1, "Expected at least one asyncio.sleep call for jitter")
        # 首次 sleep 为抖动（seq=0/total=1 时错峰项为 0），必须落在抖动上限内
        self.assertGreater(sleep_calls[0], 0.0)
        self.assertLessEqual(sleep_calls[0], jitter_max)

    async def test_different_accounts_produce_different_jitter(self) -> None:
        """不同账号的抖动延迟应该不同，避免所有请求挤在同一时刻。"""
        client = MagicMock()
        client.executor = MagicMock()
        client.executor.create_chat = AsyncMock(return_value="chat-x")

        account_pool = MagicMock()
        account_pool.max_inflight = 1
        account_pool.accounts = []

        pool = ChatIDPool(client, account_pool)
        semaphore = asyncio.Semaphore(1)

        jitters: list[float] = []
        original_sleep = asyncio.sleep

        async def capture_sleep(duration: float) -> None:
            jitters.append(duration)
            await original_sleep(0)

        with patch("backend.services.chat_id_pool.asyncio.sleep", side_effect=capture_sleep):
            await pool._create_warm_chat(semaphore, _FakeAccount("alice@example.com"), "qwen3.7-plus", "t2t")
            await pool._create_warm_chat(semaphore, _FakeAccount("bob@example.com"), "qwen3.7-plus", "t2t")
            await pool._create_warm_chat(semaphore, _FakeAccount("carol@example.com"), "qwen3.7-max", "t2t")

        # 三次调用的抖动值应该不完全相同
        unique_jitters = set(round(j, 3) for j in jitters)
        self.assertGreater(len(unique_jitters), 1, "Different accounts should produce different jitter values")

    async def test_spread_delay_grows_with_request_sequence(self) -> None:
        """错峰必须按「本轮第几个请求」铺开，而不是按账号序位。

        同一账号可能同时有 TARGET × 模型数 个待建会话，若只按账号分档，它们会拿到
        完全相同的 slot，于是同时发出 —— 单账号并发脉冲比分散还显眼。
        """
        delays = [
            ChatIDPool._spread_delay(seq, 4, "same@example.com", "qwen3.7-plus", "t2t")
            for seq in range(4)
        ]

        # 序号越大，起跑时间越晚（严格递增，抖动不改变单调性）
        for earlier, later in zip(delays, delays[1:]):
            self.assertLess(earlier, later)

    async def test_jitter_uses_time_entropy(self) -> None:
        """抖动必须带时间熵：纯确定性哈希下同一账号每次预热的等待时间完全相同。"""
        jitter_max = float(getattr(settings, "CHAT_ID_PREWARM_JITTER_SECONDS", 1.5) or 0)
        first = ChatIDPool._jitter("same@example.com", "qwen3.7-plus", "t2t")

        with patch("backend.services.chat_id_pool.time.time", return_value=1_700_000_001.0):
            second = ChatIDPool._jitter("same@example.com", "qwen3.7-plus", "t2t")

        for value in (first, second):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, jitter_max)
        # 不同时间片给出不同抖动
        self.assertNotAlmostEqual(first, second, places=6)


if __name__ == "__main__":
    unittest.main()
