import asyncio
import hashlib
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
    async def test_create_warm_chat_sleeps_deterministic_jitter_before_request(self) -> None:
        """_create_warm_chat 应在发请求前根据 email+model+chat_type 哈希产生确定性抖动延迟。"""
        client = MagicMock()
        client.executor = MagicMock()
        client.executor.create_chat = AsyncMock(return_value="chat-jitter-test")

        account_pool = MagicMock()
        account_pool.max_inflight = 1
        account_pool.accounts = []

        pool = ChatIDPool(client, account_pool)

        acc = _FakeAccount(email="jitter@example.com")
        semaphore = asyncio.Semaphore(1)

        # 计算期望的确定性抖动值
        jitter_key = f"{acc.email}|qwen3.7-plus|t2t"
        expected_hash = int(hashlib.sha256(jitter_key.encode()).hexdigest(), 16)
        expected_jitter = (expected_hash % 2000) / 1000.0  # 0~2s 范围

        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def capture_sleep(duration: float) -> None:
            sleep_calls.append(duration)
            # 不真正等待，避免测试变慢
            await original_sleep(0)

        with patch("backend.services.chat_id_pool.asyncio.sleep", side_effect=capture_sleep):
            await pool._create_warm_chat(semaphore, acc, "qwen3.7-plus", "t2t")

        # 必须至少有一次 sleep 调用
        self.assertTrue(len(sleep_calls) >= 1, "Expected at least one asyncio.sleep call for jitter")
        # 第一次 sleep 应该是确定性抖动（允许浮点误差）
        self.assertAlmostEqual(sleep_calls[0], expected_jitter, places=2)

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

    async def test_same_account_model_produces_same_jitter(self) -> None:
        """相同账号+模型组合的抖动延迟应该一致（确定性）。"""
        client = MagicMock()
        client.executor = MagicMock()
        client.executor.create_chat = AsyncMock(return_value="chat-y")

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
            await pool._create_warm_chat(semaphore, _FakeAccount("same@example.com"), "qwen3.7-plus", "t2t")
            await pool._create_warm_chat(semaphore, _FakeAccount("same@example.com"), "qwen3.7-plus", "t2t")

        self.assertEqual(len(jitters), 2)
        self.assertAlmostEqual(jitters[0], jitters[1], places=4)


if __name__ == "__main__":
    unittest.main()
