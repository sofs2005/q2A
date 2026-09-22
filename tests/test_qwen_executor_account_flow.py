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

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

from backend.core.account_pool import Account
from backend.upstream.qwen_executor import (
    QwenExecutor,
    _is_punish_error,
    _is_waf_blocked_body,
    _preview_text,
    _retry_backoff_seconds,
)


class _Pool:
    def release(self, acc):
        self.released = acc


class _RetryPool:
    def __init__(self, account):
        self.account = account
        self.invalidated = []
        self.released = []
        self.excludes = []

    async def acquire_wait(self, timeout=60, exclude=None):
        self.excludes.append(set(exclude or set()))
        if not self.excludes[-1]:
            return self.account
        return None

    def mark_invalid(self, acc):
        self.invalidated.append(acc)

    def mark_rate_limited(self, acc):
        self.rate_limited = acc

    def release(self, acc):
        self.released.append(acc)


class QwenExecutorAccountFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_preferred_account_is_used_first_attempt(self) -> None:
        """preferred_account 应在首次尝试时被使用，且 chat_id 来自预热池。"""
        account = Account(email="alice@example.com", token="token-1")
        account.fingerprint_id = "chrome146_windows"
        chat_pool = SimpleNamespace(
            remember_model=AsyncMock(),
            take=AsyncMock(return_value=("warm-chat-1", True)),
        )
        pool = _RetryPool(account)
        executor = QwenExecutor(SimpleNamespace(chat_id_pool=chat_pool), pool)
        seen = []

        async def fake_stream(acc, chat_id, model, content, has_custom_tools=False, files=None, chat_type="t2t", media_options=None):
            seen.append(("stream", acc, chat_id, model, content))
            if False:
                yield None
            return
            yield

        executor.stream = fake_stream

        events = []
        async for item in executor.chat_stream_events_with_retry("gpt-4o", "hello", preferred_account=account):
            events.append(item)

        self.assertEqual(events[0]["type"], "meta")
        self.assertIs(events[0]["acc"], account)
        self.assertEqual(events[0]["chat_id"], "warm-chat-1")
        chat_pool.take.assert_awaited_once_with("alice@example.com", "gpt-4o", "t2t")
        self.assertEqual(seen[0][0], "stream")
        self.assertIs(seen[0][1], account)
        self.assertEqual(seen[0][2], "warm-chat-1")

    async def test_create_chat_reports_waf_blocked_for_aliyun_waf_html(self) -> None:
        account = Account(email="alice@example.com", token="token-1")

        async def fake_request(method, path, token, body=None, timeout=None, account=None, **kwargs):
            return {
                "status": 200,
                "body": '<!doctypehtml><meta name="aliyun_waf_aa" content="blocked">',
            }

        executor = QwenExecutor(SimpleNamespace(_request_json=fake_request), _Pool())

        with self.assertRaisesRegex(Exception, "waf_blocked"):
            await executor.create_chat(account, "qwen3.7-plus")

    async def test_create_chat_enables_retry_waf_on_request(self) -> None:
        """create_chat 必须开 retry_waf，才能在建 chat 前/撞挑战后刷新 acw_tc。"""
        account = Account(email="alice@example.com", token="token-1")
        seen = {}

        async def fake_request(method, path, token, body=None, timeout=None, account=None, **kwargs):
            seen["kwargs"] = kwargs
            seen["path"] = path
            return {
                "status": 200,
                "body": '{"success": true, "data": {"id": "chat-waf-1"}}',
                "acw_tc": "fresh-acw",
            }

        executor = QwenExecutor(SimpleNamespace(_request_json=fake_request), _Pool())
        chat_id = await executor.create_chat(account, "qwen3.8-max")

        self.assertEqual(chat_id, "chat-waf-1")
        self.assertEqual(seen["path"], "/api/v2/chats/new")
        self.assertTrue(seen["kwargs"].get("retry_waf"))
        self.assertTrue(seen["kwargs"].get("chat_transport"))

    async def test_retry_does_not_mark_account_invalid_for_waf_blocked(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = _RetryPool(account)
        executor = QwenExecutor(SimpleNamespace(), pool)
        executor.auth_resolver.auto_heal_account = AsyncMock()

        async def fake_create_chat(acc, model, chat_type="t2t"):
            raise Exception("waf_blocked: aliyun_waf_aa")

        executor.create_chat = fake_create_chat

        with patch("backend.upstream.qwen_executor.settings.MAX_RETRIES", 1):
            with self.assertRaisesRegex(Exception, "All 1 attempts failed"):
                async for _ in executor.chat_stream_events_with_retry("qwen3.7-plus", "hello"):
                    pass

        self.assertEqual(pool.invalidated, [])
        self.assertEqual(pool.released, [account])

    async def test_retry_refreshes_waf_cookie_on_waf_blocked(self) -> None:
        """WAF 命中时应标记该账号 acw_tc 失效并后台触发刷新。"""
        import asyncio

        account = Account(email="alice@example.com", token="token-1")
        account.waf_cookies = "acw_tc=stale"
        account.waf_cookies_expires_at = 9999999999.0
        pool = _RetryPool(account)
        executor = QwenExecutor(SimpleNamespace(), pool)
        executor.auth_resolver.auto_heal_account = AsyncMock()

        async def fake_create_chat(acc, model, chat_type="t2t"):
            raise Exception("waf_blocked: stream validation challenge")

        executor.create_chat = fake_create_chat

        with patch("backend.upstream.qwen_executor.settings.MAX_RETRIES", 1):
            with self.assertRaisesRegex(Exception, "All 1 attempts failed"):
                async for _ in executor.chat_stream_events_with_retry("qwen3.7-plus", "hello"):
                    pass

        await asyncio.sleep(0)
        executor.auth_resolver.auto_heal_account.assert_called_once_with(account)
        self.assertEqual(account.waf_cookies_expires_at, 0)

    def test_qwen_validation_challenge_is_waf_blocked(self) -> None:
        body = '{"ret":["FAIL_SYS_USER_VALIDATE","RGV587_ERROR::SM::被挤爆啦"],"data":{"url":"https://chat.qwen.ai/api/v2/chat/completions/_____tmd_____/punish?x5secdata=secret&action=captcha"}}'

        self.assertTrue(_is_waf_blocked_body(body))

    def test_preview_redacts_qwen_challenge_tokens(self) -> None:
        body = "punish?x5secdata=secret-value&action=captcha&pureCaptcha=secret-captcha"

        preview = _preview_text(body)

        self.assertIn("x5secdata=<redacted>", preview)
        self.assertIn("pureCaptcha=<redacted>", preview)
        self.assertNotIn("secret-value", preview)
        self.assertNotIn("secret-captcha", preview)

    async def test_stream_raises_waf_blocked_for_qwen_validation_challenge(self) -> None:
        class FakeEngine:
            async def stream_chat_once(self, *_args, **_kwargs):
                yield {
                    "chunk": '{"ret":["FAIL_SYS_USER_VALIDATE","RGV587_ERROR::SM::被挤爆啦"],"data":{"url":"https://chat.qwen.ai/api/v2/chat/completions/_____tmd_____/punish?x5secdata=secret&action=captcha"}}'
                }

        executor = QwenExecutor(FakeEngine(), _Pool())

        with self.assertRaisesRegex(Exception, "waf_blocked"):
            async for _ in executor.stream("tok", "chat-1", "qwen3.7-plus", "hello"):
                pass

    async def test_get_chat_id_reuses_prewarmed_chat_before_create_chat(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        chat_pool = SimpleNamespace(
            remember_model=AsyncMock(),
            take=AsyncMock(return_value=("warm-chat-1", True)),
        )
        executor = QwenExecutor(SimpleNamespace(chat_id_pool=chat_pool), _Pool())
        executor.create_chat = AsyncMock(return_value="new-chat")

        chat_id, reused = await executor.get_chat_id(account, "qwen3.7-plus")

        self.assertEqual(chat_id, "warm-chat-1")
        self.assertTrue(reused)
        chat_pool.remember_model.assert_awaited_once_with("qwen3.7-plus", "t2t")
        chat_pool.take.assert_awaited_once_with("alice@example.com", "qwen3.7-plus", "t2t")
        executor.create_chat.assert_not_awaited()

    def test_punish_error_is_detected_from_all_known_markers(self) -> None:
        for marker in (
            "x5sec punish interception",
            "punish?x5secdata=abc",
            "https://chat.qwen.ai/api/v2/chat/completions/_____tmd_____/punish",
            "FAIL_SYS_USER_VALIDATE",
            "RGV587_ERROR::SM",
        ):
            self.assertTrue(_is_punish_error(marker.lower()), marker)
        # 普通 WAF 挑战不算 punish：那是接入层拦截，可自愈后重试
        self.assertFalse(_is_punish_error("waf_blocked: aliyun_waf_aa"))
        self.assertFalse(_is_punish_error("timeout"))

    def test_retry_backoff_grows_exponentially_and_caps(self) -> None:
        with (
            patch("backend.upstream.qwen_executor.settings.WAF_RETRY_BACKOFF_BASE_SECONDS", 5),
            patch("backend.upstream.qwen_executor.settings.WAF_RETRY_BACKOFF_MAX_SECONDS", 60),
            patch("backend.upstream.qwen_executor.settings.WAF_RETRY_EXTRA_COOLDOWN_SECONDS", 0),
            patch("backend.upstream.qwen_executor.random.uniform", return_value=0.0),
        ):
            self.assertEqual(_retry_backoff_seconds(0), 5)
            self.assertEqual(_retry_backoff_seconds(1), 10)
            self.assertEqual(_retry_backoff_seconds(2), 20)
            # 封顶，不无限增长
            self.assertEqual(_retry_backoff_seconds(10), 60)

    async def test_punish_does_not_retry_with_another_account(self) -> None:
        """x5sec punish 是终态：必须熔断，不得换号接着打。"""
        account = Account(email="alice@example.com", token="token-1")
        pool = _RetryPool(account)
        executor = QwenExecutor(SimpleNamespace(), pool)
        executor.auth_resolver.auto_heal_account = AsyncMock()
        calls = []

        async def fake_create_chat(acc, model, chat_type="t2t"):
            calls.append(acc.email)
            raise Exception("waf_blocked: FAIL_SYS_USER_VALIDATE /punish?x5secdata=secret")

        executor.create_chat = fake_create_chat

        # 允许 3 次重试；若未熔断会 acquire 到第二个账号（_RetryPool 排除后返回 None）
        with patch("backend.upstream.qwen_executor.settings.MAX_RETRIES", 3):
            with self.assertRaises(Exception) as ctx:
                async for _ in executor.chat_stream_events_with_retry("qwen3.7-plus", "hello"):
                    pass

        self.assertNotIn("All 3 attempts failed", str(ctx.exception))
        # 只打了一次，没有换号重试
        self.assertEqual(calls, ["alice@example.com"])
        self.assertEqual(pool.released, [account])
        # punish 不触发 401 自愈（那不是鉴权问题）
        executor.auth_resolver.auto_heal_account.assert_not_called()

    async def test_punish_on_non_stream_path_also_circuit_breaks(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = _RetryPool(account)
        calls = []

        async def fake_complete(token, chat_id, payload, account=None, timeout=None):
            calls.append(account.email)
            return {
                "status": 200,
                "body": '{"ret":["FAIL_SYS_USER_VALIDATE"],"data":{"url":"/_____tmd_____/punish?x5secdata=secret"}}',
            }

        executor = QwenExecutor(SimpleNamespace(complete_chat_once=fake_complete), pool)
        executor.create_chat = AsyncMock(return_value="chat-1")
        executor.auth_resolver.auto_heal_account = AsyncMock()

        with patch("backend.upstream.qwen_executor.settings.MAX_RETRIES", 3):
            with self.assertRaises(Exception) as ctx:
                await executor.complete_once_with_retry("qwen3.7-plus", "hello")

        self.assertNotIn("All 3 attempts failed", str(ctx.exception))
        self.assertEqual(calls, ["alice@example.com"])

    async def test_create_chat_omits_api_title_marker(self) -> None:
        """会话标题不得带 api_<毫秒> 前缀——它会永久留在账号会话列表里被识别。"""
        account = Account(email="alice@example.com", token="token-1")
        seen = {}

        async def fake_request(method, path, token, body=None, timeout=None, account=None, **kwargs):
            seen["body"] = body
            return {"status": 200, "body": '{"success": true, "data": {"id": "chat-x"}}'}

        executor = QwenExecutor(SimpleNamespace(_request_json=fake_request), _Pool())
        await executor.create_chat(account, "qwen3.8-max")

        self.assertEqual(seen["body"]["title"], "")
        self.assertNotIn("api_", seen["body"]["title"])


if __name__ == "__main__":
    unittest.main()
