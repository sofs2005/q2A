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
from backend.upstream.qwen_executor import QwenExecutor, _is_waf_blocked_body, _preview_text


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
    async def test_fixed_account_path_passes_account_objects_to_create_and_stream(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        account.fingerprint_id = "chrome146_windows"
        executor = QwenExecutor(SimpleNamespace(), _Pool())
        seen = []

        async def fake_create_chat(acc, model, chat_type="t2t"):
            seen.append(("create", acc, model, chat_type))
            return "chat-1"

        async def fake_stream(acc, chat_id, model, content, has_custom_tools=False, files=None):
            seen.append(("stream", acc, chat_id, model, content, has_custom_tools, files))
            if False:
                yield None
            return
            yield

        executor.create_chat = fake_create_chat
        executor.stream = fake_stream

        events = []
        async for item in executor.chat_stream_events_with_retry("gpt-4o", "hello", fixed_account=account):
            events.append(item)

        self.assertEqual(events[0]["type"], "meta")
        self.assertIs(events[0]["acc"], account)
        self.assertEqual(seen[0][0], "create")
        self.assertIs(seen[0][1], account)
        self.assertEqual(seen[0][2], "gpt-4o")
        self.assertEqual(seen[1][0], "stream")
        self.assertIs(seen[1][1], account)
        self.assertEqual(seen[1][2], "chat-1")

    async def test_create_chat_reports_waf_blocked_for_aliyun_waf_html(self) -> None:
        account = Account(email="alice@example.com", token="token-1")

        async def fake_request(method, path, token, body=None, timeout=None, account=None):
            return {
                "status": 200,
                "body": '<!doctypehtml><meta name="aliyun_waf_aa" content="blocked">',
            }

        executor = QwenExecutor(SimpleNamespace(_request_json=fake_request), _Pool())

        with self.assertRaisesRegex(Exception, "waf_blocked"):
            await executor.create_chat(account, "qwen3.7-plus")

    async def test_retry_does_not_mark_account_invalid_for_waf_blocked(self) -> None:
        account = Account(email="alice@example.com", token="token-1")
        pool = _RetryPool(account)
        executor = QwenExecutor(SimpleNamespace(), pool)

        async def fake_create_chat(acc, model, chat_type="t2t"):
            raise Exception("waf_blocked: aliyun_waf_aa")

        executor.create_chat = fake_create_chat

        with patch("backend.upstream.qwen_executor.settings.MAX_RETRIES", 1):
            with self.assertRaisesRegex(Exception, "All 1 attempts failed"):
                async for _ in executor.chat_stream_events_with_retry("qwen3.7-plus", "hello"):
                    pass

        self.assertEqual(pool.invalidated, [])
        self.assertEqual(pool.released, [account])

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


if __name__ == "__main__":
    unittest.main()
