"""删除/列举对话路径的 WAF 原地自愈回归测试。

覆盖缺陷③：delete_chat / list_chats 撞阿里风控（403 / 拦截页）时静默失败，
既不刷新 acw_tc 也不重试，导致自动删除对话长期无效、对话在上游堆积。
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.account_pool import Account
from backend.services.qwen_client import QwenClient


class _FakeResp:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.cookies = {}


def _client() -> QwenClient:
    # 跳过 __init__（避免构造 executor 等副作用）；被测方法只用无状态请求逻辑
    return QwenClient.__new__(QwenClient)


def _account() -> Account:
    return Account(email="a@test", token="tok", fingerprint_id="chrome-test")


class DeleteChatWafHealTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_chat_refreshes_waf_and_retries_on_challenge(self):
        client = _client()
        account = _account()

        session = MagicMock()
        session.request = AsyncMock(side_effect=[
            _FakeResp(403, "<!DOCTYPE html><html>aliyun_waf blocked</html>"),
            _FakeResp(200, ""),
        ])
        manager = MagicMock()
        manager.get_cookies = AsyncMock(return_value="acw_tc=new")
        manager.mark_expired = MagicMock()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)), \
             patch("backend.services.qwen_client.WafCookieManager") as waf_cls:
            waf_cls.get_instance.return_value = manager
            # 撞 WAF 后应自愈重试成功，不抛异常
            await client.delete_chat("tok", "chat123", account=account)

        self.assertEqual(session.request.call_count, 2)  # 首次撞 WAF + 重试一次
        manager.mark_expired.assert_called_once()         # 强制刷新 acw_tc

    async def test_delete_chat_heals_on_200_html_challenge(self):
        client = _client()
        account = _account()

        session = MagicMock()
        session.request = AsyncMock(side_effect=[
            _FakeResp(200, "<!DOCTYPE html><script>x5sec</script>"),
            _FakeResp(204, ""),
        ])
        manager = MagicMock()
        manager.get_cookies = AsyncMock(return_value="acw_tc=new")
        manager.mark_expired = MagicMock()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)), \
             patch("backend.services.qwen_client.WafCookieManager") as waf_cls:
            waf_cls.get_instance.return_value = manager
            await client.delete_chat("tok", "chat123", account=account)

        self.assertEqual(session.request.call_count, 2)
        manager.mark_expired.assert_called_once()

    async def test_delete_chat_fails_when_retry_still_returns_html_challenge(self):
        client = _client()
        account = _account()

        session = MagicMock()
        session.request = AsyncMock(side_effect=[
            _FakeResp(200, "<!DOCTYPE html><script>x5sec</script>"),
            _FakeResp(200, "<!DOCTYPE html><script>x5sec</script>"),
        ])
        manager = MagicMock()
        manager.get_cookies = AsyncMock(return_value="acw_tc=new")
        manager.mark_expired = MagicMock()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)), \
             patch("backend.services.qwen_client.WafCookieManager") as waf_cls:
            waf_cls.get_instance.return_value = manager
            with self.assertRaises(RuntimeError):
                await client.delete_chat("tok", "chat123", account=account)

        self.assertEqual(session.request.call_count, 2)

    async def test_delete_chat_no_retry_when_clean(self):
        client = _client()
        account = _account()

        session = MagicMock()
        session.request = AsyncMock(return_value=_FakeResp(200, ""))
        manager = MagicMock()
        manager.get_cookies = AsyncMock(return_value="acw_tc=ok")
        manager.mark_expired = MagicMock()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)), \
             patch("backend.services.qwen_client.WafCookieManager") as waf_cls:
            waf_cls.get_instance.return_value = manager
            await client.delete_chat("tok", "chat123", account=account)

        self.assertEqual(session.request.call_count, 1)  # 无 WAF 不重试
        manager.mark_expired.assert_not_called()

    async def test_list_chats_heals_on_waf(self):
        client = _client()
        account = _account()

        body_ok = '{"data": [{"id": "c1", "title": "api_1"}]}'
        session = MagicMock()
        session.request = AsyncMock(side_effect=[
            _FakeResp(403, "aliyun_waf intercepted"),
            _FakeResp(200, body_ok),
        ])
        manager = MagicMock()
        manager.get_cookies = AsyncMock(return_value="acw_tc=new")
        manager.mark_expired = MagicMock()

        with patch("backend.services.qwen_client.get_session", AsyncMock(return_value=session)), \
             patch("backend.services.qwen_client.WafCookieManager") as waf_cls:
            waf_cls.get_instance.return_value = manager
            chats = await client.list_chats("tok", account=account)

        self.assertEqual(session.request.call_count, 2)      # 撞 WAF 后重试
        self.assertEqual([c["id"] for c in chats], ["c1"])   # 重试成功后拿到列表


if __name__ == "__main__":
    unittest.main()
