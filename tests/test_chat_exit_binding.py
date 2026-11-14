"""`chats/new` 走出口池 + chat_id → 出口绑定（`CHAT_VIA_EXIT_POOL`）单测。

**全部用假网络层，不打真上游**（出口 IP 正在风控冷却期，真打会加重惩罚）。

覆盖：
1. `ChatExitBindings`：写入 / 命中 / TTL 过期 / LRU 淘汰 / drop / snapshot；
2. `resolve_chat_exit_mode` 与 `ExitPool.from_chat_settings` 的开关语义；
3. 建会话走出口 g → 后续 completions **仍走 g**（断言 proxies 完全一致）；
4. 出口 g 建会话撞 WAF → 换 j 重建 → completions 走 j；
5. 出口 g 连不上 → 换 j；
6. 所有出口都挂 → 回落直连（不阻断服务）；
7. 未配置出口池 → 全程直连，零代理调用；
8. 绑定过期（TTL）→ `get_chat_id` 丢弃预热的旧 chat 并重建会话；
9. 绑定表 LRU 上限生效。
"""

from __future__ import annotations

import sys
import time
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
from backend.services.chat_exit_binding import (
    DEFAULT_MAX_ENTRIES,
    DEFAULT_TTL_SECONDS,
    ChatExitBindings,
    resolve_chat_exit_mode,
)
from backend.services.exit_pool import ExitPool
from backend.services.qwen_client import QwenClient

G_EXIT = "socks5h://Default.g@10.6.6.114:2260"
J_EXIT = "socks5h://Default.j@10.6.6.114:2260"
C_EXIT = "socks5h://Default.c@10.6.6.114:2260"

WAF_HTML = (
    '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
    '<script>window.__aliyun_waf_aa=1;</script>'
    '<div id="aliyunCaptcha"></div></body></html>'
)

NEW_CHAT_PATH = "/api/v2/chats/new"


# ----------------------------------------------------------------------
# 假网络层（假 curl_cffi 会话）
# ----------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = int(status_code)
        self.text = str(text)
        self.cookies: dict[str, str] = {}


class _FakeStreamResponse:
    """流式响应替身：`async with` + `aiter_content()`。"""

    def __init__(self, status_code: int, chunks: list[bytes]) -> None:
        self.status_code = int(status_code)
        self._chunks = list(chunks)
        self.headers: dict[str, str] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_content(self):
        for chunk in self._chunks:
            yield chunk


class _FakeSession:
    """一条「经某个出口（或直连）出发」的会话；响应由网络脚本决定。"""

    def __init__(self, proxy_url, network: "_FakeExitNetwork") -> None:
        self.proxy_url = proxy_url
        self.network = network
        self.closed = False

    async def request(self, method, url, headers=None, json=None, timeout=None, **_kwargs):
        del method, headers, json, timeout
        return self.network.respond(self.proxy_url, url)

    def stream(self, method, url, headers=None, json=None, timeout=None, **_kwargs):
        del method, headers, json, timeout
        return self.network.respond_stream(self.proxy_url, url)

    async def close(self) -> None:
        self.closed = True


class _FakeExitNetwork:
    """按「出口 URL + 路径」编排响应，并记录每个请求实际走了哪个出口。"""

    def __init__(self, script, *, stream_chunks: list[bytes] | None = None) -> None:
        self.script = script
        self.stream_chunks = stream_chunks if stream_chunks is not None else [b'data: {"text":"hi"}\n\n']
        self.calls: list[tuple[str | None, str]] = []
        self.sessions: list[_FakeSession] = []

    def respond(self, proxy_url, url: str) -> _FakeResponse:
        path = str(url).split("chat.qwen.ai", 1)[-1]
        self.calls.append((proxy_url, path))
        outcome = self.script(proxy_url, path)
        if isinstance(outcome, Exception):
            raise outcome
        status, text = outcome
        return _FakeResponse(status, text)

    def respond_stream(self, proxy_url, url: str) -> _FakeStreamResponse:
        response = self.respond(proxy_url, url)
        return _FakeStreamResponse(response.status_code, self.stream_chunks if response.status_code == 200 else [response.text.encode()])

    def factory(self, proxy_url):
        """出口会话工厂（同步实现，客户端两种都支持）。"""
        session = _FakeSession(proxy_url, self)
        self.sessions.append(session)
        return session

    @property
    def proxies(self) -> list[str | None]:
        return [proxy for proxy, _path in self.calls]


class _FakeWafManager:
    """假 WAF cookie 管理器：绝不发真实请求。"""

    async def get_cookies(self, account) -> str:
        return str(getattr(account, "waf_cookies", "") or "")

    def mark_expired(self, account) -> None:
        account.waf_cookies = ""
        account.waf_cookies_expires_at = 0

    def invalidate(self, account) -> None:
        account.waf_cookies = ""

    def update_cookies(self, account, cookies) -> None:
        del account, cookies


def _fake_waf_manager():
    """patch `WafCookieManager.get_instance`，把真实网络刷新挡在单测之外。"""
    return patch(
        "backend.services.qwen_client.WafCookieManager.get_instance",
        return_value=_FakeWafManager(),
    )


def _account() -> Account:
    return Account(
        email="alice@example.com",
        token="tok-1",
        # 预置一个未过期的 acw_tc：即使补丁失效也不会触发真实刷新
        waf_cookies="acw_tc=seed",
        waf_cookies_expires_at=time.time() + 1800,
    )


def _client(exits=(G_EXIT, J_EXIT, C_EXIT), *, network=None, bindings=None) -> QwenClient:
    """构造客户端并注入出口池 / 假代理层（`exits=()` 表示不启用出口池）。"""
    client = QwenClient(
        None,
        chat_bindings=bindings,
        exit_session_factory=(network.factory if network is not None else None),
    )
    client.chat_exit_pool = ExitPool(list(exits)) if exits else None
    return client


def _chat_new_ok(chat_id: str):
    return (200, '{"success": true, "data": {"id": "%s"}}' % chat_id)


# ----------------------------------------------------------------------
# 1. 绑定表本体
# ----------------------------------------------------------------------
class ChatExitBindingsTests(unittest.TestCase):
    def test_bind_and_resolve(self) -> None:
        bindings = ChatExitBindings()
        binding = bindings.bind("chat-1", "Default.g", G_EXIT)

        self.assertIsNotNone(binding)
        self.assertEqual(bindings.count, 1)
        hit = bindings.resolve("chat-1")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.exit_id, "Default.g")
        self.assertEqual(hit.proxy_url, G_EXIT)
        # 探针视图不含代理凭据
        self.assertNotIn(G_EXIT, str(bindings.snapshot()))

    def test_empty_keys_are_ignored(self) -> None:
        bindings = ChatExitBindings()
        self.assertIsNone(bindings.bind("", "Default.g", G_EXIT))
        self.assertIsNone(bindings.bind("chat-1", "Default.g", ""))
        self.assertEqual(bindings.count, 0)
        self.assertIsNone(bindings.resolve(""))
        self.assertFalse(bindings.drop(""))

    def test_ttl_expiry_drops_binding(self) -> None:
        """绑定过期后不再返回，调用方据此重建会话。"""
        now = [1000.0]
        bindings = ChatExitBindings(ttl_seconds=60, clock=lambda: now[0])
        bindings.bind("chat-1", "Default.g", G_EXIT)

        now[0] += 59
        self.assertIsNotNone(bindings.resolve("chat-1"))
        now[0] += 2
        self.assertIsNone(bindings.resolve("chat-1"))
        self.assertEqual(bindings.count, 0)

    def test_purge_expired_sweeps_only_stale(self) -> None:
        now = [0.0]
        bindings = ChatExitBindings(ttl_seconds=10, clock=lambda: now[0])
        bindings.bind("old", "Default.g", G_EXIT)
        now[0] += 20
        bindings.bind("fresh", "Default.j", J_EXIT)

        self.assertEqual(bindings.purge_expired(), 1)
        self.assertEqual(bindings.count, 1)
        self.assertIsNotNone(bindings.resolve("fresh"))

    def test_lru_eviction_respects_max_entries(self) -> None:
        """容量上限生效：最久未用的绑定被淘汰，仍在用的被保留。"""
        bindings = ChatExitBindings(max_entries=2)
        bindings.bind("chat-1", "Default.g", G_EXIT)
        bindings.bind("chat-2", "Default.j", J_EXIT)
        # 命中 chat-1 把它提到「最近使用」，再塞第三条应淘汰 chat-2
        self.assertIsNotNone(bindings.resolve("chat-1"))
        bindings.bind("chat-3", "Default.c", C_EXIT)

        self.assertEqual(bindings.count, 2)
        self.assertIsNotNone(bindings.resolve("chat-1"))
        self.assertIsNone(bindings.resolve("chat-2"))
        self.assertIsNotNone(bindings.resolve("chat-3"))

    def test_rebind_refreshes_expiry_and_is_not_duplicated(self) -> None:
        now = [0.0]
        bindings = ChatExitBindings(ttl_seconds=10, clock=lambda: now[0])
        bindings.bind("chat-1", "Default.g", G_EXIT)
        now[0] += 5
        bindings.bind("chat-1", "Default.j", J_EXIT)
        now[0] += 6

        self.assertEqual(bindings.count, 1)
        hit = bindings.resolve("chat-1")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.exit_id, "Default.j")

    def test_defaults_and_settings_override(self) -> None:
        self.assertEqual(ChatExitBindings().ttl_seconds, DEFAULT_TTL_SECONDS)
        self.assertEqual(ChatExitBindings().max_entries, DEFAULT_MAX_ENTRIES)

        configured = ChatExitBindings.from_settings(
            SimpleNamespace(CHAT_EXIT_BINDING_TTL_SECONDS=90, CHAT_EXIT_BINDING_MAX_ENTRIES=7)
        )
        self.assertEqual(configured.ttl_seconds, 90.0)
        self.assertEqual(configured.max_entries, 7)

        # 非法 / 缺失配置退回默认值（写错 env 不该打死链路）
        fallback = ChatExitBindings.from_settings(
            SimpleNamespace(CHAT_EXIT_BINDING_TTL_SECONDS="oops", CHAT_EXIT_BINDING_MAX_ENTRIES=0)
        )
        self.assertEqual(fallback.ttl_seconds, DEFAULT_TTL_SECONDS)
        self.assertEqual(fallback.max_entries, DEFAULT_MAX_ENTRIES)


# ----------------------------------------------------------------------
# 2. 开关语义
# ----------------------------------------------------------------------
class ChatExitModeTests(unittest.TestCase):
    def test_resolve_mode_defaults_to_auto(self) -> None:
        self.assertEqual(resolve_chat_exit_mode(SimpleNamespace()), "auto")
        self.assertEqual(resolve_chat_exit_mode(SimpleNamespace(CHAT_VIA_EXIT_POOL="ALWAYS")), "always")
        self.assertEqual(resolve_chat_exit_mode(SimpleNamespace(CHAT_VIA_EXIT_POOL="never")), "never")
        # 非法值按 auto
        self.assertEqual(resolve_chat_exit_mode(SimpleNamespace(CHAT_VIA_EXIT_POOL="oops")), "auto")

    def test_pool_disabled_when_exits_empty_or_mode_never(self) -> None:
        self.assertIsNone(ExitPool.from_chat_settings(SimpleNamespace(UPLOAD_EXIT_POOL="")))
        self.assertIsNone(
            ExitPool.from_chat_settings(
                SimpleNamespace(UPLOAD_EXIT_POOL=f"{G_EXIT},{J_EXIT}", CHAT_VIA_EXIT_POOL="never")
            )
        )
        self.assertIsNone(ExitPool.from_chat_settings(None))

    def test_pool_enabled_from_shared_exit_list(self) -> None:
        pool = ExitPool.from_chat_settings(
            SimpleNamespace(UPLOAD_EXIT_POOL=f"{G_EXIT},{J_EXIT}", CHAT_VIA_EXIT_POOL="auto")
        )
        self.assertIsNotNone(pool)
        self.assertEqual(pool.exit_ids, ["Default.g", "Default.j"])
        # 上传链路被单独关掉也不影响建会话（两个开关互相独立）
        pool_always = ExitPool.from_chat_settings(
            SimpleNamespace(
                UPLOAD_EXIT_POOL=G_EXIT,
                UPLOAD_VIA_EXIT="never",
                CHAT_VIA_EXIT_POOL="always",
            )
        )
        self.assertIsNotNone(pool_always)


# ----------------------------------------------------------------------
# 3~7. 建会话 + 后续请求的出口亲和性
# ----------------------------------------------------------------------
class ChatExitAffinityTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_chat_and_completions_share_same_exit(self) -> None:
        """建会话走 g → 后续 completions 也走 g（断言 proxies 一致）。"""

        def script(proxy_url, path):
            self.assertEqual(proxy_url, G_EXIT)
            if path == NEW_CHAT_PATH:
                return _chat_new_ok("chat-g")
            if path.startswith("/api/v2/chat/completions"):
                return (200, '{"ok": true}')
            raise AssertionError(f"unexpected path: {path}")

        network = _FakeExitNetwork(script)
        client = _client(network=network)
        account = _account()

        with _fake_waf_manager(), self.assertLogs(level="INFO") as captured:
            chat_id = await client.executor.create_chat(account, "qwen4.0-max-preview")
            result = await client._request_json(
                "POST",
                f"/api/v2/chat/completions?chat_id={chat_id}",
                account.token,
                body={"stream": True},
                account=account,
                chat_transport=True,
                chat_id=chat_id,
            )

        self.assertEqual(chat_id, "chat-g")
        self.assertEqual(result["status"], 200)
        # 两次请求都钉在同一个出口上
        self.assertEqual(network.proxies, [G_EXIT, G_EXIT])
        self.assertEqual(client.chat_bindings.resolve(chat_id).exit_id, "Default.g")
        # 可观测：建会话成功 + 绑定命中
        logs = "\n".join(captured.output)
        self.assertIn("[ChatExit] create ok chat_id=chat-g exit=Default.g", logs)
        self.assertIn("[ChatExit] binding hit chat_id=chat-g exit=Default.g", logs)

    async def test_waf_on_first_exit_rotates_to_next(self) -> None:
        """出口 g 建会话撞 WAF → 换 j 重建 → completions 走 j。"""

        def script(proxy_url, path):
            if proxy_url == G_EXIT:
                return (200, WAF_HTML)  # 首次 + retry_waf 重试都是挑战页
            if path == NEW_CHAT_PATH:
                return _chat_new_ok("chat-j")
            if path.startswith("/api/v2/chat/completions"):
                return (200, '{"ok": true}')
            raise AssertionError(f"unexpected path: {path}")

        network = _FakeExitNetwork(script)
        client = _client(network=network)
        account = _account()

        with _fake_waf_manager():
            chat_id = await client.executor.create_chat(account, "qwen4.0-max-preview")
            await client._request_json(
                "POST",
                f"/api/v2/chat/completions?chat_id={chat_id}",
                account.token,
                body={},
                account=account,
                chat_transport=True,
                chat_id=chat_id,
            )

        self.assertEqual(chat_id, "chat-j")
        # g 被踢出（首次 + 一次 WAF 重试），j 建会话 + completions
        self.assertEqual(network.proxies, [G_EXIT, G_EXIT, J_EXIT, J_EXIT])
        self.assertEqual(client.chat_bindings.resolve(chat_id).exit_id, "Default.j")
        snapshot = {row["id"]: row for row in client.chat_exit_pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["failures"], 1)
        self.assertEqual(snapshot["Default.j"]["failures"], 0)
        self.assertEqual(snapshot["Default.j"]["successes"], 2)  # create + completions 都记成功

    async def test_connect_failure_rotates_to_next_exit(self) -> None:
        """出口 g 连不上（socks 连接失败）→ 换 j。"""

        def script(proxy_url, path):
            if proxy_url == G_EXIT:
                raise RuntimeError("socks connect failed")
            return _chat_new_ok("chat-j") if path == NEW_CHAT_PATH else (200, '{"ok": true}')

        network = _FakeExitNetwork(script)
        client = _client(network=network)

        with _fake_waf_manager():
            chat_id = await client.executor.create_chat(_account(), "qwen4.0-max-preview")

        self.assertEqual(chat_id, "chat-j")
        self.assertEqual(network.proxies, [G_EXIT, J_EXIT])
        self.assertEqual(client.chat_bindings.resolve(chat_id).exit_id, "Default.j")

    async def test_all_exits_blocked_falls_back_to_direct(self) -> None:
        """所有出口都被 WAF 拦下 → 回落直连（宁可慢，也不能整条链路不可用）。"""

        def script(proxy_url, path):
            if proxy_url is not None:
                return (200, WAF_HTML)
            return _chat_new_ok("chat-direct") if path == NEW_CHAT_PATH else (200, '{"ok": true}')

        network = _FakeExitNetwork(script)
        client = _client(network=network)
        direct_session = _FakeSession(None, network)

        with _fake_waf_manager(), patch(
            "backend.services.qwen_client.get_session", AsyncMock(return_value=direct_session)
        ):
            chat_id = await client.executor.create_chat(_account(), "qwen4.0-max-preview")

        self.assertEqual(chat_id, "chat-direct")
        # 3 个出口各试两次（建会话 + WAF 重试），最后直连成功
        self.assertEqual(
            network.proxies,
            [G_EXIT, G_EXIT, J_EXIT, J_EXIT, C_EXIT, C_EXIT, None],
        )
        # 直连建出来的会话没有出口绑定
        self.assertIsNone(client.chat_bindings.resolve(chat_id))

    async def test_no_exit_pool_uses_direct_transport_only(self) -> None:
        """未配置出口池 → 建会话与 completions 全程直连，零代理调用。"""

        def script(proxy_url, path):
            self.assertIsNone(proxy_url)
            return _chat_new_ok("chat-direct") if path == NEW_CHAT_PATH else (200, '{"ok": true}')

        network = _FakeExitNetwork(script)
        client = _client(exits=())  # 未启用出口池
        direct_session = _FakeSession(None, network)

        with _fake_waf_manager(), patch(
            "backend.services.qwen_client.get_session", AsyncMock(return_value=direct_session)
        ):
            chat_id = await client.executor.create_chat(_account(), "qwen4.0-max-preview")
            await client._request_json(
                "POST",
                f"/api/v2/chat/completions?chat_id={chat_id}",
                "tok-1",
                body={},
                account=_account(),
                chat_transport=True,
                chat_id=chat_id,
            )

        self.assertIsNone(client.chat_exit_pool)
        self.assertEqual(network.proxies, [None, None])
        self.assertEqual(network.sessions, [])  # 一个出口会话都没建（零代理）
        self.assertEqual(client.chat_bindings.count, 0)

    async def test_streaming_completions_use_bound_exit(self) -> None:
        """流式 completions（生产主路径）同样复用建会话时的出口。"""

        def script(proxy_url, path):
            self.assertEqual(proxy_url, G_EXIT)
            return _chat_new_ok("chat-g") if path == NEW_CHAT_PATH else (200, "")

        network = _FakeExitNetwork(script)
        client = _client(network=network)
        account = _account()

        with _fake_waf_manager():
            chat_id = await client.executor.create_chat(account, "qwen4.0-max-preview")
            events = [
                event
                async for event in client.stream_chat_once(
                    account.token, chat_id, {"stream": True}, account=account
                )
            ]

        self.assertEqual(chat_id, "chat-g")
        self.assertEqual(network.proxies, [G_EXIT, G_EXIT])
        self.assertTrue(any("chunk" in event for event in events))
        self.assertTrue(any(event.get("status") == "streamed" for event in events))
        # 出口会话用完即关，避免连接堆积
        self.assertTrue(all(session.closed for session in network.sessions))

    async def test_binding_is_dropped_when_bound_exit_is_blocked(self) -> None:
        """绑定的出口在 completions 时被 WAF 拦 → 记账并丢绑定，下次重建换出口。"""

        def script(proxy_url, path):
            if path == NEW_CHAT_PATH:
                return _chat_new_ok("chat-g")
            return (200, WAF_HTML)

        network = _FakeExitNetwork(script)
        client = _client(network=network)
        account = _account()

        with _fake_waf_manager():
            chat_id = await client.executor.create_chat(account, "qwen4.0-max-preview")
            self.assertIsNotNone(client.chat_bindings.resolve(chat_id))
            result = await client._request_json(
                "POST",
                f"/api/v2/chat/completions?chat_id={chat_id}",
                account.token,
                body={},
                account=account,
                chat_transport=True,
                chat_id=chat_id,
            )

        self.assertEqual(result["status"], 200)  # 上游 200 + WAF 页
        self.assertIsNone(client.chat_bindings.resolve(chat_id))
        snapshot = {row["id"]: row for row in client.chat_exit_pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["failures"], 1)


# ----------------------------------------------------------------------
# 8. TTL 过期 → 丢弃预热的旧 chat 并重建会话
# ----------------------------------------------------------------------
class ChatExitBindingExpiryTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_binding_forces_new_chat(self) -> None:
        now = [0.0]
        bindings = ChatExitBindings(ttl_seconds=60, clock=lambda: now[0])
        client = _client(exits=(G_EXIT,), bindings=bindings)
        client.bind_chat_exit("chat-warm", "Default.g", G_EXIT)

        chat_pool = SimpleNamespace(
            remember_model=AsyncMock(),
            take=AsyncMock(return_value=("chat-warm", True)),
        )
        client.chat_id_pool = chat_pool
        client.executor.create_chat = AsyncMock(return_value="chat-new")

        # 未过期：复用预热会话
        account = _account()
        now[0] += 30
        chat_id, reused = await client.executor.get_chat_id(account, "qwen4.0-max-preview")
        self.assertEqual((chat_id, reused), ("chat-warm", True))
        client.executor.create_chat.assert_not_awaited()

        # 过期后：预热的 chat 出口对不上 → 丢弃并重建
        now[0] += 40
        chat_id, reused = await client.executor.get_chat_id(account, "qwen4.0-max-preview")
        self.assertEqual((chat_id, reused), ("chat-new", False))
        client.executor.create_chat.assert_awaited_once_with(account, "qwen4.0-max-preview")

    async def test_missing_binding_for_prewarmed_chat_forces_new_chat(self) -> None:
        """功能上线前建的预热 chat 没有绑定 → 不能直接拿来发 completions。"""
        client = _client(exits=(G_EXIT,))
        chat_pool = SimpleNamespace(
            remember_model=AsyncMock(),
            take=AsyncMock(return_value=("chat-legacy", True)),
        )
        client.chat_id_pool = chat_pool
        client.executor.create_chat = AsyncMock(return_value="chat-new")

        chat_id, reused = await client.executor.get_chat_id(_account(), "qwen4.0-max-preview")

        self.assertEqual((chat_id, reused), ("chat-new", False))

    async def test_prewarmed_chat_reused_when_binding_present(self) -> None:
        client = _client(exits=(G_EXIT,))
        client.bind_chat_exit("chat-warm", "Default.g", G_EXIT)
        chat_pool = SimpleNamespace(
            remember_model=AsyncMock(),
            take=AsyncMock(return_value=("chat-warm", True)),
        )
        client.chat_id_pool = chat_pool
        client.executor.create_chat = AsyncMock(return_value="chat-new")

        chat_id, reused = await client.executor.get_chat_id(_account(), "qwen4.0-max-preview")

        self.assertEqual((chat_id, reused), ("chat-warm", True))
        client.executor.create_chat.assert_not_awaited()

    async def test_prewarmed_chat_reused_when_pool_disabled(self) -> None:
        """出口池未启用时，预热复用行为保持原样（不看绑定）。"""
        client = _client(exits=())
        chat_pool = SimpleNamespace(
            remember_model=AsyncMock(),
            take=AsyncMock(return_value=("chat-warm", True)),
        )
        client.chat_id_pool = chat_pool
        client.executor.create_chat = AsyncMock(return_value="chat-new")

        chat_id, reused = await client.executor.get_chat_id(_account(), "qwen4.0-max-preview")

        self.assertEqual((chat_id, reused), ("chat-warm", True))


if __name__ == "__main__":
    unittest.main()
