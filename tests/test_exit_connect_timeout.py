"""出口连接超时 + 启动探测 + `pick()` 优先最近成功出口的单测（全部假网络）。

覆盖（对应需求 1 / 2 / 3）：
1. `EXIT_CONNECT_TIMEOUT_SECONDS` 只压**建连阶段**：curl_cffi 支持元组时下发
   `(connect, read)`，不支持时整体收紧到 connect 值；读取阶段保持原超时。
2. 建连超时的请求会**快速失败**并轮到下一个出口（不再等满整体超时）。
3. 启动探测：3 个出口 1 个连不上 → 死的进冷却、其余 `state=ready`；
   探测失败不影响服务启动（后台任务，不阻塞 startup）。
4. `pick()` 优先返回 `last_success_at` 最新的出口；冷却期内跳过、到期恢复。
5. `/admin/upload/probe` 快照新增 `last_success_at` / `last_failure_at` /
   `avg_connect_ms`。
6. 探测判定口径（2026-12 事故修复）：宽预算（单出口 20s / 总 45s）、限并发 2、
   **有 HTTP 响应就算活**（401/403/429/WAF 页都不判死）、判死用 120s 短冷却且
   不动 `failures`、全灭时兜底为 available + WARNING。
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

if "pydantic_settings" not in sys.modules:
    fake_pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        pass

    fake_pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = fake_pydantic_settings

if "oss2" not in sys.modules:
    fake_oss2 = types.ModuleType("oss2")

    class StsAuth:
        pass

    class Bucket:
        pass

    fake_oss2.StsAuth = StsAuth
    fake_oss2.Bucket = Bucket
    sys.modules["oss2"] = fake_oss2

from backend.core import http_timeout
from backend.core.http_timeout import connect_read_timeout, connect_read_timeout_seconds
from backend.services.exit_pool import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    STATE_COOLING,
    STATE_READY,
    ExitPool,
    exit_id_from_url,
)
from backend.services.exit_probe import (
    probe_enabled,
    run_startup_exit_probe,
    select_pool,
    startup_exit_probe_task,
)

G_EXIT = "socks5h://Default.g@10.6.6.114:2260"
J_EXIT = "socks5h://Default.j@10.6.6.114:2260"
C_EXIT = "socks5h://Default.c@10.6.6.114:2260"

#: 单测用的「建连超时」注入值（真 5s 会让测试变慢；逻辑完全相同）
FAST_CONNECT_TIMEOUT = 0.05


def _settings(**overrides) -> SimpleNamespace:
    """探针/构造用的最小配置替身。"""
    base = {
        "UPLOAD_EXIT_POOL": f"{G_EXIT},{J_EXIT},{C_EXIT}",
        "UPLOAD_VIA_EXIT": "auto",
        "UPLOAD_EXIT_COOLDOWN_SECONDS": 600,
        "UPLOAD_EXIT_MAX_FAILURES": 1,
        "UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS": 10,
        "EXIT_CONNECT_TIMEOUT_SECONDS": FAST_CONNECT_TIMEOUT,
        "EXIT_PICK_PREFER_LAST_SUCCESS": True,
        "EXIT_PROBE_ON_STARTUP": True,
        "EXIT_PROBE_TOTAL_TIMEOUT_SECONDS": 15,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _split_timeout(value) -> tuple[float, float]:
    """把 timeout（浮点 / 元组）拆成 `(connect, read)`，便于断言。"""
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    return float(value), float(value)


# ----------------------------------------------------------------------
# 1. 连接超时工具
# ----------------------------------------------------------------------
class ConnectTimeoutHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        http_timeout.reset_capability_cache()

    def tearDown(self) -> None:
        http_timeout.reset_capability_cache()

    def test_default_connect_timeout_is_five_seconds(self) -> None:
        self.assertEqual(DEFAULT_CONNECT_TIMEOUT_SECONDS, 5.0)

    def test_tuple_keeps_read_timeout_untouched(self) -> None:
        # curl_cffi 支持元组：只压建连，读仍是原值（大模型生成慢，绝不能压到 5s）。
        # 显式固定能力位：全量跑测试时别的用例可能把 curl_cffi 换成替身模块。
        with patch.object(http_timeout, "_SUPPORTS_TIMEOUT_TUPLE", True):
            value = connect_read_timeout(120, 5)
            self.assertEqual(_split_timeout(value), (5.0, 120.0))
            self.assertEqual(connect_read_timeout_seconds(120, 5), 5.0)

    def test_fallback_tightens_whole_timeout_when_tuple_unsupported(self) -> None:
        with patch.object(http_timeout, "_SUPPORTS_TIMEOUT_TUPLE", False):
            value = connect_read_timeout(120, 5)
        self.assertEqual(value, 5.0)

    def test_no_connect_timeout_keeps_original_semantics(self) -> None:
        self.assertEqual(connect_read_timeout(120, None), 120.0)
        self.assertEqual(connect_read_timeout(120, 0), 120.0)

    def test_connect_never_exceeds_read(self) -> None:
        # connect 比 read 还宽只会误导人：以 read 为准
        with patch.object(http_timeout, "_SUPPORTS_TIMEOUT_TUPLE", True):
            self.assertEqual(_split_timeout(connect_read_timeout(3, 5)), (3.0, 3.0))


# ----------------------------------------------------------------------
# 2. 慢连接快速失败 + 轮到下一个出口（真实传输路径 QwenClient._request_json）
# ----------------------------------------------------------------------
class _RecordingSession:
    """假出口会话：按 (connect, read) 拆分 timeout，建连阶段「卡住」后超时。"""

    def __init__(self, proxy_url: str, *, dead: bool, calls: list[dict]) -> None:
        self.proxy_url = proxy_url
        self.dead = dead
        self.calls = calls
        self.closed = False

    async def request(self, method, url, headers=None, json=None, timeout=None):
        del method, url, headers, json
        connect, read = _split_timeout(timeout)
        self.calls.append({"proxy": self.proxy_url, "connect": connect, "read": read})
        if self.dead:
            # 模拟「建连卡住」：只在 connect 预算内干等，然后像 curl 一样抛超时。
            await asyncio.sleep(connect)
            raise TimeoutError("connect timed out")
        return SimpleNamespace(status_code=200, text='{"success":true}', cookies={})

    async def close(self) -> None:
        self.closed = True


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.cookies = {}


class ConnectTimeoutRequestTests(unittest.IsolatedAsyncioTestCase):
    """建连超时必须让死出口「5s 内放弃」并轮到下一个出口。"""

    def setUp(self) -> None:
        # 固定「curl_cffi 支持 timeout 元组」：全量跑测试时 curl_cffi 可能被替身替换。
        self._capability = patch.object(http_timeout, "_SUPPORTS_TIMEOUT_TUPLE", True)
        self._capability.start()

    def tearDown(self) -> None:
        self._capability.stop()
        http_timeout.reset_capability_cache()

    def _client(self, calls: list[dict], *, dead_exits: set[str]):
        from backend.services.qwen_client import QwenClient

        client = QwenClient(None)
        pool = ExitPool(
            [G_EXIT, J_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            connect_timeout_seconds=FAST_CONNECT_TIMEOUT,
        )
        client.chat_exit_pool = pool

        def factory(proxy_url: str):
            return _RecordingSession(
                proxy_url,
                dead=exit_id_from_url(proxy_url) in dead_exits,
                calls=calls,
            )

        client._exit_session_factory = factory
        return client, pool

    async def test_connect_timeout_is_applied_and_fails_fast(self) -> None:
        calls: list[dict] = []
        client, _pool = self._client(calls, dead_exits={"Default.g"})
        with patch(
            "backend.services.qwen_client.settings.EXIT_CONNECT_TIMEOUT_SECONDS",
            FAST_CONNECT_TIMEOUT,
            create=True,
        ):
            started = time.monotonic()
            dead = await client._request_json(
                "POST", "/api/v2/chats/new", "tok", {}, timeout=120.0, proxy_url=G_EXIT
            )
            dead_elapsed = time.monotonic() - started

        # 死出口：建连预算（0.05s）内放弃，而不是等满 120s 的读超时。
        self.assertEqual(dead["status"], 0)
        self.assertLess(dead_elapsed, 2.0)
        # 超时按 (connect, read) 下发：connect 收紧，read 仍是原值。
        self.assertEqual(calls[0]["connect"], FAST_CONNECT_TIMEOUT)
        self.assertEqual(calls[0]["read"], 120.0)

    async def test_slow_connect_rotates_to_next_exit(self) -> None:
        """死出口快速失败后轮到下一个出口（与 executor.create_chat 的轮转同构）。"""

        calls: list[dict] = []
        client, pool = self._client(calls, dead_exits={"Default.g"})
        tried: set[str] = set()
        winner = ""
        started = time.monotonic()
        with patch(
            "backend.services.qwen_client.settings.EXIT_CONNECT_TIMEOUT_SECONDS",
            FAST_CONNECT_TIMEOUT,
            create=True,
        ):
            while True:
                proxy_url = pool.pick(exclude=tried)
                if not proxy_url:
                    break
                exit_id = exit_id_from_url(proxy_url)
                if exit_id in tried:
                    break
                tried.add(exit_id)
                res = await client._request_json(
                    "POST", "/api/v2/chats/new", "tok", {}, timeout=30.0, proxy_url=proxy_url
                )
                if res["status"] == 200:
                    pool.mark_success(exit_id)
                    winner = exit_id
                    break
                pool.mark_failure(exit_id, "connect_failed")
        elapsed = time.monotonic() - started

        self.assertEqual(winner, "Default.j")
        self.assertEqual(len(calls), 2)
        self.assertLess(elapsed, 2.0)
        snapshot = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(snapshot["Default.g"]["state"], STATE_COOLING)
        self.assertEqual(snapshot["Default.j"]["state"], STATE_READY)
        # 之后直接命中已证明可用的出口，不再从零试错。
        self.assertEqual(exit_id_from_url(pool.pick()), "Default.j")


# ----------------------------------------------------------------------
# 3. 启动探测
# ----------------------------------------------------------------------
class _FakePool:
    """`probe_all` 替身：只记录调用，用于验证「不阻塞启动」。"""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.enabled = True
        self.exit_ids = ["Default.g"]
        self.gate = gate
        self.probed = False

    async def probe_all(self, **_kwargs):
        self.probed = True
        if self.gate is not None:
            await self.gate.wait()
        return [{"id": "Default.g", "ok": True, "reason": "", "elapsed_seconds": 0.1, "state": STATE_READY}]


def _app(**state) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(**state))


class StartupProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_all_marks_dead_exit_cooling(self) -> None:
        """"3 个出口，1 个连不上 → 死的进冷却，另外两个 ready。"""

        async def fake_fetch(proxy_url: str, _timeout: float) -> bool:
            if exit_id_from_url(proxy_url) == "Default.c":
                raise OSError("socks5 connect failed")
            return True

        pool = ExitPool(
            [G_EXIT, J_EXIT, C_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            connect_timeout_seconds=FAST_CONNECT_TIMEOUT,
            probe_fetch=fake_fetch,
        )
        results = await pool.probe_all(limit_seconds=5.0)
        by_id = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(len(results), 3)
        self.assertEqual(by_id["Default.g"]["state"], STATE_READY)
        self.assertEqual(by_id["Default.j"]["state"], STATE_READY)
        self.assertEqual(by_id["Default.c"]["state"], STATE_COOLING)
        self.assertGreater(by_id["Default.c"]["cooldown_remaining"], 0)
        dead = [row for row in results if not row["ok"]]
        self.assertEqual([row["id"] for row in dead], ["Default.c"])
        self.assertEqual(dead[0]["reason"], "connect_failed")
        # 成功的两个记了 last_success_at（之后 pick 会优先它们）
        self.assertGreater(by_id["Default.g"]["last_success_at"], 0)
        self.assertGreater(by_id["Default.j"]["last_success_at"], 0)

    async def test_probe_all_is_concurrent_and_bounded(self) -> None:
        """并发探测：3 个出口各花 0.2s，总耗时远小于串行的 0.6s。"""

        async def slow_fetch(_proxy_url: str, _timeout: float) -> bool:
            await asyncio.sleep(0.2)
            return True

        pool = ExitPool(
            [G_EXIT, J_EXIT, C_EXIT],
            probe_fetch=slow_fetch,
            connect_timeout_seconds=FAST_CONNECT_TIMEOUT,
        )
        started = time.monotonic()
        results = await pool.probe_all(limit_seconds=5.0)
        elapsed = time.monotonic() - started
        self.assertEqual(len(results), 3)
        self.assertLess(elapsed, 0.5)

    async def test_probe_budget_is_capped(self) -> None:
        """总预算耗尽时不会挂住整个探测（全灭兜底 → 视为网络抖动，池子保持可用）。"""

        async def stuck_fetch(_proxy_url: str, _timeout: float) -> bool:
            await asyncio.sleep(5.0)
            return True

        pool = ExitPool([G_EXIT, J_EXIT], probe_fetch=stuck_fetch, probe_timeout_seconds=5.0)
        started = time.monotonic()
        results = await pool.probe_all(limit_seconds=0.2)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertEqual(len(results), 2)
        # 全灭兜底：全部失败时按「探测自身抖动」处理，不惩罚池子（见需求 4）。
        self.assertTrue(all(row["ok"] for row in results))
        self.assertTrue(all(row.get("assumed_alive") for row in results))
        self.assertTrue(all(row["state"] == STATE_READY for row in pool.snapshot()))

    async def test_startup_probe_does_not_block_startup(self) -> None:
        """探测任务不能被 await 在 startup 里：先让出事件循环，探测在后台跑。"""
        gate = asyncio.Event()
        pool = _FakePool(gate=gate)
        app = _app(qwen_client=SimpleNamespace(chat_exit_pool=pool), upload_exit_pool=None, account_pool=None)
        task = asyncio.create_task(startup_exit_probe_task(app))
        # startup 已经可以往下走（这里用「事件循环还能调度别的协程」证明）
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertTrue(pool.probed)
        gate.set()
        await asyncio.wait_for(task, timeout=5.0)
        self.assertTrue(task.done())

    async def test_run_startup_exit_probe_summarises(self) -> None:
        async def fake_fetch(proxy_url: str, _timeout: float) -> bool:
            return exit_id_from_url(proxy_url) != "Default.c"

        pool = ExitPool(
            [G_EXIT, J_EXIT, C_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            probe_fetch=fake_fetch,
        )
        app = _app(
            qwen_client=SimpleNamespace(chat_exit_pool=pool),
            upload_exit_pool=None,
            account_pool=SimpleNamespace(accounts=[SimpleNamespace(valid=True, token="tok")]),
        )
        summary = await run_startup_exit_probe(app, _settings())
        self.assertEqual(summary["pool"], "chat")
        self.assertEqual((summary["total"], summary["ok"], summary["dead"]), (3, 2, 1))

    async def test_run_startup_exit_probe_skips_without_pool(self) -> None:
        app = _app(qwen_client=SimpleNamespace(chat_exit_pool=None), upload_exit_pool=None)
        summary = await run_startup_exit_probe(app, _settings())
        self.assertEqual(summary["skipped"], "no_pool")

    async def test_run_startup_exit_probe_skips_when_disabled(self) -> None:
        pool = _FakePool()
        app = _app(qwen_client=SimpleNamespace(chat_exit_pool=pool), upload_exit_pool=None)
        summary = await run_startup_exit_probe(app, _settings(EXIT_PROBE_ON_STARTUP=False))
        self.assertEqual(summary["skipped"], "disabled")
        self.assertFalse(pool.probed)

    async def test_probe_never_raises_even_on_broken_pool(self) -> None:
        class _BrokenPool:
            enabled = True

            async def probe_all(self, **_kwargs):
                raise RuntimeError("boom")

        app = _app(qwen_client=SimpleNamespace(chat_exit_pool=_BrokenPool()), upload_exit_pool=None)
        summary = await run_startup_exit_probe(app, _settings())
        self.assertTrue(summary["skipped"].startswith("error:"))

    def test_select_pool_prefers_chat_then_upload(self) -> None:
        chat = _FakePool()
        upload = _FakePool()
        app = _app(qwen_client=SimpleNamespace(chat_exit_pool=chat), upload_exit_pool=upload)
        self.assertEqual(select_pool(app), (chat, "chat"))
        app2 = _app(qwen_client=SimpleNamespace(chat_exit_pool=None), upload_exit_pool=upload)
        self.assertEqual(select_pool(app2), (upload, "upload"))
        app3 = _app(qwen_client=SimpleNamespace(chat_exit_pool=None), upload_exit_pool=None)
        self.assertEqual(select_pool(app3), (None, ""))

    def test_probe_enabled_reads_bool_and_text(self) -> None:
        self.assertTrue(probe_enabled(_settings()))
        self.assertTrue(probe_enabled(_settings(EXIT_PROBE_ON_STARTUP=True)))
        self.assertTrue(probe_enabled(_settings(EXIT_PROBE_ON_STARTUP="yes")))
        self.assertFalse(probe_enabled(_settings(EXIT_PROBE_ON_STARTUP=False)))
        self.assertFalse(probe_enabled(_settings(EXIT_PROBE_ON_STARTUP="false")))
        self.assertFalse(probe_enabled(_settings(EXIT_PROBE_ON_STARTUP="off")))


# ----------------------------------------------------------------------
# 4. pick() 排序 + 冷却 + 观测字段
# ----------------------------------------------------------------------
class PickPreferenceTests(unittest.TestCase):
    def test_pick_spreads_across_proven_exits(self) -> None:
        # 已证明可用的出口之间要轮转摊开，不能死盯最近成功的那个：
        # 旧实现取 last_success_at 最大者，生产实测 3 并发全走 Default.r。
        now = [1000.0]
        pool = ExitPool([G_EXIT, J_EXIT, C_EXIT], clock=lambda: now[0])
        pool.mark_success("Default.g")
        now[0] += 1
        pool.mark_success("Default.j")  # 更晚成功
        picks = []
        for _ in range(4):
            now[0] += 1
            picks.append(pool.pick())
        # c 没成功过 → 不参与；g / j 交替
        self.assertEqual(picks, [J_EXIT, G_EXIT, J_EXIT, G_EXIT])

    def test_pick_never_tries_unproven_while_proven_available(self) -> None:
        # 摊开不能以「从零试错死出口」为代价：只要还有 proven 的就不碰未证明的。
        pool = ExitPool([G_EXIT, J_EXIT, C_EXIT])
        pool.mark_success("Default.c")
        for _ in range(5):
            self.assertEqual(pool.pick(), C_EXIT)

    def test_pick_covers_all_exits_in_one_round(self) -> None:
        # 生产场景：8 个出口探测全通 → 一轮内应当全部轮到。
        names = "0r13f8k9"
        exits = [f"socks5h://Default.{n}@10.6.6.114:2260" for n in names]
        now = [1000.0]
        pool = ExitPool(exits, clock=lambda: now[0])
        for name in names:
            now[0] += 1
            pool.mark_success(f"Default.{name}")
        picked = set()
        for _ in range(len(names)):
            now[0] += 1
            picked.add(pool.pick())
        self.assertEqual(len(picked), len(names))

    def test_pick_rotates_while_nothing_succeeded(self) -> None:
        # 探测还没跑 / 全都没成功过：保持旧轮询语义
        pool = ExitPool([G_EXIT, J_EXIT, C_EXIT])
        self.assertEqual([pool.pick(), pool.pick(), pool.pick()], [G_EXIT, J_EXIT, C_EXIT])

    def test_pick_skips_cooling_then_recovers(self) -> None:
        now = [1000.0]
        pool = ExitPool(
            [G_EXIT, J_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            clock=lambda: now[0],
        )
        pool.mark_success("Default.g")
        pool.mark_failure("Default.g", "connect_failed")
        self.assertEqual(pool.pick(), J_EXIT)  # 冷却期内跳过
        now[0] += 601
        self.assertEqual(pool.pick(), G_EXIT)  # 到期恢复
        self.assertEqual(pool.snapshot()[0]["failures"], 0)

    def test_prefer_last_success_can_be_disabled(self) -> None:
        pool = ExitPool([G_EXIT, J_EXIT], prefer_last_success=False)
        pool.mark_success("Default.j")
        self.assertEqual(pool.pick(), G_EXIT)  # 回到纯轮询（回滚开关）

    def test_snapshot_exposes_observability_fields(self) -> None:
        pool = ExitPool([G_EXIT])
        pool.mark_success("Default.g", connect_ms=120.0)
        pool.mark_failure("Default.g", "waf", connect_ms=80.0)
        row = pool.snapshot()[0]
        for key in ("last_success_at", "last_failure_at", "avg_connect_ms"):
            self.assertIn(key, row)
        self.assertGreater(row["last_success_at"], 0)
        self.assertGreaterEqual(row["last_failure_at"], row["last_success_at"])
        # 滑窗平均：(120 + 80) / 2 = 100ms
        self.assertAlmostEqual(row["avg_connect_ms"], 100.0, places=1)

    def test_avg_connect_ms_is_running_mean(self) -> None:
        pool = ExitPool([G_EXIT])
        pool.mark_success("Default.g", connect_ms=100.0)
        pool.mark_success("Default.g", connect_ms=300.0)
        self.assertAlmostEqual(pool.snapshot()[0]["avg_connect_ms"], 200.0, places=1)

    def test_failed_probe_marks_last_failure_at(self) -> None:
        pool = ExitPool([G_EXIT], max_failures=1)
        pool.mark_failure("Default.g", "connect_failed")
        row = pool.snapshot()[0]
        self.assertGreater(row["last_failure_at"], 0)
        self.assertEqual(row["last_success_at"], 0)


# ----------------------------------------------------------------------
# 5. 探测判定口径 + 短冷却 + 全灭兜底 + 限并发（2026-12 事故修复）
# ----------------------------------------------------------------------
#: 假指纹：`_http_probe` 只用 `impersonate` / `user_agent` 两个字段
_STUB_FINGERPRINT = SimpleNamespace(impersonate="chrome", user_agent="UA-probe-test")

#: 探测端点（无需鉴权、不触发 WAF）
EXPECTED_PROBE_URL = "https://chat.qwen.ai/api/v2/models/"


class _StubHttpResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _StubCurlSession:
    """替身 `curl_cffi.requests.AsyncSession`：固定返回一个状态码，记录请求。"""

    status_code = 200
    body = "{}"
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        _StubCurlSession.calls.append({"kwargs": kwargs})

    async def get(self, url, headers=None, timeout=None):
        _StubCurlSession.calls[-1].update({"url": url, "headers": headers or {}, "timeout": timeout})
        return _StubHttpResponse(self.status_code, self.body)

    async def close(self) -> None:
        self.closed = True


class ProbeJudgementTests(unittest.IsolatedAsyncioTestCase):
    """需求 1~4：宽预算 + 限并发 + 只看连通性 + 短冷却 + 全灭兜底。"""

    def setUp(self) -> None:
        _StubCurlSession.calls = []
        _StubCurlSession.status_code = 200
        _StubCurlSession.body = "{}"

    async def test_only_connect_failure_exit_cools_and_for_120s(self) -> None:
        """3 个出口，1 个 ProxyError → 只有它冷却，另 2 个 ready，冷却 120s 不是 600s。"""

        async def fake_fetch(proxy_url: str, _timeout: float) -> bool:
            if exit_id_from_url(proxy_url) == "Default.c":
                raise OSError("ProxyError: socks5 connect failed")
            return True

        pool = ExitPool(
            [G_EXIT, J_EXIT, C_EXIT],
            max_failures=1,
            cooldown_seconds=600,  # 真实请求的重罚：探测**不该**用它
            probe_cooldown_seconds=120,  # 探测专用短冷却
            probe_fetch=fake_fetch,
        )
        results = await pool.probe_all(limit_seconds=5.0)
        by_id = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(len(results), 3)
        self.assertEqual(by_id["Default.g"]["state"], STATE_READY)
        self.assertEqual(by_id["Default.j"]["state"], STATE_READY)
        self.assertEqual(by_id["Default.c"]["state"], STATE_COOLING)
        # 冷却时长必须是 120s 量级，绝不能是 600s
        remaining = by_id["Default.c"]["cooldown_remaining"]
        self.assertGreater(remaining, 100.0)
        self.assertLessEqual(remaining, 120.0)
        dead = [row for row in results if not row["ok"]]
        self.assertEqual([row["id"] for row in dead], ["Default.c"])
        self.assertEqual(dead[0]["reason"], "proxy_error")

    async def test_probe_failure_does_not_touch_failure_counter(self) -> None:
        """探测失败**不增加** `failures`（探测是猜测，不该和真实请求失败累加）。"""

        async def fake_fetch(proxy_url: str, _timeout: float) -> bool:
            return exit_id_from_url(proxy_url) != "Default.c"

        pool = ExitPool(
            [G_EXIT, J_EXIT, C_EXIT],
            max_failures=2,
            cooldown_seconds=600,
            probe_fetch=fake_fetch,
        )
        await pool.probe_all(limit_seconds=5.0)
        by_id = {row["id"]: row for row in pool.snapshot()}
        self.assertEqual(by_id["Default.c"]["failures"], 0)
        self.assertEqual(by_id["Default.c"]["state"], STATE_COOLING)
        # 只探测一次也不会因为「达阈值」把计数推上去
        self.assertEqual(by_id["Default.g"]["failures"], 0)

    async def test_http_403_and_429_count_as_alive(self) -> None:
        """出口返回 403 / 429 → 判**活**（网络通，是上游在拒绝，不是出口死）。"""
        for status in (403, 429, 401, 200):
            _StubCurlSession.calls = []
            _StubCurlSession.status_code = status
            pool = ExitPool([G_EXIT], max_failures=1, cooldown_seconds=600)
            with patch("curl_cffi.requests.AsyncSession", _StubCurlSession), patch(
                "backend.core.browser_fingerprint.fingerprint_for_account",
                lambda _account: _STUB_FINGERPRINT,
            ):
                self.assertTrue(await pool.probe("Default.g"), f"status={status} 应判活")
            row = pool.snapshot()[0]
            self.assertEqual(row["state"], STATE_READY, f"status={status} 不该冷却")
            self.assertEqual(row["failures"], 0)
            self.assertEqual(row["last_error"], "")
            self.assertGreater(row["last_success_at"], 0)
            # 探测目标是无鉴权的轻量端点
            self.assertEqual(_StubCurlSession.calls[-1]["url"], EXPECTED_PROBE_URL)

    async def test_waf_page_still_counts_as_alive(self) -> None:
        """WAF 拼图页也是 HTTP 响应 → 出口网络通，判活（只记 last_error 供观测）。"""
        _StubCurlSession.status_code = 200
        _StubCurlSession.body = "<!DOCTYPE html><script>window.__aliyun_waf_aa=1;</script>"
        pool = ExitPool([G_EXIT], max_failures=1)
        with patch("curl_cffi.requests.AsyncSession", _StubCurlSession), patch(
            "backend.core.browser_fingerprint.fingerprint_for_account",
            lambda _account: _STUB_FINGERPRINT,
        ):
            self.assertTrue(await pool.probe("Default.g"))
        self.assertEqual(pool.snapshot()[0]["state"], STATE_READY)

    async def test_connect_error_is_dead(self) -> None:
        """真的连不上（ProxyError / DNS / 超时）→ 判死 + 短冷却。"""

        class _ExplodingSession(_StubCurlSession):
            async def get(self, url, headers=None, timeout=None):
                del url, headers, timeout
                raise OSError("Could not resolve host: chat.qwen.ai")

        pool = ExitPool([G_EXIT], max_failures=1, cooldown_seconds=600, probe_cooldown_seconds=120)
        with patch("curl_cffi.requests.AsyncSession", _ExplodingSession), patch(
            "backend.core.browser_fingerprint.fingerprint_for_account",
            lambda _account: _STUB_FINGERPRINT,
        ):
            self.assertFalse(await pool.probe("Default.g"))
        row = pool.snapshot()[0]
        self.assertEqual(row["state"], STATE_COOLING)
        self.assertIn(row["last_error"], ("dns_failed", "connect_failed", "proxy_error"))
        self.assertLessEqual(row["cooldown_remaining"], 120.0)
        self.assertGreater(row["cooldown_remaining"], 100.0)
        self.assertEqual(row["failures"], 0)

    async def test_all_exits_failing_keeps_pool_available(self) -> None:
        """全部探测失败 → 全部 available + WARNING（不惩罚整个池子）。"""

        async def dead_fetch(_proxy_url: str, _timeout: float) -> bool:
            raise OSError("socks5 connect failed")

        pool = ExitPool(
            [G_EXIT, J_EXIT, C_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            probe_cooldown_seconds=120,
            probe_fetch=dead_fetch,
        )
        with self.assertLogs("qwen2api.exit_pool", level="WARNING") as captured:
            results = await pool.probe_all(limit_seconds=5.0)
        self.assertTrue(any(
            "[ExitProbe] all exits failed probe" in line and "keeping pool available" in line
            for line in captured.output
        ))
        self.assertEqual(len(results), 3)
        self.assertTrue(all(row["ok"] for row in results))
        self.assertTrue(all(row.get("assumed_alive") for row in results))
        by_id = {row["id"]: row for row in pool.snapshot()}
        for exit_id in ("Default.g", "Default.j", "Default.c"):
            self.assertEqual(by_id[exit_id]["state"], STATE_READY)
            self.assertEqual(by_id[exit_id]["failures"], 0)
        # 池子整体可用：pick() 必须还能挑到出口
        self.assertIsNotNone(pool.pick())

    async def test_probe_concurrency_is_limited_to_two(self) -> None:
        """并发度 = 2：fake 层记录同时在跑的探测数，断言 ≤ 2。"""
        running = 0
        peak = 0

        async def counting_fetch(_proxy_url: str, _timeout: float) -> bool:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await asyncio.sleep(0.05)
                return True
            finally:
                running -= 1

        exits = [
            f"socks5h://Default.{suffix}@10.6.6.114:2260"
            for suffix in ("g", "j", "c", "e", "l", "d")
        ]
        pool = ExitPool(exits, probe_fetch=counting_fetch)
        self.assertEqual(pool.probe_concurrency, 2)
        results = await pool.probe_all(limit_seconds=10.0)
        self.assertEqual(len(results), 6)
        self.assertLessEqual(peak, 2)
        self.assertGreaterEqual(peak, 1)

    def test_startup_probe_defaults_are_relaxed(self) -> None:
        """默认预算：单出口 20s / 总 45s / 并发 2 / 探测冷却 120s。"""
        from backend.services.exit_pool import (
            DEFAULT_PROBE_COOLDOWN_SECONDS,
            DEFAULT_PROBE_CONCURRENCY,
            DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS,
            DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS,
        )

        self.assertEqual(DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS, 20.0)
        self.assertEqual(DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS, 45.0)
        self.assertEqual(DEFAULT_PROBE_CONCURRENCY, 2)
        self.assertEqual(DEFAULT_PROBE_COOLDOWN_SECONDS, 120.0)
        pool = ExitPool.from_settings(_settings())
        self.assertIsNotNone(pool)
        self.assertEqual(pool.startup_probe_timeout_seconds, 20.0)
        self.assertEqual(pool.probe_concurrency, 2)
        self.assertEqual(pool.probe_cooldown_seconds, 120.0)
        # 手工探测超时仍走 UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS（两者不混用）
        self.assertEqual(pool.probe_timeout_seconds, 10.0)


# ----------------------------------------------------------------------
# 7. 探测的建连超时必须用「探测预算」，不能套用正常请求的
#    `EXIT_CONNECT_TIMEOUT_SECONDS`（2026-12 socks5h 冷启动握手事故）
# ----------------------------------------------------------------------
#: 生产实测：socks5h 冷启动握手 + TLS 拿到首字节要 8.2s（curl 28 Connection
#: timed out @5.0s → 换 20s 预算后 200 @8.2s）。任何小于它秒数的建连上限都会
#: 把健康出口判死。
COLD_START_HANDSHAKE_SECONDS = 8.2


class ProbeUsesItsOwnConnectBudgetTests(unittest.IsolatedAsyncioTestCase):
    """探测建连预算 = 探测超时（20s），而不是正常请求的 5s。

    旧实现 `connect_read_timeout(read_timeout, self.connect_timeout_seconds)` 下发
    `(5, 20)`：出口在 5s 就被掐断 → 6 个健康出口全进冷却，池子形同废掉。
    """

    def setUp(self) -> None:
        _StubCurlSession.calls = []
        _StubCurlSession.status_code = 200
        _StubCurlSession.body = "{}"
        # 固定「curl_cffi 支持 timeout 元组」：全量跑测试时能力位可能被别的用例改过。
        self._capability = patch.object(http_timeout, "_SUPPORTS_TIMEOUT_TUPLE", True)
        self._capability.start()
        self._fingerprint = patch(
            "backend.core.browser_fingerprint.fingerprint_for_account",
            lambda _account: _STUB_FINGERPRINT,
        )
        self._fingerprint.start()

    def tearDown(self) -> None:
        self._fingerprint.stop()
        self._capability.stop()
        http_timeout.reset_capability_cache()

    async def test_probe_timeout_tuple_uses_probe_budget_for_both_parts(self) -> None:
        """fake 层记录的 timeout 元组两个分量都等于探测超时：`(20, 20)` 而非 `(5, 20)`。"""
        pool = ExitPool(
            [G_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            connect_timeout_seconds=5.0,  # 正常请求的建连上限：探测**不许**用它
            probe_timeout_seconds=20.0,  # 探测自己的预算
        )
        with patch("curl_cffi.requests.AsyncSession", _StubCurlSession):
            self.assertTrue(await pool.probe("Default.g"))

        call = _StubCurlSession.calls[-1]
        # 会话构造时就下发同一份预算（不靠 get() 覆盖）
        self.assertEqual(_split_timeout(call["kwargs"]["timeout"]), (20.0, 20.0))
        # 请求级 timeout 也必须是 (20, 20)
        self.assertEqual(_split_timeout(call["timeout"]), (20.0, 20.0))
        self.assertEqual(pool.connect_timeout_seconds, 5.0)  # 没被探测改写

    async def test_probe_survives_five_second_connect_cap(self) -> None:
        """`EXIT_CONNECT_TIMEOUT_SECONDS=5` + 探测 20s：冷启动握手不被 5s 掐断。"""

        class _ColdStartSession(_StubCurlSession):
            """按 curl 的行为模拟冷启动：建连预算不足就超时，够就返回 200。"""

            async def get(self, url, headers=None, timeout=None):
                _StubCurlSession.calls[-1].update({"url": url, "timeout": timeout})
                connect, _read = _split_timeout(timeout)
                if connect < COLD_START_HANDSHAKE_SECONDS:
                    # curl 28：连不上，且只等满建连预算就放弃。
                    raise OSError("Failed to connect: Connection timed out")
                return _StubHttpResponse(_StubCurlSession.status_code, _StubCurlSession.body)

        pool = ExitPool(
            [G_EXIT],
            max_failures=1,
            cooldown_seconds=600,
            connect_timeout_seconds=5.0,
            probe_timeout_seconds=20.0,
        )
        with patch("curl_cffi.requests.AsyncSession", _ColdStartSession):
            ok = await pool.probe("Default.g")

        self.assertTrue(ok, "探测预算 20s 时冷启动握手必须活下来（旧实现 5s 掐死）")
        connect_budget, read_budget = _split_timeout(_StubCurlSession.calls[-1]["timeout"])
        # 只有两个分量都放大到探测预算，8.2s 的握手才可能完成。
        self.assertGreaterEqual(connect_budget, COLD_START_HANDSHAKE_SECONDS)
        self.assertGreaterEqual(read_budget, COLD_START_HANDSHAKE_SECONDS)
        row = pool.snapshot()[0]
        self.assertEqual(row["state"], STATE_READY)
        self.assertEqual(row["failures"], 0)
        self.assertGreater(row["last_success_at"], 0)


if __name__ == "__main__":
    unittest.main()
