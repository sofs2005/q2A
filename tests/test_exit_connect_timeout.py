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
        """总预算耗尽时按不可用处理，不会挂住整个探测。"""

        async def stuck_fetch(_proxy_url: str, _timeout: float) -> bool:
            await asyncio.sleep(5.0)
            return True

        pool = ExitPool([G_EXIT, J_EXIT], probe_fetch=stuck_fetch, probe_timeout_seconds=5.0)
        started = time.monotonic()
        results = await pool.probe_all(limit_seconds=0.2)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertTrue(all(not row["ok"] for row in results))

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
    def test_pick_prefers_latest_success(self) -> None:
        pool = ExitPool([G_EXIT, J_EXIT, C_EXIT])
        pool.mark_success("Default.g")
        pool.mark_success("Default.j")  # 更晚成功
        self.assertEqual(pool.pick(), J_EXIT)
        self.assertEqual(pool.pick(), J_EXIT)  # 一直命中同一个，不再从零试错
        self.assertEqual(pool.pick(), J_EXIT)

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


if __name__ == "__main__":
    unittest.main()
