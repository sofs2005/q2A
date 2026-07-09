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

from backend.core.account_pool import Account, AccountPool
from backend.core.config import settings


class AccountPoolDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_min_interval_ms = settings.ACCOUNT_MIN_INTERVAL_MS
        self.original_busy_timeout = getattr(settings, "ACCOUNT_BUSY_TIMEOUT_SECONDS", None)

    async def asyncTearDown(self) -> None:
        settings.ACCOUNT_MIN_INTERVAL_MS = self.original_min_interval_ms
        if self.original_busy_timeout is None:
            if hasattr(settings, "ACCOUNT_BUSY_TIMEOUT_SECONDS"):
                delattr(settings, "ACCOUNT_BUSY_TIMEOUT_SECONDS")
        else:
            settings.ACCOUNT_BUSY_TIMEOUT_SECONDS = self.original_busy_timeout

    def _pool(self, *accounts: Account, max_inflight: int = 1) -> AccountPool:
        pool = AccountPool(db=None, max_inflight=max_inflight)
        pool.accounts = list(accounts)
        return pool

    def test_account_diagnostics_explains_ready_and_blocked_states(self) -> None:
        settings.ACCOUNT_MIN_INTERVAL_MS = 1000
        ready = Account(email="ready@example.com")
        busy = Account(email="busy@example.com")
        busy.inflight = 1
        cooldown = Account(email="cooldown@example.com")
        cooldown.last_request_started = 100.0
        rate_limited = Account(email="rate@example.com")
        rate_limited.rate_limited_until = 130.0
        invalid = Account(email="invalid@example.com")
        invalid.valid = False

        pool = self._pool(ready, busy, cooldown, rate_limited, invalid)

        with patch("backend.core.account_pool.time.time", return_value=100.2):
            diagnostics = pool.account_diagnostics()

        by_email = {item["email"]: item for item in diagnostics}
        self.assertTrue(by_email["ready@example.com"]["ready"])
        self.assertEqual(by_email["ready@example.com"]["selection_block_reason"], "ready")
        self.assertTrue(by_email["ready@example.com"]["capacity_available"])
        self.assertEqual(by_email["ready@example.com"]["next_available_in"], 0.0)

        self.assertFalse(by_email["busy@example.com"]["ready"])
        self.assertEqual(by_email["busy@example.com"]["selection_block_reason"], "busy")
        self.assertFalse(by_email["busy@example.com"]["capacity_available"])

        self.assertFalse(by_email["cooldown@example.com"]["ready"])
        self.assertEqual(by_email["cooldown@example.com"]["selection_block_reason"], "cooldown")
        self.assertAlmostEqual(by_email["cooldown@example.com"]["next_available_in"], 0.8)

        self.assertFalse(by_email["rate@example.com"]["ready"])
        self.assertTrue(by_email["rate@example.com"]["is_rate_limited"])
        self.assertEqual(by_email["rate@example.com"]["selection_block_reason"], "rate_limited")
        self.assertAlmostEqual(by_email["rate@example.com"]["next_available_in"], 29.8)

        self.assertFalse(by_email["invalid@example.com"]["ready"])
        self.assertEqual(by_email["invalid@example.com"]["selection_block_reason"], "invalid")

    async def test_acquire_records_least_loaded_selection_diagnostics(self) -> None:
        first = Account(email="first@example.com")
        first.last_request_started = 10.0
        second = Account(email="second@example.com")
        second.last_request_started = 20.0
        pool = self._pool(second, first)

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertEqual(selected.email, "first@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["strategy"], "least_loaded")
        self.assertEqual(pool.last_acquire_diagnostics["selected_email"], "first@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["ready_count"], 2)

    async def test_acquire_prefers_lower_inflight_before_older_usage(self) -> None:
        busy_old = Account(email="busy-old@example.com")
        busy_old.inflight = 1
        busy_old.last_request_started = 10.0
        idle_new = Account(email="idle-new@example.com")
        idle_new.inflight = 0
        idle_new.last_request_started = 90.0
        pool = self._pool(busy_old, idle_new, max_inflight=2)

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertEqual(selected.email, "idle-new@example.com")
        self.assertEqual(idle_new.inflight, 1)
        self.assertEqual(busy_old.inflight, 1)
        self.assertEqual(pool.last_acquire_diagnostics["strategy"], "least_loaded")

    async def test_acquire_uses_email_tie_breaker_for_equal_load_and_usage(self) -> None:
        later_email = Account(email="z-last@example.com")
        later_email.last_request_started = 10.0
        later_email.last_used = 20.0
        earlier_email = Account(email="a-first@example.com")
        earlier_email.last_request_started = 10.0
        earlier_email.last_used = 20.0
        pool = self._pool(later_email, earlier_email)

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertEqual(selected.email, "a-first@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["strategy"], "least_loaded")

    async def test_acquire_uses_email_tiebreaker_for_equal_load_and_usage(self) -> None:
        no_cookie = Account(email="a-no-cookie@example.com", token="token-1")
        no_cookie.last_request_started = 10.0
        no_cookie.last_used = 20.0
        with_cookie = Account(email="z-cookie@example.com", token="token-2", cookies="aui=1; cna=2")
        with_cookie.last_request_started = 10.0
        with_cookie.last_used = 20.0
        pool = self._pool(no_cookie, with_cookie)

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertEqual(selected.email, "a-no-cookie@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["strategy"], "least_loaded")

    async def test_acquire_preferred_records_preferred_and_fallback_diagnostics(self) -> None:
        preferred = Account(email="preferred@example.com")
        fallback = Account(email="fallback@example.com")
        pool = self._pool(preferred, fallback)

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire_preferred("preferred@example.com")

        self.assertEqual(selected.email, "preferred@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["strategy"], "preferred")
        self.assertEqual(pool.last_acquire_diagnostics["selected_email"], "preferred@example.com")

        pool.release(preferred)
        preferred.inflight = 1

        with patch("backend.core.account_pool.time.time", return_value=101.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire_preferred("preferred@example.com")

        self.assertEqual(selected.email, "fallback@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["strategy"], "fallback")
        self.assertEqual(pool.last_acquire_diagnostics["preferred_email"], "preferred@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["selected_email"], "fallback@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["preferred_block_reason"], "busy")

    async def test_acquire_wait_records_timeout_snapshot(self) -> None:
        busy = Account(email="busy@example.com")
        busy.inflight = 1
        pool = self._pool(busy)

        with patch("backend.core.account_pool.time.time", return_value=100.0):
            selected = await pool.acquire_wait(timeout=0.0)

        self.assertIsNone(selected)
        self.assertEqual(pool.last_acquire_wait_diagnostics["result"], "timeout")
        self.assertEqual(pool.last_acquire_wait_diagnostics["timeout"], 0.0)
        self.assertEqual(pool.last_acquire_wait_diagnostics["snapshot"]["ready"], 0)
        self.assertEqual(pool.last_acquire_wait_diagnostics["snapshot"]["blocked_reasons"], {"busy": 1})

    async def test_acquire_reclaims_stale_busy_account_after_timeout(self) -> None:
        settings.ACCOUNT_BUSY_TIMEOUT_SECONDS = 30
        stale = Account(email="stale@example.com")
        stale.inflight = 1
        stale.last_request_started = 100.0
        pool = self._pool(stale)

        with patch("backend.core.account_pool.time.time", return_value=131.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertIs(selected, stale)
        self.assertEqual(stale.inflight, 1)
        self.assertEqual(pool.last_acquire_diagnostics["selected_email"], "stale@example.com")

    async def test_disabled_account_is_not_acquired(self) -> None:
        disabled = Account(email="disabled@example.com")
        fallback = Account(email="fallback@example.com")
        pool = self._pool(disabled, fallback)
        pool.db = SimpleNamespace(save=AsyncMock())

        await pool.disable_accounts(["disabled@example.com"])

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertEqual(disabled.get_status_code(), "disabled")
        self.assertFalse(disabled.valid)
        self.assertEqual(selected.email, "fallback@example.com")
        self.assertEqual(pool.last_acquire_diagnostics["blocked_reasons"], {"disabled": 1})

    async def test_enabled_account_returns_to_valid_and_can_be_acquired(self) -> None:
        account = Account(email="disabled@example.com")
        account.valid = False
        account.status_code = "disabled"
        account.activation_pending = True
        pool = self._pool(account)
        pool.db = SimpleNamespace(save=AsyncMock())

        await pool.enable_accounts(["disabled@example.com"])

        with patch("backend.core.account_pool.time.time", return_value=100.0), patch("backend.core.account_pool._jitter_seconds", return_value=0.0):
            selected = await pool.acquire()

        self.assertEqual(account.get_status_code(), "valid")
        self.assertTrue(account.valid)
        self.assertFalse(account.activation_pending)
        self.assertIs(selected, account)

    def test_to_dict_persists_waf_cookies_for_round_trip(self) -> None:
        """to_dict 必须序列化 waf_cookies/expires_at，否则落盘后 acw_tc 丢失。"""
        account = Account(email="waf@example.com", token="tok")
        account.waf_cookies = "acw_tc=persist-me"
        account.waf_cookies_expires_at = 1234567890.0

        payload = account.to_dict()

        self.assertIn("waf_cookies", payload)
        self.assertIn("waf_cookies_expires_at", payload)
        self.assertEqual(payload["waf_cookies"], "acw_tc=persist-me")
        self.assertEqual(payload["waf_cookies_expires_at"], 1234567890.0)

        restored = Account(**payload)
        self.assertEqual(restored.waf_cookies, "acw_tc=persist-me")
        self.assertEqual(restored.waf_cookies_expires_at, 1234567890.0)

    def test_file_upload_blocked_state_is_independent_from_chat_rate_limit(self) -> None:
        """上传超限状态独立于 chat 限流：上传被封的号 chat 仍可用。"""
        account = Account(email="up@example.com", token="tok")
        account.file_upload_blocked_until = 200.0

        with patch("backend.core.account_pool.time.time", return_value=100.0):
            self.assertTrue(account.is_file_upload_blocked())
            self.assertFalse(account.is_rate_limited())
            self.assertTrue(account.is_available())

    def test_to_dict_persists_file_upload_blocked_until_for_round_trip(self) -> None:
        """to_dict 必须序列化 file_upload_blocked_until，否则落盘后上传限额丢失。"""
        account = Account(email="up@example.com", token="tok")
        account.file_upload_blocked_until = 1234567890.0

        payload = account.to_dict()

        self.assertIn("file_upload_blocked_until", payload)
        self.assertEqual(payload["file_upload_blocked_until"], 1234567890.0)

        restored = Account(**payload)
        self.assertEqual(restored.file_upload_blocked_until, 1234567890.0)

    def test_mark_file_upload_limited_sets_block_window(self) -> None:
        """标记上传超限后，在 retry_after 秒内该号 is_file_upload_blocked。"""
        account = Account(email="up@example.com", token="tok")
        pool = self._pool(account)

        with patch("backend.core.account_pool.time.time", return_value=100.0):
            pool.mark_file_upload_limited(account, 3600)
            self.assertTrue(account.is_file_upload_blocked())
            self.assertEqual(account.file_upload_blocked_until, 3700.0)

    def test_mark_file_upload_limited_keeps_chat_available(self) -> None:
        """上传超限不影响 chat：该号仍 is_available、未 rate_limited。"""
        account = Account(email="up@example.com", token="tok")
        pool = self._pool(account)

        with patch("backend.core.account_pool.time.time", return_value=100.0):
            pool.mark_file_upload_limited(account, 3600)
            self.assertFalse(account.is_rate_limited())
            self.assertTrue(account.is_available())

    def test_file_upload_blocked_emails_returns_only_currently_blocked(self) -> None:
        """blocked_emails 只返回当前仍超限的号，过期/未超限的不返回。"""
        blocked = Account(email="blocked@example.com", token="t1")
        blocked.file_upload_blocked_until = 200.0
        ready = Account(email="ready@example.com", token="t2")
        expired = Account(email="expired@example.com", token="t3")
        expired.file_upload_blocked_until = 50.0
        pool = self._pool(blocked, ready, expired)

        with patch("backend.core.account_pool.time.time", return_value=100.0):
            emails = pool.file_upload_blocked_emails()

        self.assertEqual(emails, {"blocked@example.com"})


if __name__ == "__main__":
    unittest.main()
