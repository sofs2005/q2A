"""账号池轮询公平性回归测试。

覆盖缺陷①：least_loaded 排序键把手动 cookies 当最高优先级，
导致带 cookies 的账号被无限连选，其他账号长期饿死。
"""
import time
import unittest

from backend.core.account_pool import Account, AccountPool


def _acc(email: str, cookies: str = "", last_request_started: float = 0.0) -> Account:
    return Account(
        email=email,
        token="tok-" + email,
        cookies=cookies,
        last_request_started=last_request_started,
    )


class _FakeDB:
    def __init__(self, data):
        self.data = data
        self.saved = None

    async def load(self):
        return self.data

    async def save(self, data):
        self.saved = data


class AccountRotationCookiesBiasTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefers_least_recently_used_over_cookies(self):
        """带 cookies 但刚用过的账号，不应压过久未使用的无 cookies 账号。"""
        now = time.time()
        pool = AccountPool(db=None)
        pool.accounts = [
            _acc("a@test", cookies="sessionid=x", last_request_started=now - 5),
            _acc("b@test", cookies="", last_request_started=now - 600),
        ]
        acc = await pool.acquire()
        self.assertIsNotNone(acc)
        # 公平轮询应选最久未使用的 b，而非因 cookies 恒排第一的 a
        self.assertEqual(acc.email, "b@test")

    async def test_rotation_selects_least_recently_used_without_cookies(self):
        """无 cookies 时应稳定选中最久未使用的账号（确保未破坏正常轮询）。"""
        now = time.time()
        pool = AccountPool(db=None)
        pool.accounts = [
            _acc("a@test", last_request_started=now - 10),
            _acc("b@test", last_request_started=now - 600),
            _acc("c@test", last_request_started=now - 100),
        ]
        acc = await pool.acquire()
        self.assertEqual(acc.email, "b@test")

    async def test_load_clears_legacy_manual_cookies(self):
        """加载账号时应清空历史手动 cookies，避免再次参与调度或请求。"""
        db = _FakeDB([
            {"email": "a@test", "token": "tok-a", "cookies": "legacy_cookie=1"},
            {"email": "b@test", "token": "tok-b", "cookies": ""},
        ])
        pool = AccountPool(db=db)

        await pool.load()

        self.assertEqual([acc.cookies for acc in pool.accounts], ["", ""])
        self.assertIsNotNone(db.saved)
        self.assertEqual([item["cookies"] for item in db.saved], ["", ""])


if __name__ == "__main__":
    unittest.main()
