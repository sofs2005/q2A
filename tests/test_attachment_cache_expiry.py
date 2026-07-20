import calendar
import unittest

from backend.services.context_attachment_manager import _resolve_attachment_cache_expires_at


class AttachmentCacheExpiryTests(unittest.TestCase):
    def test_cache_expires_before_signed_url_expiry(self) -> None:
        # 用与 x-oss-date 一致的时间作为 created_at，避免签名日期漂移干扰断言
        created_at = float(calendar.timegm((2026, 7, 20, 0, 0, 0, 0, 0, 0)))
        remote_meta = {
            "url_expires_at": created_at + 300.0,
            "remote_ref": {
                "url": (
                    "https://qwen-webui-prod.oss-accelerate.aliyuncs.com/u/f.txt"
                    "?x-oss-date=20260720T000000Z&x-oss-expires=300&x-oss-signature=abc"
                )
            },
        }
        # session TTL 很长时，应以签名 URL 安全窗口为准（300 - 30 = 270）
        expires = _resolve_attachment_cache_expires_at(
            remote_meta,
            created_at=created_at,
            session_ttl_seconds=1800,
        )
        self.assertLess(expires, created_at + 300.0)
        self.assertAlmostEqual(expires, created_at + 270.0, places=3)

    def test_session_ttl_can_be_shorter_than_url_ttl(self) -> None:
        created_at = 1_700_000_000.0
        remote_meta = {
            "remote_ref": {
                "url": "https://example.com/obj?x-oss-expires=300&x-oss-signature=abc"
            }
        }
        expires = _resolve_attachment_cache_expires_at(
            remote_meta,
            created_at=created_at,
            session_ttl_seconds=60,
        )
        self.assertEqual(expires, created_at + 60.0)


if __name__ == "__main__":
    unittest.main()
