import sys
import types
import unittest

if "curl_cffi" not in sys.modules:
    fake_curl_cffi = types.ModuleType("curl_cffi")
    fake_curl_cffi_requests = types.ModuleType("curl_cffi.requests")

    class AsyncSession:
        pass

    fake_curl_cffi_requests.AsyncSession = AsyncSession
    fake_curl_cffi.requests = fake_curl_cffi_requests
    sys.modules["curl_cffi"] = fake_curl_cffi
    sys.modules["curl_cffi.requests"] = fake_curl_cffi_requests

from backend.core.browser_fingerprint import SUPPORTED_BROWSER_FINGERPRINTS, fingerprint_for_email


class BrowserFingerprintTests(unittest.TestCase):
    def test_supported_pool_excludes_edge_and_uses_three_families(self) -> None:
        families = {fingerprint.browser for fingerprint in SUPPORTED_BROWSER_FINGERPRINTS}

        self.assertEqual(families, {"chrome", "firefox", "safari"})
        self.assertNotIn("edge", families)
        self.assertEqual(len(SUPPORTED_BROWSER_FINGERPRINTS), 10)

    def test_fingerprint_for_email_is_stable_and_uses_supported_profile(self) -> None:
        first = fingerprint_for_email("alice@example.com")
        second = fingerprint_for_email("alice@example.com")
        other = fingerprint_for_email("bob@example.com")

        self.assertEqual(first, second)
        self.assertIn(first, SUPPORTED_BROWSER_FINGERPRINTS)
        self.assertIn(other, SUPPORTED_BROWSER_FINGERPRINTS)

    def test_build_headers_includes_browser_identity_and_token(self) -> None:
        chrome = next(fingerprint for fingerprint in SUPPORTED_BROWSER_FINGERPRINTS if fingerprint.browser == "chrome")

        headers = chrome.build_headers(token="token-123")

        self.assertEqual(headers["Authorization"], "Bearer token-123")
        self.assertEqual(headers["User-Agent"], chrome.user_agent)
        self.assertEqual(headers["sec-ch-ua"], chrome.sec_ch_ua)
        self.assertEqual(headers["sec-fetch-site"], "same-origin")
        self.assertEqual(headers["Referer"], chrome.referer)
        self.assertEqual(headers["Origin"], chrome.origin)


if __name__ == "__main__":
    unittest.main()
