from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from curl_cffi.requests import AsyncSession

BASE_URL = "https://chat.qwen.ai"


@dataclass(frozen=True)
class BrowserFingerprint:
    id: str
    browser: str
    impersonate: str
    user_agent: str
    platform: str
    sec_ch_ua: str = ""
    sec_ch_ua_mobile: str = "?0"
    referer: str = f"{BASE_URL}/"
    origin: str = BASE_URL

    def build_headers(
        self,
        *,
        token: str | None = None,
        cookies: str | None = None,
        referer: str | None = None,
        content_type: str | None = "application/json",
        accept: str = "application/json, text/plain, */*",
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": referer or self.referer,
            "Origin": self.origin,
            "Connection": "keep-alive",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }
        if content_type:
            headers["Content-Type"] = content_type
        if self.sec_ch_ua:
            headers["sec-ch-ua"] = self.sec_ch_ua
            headers["sec-ch-ua-mobile"] = self.sec_ch_ua_mobile
            headers["sec-ch-ua-platform"] = self.platform
        if cookies:
            headers["Cookie"] = cookies
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers:
            headers.update(extra_headers)
        return headers


SUPPORTED_BROWSER_FINGERPRINTS: tuple[BrowserFingerprint, ...] = (
    BrowserFingerprint(
        id="chrome146_windows",
        browser="chrome",
        impersonate="chrome146",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        platform='"Windows"',
        sec_ch_ua='"Chromium";v="146", "Google Chrome";v="146", "Not_A Brand";v="99"',
    ),
    BrowserFingerprint(
        id="chrome145_macos",
        browser="chrome",
        impersonate="chrome145",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        platform='"macOS"',
        sec_ch_ua='"Chromium";v="145", "Google Chrome";v="145", "Not_A Brand";v="99"',
    ),
    BrowserFingerprint(
        id="chrome142_linux",
        browser="chrome",
        impersonate="chrome142",
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        platform='"Linux"',
        sec_ch_ua='"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    ),
    BrowserFingerprint(
        id="firefox147_windows",
        browser="firefox",
        impersonate="firefox147",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
        platform='"Windows"',
    ),
    BrowserFingerprint(
        id="firefox144_macos",
        browser="firefox",
        impersonate="firefox144",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 15.7; rv:144.0) Gecko/20100101 Firefox/144.0",
        platform='"macOS"',
    ),
    BrowserFingerprint(
        id="firefox135_linux",
        browser="firefox",
        impersonate="firefox135",
        user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:135.0) Gecko/20100101 Firefox/135.0",
        platform='"Linux"',
    ),
    BrowserFingerprint(
        id="safari2601_macos",
        browser="safari",
        impersonate="safari2601",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Safari/605.1.15",
        platform='"macOS"',
    ),
    BrowserFingerprint(
        id="safari260_macos",
        browser="safari",
        impersonate="safari260",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
        platform='"macOS"',
    ),
    BrowserFingerprint(
        id="safari184_macos",
        browser="safari",
        impersonate="safari184",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 15_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.4 Safari/605.1.15",
        platform='"macOS"',
    ),
    BrowserFingerprint(
        id="chrome136_windows",
        browser="chrome",
        impersonate="chrome136",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        platform='"Windows"',
        sec_ch_ua='"Chromium";v="136", "Google Chrome";v="136", "Not_A Brand";v="99"',
    ),
)

_FINGERPRINT_BY_ID = {fingerprint.id: fingerprint for fingerprint in SUPPORTED_BROWSER_FINGERPRINTS}
_DEFAULT_FINGERPRINT = SUPPORTED_BROWSER_FINGERPRINTS[0]
_sessions: dict[str, AsyncSession] = {}
_session_lock: asyncio.Lock | None = None


def fingerprint_id_for_email(email: str) -> str:
    normalized = str(email or "").strip().lower()
    if not normalized:
        return _DEFAULT_FINGERPRINT.id
    digest = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
    return SUPPORTED_BROWSER_FINGERPRINTS[int(digest, 16) % len(SUPPORTED_BROWSER_FINGERPRINTS)].id


def fingerprint_for_id(fingerprint_id: str | None) -> BrowserFingerprint:
    return _FINGERPRINT_BY_ID.get(str(fingerprint_id or "").strip(), _DEFAULT_FINGERPRINT)


def fingerprint_for_email(email: str) -> BrowserFingerprint:
    return fingerprint_for_id(fingerprint_id_for_email(email))


def fingerprint_for_account(account: Any | None) -> BrowserFingerprint:
    if account is None:
        return _DEFAULT_FINGERPRINT
    fingerprint_id = str(getattr(account, "fingerprint_id", "") or "").strip()
    if fingerprint_id:
        return fingerprint_for_id(fingerprint_id)
    return fingerprint_for_email(str(getattr(account, "email", "") or ""))


def new_session(fingerprint: BrowserFingerprint, *, timeout: float | None = None) -> AsyncSession:
    from backend.core.config import settings

    return AsyncSession(
        impersonate=fingerprint.impersonate,
        timeout=timeout if timeout is not None else settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS,
        allow_redirects=True,
    )


async def get_session(fingerprint: BrowserFingerprint) -> AsyncSession:
    global _session_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    session = _sessions.get(fingerprint.impersonate)
    if session is not None:
        return session
    async with _session_lock:
        session = _sessions.get(fingerprint.impersonate)
        if session is not None:
            return session
        session = new_session(fingerprint)
        _sessions[fingerprint.impersonate] = session
        return session


async def close_all_sessions() -> None:
    sessions = list(_sessions.values())
    _sessions.clear()
    for session in sessions:
        close = getattr(session, "close", None)
        if close is not None:
            await close()
