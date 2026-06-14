import asyncio
import logging
import random
import time
from typing import Optional
from backend.core.database import AsyncJsonDB
from backend.core.config import settings
from backend.core.browser_fingerprint import fingerprint_id_for_email

log = logging.getLogger("qwen2api.accounts")


def _jitter_seconds() -> float:
    low = max(0, settings.REQUEST_JITTER_MIN_MS)
    high = max(low, settings.REQUEST_JITTER_MAX_MS)
    return random.uniform(low, high) / 1000.0


class Account:
    def __init__(
        self,
        email="",
        password="",
        token="",
        cookies="",
        username="",
        activation_pending=False,
        status_code="",
        last_error="",
        **kwargs,
    ):
        self.email = email
        self.password = password
        self.token = token
        self.cookies = cookies
        self.username = username
        self.activation_pending = activation_pending
        stored_valid = kwargs.get("valid")
        self.valid = bool(stored_valid) if stored_valid is not None else not activation_pending and status_code not in ("disabled", "banned", "auth_error", "invalid")
        self.last_used = 0.0
        self.inflight = 0
        self.rate_limited_until = 0.0
        self.healing = False
        self.status_code = status_code or ("pending_activation" if activation_pending else "valid")
        self.last_error = last_error or ""
        self.last_request_started = float(kwargs.get("last_request_started", 0.0) or 0.0)
        self.last_request_finished = float(kwargs.get("last_request_finished", 0.0) or 0.0)
        self.consecutive_failures = int(kwargs.get("consecutive_failures", 0) or 0)
        self.rate_limit_strikes = int(kwargs.get("rate_limit_strikes", 0) or 0)
        self.fingerprint_id = str(kwargs.get("fingerprint_id", "") or "").strip()
        self.waf_cookies = str(kwargs.get("waf_cookies", "") or "").strip()
        self.waf_cookies_expires_at = float(kwargs.get("waf_cookies_expires_at", 0) or 0)
        self.file_upload_blocked_until = float(kwargs.get("file_upload_blocked_until", 0) or 0)

    def is_rate_limited(self) -> bool:
        return self.rate_limited_until > time.time()

    def is_file_upload_blocked(self) -> bool:
        return self.file_upload_blocked_until > time.time()

    def is_available(self) -> bool:
        return self.valid and not self.is_rate_limited()

    def next_available_at(self) -> float:
        min_interval = max(0, settings.ACCOUNT_MIN_INTERVAL_MS) / 1000.0
        return max(self.rate_limited_until, self.last_request_started + min_interval)

    def get_status_code(self) -> str:
        if self.status_code == "disabled":
            return "disabled"
        if self.activation_pending:
            return "pending_activation"
        if self.is_rate_limited():
            return "rate_limited"
        if self.valid:
            return "valid"
        if self.status_code == "banned":
            return "banned"
        if self.status_code == "auth_error":
            return "auth_error"
        return self.status_code or "invalid"

    def get_status_text(self) -> str:
        status_map = {
            "valid": "正常",
            "pending_activation": "待激活",
            "rate_limited": "限流",
            "banned": "封禁",
            "auth_error": "鉴权失败",
            "disabled": "已禁用",
            "invalid": "失效",
            "unknown": "未知",
        }
        return status_map.get(self.get_status_code(), "未知")

    def to_dict(self):
        return {
            "email": self.email,
            "password": self.password,
            "token": self.token,
            "cookies": self.cookies,
            "username": self.username,
            "activation_pending": self.activation_pending,
            "valid": self.valid,
            "status_code": self.status_code,
            "last_error": self.last_error,
            "last_request_started": self.last_request_started,
            "last_request_finished": self.last_request_finished,
            "consecutive_failures": self.consecutive_failures,
            "rate_limit_strikes": self.rate_limit_strikes,
            "fingerprint_id": self.fingerprint_id,
            "waf_cookies": self.waf_cookies,
            "waf_cookies_expires_at": self.waf_cookies_expires_at,
            "file_upload_blocked_until": self.file_upload_blocked_until,
        }


class AccountPool:
    def __init__(self, db: AsyncJsonDB, max_inflight: int = settings.MAX_INFLIGHT_PER_ACCOUNT):
        self.db = db
        self.max_inflight = max_inflight
        self.accounts: list[Account] = []
        self._lock = asyncio.Lock()
        self._waiters: list[asyncio.Event] = []
        self._sticky_email: Optional[str] = None
        self.last_acquire_diagnostics: dict = {}
        self.last_acquire_wait_diagnostics: dict = {}

    async def load(self):
        data = await self.db.load()
        self.accounts = [Account(**d) for d in data] if isinstance(data, list) else []
        migrated = self._assign_missing_fingerprints()
        if migrated:
            await self.save()
        log.info(f"Loaded {len(self.accounts)} upstream account(s)")

    def _assign_missing_fingerprints(self) -> bool:
        changed = False
        for account in self.accounts:
            if account.fingerprint_id:
                continue
            account.fingerprint_id = fingerprint_id_for_email(account.email)
            changed = True
        return changed

    async def save(self):
        await self.db.save([a.to_dict() for a in self.accounts])

    async def add(self, account: Account):
        async with self._lock:
            self.accounts = [a for a in self.accounts if a.email != account.email]
            self.accounts.append(account)
        await self.save()

    async def remove(self, email: str):
        async with self._lock:
            self.accounts = [a for a in self.accounts if a.email != email]
        await self.save()

    def set_max_inflight(self, value: int):
        self.max_inflight = max(1, int(value))

    def get_by_email(self, email: str) -> Optional[Account]:
        return next((a for a in self.accounts if a.email == email), None)

    async def disable_accounts(self, emails: list[str]) -> list[dict]:
        results: list[dict] = []
        changed = False

        async with self._lock:
            accounts_by_email = {account.email: account for account in self.accounts}
            for email in emails:
                account = accounts_by_email.get(email)
                if not account:
                    results.append({"email": email, "status": "skipped", "reason": "missing_account"})
                    continue
                if account.get_status_code() == "disabled":
                    results.append({"email": email, "status": "skipped", "reason": "already_disabled"})
                    continue

                account.valid = False
                account.activation_pending = False
                account.status_code = "disabled"
                account.last_error = ""
                if self._sticky_email == account.email:
                    self._sticky_email = None
                results.append({"email": email, "status": "success", "account_status": "disabled"})
                changed = True

        if changed:
            await self.save()
        return results

    async def enable_accounts(self, emails: list[str]) -> list[dict]:
        results: list[dict] = []
        changed = False

        async with self._lock:
            accounts_by_email = {account.email: account for account in self.accounts}
            for email in emails:
                account = accounts_by_email.get(email)
                if not account:
                    results.append({"email": email, "status": "skipped", "reason": "missing_account"})
                    continue
                if account.get_status_code() != "disabled":
                    results.append({"email": email, "status": "skipped", "reason": "already_enabled"})
                    continue

                account.valid = True
                account.activation_pending = False
                account.status_code = "valid"
                account.rate_limited_until = 0.0
                account.last_error = ""
                results.append({"email": email, "status": "success", "account_status": "valid"})
                changed = True

        if changed:
            await self.save()
        return results

    def _reclaim_stale_inflight(self, now: float) -> None:
        timeout = max(0.0, float(getattr(settings, "ACCOUNT_BUSY_TIMEOUT_SECONDS", 0) or 0))
        if timeout <= 0:
            return
        for acc in self.accounts:
            if acc.inflight <= 0 or acc.last_request_started <= 0:
                continue
            age = now - acc.last_request_started
            if age < timeout:
                continue
            log.warning(
                "[账号池] reclaim_stale_busy email=%s inflight=%s age=%.3fs timeout=%.3fs",
                acc.email,
                acc.inflight,
                age,
                timeout,
            )
            acc.inflight = 0
            acc.last_request_finished = now

    def _account_diagnostic(self, acc: Account, now: float, exclude: set | None = None) -> dict:
        min_interval = max(0, settings.ACCOUNT_MIN_INTERVAL_MS) / 1000.0
        next_available_at = max(acc.rate_limited_until, acc.last_request_started + min_interval)
        next_available_in = round(max(0.0, next_available_at - now), 3)
        is_excluded = bool(exclude and acc.email in exclude)
        is_rate_limited = acc.rate_limited_until > now
        capacity_available = acc.inflight < self.max_inflight

        if is_excluded:
            reason = "excluded"
        elif acc.get_status_code() == "disabled":
            reason = "disabled"
        elif acc.activation_pending:
            reason = "pending_activation"
        elif not acc.valid:
            reason = acc.status_code if acc.status_code and acc.status_code != "valid" else "invalid"
        elif is_rate_limited:
            reason = "rate_limited"
        elif not capacity_available:
            reason = "busy"
        elif next_available_at > now:
            reason = "cooldown"
        else:
            reason = "ready"

        return {
            "email": acc.email,
            "valid": acc.valid,
            "status_code": acc.get_status_code(),
            "status_text": acc.get_status_text(),
            "ready": reason == "ready",
            "selection_block_reason": reason,
            "inflight": acc.inflight,
            "max_inflight": self.max_inflight,
            "capacity_available": capacity_available,
            "is_rate_limited": is_rate_limited,
            "rate_limited_until": acc.rate_limited_until,
            "next_available_at": next_available_at,
            "next_available_in": next_available_in,
            "last_used": acc.last_used,
            "last_request_started": acc.last_request_started,
            "last_request_finished": acc.last_request_finished,
        }

    def account_diagnostics(self, exclude: set | None = None) -> list[dict]:
        now = time.time()
        return [self._account_diagnostic(acc, now, exclude) for acc in self.accounts]

    def _scheduler_snapshot(self, now: float, exclude: set | None = None) -> dict:
        diagnostics = [self._account_diagnostic(acc, now, exclude) for acc in self.accounts]
        blocked_reasons: dict[str, int] = {}
        for item in diagnostics:
            if item["ready"]:
                continue
            reason = item["selection_block_reason"]
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        ready_count = sum(1 for item in diagnostics if item["ready"])
        next_candidates = [item["next_available_at"] for item in diagnostics if item["valid"] and item["selection_block_reason"] != "excluded"]
        next_ready_at = min(next_candidates, default=0.0)
        return {
            "total": len(diagnostics),
            "ready": ready_count,
            "blocked": len(diagnostics) - ready_count,
            "blocked_reasons": blocked_reasons,
            "in_use": sum(item["inflight"] for item in diagnostics),
            "waiting": len(self._waiters),
            "max_inflight": self.max_inflight,
            "account_min_interval_ms": getattr(settings, "ACCOUNT_MIN_INTERVAL_MS", 0),
            "next_ready_at": next_ready_at,
            "next_ready_in": round(max(0.0, next_ready_at - now), 3) if next_ready_at else 0.0,
        }

    def _record_acquire_diagnostics(
        self,
        *,
        strategy: str,
        selected_email: str | None,
        diagnostics: list[dict],
        now: float,
        preferred_email: str | None = None,
        preferred_block_reason: str | None = None,
        exclude: set | None = None,
    ) -> None:
        ready_count = sum(1 for item in diagnostics if item["ready"])
        blocked_reasons: dict[str, int] = {}
        for item in diagnostics:
            if item["ready"]:
                continue
            reason = item["selection_block_reason"]
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        self.last_acquire_diagnostics = {
            "strategy": strategy,
            "selected_email": selected_email,
            "preferred_email": preferred_email,
            "preferred_block_reason": preferred_block_reason,
            "ready_count": ready_count,
            "blocked_count": len(diagnostics) - ready_count,
            "blocked_reasons": blocked_reasons,
            "snapshot": self._scheduler_snapshot(now, exclude),
        }

    async def acquire_preferred(self, preferred_email: str | None = None, exclude: set = None) -> Optional[Account]:
        if not preferred_email:
            return await self.acquire(exclude)
        preferred_block_reason = "missing"
        async with self._lock:
            now = time.time()
            self._reclaim_stale_inflight(now)
            diagnostics = [self._account_diagnostic(acc, now, exclude) for acc in self.accounts]
            preferred_diag = next((item for item in diagnostics if item["email"] == preferred_email), None)
            preferred = next((a for a in self.accounts if a.email == preferred_email), None)
            if preferred_diag:
                preferred_block_reason = preferred_diag["selection_block_reason"]
            if preferred and preferred_diag and preferred_diag["ready"]:
                preferred.inflight += 1
                preferred.last_used = now
                preferred.last_request_started = now + _jitter_seconds()
                self._sticky_email = preferred.email
                self._record_acquire_diagnostics(
                    strategy="preferred",
                    selected_email=preferred.email,
                    diagnostics=diagnostics,
                    now=now,
                    preferred_email=preferred_email,
                    exclude=exclude,
                )
                log.info("[账号池] acquire_selected strategy=preferred email=%s ready=%s", preferred.email, self.last_acquire_diagnostics["ready_count"])
                return preferred

        acc = await self.acquire(exclude)
        fallback_diag = dict(self.last_acquire_diagnostics)
        fallback_diag.update({
            "strategy": "fallback",
            "preferred_email": preferred_email,
            "preferred_block_reason": preferred_block_reason,
        })
        self.last_acquire_diagnostics = fallback_diag
        if acc:
            log.info("[账号池] acquire_selected strategy=fallback preferred=%s email=%s reason=%s", preferred_email, acc.email, preferred_block_reason)
        return acc

    async def acquire_wait_preferred(self, preferred_email: str | None = None, timeout: float = 60, exclude: set = None) -> Optional[Account]:
        deadline = time.time() + timeout
        while True:
            acc = await self.acquire_preferred(preferred_email, exclude)
            if acc:
                return acc
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            evt = asyncio.Event()
            self._waiters.append(evt)
            try:
                await asyncio.wait_for(evt.wait(), timeout=min(remaining, 0.5))
            except asyncio.TimeoutError:
                pass
            finally:
                if evt in self._waiters:
                    self._waiters.remove(evt)

    async def acquire(self, exclude: set = None) -> Optional[Account]:
        strategy = "least_loaded"
        async with self._lock:
            now = time.time()
            self._reclaim_stale_inflight(now)
            diagnostics = [self._account_diagnostic(acc, now, exclude) for acc in self.accounts]
            ready_emails = {item["email"] for item in diagnostics if item["ready"]}
            ready = [a for a in self.accounts if a.email in ready_emails]
            if not ready:
                self._record_acquire_diagnostics(
                    strategy=strategy,
                    selected_email=None,
                    diagnostics=diagnostics,
                    now=now,
                    exclude=exclude,
                )
                return None

            ready.sort(key=lambda a: (
                a.inflight,
                0 if str(getattr(a, "cookies", "") or "").strip() else 1,
                a.last_request_started or 0.0,
                a.last_used or 0.0,
                a.email or "",
            ))
            best = ready[0]
            best.inflight += 1
            best.last_used = now
            best.last_request_started = now + _jitter_seconds()
            self._sticky_email = best.email if len(ready) == 1 else None
            self._record_acquire_diagnostics(
                strategy=strategy,
                selected_email=best.email,
                diagnostics=diagnostics,
                now=now,
                exclude=exclude,
            )
            log.info("[账号池] acquire_selected strategy=%s email=%s ready=%s", strategy, best.email, self.last_acquire_diagnostics["ready_count"])
            return best

    async def acquire_wait(self, timeout: float = 60, exclude: set = None) -> Optional[Account]:
        deadline = time.time() + timeout
        while True:
            acc = await self.acquire(exclude)
            if acc:
                self.last_acquire_wait_diagnostics = {
                    "result": "selected",
                    "timeout": timeout,
                    "selected_email": acc.email,
                    "snapshot": self.last_acquire_diagnostics.get("snapshot", {}),
                }
                return acc

            async with self._lock:
                now = time.time()
                candidates = [
                    a for a in self.accounts
                    if a.valid and (not exclude or a.email not in exclude)
                ]
                snapshot = self._scheduler_snapshot(now, exclude)
                if not candidates:
                    self.last_acquire_wait_diagnostics = {
                        "result": "no_candidates",
                        "timeout": timeout,
                        "snapshot": snapshot,
                    }
                    log.warning("[账号池] acquire_wait_no_candidates snapshot=%s", snapshot)
                    return None
                next_ready_at = min((a.next_available_at() for a in candidates), default=now)

            remaining = deadline - time.time()
            if remaining <= 0:
                self.last_acquire_wait_diagnostics = {
                    "result": "timeout",
                    "timeout": timeout,
                    "snapshot": snapshot,
                }
                log.warning(
                    "[账号池] acquire_wait_timeout timeout=%s ready=%s blocked_reasons=%s waiting=%s",
                    timeout,
                    snapshot["ready"],
                    snapshot["blocked_reasons"],
                    snapshot["waiting"],
                )
                return None

            evt = asyncio.Event()
            self._waiters.append(evt)
            wait_timeout = min(remaining, max(0.05, next_ready_at - time.time() + 0.05))
            try:
                await asyncio.wait_for(evt.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                pass
            finally:
                if evt in self._waiters:
                    self._waiters.remove(evt)

    def release(self, acc: Account):
        acc.inflight = max(0, acc.inflight - 1)
        acc.last_request_finished = time.time()
        if self._waiters:
            evt = self._waiters.pop(0)
            evt.set()

    def mark_invalid(self, acc: Account, reason: str = "invalid", error_message: str = ""):
        acc.valid = False
        acc.status_code = reason or "invalid"
        acc.last_error = error_message or acc.last_error
        acc.consecutive_failures += 1
        if reason == "pending_activation":
            acc.activation_pending = True
        if self._sticky_email == acc.email:
            self._sticky_email = None
        log.warning(f"[账号] {acc.email} 已标记为不可用，状态={acc.status_code}")

    def mark_success(self, acc: Account):
        acc.consecutive_failures = 0
        acc.rate_limit_strikes = 0
        if acc.status_code == "rate_limited":
            acc.status_code = "valid"
        if not acc.activation_pending:
            acc.valid = True

    def mark_rate_limited(self, acc: Account, cooldown: int | None = None, error_message: str = ""):
        acc.rate_limit_strikes += 1
        base = cooldown if cooldown is not None else settings.RATE_LIMIT_BASE_COOLDOWN
        dynamic = min(settings.RATE_LIMIT_MAX_COOLDOWN, int(base * (2 ** max(0, acc.rate_limit_strikes - 1))))
        dynamic += int(_jitter_seconds())
        acc.rate_limited_until = time.time() + dynamic
        acc.status_code = "rate_limited"
        acc.last_error = error_message or acc.last_error
        if self._sticky_email == acc.email:
            self._sticky_email = None
        log.warning(f"[账号] {acc.email} 已限流冷却 {dynamic} 秒")

    def mark_file_upload_limited(self, acc: Account, retry_after_seconds: float, error_message: str = ""):
        retry_after = max(0.0, float(retry_after_seconds or 0))
        acc.file_upload_blocked_until = time.time() + retry_after
        acc.last_error = error_message or acc.last_error
        log.warning(f"[账号] {acc.email} 文件上传额度超限，{int(retry_after)} 秒内不再用于上传")

    def file_upload_blocked_emails(self) -> set:
        return {a.email for a in self.accounts if a.is_file_upload_blocked()}

    def status(self):
        available = [a for a in self.accounts if a.is_available()]
        rate_limited = [a for a in self.accounts if a.get_status_code() == "rate_limited"]
        disabled = [a for a in self.accounts if a.get_status_code() == "disabled"]
        invalid = [a for a in self.accounts if a.get_status_code() not in ("valid", "rate_limited", "disabled")]
        activation_pending = [a for a in self.accounts if a.get_status_code() == "pending_activation"]
        banned = [a for a in self.accounts if a.get_status_code() == "banned"]
        now = time.time()
        snapshot = self._scheduler_snapshot(now)
        return {
            "total": len(self.accounts),
            "valid": len(available),
            "ready": snapshot["ready"],
            "blocked": snapshot["blocked"],
            "blocked_reasons": snapshot["blocked_reasons"],
            "rate_limited": len(rate_limited),
            "disabled": len(disabled),
            "invalid": len(invalid),
            "activation_pending": len(activation_pending),
            "banned": len(banned),
            "in_use": snapshot["in_use"],
            "max_inflight": self.max_inflight,
            "waiting": len(self._waiters),
            "account_min_interval_ms": snapshot["account_min_interval_ms"],
            "next_ready_at": snapshot["next_ready_at"],
            "next_ready_in": snapshot["next_ready_in"],
            "last_acquire_diagnostics": self.last_acquire_diagnostics,
            "last_acquire_wait_diagnostics": self.last_acquire_wait_diagnostics,
        }
