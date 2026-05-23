from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Hashable


def _consume_future_exception(future: asyncio.Future) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except Exception:
        return


@dataclass(slots=True)
class SingleflightEntry:
    key: tuple[Hashable, ...]
    owner_id: str
    created_at: float
    future: asyncio.Future


class RequestSingleflight:
    def __init__(self, *, result_ttl_seconds: float = 60.0, now=time.monotonic):
        self.result_ttl_seconds = result_ttl_seconds
        self.now = now
        self._lock = asyncio.Lock()
        self._inflight: dict[tuple[Hashable, ...], SingleflightEntry] = {}
        self._completed: dict[tuple[Hashable, ...], tuple[float, Any]] = {}

    async def start_or_join(self, key: tuple[Hashable, ...], *, owner_id: str) -> tuple[SingleflightEntry | None, bool, Any | None]:
        async with self._lock:
            self._prune_completed()
            completed = self._completed.get(key)
            if completed is not None:
                return None, False, completed[1]
            existing = self._inflight.get(key)
            if existing is not None:
                return existing, False, None
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(_consume_future_exception)
            entry = SingleflightEntry(
                key=key,
                owner_id=owner_id,
                created_at=self.now(),
                future=future,
            )
            self._inflight[key] = entry
            return entry, True, None

    async def complete(self, key: tuple[Hashable, ...], result: Any) -> None:
        async with self._lock:
            entry = self._inflight.pop(key, None)
            self._completed[key] = (self.now(), result)
            self._prune_completed()
            if entry is not None and not entry.future.done():
                entry.future.set_result(result)

    async def fail(self, key: tuple[Hashable, ...], exc: BaseException) -> None:
        async with self._lock:
            entry = self._inflight.pop(key, None)
            if entry is not None and not entry.future.done():
                entry.future.set_exception(exc)

    async def forget(self, key: tuple[Hashable, ...]) -> None:
        async with self._lock:
            self._inflight.pop(key, None)
            self._completed.pop(key, None)

    def _prune_completed(self) -> None:
        cutoff = self.now() - self.result_ttl_seconds
        expired = [key for key, (created_at, _result) in self._completed.items() if created_at < cutoff]
        for key in expired:
            self._completed.pop(key, None)
