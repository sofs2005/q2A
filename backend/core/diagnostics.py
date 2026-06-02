from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, fields, replace
from typing import Any

log = logging.getLogger("qwen2api.diagnostics")


@dataclass(slots=True)
class ActiveRequestDiagnostic:
    req_id: str = "-"
    surface: str = "-"
    requested_model: str = "-"
    resolved_model: str = "-"
    chat_id: str = "-"
    stream_attempt: str = "-"
    upstream_attempt: str = "-"
    stream_stage: str = "-"
    method: str = "-"
    path: str = "-"
    client: str = "-"
    status: str = "-"
    updated_at: float = 0.0


class DiagnosticRegistry:
    def __init__(self) -> None:
        self._snapshot = ActiveRequestDiagnostic()
        self._field_names = {field.name for field in fields(ActiveRequestDiagnostic)}

    def snapshot(self) -> ActiveRequestDiagnostic:
        return replace(self._snapshot)

    def reset(self) -> ActiveRequestDiagnostic:
        self._snapshot = ActiveRequestDiagnostic()
        return self.snapshot()

    def set(self, snapshot: ActiveRequestDiagnostic | None) -> ActiveRequestDiagnostic:
        self._snapshot = replace(snapshot) if snapshot is not None else ActiveRequestDiagnostic()
        return self.snapshot()

    def update(self, **kwargs: Any) -> ActiveRequestDiagnostic:
        updated = self.snapshot()
        changed = False
        for key, value in kwargs.items():
            if key not in self._field_names or value is None:
                continue
            if key == "updated_at":
                setattr(updated, key, float(value))
            else:
                setattr(updated, key, str(value))
            changed = True
        if changed:
            if updated.updated_at <= 0:
                updated.updated_at = time.perf_counter()
            self._snapshot = updated
        return self.snapshot()


ACTIVE_REQUEST_DIAGNOSTICS = DiagnosticRegistry()


def reset_active_request_diagnostic() -> ActiveRequestDiagnostic:
    return ACTIVE_REQUEST_DIAGNOSTICS.reset()


def get_active_request_diagnostic() -> ActiveRequestDiagnostic:
    return ACTIVE_REQUEST_DIAGNOSTICS.snapshot()


def update_active_request_diagnostic(**kwargs: Any) -> ActiveRequestDiagnostic:
    return ACTIVE_REQUEST_DIAGNOSTICS.update(**kwargs)


def restore_active_request_diagnostic(snapshot: ActiveRequestDiagnostic | None) -> ActiveRequestDiagnostic:
    return ACTIVE_REQUEST_DIAGNOSTICS.set(snapshot)


def format_active_request_diagnostic(snapshot: ActiveRequestDiagnostic | None = None) -> str:
    current = snapshot or get_active_request_diagnostic()
    items = (
        ("req_id", current.req_id),
        ("surface", current.surface),
        ("requested_model", current.requested_model),
        ("resolved_model", current.resolved_model),
        ("chat_id", current.chat_id),
        ("stream_attempt", current.stream_attempt),
        ("upstream_attempt", current.upstream_attempt),
        ("stream_stage", current.stream_stage),
        ("method", current.method),
        ("path", current.path),
        ("client", current.client),
        ("status", current.status),
    )
    rendered = [f"{key}={value}" for key, value in items if value not in ("", "-", None)]
    return " ".join(rendered) if rendered else "-"


def format_event_loop_lag_warning(*, lag_seconds: float, poll_seconds: float, snapshot: ActiveRequestDiagnostic | None = None) -> str:
    return (
        f"[Diagnostics] event_loop_lag lag={lag_seconds:.3f}s poll={poll_seconds:.3f}s "
        f"active={format_active_request_diagnostic(snapshot)}"
    )


async def event_loop_lag_watchdog(
    *,
    interval_seconds: float,
    threshold_seconds: float,
    stop_event: asyncio.Event | None = None,
    logger: logging.Logger = log,
) -> None:
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval_seconds
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        await asyncio.sleep(interval_seconds)
        if stop_event is not None and stop_event.is_set():
            return
        now = loop.time()
        lag = now - expected
        if lag >= threshold_seconds:
            logger.warning(
                format_event_loop_lag_warning(
                    lag_seconds=lag,
                    poll_seconds=interval_seconds,
                    snapshot=get_active_request_diagnostic(),
                )
            )
        expected = now + interval_seconds


def install_stack_dump_handler(*, settings: Any, faulthandler_module: Any, signal_module: Any, stream: Any) -> bool:
    if not bool(getattr(settings, "DIAGNOSTIC_STACK_DUMP_ENABLED", False)):
        return False

    sigusr1 = getattr(signal_module, "SIGUSR1", None)
    if sigusr1 is None:
        log.warning("[Diagnostics] SIGUSR1 stack dump is unavailable on this platform")
        return False

    faulthandler_module.enable(file=stream, all_threads=True)
    faulthandler_module.register(sigusr1, file=stream, all_threads=True)
    log.info("[Diagnostics] SIGUSR1 stack dump handler enabled")
    return True
