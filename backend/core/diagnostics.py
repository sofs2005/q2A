from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("qwen2api.diagnostics")


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
