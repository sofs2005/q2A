"""文档 `parse`（`/api/v2/files/parse`）进程级熔断。

背景（2026-11 实测）：上游 `/api/v2/files/parse` 被阿里云 WAF 拦死，**换出口也
过不去**（出口 g/j/l 与直连全部返回 WAF 页面），而 `getstsToken` 走出口池是好的。
后果是「文档类附件必然失败」：每个请求都要先白等一次 10~20s 的
`getstsToken` + OSS PUT + `files/parse`，才降级成内联文本。带 tools / 大上下文的
请求每次都会生成 `qwen2api_context_history.txt`（网关自己造的历史上下文文件），
于是**每个普通请求**都在白付这笔时间。

本模块把「上游 parse 坏了」这件事记在进程内存里（不落盘、不跨进程）：

- `files/parse` 连续失败达 `DOCUMENT_PARSE_MAX_FAILURES`（默认 2）次，**或**直接
  撞 WAF 页面 → 打开熔断，冷却 `DOCUMENT_PARSE_COOLDOWN_SECONDS`（默认 600s）；
- 冷却期内调用方（`context_attachment_manager`）**直接跳过「上传 + parse」**，
  走内联文本降级 —— 省掉那次注定失败的 10~20s；
- 冷却到期后放行一次「探针」请求（half-open）：成功则清零关闭，失败立刻重新熔断。

熔断**只影响文档类**（需要 parse 才能被模型读到的附件）；图片 / 音频 / 视频
走 data URI 内联、纯文本类附件在网关本地解码注入，都不经过这里。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger("qwen2api.document_parse_breaker")

#: 连续失败多少次把 parse 投入冷却（默认与历史上传出口熔断一致：2 次）。
_DEFAULT_MAX_FAILURES = 2
#: 熔断冷却时长（秒）。WAF 是分钟级以上的状态，10 分钟足够避免「每个请求都去撞」。
_DEFAULT_COOLDOWN_SECONDS = 600.0


def _positive_int(value: Any, fallback: int) -> int:
    """把配置值收敛成正整数；非法 / 非正值退回默认。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _positive_float(value: Any, fallback: float) -> float:
    """把配置值收敛成正浮点；非法 / 非正值退回默认。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


class DocumentParseBreaker:
    """文档 parse 的进程级熔断器（closed / open / half-open 三态）。

    状态机：

    - **closed**：正常放行，累计失败次数；
    - **open**：`blocked()` 返回 True，调用方直接降级，不再发上游请求；
    - **half-open**：冷却到期后 `blocked()` 返回 False 且只放行一次探针；探针失败
      立刻重新 open（不等累计阈值），探针成功则回到 closed。

    线程安全：网关是单进程多协程（asyncio），但 `record_failure` 也可能被
    `asyncio.to_thread` 里的路径碰到，故用 `threading.Lock` 保护状态。
    """

    def __init__(
        self,
        *,
        max_failures: int = _DEFAULT_MAX_FAILURES,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock: Callable[[], float] = clock or time.monotonic
        self._lock = threading.Lock()
        self._max_failures = _positive_int(max_failures, _DEFAULT_MAX_FAILURES)
        self._cooldown_seconds = _positive_float(cooldown_seconds, _DEFAULT_COOLDOWN_SECONDS)
        self._failures = 0
        self._open_until: float | None = None
        self._half_open = False
        self._last_reason = ""

    # ---- 配置 ----

    def configure(self, *, max_failures: Any = None, cooldown_seconds: Any = None) -> None:
        """按运行期配置更新阈值。**不重置已有状态**（配置变化不该清掉熔断）。"""
        with self._lock:
            if max_failures is not None:
                self._max_failures = _positive_int(max_failures, self._max_failures)
            if cooldown_seconds is not None:
                self._cooldown_seconds = _positive_float(cooldown_seconds, self._cooldown_seconds)

    @property
    def max_failures(self) -> int:
        return self._max_failures

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    @property
    def last_reason(self) -> str:
        with self._lock:
            return self._last_reason

    @property
    def is_half_open(self) -> bool:
        with self._lock:
            return self._half_open

    def open_until(self) -> float | None:
        """熔断到期时刻（`clock()` 口径）；未熔断返回 None。"""
        with self._lock:
            return self._open_until

    def remaining_seconds(self) -> float:
        """剩余冷却秒数；未熔断返回 0。"""
        with self._lock:
            if self._open_until is None:
                return 0.0
            return max(0.0, self._open_until - self._clock())

    # ---- 状态查询 / 记账 ----

    def blocked(self) -> bool:
        """当前是否应跳过「上传 + parse」。

        Returns:
            True 表示熔断中（调用方直接走文本降级）；False 表示可以尝试。
            冷却到期时会就地转为 half-open 并返回 False —— 这次请求就是探针。
        """
        with self._lock:
            if self._open_until is None:
                return False
            now = self._clock()
            if now < self._open_until:
                return True
            # 冷却到期：放行一次探针。失败会在 record_failure() 里立刻重新熔断。
            self._open_until = None
            self._half_open = True
            self._failures = 0
            reason = self._last_reason or "unknown"
            log.info(
                "[DocParseBreaker] cooldown expired after reason=%s → probing next document parse",
                reason,
            )
            return False

    def record_failure(self, reason: str = "", *, immediate: bool = False) -> bool:
        """记一次 parse 失败。

        Args:
            reason: 失败原因短标签（`waf` / `parse_failed` / 异常类名），只用于日志。
            immediate: True 表示「不必等累计阈值，本次就熔断」（撞 WAF 用）。

        Returns:
            True 表示本次调用打开了熔断。
        """
        with self._lock:
            now = self._clock()
            self._last_reason = str(reason or "parse_failed")[:80]
            if self._half_open:
                # 探针失败：不等累计阈值，立刻重新熔断。
                self._half_open = False
                self._failures = 0
                self._open_until = now + self._cooldown_seconds
                opened = True
            else:
                self._failures += 1
                opened = bool(immediate) or self._failures >= self._max_failures
                if opened:
                    self._failures = 0
                    self._open_until = now + self._cooldown_seconds
            if not opened:
                return False
            cooldown = self._cooldown_seconds
            reason_text = self._last_reason
        log.warning(
            "[DocParseBreaker] document parse broken reason=%s → skip upload+parse for %.0fs",
            reason_text,
            cooldown,
        )
        return True

    def record_success(self) -> None:
        """记一次 parse 成功：清零并关闭熔断。"""
        with self._lock:
            was_active = self._open_until is not None or self._half_open or self._failures > 0
            previous_reason = self._last_reason
            self._failures = 0
            self._open_until = None
            self._half_open = False
            self._last_reason = ""
        if was_active:
            log.info("[DocParseBreaker] document parse ok (was reason=%s) → breaker closed", previous_reason or "n/a")

    def reset(self) -> None:
        """无条件清零（测试 / 运维手动恢复用）。"""
        with self._lock:
            self._failures = 0
            self._open_until = None
            self._half_open = False
            self._last_reason = ""


def _config_from_settings(settings_obj: Any) -> tuple[Any, Any]:
    """从配置对象读阈值；配置对象缺失该字段时返回 (None, None) 走默认值。"""
    source = settings_obj
    if source is None:
        try:
            from backend.core.config import settings as runtime_settings

            source = runtime_settings
        except Exception:  # pragma: no cover - 配置模块缺失时保持默认值
            return None, None
    return (
        getattr(source, "DOCUMENT_PARSE_MAX_FAILURES", None),
        getattr(source, "DOCUMENT_PARSE_COOLDOWN_SECONDS", None),
    )


#: 进程级单例：所有请求共享同一份熔断状态。
_BREAKER = DocumentParseBreaker()


def get_document_parse_breaker(settings_obj: Any = None) -> DocumentParseBreaker:
    """取进程级熔断器，并按当前配置同步阈值。"""
    max_failures, cooldown_seconds = _config_from_settings(settings_obj)
    if max_failures is not None or cooldown_seconds is not None:
        _BREAKER.configure(max_failures=max_failures, cooldown_seconds=cooldown_seconds)
    return _BREAKER


def document_parse_blocked(settings_obj: Any = None) -> bool:
    """便捷判定：当前是否应跳过文档的上传 + parse。"""
    return get_document_parse_breaker(settings_obj).blocked()


def reset_document_parse_breaker() -> None:
    """清零进程级熔断状态（单测 / 手动恢复用）。"""
    _BREAKER.reset()
