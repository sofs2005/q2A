"""上传出口池：把 `getstsToken` 挂到不同粘性出口 IP 上，绕开出口级 WAF 惩罚。

背景（2026-12 生产实测）
------------------------
用户的代理系统给的是「一个端口 + 不同后缀 = 不同粘性出口」：

    socks5h://Default.g@10.6.6.114:2260
    socks5h://Default.j@10.6.6.114:2260
    ...

实测 12 个后缀里 8 个可用：从这 8 个出口打 `/api/v2/files/getstsToken`
**直接返回 `file_id`，完全不触发阿里云 WAF 拼图验证码**。原因是阿里云的
人机校验惩罚是**出口级**的（按出口 IP 记分），换一个干净出口就绕过去了。

于是上传链路的正确姿势变成：

1. `getstsToken` 挂在出口池里的某个出口上（只有这一步需要代理）；
2. 拿到 STS 后 OSS PUT 直连（走代理反而更慢）；
3. `chats/new` **也**挂出口池（2026-12 追加，见下）。因为「上游会话与出口 IP
   绑定」，同一个 chat_id 后续的 `completions` 必须回到同一个出口 —— 该亲和性
   由 `backend.services.chat_exit_binding` 记录、`qwen_client` 查表复用。

出口本身不稳：有的 SOCKS5 connect 直接失败，有的 25s 超时，所以需要
「健康探测 + 连续失败冷却 + 轮询挑下一个」。本模块只负责**选出口 / 记账 /
探测**三件事，怎么发请求由调用方（`upstream_file_uploader`）决定。

安全约束
--------
代理 URL 里带凭据（`Default.g` 是用户名，也可能有密码），所以**日志里只打
出口后缀 id**（如 `Default.g`），任何情况下不把整条代理串写进日志。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import unquote, urlparse

from backend.core.http_timeout import connect_read_timeout

log = logging.getLogger("qwen2api.exit_pool")

#: 连续失败多少次进入冷却（可被 `UPLOAD_EXIT_MAX_FAILURES` 覆盖）
DEFAULT_MAX_FAILURES = 2
#: 冷却时长（秒，可被 `UPLOAD_EXIT_COOLDOWN_SECONDS` 覆盖）
DEFAULT_COOLDOWN_SECONDS = 600.0
#: 单次健康探测超时（秒，可被 `UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS` 覆盖）。
#: **只用于 `/admin/upload/probe` 的手工单次探测**；启动探测另有
#: `DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS`（两者语义不同，别混用）。
DEFAULT_PROBE_TIMEOUT_SECONDS = 10.0
#: 出口请求的**建连阶段**超时（秒，可被 `EXIT_CONNECT_TIMEOUT_SECONDS` 覆盖）。
#: 只压建连、不压读取：大模型生成可能很慢，读超时压到 5s 会误杀正常请求。
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
#: **启动探测**的单出口超时（秒，可被 `EXIT_PROBE_TIMEOUT_SECONDS` 覆盖）。
#: 2026-12 事故：socks5h 冷启动握手 + TLS 在生产上要 3~4s，6 个出口挤在 5s 总预算
#: 里全部超时 → 全判死 → 冷却 600s → 出口池整体不可用 → `chats/new` 回落直连被
#: WAF 打死。这里放宽到 20s，并且与 `UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS` 分开配置。
DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS = 20.0
#: 启动探测的**总**耗时上限（秒，可被 `EXIT_PROBE_TOTAL_TIMEOUT_SECONDS` 覆盖）。
#: 6 个出口 × 20s 单超时 ÷ 并发 2 ≈ 60s 上界，45s 覆盖绝大多数正常批次。
DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS = 45.0
#: 启动探测的并发度（可被 `EXIT_PROBE_CONCURRENCY` 覆盖）。
#: socks5 握手排队做：6 条隧道同时握手会互相拖慢，串成 2 条反而整体更快更稳。
DEFAULT_PROBE_CONCURRENCY = 2
#: **探测**判死后的冷却时长（秒，可被 `EXIT_PROBE_COOLDOWN_SECONDS` 覆盖）。
#: 探测只是「猜测」，不套用真实请求的 600s 重罚：猜错了 120s 就能自己恢复。
DEFAULT_PROBE_COOLDOWN_SECONDS = 120.0

#: 出口状态
STATE_READY = "ready"
STATE_COOLING = "cooldown"

#: `UPLOAD_VIA_EXIT` 的合法取值；非法值一律按 `auto` 处理（不要因为写错一个
#: env 就把上传链路打死）。
_VALID_MODES = ("auto", "always", "never")

#: WAF 验证页特征（与 upstream_file_uploader 保持同一语义，这里不复用是为了
#: 避免 exit_pool ←→ upstream_file_uploader 的循环 import）。
_WAF_MARKERS = (
    "aliyun_waf",
    "aliyuncaptcha",
    "<!doctype html",
    "sceneid=",
)

#: 探测出口用的上游轻量接口：**不需要鉴权**，实测 200 / 4902B，不触发 WAF。
#: 绝不用 `chat/completions`（烧额度 + 招风控），也不再用 `/api/v2/users/status`
#: （需要 token，且 401 会被旧逻辑误判成「出口死」）。
_PROBE_PATH = "/api/v2/models/"

#: 探测判死的原因集合（**只有连不上才算死**）：DNS / 代理握手 / 超时。
CONNECT_FAILURE_REASONS = frozenset(
    {
        "connect_failed",
        "connect_timeout",
        "proxy_error",
        "dns_failed",
        "probe_timeout",
    }
)


@dataclass(frozen=True)
class ProbeOutcome:
    """一次出口探测的结论。

    判定口径（2026-12 事故后收紧）：**只要拿到任何 HTTP 响应就算活**。
    上游的 401 / 403 / 429 / 5xx 是「鉴权 / 限流 / 上游」的事，与出口 IP 能否
    连通无关；只有 `ProxyError` / 握手失败 / DNS 失败 / 超时才说明出口真的死了。

    Attributes:
        ok: 出口是否可用。
        reason: 观测用的原因串（`http_200` / `http_403` / `connect_failed` ...）。
        responded: 是否拿到了 HTTP 响应（`ok` 的主要判据）。
        status: HTTP 状态码；没拿到响应时为 0。
    """

    ok: bool
    reason: str = ""
    responded: bool = False
    status: int = 0


def _classify_connect_error(exc: BaseException) -> str:
    """把「连不上」的异常归类成观测用的原因串（永远属于 `CONNECT_FAILURE_REASONS`）。"""
    name = type(exc).__name__.lower()
    text = f"{name} {str(exc)}".lower()
    if isinstance(exc, asyncio.TimeoutError) or "timeout" in text or "timed out" in text:
        reason = "connect_timeout"
    elif "proxy" in text:
        reason = "proxy_error"
    elif "dns" in text or "resolve" in text or "name or service" in text or "getaddrinfo" in text:
        reason = "dns_failed"
    else:
        reason = "connect_failed"
    # 兜底：返回值必须落在「连不上」这个集合里，否则调用方会把上游错误误判成出口死。
    return reason if reason in CONNECT_FAILURE_REASONS else "connect_failed"


def _coerce_probe_outcome(raw: Any) -> ProbeOutcome:
    """把注入的假探测结果（bool / `(ok, reason)` / `ProbeOutcome`）归一成结论。

    单测与外部调用方历史上只返回 `bool`，这里保持向后兼容：`True` 视为「有响应」，
    `False` 视为「连不上」（`connect_failed`）。

    Args:
        raw: `probe_fetch` 的返回值。

    Returns:
        归一后的 :class:`ProbeOutcome`。
    """
    if isinstance(raw, ProbeOutcome):
        return raw
    if isinstance(raw, tuple):
        items = list(raw)
        ok = bool(items[0]) if items else False
        reason = str(items[1] or "") if len(items) > 1 else ""
        responded = bool(items[2]) if len(items) > 2 else ok
        if not reason:
            reason = "" if ok else "connect_failed"
        return ProbeOutcome(ok=ok, reason=reason, responded=responded)
    ok = bool(raw)
    return ProbeOutcome(ok=ok, reason="" if ok else "connect_failed", responded=ok)



def looks_like_waf(body: str) -> bool:
    """判断响应体是否阿里云 WAF 验证页面（本地副本，避免循环 import）。"""
    lowered = str(body or "").lower()
    return any(marker in lowered for marker in _WAF_MARKERS)


def exit_id_from_url(proxy_url: str) -> str:
    """从代理 URL 里取出口标识：`socks5h://Default.g@host:port` → `Default.g`。

    只取 userinfo（`Default.g`），**不含 host / port / 密码**，因此可以安全地
    写进日志。userinfo 为空时退回 `host:port`（无凭据的普通代理）。

    Args:
        proxy_url: 完整代理串，如 `socks5h://Default.g@10.6.6.114:2260`。

    Returns:
        出口 id；无法解析时返回空串。
    """
    text = str(proxy_url or "").strip()
    if not text:
        return ""
    candidate = text if "://" in text else f"socks5h://{text}"
    try:
        parsed = urlparse(candidate)
        username = unquote(parsed.username or "").strip()
        if username:
            return username
        host = str(parsed.hostname or "").strip()
        port = parsed.port
    except Exception:
        return ""
    if host and port:
        return f"{host}:{port}"
    return host


def parse_exit_list(raw: Any) -> list[str]:
    """把 `UPLOAD_EXIT_POOL` 的配置解析成代理 URL 列表。

    Args:
        raw: 逗号分隔字符串（也接受已解析的 list / tuple）。

    Returns:
        去重后的代理 URL 列表（保持配置顺序），空项自动丢弃。
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items: Iterable[Any] = raw
    else:
        items = str(raw).replace("\n", ",").split(",")
    exits: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = str(item or "").strip()
        if not url:
            continue
        exit_id = exit_id_from_url(url)
        if not exit_id or exit_id in seen:
            continue
        seen.add(exit_id)
        exits.append(url)
    return exits


def resolve_exit_mode(settings_obj: Any) -> str:
    """解析 `UPLOAD_VIA_EXIT`：auto / always / never，非法值按 auto。"""
    raw = str(getattr(settings_obj, "UPLOAD_VIA_EXIT", "auto") or "auto").strip().lower()
    return raw if raw in _VALID_MODES else "auto"


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


#: `avg_connect_ms` 滑动平均的样本上限：超过后按指数衰减，让近期样本占主导
_CONNECT_SAMPLE_CAP = 8


def _accumulate_connect_ms(state: "_ExitState", connect_ms: float | None) -> None:
    """把一次耗时样本并进 `avg_connect_ms`（简单滑动平均，样本上限后转指数衰减）。

    Args:
        state: 出口状态。
        connect_ms: 本次耗时（毫秒）；None / 负数 / 非法值直接忽略。
    """
    if connect_ms is None:
        return
    try:
        sample = float(connect_ms)
    except (TypeError, ValueError):
        return
    if sample < 0:
        return
    state.connect_samples += 1
    if state.connect_samples == 1:
        # 首个样本直接作为初值，避免从 0 慢慢爬上来。
        state.avg_connect_ms = sample
        return
    if state.connect_samples <= _CONNECT_SAMPLE_CAP:
        # 样本未满：标准增量平均。
        weight = 1.0 / float(state.connect_samples)
    else:
        # 样本已满：固定权重 1/(cap+1)，近 8 次占主导，老样本自然淡出。
        weight = 1.0 / float(_CONNECT_SAMPLE_CAP + 1)
    state.avg_connect_ms = state.avg_connect_ms + (sample - state.avg_connect_ms) * weight


@dataclass
class _ExitState:
    """单个出口的运行时状态。"""

    exit_id: str
    url: str
    failures: int = 0
    successes: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""
    last_used_at: float = 0.0
    #: 最近一次成功的时间戳（`wall_clock`，默认 `time.time()` 的 epoch 秒）。
    #: `pick()` 用它把「已证明可用」的出口排到最前面 —— 生产上出口一旦通了就
    #: 该直接命中它，不要再从零试错。
    last_success_at: float = 0.0
    #: 最近一次失败的时间戳（epoch 秒，只用于观测）。
    last_failure_at: float = 0.0
    #: 建连耗时滑动平均（毫秒）；没测到过就是 0。
    avg_connect_ms: float = 0.0
    #: 已累计的建连耗时样本数（用于把滑动平均从「首个样本」平滑起算）。
    connect_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ExitPool:
    """轮询式出口池：挑出口 / 记成功失败 / 冷却剔除 / 健康探测。

    并发安全：`pick` / `mark_*` / `snapshot` 都是同步方法，只做 dict 上的整数与
    浮点读写，在同一个事件循环里天然原子；`probe` 是唯一 await 的入口，用
    `asyncio.Lock` 串行化「探测 → 记账」，避免并发探测把失败计数翻倍。
    """

    def __init__(
        self,
        exits: list[str],
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_failures: int = DEFAULT_MAX_FAILURES,
        probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        startup_probe_timeout_seconds: float = DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS,
        probe_cooldown_seconds: float = DEFAULT_PROBE_COOLDOWN_SECONDS,
        probe_concurrency: int = DEFAULT_PROBE_CONCURRENCY,
        prefer_last_success: bool = True,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        probe_fetch: Callable[[str, float], Awaitable[bool]] | None = None,
    ) -> None:
        """初始化出口池。

        Args:
            exits: 代理 URL 列表（`socks5h://Default.g@host:port`）。
            cooldown_seconds: 冷却时长；连续失败达阈值后该出口暂停这么久。
            max_failures: 连续失败多少次进入冷却。
            probe_timeout_seconds: 单次探测的整体超时（读阶段，`/admin/upload/probe`）。
            connect_timeout_seconds: 出口请求的**建连阶段**超时（只压 connect）。
            startup_probe_timeout_seconds: **启动探测**的单出口超时
                （`EXIT_PROBE_TIMEOUT_SECONDS`，默认 20s；与上面那个分开配置）。
            probe_cooldown_seconds: 探测判死后的**短冷却**
                （`EXIT_PROBE_COOLDOWN_SECONDS`，默认 120s）。
            probe_concurrency: 启动探测的并发度（`EXIT_PROBE_CONCURRENCY`，默认 2）。
            prefer_last_success: `pick()` 是否优先返回最近成功的出口
                （`EXIT_PICK_PREFER_LAST_SUCCESS`，默认开）。
            clock: 单调时钟函数（冷却计算用），默认 `time.monotonic`。
            wall_clock: 墙钟函数（`last_success_at` / `last_failure_at` 用，
                给运维看的是 epoch 秒），默认 `time.time`。
            probe_fetch: 探测实现（单测注入假网络用）；签名
                `async (proxy_url, timeout) -> bool`，为空时用真实 HTTP 探测。
        """
        self._states: dict[str, _ExitState] = {}
        self._order: list[str] = []
        for url in parse_exit_list(exits):
            exit_id = exit_id_from_url(url)
            if not exit_id or exit_id in self._states:
                continue
            self._states[exit_id] = _ExitState(exit_id=exit_id, url=url)
            self._order.append(exit_id)
        self._cursor = 0
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.max_failures = max(1, int(max_failures))
        self.probe_timeout_seconds = max(0.1, float(probe_timeout_seconds))
        self.connect_timeout_seconds = max(0.0, float(connect_timeout_seconds))
        # 启动探测与手工探测的参数完全分开：socks5h 冷启动握手 + TLS 要 3~4s，
        # 用 10s 那个值去并发 6 个出口必然全超时（2026-12 事故）。
        self.startup_probe_timeout_seconds = max(0.1, float(startup_probe_timeout_seconds))
        self.probe_cooldown_seconds = max(0.0, float(probe_cooldown_seconds))
        self.probe_concurrency = max(1, int(probe_concurrency))
        self.prefer_last_success = bool(prefer_last_success)
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._probe_fetch = probe_fetch
        self._lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings_obj: Any) -> "ExitPool | None":
        """按配置构造出口池；关闭或池为空时返回 None（调用方据此走原路径）。

        `UPLOAD_VIA_EXIT=auto`（默认）与 `always` 都是「池非空即启用」；
        `never` 明确关闭——上传链路一行代理都不会碰。
        """
        if settings_obj is None:
            return None
        if resolve_exit_mode(settings_obj) == "never":
            return None
        return cls._from_exit_config(settings_obj)

    @classmethod
    def from_chat_settings(cls, settings_obj: Any) -> "ExitPool | None":
        """按 `CHAT_VIA_EXIT_POOL` 构造「建会话用」出口池；关闭或池为空时返回 None。

        与 :meth:`from_settings` 复用同一份 `UPLOAD_EXIT_POOL` 出口列表和同一套
        冷却 / 失败阈值配置，但开关独立：运营可以用 `UPLOAD_VIA_EXIT=never` 关掉
        上传代理，同时用 `CHAT_VIA_EXIT_POOL=always` 保住建会话链路（生产故障里
        正是 `chats/new` 被出口级 WAF 打死，必须先救它）。

        Args:
            settings_obj: 配置对象。

        Returns:
            出口池；未配置出口列表或 `CHAT_VIA_EXIT_POOL=never` 时为 None。
        """
        # 延迟 import：chat_exit_binding 不依赖 exit_pool，这里反向引用只为避免
        # 两个模块在 import 期互相拉扯（exit_pool 是底层模块，尽量少依赖）。
        from backend.services.chat_exit_binding import resolve_chat_exit_mode

        if settings_obj is None:
            return None
        if resolve_chat_exit_mode(settings_obj) == "never":
            return None
        return cls._from_exit_config(settings_obj)

    @classmethod
    def _from_exit_config(cls, settings_obj: Any) -> "ExitPool | None":
        """用 `UPLOAD_EXIT_POOL` + 冷却/阈值配置构造池；出口列表为空时返回 None。"""
        exits = parse_exit_list(getattr(settings_obj, "UPLOAD_EXIT_POOL", ""))
        if not exits:
            return None
        return cls(
            exits,
            cooldown_seconds=_as_float(
                getattr(settings_obj, "UPLOAD_EXIT_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS),
                DEFAULT_COOLDOWN_SECONDS,
            ),
            max_failures=_as_int(
                getattr(settings_obj, "UPLOAD_EXIT_MAX_FAILURES", DEFAULT_MAX_FAILURES),
                DEFAULT_MAX_FAILURES,
            ),
            probe_timeout_seconds=_as_float(
                getattr(settings_obj, "UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS", DEFAULT_PROBE_TIMEOUT_SECONDS),
                DEFAULT_PROBE_TIMEOUT_SECONDS,
            ),
            connect_timeout_seconds=_as_float(
                getattr(settings_obj, "EXIT_CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT_SECONDS),
                DEFAULT_CONNECT_TIMEOUT_SECONDS,
            ),
            # 启动探测：单出口 20s（与 UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS 分开）、
            # 判死短冷却 120s（探测是猜测，不该套用真实请求的 600s 重罚）、
            # 并发 2（socks5 握手排队做，避免互相拖慢）。
            startup_probe_timeout_seconds=_as_float(
                getattr(
                    settings_obj,
                    "EXIT_PROBE_TIMEOUT_SECONDS",
                    DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS,
                ),
                DEFAULT_STARTUP_PROBE_TIMEOUT_SECONDS,
            ),
            probe_cooldown_seconds=_as_float(
                getattr(settings_obj, "EXIT_PROBE_COOLDOWN_SECONDS", DEFAULT_PROBE_COOLDOWN_SECONDS),
                DEFAULT_PROBE_COOLDOWN_SECONDS,
            ),
            probe_concurrency=_as_int(
                getattr(settings_obj, "EXIT_PROBE_CONCURRENCY", DEFAULT_PROBE_CONCURRENCY),
                DEFAULT_PROBE_CONCURRENCY,
            ),
            # 未显式配置时默认开启「优先最近成功的出口」。
            prefer_last_success=bool(
                getattr(settings_obj, "EXIT_PICK_PREFER_LAST_SUCCESS", True)
            ),
        )

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """池里至少有一个出口时才启用。"""
        return bool(self._order)

    @property
    def exit_ids(self) -> list[str]:
        """全部出口 id（配置顺序）。"""
        return list(self._order)

    def proxy_url(self, exit_id: str) -> str:
        """出口 id → 完整代理 URL；不存在时返回空串。"""
        state = self._states.get(str(exit_id or ""))
        return state.url if state is not None else ""

    # ------------------------------------------------------------------
    # 选出口
    # ------------------------------------------------------------------
    def pick(self, exclude: set[str] | None = None) -> str | None:
        """挑一个可用出口，返回**代理 URL**；没有可用出口时返回 None。

        排序（`prefer_last_success=True`，默认）：

        1. 先排除 `exclude`（本轮已试过的）与冷却中的出口（冷却到期的顺手复位）；
        2. 在剩下的出口里，**优先返回 `last_success_at` 最新的那个** —— 出口一旦
           证明可用就直接命中它，不再从零试错（生产上 4 个死出口各等 10s 的浪费
           就出在这里）。同样「最近成功」时退化为轮询，保证负载不会死盯一个出口；
        3. 从来没过成功的出口（`last_success_at=0`）排在已证明可用的之后，按轮询
           顺序取第一个 —— 探测还没跑完 / 全部不可用时行为与旧版一致。

        Args:
            exclude: 本轮要跳过的出口 id 集合。

        Returns:
            代理 URL（可直接塞进 curl_cffi 的 `proxies`），或 None。
        """
        if not self._order:
            return None
        excluded = {str(item or "") for item in (exclude or set())}
        now = self._clock()
        total = len(self._order)
        candidates: list[tuple[int, _ExitState]] = []
        for offset in range(total):
            index = (self._cursor + offset) % total
            exit_id = self._order[index]
            if exit_id in excluded:
                continue
            state = self._states[exit_id]
            if state.cooldown_until > now:
                continue
            if state.cooldown_until:
                # 冷却到期：清零计数，给这个出口一轮全新的机会。
                state.cooldown_until = 0.0
                state.failures = 0
            candidates.append((index, state))
        if not candidates:
            return None

        chosen_index, chosen_state = candidates[0]
        if self.prefer_last_success:
            proven = [item for item in candidates if item[1].last_success_at > 0.0]
            if proven:
                # key：最近成功时间最大 → 优先；同时间再按「连续失败更少」→ 轮询。
                chosen_index, chosen_state = max(
                    proven,
                    key=lambda item: (item[1].last_success_at, -item[1].failures),
                )
        self._cursor = (chosen_index + 1) % total
        chosen_state.last_used_at = now
        return chosen_state.url

    # ------------------------------------------------------------------
    # 记账
    # ------------------------------------------------------------------
    def mark_success(self, exit_id: str, *, connect_ms: float | None = None) -> None:
        """记一次成功：失败计数清零、解除冷却、刷新 `last_success_at`。

        Args:
            exit_id: 出口 id（`Default.g` 这种后缀）。
            connect_ms: 本次经该出口的耗时（毫秒，可选）。给了就并进
                `avg_connect_ms` —— 只在轻量请求（探测 / getstsToken / 建会话）上
                采样，不拿大模型生成的长耗时污染这个指标。
        """
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return
        state.failures = 0
        state.cooldown_until = 0.0
        state.successes += 1
        state.last_error = ""
        state.last_success_at = float(self._wall_clock())
        _accumulate_connect_ms(state, connect_ms)

    def mark_failure(
        self,
        exit_id: str,
        reason: str,
        *,
        connect_ms: float | None = None,
    ) -> None:
        """记一次失败；连续失败达 `max_failures` 时把该出口投入冷却。

        Args:
            exit_id: 出口 id（`Default.g` 这种后缀，不含凭据）。
            reason: 失败原因（`waf` / `http_403` / `connect_failed` ...），只用于
                观测，不会被当作判断依据。
            connect_ms: 失败前已经等掉的耗时（毫秒，可选），并进 `avg_connect_ms`。
        """
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return
        state.failures += 1
        state.last_error = str(reason or "")[:200]
        state.last_failure_at = float(self._wall_clock())
        _accumulate_connect_ms(state, connect_ms)
        if state.failures >= self.max_failures:
            state.cooldown_until = self._clock() + self.cooldown_seconds
            log.warning(
                "[UploadExit] exit=%s cooling failures=%s cooldown=%.0fs last_error=%s",
                state.exit_id,
                state.failures,
                self.cooldown_seconds,
                state.last_error,
            )

    def mark_probe_failure(
        self,
        exit_id: str,
        reason: str,
        *,
        connect_ms: float | None = None,
    ) -> None:
        """记一次**探测**失败：短冷却（`EXIT_PROBE_COOLDOWN_SECONDS`），不动失败计数。

        探测只是「猜」出口死没死，猜错了要能很快自我纠正，所以：

        1. **不增加** `failures`（主失败计数只由真实请求累加，避免探测与真实失败
           叠加后触发 600s 重罚）；
        2. 冷却时长用 `probe_cooldown_seconds`（默认 120s），而不是
           `cooldown_seconds`（默认 600s）。

        Args:
            exit_id: 出口 id（`Default.g` 这种后缀，不含凭据）。
            reason: 失败原因（`connect_failed` / `proxy_error` / `connect_timeout` ...）。
            connect_ms: 失败前已经等掉的耗时（毫秒，可选），并进 `avg_connect_ms`。
        """
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return
        state.last_error = str(reason or "")[:200]
        state.last_failure_at = float(self._wall_clock())
        _accumulate_connect_ms(state, connect_ms)
        # 注意：这里刻意 **不** 执行 `state.failures += 1`。
        if self.probe_cooldown_seconds > 0:
            state.cooldown_until = max(
                state.cooldown_until,
                self._clock() + self.probe_cooldown_seconds,
            )

    def mark_probe_alive(self, exit_id: str) -> None:
        """把探测判死的出口「翻回可用」（全灭兜底用）：清冷却，不动失败计数。"""
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return
        state.cooldown_until = 0.0
        state.last_error = ""

    def snapshot(self) -> list[dict[str, Any]]:
        """给 `/admin/upload/probe` 用的只读快照（不含代理凭据）。"""
        now = self._clock()
        rows: list[dict[str, Any]] = []
        for exit_id in self._order:
            state = self._states[exit_id]
            remaining = max(0.0, state.cooldown_until - now) if state.cooldown_until else 0.0
            rows.append(
                {
                    "id": state.exit_id,
                    "state": STATE_COOLING if remaining > 0 else STATE_READY,
                    "failures": state.failures,
                    "successes": state.successes,
                    "cooldown_remaining": round(remaining, 1),
                    "last_error": state.last_error,
                    # 出口健康画像（epoch 秒 / 毫秒），运维据此判断该不该换出口。
                    "last_success_at": round(state.last_success_at, 3),
                    "last_failure_at": round(state.last_failure_at, 3),
                    "avg_connect_ms": round(state.avg_connect_ms, 1),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # 健康探测
    # ------------------------------------------------------------------
    async def probe(self, exit_id: str, *, token: str = "", account: Any = None) -> bool:
        """经指定出口 GET 一次上游轻量接口，判断出口是否可用。

        判定口径：**拿到任何 HTTP 响应就算活**（200/401/403/429 都算），只有连不上
        （ProxyError / 握手失败 / DNS 失败 / 超时）才算死。判死走
        :meth:`mark_probe_failure`（短冷却、不动失败计数）。**永不抛异常**。

        Args:
            exit_id: 出口 id。
            token: 可选上游 token（探测端点不需要，但带上更接近真实链路）。
            account: 可选账号对象，用它的浏览器指纹发请求（与真实链路一致）。

        Returns:
            True 表示该出口当前可用。
        """
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return False
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                outcome = await self._probe_outcome(
                    state.url,
                    token=token,
                    account=account,
                    timeout=self.probe_timeout_seconds,
                )
            except Exception as exc:  # 探测不允许把异常抛给 /admin
                log.warning(
                    "[UploadExit] probe error exit=%s error=%s",
                    state.exit_id,
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                )
                outcome = ProbeOutcome(ok=False, reason="connect_failed")
            if outcome.ok:
                self.mark_success(state.exit_id)
                log.info("[UploadExit] probe ok exit=%s reason=%s", state.exit_id, outcome.reason)
            else:
                self.mark_probe_failure(state.exit_id, outcome.reason)
                log.warning(
                    "[UploadExit] probe failed exit=%s reason=%s cooldown=%.0fs",
                    state.exit_id,
                    outcome.reason,
                    self.probe_cooldown_seconds,
                )
            return bool(outcome.ok)

    async def _do_probe(
        self,
        proxy_url: str,
        *,
        token: str,
        account: Any,
        timeout: float | None = None,
    ) -> ProbeOutcome:
        """执行一次探测：优先用注入的假实现（单测），否则发真实请求。"""
        probe_timeout = (
            float(timeout) if timeout is not None and timeout > 0 else self.probe_timeout_seconds
        )
        if self._probe_fetch is not None:
            # 单测的假探测层历史上只返回 bool，这里归一成结论（True → 有响应）。
            return _coerce_probe_outcome(await self._probe_fetch(proxy_url, probe_timeout))
        return await self._http_probe(proxy_url, token=token, account=account, timeout=probe_timeout)

    async def _http_probe(
        self,
        proxy_url: str,
        *,
        token: str,
        account: Any,
        timeout: float | None = None,
    ) -> ProbeOutcome:
        """真实探测：curl_cffi + 该出口代理 GET `/api/v2/models/`。

        判定：**有任何 HTTP 响应就算出口活着**。`401/403/429/5xx` 是上游的事
        （鉴权 / 限流 / 上游故障），与「出口 IP 能不能连通」无关；旧实现用
        `status == 200` 判定，一次 401 就把健康出口打进 600s 冷却（2026-12 事故）。

        Args:
            proxy_url: 该出口的代理 URL。
            token: 可选上游 token（探测端点不需要鉴权，带上更贴近真实链路）。
            account: 可选账号（决定浏览器指纹）。
            timeout: 单次探测超时（秒）。

        Returns:
            :class:`ProbeOutcome`；只有连不上时 `ok=False`。
        """
        from curl_cffi.requests import AsyncSession

        from backend.core.browser_fingerprint import fingerprint_for_account
        from backend.services.auth_resolver import BASE_URL

        read_timeout = float(timeout) if timeout is not None and timeout > 0 else self.probe_timeout_seconds
        timeouts = connect_read_timeout(read_timeout, self.connect_timeout_seconds)
        fingerprint = fingerprint_for_account(account)
        session = AsyncSession(
            impersonate=fingerprint.impersonate,
            timeout=timeouts,
            allow_redirects=True,
            proxies={"http": proxy_url, "https": proxy_url},
        )
        try:
            headers = {
                "Accept": "application/json, text/plain, */*",
                "User-Agent": fingerprint.user_agent,
                "Referer": f"{BASE_URL}/",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = await session.get(
                f"{BASE_URL}{_PROBE_PATH}",
                headers=headers,
                timeout=timeouts,
            )
            status = int(getattr(response, "status_code", 0) or 0)
            body = str(getattr(response, "text", "") or "")
            if status <= 0:
                # 没有状态码 = 压根没拿到 HTTP 响应。
                return ProbeOutcome(ok=False, reason="connect_failed", responded=False, status=0)
            if looks_like_waf(body):
                # WAF 页也是「网络通」的证据（出口能到上游），只是被记分了；
                # 记进 last_error 供运维观测，但**不判死**。
                return ProbeOutcome(
                    ok=True,
                    reason=f"waf_page_{status}",
                    responded=True,
                    status=status,
                )
            return ProbeOutcome(ok=True, reason=f"http_{status}", responded=True, status=status)
        except Exception as exc:
            # ProxyError / 握手失败 / DNS 失败 / 读超时 —— 只有这些算出口死。
            return ProbeOutcome(ok=False, reason=_classify_connect_error(exc), responded=False)
        finally:
            try:
                await session.close()
            except Exception:  # pragma: no cover - 关闭失败不影响探测结论
                log.debug("[UploadExit] probe session close failed", exc_info=True)

    async def _probe_outcome(
        self,
        proxy_url: str,
        *,
        token: str,
        account: Any,
        timeout: float | None = None,
    ) -> ProbeOutcome:
        """探测一次并给出结论；**永不抛异常**。

        Args:
            proxy_url: 该出口的代理 URL。
            token: 可选上游 token。
            account: 可选账号（决定浏览器指纹）。
            timeout: 单次探测超时（秒）。

        Returns:
            :class:`ProbeOutcome`；连不上时 `ok=False`，reason ∈
            `CONNECT_FAILURE_REASONS`。
        """
        probe_timeout = (
            float(timeout) if timeout is not None and timeout > 0 else self.probe_timeout_seconds
        )
        try:
            raw = await asyncio.wait_for(
                self._do_probe(proxy_url, token=token, account=account, timeout=probe_timeout),
                timeout=probe_timeout,
            )
        except asyncio.TimeoutError:
            # 整体超时 = 拿不到响应 = 出口连不上（唯一该判死的情形之一）。
            return ProbeOutcome(ok=False, reason="probe_timeout")
        except Exception as exc:
            log.debug(
                "[UploadExit] probe error error=%s",
                f"{type(exc).__name__}: {str(exc)[:120]}",
            )
            return ProbeOutcome(ok=False, reason=_classify_connect_error(exc))
        return _coerce_probe_outcome(raw)

    async def probe_all(
        self,
        *,
        token: str = "",
        account: Any = None,
        limit_seconds: float = DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS,
        only: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """**限并发**探测所有出口，只把「连不上」的踢进**短**冷却。

        启动时调一次：把「已经连不上」的出口提前踢出轮转，后续请求就不会再挨个
        试死出口（生产上 4 个死出口 × 10s ≈ 21s 的白等由此消失）。

        - 并发：`asyncio.Semaphore(probe_concurrency)`（默认 2）—— socks5h 冷启动
          握手排队做，6 条隧道同时握手只会互相拖慢；总耗时受 `limit_seconds`
          约束（默认 45s）；
        - 端点：轻量 GET `/api/v2/models/`（无需鉴权，实测 200 / 4902B，不触发
          WAF），**绝不打 chat/completions**（会烧额度、招风控）；
        - 判定：**有任何 HTTP 响应就算活**；只有连不上才 `mark_probe_failure`
          （短冷却 120s、不动 `failures`）；
        - 兜底：**全部出口都探测失败**时按「网络抖动 / 预算不够」处理，全部视为
          available 并打 WARNING —— 避免探测自身的抖动把整个池子打死。

        Args:
            token: 可选上游 token。
            account: 可选账号对象（决定浏览器指纹，与真实链路一致）。
            limit_seconds: 整批探测的总耗时上限（秒）。
            only: 只探测这些出口 id（None 表示全部）。

        Returns:
            每个出口一条结果：`{id, ok, reason, elapsed_seconds, state}`；
            全灭兜底时额外带 `assumed_alive=True`。
        """
        if not self._order:
            return []
        targets = [
            exit_id
            for exit_id in self._order
            if only is None or exit_id in {str(item or "") for item in only}
        ]
        if not targets:
            return []
        budget = max(0.1, float(limit_seconds))
        # socks5 握手排队做：并发度限制在 2，避免 6 条隧道同时握手互相拖慢。
        semaphore = asyncio.Semaphore(self.probe_concurrency)
        tasks = [
            asyncio.ensure_future(
                self._probe_one(
                    exit_id,
                    token=token,
                    account=account,
                    budget=budget,
                    semaphore=semaphore,
                )
            )
            for exit_id in targets
        ]
        try:
            gathered = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=budget)
        except asyncio.TimeoutError:
            # 总预算耗尽：把没跑完的探测取消，本轮结论按「不可用」处理。
            for task in tasks:
                if not task.done():
                    task.cancel()
            log.warning(
                "[ExitProbe] budget exhausted budget=%.0fs exits=%s concurrency=%s",
                budget,
                len(targets),
                self.probe_concurrency,
            )
            gathered = [
                {"id": exit_id, "ok": False, "reason": "probe_timeout", "elapsed_seconds": budget, "state": STATE_READY}
                for exit_id in targets
            ]
        results: list[dict[str, Any]] = []
        for item in gathered:
            if isinstance(item, Exception):  # pragma: no cover - _probe_one 内部已兜住
                log.debug("[ExitProbe] probe crashed error=%s", type(item).__name__)
                continue
            results.append(item)
        if results and all(not row.get("ok") for row in results):
            # 全灭兜底：所有出口同时探测失败，更可能是**探测自身**的问题
            # （网络抖动 / 预算不够 / 代理网关短暂抽风），而不是 6 个出口一起死。
            # 此时惩罚整个池子会把链路推到直连 → 被 WAF 打死（2026-12 事故）。
            log.warning("[ExitProbe] all exits failed probe — assuming network blip, keeping pool available")
            for row in results:
                self.mark_probe_alive(str(row.get("id", "")))
                row["ok"] = True
                row["assumed_alive"] = True
                row["state"] = STATE_READY
        return results

    async def _probe_one(
        self,
        exit_id: str,
        *,
        token: str,
        account: Any,
        budget: float,
        semaphore: asyncio.Semaphore | None = None,
    ) -> dict[str, Any]:
        """探测单个出口并记账（`probe_all` 的 worker）。

        Args:
            exit_id: 出口 id。
            token: 可选上游 token。
            account: 可选账号（决定浏览器指纹）。
            budget: 整批探测的总预算（秒），单出口超时不会超过它。
            semaphore: 限并发用的信号量（None 表示不限制，单测可省）。
        """
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return {"id": str(exit_id or ""), "ok": False, "reason": "unknown_exit", "elapsed_seconds": 0.0, "state": STATE_READY}
        started = time.monotonic()
        timeout = min(self.startup_probe_timeout_seconds, budget)
        if semaphore is None:
            outcome = await self._probe_outcome(
                state.url, token=token, account=account, timeout=timeout
            )
        else:
            # 只有真正发探测的这段被信号量卡住，记账不占并发名额。
            async with semaphore:
                outcome = await self._probe_outcome(
                    state.url, token=token, account=account, timeout=timeout
                )
        elapsed = max(0.0, time.monotonic() - started)
        connect_ms = elapsed * 1000.0
        if outcome.ok:
            self.mark_success(state.exit_id, connect_ms=connect_ms)
            log.info("[ExitProbe] ok exit=%s %.1fs reason=%s", state.exit_id, elapsed, outcome.reason)
        else:
            # 探测判死：**短冷却**、**不动 failures**（探测只是猜测，不该重罚）。
            self.mark_probe_failure(state.exit_id, outcome.reason, connect_ms=connect_ms)
            cooling = state.cooldown_until > self._clock()
            if cooling:
                log.warning(
                    "[ExitProbe] dead exit=%s reason=%s → cooldown %.0fs",
                    state.exit_id,
                    outcome.reason,
                    self.probe_cooldown_seconds,
                )
            else:
                log.warning(
                    "[ExitProbe] dead exit=%s reason=%s failures=%s/%s",
                    state.exit_id,
                    outcome.reason,
                    state.failures,
                    self.max_failures,
                )
        return {
            "id": state.exit_id,
            "ok": bool(outcome.ok),
            "reason": outcome.reason,
            "elapsed_seconds": round(elapsed, 3),
            "state": STATE_COOLING if state.cooldown_until > self._clock() else STATE_READY,
        }

