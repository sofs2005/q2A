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
#: 单次健康探测超时（秒，可被 `UPLOAD_EXIT_PROBE_TIMEOUT_SECONDS` 覆盖）
DEFAULT_PROBE_TIMEOUT_SECONDS = 10.0
#: 出口请求的**建连阶段**超时（秒，可被 `EXIT_CONNECT_TIMEOUT_SECONDS` 覆盖）。
#: 只压建连、不压读取：大模型生成可能很慢，读超时压到 5s 会误杀正常请求。
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
#: 启动探测的总耗时上限（秒，可被 `EXIT_PROBE_TOTAL_TIMEOUT_SECONDS` 覆盖）
DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS = 15.0

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

#: 探测出口用的上游状态接口（无需业务参数，200 即视为出口可用）。
_PROBE_PATH = "/api/v2/users/status"


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
            probe_timeout_seconds: 单次探测的整体超时（读阶段）。
            connect_timeout_seconds: 出口请求的**建连阶段**超时（只压 connect）。
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
        """经指定出口 GET 一次上游状态接口，判断出口是否可用。

        成功记 `mark_success`，失败记 `mark_failure`（连续失败会进入冷却，从而
        被 `pick` 跳过）。**永不抛异常**：探测本身失败就是「不可用」。

        Args:
            exit_id: 出口 id。
            token: 可选上游 token（`/api/v2/users/status` 需要鉴权）。
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
                ok = await self._do_probe(state.url, token=token, account=account)
            except Exception as exc:  # 探测不允许把异常抛给 /admin
                log.warning(
                    "[UploadExit] probe error exit=%s error=%s",
                    state.exit_id,
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                )
                ok = False
            if ok:
                self.mark_success(state.exit_id)
                log.info("[UploadExit] probe ok exit=%s", state.exit_id)
            else:
                self.mark_failure(state.exit_id, "probe_failed")
                log.warning("[UploadExit] probe failed exit=%s", state.exit_id)
            return bool(ok)

    async def _do_probe(
        self,
        proxy_url: str,
        *,
        token: str,
        account: Any,
        timeout: float | None = None,
    ) -> bool:
        """执行一次探测：优先用注入的假实现（单测），否则发真实请求。"""
        probe_timeout = (
            float(timeout) if timeout is not None and timeout > 0 else self.probe_timeout_seconds
        )
        if self._probe_fetch is not None:
            return bool(await self._probe_fetch(proxy_url, probe_timeout))
        return await self._http_probe(proxy_url, token=token, account=account, timeout=probe_timeout)

    async def _http_probe(
        self,
        proxy_url: str,
        *,
        token: str,
        account: Any,
        timeout: float | None = None,
    ) -> bool:
        """真实探测：curl_cffi + 该出口代理 GET `/api/v2/users/status`。"""
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
            return status == 200 and not looks_like_waf(body)
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
    ) -> tuple[bool, str]:
        """探测一次并给出 `(是否可用, 失败原因)`；**永不抛异常**。

        Args:
            proxy_url: 该出口的代理 URL。
            token: 可选上游 token。
            account: 可选账号（决定浏览器指纹）。
            timeout: 单次探测超时（秒）。

        Returns:
            `(True, "")` 或 `(False, reason)`；reason ∈
            `connect_failed` / `probe_failed`。
        """
        probe_timeout = (
            float(timeout) if timeout is not None and timeout > 0 else self.probe_timeout_seconds
        )
        try:
            ok = await asyncio.wait_for(
                self._do_probe(proxy_url, token=token, account=account, timeout=probe_timeout),
                timeout=probe_timeout,
            )
        except asyncio.TimeoutError:
            return False, "connect_failed"
        except Exception as exc:
            log.debug(
                "[UploadExit] probe error error=%s",
                f"{type(exc).__name__}: {str(exc)[:120]}",
            )
            return False, "connect_failed"
        return (True, "") if ok else (False, "probe_failed")

    async def probe_all(
        self,
        *,
        token: str = "",
        account: Any = None,
        limit_seconds: float = DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS,
        only: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """**并发**探测所有出口，失败的立即 `mark_failure`（达阈值即进冷却）。

        启动时调一次：把「已经连不上」的出口提前踢出轮转，后续请求就不会再挨个
        试死出口（生产上 4 个死出口 × 10s ≈ 21s 的白等由此消失）。

        - 并发：`asyncio.gather`，总耗时受 `limit_seconds` 约束（默认 15s）；
        - 端点：轻量 GET（默认 `/api/v2/users/status`），**绝不打 chat/completions**
          （会烧额度、招风控）；
        - 失败即记 `connect_failed` → 连续失败达 `max_failures` 立刻冷却。

        Args:
            token: 可选上游 token。
            account: 可选账号对象（决定浏览器指纹，与真实链路一致）。
            limit_seconds: 整批探测的总耗时上限（秒）。
            only: 只探测这些出口 id（None 表示全部）。

        Returns:
            每个出口一条结果：`{id, ok, reason, elapsed_seconds, state}`。
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
        tasks = [
            asyncio.ensure_future(self._probe_one(exit_id, token=token, account=account, budget=budget))
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
                "[ExitProbe] budget exhausted budget=%.0fs exits=%s",
                budget,
                len(targets),
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
        return results

    async def _probe_one(
        self,
        exit_id: str,
        *,
        token: str,
        account: Any,
        budget: float,
    ) -> dict[str, Any]:
        """探测单个出口并记账（`probe_all` 的 worker）。"""
        state = self._states.get(str(exit_id or ""))
        if state is None:
            return {"id": str(exit_id or ""), "ok": False, "reason": "unknown_exit", "elapsed_seconds": 0.0, "state": STATE_READY}
        started = time.monotonic()
        ok, reason = await self._probe_outcome(
            state.url,
            token=token,
            account=account,
            timeout=min(self.probe_timeout_seconds, budget),
        )
        elapsed = max(0.0, time.monotonic() - started)
        connect_ms = elapsed * 1000.0
        if ok:
            self.mark_success(state.exit_id, connect_ms=connect_ms)
            log.info("[ExitProbe] ok exit=%s %.1fs", state.exit_id, elapsed)
        else:
            self.mark_failure(state.exit_id, reason, connect_ms=connect_ms)
            cooling = state.cooldown_until > self._clock()
            if cooling:
                log.warning(
                    "[ExitProbe] dead exit=%s reason=%s → cooldown %.0fs",
                    state.exit_id,
                    reason,
                    self.cooldown_seconds,
                )
            else:
                log.warning(
                    "[ExitProbe] dead exit=%s reason=%s failures=%s/%s",
                    state.exit_id,
                    reason,
                    state.failures,
                    self.max_failures,
                )
        return {
            "id": state.exit_id,
            "ok": bool(ok),
            "reason": reason,
            "elapsed_seconds": round(elapsed, 3),
            "state": STATE_COOLING if state.cooldown_until > self._clock() else STATE_READY,
        }

