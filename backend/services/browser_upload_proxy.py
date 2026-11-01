"""浏览器上传代理：用远程 CDP 浏览器过阿里云拼图验证码，换 `getstsToken` 的 STS。

## 为什么这么设计

实测结论见 `docs/superpowers/specs/2026-10-31-multimodal-attachments-findings.md`
§8 / §9：

- `POST /api/v2/files/getstsToken` 被阿里云 WAF 的 **PUZZLE 拼图验证码**拦住，纯 HTTP
  客户端（含 curl_cffi 指纹伪装）一律拿到 WAF HTML；人工在真浏览器里拖动可以通过。
- 通过后接口返回 JSON（`file_id` / `file_path` / `file_url` / `bucketname` /
  `endpoint` / STS 临时密钥），网页端再自己 PUT 到 OSS。
- 一轮验证码不一定够（实测出现过 1 轮，也出现过 2 轮），必须循环到拿回 JSON。
- 鼠标位移 ↔ 拼图位移**不是 1:1**，是二次映射（见 :data:`BUILTIN_QUADRATIC_A`）。

因此本模块**只让浏览器做两件事**：过验证码 + 调 `getstsToken`。OSS PUT 与 ref 构造
仍然走 Python 侧（`oss2` + :mod:`backend.services.sts_ref_builder`），不依赖网页端的
内部状态，也不用往页面里塞消息。

## 关键实现

1. **惰性连接**：`playwright.chromium.connect_over_cdp(settings.UPLOAD_CDP_ENDPOINT)`，
   复用已有的 context / page（远程 browserless 或远程 Chrome）。
2. **登录态注入**：首次用某账号时用 `context.add_init_script` 写
   `localStorage.token` / `at_expire_time`，再把账号 cookies（含 `waf_cookies`）
   `context.add_cookies` 进去，然后 `goto` 首页并 `Escape` 关掉欢迎弹窗
   （有弹窗时 drop / 请求会异常）。
3. **在页面上下文 fetch**：`page.evaluate` 里调 `fetch('/api/v2/files/getstsToken')`，
   这样 cookie / TLS 指纹 / 设备指纹都由真浏览器承担。
4. **缺口定位**：:mod:`backend.services.puzzle_solver`（纯计算）。
5. **拖动比例自动标定**：`UPLOAD_DRAG_MAPPING=auto` 时读
   `data/drag_calibration.json`；没有缓存就做两点标定（拖 40px、80px，按住时读
   `#aliyunCaptcha-puzzle.x - #aliyunCaptcha-img.x`），拟合 `piece = a*mouse² + b*mouse`
   并落盘。标定会把当前拼图拖废，属预期成本。
6. **CDP 合成拖动**：`Input.dispatchMouseEvent`（不用 `page.mouse`），轨迹由
   :func:`build_drag_trajectory` 生成：总时长 2.6~3.4s、250~320 个 move 事件、
   相邻间隔 `max(1, gauss(8, 3))` ms、2~3 处 300~700ms 停顿、y 抖动 ±5px、
   起手前 hover 200~400ms。
7. **账号级熔断**：连续 `UPLOAD_CAPTCHA_MAX_ROUNDS` 轮失败后，该账号 10 分钟内不再
   尝试浏览器上传，避免把风控分越打越高。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from backend.services.puzzle_solver import PuzzleSolveError, locate_gap
from backend.services.sts_ref_builder import upload_filetype_from_content_type


log = logging.getLogger("qwen2api.browser_upload_proxy")

try:  # pragma: no cover - playwright 是可选依赖（生产镜像已装）
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover
    async_playwright = None

try:  # pragma: no cover - 指纹模块依赖 curl_cffi，缺失时退化为最小头
    from backend.core.browser_fingerprint import fingerprint_for_account
except Exception:  # pragma: no cover
    fingerprint_for_account = None

try:
    from backend.core.config import DEFAULT_QWEN_BX_VERSION, DEFAULT_QWEN_WEB_VERSION
except Exception:  # pragma: no cover
    DEFAULT_QWEN_WEB_VERSION = "0.3.28"
    DEFAULT_QWEN_BX_VERSION = "2.5.37"


QWEN_BASE_URL = "https://chat.qwen.ai"
GETSTS_PATH = "/api/v2/files/getstsToken"

# ---- 拖动映射 ----------------------------------------------------------------

DRAG_MAPPING_AUTO = "auto"
DRAG_MAPPING_LINEAR = "linear"
DRAG_MAPPING_QUADRATIC = "quadratic"
DRAG_MAPPING_MODES = (DRAG_MAPPING_AUTO, DRAG_MAPPING_LINEAR, DRAG_MAPPING_QUADRATIC)

#: 实测拟合出的二次映射常数：`piece_dx = a * mouse_dx^2 + b * mouse_dx`。
#: 8 个实测点拟合误差 < 0.1px（例：拼图要动 157px，鼠标要拖 199.7px）。
BUILTIN_QUADRATIC_A = 0.003555
BUILTIN_QUADRATIC_B = 0.075899

#: 二次项小于该值即视为线性（标定时两点几乎共线）。
QUADRATIC_EPS = 1e-6

# ---- 验证码 DOM --------------------------------------------------------------

SLIDER_SELECTOR = "#aliyunCaptcha-sliding-slider"
PUZZLE_SELECTOR = "#aliyunCaptcha-puzzle"
IMAGE_SELECTOR = "#aliyunCaptcha-img"
SLIDER_TRACK_SELECTOR = "#aliyunCaptcha-sliding-body"

#: 标定用的两个鼠标位移（px）。
CALIBRATION_PROBE_OFFSETS: tuple[float, ...] = (40.0, 80.0)

# ---- 熔断 --------------------------------------------------------------------

DEFAULT_CAPTCHA_COOLDOWN_SECONDS = 600.0
CAPTCHA_RECENT_WINDOW_SECONDS = 600.0

#: 上游没给 `num`（小时数）时的保守重试间隔（与 upstream_file_uploader 一致）。
DEFAULT_UPLOAD_RATE_LIMIT_RETRY_SECONDS = 3600.0

# ---- WAF 页面特征（与 upstream_file_uploader 保持一致的语义）------------------

_WAF_CHALLENGE_MARKERS = (
    "aliyun_waf",
    "aliyuncaptcha",
    "<!doctype html",
    "sceneid=",
)

_VERIFY_CODE_RE = re.compile(r"\b([TF]\d{3})\b")

# fetch() 不能设置的「禁止头」，浏览器自己会带（含 Cookie / UA / Referer / Origin）。
_FORBIDDEN_FETCH_HEADERS = frozenset(
    {
        "cookie",
        "user-agent",
        "referer",
        "origin",
        "host",
        "content-length",
        "connection",
        "accept-encoding",
    }
)

# ---- 页面内 JS --------------------------------------------------------------

_FETCH_STS_JS = """
async ([url, headers, body]) => {
  const resp = await fetch(url, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(body),
    credentials: 'include',
    mode: 'cors',
  });
  const text = await resp.text();
  return {status: resp.status, body: text};
}
"""

_CAPTCHA_ASSETS_JS = """
() => {
  const entries = performance.getEntriesByType('resource').map((e) => (e && e.name) || '');
  const pick = (suffix) => {
    const hits = entries.filter((u) => u.indexOf(suffix) >= 0);
    return hits.length ? hits[hits.length - 1] : '';
  };
  return {shadow: pick('/shadow.png'), back: pick('/back.png')};
}
"""

_IMAGE_BOX_JS = """
() => {
  const el = document.querySelector('#aliyunCaptcha-img');
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  return {width: rect.width, height: rect.height};
}
"""

_SLIDER_BOX_JS = """
() => {
  const el = document.querySelector('#aliyunCaptcha-sliding-slider');
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
}
"""

_PUZZLE_OFFSET_JS = """
() => {
  const puzzle = document.querySelector('#aliyunCaptcha-puzzle');
  const img = document.querySelector('#aliyunCaptcha-img');
  if (!puzzle || !img) return null;
  return puzzle.getBoundingClientRect().x - img.getBoundingClientRect().x;
}
"""


class BrowserUploadUnavailable(Exception):
    """浏览器上传不可用（未配置 / CDP 连不上 / 账号冷却中）。

    上层捕获后退化为内联；这**不是**错误路径，只打 warning。
    """


class BrowserUploadError(Exception):
    """浏览器上传可用但本次失败（验证码没通过 / 上游返回异常）。"""


class BrowserUploadRateLimited(BrowserUploadError):
    """上游 `getstsToken` 返回 `RateLimited`（每日上传配额用尽）。

    上层（`context_attachment_manager`）把它翻译成 `FileUploadRateLimitedError`，
    复用已有的「标记超限号 → 换号」逻辑。
    """

    def __init__(self, retry_after_seconds: float = 3600.0, message: str = "") -> None:
        self.retry_after_seconds = max(0.0, float(retry_after_seconds or 0))
        super().__init__(
            message or f"browser upload rate limited; retry after {self.retry_after_seconds:.0f}s"
        )


# ============================================================================
# 拖动映射
# ============================================================================


def normalize_drag_mapping(value: str) -> str:
    """归一化 `UPLOAD_DRAG_MAPPING`；非法值退回 `auto`。"""
    text = str(value or "").strip().lower()
    return text if text in DRAG_MAPPING_MODES else DRAG_MAPPING_AUTO


def linear_mouse_dx(piece_dx: float) -> float:
    """1:1 映射：鼠标位移 == 拼图位移。"""
    return max(0.0, float(piece_dx))


def quadratic_mouse_dx(
    piece_dx: float,
    a: float = BUILTIN_QUADRATIC_A,
    b: float = BUILTIN_QUADRATIC_B,
) -> float:
    """由拼图位移反解鼠标位移。

    `piece = a*mouse² + b*mouse`  ⇒
    `mouse = (-b + sqrt(b² + 4a*piece)) / (2a)`。

    `a` 接近 0（或为负）时退化为线性 `piece / b`，避免除零 / 开负方。
    """
    target = max(0.0, float(piece_dx))
    a_value = float(a or 0.0)
    b_value = float(b or 0.0)
    if a_value <= QUADRATIC_EPS:
        if b_value <= 0:
            return target
        return target / b_value
    discriminant = b_value * b_value + 4.0 * a_value * target
    return (-b_value + math.sqrt(max(0.0, discriminant))) / (2.0 * a_value)


@dataclass
class DragCalibration:
    """鼠标位移 ↔ 拼图位移的标定结果（`piece = a*mouse² + b*mouse`）。"""

    a: float = 0.0
    b: float = 1.0
    measured_at: float = 0.0
    samples: list[dict[str, float]] = field(default_factory=list)
    source: str = "measured"

    @property
    def is_linear(self) -> bool:
        return abs(float(self.a)) <= QUADRATIC_EPS

    def piece_dx(self, mouse_dx: float) -> float:
        """正向映射：鼠标位移 → 拼图位移。"""
        mouse = max(0.0, float(mouse_dx))
        return float(self.a) * mouse * mouse + float(self.b) * mouse

    def mouse_dx(self, piece_dx: float) -> float:
        """反向映射：拼图位移 → 需要拖多少鼠标。"""
        if self.is_linear:
            return linear_mouse_dx(piece_dx) / self.b if self.b > 0 else linear_mouse_dx(piece_dx)
        return quadratic_mouse_dx(piece_dx, self.a, self.b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": float(self.a),
            "b": float(self.b),
            "measured_at": float(self.measured_at),
            "samples": [dict(item) for item in self.samples],
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DragCalibration | None":
        """从 JSON dict 还原；字段非法返回 None（当作没有缓存）。"""
        if not isinstance(payload, dict):
            return None
        try:
            a = float(payload.get("a"))
            b = float(payload.get("b"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(a) or not math.isfinite(b) or b <= 0 or a < 0:
            return None
        samples: list[dict[str, float]] = []
        raw_samples = payload.get("samples")
        if isinstance(raw_samples, list):
            for item in raw_samples:
                if not isinstance(item, dict):
                    continue
                try:
                    samples.append(
                        {
                            "mouse_dx": float(item.get("mouse_dx")),
                            "piece_dx": float(item.get("piece_dx")),
                        }
                    )
                except (TypeError, ValueError):
                    continue
        try:
            measured_at = float(payload.get("measured_at") or 0.0)
        except (TypeError, ValueError):
            measured_at = 0.0
        return cls(
            a=a,
            b=b,
            measured_at=measured_at,
            samples=samples,
            source=str(payload.get("source") or "cache"),
        )


def fit_calibration(
    samples: Iterable[tuple[float, float] | dict[str, float]],
    *,
    now: float | None = None,
) -> "DragCalibration | None":
    """用 `(mouse_dx, piece_dx)` 采样点拟合 `piece = a*mouse² + b*mouse`。

    - 两个点（标定的常规情况）：解 2×2 线性方程组；
    - 两个以上：最小二乘（正规方程）；
    - 点太少 / 鼠标位移无差异：返回 None（调用方退回内置二次常数）；
    - 解出的二次项 ≤ 0（非物理）：退化为过原点的线性拟合。

    Args:
        samples: 采样点，元素可以是 `(mouse_dx, piece_dx)` 元组或同名字典。
        now: 注入的时间戳，便于测试。

    Returns:
        :class:`DragCalibration` 或 None。
    """
    points: list[tuple[float, float]] = []
    for item in samples:
        if isinstance(item, dict):
            try:
                mouse = float(item.get("mouse_dx"))
                piece = float(item.get("piece_dx"))
            except (TypeError, ValueError):
                continue
        else:
            try:
                mouse, piece = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
        if not math.isfinite(mouse) or not math.isfinite(piece):
            continue
        if mouse <= 0 or piece <= 0:
            continue
        points.append((mouse, piece))
    if len(points) < 2:
        return None
    if max(mouse for mouse, _ in points) - min(mouse for mouse, _ in points) <= QUADRATIC_EPS:
        return None

    # 正规方程：Σ(a*m² + b*m) 拟合 Σp，特征向量 [m², m]。
    s_m4 = sum(mouse**4 for mouse, _ in points)
    s_m3 = sum(mouse**3 for mouse, _ in points)
    s_m2 = sum(mouse**2 for mouse, _ in points)
    s_m2p = sum(mouse**2 * piece for mouse, piece in points)
    s_mp = sum(mouse * piece for mouse, piece in points)
    det = s_m4 * s_m2 - s_m3 * s_m3
    a_value = 0.0
    b_value = 0.0
    if abs(det) > 1e-9:
        a_value = (s_m2p * s_m2 - s_mp * s_m3) / det
        b_value = (s_m4 * s_mp - s_m3 * s_m2p) / det
    if not math.isfinite(a_value) or not math.isfinite(b_value) or a_value <= QUADRATIC_EPS or b_value <= 0:
        # 退化：过原点的线性最小二乘 b = Σ(m*p) / Σ(m²)
        b_value = s_mp / s_m2 if s_m2 > 0 else 0.0
        a_value = 0.0
    if b_value <= 0 or not math.isfinite(b_value):
        return None
    return DragCalibration(
        a=a_value,
        b=b_value,
        measured_at=float(now if now is not None else time.time()),
        samples=[{"mouse_dx": mouse, "piece_dx": piece} for mouse, piece in points],
        source="measured",
    )


def load_calibration(path: str | Path) -> "DragCalibration | None":
    """读 `data/drag_calibration.json`；缺失 / 损坏 / 字段非法都返回 None。"""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return DragCalibration.from_dict(payload if isinstance(payload, dict) else {})


def save_calibration(path: str | Path, calibration: DragCalibration) -> None:
    """把标定结果落盘（父目录自动创建）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mouse_dx_for_piece(
    piece_dx: float,
    *,
    mapping: str = DRAG_MAPPING_AUTO,
    calibration: "DragCalibration | None" = None,
) -> float:
    """按映射模式把「拼图位移」换算成「鼠标位移」。

    - `linear`：1:1
    - `quadratic`：内置二次常数反解
    - `auto`：有标定就用标定，没有就退回内置二次常数
    """
    mode = normalize_drag_mapping(mapping)
    if mode == DRAG_MAPPING_LINEAR:
        return linear_mouse_dx(piece_dx)
    if mode == DRAG_MAPPING_QUADRATIC:
        return quadratic_mouse_dx(piece_dx)
    if calibration is not None:
        return calibration.mouse_dx(piece_dx)
    return quadratic_mouse_dx(piece_dx)


@dataclass(frozen=True)
class MappingResolution:
    """一次拖动前的映射决策结果。"""

    mode: str
    calibration: "DragCalibration | None" = None
    calibrated_now: bool = False

    def mouse_dx(self, piece_dx: float) -> float:
        return mouse_dx_for_piece(piece_dx, mapping=self.mode, calibration=self.calibration)


# ============================================================================
# 拖动轨迹
# ============================================================================


@dataclass(frozen=True)
class DragMove:
    """一个 `Input.dispatchMouseEvent(mouseMoved)` 事件。"""

    x: float
    y: float
    #: 派发该事件前需要 sleep 的毫秒数（已含插入的停顿）。
    delay_ms: float
    #: 该步是否插入了「思考停顿」（仅用于日志/调试）。
    pause: bool = False


def build_drag_trajectory(
    start_x: float,
    start_y: float,
    mouse_dx: float,
    *,
    rng: random.Random | None = None,
    move_count_range: tuple[int, int] = (250, 320),
    pause_count_range: tuple[int, int] = (2, 3),
    pause_ms_range: tuple[float, float] = (300.0, 700.0),
    duration_range: tuple[float, float] = (2.6, 3.4),
    y_jitter: float = 5.0,
) -> list[DragMove]:
    """生成拟人拖动轨迹（与真人实测的 121 事件 / 3.13s / 中位 8ms 同量级）。

    约束（实测统计出来的参数，硬编码在这里以便单测直接断言）：

    - 总时长落在 `duration_range`（默认 2.6~3.4s）；
    - move 事件数落在 `move_count_range`（默认 250~320）；
    - 相邻间隔形状是 `max(1, gauss(8, 3))` ms，整体缩放到目标时长；
    - 插入 `pause_count_range` 处 300~700ms 停顿；
    - y 抖动 ±`y_jitter`；
    - 终点精确落在 `start_x + mouse_dx`。

    Returns:
        `DragMove` 列表（最后一个事件的 x 精确等于 `start_x + mouse_dx`）。
    """
    generator = rng or random.Random()
    distance = max(1.0, float(mouse_dx))
    move_count = max(2, int(generator.randint(int(move_count_range[0]), int(move_count_range[1]))))
    pause_count = max(0, int(generator.randint(int(pause_count_range[0]), int(pause_count_range[1]))))
    target_ms = generator.uniform(float(duration_range[0]), float(duration_range[1])) * 1000.0

    pauses = [generator.uniform(float(pause_ms_range[0]), float(pause_ms_range[1])) for _ in range(pause_count)]
    # 停顿最多占目标时长的一半，避免把间隔挤到 <1ms（用「压低单次停顿上限」而不是
    # 事后等比缩放，这样每次停顿仍落在 300~700ms 的真实区间内）。
    if pauses:
        pause_cap = min(
            float(pause_ms_range[1]),
            max(float(pause_ms_range[0]), 0.5 * target_ms / len(pauses)),
        )
        pauses = [generator.uniform(float(pause_ms_range[0]), pause_cap) for _ in range(len(pauses))]
    pause_budget = sum(pauses)
    move_budget = max(float(move_count), target_ms - pause_budget)

    # 间隔形状：max(1, gauss(8, 3))，再整体缩放到剩余时长。
    raw_intervals = [max(1.0, min(20.0, generator.gauss(8.0, 3.0))) for _ in range(move_count)]
    raw_total = sum(raw_intervals)
    scale = move_budget / raw_total if raw_total > 0 else 1.0
    intervals = [max(1.0, value * scale) for value in raw_intervals]
    # clamp 会引入微小误差，按比例再修一次总量。
    drift = move_budget - sum(intervals)
    if abs(drift) > 1e-9:
        adjust = drift / len(intervals)
        intervals = [max(1.0, value + adjust) for value in intervals]

    # 停顿插入位置：避开开头 10% 与结尾 10%。
    pause_slots: set[int] = set()
    if pauses:
        low = max(1, int(move_count * 0.1))
        high = max(low + 1, int(move_count * 0.9))
        candidates = list(range(low, high))
        generator.shuffle(candidates)
        pause_slots = set(candidates[: len(pauses)])

    moves: list[DragMove] = []
    pause_cursor = 0
    previous_y = float(start_y)
    previous_x = float(start_x)
    for index in range(move_count):
        progress = (index + 1) / move_count
        # 起手快、收尾慢（ease-out），再叠一点随机抖动。
        eased = 1.0 - (1.0 - progress) ** 1.35
        jitter = generator.uniform(-0.6, 0.6)
        x = start_x + distance * eased + jitter
        if index == move_count - 1:
            x = start_x + distance
        x = min(max(x, min(start_x, start_x + distance)), max(start_x, start_x + distance))
        # 不允许倒退：对位只看最后一帧，抖动只往前叠。
        x = max(x, previous_x)
        previous_x = x
        # y 用均值回归的随机游走，幅度 ±y_jitter。
        previous_y = previous_y + generator.gauss(0.0, 1.2)
        previous_y = max(start_y - y_jitter, min(start_y + y_jitter, previous_y))
        delay_ms = intervals[index]
        is_pause = False
        if index in pause_slots and pause_cursor < len(pauses):
            delay_ms += pauses[pause_cursor]
            pause_cursor += 1
            is_pause = True
        moves.append(DragMove(x=round(x, 3), y=round(previous_y, 3), delay_ms=round(delay_ms, 3), pause=is_pause))
    return moves


# ============================================================================
# 头 / cookie 工具
# ============================================================================


def looks_like_waf_challenge(body: str) -> bool:
    """响应体是否阿里云 WAF 验证页面（与 uploader 侧同一套特征）。"""
    lowered = str(body or "").lower()
    return any(marker in lowered for marker in _WAF_CHALLENGE_MARKERS)


def parse_cookie_header(text: str) -> dict[str, str]:
    """把 `a=1; b=2` 形式的 cookie 串解析成 dict（靠后的同名值覆盖靠前的）。"""
    parsed: dict[str, str] = {}
    for part in str(text or "").split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        if name:
            parsed[name] = value.strip()
    return parsed


def cookies_for_context(text: str, *, domain: str = ".qwen.ai") -> list[dict[str, Any]]:
    """把 cookie 串转成 playwright `context.add_cookies` 需要的结构。"""
    return [
        {"name": name, "value": value, "domain": domain, "path": "/"}
        for name, value in parse_cookie_header(text).items()
    ]


def token_expire_ms(token: str, *, now: float | None = None, default_ttl_seconds: float = 86400.0) -> int:
    """从 JWT 的 `exp` 推出网页端 `at_expire_time`（毫秒时间戳）。

    解不出来就用 `now + default_ttl_seconds`，保证 localStorage 里有个像样的值。
    """
    current = float(now if now is not None else time.time())
    parts = str(token or "").split(".")
    if len(parts) >= 2:
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload + padding)
            data = json.loads(decoded.decode("utf-8"))
            exp = float(data.get("exp"))
            if math.isfinite(exp) and exp > 0:
                return int(exp * 1000)
        except Exception:
            pass
    return int((current + float(default_ttl_seconds)) * 1000)


def extract_verify_code(text: str) -> str:
    """从 verify 响应里抠出 `T001` / `F001` / `F015` 这类结果码。"""
    matched = _VERIFY_CODE_RE.search(str(text or ""))
    return matched.group(1) if matched else ""


def build_request_headers(account: Any, *, request_id: str | None = None) -> dict[str, str]:
    """构造 `getstsToken` 的请求头（语义同 `QwenClient._build_chat_transport_headers`）。

    Cookie 由浏览器自己带（`credentials: 'include'`），因此这里**不**放 Cookie 头。
    """
    token = str(getattr(account, "token", "") or "")
    headers: dict[str, str] = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/json",
        "X-Request-Id": request_id or str(uuid.uuid4()),
        "Version": str(getattr(_settings_ref(), "QWEN_WEB_VERSION", "") or DEFAULT_QWEN_WEB_VERSION),
        "bx-v": str(getattr(_settings_ref(), "QWEN_BX_VERSION", "") or DEFAULT_QWEN_BX_VERSION),
        "source": "web",
        "Timezone": time.strftime("%a %b %d %Y %H:%M:%S GMT%z", time.localtime()),
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    fingerprint = None
    if fingerprint_for_account is not None:
        try:
            fingerprint = fingerprint_for_account(account)
        except Exception:
            fingerprint = None
    if fingerprint is not None:
        user_agent = str(getattr(fingerprint, "user_agent", "") or "")
        if user_agent:
            headers["User-Agent"] = user_agent
        sec_ch_ua = str(getattr(fingerprint, "sec_ch_ua", "") or "")
        if sec_ch_ua:
            headers["sec-ch-ua"] = sec_ch_ua
            headers["sec-ch-ua-mobile"] = str(getattr(fingerprint, "sec_ch_ua_mobile", "?0") or "?0")
            headers["sec-ch-ua-platform"] = str(getattr(fingerprint, "platform", "") or "")
    return headers


def sanitize_fetch_headers(headers: dict[str, str]) -> dict[str, str]:
    """剔除浏览器禁止脚本设置的请求头（它们由浏览器自己带）。"""
    return {
        name: value
        for name, value in (headers or {}).items()
        if str(name).strip().lower() not in _FORBIDDEN_FETCH_HEADERS
    }


_SETTINGS_SINGLETON: Any = None


def _settings_ref() -> Any:
    """惰性拿全局 settings（避免模块导入期就依赖 pydantic-settings）。"""
    global _SETTINGS_SINGLETON
    if _SETTINGS_SINGLETON is None:
        try:
            from backend.core.config import settings as backend_settings

            _SETTINGS_SINGLETON = backend_settings
        except Exception:  # pragma: no cover
            _SETTINGS_SINGLETON = _NullSettings()
    return _SETTINGS_SINGLETON


class _NullSettings:
    """config 不可用时的空配置（单测 / 极端环境兜底）。"""

    UPLOAD_CDP_ENDPOINT = ""
    UPLOAD_VIA_BROWSER = "auto"
    UPLOAD_CAPTCHA_MAX_ROUNDS = 3
    UPLOAD_CAPTCHA_TIMEOUT_SECONDS = 30
    UPLOAD_DRAG_MAPPING = "auto"
    UPLOAD_BROWSER_TIMEOUT_SECONDS = 120
    UPLOAD_DRAG_CALIBRATION_FILE = "data/drag_calibration.json"


def default_calibration_path() -> Path:
    """默认标定缓存路径：`data/drag_calibration.json`（可由配置覆盖）。"""
    configured = str(getattr(_settings_ref(), "UPLOAD_DRAG_CALIBRATION_FILE", "") or "").strip()
    if configured:
        return Path(configured)
    return Path("data") / "drag_calibration.json"


# ============================================================================
# 代理主体
# ============================================================================


class BrowserUploadProxy:
    """浏览器上传代理（单实例、单 page、串行使用）。"""

    def __init__(
        self,
        settings_obj: Any | None = None,
        *,
        calibration_path: str | Path | None = None,
        clock: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings_obj if settings_obj is not None else _settings_ref()
        self._calibration_path = Path(calibration_path) if calibration_path else default_calibration_path()
        self._clock = clock or time.time
        self._rng = rng or random.Random()

        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._cdp: Any = None

        # 连接管理锁 + 会话串行锁（一个 page，必须串行用）。
        self._connect_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()

        self._seeded_accounts: set[str] = set()
        self._failure_streak: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._calibration: DragCalibration | None = None
        self._calibration_loaded = False
        self._last_captcha_pass_at = 0.0
        self._last_verify_code = ""
        self._last_stats: dict[str, Any] = {}

    # -- 配置读取 -----------------------------------------------------------

    def _cfg(self, name: str, default: Any) -> Any:
        value = getattr(self.settings, name, default)
        return default if value is None else value

    @property
    def endpoint(self) -> str:
        return str(self._cfg("UPLOAD_CDP_ENDPOINT", "") or "").strip()

    @property
    def upload_mode(self) -> str:
        mode = str(self._cfg("UPLOAD_VIA_BROWSER", "auto") or "auto").strip().lower()
        return mode if mode in {"auto", "always", "never"} else "auto"

    @property
    def drag_mapping_mode(self) -> str:
        return normalize_drag_mapping(self._cfg("UPLOAD_DRAG_MAPPING", DRAG_MAPPING_AUTO))

    @property
    def max_captcha_rounds(self) -> int:
        try:
            return max(1, int(self._cfg("UPLOAD_CAPTCHA_MAX_ROUNDS", 3)))
        except (TypeError, ValueError):
            return 3

    @property
    def captcha_timeout_seconds(self) -> float:
        try:
            return max(1.0, float(self._cfg("UPLOAD_CAPTCHA_TIMEOUT_SECONDS", 30)))
        except (TypeError, ValueError):
            return 30.0

    @property
    def browser_timeout_seconds(self) -> float:
        try:
            return max(1.0, float(self._cfg("UPLOAD_BROWSER_TIMEOUT_SECONDS", 120)))
        except (TypeError, ValueError):
            return 120.0

    @property
    def enabled(self) -> bool:
        """是否可能启用：endpoint 已配置且模式不是 never。"""
        return bool(self.endpoint) and self.upload_mode != "never"

    # -- 熔断 ---------------------------------------------------------------

    def is_cooled_down(self, email: str) -> bool:
        """该账号是否还在冷却窗口内（10 分钟内不再尝试浏览器上传）。"""
        key = str(email or "")
        until = float(self._cooldown_until.get(key, 0.0) or 0.0)
        return until > self._clock()

    def cooldown_remaining(self, email: str) -> float:
        until = float(self._cooldown_until.get(str(email or ""), 0.0) or 0.0)
        return max(0.0, until - self._clock())

    def failure_streak(self, email: str) -> int:
        return int(self._failure_streak.get(str(email or ""), 0) or 0)

    def _register_failure(self, email: str, count: int = 1) -> None:
        key = str(email or "")
        streak = self.failure_streak(key) + max(1, int(count))
        self._failure_streak[key] = streak
        if streak >= self.max_captcha_rounds:
            self._cooldown_until[key] = self._clock() + DEFAULT_CAPTCHA_COOLDOWN_SECONDS
            log.warning(
                "[BrowserUpload] circuit breaker open account=%s streak=%s cooldown=%ss",
                key,
                streak,
                int(DEFAULT_CAPTCHA_COOLDOWN_SECONDS),
            )

    def _on_success(self, email: str) -> None:
        key = str(email or "")
        self._failure_streak[key] = 0
        self._cooldown_until.pop(key, None)

    def recent_captcha_pass(self, *, window_seconds: float = CAPTCHA_RECENT_WINDOW_SECONDS) -> bool:
        """最近是否成功过验证码（运维探针用）。"""
        return bool(self._last_captcha_pass_at) and (self._clock() - self._last_captcha_pass_at) <= window_seconds

    # -- 可被单测替换的小工具 ------------------------------------------------

    async def _sleep(self, seconds: float) -> None:
        """所有等待都走这里，单测替换成 no-op 即可零耗时跑完整流程。"""
        delay = float(seconds or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)

    async def _captcha_assets(self, page: Any) -> tuple[str, str] | None:
        """从 `performance.getEntriesByType('resource')` 里取最近的一对素材 URL。"""
        try:
            payload = await page.evaluate(_CAPTCHA_ASSETS_JS)
        except Exception as exc:
            log.warning("[BrowserUpload] read captcha resource entries failed: %s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        shadow = str(payload.get("shadow") or "").strip()
        back = str(payload.get("back") or "").strip()
        if not shadow or not back:
            return None
        return shadow, back

    async def _download(self, page: Any, url: str) -> bytes:
        """用 `page.request.get` 下载素材（复用浏览器上下文的 cookie / 代理）。"""
        response = await page.request.get(url)
        body = await response.body()
        return bytes(body or b"")

    async def _display_width(self, page: Any) -> float:
        """`#aliyunCaptcha-img` 的显示宽度；读不到就用实测 300。"""
        try:
            payload = await page.evaluate(_IMAGE_BOX_JS)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            try:
                width = float(payload.get("width") or 0.0)
            except (TypeError, ValueError):
                width = 0.0
            if width > 0:
                return width
        return 300.0

    async def _slider_box(self, page: Any) -> dict[str, float] | None:
        try:
            payload = await page.evaluate(_SLIDER_BOX_JS)
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            return None
        try:
            return {
                "x": float(payload.get("x") or 0.0),
                "y": float(payload.get("y") or 0.0),
                "width": float(payload.get("width") or 0.0),
                "height": float(payload.get("height") or 0.0),
            }
        except (TypeError, ValueError):
            return None

    async def _puzzle_offset(self, page: Any) -> float | None:
        """拼图块相对背景图左边缘的位移（按住时读，显示坐标）。"""
        try:
            value = await page.evaluate(_PUZZLE_OFFSET_JS)
        except Exception:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _slider_visible(self, page: Any) -> bool:
        """验证码滑块是否还在（还在 = 本轮没通过）。"""
        query = getattr(page, "query_selector", None)
        if not callable(query):
            return False
        try:
            element = await query(SLIDER_SELECTOR)
        except Exception:
            return False
        if element is None:
            return False
        is_visible = getattr(element, "is_visible", None)
        if callable(is_visible):
            try:
                return bool(await is_visible())
            except Exception:
                return True
        return True

    # -- 连接 / 登录态 ------------------------------------------------------

    async def _ensure_page(self, account: Any) -> tuple[Any, Any]:
        """拿到可用的 `(page, cdp)`；必要时连接、注入登录态。"""
        async with self._connect_lock:
            await self._ensure_connected_locked()
            await self._seed_account_locked(account)
            page = self._page
            cdp = self._cdp
        if page is None or cdp is None:
            raise BrowserUploadUnavailable("browser page is not available")
        return page, cdp

    def _page_is_usable(self) -> bool:
        page = self._page
        if page is None:
            return False
        checker = getattr(page, "is_closed", None)
        if callable(checker):
            try:
                if checker():
                    return False
            except Exception:
                return False
        return True

    async def _ensure_connected_locked(self) -> None:
        """（持锁）建立 / 复用 CDP 连接。"""
        if self._page_is_usable():
            return
        if async_playwright is None:
            raise BrowserUploadUnavailable("playwright is not installed")
        endpoint = self.endpoint
        if not endpoint:
            raise BrowserUploadUnavailable("UPLOAD_CDP_ENDPOINT is not configured")
        if self._playwright is None:
            try:
                self._playwright = await async_playwright().start()
            except Exception as exc:
                raise BrowserUploadUnavailable(f"playwright start failed: {exc}") from exc
        try:
            browser = await self._playwright.chromium.connect_over_cdp(endpoint)
        except Exception as exc:
            raise BrowserUploadUnavailable(f"connect_over_cdp({endpoint}) failed: {exc}") from exc
        self._browser = browser
        try:
            contexts = list(getattr(browser, "contexts", []) or [])
            context = contexts[0] if contexts else await browser.new_context()
            self._context = context
            pages = list(getattr(context, "pages", []) or [])
            page = pages[0] if pages else await context.new_page()
            self._page = page
            self._cdp = await context.new_cdp_session(page)
        except Exception as exc:
            raise BrowserUploadUnavailable(f"prepare browser page failed: {exc}") from exc
        await self._install_response_hook(self._page)
        log.info("[BrowserUpload] connected cdp endpoint=%s", endpoint)

    async def _install_response_hook(self, page: Any) -> None:
        """监听 verify 响应，记录 `T001` / `F001` 之类的验证码结果码。"""
        register = getattr(page, "on", None)
        if not callable(register):
            return

        async def _handler(response: Any) -> None:
            try:
                url = str(getattr(response, "url", "") or "")
                if "verify" not in url.lower():
                    return
                text = await response.text()
                code = extract_verify_code(text)
                if code:
                    self._last_verify_code = code
            except Exception:
                return

        try:
            register("response", _handler)
        except Exception:
            return

    async def _seed_account_locked(self, account: Any) -> None:
        """（持锁）首次使用该账号时注入登录态。"""
        email = str(getattr(account, "email", "") or "")
        token = str(getattr(account, "token", "") or "")
        if not token:
            raise BrowserUploadUnavailable(f"account {email} has no token")
        key = f"{email}:{token[-12:]}"
        if key in self._seeded_accounts:
            return
        context = self._context
        page = self._page
        if context is None or page is None:
            raise BrowserUploadUnavailable("browser context is not ready")

        expire_ms = token_expire_ms(token, now=self._clock())
        init_script = (
            "(() => {\n"
            "  try {\n"
            f"    localStorage.setItem('token', {json.dumps(token)});\n"
            f"    localStorage.setItem('at_expire_time', {json.dumps(str(expire_ms))});\n"
            "  } catch (e) {}\n"
            "})();\n"
        )
        try:
            await context.add_init_script(script=init_script)
        except Exception as exc:
            raise BrowserUploadUnavailable(f"inject localStorage failed: {exc}") from exc

        cookies: list[dict[str, Any]] = []
        cookies.extend(cookies_for_context(str(getattr(account, "cookies", "") or "")))
        cookies.extend(cookies_for_context(self._valid_waf_cookie(account)))
        if cookies:
            try:
                await context.add_cookies(cookies)
            except Exception as exc:
                log.warning("[BrowserUpload] add_cookies failed account=%s error=%s", email, exc)

        try:
            await page.goto(f"{QWEN_BASE_URL}/", wait_until="domcontentloaded", timeout=int(self.browser_timeout_seconds * 1000))
        except Exception as exc:
            raise BrowserUploadUnavailable(f"goto {QWEN_BASE_URL} failed: {exc}") from exc
        # 实测：欢迎弹窗存在时 drop / 请求会异常，先 Escape 关掉。
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
        self._seeded_accounts.add(key)
        log.info("[BrowserUpload] seeded login state account=%s cookies=%s", email, len(cookies))

    @staticmethod
    def _valid_waf_cookie(account: Any) -> str:
        """未过期的 WAF cookie（`acw_tc` 等）；过期或缺失返回空串。"""
        waf = str(getattr(account, "waf_cookies", "") or "").strip()
        if not waf:
            return ""
        try:
            expires = float(getattr(account, "waf_cookies_expires_at", 0) or 0)
        except (TypeError, ValueError):
            expires = 0.0
        if expires <= 0 or expires <= time.time():
            return ""
        return waf

    # -- getstsToken --------------------------------------------------------

    async def _request_sts(
        self,
        page: Any,
        account: Any,
        filename: str,
        content_type: str,
        size: int,
    ) -> tuple[str, Any]:
        """在页面上下文里调 `getstsToken`。

        Returns:
            `("json", sts_data)` 或 `("waf", body)`。

        Raises:
            BrowserUploadError: 非 WAF 的硬失败（网络 / 非 JSON / 缺 file_id / 配额）。
        """
        headers = sanitize_fetch_headers(build_request_headers(account))
        body = {
            "filename": filename,
            # 官网网页端传字符串；与官方抓包保持一致。
            "filesize": str(int(size)),
            "filetype": upload_filetype_from_content_type(content_type),
        }
        try:
            result = await page.evaluate(_FETCH_STS_JS, [f"{QWEN_BASE_URL}{GETSTS_PATH}", headers, body])
        except Exception as exc:
            raise BrowserUploadError(f"page fetch {GETSTS_PATH} failed: {exc}") from exc
        if not isinstance(result, dict):
            raise BrowserUploadError(f"page fetch {GETSTS_PATH} returned unexpected payload: {result!r}")
        status = int(result.get("status") or 0)
        text = str(result.get("body") or "")
        if looks_like_waf_challenge(text):
            return "waf", text
        try:
            parsed = json.loads(text or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise BrowserUploadError(
                f"{GETSTS_PATH} returned non-JSON body (status={status}): {text[:200]}"
            ) from exc
        if status != 200:
            raise BrowserUploadError(f"{GETSTS_PATH} failed: status={status} body={text[:200]}")
        payload = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(payload, dict) or not str(payload.get("file_id") or "").strip():
            code = str((payload or {}).get("code") or "") if isinstance(payload, dict) else ""
            if code == "RateLimited":
                try:
                    hours = float((payload or {}).get("num"))
                except (TypeError, ValueError):
                    hours = 0.0
                retry_after = hours * 3600.0 if hours > 0 else DEFAULT_UPLOAD_RATE_LIMIT_RETRY_SECONDS
                raise BrowserUploadRateLimited(
                    retry_after_seconds=retry_after,
                    message=str((payload or {}).get("details") or "file upload rate limited"),
                )
            raise BrowserUploadError(f"{GETSTS_PATH} missing file_id: {text[:200]}")
        return "json", payload

    # -- 验证码求解 ---------------------------------------------------------

    async def solve_puzzle(self, page: Any, cdp: Any) -> str | None:
        """等验证码 → 定位缺口 → 标定/换算 → CDP 拖动 → 校验。

        Returns:
            通过时返回 verify 结果码（拿不到真实响应码时用 `T001` 表示「滑块消失」），
            失败返回 None（调用方会重新发请求拿新拼图）。
        """
        timeout_ms = int(self.captcha_timeout_seconds * 1000)
        try:
            await page.wait_for_selector(SLIDER_SELECTOR, timeout=timeout_ms)
        except Exception as exc:
            log.warning("[BrowserUpload] captcha slider did not appear: %s", exc)
            return None
        # 验证码有入场动画：等动画结束再读坐标 / 素材，否则会拿到过渡中间值。
        await self._sleep(self._rng.uniform(0.4, 0.8))

        mapping_mode = self.drag_mapping_mode
        # 最多 3 轮：auto 标定会把当前拼图拖废，需要重新取素材再拖一次。
        for _attempt in range(3):
            mapping = await self._resolve_mapping(page, cdp, mode=mapping_mode)
            if mapping.calibrated_now:
                # 标定拖废了当前拼图，等风控下发新拼图（素材 URL 会变）再继续。
                await self._sleep(self._rng.uniform(1.0, 1.8))
                continue
            assets = await self._captcha_assets(page)
            if assets is None:
                log.warning("[BrowserUpload] captcha assets not found (shadow/back)")
                return None
            shadow_url, back_url = assets
            try:
                shadow = await self._download(page, shadow_url)
                back = await self._download(page, back_url)
            except Exception as exc:
                log.warning("[BrowserUpload] captcha asset download failed: %s", exc)
                return None
            display_width = await self._display_width(page)
            try:
                solution = await asyncio.to_thread(
                    locate_gap, back, shadow, display_width=display_width
                )
            except PuzzleSolveError as exc:
                log.warning("[BrowserUpload] puzzle solve failed: %s", exc)
                return None
            mouse_dx = mapping.mouse_dx(solution.dx_display)
            box = await self._slider_box(page)
            if box is None:
                log.warning("[BrowserUpload] captcha slider box unavailable")
                return None
            start_x = box["x"] + box["width"] / 2.0
            start_y = box["y"] + box["height"] / 2.0
            self._last_verify_code = ""
            await self._drag(cdp, start_x, start_y, mouse_dx)
            await self._sleep(self._rng.uniform(3.0, 5.0))
            passed = not await self._slider_visible(page)
            verify = self._last_verify_code or ("T001" if passed else "F001")
            log.info(
                "[BrowserUpload] captcha drag mapping=%s piece_dx=%.2f mouse_dx=%.2f verify=%s passed=%s",
                mapping.mode,
                solution.dx_display,
                mouse_dx,
                verify,
                passed,
            )
            return verify if passed else None
        log.warning("[BrowserUpload] captcha solve aborted after calibration rounds")
        return None

    async def _resolve_mapping(self, page: Any, cdp: Any, *, mode: str) -> MappingResolution:
        """决定本次拖动用哪种映射；`auto` 且无缓存时现场标定。"""
        normalized = normalize_drag_mapping(mode)
        if normalized == DRAG_MAPPING_LINEAR:
            return MappingResolution(mode=DRAG_MAPPING_LINEAR)
        if normalized == DRAG_MAPPING_QUADRATIC:
            return MappingResolution(mode=DRAG_MAPPING_QUADRATIC)
        cached = self.cached_calibration()
        if cached is not None:
            return MappingResolution(mode=DRAG_MAPPING_AUTO, calibration=cached)
        measured = await self.calibrate_drag(page, cdp)
        return MappingResolution(
            mode=DRAG_MAPPING_AUTO,
            calibration=measured,
            calibrated_now=measured is not None,
        )

    def cached_calibration(self) -> DragCalibration | None:
        """读一次 `data/drag_calibration.json` 并缓存到内存。"""
        if not self._calibration_loaded:
            self._calibration_loaded = True
            self._calibration = load_calibration(self._calibration_path)
            if self._calibration is not None:
                log.info(
                    "[BrowserUpload] drag calibration loaded path=%s a=%.6f b=%.6f source=%s",
                    self._calibration_path,
                    self._calibration.a,
                    self._calibration.b,
                    self._calibration.source,
                )
        return self._calibration

    async def calibrate_drag(self, page: Any, cdp: Any) -> DragCalibration | None:
        """两点标定：拖 40px / 80px，按住时读拼图位移，拟合二次曲线并落盘。

        标定本身会让当前拼图失败、触发新拼图，属预期成本。
        """
        samples: list[tuple[float, float]] = []
        for target in CALIBRATION_PROBE_OFFSETS:
            offset = await self._probe_hold_offset(page, cdp, target)
            if offset is None or offset <= 0:
                log.warning(
                    "[BrowserUpload] drag calibration probe failed mouse_dx=%.1f offset=%s",
                    target,
                    offset,
                )
                return None
            samples.append((float(target), float(offset)))
            await self._sleep(self._rng.uniform(0.5, 0.9))
        calibration = fit_calibration(samples, now=self._clock())
        if calibration is None:
            log.warning("[BrowserUpload] drag calibration fit failed samples=%s", samples)
            return None
        self._calibration = calibration
        self._calibration_loaded = True
        try:
            save_calibration(self._calibration_path, calibration)
        except OSError as exc:
            log.warning("[BrowserUpload] save drag calibration failed path=%s error=%s", self._calibration_path, exc)
        log.info(
            "[BrowserUpload] drag calibration measured a=%.6f b=%.6f samples=%s path=%s",
            calibration.a,
            calibration.b,
            samples,
            self._calibration_path,
        )
        return calibration

    async def _probe_hold_offset(self, page: Any, cdp: Any, mouse_dx: float) -> float | None:
        """按住滑块拖 `mouse_dx`，等过渡结束后读拼图位移，再松开。"""
        box = await self._slider_box(page)
        if box is None:
            return None
        start_x = box["x"] + box["width"] / 2.0
        start_y = box["y"] + box["height"] / 2.0
        await self._cdp_press(cdp, start_x, start_y)
        try:
            steps = 12
            for index in range(1, steps + 1):
                await self._sleep(0.016)
                await self._cdp_move(cdp, start_x + mouse_dx * index / steps, start_y)
            # 必须等 CSS 过渡结束再读，否则读到的是动画中间值（历史上被这个坑过）。
            await self._sleep(0.6)
            offset = await self._puzzle_offset(page)
        finally:
            await self._cdp_release(cdp, start_x + mouse_dx, start_y)
        return offset

    # -- CDP 拖动 -----------------------------------------------------------

    async def _cdp_press(self, cdp: Any, x: float, y: float) -> None:
        await cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": float(x),
                "y": float(y),
                "button": "left",
                "buttons": 1,
                "clickCount": 1,
                "force": 0.5,
            },
        )

    async def _cdp_move(self, cdp: Any, x: float, y: float) -> None:
        await cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseMoved",
                "x": float(x),
                "y": float(y),
                "button": "left",
                "buttons": 1,
                "clickCount": 0,
                "force": 0.5,
            },
        )

    async def _cdp_release(self, cdp: Any, x: float, y: float) -> None:
        await cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": float(x),
                "y": float(y),
                "button": "left",
                "buttons": 0,
                "clickCount": 1,
                "force": 0.5,
            },
        )

    async def _drag(self, cdp: Any, start_x: float, start_y: float, mouse_dx: float) -> None:
        """用 CDP 合成一次拟人拖动（`page.mouse` 的事件序列会被判定为脚本）。"""
        moves = build_drag_trajectory(start_x, start_y, mouse_dx, rng=self._rng)
        # 起手前先在滑块上 hover 200~400ms。
        await self._sleep(self._rng.uniform(0.2, 0.4))
        await cdp.send(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseMoved",
                "x": float(start_x),
                "y": float(start_y),
                "button": "none",
                "buttons": 0,
                "clickCount": 0,
            },
        )
        await self._cdp_press(cdp, start_x, start_y)
        for move in moves:
            if move.delay_ms > 0:
                await self._sleep(move.delay_ms / 1000.0)
            await self._cdp_move(cdp, move.x, move.y)
        last = moves[-1] if moves else DragMove(x=start_x + mouse_dx, y=start_y, delay_ms=0.0)
        await self._cdp_release(cdp, last.x, last.y)

    # -- 对外主流程 ---------------------------------------------------------

    async def fetch_sts(
        self,
        account: Any,
        filename: str,
        content_type: str,
        size: int,
    ) -> dict[str, Any] | None:
        """过验证码并拿到 `getstsToken` 的 `data`（含 STS 临时凭证）。

        Args:
            account: 账号对象（需要 `email` / `token` / `cookies` / `waf_cookies`）。
            filename: 上传文件名。
            content_type: 附件 mime。
            size: 原始字节数。

        Returns:
            STS `data` 字典（含 `file_id` / `file_path` / `file_url` / `bucketname` /
            `endpoint` / 临时密钥）；浏览器上传被关闭时返回 None。

        Raises:
            BrowserUploadUnavailable: 未配置 / 连不上 / 账号冷却中。
            BrowserUploadError: 验证码连续失败或上游返回异常。
        """
        if not self.enabled:
            return None
        email = str(getattr(account, "email", "") or "")
        if self.is_cooled_down(email):
            raise BrowserUploadUnavailable(
                f"browser upload cooling down for {email} ({int(self.cooldown_remaining(email))}s left)"
            )
        started = self._clock()
        page, cdp = await self._ensure_page(account)
        max_rounds = self.max_captcha_rounds
        rounds = 0
        verify = ""
        async with self._session_lock:
            while True:
                try:
                    outcome, payload = await self._request_sts(page, account, filename, content_type, size)
                except BrowserUploadError:
                    self._register_failure(email)
                    self._log_upload(account, filename, size, rounds, verify, started, ok=False)
                    raise
                if outcome == "json":
                    self._on_success(email)
                    self._log_upload(account, filename, size, rounds, verify, started, ok=True)
                    return payload
                # WAF 页面：求解验证码
                if rounds >= max_rounds:
                    self._register_failure(email, count=max_rounds)
                    self._log_upload(account, filename, size, rounds, verify, started, ok=False)
                    raise BrowserUploadError(
                        f"captcha not passed after {rounds} rounds ({GETSTS_PATH} still returns WAF page)"
                    )
                rounds += 1
                solved = await self.solve_puzzle(page, cdp)
                if solved:
                    verify = solved
                    self._last_captcha_pass_at = self._clock()
                else:
                    self._register_failure(email)
                    await self._sleep(self._rng.uniform(0.5, 1.2))

    def _log_upload(
        self,
        account: Any,
        filename: str,
        size: int,
        rounds: int,
        verify: str,
        started: float,
        *,
        ok: bool,
    ) -> None:
        elapsed = max(0.0, self._clock() - started)
        self._last_stats = {
            "account": str(getattr(account, "email", "") or ""),
            "file": str(filename or ""),
            "size": int(size or 0),
            "rounds": int(rounds),
            "verify": str(verify or ""),
            "elapsed": round(elapsed, 3),
            "ok": bool(ok),
        }
        log.info(
            "[BrowserUpload] account=%s file=%s size=%s rounds=%s verify=%s elapsed=%.2fs ok=%s",
            self._last_stats["account"],
            self._last_stats["file"],
            self._last_stats["size"],
            self._last_stats["rounds"],
            self._last_stats["verify"] or "-",
            elapsed,
            ok,
        )

    @property
    def last_stats(self) -> dict[str, Any]:
        """最近一次上传的结构化统计（运维接口用）。"""
        return dict(self._last_stats)

    async def probe(self) -> dict[str, Any]:
        """运维探针：连一次 CDP，回报连接状态与浏览器版本。"""
        result: dict[str, Any] = {
            "connected": False,
            "browser_version": "",
            "captcha_passed_recently": self.recent_captcha_pass(),
            "endpoint": self.endpoint,
            "drag_mapping": self.drag_mapping_mode,
            "calibration": None,
        }
        calibration = self.cached_calibration()
        if calibration is not None:
            result["calibration"] = {
                "a": calibration.a,
                "b": calibration.b,
                "source": calibration.source,
                "measured_at": calibration.measured_at,
            }
        if not self.endpoint:
            result["error"] = "UPLOAD_CDP_ENDPOINT is not configured"
            return result
        try:
            async with self._connect_lock:
                await self._ensure_connected_locked()
                browser = self._browser
            version = ""
            getter = getattr(browser, "version", "")
            version = str(getter() if callable(getter) else (getter or ""))
            result["connected"] = True
            result["browser_version"] = version
        except Exception as exc:
            result["error"] = str(exc)
        return result

    async def close(self) -> None:
        """释放 CDP 连接与 playwright（进程退出时调用）。"""
        async with self._connect_lock:
            cdp, browser, playwright = self._cdp, self._browser, self._playwright
            self._cdp = None
            self._browser = None
            self._context = None
            self._page = None
            self._playwright = None
            self._seeded_accounts.clear()
            if cdp is not None:
                detach = getattr(cdp, "detach", None)
                if callable(detach):
                    try:
                        await detach()
                    except Exception:
                        pass
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass
            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception:
                    pass
