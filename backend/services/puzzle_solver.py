"""阿里云 PUZZLE（拼图）验证码缺口定位 —— 纯计算，无任何 I/O。

实测结论见 `docs/superpowers/specs/2026-10-31-multimodal-attachments-findings.md`
§8.3 / §9.3。素材由阿里云 CDN 下发：

    https://static-captcha-sgp.aliyuncs.com/qst/PUZZLE/online/903/<uuid>/shadow.png
    https://static-captcha-sgp.aliyuncs.com/qst/PUZZLE/online/903/<uuid>/back.png

- `shadow.png`：52×200 的「拼图块」素材，只有拼图块本身不透明（alpha 高），
  其余透明。它的不透明像素 bbox 给出拼图形状 + 缺口所在 y 带。
- `back.png`：296×200 的背景图，缺口处是「亮 + 低饱和」的拼图块形状。

定位算法（实测误差 ≤2px，已用叠加图逐像素核对）：

1. 对 `shadow.png` 取 `alpha > 150` 的像素，得到拼图掩码与 bbox
   `(piece_x0, piece_y0, piece_width, piece_height)`。
2. 对 `back.png` 每个像素算 ``score = max(R, G, B) / 255 - 1.2 * saturation``
   （saturation = (max - min) / max，HSV 定义）。
3. 用拼图掩码当权重，在拼图 y 带内做水平滑窗，取掩码内 score 均值最大的窗口
   起始 x，即缺口左边界 `x_best`。
4. `dx_file = x_best - piece_x0`（文件坐标）；显示层位移
   `dx_display = dx_file * (display_width / back_width)`。

本模块只依赖 Pillow（**惰性导入**，见 :func:`_load_pillow`），不依赖 numpy，单张
296×200 的图滑窗约 25 万次加法，耗时 ~0.2s，对验证码场景完全够用。

## 为什么 Pillow 必须惰性导入

生产镜像 `ghcr.io/sofs2005/q2a:dev` 目前不带 pillow（`requirements.txt` 里有，但镜像
还没重建）。若在模块顶层 `from PIL import Image`，缺依赖会直接打死
:mod:`backend.services.browser_upload_proxy`，进而打死网关进程 —— 「可选依赖缺失」
变成「整个服务起不来」，这是不可接受的降级。

因此约定：**导入本模块、启动服务、内联附件链路一律不受 Pillow 影响**；只有真正调用
:func:`locate_gap` 时才抛 :class:`PuzzleSolverUnavailable`（`PuzzleSolveError` 的子类），
由 `browser_upload_proxy` 识别为「浏览器上传不可用」并降级回内联 / 400。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any


#: 显示层 `#aliyunCaptcha-img` 的实测尺寸（300×200），back.png 是 296×200。
DEFAULT_DISPLAY_WIDTH = 300.0
DEFAULT_DISPLAY_HEIGHT = 200.0

#: 拼图块判定阈值：shadow.png 的 alpha 通道。
PIECE_ALPHA_THRESHOLD = 150

#: 缺口打分里的饱和度惩罚系数（实测最优值）。
SATURATION_PENALTY = 1.2

#: Pillow 缺失时对外统一的一句话（写进异常 message 与 WARNING 日志）。
PILLOW_MISSING_MESSAGE = "pillow not installed, browser upload fallback disabled"


def _load_pillow() -> Any:
    """惰性导入 Pillow 的 `Image` 模块；未安装（或装坏）时返回 None。

    与 :func:`backend.services.inline_media_ref._load_pillow` 保持同一写法：
    调用方拿到 None 必须自己降级，绝不能让 ImportError 冒到模块顶层。

    Returns:
        `PIL.Image` 模块对象，或 None（表示 Pillow 不可用）。
    """
    try:
        from PIL import Image  # type: ignore import-not-found
    except Exception:  # pragma: no cover - 生产镜像未装 pillow 时走这里
        return None
    return Image


def pillow_available() -> bool:
    """Pillow 是否可用（运维探针 / 单测用；不影响模块导入）。"""
    return _load_pillow() is not None


class PuzzleSolveError(Exception):
    """拼图素材无法解析 / 无法定位缺口。

    上层（browser_upload_proxy）捕获后退化为「本轮失败」，等下一轮新拼图，
    不能把整条上传链路打成异常。
    """


class PuzzleSolverUnavailable(PuzzleSolveError):
    """Pillow 未安装 —— 整个求解器不可用（不是某一张素材的问题）。

    继承 :class:`PuzzleSolveError`，所以既有的 `except PuzzleSolveError` 调用方
    自动拿到安全行为（本轮失败）；`browser_upload_proxy` 再单独识别它，直接降级为
    「浏览器上传不可用」，不消耗验证码轮次、也不计入熔断。
    """

    #: 便于 `getattr(exc, "unavailable", False)` 风格的判断。
    unavailable = True

    def __init__(self, message: str = PILLOW_MISSING_MESSAGE) -> None:
        super().__init__(str(message or "") or PILLOW_MISSING_MESSAGE)


@dataclass(frozen=True)
class PuzzleSolution:
    """一次缺口定位的结果（含文件坐标与显示坐标两套位移）。"""

    #: 文件坐标下的水平位移（back.png 像素）。
    dx_file: float
    #: 缺口窗口的掩码内平均得分（越大越可信）。
    score: float
    #: back.png 尺寸。
    image_width: int
    image_height: int
    #: 拼图块在 shadow.png 中的 bbox。
    piece_x0: int
    piece_y0: int
    piece_width: int
    piece_height: int
    #: 显示层宽度（`#aliyunCaptcha-img` 的 CSS 像素宽）。
    display_width: float

    @property
    def scale(self) -> float:
        """文件坐标 → 显示坐标的比例。"""
        if self.image_width <= 0:
            return 1.0
        return float(self.display_width) / float(self.image_width)

    @property
    def dx_display(self) -> float:
        """显示层需要移动的像素数（拼图块自身位移）。"""
        return float(self.dx_file) * self.scale

    @property
    def dx(self) -> float:
        """`dx_display` 的别名，便于调用方直接取用。"""
        return self.dx_display

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx_file": round(self.dx_file, 3),
            "dx_display": round(self.dx_display, 3),
            "scale": round(self.scale, 6),
            "score": round(self.score, 6),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "piece_x0": self.piece_x0,
            "piece_y0": self.piece_y0,
            "piece_width": self.piece_width,
            "piece_height": self.piece_height,
            "display_width": self.display_width,
        }


def _load_rgba(raw: bytes, *, label: str) -> Any:
    """把字节解码成 RGBA 图像；失败统一抛 :class:`PuzzleSolveError`。

    Pillow 缺失时抛 :class:`PuzzleSolverUnavailable`（Pillow 只在真正解码时才需要）。
    """
    data = bytes(raw or b"")
    if not data:
        raise PuzzleSolveError(f"{label}.png is empty")
    Image = _load_pillow()
    if Image is None:
        raise PuzzleSolverUnavailable()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGBA")
    except Exception as exc:  # Pillow 抛的异常类型五花八门
        raise PuzzleSolveError(f"{label}.png decode failed: {exc}") from exc


def _piece_mask(
    shadow: Any,
    alpha_threshold: int = PIECE_ALPHA_THRESHOLD,
) -> tuple[dict[int, list[int]], int, int, int, int]:
    """从 shadow.png 提取拼图掩码。

    Args:
        shadow: RGBA 的 shadow.png。
        alpha_threshold: alpha 高于该值的像素算拼图块本体（实测 150 最稳）。

    Returns:
        `(rows, x0, y0, width, height)`：`rows[dy]` 是该行所有不透明像素相对 bbox
        左边界的 dx 列表（用于滑窗内求均值），其余为 bbox。
    """
    width, height = shadow.size
    if width <= 0 or height <= 0:
        raise PuzzleSolveError("shadow.png has zero size")
    threshold = int(alpha_threshold or PIECE_ALPHA_THRESHOLD)
    alpha = shadow.getchannel("A").load()
    pixels: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if alpha[x, y] > threshold:
                pixels.append((x, y))
    if not pixels:
        raise PuzzleSolveError("shadow.png has no opaque puzzle piece")
    xs = [x for x, _ in pixels]
    ys = [y for _, y in pixels]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    rows: dict[int, list[int]] = {}
    for x, y in pixels:
        rows.setdefault(y - y0, []).append(x - x0)
    return rows, x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


def _score_band(back: Any, y0: int, y1: int) -> list[list[float]]:
    """计算 back.png 在 `[y0, y1]` y 带内的缺口得分矩阵（相对 y0 下标）。"""
    width, height = back.size
    band_top = max(0, min(y0, height - 1))
    band_bottom = max(band_top, min(y1, height - 1))
    pixels = back.load()
    band: list[list[float]] = []
    for y in range(band_top, band_bottom + 1):
        row: list[float] = []
        for x in range(width):
            r, g, b, _a = pixels[x, y]
            max_channel = r if r >= g and r >= b else (g if g >= b else b)
            min_channel = r if r <= g and r <= b else (g if g <= b else b)
            value = max_channel / 255.0
            if max_channel > 0:
                saturation = (max_channel - min_channel) / max_channel
            else:
                saturation = 0.0
            row.append(value - SATURATION_PENALTY * saturation)
        band.append(row)
    return band


def locate_gap(
    back_png: bytes,
    shadow_png: bytes,
    *,
    display_width: float = DEFAULT_DISPLAY_WIDTH,
    alpha_threshold: int = PIECE_ALPHA_THRESHOLD,
) -> PuzzleSolution:
    """定位拼图缺口，返回拼图块需要移动的位移。

    Args:
        back_png: `back.png` 原始字节（背景图）。
        shadow_png: `shadow.png` 原始字节（拼图块素材）。
        display_width: 显示层宽度（`#aliyunCaptcha-img` 的 CSS 像素宽），
            用于把文件坐标换算成显示坐标；默认取实测 300。
        alpha_threshold: 拼图块 alpha 阈值（仅用于日志/兼容，内部统一用常量）。

    Returns:
        :class:`PuzzleSolution`。

    Raises:
        PuzzleSolverUnavailable: Pillow 未安装（浏览器上传降级为不可用）。
        PuzzleSolveError: 素材为空 / 解码失败 / 找不到拼图块 / 拼图比背景还宽。
    """
    # Pillow 缺失是最常见的部署态（镜像没重建），必须在任何解码前给出明确异常，
    # 否则会先被「素材为空 / 解码失败」这类噪声错误掩盖。
    if _load_pillow() is None:
        raise PuzzleSolverUnavailable()
    back = _load_rgba(back_png, label="back")
    shadow = _load_rgba(shadow_png, label="shadow")

    rows, piece_x0, piece_y0, piece_width, piece_height = _piece_mask(shadow, alpha_threshold)
    image_width, image_height = back.size
    if piece_width > image_width:
        raise PuzzleSolveError(
            f"puzzle piece ({piece_width}px) is wider than back.png ({image_width}px)"
        )
    if piece_height > image_height:
        raise PuzzleSolveError(
            f"puzzle piece ({piece_height}px) is taller than back.png ({image_height}px)"
        )

    band = _score_band(back, piece_y0, piece_y0 + piece_height - 1)
    band_rows = len(band)
    mask_count = sum(len(dxs) for dxs in rows.values())
    if mask_count <= 0 or band_rows <= 0:
        raise PuzzleSolveError("empty puzzle mask or empty score band")

    max_start = image_width - piece_width
    best_score = float("-inf")
    best_start = 0
    for start in range(0, max_start + 1):
        total = 0.0
        for dy, dxs in rows.items():
            if dy < 0 or dy >= band_rows:
                continue
            row = band[dy]
            for dx in dxs:
                x = start + dx
                if 0 <= x < image_width:
                    total += row[x]
        mean = total / mask_count
        if mean > best_score:
            best_score = mean
            best_start = start

    if best_score == float("-inf"):
        raise PuzzleSolveError("sliding window produced no candidate")

    return PuzzleSolution(
        dx_file=float(best_start - piece_x0),
        score=float(best_score),
        image_width=int(image_width),
        image_height=int(image_height),
        piece_x0=int(piece_x0),
        piece_y0=int(piece_y0),
        piece_width=int(piece_width),
        piece_height=int(piece_height),
        display_width=float(display_width or DEFAULT_DISPLAY_WIDTH),
    )


def locate_piece_dx(
    back_png: bytes,
    shadow_png: bytes,
    *,
    display_width: float = DEFAULT_DISPLAY_WIDTH,
) -> float:
    """便捷函数：只返回显示层需要移动的拼图位移（px）。

    Raises:
        PuzzleSolverUnavailable: Pillow 未安装。
        PuzzleSolveError: 素材无法解析 / 定位失败。
    """
    return locate_gap(back_png, shadow_png, display_width=display_width).dx_display
