"""拼图缺口定位单测：用两组真实 `shadow.png` / `back.png` 核对位移。

fixture 来自实测现场（`generated/q2a-mm/cap/1`、`cap/2`，已复制到
`tests/fixtures/captcha/`）：

- `cap1`：人工核对过的真值 ≈ **218px**（显示坐标，`#aliyunCaptcha-img` 300px 宽）
- `cap2`：人工核对过的真值 ≈ **153px**

算法说明见 `backend/services/puzzle_solver.py` 与 findings 文档 §8.3。
"""

import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path

from PIL import Image

from backend.services import puzzle_solver
from backend.services.puzzle_solver import (
    DEFAULT_DISPLAY_WIDTH,
    PILLOW_MISSING_MESSAGE,
    PuzzleSolveError,
    PuzzleSolverUnavailable,
    locate_gap,
    locate_piece_dx,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "captcha"

#: 人工用叠加图逐像素核对过的真值（显示坐标）。
CAPTCHA_TRUTH = {
    "cap1": 218.0,
    "cap2": 153.0,
}

#: 允许误差（实测算法误差 ≤2px，这里留 1px 余量）。
TRUTH_TOLERANCE_PX = 3.0


def _read(capture: str, name: str) -> bytes:
    return (FIXTURES / capture / name).read_bytes()


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@contextlib.contextmanager
def _pillow_missing():
    """模拟「生产镜像没装 pillow」：让 `import PIL` 失败，退出后完整还原。

    用 `sys.modules["PIL"] = None` 而不是 monkeypatch `_load_pillow`，这样验证的是
    真实的 `from PIL import Image` 失败路径（含子模块）。
    """
    saved = {name: module for name, module in sys.modules.items() if name == "PIL" or name.startswith("PIL.")}
    for name in saved:
        del sys.modules[name]
    sys.modules["PIL"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.modules.pop("PIL", None)
        for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
            del sys.modules[name]
        sys.modules.update(saved)


class PuzzleSolverTruthTests(unittest.TestCase):
    """真实素材上的位移精度（核心回归）。"""

    def test_cap1_matches_hand_checked_truth(self) -> None:
        solution = locate_gap(_read("cap1", "back.png"), _read("cap1", "shadow.png"))
        self.assertAlmostEqual(solution.dx_display, CAPTCHA_TRUTH["cap1"], delta=TRUTH_TOLERANCE_PX)
        # back.png 实测 296×200，显示层 300×200 → scale ≈ 1.0135
        self.assertEqual(solution.image_width, 296)
        self.assertEqual(solution.image_height, 200)
        self.assertAlmostEqual(solution.scale, 300.0 / 296.0, places=6)
        # 拼图块 bbox（shadow.png 的 alpha>150 区域）
        self.assertEqual(solution.piece_x0, 2)
        self.assertEqual(solution.piece_width, 49)
        self.assertEqual(solution.piece_height, 50)
        self.assertGreaterEqual(solution.piece_y0, 0)
        self.assertLessEqual(solution.piece_y0 + solution.piece_height, 200)
        self.assertGreater(solution.score, 0.0)

    def test_cap2_matches_hand_checked_truth(self) -> None:
        solution = locate_gap(_read("cap2", "back.png"), _read("cap2", "shadow.png"))
        self.assertAlmostEqual(solution.dx_display, CAPTCHA_TRUTH["cap2"], delta=TRUTH_TOLERANCE_PX)
        self.assertEqual(solution.image_width, 296)
        self.assertEqual(solution.piece_x0, 2)
        self.assertEqual(solution.piece_width, 49)

    def test_file_coordinates_match_truth_divided_by_scale(self) -> None:
        for capture, truth in CAPTCHA_TRUTH.items():
            solution = locate_gap(_read(capture, "back.png"), _read(capture, "shadow.png"))
            expected_file_dx = truth * solution.image_width / DEFAULT_DISPLAY_WIDTH
            self.assertAlmostEqual(solution.dx_file, expected_file_dx, delta=TRUTH_TOLERANCE_PX)

    def test_two_captures_are_distinct(self) -> None:
        first = locate_gap(_read("cap1", "back.png"), _read("cap1", "shadow.png"))
        second = locate_gap(_read("cap2", "back.png"), _read("cap2", "shadow.png"))
        self.assertNotAlmostEqual(first.dx_display, second.dx_display, delta=10.0)

    def test_result_is_deterministic(self) -> None:
        back = _read("cap1", "back.png")
        shadow = _read("cap1", "shadow.png")
        first = locate_gap(back, shadow)
        second = locate_gap(back, shadow)
        self.assertEqual(first.to_dict(), second.to_dict())


class PuzzleSolverScalingTests(unittest.TestCase):
    """显示层宽度决定最终位移（CSS 缩放不能写死）。"""

    def test_display_width_scales_dx_linearly(self) -> None:
        back = _read("cap2", "back.png")
        shadow = _read("cap2", "shadow.png")
        wide = locate_gap(back, shadow, display_width=300.0)
        narrow = locate_gap(back, shadow, display_width=150.0)
        self.assertAlmostEqual(narrow.dx_file, wide.dx_file, places=6)
        self.assertAlmostEqual(narrow.dx_display, wide.dx_display / 2.0, places=6)

    def test_display_width_defaults_to_300(self) -> None:
        solution = locate_gap(_read("cap1", "back.png"), _read("cap1", "shadow.png"))
        self.assertEqual(solution.display_width, DEFAULT_DISPLAY_WIDTH)

    def test_locate_piece_dx_matches_locate_gap(self) -> None:
        back = _read("cap2", "back.png")
        shadow = _read("cap2", "shadow.png")
        self.assertAlmostEqual(
            locate_piece_dx(back, shadow),
            locate_gap(back, shadow).dx_display,
            places=6,
        )


class PuzzleSolverErrorTests(unittest.TestCase):
    """坏素材必须抛 PuzzleSolveError（上层据此判定「本轮失败」，不能崩）。"""

    def test_empty_back_png_raises(self) -> None:
        with self.assertRaises(PuzzleSolveError):
            locate_gap(b"", _read("cap1", "shadow.png"))

    def test_empty_shadow_png_raises(self) -> None:
        with self.assertRaises(PuzzleSolveError):
            locate_gap(_read("cap1", "back.png"), b"")

    def test_garbage_bytes_raise(self) -> None:
        with self.assertRaises(PuzzleSolveError):
            locate_gap(b"not-a-png", b"not-a-png")

    def test_transparent_shadow_raises(self) -> None:
        transparent = _png_bytes((52, 200), (0, 0, 0, 0))
        with self.assertRaises(PuzzleSolveError):
            locate_gap(_read("cap1", "back.png"), transparent)

    def test_oversized_piece_raises(self) -> None:
        oversized = _png_bytes((400, 200), (255, 255, 255, 255))
        with self.assertRaises(PuzzleSolveError):
            locate_gap(_read("cap1", "back.png"), oversized)


class PillowMissingTests(unittest.TestCase):
    """pillow 缺失时：导入 / 启动不受影响，只有真正求解才报 PuzzleSolverUnavailable。

    对应线上事故：`puzzle_solver` 在模块顶层 `from PIL import Image`，而生产镜像
    `ghcr.io/sofs2005/q2a:dev` 没装 pillow，导致整个网关起不来。
    """

    def test_module_import_does_not_require_pillow(self) -> None:
        saved = sys.modules.pop("backend.services.puzzle_solver", None)
        try:
            with _pillow_missing():
                # 1) 模块在 pillow 缺失时照样能导入
                module = importlib.import_module("backend.services.puzzle_solver")
                # 2) 惰性导入确实返回 None，且探针如实报告
                self.assertIsNone(module._load_pillow())
                self.assertFalse(module.pillow_available())
                # 3) 新异常是 PuzzleSolveError 的子类（既有 except 分支依然安全）
                self.assertTrue(issubclass(module.PuzzleSolverUnavailable, module.PuzzleSolveError))
        finally:
            sys.modules.pop("backend.services.puzzle_solver", None)
            if saved is not None:
                sys.modules["backend.services.puzzle_solver"] = saved

    def test_locate_gap_raises_unavailable_without_pillow(self) -> None:
        back = _read("cap1", "back.png")
        shadow = _read("cap1", "shadow.png")
        with _pillow_missing():
            with self.assertRaises(PuzzleSolverUnavailable) as ctx:
                locate_gap(back, shadow)
            # 坏素材也要先报「不可用」，不能被「素材为空」之类的噪声掩盖
            with self.assertRaises(PuzzleSolverUnavailable):
                locate_gap(b"", b"")
            with self.assertRaises(PuzzleSolverUnavailable):
                locate_piece_dx(back, shadow)
        message = str(ctx.exception)
        self.assertIn("pillow not installed", message)
        self.assertIn("browser upload fallback disabled", message)
        self.assertEqual(message, PILLOW_MISSING_MESSAGE)
        # 标记位便于调用方用 getattr 判断
        self.assertTrue(getattr(ctx.exception, "unavailable", False))

    def test_decode_helper_raises_unavailable_without_pillow(self) -> None:
        with _pillow_missing():
            with self.assertRaises(PuzzleSolverUnavailable):
                puzzle_solver._load_rgba(b"anything", label="back")

    def test_pillow_restored_after_simulation(self) -> None:
        # 其他用例仍要能正常跑：模拟结束后 PIL 必须真的回来了
        with _pillow_missing():
            self.assertIsNone(puzzle_solver._load_pillow())
        self.assertTrue(puzzle_solver.pillow_available())
        solution = locate_gap(_read("cap1", "back.png"), _read("cap1", "shadow.png"))
        self.assertAlmostEqual(solution.dx_display, CAPTCHA_TRUTH["cap1"], delta=TRUTH_TOLERANCE_PX)


if __name__ == "__main__":
    unittest.main()
