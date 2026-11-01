"""拼图缺口定位单测：用两组真实 `shadow.png` / `back.png` 核对位移。

fixture 来自实测现场（`generated/q2a-mm/cap/1`、`cap/2`，已复制到
`tests/fixtures/captcha/`）：

- `cap1`：人工核对过的真值 ≈ **218px**（显示坐标，`#aliyunCaptcha-img` 300px 宽）
- `cap2`：人工核对过的真值 ≈ **153px**

算法说明见 `backend/services/puzzle_solver.py` 与 findings 文档 §8.3。
"""

import io
import unittest
from pathlib import Path

from PIL import Image

from backend.services.puzzle_solver import (
    DEFAULT_DISPLAY_WIDTH,
    PuzzleSolveError,
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


if __name__ == "__main__":
    unittest.main()
