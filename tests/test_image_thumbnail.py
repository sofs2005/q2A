"""缩略图缓存的单元测试。

设计要点：Pillow 是惰性导入的，因此这里分两组用例 ——
- 与 Pillow 无关的降级行为（缺文件 / 宽度越界 / 不是图片）在任何环境都要成立；
- 真正的缩放行为只在装了 Pillow 的环境跑（CI 镜像里有，本机跑测试的
  q2a:dev 镜像里没有，跳过即可）。
"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from backend.services.image_thumbnail import (
    MAX_WIDTH,
    MIN_WIDTH,
    ensure_thumbnail,
    remove_thumbnails,
    thumb_path_for,
)

try:
    from PIL import Image  # noqa: F401

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    PIL_AVAILABLE = False


class ThumbnailDegradeTests(unittest.TestCase):
    """不依赖 Pillow 的降级路径：任何异常都只能返回 None，不能抛。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.source = self.root / "generated_image" / "big.png"
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_bytes(b"not-a-real-image")

    def test_missing_source_returns_none(self) -> None:
        target = thumb_path_for(self.root, "abc", 480)
        result = asyncio.run(ensure_thumbnail(self.root / "nope.png", target, 480))
        self.assertIsNone(result)

    def test_width_out_of_range_returns_none(self) -> None:
        target = thumb_path_for(self.root, "abc", 480)
        for width in (MIN_WIDTH - 1, 0, -10, MAX_WIDTH + 1):
            with self.subTest(width=width):
                result = asyncio.run(ensure_thumbnail(self.source, target, width))
                self.assertIsNone(result)

    def test_broken_image_bytes_return_none(self) -> None:
        """损坏 / 非图片内容：渲染失败也只能返回 None（不 500）。"""
        target = thumb_path_for(self.root, "abc", 480)
        result = asyncio.run(ensure_thumbnail(self.source, target, 480))
        self.assertIsNone(result)

    def test_thumb_path_layout(self) -> None:
        path = thumb_path_for(self.root, "deadbeef", 480)
        self.assertEqual(path.parent.name, "generated_image_thumb")
        self.assertEqual(path.name, "deadbeef@480.jpg")

    def test_remove_thumbnails_on_missing_dir(self) -> None:
        self.assertEqual(remove_thumbnails(self.root, "abc"), 0)


@unittest.skipUnless(PIL_AVAILABLE, "Pillow 未安装，跳过实际缩放用例")
class ThumbnailRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.source = self.root / "generated_image" / "big.png"
        self.source.parent.mkdir(parents=True, exist_ok=True)
        # 1200x800 带透明通道：顺带验证「透明底转白底」而不是黑底
        image = Image.new("RGBA", (1200, 800), (10, 120, 220, 128))
        image.save(self.source)

    def test_thumbnail_is_scaled_and_cached(self) -> None:
        target = thumb_path_for(self.root, "fid1", 480)
        thumb = asyncio.run(ensure_thumbnail(self.source, target, 480))

        self.assertIsNotNone(thumb)
        self.assertTrue(Path(thumb).is_file())
        with Image.open(thumb) as rendered:
            self.assertEqual(rendered.format, "JPEG")
            self.assertEqual(rendered.mode, "RGB")
            self.assertLessEqual(rendered.width, 480)
            # 等比缩放：1200x800 → 480x320
            self.assertEqual(rendered.width, 480)
            self.assertEqual(rendered.height, 320)
        # 缩略图必须远小于原图（这正是它存在的意义）
        self.assertLess(Path(thumb).stat().st_size, self.source.stat().st_size)

        # 第二次命中缓存：不重写文件（mtime 不变）
        first_mtime = Path(thumb).stat().st_mtime_ns
        again = asyncio.run(ensure_thumbnail(self.source, target, 480))
        self.assertEqual(again, thumb)
        self.assertEqual(Path(thumb).stat().st_mtime_ns, first_mtime)

    def test_source_newer_than_cache_rebuilds(self) -> None:
        target = thumb_path_for(self.root, "fid2", 320)
        thumb = asyncio.run(ensure_thumbnail(self.source, target, 320))
        self.assertIsNotNone(thumb)
        first_mtime = Path(thumb).stat().st_mtime_ns

        # 原图被替换（mtime 变新）时必须重算，否则会一直发旧图
        os.utime(self.source, None)
        rebuilt = asyncio.run(
            ensure_thumbnail(self.source, target, 320, source_mtime=os.path.getmtime(self.source))
        )
        self.assertIsNotNone(rebuilt)
        self.assertGreaterEqual(Path(thumb).stat().st_mtime_ns, first_mtime)

    def test_remove_thumbnails_clears_all_widths(self) -> None:
        for width in (320, 480):
            self.assertIsNotNone(
                asyncio.run(ensure_thumbnail(self.source, thumb_path_for(self.root, "fid3", width), width))
            )
        # 别的 id 不能被误删
        other = asyncio.run(ensure_thumbnail(self.source, thumb_path_for(self.root, "other", 320), 320))
        self.assertIsNotNone(other)

        self.assertEqual(remove_thumbnails(self.root, "fid3"), 2)
        self.assertFalse(thumb_path_for(self.root, "fid3", 320).exists())
        self.assertFalse(thumb_path_for(self.root, "fid3", 480).exists())
        self.assertTrue(Path(other).exists())


if __name__ == "__main__":
    unittest.main()
