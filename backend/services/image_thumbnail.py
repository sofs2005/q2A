"""生成物缩略图：把大图按宽度等比缩小并缓存为 JPEG。

为什么单独一个模块：媒体库一页要展示几十张生成图，而原图常见 5~10MB
（PNG）。直接铺原图会把首屏流量拉到几百 MB。这里按需生成缩略图并落盘缓存，
命中缓存时只读一个小 JPEG。

Pillow 是**惰性导入**的：镜像里缺 Pillow 时（历史上有过这种镜像）本模块
一律返回 `None`，调用方退回原图，功能不受影响。
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

log = logging.getLogger("qwen2api.image_thumbnail")

# 缓存目录名（位于 file_store.root_dir 下）
THUMB_DIR_NAME = "generated_image_thumb"

# 宽度护栏：太小没意义，太大不如直接用原图
MIN_WIDTH = 32
MAX_WIDTH = 2048

# 缩略图 JPEG 质量
JPEG_QUALITY = 82


def thumb_path_for(root_dir: str | Path, file_id: str, width: int) -> Path:
    """缩略图缓存路径：`<root>/generated_image_thumb/<file_id>@<width>.jpg`。"""
    return Path(root_dir) / THUMB_DIR_NAME / f"{file_id}@{width}.jpg"


def _render(source: Path, target: Path, width: int) -> bool:
    """同步渲染缩略图（在调用方丢进线程池执行）。Pillow 缺失或图片损坏返回 False。"""
    try:
        from PIL import Image, ImageOps
    except Exception:  # 缺 Pillow / 装坏了都不该让接口 500
        return False

    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            # 透明图（RGBA/P/LA）先铺白底，否则转 JPEG 会出现黑底
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                rgba = image.convert("RGBA")
                background.paste(rgba, mask=rgba.split()[-1])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # 等比缩到指定宽度；高度不设上限（极端长图也不会被裁）
            image.thumbnail((width, width * 16))
            target.parent.mkdir(parents=True, exist_ok=True)
            # 先写临时文件再原子替换：并发请求不会读到写了一半的 JPEG
            tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                image.save(tmp, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                os.replace(tmp, target)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        return True
    except Exception as exc:
        log.debug("[Thumb] render failed source=%s error=%s", source, exc)
        return False


async def ensure_thumbnail(
    source: str | Path,
    target: str | Path,
    width: int,
    *,
    source_mtime: float | None = None,
) -> Path | None:
    """确保缩略图存在并返回其路径；无法生成时返回 None（调用方退回原图）。

    - 宽度越界（< MIN_WIDTH 或 > MAX_WIDTH）直接放弃，避免被拿来当图片缩放服务
    - 缓存文件比原图新（或 mtime 未知）即视为命中，不重复解码
    """
    if width < MIN_WIDTH or width > MAX_WIDTH:
        return None

    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        return None

    if target_path.is_file() and target_path.stat().st_size > 0:
        if source_mtime is None:
            return target_path
        try:
            if target_path.stat().st_mtime >= float(source_mtime):
                return target_path
        except OSError:
            pass

    rendered = await asyncio.to_thread(_render, source_path, target_path, width)
    return target_path if rendered and target_path.is_file() else None


def remove_thumbnails(root_dir: str | Path, file_id: str) -> int:
    """删除某个生成物的全部缩略图缓存，返回删除个数。

    生成物被删除后缩略图必须一起清掉，否则缓存目录会留下永远没人引用的文件。
    """
    directory = Path(root_dir) / THUMB_DIR_NAME
    if not directory.is_dir():
        return 0

    removed = 0
    for candidate in directory.glob(f"{file_id}@*.jpg"):
        try:
            candidate.unlink()
            removed += 1
        except OSError:
            continue
    return removed
