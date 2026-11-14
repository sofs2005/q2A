"""视频引擎 / 规格字段（extra.meta.model + resolution + duration）相关单测。

覆盖三件事：
1. chat 级 model 的死模型修复（DEFAULT_VIDEO_MODEL / VIDEO_MODEL_MAP）
2. resolution / duration 的宽松校验（非法值回落默认 + WARNING，绝不抛错）
3. `extra.wanx.video_spec` 自检日志（字段名是否生效的唯一验证手段）
"""
import sys
import unittest

#: 历史测试在 import 期往 `sys.modules` 里塞假包且从不还原（见
#: tests/test_inline_attachment_limits.py 的说明），全量跑时会污染真实导入。
_POISONED_ROOTS: tuple = ("fastapi", "pydantic", "pydantic_settings")


def _is_fake_module(module: object) -> bool:
    """真包一定有 __file__（普通模块）或 __path__（namespace package）。"""
    return (
        module is not None
        and getattr(module, "__file__", None) is None
        and not hasattr(module, "__path__")
    )


def _import_videos_module():
    """导入真实 backend.api.videos，不被其它测试注入的假 fastapi/pydantic 影响。"""
    fake_names = [
        name
        for name, module in list(sys.modules.items())
        if name.split(".", 1)[0] in _POISONED_ROOTS and _is_fake_module(module)
    ]
    if not fake_names:  # 没有污染：直接走正常导入
        from backend.api import videos as module

        return module

    saved = {name: sys.modules[name] for name in fake_names}
    for name in fake_names:
        del sys.modules[name]
    try:
        from backend.api import videos as module

        return module
    finally:
        # 还原现场：清掉本次导入的真实模块（含子模块），再把假模块放回去。
        for name in [
            n for n in list(sys.modules) if n.split(".", 1)[0] in _POISONED_ROOTS
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


videos = _import_videos_module()


# 生产实测抓到的上游回显结构（happyhorse-1.1 → happyhorse-1.1-t2v）
CAPTURED_VIDEO_SPEC_BODY = {
    "success": True,
    "data": {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "wanx": {
                        "task_id": "3f2a9c1e-0000-4aaa-8bbb-ccccdddd0000",
                        "video_spec": {
                            "model": "happyhorse-1.1-t2v",
                            "resolution": "1080P",
                            "duration": 10,
                            "watermark": False,
                            "size": "16:9",
                            "ratio": "16:9",
                        },
                    }
                },
            }
        ]
    },
}

OLD_FORMAT_BODY = {
    "data": {
        "messages": [
            {"role": "assistant", "extra": {"wanx": {"task_id": "old-task-id"}}}
        ]
    }
}


class VideoModelRoutingTests(unittest.TestCase):
    """chat 级 model 只负责触发 t2v 通道，死模型必须全部改指 qwen3.8-max。"""

    def test_default_model_is_qwen38_max(self) -> None:
        self.assertEqual(videos.DEFAULT_VIDEO_MODEL, "qwen3.8-max")

    def test_all_aliases_resolve_to_qwen38_max(self) -> None:
        for alias in (
            "sora",
            "qwen-video",
            "qwen3.6-plus",
            "qwen3.6-plus-video",
            "qwen3.8-max",
        ):
            with self.subTest(alias=alias):
                self.assertEqual(videos._resolve_video_model(alias), "qwen3.8-max")

    def test_missing_or_unknown_model_resolves_to_default(self) -> None:
        for requested in (None, "", "unknown-model"):
            with self.subTest(requested=requested):
                self.assertEqual(
                    videos._resolve_video_model(requested), "qwen3.8-max"
                )

    def test_no_dead_model_left_in_map(self) -> None:
        """qwen3.6-plus 上游已删除（Not_Found），映射表里不能再指向它。"""
        for alias, target in videos.VIDEO_MODEL_MAP.items():
            with self.subTest(alias=alias):
                self.assertEqual(target, "qwen3.8-max")


class VideoSpecNormalizationTests(unittest.TestCase):
    """resolution / duration 校验：非法值回落默认，只打 WARNING，不抛异常。"""

    def test_resolution_valid_values_pass_through(self) -> None:
        self.assertEqual(videos._normalize_resolution("720P"), "720P")
        self.assertEqual(videos._normalize_resolution("1080p"), "1080P")

    def test_resolution_invalid_falls_back_with_warning(self) -> None:
        with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
            self.assertEqual(videos._normalize_resolution("4K"), "720P")
        self.assertIn("invalid resolution", "\n".join(captured.output))

    def test_resolution_missing_falls_back_silently(self) -> None:
        self.assertEqual(videos._normalize_resolution(None), "720P")
        self.assertEqual(videos._normalize_resolution("   "), "720P")

    def test_duration_valid_values_pass_through(self) -> None:
        self.assertEqual(videos._normalize_duration(5), 5)
        self.assertEqual(videos._normalize_duration(10), 10)
        self.assertEqual(videos._normalize_duration("15"), 15)

    def test_duration_invalid_falls_back_with_warning(self) -> None:
        for bad in (7, 0, -3):
            with self.subTest(value=bad):
                with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
                    self.assertEqual(videos._normalize_duration(bad), 5)
                self.assertIn("invalid duration", "\n".join(captured.output))

    def test_duration_unparsable_does_not_raise(self) -> None:
        with self.assertLogs("qwen2api.videos", level="WARNING"):
            self.assertEqual(videos._normalize_duration("abc"), 5)
        self.assertEqual(videos._normalize_duration(None), 5)
        self.assertEqual(videos._normalize_duration(""), 5)


class VideoSpecLoggingTests(unittest.TestCase):
    """video_spec 自检日志：有字段时打印完整规格，缺失时只打 missing 日志。"""

    def test_extract_video_spec_from_captured_body(self) -> None:
        spec = videos._extract_video_spec(CAPTURED_VIDEO_SPEC_BODY)
        self.assertIsNotNone(spec)
        self.assertEqual(spec["model"], "happyhorse-1.1-t2v")
        self.assertEqual(spec["resolution"], "1080P")
        self.assertEqual(spec["duration"], 10)
        self.assertEqual(spec["ratio"], "16:9")
        self.assertEqual(spec["watermark"], False)

    def test_extract_video_spec_returns_none_for_old_format(self) -> None:
        self.assertIsNone(videos._extract_video_spec(OLD_FORMAT_BODY))
        self.assertIsNone(videos._extract_video_spec(None))
        self.assertIsNone(videos._extract_video_spec({}))

    def test_log_video_spec_matches_captured_fields(self) -> None:
        with self.assertLogs("qwen2api.videos", level="INFO") as captured:
            spec = videos._log_video_spec(
                CAPTURED_VIDEO_SPEC_BODY,
                expected_resolution="1080P",
                expected_duration=10,
            )

        self.assertIsNotNone(spec)
        text = "\n".join(captured.output)
        self.assertIn("video_spec", text)
        self.assertIn("model=happyhorse-1.1-t2v", text)
        self.assertIn("resolution=1080P", text)
        self.assertIn("duration=10", text)
        self.assertIn("ratio=16:9", text)
        self.assertIn("watermark=False", text)

    def test_log_video_spec_warns_on_mismatch(self) -> None:
        with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
            videos._log_video_spec(
                CAPTURED_VIDEO_SPEC_BODY,
                expected_resolution="720P",
                expected_duration=5,
            )
        text = "\n".join(captured.output)
        self.assertIn("resolution mismatch", text)
        self.assertIn("duration mismatch", text)

    def test_log_video_spec_missing_does_not_raise(self) -> None:
        with self.assertLogs("qwen2api.videos", level="INFO") as captured:
            result = videos._log_video_spec(OLD_FORMAT_BODY)
        self.assertIsNone(result)
        self.assertIn(
            "video_spec missing (old task format)", "\n".join(captured.output)
        )


if __name__ == "__main__":
    unittest.main()
