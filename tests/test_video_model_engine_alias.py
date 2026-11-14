"""`model` 直接写引擎名（VIDEO_ENGINE_IDS）的路由与 `extra.meta` 下发单测。

覆盖五件事：
1. `model` 填引擎名（happyhorse-1.1 / wan-2.7）→ chat 级 model 仍是
   qwen3.8-max，`messages[0].extra.meta.model` 等于请求的引擎名
2. `model` 填 chat 级别名（qwen-video / sora）→ `extra.meta.model` 用配置默认
   （VIDEO_GENERATION_META_MODEL）
3. 配置默认为空字符串 + 别名 → `extra.meta` 里没有 `model` 键
4. 引擎别名与配置默认冲突时**别名优先**
5. media_options 的 `engine` 键只在显式指定引擎时出现

注意：抓包链路是 videos.py（解析 / 组装 media_options） →
payload_builder.build_chat_payload（真正拼 extra.meta），因此这里直接串起来断言。
"""
import sys
import unittest
from unittest.mock import patch

from backend.core.config import settings
from backend.upstream.payload_builder import build_chat_payload

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


def _route_and_build(
    requested,
    meta_model: str = "happyhorse-1.1",
    duration: int = 5,
    resolution: str = "720P",
    ratio: str = "16:9",
) -> tuple:
    """复刻接口的组装链路：解析 model → 组装 media_options → 拼 payload。

    Args:
        requested: 请求体的 `model` 原始值（可能为 None / 别名 / 引擎名）。
        meta_model: 模拟的配置项 VIDEO_GENERATION_META_MODEL 取值。
        duration: 请求时长。
        resolution: 请求分辨率。
        ratio: 请求宽高比。

    Returns:
        ``(chat_model, engine_override, media_options, payload)``。
    """
    chat_model, engine_override = videos._resolve_video_model_and_engine(requested)
    media_options = videos._build_video_media_options(
        ratio, duration, resolution, engine_override,
    )
    with patch.object(settings, "VIDEO_GENERATION_META_MODEL", meta_model):
        payload = build_chat_payload(
            "chat-1",
            chat_model,
            "一只猫在跑",
            chat_type="t2v",
            media_options=media_options,
        )
    return chat_model, engine_override, media_options, payload


def _meta(payload: dict) -> dict:
    """取 `messages[0].extra.meta`。"""
    return payload["messages"][0]["extra"]["meta"]


class VideoEngineAliasRoutingTests(unittest.TestCase):
    """引擎名直接写在 `model` 里：chat 级 model 仍走 qwen3.8-max，引擎按请求下发。"""

    def test_engine_ids_are_the_documented_three(self) -> None:
        self.assertEqual(
            videos.VIDEO_ENGINE_IDS,
            {"happyhorse-1.1", "happyhorse-1.0", "wan-2.7"},
        )

    def test_happyhorse_11_alias_sets_meta_model(self) -> None:
        chat_model, engine, options, payload = _route_and_build("happyhorse-1.1")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertEqual(engine, "happyhorse-1.1")
        self.assertEqual(options["engine"], "happyhorse-1.1")
        self.assertEqual(_meta(payload)["model"], "happyhorse-1.1")
        # chat 级 model 与引擎是两件事：payload 顶层 model 仍是 qwen3.8-max
        self.assertEqual(payload["model"], "qwen3.8-max")

    def test_wan_27_alias_sets_meta_model(self) -> None:
        chat_model, engine, _, payload = _route_and_build("wan-2.7")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertEqual(engine, "wan-2.7")
        self.assertEqual(_meta(payload)["model"], "wan-2.7")

    def test_happyhorse_10_alias_sets_meta_model(self) -> None:
        _, _, _, payload = _route_and_build("happyhorse-1.0")

        self.assertEqual(_meta(payload)["model"], "happyhorse-1.0")

    def test_engine_alias_is_stripped(self) -> None:
        chat_model, engine, _, payload = _route_and_build("  wan-2.7  ")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertEqual(engine, "wan-2.7")
        self.assertEqual(_meta(payload)["model"], "wan-2.7")

    def test_engine_alias_overrides_config_default(self) -> None:
        """别名 + 配置默认冲突时：别名优先。"""
        _, engine, _, payload = _route_and_build("wan-2.7", meta_model="happyhorse-1.0")

        self.assertEqual(engine, "wan-2.7")
        self.assertEqual(_meta(payload)["model"], "wan-2.7")

    def test_engine_alias_works_when_config_default_empty(self) -> None:
        """配置默认留空时，别名依然把引擎下发出去。"""
        _, _, _, payload = _route_and_build("happyhorse-1.0", meta_model="")

        self.assertEqual(_meta(payload)["model"], "happyhorse-1.0")


class VideoChatModelAliasTests(unittest.TestCase):
    """chat 级别名：引擎用配置默认，media_options 里没有 engine 键。"""

    def test_qwen_video_alias_uses_config_default_engine(self) -> None:
        chat_model, engine, options, payload = _route_and_build("qwen-video")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertIsNone(engine)
        self.assertNotIn("engine", options)
        self.assertEqual(_meta(payload)["model"], "happyhorse-1.1")

    def test_sora_alias_uses_config_default_engine(self) -> None:
        chat_model, engine, options, payload = _route_and_build("sora")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertIsNone(engine)
        self.assertNotIn("engine", options)
        self.assertEqual(_meta(payload)["model"], "happyhorse-1.1")

    def test_empty_config_default_omits_meta_model_key(self) -> None:
        """VIDEO_GENERATION_META_MODEL="" + 别名 → extra.meta 里没有 model 键。"""
        chat_model, _, _, payload = _route_and_build("qwen-video", meta_model="")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertNotIn("model", _meta(payload))
        # 规格字段照发，不受引擎缺省影响
        self.assertEqual(_meta(payload)["resolution"], "720P")
        self.assertEqual(_meta(payload)["duration"], 5)

    def test_unknown_model_falls_back_to_default_engine(self) -> None:
        chat_model, engine, options, payload = _route_and_build("no-such-model")

        self.assertEqual(chat_model, "qwen3.8-max")
        self.assertIsNone(engine)
        self.assertNotIn("engine", options)
        self.assertEqual(_meta(payload)["model"], "happyhorse-1.1")

    def test_missing_model_falls_back_to_default_engine(self) -> None:
        for requested in (None, "", "   "):
            with self.subTest(requested=requested):
                chat_model, engine, _, payload = _route_and_build(requested)

                self.assertEqual(chat_model, "qwen3.8-max")
                self.assertIsNone(engine)
                self.assertEqual(_meta(payload)["model"], "happyhorse-1.1")


class VideoMediaOptionsTests(unittest.TestCase):
    """`_build_video_media_options` / `_resolve_video_model` 的契约。"""

    def test_media_options_without_engine_has_ratio_size_duration_resolution(self) -> None:
        options = videos._build_video_media_options("9:16", 10, "1080P")

        self.assertEqual(
            options,
            {"ratio": "9:16", "size": "9:16", "duration": 10, "resolution": "1080P"},
        )

    def test_media_options_engine_none_or_blank_is_omitted(self) -> None:
        for engine in (None, "", "   "):
            with self.subTest(engine=engine):
                options = videos._build_video_media_options("16:9", 5, "720P", engine)

                self.assertNotIn("engine", options)

    def test_media_options_engine_is_stripped_when_given(self) -> None:
        options = videos._build_video_media_options("16:9", 5, "720P", " wan-2.7 ")

        self.assertEqual(options["engine"], "wan-2.7")

    def test_resolve_video_model_keeps_string_signature(self) -> None:
        """兼容旧调用方：`_resolve_video_model` 仍返回单个字符串。"""
        self.assertEqual(videos._resolve_video_model("qwen-video"), "qwen3.8-max")
        self.assertEqual(videos._resolve_video_model("happyhorse-1.1"), "qwen3.8-max")
        self.assertEqual(videos._resolve_video_model(None), "qwen3.8-max")

    def test_engine_alias_keeps_resolution_and_duration(self) -> None:
        """引擎别名不应影响规格字段。"""
        _, _, _, payload = _route_and_build(
            "happyhorse-1.0", duration=10, resolution="1080P", ratio="9:16",
        )
        meta = _meta(payload)

        self.assertEqual(meta["model"], "happyhorse-1.0")
        self.assertEqual(meta["duration"], 10)
        self.assertEqual(meta["resolution"], "1080P")
        self.assertEqual(meta["aspectRatio"], "9:16")


if __name__ == "__main__":
    unittest.main()
