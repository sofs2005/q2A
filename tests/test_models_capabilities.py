"""`/v1/models` 上报生图 / 生视频能力，以及生图能力别名的解析。

背景：q2a 的生图和生视频**没有独立上游模型** —— 生图是「默认对话模型 + t2i
通道」，生视频是「默认对话模型 + t2v 通道 + 引擎名」。因此这两个能力此前在
`/v1/models` 里完全不可见，OpenAI 兼容客户端拉完列表只显示对话入口。

本文件锁住三件事：

1. `/v1/models` 列表里必须有生图 / 生视频 ID，且带 `capabilities`；
2. `/v1/models/{id}` 对这些 ID 不再 404；
3. 生图别名（`qwen-image` 等，都是 `qwen` 开头）必须**先**被识别成别名，
   否则会被 `_resolve_image_model` 的 `qwen*` 透传分支当成 chat 级模型发给上游。
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from types import SimpleNamespace


def _is_fake_module(module: object) -> bool:
    """判断 `sys.modules` 里的条目是不是别的测试手搓的假模块。

    真包一定有 `__file__`（普通模块）或 `__path__`（namespace package）；
    `types.ModuleType("fastapi")` 塞进去的假模块两者都没有。
    """
    return module is not None and getattr(module, "__file__", None) is None and not hasattr(module, "__path__")


def _load_api_modules() -> SimpleNamespace:
    """导入被测模块，不受其它测试注入的假 `fastapi` / `pydantic` 影响。

    污染源：`tests/test_admin_account_status.py` / `test_admin_chat_clear.py` /
    `test_admin_personalization.py` / `test_admin_verify_account_flow.py` 在
    **import 期**执行 `sys.modules["fastapi"] = types.ModuleType("fastapi")`
    且从不还原。pytest 全量跑时这些假包一直留在 `sys.modules` 里，于是本模块
    `from backend.api.images import ...` 会在**收集阶段**就抛
    `No module named 'fastapi.responses'; 'fastapi' is not a package`
    —— 单跑本文件绿、全量红的根因。

    这里临时摘掉假模块、导入真实包，导入完成后把 `sys.modules` 原样还原：
    既不依赖全局状态，也不改变其它测试看到的环境（同
    `tests/test_inline_attachment_limits.py` 的处理方式）。

    Returns:
        模块级名字都在里面的 namespace，含 `HTTPException` —— 测试运行期
        不能再 `from fastapi import HTTPException`，那时假模块已经被还原回去，
        拿到的会是假类，`assertRaises` 抓不到真异常。
    """
    roots = ("fastapi", "pydantic", "pydantic_settings")
    fake_names = [
        name
        for name, module in list(sys.modules.items())
        if name.split(".", 1)[0] in roots and _is_fake_module(module)
    ]
    saved = {name: sys.modules[name] for name in fake_names}
    for name in fake_names:
        del sys.modules[name]
    try:
        from fastapi import HTTPException

        from backend.api.images import _default_image_model, _resolve_image_model
        from backend.api.models import (
            CAP_CHAT,
            CAP_IMAGE,
            CAP_VIDEO,
            _build_model_list_payload,
            _capability_for,
            _capability_models,
            get_model,
        )
        from backend.api.videos import VIDEO_ENGINE_IDS
        from backend.core.config import MODEL_MAP, settings
    finally:
        sys.modules.update(saved)
    return SimpleNamespace(
        HTTPException=HTTPException,
        _default_image_model=_default_image_model,
        _resolve_image_model=_resolve_image_model,
        CAP_CHAT=CAP_CHAT,
        CAP_IMAGE=CAP_IMAGE,
        CAP_VIDEO=CAP_VIDEO,
        _build_model_list_payload=_build_model_list_payload,
        _capability_for=_capability_for,
        _capability_models=_capability_models,
        get_model=get_model,
        VIDEO_ENGINE_IDS=VIDEO_ENGINE_IDS,
        MODEL_MAP=MODEL_MAP,
        settings=settings,
    )


_api = _load_api_modules()
HTTPException = _api.HTTPException
_default_image_model = _api._default_image_model
_resolve_image_model = _api._resolve_image_model
CAP_CHAT = _api.CAP_CHAT
CAP_IMAGE = _api.CAP_IMAGE
CAP_VIDEO = _api.CAP_VIDEO
_build_model_list_payload = _api._build_model_list_payload
_capability_for = _api._capability_for
_capability_models = _api._capability_models
get_model = _api.get_model
VIDEO_ENGINE_IDS = _api.VIDEO_ENGINE_IDS
MODEL_MAP = _api.MODEL_MAP
settings = _api.settings


def _run(coro):
    return asyncio.run(coro)


def _payload(response) -> dict:
    return json.loads(response.body)


class ModelListTests(unittest.TestCase):
    def test_image_models_are_listed(self) -> None:
        data = _build_model_list_payload()["data"]
        by_id = {row["id"]: row for row in data}
        for model_id in settings.IMAGE_MODEL_IDS:
            self.assertIn(model_id, by_id, f"生图模型 {model_id} 未出现在 /v1/models")
            self.assertEqual(by_id[model_id]["capabilities"], [CAP_IMAGE])

    def test_video_engines_are_listed(self) -> None:
        data = _build_model_list_payload()["data"]
        by_id = {row["id"]: row for row in data}
        for engine in VIDEO_ENGINE_IDS:
            self.assertIn(engine, by_id, f"生视频引擎 {engine} 未出现在 /v1/models")
            self.assertEqual(by_id[engine]["capabilities"], [CAP_VIDEO])

    def test_chat_models_still_listed_with_chat_capability(self) -> None:
        data = _build_model_list_payload()["data"]
        by_id = {row["id"]: row for row in data}
        for alias in MODEL_MAP:
            self.assertIn(alias, by_id)
            self.assertEqual(by_id[alias]["capabilities"], [CAP_CHAT])

    def test_ids_are_unique(self) -> None:
        ids = [row["id"] for row in _build_model_list_payload()["data"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_output_order_is_stable(self) -> None:
        # 客户端会 diff 这个列表，顺序抖动会导致无意义的刷新。
        first = [row["id"] for row in _build_model_list_payload()["data"]]
        second = [row["id"] for row in _build_model_list_payload()["data"]]
        self.assertEqual(first, second)


class CapabilityLookupTests(unittest.TestCase):
    def test_lookup_image_and_video(self) -> None:
        self.assertEqual(_capability_for("qwen-image"), (CAP_IMAGE,))
        self.assertEqual(_capability_for("happyhorse-1.1"), (CAP_VIDEO,))

    def test_lookup_is_case_insensitive(self) -> None:
        self.assertEqual(_capability_for("QWEN-IMAGE"), (CAP_IMAGE,))
        self.assertEqual(_capability_for("HappyHorse-1.1"), (CAP_VIDEO,))

    def test_lookup_unknown_returns_none(self) -> None:
        self.assertIsNone(_capability_for("qwen3.8-max"))
        self.assertIsNone(_capability_for(""))
        self.assertIsNone(_capability_for(None))  # type: ignore[arg-type]

    def test_capability_models_cover_both_kinds(self) -> None:
        caps = dict(_capability_models())
        self.assertIn(CAP_IMAGE, {c for v in caps.values() for c in v})
        self.assertIn(CAP_VIDEO, {c for v in caps.values() for c in v})


class GetModelTests(unittest.TestCase):
    def test_image_alias_returns_200_with_capability(self) -> None:
        body = _payload(_run(get_model("qwen-image")))
        self.assertEqual(body["id"], "qwen-image")
        self.assertEqual(body["capabilities"], [CAP_IMAGE])
        # resolved_model 应指向真实的 chat 级默认模型，而不是别名本身
        self.assertEqual(body["resolved_model"], settings.IMAGE_GENERATION_MODEL)

    def test_video_engine_returns_200_with_capability(self) -> None:
        body = _payload(_run(get_model("happyhorse-1.1")))
        self.assertEqual(body["capabilities"], [CAP_VIDEO])

    def test_unknown_model_still_404(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            _run(get_model("definitely-not-a-model"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_chat_alias_still_resolves(self) -> None:
        body = _payload(_run(get_model("qwen-max")))
        self.assertIn("qwen-max", MODEL_MAP)
        self.assertEqual(body["resolved_model"], MODEL_MAP["qwen-max"])


class ImageAliasResolutionTests(unittest.TestCase):
    """`qwen-image` 必须以**别名**身份落到默认模型，不能被当作上游 chat 模型透传。"""

    def test_alias_maps_to_default_not_itself(self) -> None:
        default = _default_image_model()
        for alias in settings.IMAGE_MODEL_IDS:
            self.assertEqual(_resolve_image_model(alias), default, f"{alias} 未映射回默认模型")

    def test_alias_beats_qwen_passthrough(self) -> None:
        # 关键回归：别名的前缀也是 `qwen`，透传分支会把它原样发给上游。
        self.assertNotEqual(_resolve_image_model("qwen-image"), "qwen-image")
        self.assertNotEqual(_resolve_image_model("qwen-image-3.0-pro"), "qwen-image-3.0-pro")

    def test_real_qwen_model_still_passes_through(self) -> None:
        self.assertEqual(_resolve_image_model("qwen3.8-max"), "qwen3.8-max")

    def test_dall_e_falls_back_to_default(self) -> None:
        self.assertEqual(_resolve_image_model("dall-e-3"), _default_image_model())

    def test_empty_falls_back_to_default(self) -> None:
        self.assertEqual(_resolve_image_model(None), _default_image_model())
        self.assertEqual(_resolve_image_model("  "), _default_image_model())


if __name__ == "__main__":
    unittest.main()
