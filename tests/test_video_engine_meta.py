"""视频引擎 / 规格字段（extra.meta.model + resolution + duration）相关单测。

覆盖四件事：
1. chat 级 model 的死模型修复（DEFAULT_VIDEO_MODEL / VIDEO_MODEL_MAP）
2. resolution / duration 的宽松校验（非法值回落默认 + WARNING，绝不抛错）
3. `extra.wanx.video_spec` 自检日志（字段名是否生效的唯一验证手段）
4. 上游「仅会员可用」拒绝的识别与可读报错（400 + 明确 detail）
"""
import json
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

#: 生产实测：免费号请求 1080P / 10s / 15s 时上游的拒绝响应（无 task_id）。
MEMBER_ONLY_BODY = {
    "success": True,
    "data": {
        "messages": [
            {
                "role": "assistant",
                "content": "请求非法：当前视频规格仅会员可用，请开通或升级 Qwen 会员后重试。",
                "extra": {"update_member": True, "quota_limit": True},
            }
        ]
    }
}

#: 分段 content 形态（content: [{"text": "..."}]）。
MEMBER_ONLY_SEGMENTED_BODY = {
    "data": {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "当前视频规格仅会员可用"}],
                "extra": {"update_member": True, "quota_limit": True},
            }
        ]
    }
}

#: 额度耗尽但不是「规格问题」：不能误判成会员限制。
QUOTA_EXHAUSTED_BODY = {
    "data": {
        "messages": [
            {
                "role": "assistant",
                "content": "目前服务访问量较大，请稍后再试。",
                "extra": {"quota_limit": True},
            }
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


class _FakeAccount:
    """_run_generation 只用到 email / token 两个属性。"""

    def __init__(self, email: str = "demo@example.com") -> None:
        self.email = email
        self.token = "fake-token"


class _FakeAccountPool:
    """记录 release / mark_video_limited 调用，用于断言账号是否被误标限流。"""

    def __init__(self) -> None:
        self.released: list = []
        self.video_limited: list = []

    def release(self, acc) -> None:
        self.released.append(acc)

    def mark_video_limited(self, acc, error_message: str = "") -> None:
        self.video_limited.append((acc, error_message))


class _FakeClient:
    """返回固定 completions body 的假 Qwen 客户端。"""

    def __init__(self, body: str) -> None:
        self._body = body
        self.acc = _FakeAccount()
        self.account_pool = _FakeAccountPool()
        self.deleted_chats: list = []

    async def complete_once_with_retry(
        self, model, prompt, has_custom_tools=False, chat_type="", media_options=None
    ) -> dict:
        return {"acc": self.acc, "chat_id": "fake-chat-id", "body": self._body}

    async def delete_chat(self, token, chat_id, account=None) -> bool:
        self.deleted_chats.append(chat_id)
        return True


class VideoMemberOnlyDetectionTests(unittest.TestCase):
    """`_find_member_only_rejection`：quota_limit 且 content 含会员字样才算命中。"""

    def test_detects_captured_member_only_body(self) -> None:
        text = videos._find_member_only_rejection(MEMBER_ONLY_BODY)
        self.assertIsNotNone(text)
        self.assertIn("会员", text)

    def test_detects_segmented_content(self) -> None:
        text = videos._find_member_only_rejection(MEMBER_ONLY_SEGMENTED_BODY)
        self.assertIsNotNone(text)
        self.assertIn("会员", text)

    def test_quota_exhausted_without_member_wording_is_not_member_only(self) -> None:
        self.assertIsNone(videos._find_member_only_rejection(QUOTA_EXHAUSTED_BODY))

    def test_member_wording_without_quota_flag_is_not_member_only(self) -> None:
        body = {
            "data": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "当前视频规格仅会员可用",
                        "extra": {"update_member": True},
                    }
                ]
            }
        }
        self.assertIsNone(videos._find_member_only_rejection(body))

    def test_benign_bodies_are_not_member_only(self) -> None:
        for payload in (None, {}, OLD_FORMAT_BODY, CAPTURED_VIDEO_SPEC_BODY, []):
            with self.subTest(payload=payload):
                self.assertIsNone(videos._find_member_only_rejection(payload))


class VideoGenerationErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    """无 task_id 的两种成因要分开报：会员规格 → 400，其它 → 500。"""

    async def test_member_only_spec_raises_400_with_readable_detail(self) -> None:
        client = _FakeClient(json.dumps(MEMBER_ONLY_BODY, ensure_ascii=False))

        with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
            with self.assertRaises(videos.HTTPException) as ctx:
                await videos._run_generation(
                    client, "qwen3.8-max", "一只猫", 1, 10, "16:9", {},
                    resolution="1080P",
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, videos.MEMBER_ONLY_DETAIL)
        self.assertIn("Qwen 会员", ctx.exception.detail)
        self.assertIn("720P", ctx.exception.detail)

        text = "\n".join(captured.output)
        self.assertIn("member_only spec rejected", text)
        self.assertIn("resolution=1080P", text)
        self.assertIn("duration=10", text)
        self.assertIn("account=demo@example.com", text)

        # 规格问题不是账号额度问题：账号不能被标记为视频不可用
        self.assertEqual(client.account_pool.video_limited, [])
        # 会话仍要释放
        self.assertEqual(client.account_pool.released, [client.acc])

    async def test_other_missing_task_id_still_raises_500(self) -> None:
        client = _FakeClient(json.dumps(QUOTA_EXHAUSTED_BODY, ensure_ascii=False))

        with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
            with self.assertRaises(videos.HTTPException) as ctx:
                await videos._run_generation(
                    client, "qwen3.8-max", "一只猫", 1, 5, "16:9", {},
                    resolution="720P",
                )

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail, "Video generation returned no task_id"
        )
        self.assertIn("no task_id", "\n".join(captured.output))

    async def test_other_failures_still_500(self) -> None:
        client = _FakeClient("not-json-at-all")

        with self.assertLogs("qwen2api.videos", level="WARNING"):
            with self.assertRaises(videos.HTTPException) as ctx:
                await videos._run_generation(
                    client, "qwen3.8-max", "一只猫", 1, 5, "16:9", {},
                    resolution="720P",
                )

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail, "Video generation returned no task_id"
        )


if __name__ == "__main__":
    unittest.main()
