"""视频引擎 / 规格字段（extra.meta.model + resolution + duration）相关单测。

覆盖四件事：
1. chat 级 model 的死模型修复（DEFAULT_VIDEO_MODEL / VIDEO_MODEL_MAP）
2. resolution / duration 的宽松校验（非法值回落默认 + WARNING，绝不抛错）
3. `extra.wanx.video_spec` 自检日志（字段名是否生效的唯一验证手段）
4. 上游「仅会员可用」拒绝的识别与可读报错：
   - 会员规格（1080P / 10s / 15s）→ 400，且不换号、不标限流
   - 免费规格（720P + 5s）→ 该号当天免费额度用完 → 标限流 + 换号重试，
     全部用完回 429；冷却期内的账号不再被选中
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
        self.received_excludes: list = []

    async def complete_once_with_retry(
        self, model, prompt, has_custom_tools=False, chat_type="", media_options=None,
        exclude=None,
    ) -> dict:
        self.received_excludes.append(set(exclude or ()))
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


# ---------------------------------------------------------------------------
# 免费规格被打回 = 该号当天视频额度用完 → 换号重试（而不是打挂整个请求）
# ---------------------------------------------------------------------------

#: 免费规格成功返回：body 里直接带视频 URL，无需轮询任务状态。
SUCCESS_BODY = {
    "data": {
        "messages": [
            {"role": "assistant", "content": "https://cdn.example.com/v/ok.mp4"}
        ]
    }
}

#: 免费规格成功返回的视频 URL（便于断言）
SUCCESS_VIDEO_URL = "https://cdn.example.com/v/ok.mp4"


class _PoolRecorder:
    """带视频冷却集的账号池替身：记录 release / mark_video_limited 调用。

    `mark_video_limited` 会同步把账号加进冷却集，模拟真实池的行为
    （真实池里该字段是 `video_blocked_until`，冷却期内不再被选中）。
    """

    def __init__(self, blocked: set | None = None) -> None:
        self.blocked = set(blocked or ())
        self.released: list = []
        self.video_limited: list = []

    def release(self, acc) -> None:
        self.released.append(getattr(acc, "email", ""))

    def mark_video_limited(self, acc, error_message: str = "") -> None:
        self.video_limited.append(getattr(acc, "email", ""))
        self.blocked.add(getattr(acc, "email", ""))

    def video_blocked_emails(self) -> set:
        return set(self.blocked)


class _PooledClient:
    """带账号池语义的假客户端：每次调用挑第一个未被 `exclude` 的账号。

    真实执行器（`complete_once_with_retry`）就是这样用 `exclude` 换号的，
    这个替身让「换号是否真的发生」可以被断言。
    """

    def __init__(
        self,
        emails: list,
        bodies: dict | None = None,
        blocked: set | None = None,
    ) -> None:
        self.emails = list(emails)
        self.bodies = dict(bodies or {})
        self.account_pool = _PoolRecorder(blocked)
        self.selected: list = []
        self.received_excludes: list = []
        self.deleted_chats: list = []

    async def complete_once_with_retry(
        self, model, prompt, has_custom_tools=False, chat_type="", media_options=None,
        exclude=None,
    ) -> dict:
        excluded = set(exclude or ())
        self.received_excludes.append(set(excluded))
        for email in self.emails:
            if email in excluded:
                continue
            self.selected.append(email)
            body = self.bodies.get(email, SUCCESS_BODY)
            return {
                "acc": _FakeAccount(email),
                "chat_id": f"chat-{email}",
                "body": json.dumps(body, ensure_ascii=False),
            }
        raise Exception("No available accounts in pool (all busy or rate limited)")

    async def delete_chat(self, token, chat_id, account=None) -> bool:
        self.deleted_chats.append(chat_id)
        return True


class FreeSpecDetectionTests(unittest.TestCase):
    """`_is_free_video_spec`：只有 720P + 5s 才算免费规格。"""

    def test_720p_5s_is_free_spec(self) -> None:
        self.assertTrue(videos._is_free_video_spec("720P", 5))

    def test_member_specs_are_not_free(self) -> None:
        for resolution, duration in (
            ("1080P", 5),
            ("720P", 10),
            ("1080P", 10),
            ("720P", 15),
            ("1080P", 15),
        ):
            with self.subTest(resolution=resolution, duration=duration):
                self.assertFalse(videos._is_free_video_spec(resolution, duration))

    def test_garbage_duration_is_not_free(self) -> None:
        self.assertFalse(videos._is_free_video_spec("720P", "abc"))
        self.assertFalse(videos._is_free_video_spec("720P", None))

    def test_video_blocked_emails_handles_pool_without_method(self) -> None:
        self.assertEqual(videos._video_blocked_emails(None), set())
        self.assertEqual(videos._video_blocked_emails(object()), set())


class FreeSpecQuotaRetryTests(unittest.IsolatedAsyncioTestCase):
    """免费规格被打回 → 换号重试；会员规格被打回 → 直接 400 不换号。"""

    async def test_quota_limited_account_switches_and_succeeds(self) -> None:
        client = _PooledClient(
            ["a@example.com", "b@example.com"],
            bodies={"a@example.com": MEMBER_ONLY_BODY},
        )

        with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
            result = await videos._run_generation(
                client, "qwen3.8-max", "一只猫", 1, 5, "16:9", {},
                resolution="720P",
            )

        # 1) 账号 A 被打回后自动换号，B 成功 → 200 语义
        self.assertEqual(client.selected, ["a@example.com", "b@example.com"])
        self.assertEqual(result["data"][0]["url"], SUCCESS_VIDEO_URL)
        self.assertEqual(result["data"][0]["resolution"], "720P")

        # 2) 账号 A 被标记视频限流（当天免费额度用完），并进入 exclude
        self.assertEqual(client.account_pool.video_limited, ["a@example.com"])
        self.assertIn("a@example.com", client.received_excludes[1])
        self.assertIn("a@example.com", client.account_pool.blocked)

        # 3) 两个号各释放一次，不重复释放
        self.assertEqual(
            client.account_pool.released, ["a@example.com", "b@example.com"]
        )

        text = "\n".join(captured.output)
        self.assertIn("account video quota exhausted", text)
        self.assertIn("account=a@example.com", text)
        self.assertIn("retry with another account", text)

    async def test_all_accounts_quota_limited_returns_429(self) -> None:
        emails = ["a@example.com", "b@example.com", "c@example.com"]
        client = _PooledClient(
            emails, bodies={email: MEMBER_ONLY_BODY for email in emails}
        )

        with self.assertLogs("qwen2api.videos", level="WARNING"):
            with self.assertRaises(videos.HTTPException) as ctx:
                await videos._run_generation(
                    client, "qwen3.8-max", "一只猫", 1, 5, "16:9", {},
                    resolution="720P",
                )

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(
            ctx.exception.detail, videos.VIDEO_QUOTA_EXHAUSTED_DETAIL
        )
        self.assertIn("免费视频额度已用完", ctx.exception.detail)
        self.assertIn("1 个/天", ctx.exception.detail)

        # 每个号只试一次（exclude 累积生效），上限 = VIDEO_QUOTA_RETRY_ATTEMPTS
        self.assertEqual(client.selected, emails)
        self.assertEqual(len(client.selected), videos.VIDEO_QUOTA_RETRY_ATTEMPTS)
        self.assertEqual(len(set(client.selected)), len(client.selected))
        # 每个号都被标记限流且各释放一次
        self.assertEqual(client.account_pool.video_limited, emails)
        self.assertEqual(client.account_pool.released, emails)

    async def test_member_spec_returns_400_without_switching_accounts(self) -> None:
        for resolution, duration in (("1080P", 5), ("720P", 10), ("1080P", 15)):
            with self.subTest(resolution=resolution, duration=duration):
                emails = ["a@example.com", "b@example.com"]
                client = _PooledClient(
                    emails, bodies={email: MEMBER_ONLY_BODY for email in emails}
                )

                with self.assertLogs("qwen2api.videos", level="WARNING") as captured:
                    with self.assertRaises(videos.HTTPException) as ctx:
                        await videos._run_generation(
                            client, "qwen3.8-max", "一只猫", 1, duration, "16:9", {},
                            resolution=resolution,
                        )

                self.assertEqual(ctx.exception.status_code, 400)
                self.assertEqual(
                    ctx.exception.detail, videos.MEMBER_ONLY_DETAIL
                )
                # 只请求一次：规格问题换号无用，不白烧账号池
                self.assertEqual(client.selected, ["a@example.com"])
                self.assertEqual(len(client.received_excludes), 1)
                # 不误标账号限流，且该号仍被正常释放
                self.assertEqual(client.account_pool.video_limited, [])
                self.assertEqual(client.account_pool.released, ["a@example.com"])
                self.assertIn(
                    "member_only spec rejected", "\n".join(captured.output)
                )

    async def test_blocked_accounts_are_skipped_while_cooling(self) -> None:
        client = _PooledClient(
            ["a@example.com", "b@example.com"],
            blocked={"a@example.com"},
        )

        result = await videos._run_generation(
            client, "qwen3.8-max", "一只猫", 1, 5, "16:9", {},
            resolution="720P",
        )

        # 冷却期内的 A 一开始就被排除，直接选 B
        self.assertEqual(client.selected, ["b@example.com"])
        self.assertIn("a@example.com", client.received_excludes[0])
        self.assertEqual(result["data"][0]["url"], SUCCESS_VIDEO_URL)
        self.assertEqual(client.account_pool.released, ["b@example.com"])
        self.assertEqual(client.account_pool.video_limited, [])

    async def test_all_accounts_cooling_returns_429(self) -> None:
        client = _PooledClient(
            ["a@example.com", "b@example.com"],
            blocked={"a@example.com", "b@example.com"},
        )

        with self.assertLogs("qwen2api.videos", level="WARNING"):
            with self.assertRaises(videos.HTTPException) as ctx:
                await videos._run_generation(
                    client, "qwen3.8-max", "一只猫", 1, 5, "16:9", {},
                    resolution="720P",
                )

        # 池子里只剩冷却号 → 语义上同样是「额度都用完了」
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(
            ctx.exception.detail, videos.VIDEO_QUOTA_EXHAUSTED_DETAIL
        )
        self.assertEqual(client.selected, [])
        self.assertEqual(client.account_pool.released, [])


if __name__ == "__main__":
    unittest.main()
