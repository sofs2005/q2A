import unittest

from backend.services.token_calc import (
    calculate_usage,
    effective_extra_prompt_tokens,
    merge_upstream_usage,
    resolve_usage,
)


class MergeUpstreamUsageTests(unittest.TestCase):
    def test_returns_local_unchanged_when_upstream_missing(self) -> None:
        local = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        for upstream in (None, {}, "junk", 42, []):
            self.assertEqual(merge_upstream_usage(local, upstream), local)

    def test_returns_local_when_upstream_has_no_token_fields(self) -> None:
        local = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        self.assertEqual(merge_upstream_usage(local, {"foo": 1}), local)
        self.assertEqual(merge_upstream_usage(local, {"prompt_tokens": None}), local)

    def test_overrides_both_directions(self) -> None:
        local = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        merged = merge_upstream_usage(
            local, {"prompt_tokens": 85, "completion_tokens": 12, "total_tokens": 97}
        )

        self.assertEqual(merged, {"prompt_tokens": 85, "completion_tokens": 12, "total_tokens": 97})

    def test_falls_back_per_field_when_one_side_missing(self) -> None:
        local = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        merged = merge_upstream_usage(local, {"prompt_tokens": 85})

        self.assertEqual(merged["prompt_tokens"], 85)
        self.assertEqual(merged["completion_tokens"], 5)
        self.assertEqual(merged["total_tokens"], 90)

    def test_derives_total_when_upstream_omits_it(self) -> None:
        local = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        merged = merge_upstream_usage(local, {"prompt_tokens": 3, "output_tokens": 4})

        self.assertEqual(merged["total_tokens"], 3 + local["completion_tokens"])

    def test_rejects_invalid_values(self) -> None:
        local = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        # bool 与负数视为非法 → 逐项回落本地值
        merged = merge_upstream_usage(local, {"prompt_tokens": True, "completion_tokens": -3})

        self.assertEqual(merged, local)

    def test_preserves_upstream_details(self) -> None:
        local = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        details = {"output_tokens_details": {"reasoning_tokens": 7}}

        merged = merge_upstream_usage(
            local,
            {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "_upstream_details": details},
        )

        self.assertEqual(merged["_upstream_details"], details)


class ResolveUsageTests(unittest.TestCase):
    def test_falls_back_to_local_estimate(self) -> None:
        prompt, answer = "hello world", "hi there"

        self.assertEqual(resolve_usage(prompt, answer), calculate_usage(prompt, answer))

    def test_prefers_upstream_when_present(self) -> None:
        resolved = resolve_usage(
            "hello world",
            "hi there",
            upstream_usage={"prompt_tokens": 85, "completion_tokens": 12, "total_tokens": 97},
        )

        self.assertEqual(
            resolved, {"prompt_tokens": 85, "completion_tokens": 12, "total_tokens": 97}
        )

    def test_extra_prompt_tokens_still_applied_by_default(self) -> None:
        """默认保守：上游未确认含附件时继续叠加 extra_prompt_tokens。"""
        resolved = resolve_usage("hello", "hi", extra_prompt_tokens=50)

        self.assertEqual(resolved["prompt_tokens"], calculate_usage("hello", "hi")["prompt_tokens"] + 50)

    def test_extra_prompt_tokens_dropped_when_flag_enabled(self) -> None:
        from unittest.mock import patch

        from backend.core.config import settings

        upstream = {"prompt_tokens": 85, "completion_tokens": 12, "total_tokens": 97}
        with patch.object(settings, "QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS", True):
            self.assertEqual(effective_extra_prompt_tokens(50, upstream_usage=upstream), 0)
            self.assertEqual(effective_extra_prompt_tokens(50, upstream_usage=None), 50)

    def test_extra_prompt_tokens_kept_when_flag_disabled(self) -> None:
        from unittest.mock import patch

        from backend.core.config import settings

        upstream = {"prompt_tokens": 85, "completion_tokens": 12, "total_tokens": 97}
        with patch.object(settings, "QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS", False):
            self.assertEqual(effective_extra_prompt_tokens(50, upstream_usage=upstream), 50)


if __name__ == "__main__":
    unittest.main()
