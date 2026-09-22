import unittest

from backend.upstream.sse_consumer import parse_sse_chunk


class SseConsumerDiagnosticsTests(unittest.TestCase):
    def test_logs_diagnostic_for_json_event_without_choices(self) -> None:
        chunk = 'data: {"success": false, "msg": "blocked", "code": 400}\n\n'

        with self.assertLogs("qwen2api.sse", level="WARNING") as logs:
            parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed, [])
        self.assertIn("unparsed json event", "\n".join(logs.output))
        self.assertIn("success", "\n".join(logs.output))

    def test_logs_diagnostic_for_non_json_data_line(self) -> None:
        chunk = "data: upstream plain text response\n\n"

        with self.assertLogs("qwen2api.sse", level="WARNING") as logs:
            parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed, [])
        self.assertIn("non-json data line", "\n".join(logs.output))
        self.assertIn("upstream plain text response", "\n".join(logs.output))

    def test_parses_plain_content_event_like_upstream_go_version(self) -> None:
        chunk = 'data: {"content": "hello"}\n\n'

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed, [{"type": "delta", "phase": "answer", "content": "hello", "status": "", "extra": {}}])

    def test_parses_nested_data_event_like_upstream_go_version(self) -> None:
        chunk = 'data: {"data": {"answer": "hello"}}\n\n'

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed, [{"type": "delta", "phase": "answer", "content": "hello", "status": "", "extra": {}}])

    def test_parses_response_created_as_lifecycle(self) -> None:
        chunk = (
            'data: {"response.created":{"chat_id":"c1","parent_id":"p1",'
            '"response_id":"r1","response_index":"0"}}\n\n'
        )

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["type"], "lifecycle")
        self.assertEqual(parsed[0]["phase"], "response.created")
        self.assertEqual(parsed[0]["extra"]["response_id"], "r1")

    def test_parses_upstream_error_event(self) -> None:
        chunk = (
            'data: {"error":{"code":"invalid_input","details":"输入或附件无效。请检查后重试。"},'
            '"response_id":"r1","response_index":0}\n\n'
        )

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["type"], "error")
        self.assertEqual(parsed[0]["code"], "invalid_input")
        self.assertIn("附件", parsed[0]["details"])
        self.assertEqual(parsed[0]["response_id"], "r1")

    def test_parses_quota_limit_error_event(self) -> None:
        chunk = (
            'data: {"error":{"code":"quota_limit","details":"目前服务访问量较大，请稍后再试。"},'
            '"response_id":"r2","response_index":0}\n\n'
        )

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed[0]["type"], "error")
        self.assertEqual(parsed[0]["code"], "quota_limit")

    def test_parses_response_info_keep_alive_as_lifecycle(self) -> None:
        """心跳帧不当 unparsed，否则会抬高 chunk_count 却不抬高 parsed_event_count。"""
        chunk = 'data: {"response.info":{"action":"keep_alive"}}\n\n'

        with self.assertNoLogs("qwen2api.sse", level="WARNING"):
            parsed = parse_sse_chunk(chunk)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["type"], "lifecycle")
        self.assertEqual(parsed[0]["phase"], "response.info")
        self.assertEqual(parsed[0]["status"], "keep_alive")

    def test_attaches_normalized_usage_to_delta_event(self) -> None:
        chunk = (
            'data: {"choices":[{"delta":{"role":"assistant","content":"We",'
            '"phase":"answer","status":"typing"}}],"response_id":"r1",'
            '"usage":{"input_tokens":85,"output_tokens":12,"total_tokens":97,'
            '"output_tokens_details":{"reasoning_tokens":4,"text_tokens":8},'
            '"prompt_tokens_details":{"cached_tokens":10}}}\n\n'
        )

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(len(parsed), 1)
        usage = parsed[0]["usage"]
        self.assertEqual(usage["prompt_tokens"], 85)
        self.assertEqual(usage["completion_tokens"], 12)
        self.assertEqual(usage["total_tokens"], 97)
        self.assertEqual(usage["_upstream_details"]["output_tokens_details"]["reasoning_tokens"], 4)
        self.assertEqual(usage["_upstream_details"]["prompt_tokens_details"]["cached_tokens"], 10)

    def test_derives_total_tokens_when_upstream_omits_it(self) -> None:
        chunk = 'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"input_tokens":3,"output_tokens":4}}\n\n'

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed[0]["usage"]["total_tokens"], 7)
        self.assertNotIn("_upstream_details", parsed[0]["usage"])

    def test_does_not_add_usage_key_when_absent_or_null(self) -> None:
        """条件加键是 test_parses_plain_content_event_* 全等断言不破的前提。"""
        no_usage = parse_sse_chunk('data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        null_usage = parse_sse_chunk('data: {"choices":[{"delta":{"content":"x"}}],"usage":null}\n\n')
        junk_usage = parse_sse_chunk('data: {"choices":[{"delta":{"content":"x"}}],"usage":{"foo":1}}\n\n')

        for parsed in (no_usage, null_usage, junk_usage):
            self.assertNotIn("usage", parsed[0])

    def test_keep_alive_frame_with_usage_is_normalized(self) -> None:
        chunk = 'data: {"response.info":{"action":"keep_alive"},"usage":{"input_tokens":1,"output_tokens":2}}\n\n'

        parsed = parse_sse_chunk(chunk)

        self.assertEqual(parsed[0]["type"], "lifecycle")
        self.assertEqual(parsed[0]["usage"]["total_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
