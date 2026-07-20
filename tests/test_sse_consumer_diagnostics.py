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


if __name__ == "__main__":
    unittest.main()
