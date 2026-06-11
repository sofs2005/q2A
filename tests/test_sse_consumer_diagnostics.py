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


if __name__ == "__main__":
    unittest.main()
