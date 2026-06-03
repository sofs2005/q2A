import unittest

from backend.toolcore.stream_sieve import ToolStreamSieve


class ToolStreamSieveTests(unittest.TestCase):
    def test_complete_tool_block_extracted_from_streamed_chunks(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk('##TOOL_CALL##\n')
        second = sieve.process_chunk('{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##')

        self.assertEqual(first, [])
        tool_events = [event for event in second if event.get("type") == "tool_calls"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["calls"][0]["name"], "Read")

    def test_partial_tool_block_is_held_until_complete(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        events = sieve.process_chunk('##TOOL_CALL##\n{"name": "Read"')

        self.assertEqual(events, [])

        final_events = sieve.process_chunk(', "input": {"file_path": "README.md"}}\n##END_CALL##')
        tool_events = [event for event in final_events if event.get("type") == "tool_calls"]
        self.assertEqual(len(tool_events), 1)

    def test_fenced_example_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        content = '```json\n##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##\n```'
        events = sieve.process_chunk(content)
        events.extend(sieve.flush())

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, content)

    def test_complete_dsml_block_extracted_from_streamed_chunks(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk("prefix <|DSML|tool_calls>\n")
        second = sieve.process_chunk(
            '  <|DSML|invoke name="Read">\n'
            '    <|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>\n'
            "  </|DSML|invoke>\n"
            "</|DSML|tool_calls> suffix"
        )
        events = first + second + sieve.flush()

        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        tool_events = [event for event in events if event.get("type") == "tool_calls"]
        self.assertEqual(text, "prefix  suffix")
        self.assertEqual(tool_events, [{"type": "tool_calls", "calls": [{"name": "Read", "input": {"file_path": "README.md"}}]}])

    def test_partial_dsml_tag_is_held_until_complete(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk("<|DSML|tool_")
        second = sieve.process_chunk(
            'calls><|DSML|invoke name="Read"><|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
        )

        self.assertEqual(first, [])
        tool_events = [event for event in second if event.get("type") == "tool_calls"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["calls"][0]["input"], {"file_path": "README.md"})

    def test_drifted_dsml_block_is_intercepted(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        events = sieve.process_chunk(
            '＜！DSML！tool_calls＞＜！DSML！invoke name＝“Read”＞'
            '＜！DSML！parameter name＝“file_path”＞README.md＜！/DSML！parameter＞'
            '＜！/DSML！invoke＞＜！/DSML！tool_calls＞'
        )

        tool_events = [event for event in events if event.get("type") == "tool_calls"]
        self.assertEqual(tool_events[0]["calls"], [{"name": "Read", "input": {"file_path": "README.md"}}])

    def test_dsml_inside_fenced_example_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        content = (
            "```xml\n"
            "<|DSML|tool_calls><|DSML|invoke name=\"Read\"></|DSML|invoke></|DSML|tool_calls>\n"
            "```"
        )
        events = sieve.process_chunk(content)
        events.extend(sieve.flush())

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, content)

    def test_dsml_inside_inline_code_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        content = '`<|DSML|tool_calls><|DSML|invoke name="Read"></|DSML|invoke></|DSML|tool_calls>`'
        events = sieve.process_chunk(content)
        events.extend(sieve.flush())

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, content)

    def test_split_fenced_dsml_example_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk("```xml\n<|DSML|tool_calls>")
        second = sieve.process_chunk('<|DSML|invoke name="Read"></|DSML|invoke></|DSML|tool_calls>\n```')
        events = first + second + sieve.flush()

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, '```xml\n<|DSML|tool_calls><|DSML|invoke name="Read"></|DSML|invoke></|DSML|tool_calls>\n```')

    def test_multiple_dsml_blocks_are_drained_without_markup_leak(self) -> None:
        sieve = ToolStreamSieve(["Read", "Write"])
        events = sieve.process_chunk(
            '<|DSML|tool_calls><|DSML|invoke name="Read"><|DSML|parameter name="file_path"><![CDATA[a.md]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
            '<|DSML|tool_calls><|DSML|invoke name="Write"><|DSML|parameter name="content"><![CDATA[x]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
        )
        events.extend(sieve.flush())

        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        calls = [call for event in events if event.get("type") == "tool_calls" for call in event["calls"]]
        self.assertEqual(text, "")
        self.assertEqual([call["name"] for call in calls], ["Read", "Write"])

    def test_legacy_marker_outside_fenced_code_still_parses(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        events = sieve.process_chunk(
            '```json\n##TOOL_CALL##\n{"name": "Read"}\n##END_CALL##\n```\n'
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##'
        )

        tool_events = [event for event in events if event.get("type") == "tool_calls"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["calls"][0]["input"], {"file_path": "README.md"})

    def test_split_fenced_legacy_example_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk('```json\n##TOOL_CALL##\n')
        second = sieve.process_chunk('{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##\n```')
        events = first + second + sieve.flush()

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, '```json\n##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "README.md"}}\n##END_CALL##\n```')

    def test_split_inline_dsml_example_remains_plain_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        first = sieve.process_chunk('`<|DSML|tool_calls>')
        second = sieve.process_chunk('<|DSML|invoke name="Read"></|DSML|invoke></|DSML|tool_calls>`')
        events = first + second + sieve.flush()

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, '`<|DSML|tool_calls><|DSML|invoke name="Read"></|DSML|invoke></|DSML|tool_calls>`')

    def test_incomplete_dsml_tool_block_does_not_flush_as_text(self) -> None:
        sieve = ToolStreamSieve(["bridge-23"])
        sieve.process_chunk(
            '<|DSML|tool_calls>\n'
            '  <|DSML|invoke name="bridge-23">\n'
            '    <|DSML|parameter name="query"><![CDATA[AI artificial intelligence open source LLM breakthrough last 2 hours]]></|DSML|parameter>\n'
            '    <'
        )
        events = sieve.flush()

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertEqual(text, "")

    def test_incomplete_tool_block_flushes_as_text(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        sieve.process_chunk('##TOOL_CALL##\n{"name": "Read"')
        events = sieve.flush()

        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertIn('##TOOL_CALL##', text)

    def test_complete_dsml_block_with_empty_parameters_is_extracted_without_markup_leak(self) -> None:
        sieve = ToolStreamSieve(["bridge-24"])
        events = sieve.process_chunk(
            '<|DSML|tool_calls>\n'
            '<|DSML|invoke name="bridge-24">\n'
            '<|DSML|parameter name="taskId"></|DSML|parameter>\n'
            '<|DSML|parameter name="status"></|DSML|parameter>\n'
            '</|DSML|invoke>\n'
            '</|DSML|tool_calls>'
        )
        events.extend(sieve.flush())

        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        tool_events = [event for event in events if event.get("type") == "tool_calls"]
        self.assertEqual(text, "")
        self.assertEqual(
            tool_events,
            [{"type": "tool_calls", "calls": [{"name": "bridge-24", "input": {"taskId": "", "status": ""}}]}],
        )

    def test_incomplete_dsml_capture_waits_for_close_before_full_parse(self) -> None:
        sieve = ToolStreamSieve(["bridge-24"])

        with unittest.mock.patch(
            "backend.toolcore.stream_sieve.consume_dsml_tool_capture",
            side_effect=AssertionError("incomplete DSML should not run full capture parse per chunk"),
        ):
            self.assertEqual(sieve.process_chunk('<|DSML|tool_calls>\n'), [])
            self.assertEqual(sieve.process_chunk('<|DSML|invoke name="bridge-24">\n'), [])
            self.assertEqual(sieve.process_chunk('<|DSML|parameter name="taskId">'), [])

        events = sieve.process_chunk('</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>')
        tool_events = [event for event in events if event.get("type") == "tool_calls"]
        self.assertEqual(tool_events[0]["calls"], [{"name": "bridge-24", "input": {"taskId": ""}}])

    def test_closed_dsml_with_unknown_tool_does_not_stay_capturing_or_leak_markup(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        events = sieve.process_chunk(
            '<|DSML|tool_calls>'
            '<|DSML|invoke name="exec"></|DSML|invoke>'
            '</|DSML|tool_calls> after'
        )
        events.extend(sieve.flush())

        text = "".join(event.get("text", "") for event in events if event.get("type") == "content")
        self.assertFalse(sieve.capturing)
        self.assertFalse(any(event.get("type") == "tool_calls" for event in events))
        self.assertNotIn("DSML", text)
        self.assertEqual(text, " after")

    def test_oversized_incomplete_dsml_capture_degrades_to_content(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        events = sieve.process_chunk('<|DSML|tool_calls><|DSML|invoke name="Read">')
        self.assertEqual(events, [])

        emitted: list[dict[str, object]] = []
        for _ in range(70):
            emitted.extend(sieve.process_chunk("x" * 1024))
            if emitted:
                break

        self.assertFalse(sieve.capturing)
        self.assertFalse(any(event.get("type") == "tool_calls" for event in emitted))
        text = "".join(str(event.get("text", "")) for event in emitted if event.get("type") == "content")
        self.assertIn('<|DSML|tool_calls>', text)
        self.assertGreater(len(text), 64 * 1024)

    def test_unclosed_fenced_code_streams_content_without_pending_growth(self) -> None:
        sieve = ToolStreamSieve(["Read"])
        emitted_text = []

        for index in range(120):
            chunk = "```go\n" if index == 0 else "func x(){ println(\"hello\") }\n" * 10
            events = sieve.process_chunk(chunk)
            emitted_text.extend(event.get("text", "") for event in events if event.get("type") == "content")

        self.assertFalse(any("tool_calls" == event.get("type") for event in events))
        self.assertGreater(len("".join(emitted_text)), 10_000)
        self.assertLess(len(sieve.pending), 4096)


if __name__ == "__main__":
    unittest.main()
