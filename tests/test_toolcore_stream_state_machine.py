import unittest

from backend.toolcore.stream_state_machine import ToolStreamStateMachine


class ToolCoreStreamStateMachineTests(unittest.TestCase):
    def test_partial_tool_wrapper_does_not_leak_before_classification(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        events = machine.process_text_delta('##TOOL_CALL##\n{"name": "Read"')

        self.assertEqual(events, [])

    def test_cross_chunk_marker_is_held_until_safe(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        events1 = machine.process_text_delta("##TOOL_C")
        events2 = machine.process_text_delta('ALL##\n{"name": "Read", "input": {"path": "README.md"}}\n##END_CALL##')

        self.assertEqual(events1, [])
        self.assertTrue(any(event.type == "tool_calls" for event in events2))

    def test_malformed_wrapper_is_suppressed_if_later_tool_call_wins(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        machine.process_text_delta('##TOOL_CALL##\n{"name": "exec", "input": {"command": "ls -la /tmp"')
        machine.process_tool_calls([{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}])
        events = machine.flush(final_tool_use=True)

        self.assertFalse(any(event.type == "content" and event.text and "##TOOL_CALL##" in event.text for event in events))

    def test_unknown_dsml_wrapper_is_suppressed_if_later_tool_call_wins(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        events = machine.process_text_delta(
            '<|DSML|tool_calls><|DSML|invoke name="exec"><|DSML|parameter name="command"><![CDATA[ls]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
        )
        machine.process_tool_calls([{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}])
        flushed = machine.flush(final_tool_use=True)

        self.assertEqual(events, [])
        self.assertFalse(any(event.type == "content" and event.text and "DSML" in event.text for event in flushed))

    def test_unknown_dsml_wrapper_does_not_flush_as_text_without_final_tool_use(self) -> None:
        machine = ToolStreamStateMachine(["Read"])
        content = '<|DSML|tool_calls><|DSML|invoke name="exec"></|DSML|invoke></|DSML|tool_calls>'

        self.assertEqual(machine.process_text_delta(content), [])
        flushed = machine.flush(final_tool_use=False)

        text = "".join(event.text or "" for event in flushed if event.type == "content")
        self.assertEqual(text, "")

    def test_failed_attempt_output_is_isolated_from_later_success(self) -> None:
        machine = ToolStreamStateMachine(["Read"])

        machine.process_text_delta("Tool exec does not exists.")
        machine.reset_attempt()
        events = machine.process_tool_calls([{"id": "call_1", "name": "Read", "input": {"path": "README.md"}}])

        self.assertTrue(any(event.type == "tool_calls" for event in events))
        flushed = machine.flush(final_tool_use=True)
        self.assertFalse(any(event.type == "content" and event.text and "Tool exec does not exists." in event.text for event in flushed))


if __name__ == "__main__":
    unittest.main()
