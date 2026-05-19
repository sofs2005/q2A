import unittest

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import RuntimeAttemptState, evaluate_retry_directive, extract_blocked_tool_names, request_max_attempts, should_retry_textual_tool_contract


class ExecutionToolChoiceRetryTests(unittest.TestCase):
    def _request(self) -> StandardRequest:
        return StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "Read", "parameters": {}}, {"name": "Write", "parameters": {}}],
            tool_names=["Read", "Write"],
            tool_enabled=True,
            tool_choice_mode="required",
            required_tool_name="Read",
        )

    def test_required_tool_choice_retries_when_no_tool_call_present(self) -> None:
        retry = evaluate_retry_directive(
            request=self._request(),
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(answer_text="plain text response", emitted_visible_output=True),
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "required_tool_choice_missing_tool_call")

    def test_required_tool_choice_retries_when_wrong_tool_is_called(self) -> None:
        retry = evaluate_retry_directive(
            request=self._request(),
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(
                answer_text='<tool_call>{"name": "Write", "input": {"file_path": "a.txt", "content": "x"}}</tool_call>',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "required_tool_choice_wrong_tool:Write")

    def test_tool_choice_none_blocks_tool_call(self) -> None:
        request = self._request()
        request.tool_choice_mode = "none"
        request.required_tool_name = None

        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(
                answer_text='<tool_call>{"name": "Read", "input": {"file_path": "a.txt"}}</tool_call>',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "tool_choice_none_blocked_tool_call")

    def test_repeated_same_tool_does_not_cross_user_turn_boundary(self) -> None:
        request = StandardRequest(
            prompt="prompt",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "exec", "parameters": {}}],
            tool_names=["exec"],
            tool_enabled=True,
        )

        history_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": '{"command": "mcporter call amap.maps_direction_driving --origin \\"上海市\\" --destination \\"无锡市\\""}',
                        },
                    }
                ],
            },
            {"role": "user", "content": "用高德查一下上海到无锡的路线做成卡片发给我"},
        ]

        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=history_messages,
            attempt_index=0,
            max_attempts=2,
            state=RuntimeAttemptState(
                answer_text='##TOOL_CALL##\n{"name": "exec", "input": {"command": "mcporter call amap.maps_direction_driving --origin \\"上海市\\" --destination \\"无锡市\\""}}\n##END_CALL##',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertFalse(retry.retry)

    def test_analysis_task_does_not_retry_first_same_read(self) -> None:
        request = StandardRequest(
            prompt="Human: analyze this local script and explain how it works\n\nAssistant:",
            response_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            surface="openai",
            tools=[{"name": "read", "parameters": {}}],
            tool_names=["read"],
            tool_enabled=True,
        )

        history_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "script.py"}',
                        },
                    }
                ],
            }
        ]

        retry = evaluate_retry_directive(
            request=request,
            current_prompt=request.prompt,
            history_messages=history_messages,
            attempt_index=0,
            max_attempts=2,
            state=RuntimeAttemptState(
                answer_text='##TOOL_CALL##\n{"name": "read", "input": {"path": "script.py"}}\n##END_CALL##',
                emitted_visible_output=True,
            ),
            allow_after_visible_output=True,
        )

        self.assertFalse(retry.retry)

    def test_incomplete_dsml_tool_contract_triggers_retry(self) -> None:
        request = self._request()
        request.tool_choice_mode = "auto"
        request.required_tool_name = None
        answer_text = '<|DSML|tool_calls>\n<|DSML|invoke name="Read">\n<|\n</|DSML|tool_calls>'

        self.assertTrue(should_retry_textual_tool_contract(answer_text))

        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(answer_text=answer_text),
        )

        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "unparsed_textual_tool_contract:Read")

    def test_mapped_bridge_missing_tool_error_retries(self) -> None:
        request = self._request()
        request.tool_choice_mode = "auto"
        request.required_tool_name = None
        request.tool_names = ["bridge-7"]

        blocked = extract_blocked_tool_names("Tool bridge-7 does not exists.", request.tool_names)
        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(answer_text="Tool bridge-7 does not exists.", blocked_tool_names=blocked),
            allow_after_visible_output=True,
        )

        self.assertEqual(blocked, ["bridge-7"])
        self.assertTrue(retry.retry)
        self.assertEqual(retry.reason, "blocked_tool_name:bridge-7")

    def test_unmapped_bridge_missing_tool_error_does_not_retry(self) -> None:
        request = self._request()
        request.tool_choice_mode = "auto"
        request.required_tool_name = None
        request.tool_names = ["bridge-7"]

        blocked = extract_blocked_tool_names("Tool bridge-999 does not exists.", request.tool_names)
        retry = evaluate_retry_directive(
            request=request,
            current_prompt="prompt",
            history_messages=[],
            attempt_index=0,
            max_attempts=3,
            state=RuntimeAttemptState(answer_text="Tool bridge-999 does not exists.", blocked_tool_names=blocked),
            allow_after_visible_output=True,
        )

        self.assertEqual(blocked, [])
        self.assertFalse(retry.retry)

    def test_bridge_missing_tool_error_with_empty_mapping_does_not_retry(self) -> None:
        self.assertEqual(extract_blocked_tool_names("Tool bridge-7 does not exists.", []), [])

    def test_auto_tool_requests_allow_two_recovery_retries(self) -> None:
        request = self._request()
        request.tool_choice_mode = "auto"
        request.required_tool_name = None

        self.assertEqual(request_max_attempts(request), 3)


if __name__ == "__main__":
    unittest.main()
