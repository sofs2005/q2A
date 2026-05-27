import unittest

from backend.toolcore.formatter import (
    build_canonical_anthropic_message,
    build_canonical_gemini_payload,
    build_canonical_openai_chat_payload,
    build_canonical_openai_responses_payload,
)
from backend.services.token_calc import calculate_usage, count_tokens


class ToolCoreFormatterTests(unittest.TestCase):
    def test_openai_chat_formatter_renders_tool_calls(self) -> None:
        payload = build_canonical_openai_chat_payload(
            completion_id="chatcmpl_1",
            created=1,
            model_name="gpt-4.1",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "Read")

    def test_openai_responses_formatter_renders_function_call_items(self) -> None:
        payload = build_canonical_openai_responses_payload(
            response_id="resp_1",
            created=1,
            model_name="gpt-4.1",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["output"][0]["name"], "Read")

    def test_anthropic_formatter_renders_tool_use_blocks(self) -> None:
        payload = build_canonical_anthropic_message(
            msg_id="msg_1",
            model_name="claude-3-5-sonnet",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0]["type"], "tool_use")

    def test_gemini_formatter_renders_text_payload(self) -> None:
        payload = build_canonical_gemini_payload(answer_text="hello")

        self.assertEqual(payload["candidates"][0]["content"]["parts"][0]["text"], "hello")

    def test_gemini_formatter_renders_function_call_payload(self) -> None:
        payload = build_canonical_gemini_payload(
            answer_text="",
            tool_calls=[{"name": "Read", "input": {"file_path": "README.md"}}],
        )

        part = payload["candidates"][0]["content"]["parts"][0]["functionCall"]
        self.assertEqual(part["name"], "Read")
        self.assertEqual(part["args"], {"file_path": "README.md"})
        self.assertEqual(payload["candidates"][0]["finishReason"], "STOP")

    def test_openai_chat_formatter_counts_tool_calls_as_completion_usage(self) -> None:
        payload = build_canonical_openai_chat_payload(
            completion_id="chatcmpl_tool_usage",
            created=1,
            model_name="gpt-4.1",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertGreater(payload["usage"]["completion_tokens"], 0)
        self.assertEqual(payload["usage"]["total_tokens"], payload["usage"]["prompt_tokens"] + payload["usage"]["completion_tokens"])

    def test_openai_responses_formatter_counts_tool_calls_as_output_usage(self) -> None:
        payload = build_canonical_openai_responses_payload(
            response_id="resp_tool_usage",
            created=1,
            model_name="gpt-4.1",
            prompt="prompt",
            answer_text="",
            reasoning_text="",
            directives=[{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
        )

        self.assertGreater(payload["usage"]["output_tokens"], 0)
        self.assertEqual(payload["usage"]["total_tokens"], payload["usage"]["input_tokens"] + payload["usage"]["output_tokens"])

    def test_openai_chat_formatter_reports_token_usage(self) -> None:
        prompt = "hello world hello world hello world"
        answer_text = "I will review the project structure first."
        payload = build_canonical_openai_chat_payload(
            completion_id="chatcmpl_usage",
            created=1,
            model_name="gpt-4.1",
            prompt=prompt,
            answer_text=answer_text,
            reasoning_text="",
            directives=[],
        )

        self.assertEqual(payload["usage"], calculate_usage(prompt, answer_text))
        self.assertNotEqual(payload["usage"]["prompt_tokens"], len(prompt))

    def test_openai_responses_formatter_reports_token_usage(self) -> None:
        prompt = "The quick brown fox jumps over the lazy dog."
        answer_text = "The logs show success, but latency is higher than expected."
        reasoning_text = "Check the status code first, then inspect latency."
        payload = build_canonical_openai_responses_payload(
            response_id="resp_usage",
            created=1,
            model_name="gpt-4.1",
            prompt=prompt,
            answer_text=answer_text,
            reasoning_text=reasoning_text,
            directives=[],
        )

        self.assertEqual(payload["usage"]["input_tokens"], count_tokens(prompt))
        self.assertEqual(payload["usage"]["output_tokens"], count_tokens(answer_text))
        self.assertEqual(payload["usage"]["total_tokens"], count_tokens(prompt) + count_tokens(answer_text))
        self.assertEqual(payload["usage"]["output_tokens_details"]["reasoning_tokens"], count_tokens(reasoning_text))
        self.assertNotEqual(payload["usage"]["input_tokens"], len(prompt))

    def test_anthropic_formatter_reports_token_usage(self) -> None:
        prompt = "hello world hello world hello world"
        answer_text = "I will keep the answer concise."
        payload = build_canonical_anthropic_message(
            msg_id="msg_usage",
            model_name="claude-3-5-sonnet",
            prompt=prompt,
            answer_text=answer_text,
            reasoning_text="",
            directives=[],
        )

        self.assertEqual(payload["usage"]["input_tokens"], count_tokens(prompt))
        self.assertEqual(payload["usage"]["output_tokens"], count_tokens(answer_text))
        self.assertNotEqual(payload["usage"]["input_tokens"], len(prompt))


if __name__ == "__main__":
    unittest.main()
