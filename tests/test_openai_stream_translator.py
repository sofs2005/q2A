import json
import unittest
from unittest.mock import patch

from backend.services.openai_stream_translator import OpenAIStreamTranslator
from backend.services.token_calc import calculate_usage
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcore.types import ToolDefinition


class OpenAIStreamTranslatorTests(unittest.TestCase):
    def test_emit_tool_calls_uses_full_arguments_without_splitting(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_test",
            created=1,
            model_name="gpt-4.1",
            client_profile="claude_code_openai",
        )

        translator.emit_tool_calls([
            {
                "id": "call_1",
                "name": "Read",
                "input": {"file_path": "a" * 300},
            }
        ])

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        tool_call_chunks = [payload["choices"][0]["delta"]["tool_calls"][0] for payload in payloads if payload["choices"][0]["delta"].get("tool_calls")]

        self.assertEqual(len(tool_call_chunks), 1)
        self.assertEqual(tool_call_chunks[0]["function"]["name"], "Read")
        self.assertEqual(tool_call_chunks[0]["function"]["arguments"], json.dumps({"file_path": "a" * 300}, ensure_ascii=False))

    def test_emit_tool_calls_maps_gateway_name_back_to_client_name(self) -> None:
        catalog = ToolCatalog([
            ToolDefinition(name="exec", client_name="exec", model_name="bridge-0"),
        ])
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_test",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            tool_catalog=catalog,
        )

        translator.emit_tool_calls([
            {
                "id": "call_1",
                "name": "bridge-0",
                "input": {"command": "echo hi"},
            }
        ])

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        tool_call_chunks = [payload["choices"][0]["delta"]["tool_calls"][0] for payload in payloads if payload["choices"][0]["delta"].get("tool_calls")]

        self.assertEqual(tool_call_chunks[0]["function"]["name"], "exec")
        self.assertNotEqual(tool_call_chunks[0]["function"]["name"], "bridge-0")

    def test_safe_text_delta_bypasses_state_machine_scanning(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_safe",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )

        class FailingStateMachine:
            def process_text_delta(self, _text):
                raise AssertionError("safe text should not be parsed again")

        translator.state_machine = FailingStateMachine()
        text = "plain safe literal text"

        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, text, None)

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )
        self.assertEqual(content_text, text)
        self.assertEqual(translator.answer_fragments, [text])

    def test_safe_text_delta_does_not_emit_raw_dsml_tool_markup(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_safe_dsml",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )
        text = (
            "visible prefix\n"
            '<|DSML|tool_calls>\n'
            '  <|DSML|invoke name="bridge-15">\n'
            '    <|DSML|parameter name="file_path"></|DSML|parameter>\n'
            '    <|DSML|parameter name="limit"></|DSML|parameter>\n'
            '    <|DSML|parameter name="offset"></|DSML|parameter>\n'
            '  </|DSML|invoke>\n'
            '</|DSML|tool_calls>\n'
            "visible suffix"
        )

        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, text, None)

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )
        self.assertEqual(content_text, "visible prefix")
        self.assertNotIn("<|DSML|", content_text)

    def test_safe_text_delta_holds_cross_chunk_dsml_prefix(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_safe_split_dsml",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )

        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, "visible <|DS", None)
        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, 'ML|parameter name="file_path"><![CDATA[secret', None)

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )
        self.assertEqual(content_text, "visible")
        self.assertNotIn("<|DS", content_text)
        self.assertNotIn("<|DSML|", content_text)
        self.assertNotIn("<![CDATA[", content_text)

    def test_safe_text_delta_holds_cross_chunk_closing_dsml_and_cdata_prefixes(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_safe_split_close",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )

        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, "visible </|DS", None)
        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, "ML|tool_calls>", None)
        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, " more <![CD", None)
        translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, "ATA[hidden", None)

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )
        self.assertEqual(content_text, "visible more")
        self.assertNotIn("</|DS", content_text)
        self.assertNotIn("<![CDATA[", content_text)

    def test_safe_text_delta_with_plain_angle_bracket_stays_on_fast_path(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_safe_angle",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )
        text = "plain comparison: a < b"

        with patch(
            "backend.services.openai_stream_translator.find_tool_markup_tag_outside_ignored",
            side_effect=AssertionError("plain safe text should not be scanned"),
        ):
            translator.on_delta({"phase": "answer", "_qwen2api_safe_text": True}, text, None)

        payloads = [json.loads(chunk[6:].strip()) for chunk in translator.pending_chunks if chunk.startswith("data: ")]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )
        self.assertEqual(content_text, text)

    def test_finalize_emits_usage_as_separate_empty_choices_chunk(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_usage",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
        )
        translator.on_delta({"phase": "answer"}, "hello", None)
        usage = calculate_usage("prompt text", "hello")

        chunks = translator.finalize("stop", usage=usage)

        payloads = [json.loads(chunk[6:].strip()) for chunk in chunks if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"]
        finish_payload = payloads[-2]
        usage_payload = payloads[-1]
        self.assertEqual(finish_payload["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("usage", finish_payload)
        self.assertEqual(usage_payload["choices"], [])
        self.assertEqual(usage_payload["usage"], usage)

    def test_finalize_tool_calls_uses_call_id_full_arguments_and_separate_usage_chunk(self) -> None:
        directive = type(
            "Directive",
            (),
            {
                "stop_reason": "tool_use",
                "tool_blocks": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123_1",
                        "name": "image_generate",
                        "input": {"prompt": "x" * 300},
                    }
                ],
            },
        )()
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_tool_usage",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            build_final_directive=lambda _text: directive,
            allowed_tool_names=["image_generate"],
        )
        translator.on_delta({"phase": "answer"}, "visible progress", None)
        usage = calculate_usage("prompt text", "")

        chunks = translator.finalize("stop", usage=usage)

        payloads = [json.loads(chunk[6:].strip()) for chunk in chunks if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"]
        tool_call_chunks = [payload["choices"][0]["delta"]["tool_calls"][0] for payload in payloads if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")]
        finish_payload = payloads[-2]
        usage_payload = payloads[-1]

        self.assertEqual(len(tool_call_chunks), 1)
        self.assertRegex(tool_call_chunks[0]["id"], r"^call_[0-9a-f]+$")
        self.assertEqual(tool_call_chunks[0]["function"]["name"], "image_generate")
        self.assertEqual(tool_call_chunks[0]["function"]["arguments"], json.dumps({"prompt": "x" * 300}, ensure_ascii=False))
        self.assertEqual(finish_payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("usage", finish_payload)
        self.assertEqual(usage_payload["choices"], [])
        self.assertEqual(usage_payload["usage"], usage)

    def test_finalize_corrects_stop_to_tool_calls_after_tool_call_emitted(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_finish_tool",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )

        translator.emit_tool_calls([{"id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}])
        chunks = translator.finalize("stop")

        payloads = [json.loads(chunk[6:].strip()) for chunk in chunks if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"]
        finish_payload = payloads[-1]
        self.assertEqual(finish_payload["choices"][0]["finish_reason"], "tool_calls")

    def test_stream_tool_call_preserves_preceding_content_chunks(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_tool",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["execute_code"],
        )

        translator.on_delta({"phase": "answer"}, "temporary answer", None)
        translator.on_delta(
            {"phase": "tool_call"},
            None,
            [{"id": "call_1", "name": "execute_code", "input": {"code": "print(1)"}}],
        )

        chunks = translator.finalize("tool_calls")
        payloads = [json.loads(chunk[6:].strip()) for chunk in chunks if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )
        tool_call_chunks = [
            payload
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("tool_calls")
        ]
        finish_reasons = [
            payload["choices"][0].get("finish_reason")
            for payload in payloads
            if payload["choices"]
        ]

        self.assertEqual(content_text, "temporary answer")
        self.assertTrue(tool_call_chunks)
        self.assertIn("tool_calls", finish_reasons)

    def test_finalize_drops_incomplete_tool_wrapper_text_when_valid_tool_call_exists(self) -> None:
        directive = type(
            "Directive",
            (),
            {
                "stop_reason": "tool_use",
                "tool_blocks": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "README.md"}}],
            },
        )()
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_test",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            build_final_directive=lambda _text: directive,
            allowed_tool_names=["Read"],
            toolcore_enabled=True,
        )

        translator.on_delta({"phase": "answer"}, '##TOOL_CALL##\n{"name": "exec", "input": {"command": "ls -la', None)
        translator.on_delta({"phase": "answer"}, ' /tmp"}}\n##TOOL_CALL##\n{"name": "Read", "input": {"path": "README.md"}}\n##END_CALL##', None)

        chunks = translator.finalize("stop")
        payloads = [json.loads(chunk[6:].strip()) for chunk in chunks if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"][0]["delta"].get("content")
        )
        emitted_tool_calls = [payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls")]

        self.assertEqual(content_text, "")
        self.assertTrue(emitted_tool_calls)

    def test_content_chunks_strip_broken_dsml_control_markup(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_broken_dsml",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
            allowed_tool_names=["Read"],
        )

        translator.on_delta({"phase": "answer"}, "visible text <|DSML|parameter name=\"x\"><![CDATA[secret", None)
        chunks = translator.finalize("stop")

        payloads = [json.loads(chunk[6:].strip()) for chunk in chunks if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]"]
        content_text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"].get("content")
        )

        self.assertEqual(content_text, "visible text ")
        self.assertNotIn("<|DSML|", content_text)
        self.assertNotIn("<![CDATA[", content_text)


if __name__ == "__main__":
    unittest.main()
