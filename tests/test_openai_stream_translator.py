import json
import unittest

from backend.services.openai_stream_translator import OpenAIStreamTranslator
from backend.services.token_calc import calculate_usage
from backend.toolcore.tool_catalog import ToolCatalog
from backend.toolcore.types import ToolDefinition


class OpenAIStreamTranslatorTests(unittest.TestCase):
    def test_emit_tool_calls_splits_arguments_into_multiple_chunks(self) -> None:
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl_test",
            created=1,
            model_name="gpt-4.1",
            client_profile="openclaw_openai",
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

        self.assertEqual(tool_call_chunks[0]["function"]["name"], "Read")
        self.assertEqual(tool_call_chunks[0]["function"]["arguments"], "")
        rebuilt = "".join(chunk["function"].get("arguments", "") for chunk in tool_call_chunks[1:])
        self.assertEqual(rebuilt, json.dumps({"file_path": "a" * 300}, ensure_ascii=False))
        self.assertGreater(len(tool_call_chunks), 2)

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

    def test_finalize_can_emit_token_usage_chunk(self) -> None:
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
        usage_payload = payloads[-1]
        self.assertEqual(usage_payload["choices"], [])
        self.assertEqual(usage_payload["usage"], usage)

    def test_stream_tool_call_discards_preceding_content_chunks(self) -> None:
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

        self.assertEqual(content_text, "")
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


if __name__ == "__main__":
    unittest.main()
