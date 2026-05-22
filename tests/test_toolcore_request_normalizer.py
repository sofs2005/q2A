from __future__ import annotations

import unittest

from backend.toolcore.request_normalizer import (
    normalize_anthropic_request,
    normalize_chat_request,
    normalize_gemini_request,
    normalize_responses_request,
    to_prompt_payload,
)
from backend.toolcore.types import ToolChoicePolicy, ToolCoreRequest


class ChatRequestNormalizationTests(unittest.TestCase):
    def test_chat_request_with_tools(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )

        self.assertIsInstance(result, ToolCoreRequest)
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].name, "get_weather")
        self.assertEqual(result.tools[0].description, "Get weather for a location")
        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.AUTO)
        self.assertEqual(result.messages, [{"role": "user", "content": "What's the weather in Tokyo?"}])

    def test_chat_request_required_tool_choice(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "Call a tool"}],
                "tools": [{"type": "function", "function": {"name": "test_tool"}}],
                "tool_choice": "required",
            }
        )

        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.REQUIRED)

    def test_chat_request_none_tool_choice(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "Don't call tools"}],
                "tools": [{"type": "function", "function": {"name": "test_tool"}}],
                "tool_choice": "none",
            }
        )

        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.NONE)

    def test_chat_request_specific_function_tool_choice(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "Call get_weather"}],
                "tools": [
                    {"type": "function", "function": {"name": "get_weather"}},
                    {"type": "function", "function": {"name": "get_time"}},
                ],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            }
        )

        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.FORCED)
        self.assertEqual(result.forced_tool_name, "get_weather")

    def test_prompt_payload_uses_bridge_slots_and_forced_choice_mapping(self) -> None:
        result = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "run code"}],
                "tools": [
                    {"type": "function", "function": {"name": "exec", "description": "Run shell", "parameters": {}}},
                    {"type": "function", "function": {"name": "read", "description": "Read file", "parameters": {}}},
                ],
                "tool_choice": {"type": "function", "function": {"name": "exec"}},
            }
        )

        payload = to_prompt_payload(result, model="gpt-4.1")

        self.assertEqual([tool["name"] for tool in payload["tools"]], ["bridge-0", "bridge-1"])
        self.assertEqual(payload["tool_choice"], {"type": "function", "function": {"name": "bridge-0"}})
        self.assertEqual(result.tool_catalog.get_client_name("exec"), "exec")
        self.assertEqual(result.tool_catalog.get_model_name("read"), "bridge-1")

    def test_chat_request_invalid_tool_choice_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid tool_choice"):
            normalize_chat_request(
                {
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "tool_choice": "invalid_value",
                }
            )


class ResponsesRequestNormalizationTests(unittest.TestCase):
    def test_responses_request_with_tools(self) -> None:
        result = normalize_responses_request(
            {
                "input": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather info",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )

        self.assertIsInstance(result, ToolCoreRequest)
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].name, "get_weather")
        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.AUTO)
        self.assertEqual(result.messages, [{"role": "user", "content": "What's the weather?"}])

    def test_responses_request_with_function_call_output(self) -> None:
        result = normalize_responses_request(
            {
                "input": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                            }
                        ],
                    }
                ],
                "tools": [],
                "function_call_output": {"call_id": "call_123", "output": "Sunny, 25°C"},
            }
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].call_id, "call_123")
        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.tool_calls[0].input, {"city": "Tokyo"})
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0].call_id, "call_123")
        self.assertEqual(result.tool_results[0].output, "Sunny, 25°C")

    def test_responses_request_without_tools_rejects_tool_choice(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool_choice provided but no tools"):
            normalize_responses_request(
                {
                    "input": [{"role": "user", "content": "Hi"}],
                    "tools": [],
                    "tool_choice": "auto",
                }
            )

    def test_responses_request_invalid_tool_choice_raises_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid tool_choice"):
            normalize_responses_request(
                {
                    "input": [{"role": "user", "content": "Test"}],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                    "tool_choice": "invalid",
                }
            )


class CrossSurfaceNormalizationTests(unittest.TestCase):
    def test_anthropic_request_normalizes_to_same_toolcore_shape(self) -> None:
        result = normalize_anthropic_request(
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            }
        )

        self.assertEqual([tool.name for tool in result.tools], ["Read"])
        self.assertEqual([tool.model_name for tool in result.tools], ["bridge-0"])
        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.FORCED)
        self.assertEqual(result.forced_tool_name, "Read")

    def test_anthropic_prompt_payload_maps_forced_tool_choice_to_bridge_slot(self) -> None:
        result = normalize_anthropic_request(
            {
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "tool", "name": "Read"},
            }
        )

        payload = to_prompt_payload(result, model="claude-3-5-sonnet")

        self.assertEqual([tool["name"] for tool in payload["tools"]], ["bridge-0"])
        self.assertEqual(payload["tool_choice"], {"type": "function", "function": {"name": "bridge-0"}})

    def test_gemini_request_normalizes_to_same_toolcore_shape(self) -> None:
        result = normalize_gemini_request(
            {
                "contents": [{"role": "user", "parts": [{"text": "read the file"}]}],
                "tools": [{"functionDeclarations": [{"name": "Read", "description": "Read file", "parameters": {"type": "object"}}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            },
            model="gemini-2.5-pro",
            force_stream=False,
        )

        self.assertEqual(result.messages, [{"role": "user", "content": "read the file"}])
        self.assertEqual([tool.name for tool in result.tools], ["Read"])
        self.assertEqual([tool.model_name for tool in result.tools], ["bridge-0"])
        self.assertEqual(result.tool_choice_policy, ToolChoicePolicy.REQUIRED)

    def test_equivalent_requests_across_surfaces_produce_equivalent_core_fields(self) -> None:
        openai_chat = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"type": "function", "function": {"name": "Read", "description": "Read file", "parameters": {"type": "object"}}}],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            }
        )
        anthropic = normalize_anthropic_request(
            {
                "messages": [{"role": "user", "content": "read the file"}],
                "tools": [{"name": "Read", "description": "Read file", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            }
        )

        self.assertEqual(openai_chat.messages, anthropic.messages)
        self.assertEqual([tool.name for tool in openai_chat.tools], [tool.name for tool in anthropic.tools])
        self.assertEqual(openai_chat.tool_choice_policy, anthropic.tool_choice_policy)
        self.assertEqual(openai_chat.forced_tool_name, anthropic.forced_tool_name)


class SkillCatalogPropagationTests(unittest.TestCase):
    def test_raw_skill_catalog_survives_normalization_into_prompt_payload(self) -> None:
        request = normalize_chat_request(
            {
                "messages": [{"role": "user", "content": "read the file"}],
                "_skill_catalog": {
                    "agent-orchestrator": {
                        "location": "~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md",
                        "description": "Coordinate multi-step work across tools.",
                        "aliases": ["orchestrator"],
                        "source": "openclaw",
                    }
                },
            }
        )

        self.assertIsNotNone(request.skill_catalog)
        assert request.skill_catalog is not None
        self.assertEqual(request.skill_catalog.resolve("orchestrator").name, "agent-orchestrator")

        payload = to_prompt_payload(request, model="gpt-4.1")

        self.assertIn("_skill_catalog", payload)
        self.assertIs(payload["_skill_catalog"], request.skill_catalog)
        self.assertEqual(payload["_skill_catalog"].resolve("agent-orchestrator").location, "~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md")


if __name__ == "__main__":
    unittest.main()
