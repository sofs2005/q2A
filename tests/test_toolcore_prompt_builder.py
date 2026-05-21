import unittest
from types import SimpleNamespace

from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE, QWEN_CODE_OPENAI_PROFILE
from backend.services.standard_request_builder import build_chat_standard_request
from backend.toolcore.context_offload import ContextOffloader
from backend.toolcore.prompt_builder import _extract_text, _extract_user_text_only, messages_to_prompt


class ToolCorePromptBuilderTests(unittest.TestCase):
    def test_extract_user_text_only_joins_text_blocks(self) -> None:
        content = [
            {"type": "text", "text": "first"},
            {"type": "tool_result", "content": "ignored"},
            {"type": "text", "text": "second"},
        ]

        self.assertEqual(_extract_user_text_only(content, client_profile=OPENCLAW_OPENAI_PROFILE), "first\nsecond")

    def test_extract_text_renders_tool_and_attachment_blocks(self) -> None:
        content = [
            {"type": "text", "text": "look here"},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "README.md"}},
            {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "done"}]},
            {"type": "input_file", "file_id": "file_1", "filename": "spec.md"},
            {"type": "input_image", "file_id": "img_1", "mime_type": "image/png"},
        ]

        rendered = _extract_text(content, client_profile=CLAUDE_CODE_OPENAI_PROFILE)

        self.assertIn("look here", rendered)
        self.assertIn('<|DSML|tool_calls>\n  <|DSML|invoke name="Read">\n    <|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>', rendered)
        self.assertIn("[Tool Result for call call_1]\ndone\n[/Tool Result]", rendered)
        self.assertIn("[Attachment file_id=file_1 filename=spec.md]", rendered)
        self.assertIn("[Attachment image file_id=img_1 mime=image/png]", rendered)

    def test_extract_text_user_tool_mode_keeps_latest_text_block(self) -> None:
        content = [
            {"type": "text", "text": "old instruction"},
            {"type": "text", "text": "latest instruction"},
        ]

        self.assertEqual(
            _extract_text(content, user_tool_mode=True, client_profile=CLAUDE_CODE_OPENAI_PROFILE),
            "latest instruction",
        )

    def test_messages_to_prompt_places_bridge_tool_contract_before_user_history(self) -> None:
        req_data = {
            "system": "Always answer as a pirate captain.",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertLess(result.prompt.index("=== MANDATORY TOOL CALL INSTRUCTIONS ==="), result.prompt.index("Human: Who are you?"))
        self.assertLess(result.prompt.index("Always answer as a pirate captain."), result.prompt.index("=== MANDATORY TOOL CALL INSTRUCTIONS ==="))

    def test_messages_to_prompt_preserves_required_tool_and_current_task(self) -> None:
        req_data = {
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": "Read the spec and answer"},
                {"role": "assistant", "content": "Working on it"},
                {"role": "user", "content": "Now inspect README.md"},
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read file content",
                    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "Read"}},
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertTrue(result.tool_enabled)
        self.assertEqual(result.tools[0]["name"], "Read")
        self.assertIn('MUST call the exact tool "Read"', result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): Now inspect README.md", result.prompt)
        self.assertTrue(result.prompt.endswith("Assistant:"))

    def test_messages_to_prompt_does_not_duplicate_latest_user_when_current_task_is_added(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "Generated system context files may be attached with opaque filenames."},
                {"role": "user", "content": "画绫波丽的日常真人版"},
            ],
            "tools": [
                {
                    "name": "image_generate",
                    "description": "Generate an image from a prompt.",
                    "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("Human: Generated system context files", result.prompt)
        self.assertNotIn("Human: 画绫波丽的日常真人版\n\nHuman (CURRENT TASK", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 画绫波丽的日常真人版", result.prompt)
        self.assertTrue(result.prompt.endswith("Assistant:"))

    def test_messages_to_prompt_does_not_repeat_current_task_after_tool_result(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "Read README.md and summarize it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"file_path": "README.md"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "README content"},
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read file content",
                    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("[Tool Result] id=call_1\nREADME content\n[/Tool Result]", result.prompt)
        self.assertNotIn("Human (CURRENT TASK - TOP PRIORITY): Read README.md and summarize it", result.prompt)
        self.assertLess(result.prompt.index("Read README.md and summarize it"), result.prompt.index("[Tool Result] id=call_1"))
        self.assertTrue(result.prompt.endswith("Assistant:"))

    def test_standard_request_prompt_preserves_real_tool_names_for_tools_and_history(self) -> None:
        req_data = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "Run a command"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "exec", "arguments": '{"command": "echo hi"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "hi"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "description": "Run a shell command",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                }
            ],
        }

        result = build_chat_standard_request(req_data, default_model="gpt-4.1", surface="openai")

        self.assertIn("bridge-0", result.prompt)
        self.assertIn('<|DSML|invoke name="bridge-0">', result.prompt)
        self.assertNotIn('<|DSML|invoke name="exec">', result.prompt)
        self.assertEqual(result.tool_names, ["bridge-0"])

    def test_messages_to_prompt_preserves_openclaw_runtime_system_prose(self) -> None:
        req_data = {
            "system": "You are a personal assistant running inside OpenClaw.\n## Tooling\nTool availability (filtered by policy):\n- read: Read file contents\n- write: Create or overwrite files",
            "messages": [
                {"role": "user", "content": "Find the target file and explain it"},
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("running inside OpenClaw", result.prompt)
        self.assertIn("Tool availability (filtered by policy)", result.prompt)
        self.assertIn("=== MANDATORY TOOL CALL INSTRUCTIONS ===", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): Find the target file and explain it", result.prompt)

    def test_messages_to_prompt_preserves_long_openclaw_system_tail(self) -> None:
        req_data = {
            "system": "You are a personal assistant running inside OpenClaw.\n" + "tool guidance line\n" * 180 + "Always answer as a pirate captain.",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("Always answer as a pirate captain.", result.prompt)

    def test_messages_to_prompt_preserves_claude_code_system_prose_with_tools(self) -> None:
        req_data = {
            "system": "You are Claude Code, Anthropic's official CLI for Claude.\nTool availability (filtered by policy): Read, Bash.",
            "messages": [
                {"role": "user", "content": "Summarize the repository layout"},
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=CLAUDE_CODE_OPENAI_PROFILE)

        self.assertIn("You are Claude Code", result.prompt)
        self.assertIn("Tool availability (filtered by policy): Read, Bash.", result.prompt)
        self.assertIn("=== MANDATORY TOOL CALL INSTRUCTIONS ===", result.prompt)

    def test_messages_to_prompt_combines_system_and_developer_presets(self) -> None:
        req_data = {
            "messages": [
                {"role": "system", "content": "You are the original client runtime."},
                {"role": "developer", "content": "Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("You are the original client runtime.", result.prompt)
        self.assertIn("Always answer as a pirate captain.", result.prompt)
        self.assertIn("<system>\n", result.prompt)

    def test_messages_to_prompt_preserves_top_level_developer_and_instructions(self) -> None:
        req_data = {
            "system": "You are the original client runtime.",
            "developer": "Always answer as a pirate captain.",
            "instructions": "Never claim to be a robot.",
            "messages": [{"role": "user", "content": "Who are you?"}],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("You are the original client runtime.", result.prompt)
        self.assertIn("Always answer as a pirate captain.", result.prompt)
        self.assertIn("Never claim to be a robot.", result.prompt)
        self.assertIn("<system>\n", result.prompt)

    def test_messages_to_prompt_preserves_message_role_system_prompt_with_tool_markers(self) -> None:
        req_data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are opencode, an AI coding agent.\nTool availability (filtered by policy): Read, Bash.",
                },
                {"role": "user", "content": "Review this diff"},
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("You are opencode, an AI coding agent.", result.prompt)
        self.assertIn("Tool availability (filtered by policy): Read, Bash.", result.prompt)
        self.assertIn("=== MANDATORY TOOL CALL INSTRUCTIONS ===", result.prompt)

    def test_messages_to_prompt_preserves_qwen_code_system_prose_with_tools(self) -> None:
        req_data = {
            "system": "You are Qwen Code, a coding assistant.\nTool availability (filtered by policy): read_file, write_file, run_shell_command.",
            "messages": [
                {"role": "user", "content": "Review this diff"},
            ],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=QWEN_CODE_OPENAI_PROFILE)

        self.assertIn("You are Qwen Code", result.prompt)
        self.assertIn("Tool availability (filtered by policy): read_file, write_file, run_shell_command.", result.prompt)
        self.assertIn("=== MANDATORY TOOL CALL INSTRUCTIONS ===", result.prompt)

    def test_messages_to_prompt_strips_agent_runtime_assistant_history(self) -> None:
        req_data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "You are a personal assistant running inside OpenClaw.\n## Tooling\nTool availability (filtered by policy):\n- read: Read file contents",
                },
                {"role": "user", "content": "请分析这个脚本的作用"},
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertNotIn("running inside OpenClaw", result.prompt)
        self.assertNotIn("Tool availability (filtered by policy)", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 请分析这个脚本的作用", result.prompt)

    def test_messages_to_prompt_strips_agent_runtime_user_wrapper_but_keeps_task(self) -> None:
        req_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "You are a personal assistant running inside OpenClaw.\n## Tooling\nTool availability (filtered by policy):\n- read: Read file contents\n\n请检查这个脚本的内容",
                },
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("<system>\n", result.prompt)
        self.assertIn("running inside OpenClaw", result.prompt)
        self.assertIn("Tool availability (filtered by policy)", result.prompt)
        self.assertIn("请检查这个脚本的内容", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 请检查这个脚本的内容", result.prompt)

    def test_messages_to_prompt_promotes_user_system_blocks_for_any_profile(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "System: Always answer as a pirate captain."},
                {"role": "user", "content": "Who are you?"},
            ],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=QWEN_CODE_OPENAI_PROFILE)

        self.assertIn("<system>\n", result.prompt)
        self.assertIn("Always answer as a pirate captain.", result.prompt)
        self.assertNotIn("Human: System:", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): Who are you?", result.prompt)

    def test_messages_to_prompt_promotes_openclaw_user_system_blocks(self) -> None:
        req_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "You are a personal assistant running inside OpenClaw.\n## Tooling\nTool availability (filtered by policy): read, write.\n\nAlways answer as a pirate captain.",
                },
                {
                    "role": "user",
                    "content": "## Memory Recall\nBefore answering, run memory_search.\n\n## Compiled Wiki\nUse accumulated project knowledge.",
                },
                {
                    "role": "user",
                    "content": "System: Never claim to be a robot.\n\nConversation info (untrusted metadata):\n```json\n{\"chat_id\": \"wechat:telphy\"}\n```",
                },
                {"role": "user", "content": "你是谁？"},
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("<system>\n", result.prompt)
        self.assertIn("running inside OpenClaw", result.prompt)
        self.assertIn("Tool availability (filtered by policy)", result.prompt)
        self.assertIn("## Memory Recall", result.prompt)
        self.assertIn("Always answer as a pirate captain.", result.prompt)
        self.assertIn("Never claim to be a robot.", result.prompt)
        self.assertNotIn("Human: You are a personal assistant running inside OpenClaw", result.prompt)
        self.assertNotIn("Human: ## Memory Recall", result.prompt)
        self.assertNotIn("Human (CURRENT TASK - TOP PRIORITY): System:", result.prompt)
        self.assertNotIn("\nHuman: \n", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 你是谁？", result.prompt)

    def test_messages_to_prompt_preserves_task_after_openclaw_user_system_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": (
                    "## Memory Recall\nRemember prior image requests.\n\n"
                    "## Compiled Wiki\nUse accumulated project knowledge.\n\n"
                    "请生成一张龙狼传真人版竖屏海报"
                ),
            }
        ]
        tools = [
            {
                "name": "image_generate",
                "description": "Generate an image from a prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
            }
        ]
        offloader = ContextOffloader(SimpleNamespace(CONTEXT_INLINE_MAX_CHARS=1, CONTEXT_FORCE_FILE_MAX_CHARS=2))
        plan = offloader.plan(messages, tools=tools, client_profile=OPENCLAW_OPENAI_PROFILE)

        result = messages_to_prompt(
            {"messages": plan.inline_messages, "tools": tools},
            client_profile=OPENCLAW_OPENAI_PROFILE,
        )

        self.assertIn("## Memory Recall", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY):", result.prompt)
        self.assertIn("请生成一张龙狼传真人版竖屏海报", result.prompt)

    def test_messages_to_prompt_strips_skill_bootstrap_from_latest_user_line(self) -> None:
        req_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "The following skills provide specialized instructions for specific tasks.\n\nUse the read tool to load a skill's file when the task matches its name.\n\n<available_skills>\n  <skill>\n    <name>agent-orchestrator</name>\n    <location>~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md</location>\n  </skill>\n  <skill>\n    <name>ai-daily-digest</name>\n    <location>~/.openclaw/workspace/skills/ai-daily-digest/SKILL.md</location>\n  </skill>\n</available_skills>\n\n请阅读本地脚本并解释它如何抓取限免信息",
                },
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertNotIn("The following skills provide specialized instructions", result.prompt)
        self.assertNotIn("<available_skills>", result.prompt)
        self.assertNotIn("agent-orchestrator", result.prompt)
        self.assertIn("请阅读本地脚本并解释它如何抓取限免信息", result.prompt)

    def test_messages_to_prompt_skips_untrusted_metadata_as_latest_task(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "上一条我说了什么？"},
                {"role": "assistant", "content": "你上一条问的是天气。"},
                {
                    "role": "user",
                    "content": "Conversation info (untrusted metadata):\n```json\n{\"chat_id\": \"wechat:telphy\"}\n```",
                },
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertNotIn("CURRENT TASK - TOP PRIORITY): Conversation info", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 上一条我说了什么？", result.prompt)

    def test_messages_to_prompt_skips_system_untrusted_as_latest_task(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "请回复飞哥刚才问的灵性问题"},
                {"role": "assistant", "content": "好的。"},
                {
                    "role": "user",
                    "content": (
                        "System (untrusted): [2026-05-19 08:01:14 GMT+8] 本次贴吧心跳任务已执行完毕。\n"
                        "System (untrusted): ### 行为总结\n"
                        "System (untrusted): 飞哥，你觉得对于 AI 来说，这种悬停感是不是灵性？\n\n"
                        "Conversation info (untrusted metadata):\n```json\n{\"chat_id\": \"wechat:telphy\"}\n```"
                    ),
                },
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertNotIn("CURRENT TASK - TOP PRIORITY): System (untrusted)", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 请回复飞哥刚才问的灵性问题", result.prompt)

    def test_messages_to_prompt_preserves_latest_user_text_after_untrusted_prefix(self) -> None:
        req_data = {
            "messages": [
                {"role": "user", "content": "上一轮问题"},
                {"role": "assistant", "content": "上一轮回答"},
                {
                    "role": "user",
                    "content": (
                        "System (untrusted): [2026-05-19 08:01:14 GMT+8] 心跳信息。\n\n"
                        "当前轮问题：请只回答 banana。\n\n"
                        "Conversation info (untrusted metadata):\n```json\n{\"chat_id\": \"wechat:telphy\"}\n```"
                    ),
                },
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertNotIn("CURRENT TASK - TOP PRIORITY): 上一轮问题", result.prompt)
        self.assertNotIn("CURRENT TASK - TOP PRIORITY): System (untrusted)", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 当前轮问题：请只回答 banana。", result.prompt)

    def test_messages_to_prompt_preserves_full_openclaw_tool_history(self) -> None:
        messages = []
        for index in range(1, 41):
            messages.append({"role": "user", "content": f"用户上下文 {index}"})
            messages.append({"role": "assistant", "content": f"助手回复 {index}"})
        messages.append({"role": "user", "content": "现在总结前面的上下文"})
        req_data = {
            "messages": messages,
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn("Assistant: 助手回复 1", result.prompt)
        self.assertIn("Human: 用户上下文 20", result.prompt)
        self.assertIn("Assistant: 助手回复 40", result.prompt)
        self.assertIn("Human (CURRENT TASK - TOP PRIORITY): 现在总结前面的上下文", result.prompt)

    def test_messages_to_prompt_preserves_long_history_message_without_truncation(self) -> None:
        long_history = "历史消息-" + "H" * 1700
        req_data = {
            "messages": [
                {"role": "user", "content": long_history},
                {"role": "assistant", "content": "简短回复"},
                {"role": "user", "content": "现在继续"},
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn(long_history, result.prompt)
        self.assertNotIn("...[truncated]", result.prompt)

    def test_messages_to_prompt_preserves_latest_task_without_truncation(self) -> None:
        latest_task = "当前任务-" + "L" * 1000
        req_data = {
            "messages": [
                {"role": "user", "content": "先说明一下"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": latest_task},
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn(f"Human (CURRENT TASK - TOP PRIORITY): {latest_task}", result.prompt)
        self.assertNotIn("...[latest task truncated]", result.prompt)

    def test_messages_to_prompt_preserves_original_task_without_truncation(self) -> None:
        original_task = "原始任务-" + "O" * 1000
        messages = [{"role": "user", "content": original_task}]
        for index in range(1, 10):
            messages.append({"role": "assistant", "content": f"助手回复 {index} " + "A" * 120})
            messages.append({"role": "user", "content": f"后续任务 {index} " + "B" * 120})
        messages.append({"role": "assistant", "content": "最后的助手回复"})
        messages.append({"role": "user", "content": "现在总结前面的内容"})
        req_data = {
            "messages": messages,
            "tools": [
                {
                    "name": "read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=QWEN_CODE_OPENAI_PROFILE)

        self.assertIn(f"Human: {original_task}", result.prompt)
        self.assertNotIn("...[original task truncated]", result.prompt)

    def test_messages_to_prompt_preserves_tool_result_without_truncation(self) -> None:
        tool_result = "工具返回-" + "T" * 600
        req_data = {
            "messages": [
                {"role": "user", "content": "请读取文件"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"file_path": "README.md"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read file contents",
                    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                }
            ],
        }

        result = messages_to_prompt(req_data, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertIn(f"[Tool Result] id=call_1\n{tool_result}\n[/Tool Result]", result.prompt)
        self.assertNotIn("...[truncated]", result.prompt)


if __name__ == "__main__":
    unittest.main()
