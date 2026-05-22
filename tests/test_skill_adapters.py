from __future__ import annotations

import unittest

from backend.services.client_profiles import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.skills.adapters import extract_skill_catalog_from_messages


class SkillAdapterTests(unittest.TestCase):
    def test_extract_skill_catalog_from_openclaw_bootstrap(self) -> None:
        messages = [
            {
                "role": "user",
                "content": (
                    "The following skills provide specialized instructions for specific tasks.\n\n"
                    "<available_skills>\n"
                    "  <skill>\n"
                    "    <name>agent-orchestrator</name>\n"
                    "    <location>~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md</location>\n"
                    "    <description>Coordinate multi-step work across tools.</description>\n"
                    "    <aliases>orchestrator, agent-coordinator</aliases>\n"
                    "    <source>openclaw</source>\n"
                    "  </skill>\n"
                    "  <skill>\n"
                    "    <name>ai-daily-digest</name>\n"
                    "    <location>~/.openclaw/workspace/skills/ai-daily-digest/SKILL.md</location>\n"
                    "  </skill>\n"
                    "</available_skills>\n\n"
                    "请阅读本地脚本并解释它如何抓取限免信息"
                ),
            }
        ]

        catalog = extract_skill_catalog_from_messages(messages, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertEqual([descriptor.name for descriptor in catalog], ["agent-orchestrator", "ai-daily-digest"])
        self.assertEqual(catalog.resolve("orchestrator").name, "agent-orchestrator")
        descriptor = catalog.resolve("agent-orchestrator")
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertEqual(descriptor.location, "~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md")
        self.assertEqual(descriptor.description, "Coordinate multi-step work across tools.")
        self.assertEqual(descriptor.aliases, ("orchestrator", "agent-coordinator"))
        self.assertEqual(descriptor.source, "openclaw")

    def test_extract_skill_catalog_handles_nested_content_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Ignore this prefix"},
                    {
                        "type": "group",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "<available_skills>\n"
                                    "  <skill>\n"
                                    "    <name>agent-orchestrator</name>\n"
                                    "    <location>~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md</location>\n"
                                    "  </skill>\n"
                                    "</available_skills>"
                                ),
                            }
                        ],
                    },
                ],
            }
        ]

        catalog = extract_skill_catalog_from_messages(messages, client_profile=OPENCLAW_OPENAI_PROFILE)

        self.assertEqual([descriptor.name for descriptor in catalog], ["agent-orchestrator"])
        self.assertEqual(catalog.resolve("agent-orchestrator").location, "~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md")

    def test_extract_skill_catalog_ignores_non_openclaw_profiles(self) -> None:
        messages = [
            {
                "role": "user",
                "content": (
                    "<available_skills>\n"
                    "  <skill>\n"
                    "    <name>agent-orchestrator</name>\n"
                    "    <location>~/.openclaw/workspace/skills/agent-orchestrator/SKILL.md</location>\n"
                    "  </skill>\n"
                    "</available_skills>"
                ),
            }
        ]

        catalog = extract_skill_catalog_from_messages(messages, client_profile=CLAUDE_CODE_OPENAI_PROFILE)

        self.assertEqual(list(catalog), [])


if __name__ == "__main__":
    unittest.main()
