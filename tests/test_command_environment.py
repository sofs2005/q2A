import unittest

from backend.services.command_environment import (
    CommandEnvironment,
    detect_command_environment,
    format_command_environment_hint,
)
from backend.services.standard_request_builder import build_chat_standard_request


class CommandEnvironmentTests(unittest.TestCase):
    def test_detects_explicit_shell_and_platform_hints(self) -> None:
        env = detect_command_environment(
            headers={
                "x-shell": "Bash",
                "x-platform": "Linux",
            },
            request_data={"metadata": {"shell": "PowerShell", "platform": "Windows"}},
        )

        self.assertEqual(env.shell, "powershell")
        self.assertEqual(env.platform, "windows")
        self.assertEqual(env.source, "explicit")
        self.assertEqual(format_command_environment_hint(env), "powershell/windows")

    def test_detects_header_hints_when_metadata_is_missing(self) -> None:
        env = detect_command_environment(
            headers={"x-shell": "Bash", "x-platform": "Linux"},
            request_data={},
        )

        self.assertEqual(env.shell, "bash")
        self.assertEqual(env.platform, "linux")
        self.assertEqual(env.source, "headers")
        self.assertEqual(format_command_environment_hint(env), "bash/linux")

    def test_returns_unknown_environment_when_no_hint_exists(self) -> None:
        env = detect_command_environment(headers={}, request_data={})

        self.assertEqual(env.shell, "unknown")
        self.assertEqual(env.platform, "unknown")
        self.assertEqual(env.source, "unknown")
        self.assertEqual(format_command_environment_hint(env), "unknown")

    def test_standard_request_builder_carries_command_environment(self) -> None:
        command_environment = CommandEnvironment(shell="bash", platform="linux", source="explicit")

        request = build_chat_standard_request(
            {"messages": [{"role": "user", "content": "List files."}]},
            default_model="gpt-4.1",
            surface="openai",
            command_environment=command_environment,
        )

        self.assertEqual(request.command_environment, command_environment)


if __name__ == "__main__":
    unittest.main()
