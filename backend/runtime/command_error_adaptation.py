from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


@dataclass(slots=True, frozen=True)
class CommandErrorClassification:
    kind: str
    shell: str = "unknown"
    confidence: str = "low"


def _environment_shell(command_environment: Any = None) -> str:
    return str(getattr(command_environment, "shell", "unknown") or "unknown").strip().lower() or "unknown"


def looks_like_command_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "parsererror",
            "traceback",
            "error:",
            "exception",
            "missing file specification after redirection operator",
            "unexpected eof",
            "unterminated",
            "no closing quotation",
            "missing terminating",
            "not recognized as the name of a cmdlet",
            "is not recognized as an internal or external command",
        )
    ) or re.search(r"(?m)^[^\n:]+:\s*command not found\b", lowered) is not None


def classify_command_error(error_text: str, *, command_environment: Any = None) -> CommandErrorClassification:
    text = str(error_text or "")
    lowered = text.lower()
    shell = _environment_shell(command_environment)

    if "missing file specification after redirection operator" in lowered:
        return CommandErrorClassification(kind="shell_syntax_error", shell="powershell", confidence="high")

    if any(marker in lowered for marker in ("unexpected eof", "unterminated", "no closing quotation", "missing terminating")):
        return CommandErrorClassification(kind="quote_balance_error", shell=shell, confidence="medium")

    if (
        re.search(r"(?m)^[^\n:]+:\s*command not found\b", lowered)
        or "not recognized as the name of a cmdlet" in lowered
        or "is not recognized as an internal or external command" in lowered
    ):
        return CommandErrorClassification(kind="missing_command_error", shell=shell, confidence="medium")

    return CommandErrorClassification(kind="unknown_error", shell=shell, confidence="low")


def build_command_error_retry_prompt(*, classification: CommandErrorClassification, current_prompt: str, command_environment: Any = None) -> str:
    if "[Command repair reminder]" in str(current_prompt or ""):
        return current_prompt
    shell = classification.shell if classification.shell != "unknown" else _environment_shell(command_environment)
    if shell == "powershell":
        repair_hint = "PowerShell does not support POSIX here-documents; prefer @' ... '@ | python - or a temporary .py file, and keep nested quotes balanced."
    elif shell in {"bash", "zsh", "sh"}:
        repair_hint = "Use syntax that is valid for the explicit POSIX shell; prefer a quoted here-document only when the shell is POSIX, otherwise use stdin or a temporary file."
    else:
        repair_hint = "Use stdin or a temporary script file and avoid shell-specific syntax until the execution environment is clear."

    return (
        f"{current_prompt}\n\n"
        "[Command repair reminder]\n"
        f"The previous read-only command failed with `{classification.kind}`. "
        f"{repair_hint}\n"
        "Retry by emitting a corrected tool call only; do not repeat the same invalid command text."
    )
