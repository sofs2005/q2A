import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.core.diagnostics import (
    format_event_loop_lag_warning,
    get_active_request_diagnostic,
    install_stack_dump_handler,
    reset_active_request_diagnostic,
)
from backend.core.request_logging import request_context, update_request_context


class RuntimeDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_active_request_diagnostic()

    def tearDown(self) -> None:
        reset_active_request_diagnostic()

    def test_installs_sigusr1_stack_dump_handler_when_enabled(self) -> None:
        faulthandler = SimpleNamespace(enable=Mock(), register=Mock())
        signal_module = SimpleNamespace(SIGUSR1=object())
        settings = SimpleNamespace(DIAGNOSTIC_STACK_DUMP_ENABLED=True)
        stream = object()

        installed = install_stack_dump_handler(
            settings=settings,
            faulthandler_module=faulthandler,
            signal_module=signal_module,
            stream=stream,
        )

        self.assertTrue(installed)
        faulthandler.enable.assert_called_once_with(file=stream, all_threads=True)
        faulthandler.register.assert_called_once_with(signal_module.SIGUSR1, file=stream, all_threads=True)

    def test_skips_stack_dump_handler_when_disabled(self) -> None:
        faulthandler = SimpleNamespace(enable=Mock(), register=Mock())
        signal_module = SimpleNamespace(SIGUSR1=object())
        settings = SimpleNamespace(DIAGNOSTIC_STACK_DUMP_ENABLED=False)

        installed = install_stack_dump_handler(
            settings=settings,
            faulthandler_module=faulthandler,
            signal_module=signal_module,
            stream=object(),
        )

        self.assertFalse(installed)
        faulthandler.enable.assert_not_called()
        faulthandler.register.assert_not_called()

    def test_request_context_updates_active_diagnostic_context(self) -> None:
        with request_context(
            req_id="req_1",
            surface="openai",
            requested_model="gpt-4.1",
            resolved_model="qwen3.6-plus",
            chat_id="chat_1",
            stream_stage="tool_sieve",
        ):
            snapshot = get_active_request_diagnostic()
            self.assertEqual(snapshot.req_id, "req_1")
            self.assertEqual(snapshot.surface, "openai")
            self.assertEqual(snapshot.requested_model, "gpt-4.1")
            self.assertEqual(snapshot.resolved_model, "qwen3.6-plus")
            self.assertEqual(snapshot.chat_id, "chat_1")
            self.assertEqual(snapshot.stream_stage, "tool_sieve")

        self.assertEqual(get_active_request_diagnostic().req_id, "-")

    def test_event_loop_lag_warning_includes_active_request_context(self) -> None:
        with request_context(
            req_id="req_2",
            surface="models",
            requested_model="-",
            resolved_model="-",
            stream_stage="watchdog",
        ):
            message = format_event_loop_lag_warning(
                lag_seconds=12.345,
                poll_seconds=1.0,
                snapshot=get_active_request_diagnostic(),
            )

        self.assertIn("[Diagnostics] event_loop_lag", message)
        self.assertIn("lag=12.345s", message)
        self.assertIn("poll=1.000s", message)
        self.assertIn("req_id=req_2", message)
        self.assertIn("surface=models", message)
        self.assertIn("stream_stage=watchdog", message)

    def test_update_request_context_skips_redundant_same_stream_stage_updates(self) -> None:
        with request_context(
            req_id="req_3",
            surface="openai",
            stream_stage="tool_sieve_emit",
        ):
            with patch("backend.core.request_logging.update_active_request_diagnostic") as update_diag:
                update_request_context(stream_stage="tool_sieve_emit")

        update_diag.assert_not_called()


if __name__ == "__main__":
    unittest.main()
