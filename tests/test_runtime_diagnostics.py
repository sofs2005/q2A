import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from backend.core.diagnostics import install_stack_dump_handler


class RuntimeDiagnosticsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
