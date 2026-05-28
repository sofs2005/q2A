import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.core.config import settings
from backend.runtime.execution import cleanup_runtime_resources
from backend.services.context_cleanup import context_cleanup_loop
from backend.services.garbage_collector import garbage_collect_chats


class UpstreamAutoDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_cleanup_releases_account_without_deleting_chat_by_default(self) -> None:
        acc = SimpleNamespace(token="token-1")
        client = SimpleNamespace(
            account_pool=SimpleNamespace(release=AsyncMock()),
            delete_chat=AsyncMock(),
        )
        client.account_pool.release = unittest.mock.Mock()

        with patch.object(settings, "UPSTREAM_AUTO_DELETE_ENABLED", False):
            await cleanup_runtime_resources(client, acc, "chat-1")
            await asyncio.sleep(0)

        client.account_pool.release.assert_called_once_with(acc)
        client.delete_chat.assert_not_called()

    async def test_context_cleanup_keeps_expired_upstream_records_by_default(self) -> None:
        class StopLoop(Exception):
            pass

        record = SimpleNamespace(
            account_email="user@example.com",
            chat_id="chat-1",
            session_key="session-1",
            uploaded_files=[{"file_id": "file-1"}],
        )
        acc = SimpleNamespace(token="token-1")
        app = SimpleNamespace(
            state=SimpleNamespace(
                context_offloader=SimpleNamespace(settings=SimpleNamespace(CONTEXT_ATTACHMENT_TTL_SECONDS=1)),
                file_store=SimpleNamespace(cleanup_expired=AsyncMock()),
                session_affinity=SimpleNamespace(cleanup_expired=AsyncMock(return_value=[record])),
                upstream_file_cache=SimpleNamespace(cleanup_expired=AsyncMock()),
                account_pool=SimpleNamespace(get_by_email=unittest.mock.Mock(return_value=acc)),
                qwen_client=SimpleNamespace(delete_chat=AsyncMock()),
                upstream_file_uploader=SimpleNamespace(delete_remote_file=AsyncMock()),
            )
        )

        with patch.object(settings, "UPSTREAM_AUTO_DELETE_ENABLED", False):
            with patch("backend.services.context_cleanup.asyncio.sleep", AsyncMock(side_effect=StopLoop)):
                with self.assertRaises(StopLoop):
                    await context_cleanup_loop(app, interval_seconds=60)

        app.state.qwen_client.delete_chat.assert_not_called()
        app.state.upstream_file_uploader.delete_remote_file.assert_not_called()

    async def test_garbage_collector_keeps_api_chats_by_default(self) -> None:
        class StopLoop(Exception):
            pass

        acc = SimpleNamespace(is_available=unittest.mock.Mock(return_value=True), token="token-1", email="user@example.com")
        client = SimpleNamespace(
            account_pool=SimpleNamespace(accounts=[acc]),
            list_chats=AsyncMock(return_value=[{"id": "chat-1", "title": "api_debug"}]),
            delete_chat=AsyncMock(),
        )
        app = SimpleNamespace(
            state=SimpleNamespace(
                qwen_client=client,
                session_affinity=SimpleNamespace(active_chat_ids=unittest.mock.Mock(return_value=set())),
            )
        )

        with patch.object(settings, "UPSTREAM_AUTO_DELETE_ENABLED", False):
            with patch("backend.services.garbage_collector.asyncio.sleep", AsyncMock(side_effect=[None, StopLoop])):
                with self.assertRaises(StopLoop):
                    await garbage_collect_chats(app)

        client.list_chats.assert_not_called()
        client.delete_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
