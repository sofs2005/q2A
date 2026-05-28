import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

FASTAPI_RUNTIME_AVAILABLE = True
FASTAPI_RUNTIME_IMPORT_ERROR = ""

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import images
    from backend.core.config import settings
except (ModuleNotFoundError, ImportError) as exc:
    FASTAPI_RUNTIME_AVAILABLE = False
    FASTAPI_RUNTIME_IMPORT_ERROR = str(exc)
    FastAPI = None
    TestClient = None
    images = None
    settings = None


@unittest.skipUnless(
    FASTAPI_RUNTIME_AVAILABLE,
    f"FastAPI runtime dependencies unavailable: {FASTAPI_RUNTIME_IMPORT_ERROR}",
)
class ImagesHttpTests(unittest.TestCase):
    def test_create_image_does_not_require_list_chats(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, fixed_account=None, existing_chat_id=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            yield {
                "type": "event",
                "event": {
                    "choices": [
                        {
                            "delta": {
                                "content": '![result](https://cdn.qwenlm.ai/image-1.png)'
                            }
                        }
                    ]
                },
            }

        app = FastAPI()
        app.include_router(images.router)
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.6-plus"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"][0]["url"], "https://cdn.qwenlm.ai/image-1.png")
        app.state.qwen_client.account_pool.release.assert_called_once_with(acc)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)

    def test_create_image_uses_current_chat_fallback_when_stream_payload_has_no_url(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, fixed_account=None, existing_chat_id=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            yield {"type": "event", "event": {"choices": [{"delta": {"content": "image ready"}}]}}

        current_chat = {
            "id": "chat-1",
            "title": "api_image",
            "messages": [{"content": "![fallback](https://cdn.qwenlm.ai/fallback-image.png)"}],
        }
        app = FastAPI()
        app.include_router(images.router)
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            list_chats=AsyncMock(return_value=[current_chat]),
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )
        app.state.qwen_client.account_pool.release = Mock(side_effect=lambda account: setattr(account, "inflight", max(0, getattr(account, "inflight", 0) - 1)))

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.6-plus"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"][0]["url"], "https://cdn.qwenlm.ai/fallback-image.png")
        app.state.qwen_client.list_chats.assert_awaited_once_with("token-1", limit=20, account=acc)
        app.state.qwen_client.account_pool.release.assert_called_once_with(acc)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)

    def test_create_image_does_not_double_release_after_stream_failure(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, fixed_account=None, existing_chat_id=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            app.state.qwen_client.account_pool.release(acc)
            raise RuntimeError("upstream failed")

        app = FastAPI()
        app.include_router(images.router)
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )
        app.state.qwen_client.account_pool.release = Mock(side_effect=lambda account: setattr(account, "inflight", max(0, getattr(account, "inflight", 0) - 1)))

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.6-plus"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(app.state.qwen_client.account_pool.release.call_count, 1)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)


if __name__ == "__main__":
    unittest.main()
