import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

FASTAPI_RUNTIME_AVAILABLE = True
FASTAPI_RUNTIME_IMPORT_ERROR = ""

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import images
    from backend.core.config import settings
    from backend.services.file_store import LocalFileStore
except (ModuleNotFoundError, ImportError) as exc:
    FASTAPI_RUNTIME_AVAILABLE = False
    FASTAPI_RUNTIME_IMPORT_ERROR = str(exc)
    FastAPI = None
    TestClient = None
    images = None
    settings = None
    LocalFileStore = None


@unittest.skipUnless(
    FASTAPI_RUNTIME_AVAILABLE,
    f"FastAPI runtime dependencies unavailable: {FASTAPI_RUNTIME_IMPORT_ERROR}",
)
class ImagesHttpTests(unittest.TestCase):
    def test_resolve_image_model_uses_env_default(self) -> None:
        from unittest.mock import patch

        with patch.object(settings, "IMAGE_GENERATION_MODEL", "qwen3.9-max-preview"):
            self.assertEqual(images._resolve_image_model(None), "qwen3.9-max-preview")
            self.assertEqual(images._resolve_image_model(""), "qwen3.9-max-preview")
            self.assertEqual(images._resolve_image_model("dall-e-3"), "qwen3.9-max-preview")
            self.assertEqual(images._resolve_image_model("qwen3.8-max-preview"), "qwen3.8-max-preview")

    def test_extract_image_urls_from_function_tool_result_extra(self) -> None:
        """官网 image_gen：image_list 与 tool_result 是同图两条 CDN 路径，只取 image_list。"""
        event = {
            "type": "delta",
            "phase": "image_gen_tool",
            "content": "",
            "status": "finished",
            "extra": {
                "tool_result": [
                    {"image": "https://cdn.qwenlm.ai/output/u/t2i/c/a.png?key=k1"},
                ],
                "image_list": [
                    {"image": "https://cdn.qwenlm.ai/output/u/image_gen/r/b.png?key=k2"},
                ],
            },
        }
        urls = images._extract_image_urls_from_events([event])
        self.assertEqual(
            urls,
            ["https://cdn.qwenlm.ai/output/u/image_gen/r/b.png?key=k2"],
        )

    def test_extract_image_urls_falls_back_to_tool_result_when_no_image_list(self) -> None:
        event = {
            "type": "delta",
            "phase": "image_gen_tool",
            "content": "",
            "status": "finished",
            "extra": {
                "tool_result": [
                    {"image": "https://cdn.qwenlm.ai/output/u/t2i/c/a.png?key=k1"},
                ],
            },
        }
        urls = images._extract_image_urls_from_events([event])
        self.assertEqual(urls, ["https://cdn.qwenlm.ai/output/u/t2i/c/a.png?key=k1"])

    def test_create_image_rehosts_cdn_url_to_local_content(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)
        png_bytes = b"\x89PNG\r\n\x1a\nfake-image"

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, preferred_account=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            yield {
                "type": "event",
                "event": {
                    "type": "delta",
                    "phase": "image_gen_tool",
                    "content": "",
                    "status": "finished",
                    "extra": {
                        "image_list": [
                            {"image": "https://cdn.qwenlm.ai/output/u/image_gen/r/1.png?key=abc"},
                        ],
                    },
                },
            }
            yield {
                "type": "event",
                "event": {
                    "type": "delta",
                    "phase": "answer",
                    "content": "已经为你画好了",
                    "status": "finished",
                },
            }

        file_store = SimpleNamespace(
            save_bytes=AsyncMock(
                return_value={
                    "id": "fileid123",
                    "path": "/tmp/fileid123.png",
                    "filename": "generated.png",
                    "content_type": "image/png",
                    "size": len(png_bytes),
                    "created_at": time.time(),
                    "purpose": "generated_image",
                }
            )
        )

        app = FastAPI()
        app.include_router(images.router)
        app.state.file_store = file_store
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            download_url=AsyncMock(return_value=(png_bytes, "image/png")),
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.8-max-preview"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(
            payload["data"][0]["url"].endswith("/v1/images/content/fileid123"),
            msg=payload["data"][0]["url"],
        )
        self.assertNotIn("cdn.qwenlm.ai", payload["data"][0]["url"])
        app.state.qwen_client.download_url.assert_awaited()
        file_store.save_bytes.assert_awaited()
        save_args = file_store.save_bytes.await_args
        purpose = save_args.kwargs.get("purpose")
        if purpose is None and save_args.args and len(save_args.args) >= 4:
            purpose = save_args.args[3]
        self.assertEqual(purpose, "generated_image")
        app.state.qwen_client.account_pool.release.assert_called_once_with(acc)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)

    def test_create_image_n_two_runs_upstream_twice(self) -> None:
        """n>1 时上游一次 image_gen 通常只出一张，需循环请求凑满 n。"""
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=0)
        call_count = {"n": 0}
        png_a = b"\x89PNG\r\n\x1a\nimage-A"
        png_b = b"\x89PNG\r\n\x1a\nimage-B-different"

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, preferred_account=None):
            call_count["n"] += 1
            idx = call_count["n"]
            acc.inflight += 1
            yield {"type": "meta", "acc": acc, "chat_id": f"chat-{idx}"}
            yield {
                "type": "event",
                "event": {
                    "type": "delta",
                    "phase": "image_gen_tool",
                    "content": "",
                    "status": "finished",
                    "extra": {
                        "image_list": [
                            {"image": f"https://cdn.qwenlm.ai/output/u/image_gen/r/{idx}.png?key=k{idx}"},
                        ],
                    },
                },
            }

        async def fake_download(url, account=None, timeout=None):
            if url.endswith("1.png?key=k1") or "/1.png" in url:
                return png_a, "image/png"
            return png_b, "image/png"

        save_ids = {"n": 0}

        async def fake_save_bytes(filename, content_type, raw, purpose, owner_token=None):
            save_ids["n"] += 1
            fid = f"file{save_ids['n']}"
            return {
                "id": fid,
                "path": f"/tmp/{fid}.png",
                "filename": filename,
                "content_type": content_type,
                "size": len(raw),
                "created_at": time.time(),
                "purpose": purpose,
            }

        app = FastAPI()
        app.include_router(images.router)
        app.state.file_store = SimpleNamespace(save_bytes=AsyncMock(side_effect=fake_save_bytes))
        release = Mock(side_effect=lambda account: setattr(account, "inflight", max(0, getattr(account, "inflight", 0) - 1)))
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            download_url=AsyncMock(side_effect=fake_download),
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=release),
        )

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成两张图", "n": 2, "model": "qwen3.8-max-preview"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["data"]), 2)
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(app.state.qwen_client.delete_chat.await_count, 2)
        self.assertEqual(release.call_count, 2)

    def test_create_image_does_not_require_list_chats(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)
        png_bytes = b"\x89PNG\r\n\x1a\nfake"

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, preferred_account=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            yield {
                "type": "event",
                "event": {
                    "choices": [
                        {
                            "delta": {
                                "content": "![result](https://cdn.qwenlm.ai/image-1.png)"
                            }
                        }
                    ]
                },
            }

        file_store = SimpleNamespace(
            save_bytes=AsyncMock(
                return_value={
                    "id": "legacy1",
                    "path": "/tmp/legacy1.png",
                    "filename": "generated.png",
                    "content_type": "image/png",
                    "size": len(png_bytes),
                    "created_at": time.time(),
                    "purpose": "generated_image",
                }
            )
        )

        app = FastAPI()
        app.include_router(images.router)
        app.state.file_store = file_store
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            download_url=AsyncMock(return_value=(png_bytes, "image/png")),
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.8-max-preview"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["data"][0]["url"].endswith("/v1/images/content/legacy1"))
        app.state.qwen_client.account_pool.release.assert_called_once_with(acc)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)

    def test_create_image_uses_current_chat_fallback_when_stream_payload_has_no_url(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)
        png_bytes = b"\x89PNG\r\n\x1a\nfallback"

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, preferred_account=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            yield {"type": "event", "event": {"choices": [{"delta": {"content": "image ready"}}]}}

        current_chat = {
            "id": "chat-1",
            "title": "api_image",
            "messages": [{"content": "![fallback](https://cdn.qwenlm.ai/fallback-image.png)"}],
        }
        file_store = SimpleNamespace(
            save_bytes=AsyncMock(
                return_value={
                    "id": "fallback1",
                    "path": "/tmp/fallback1.png",
                    "filename": "generated.png",
                    "content_type": "image/png",
                    "size": len(png_bytes),
                    "created_at": time.time(),
                    "purpose": "generated_image",
                }
            )
        )
        app = FastAPI()
        app.include_router(images.router)
        app.state.file_store = file_store
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            list_chats=AsyncMock(return_value=[current_chat]),
            download_url=AsyncMock(return_value=(png_bytes, "image/png")),
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )
        app.state.qwen_client.account_pool.release = Mock(side_effect=lambda account: setattr(account, "inflight", max(0, getattr(account, "inflight", 0) - 1)))

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.8-max-preview"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["data"][0]["url"].endswith("/v1/images/content/fallback1"))
        app.state.qwen_client.list_chats.assert_awaited_once_with("token-1", limit=20, account=acc)
        app.state.qwen_client.account_pool.release.assert_called_once_with(acc)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)

    def test_create_image_does_not_double_release_after_stream_failure(self) -> None:
        acc = SimpleNamespace(token="token-1", email="user@example.com", inflight=1)

        async def fake_stream_events_with_retry(model, content, has_custom_tools=False, files=None, preferred_account=None):
            yield {"type": "meta", "acc": acc, "chat_id": "chat-1"}
            app.state.qwen_client.account_pool.release(acc)
            raise RuntimeError("upstream failed")

        app = FastAPI()
        app.include_router(images.router)
        app.state.file_store = SimpleNamespace(save_bytes=AsyncMock())
        app.state.qwen_client = SimpleNamespace(
            chat_stream_events_with_retry=fake_stream_events_with_retry,
            download_url=AsyncMock(),
            delete_chat=AsyncMock(),
            account_pool=SimpleNamespace(release=Mock()),
        )
        app.state.qwen_client.account_pool.release = Mock(side_effect=lambda account: setattr(account, "inflight", max(0, getattr(account, "inflight", 0) - 1)))

        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.ADMIN_KEY}"},
            json={"prompt": "生成一张图", "n": 1, "model": "qwen3.8-max-preview"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(app.state.qwen_client.account_pool.release.call_count, 1)
        app.state.qwen_client.delete_chat.assert_awaited_once_with("token-1", "chat-1", account=acc)

    def test_get_image_content_serves_rehosted_bytes(self) -> None:
        import asyncio
        import tempfile

        png_bytes = b"\x89PNG\r\n\x1a\ncontent"

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)

            async def _seed():
                return await store.save_bytes(
                    "cat.png",
                    "image/png",
                    png_bytes,
                    "generated_image",
                )

            meta = asyncio.run(_seed())

            app = FastAPI()
            app.include_router(images.router)
            app.state.file_store = store

            client = TestClient(app)
            response = client.get(f"/v1/images/content/{meta['id']}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, png_bytes)
            self.assertIn("image/png", response.headers.get("content-type", ""))

    def test_cleanup_expired_can_filter_by_purpose(self) -> None:
        import asyncio
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)

            async def _run():
                old = await store.save_bytes("old.png", "image/png", b"old", "generated_image")
                keep_ctx = await store.save_bytes("ctx.txt", "text/plain", b"ctx", "context")
                fresh_img = await store.save_bytes("fresh.png", "image/png", b"fresh", "generated_image")
                # 把 old generated 与 context 时间戳拨到过去
                store._metadata[old["id"]]["created_at"] = time.time() - 7200
                store._metadata[keep_ctx["id"]]["created_at"] = time.time() - 7200
                await store.save()

                # 通用清理跳过 generated_image → 只清 context
                removed_ctx = await store.cleanup_expired(3600, exclude_purpose="generated_image")
                self.assertEqual(removed_ctx, 1)
                self.assertIsNone(await store.get(keep_ctx["id"]))
                self.assertIsNotNone(await store.get(old["id"]))

                # 生图独立 TTL 清理
                removed = await store.cleanup_expired(3600, purpose="generated_image")
                self.assertEqual(removed, 1)
                self.assertIsNone(await store.get(old["id"]))
                self.assertIsNotNone(await store.get(fresh_img["id"]))
                self.assertFalse(Path(old["path"]).exists())
                self.assertTrue(Path(fresh_img["path"]).exists())

            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
