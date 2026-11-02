"""媒体库后台接口 + 视频回源落盘的单元测试。

覆盖点：
- GET /api/admin/media：kind 过滤、关键字搜索、分页与 stats、鉴权
- DELETE /api/admin/media/{id}、POST /api/admin/media/delete：单删/批删
- LocalFileStore.cleanup_expired 的 exclude_purpose / purpose 语义
- videos._rehost_video：下载成功落盘、下载异常退回原 URL、体积超限退回原 URL
"""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

FASTAPI_RUNTIME_AVAILABLE = True
FASTAPI_RUNTIME_IMPORT_ERROR = ""

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api import media, videos
    from backend.core.config import settings
    from backend.services.file_store import LocalFileStore
    from backend.services.media_purposes import (
        GENERATED_IMAGE_PURPOSE,
        GENERATED_VIDEO_PURPOSE,
        MEDIA_PURPOSES,
    )
except (ModuleNotFoundError, ImportError) as exc:
    FASTAPI_RUNTIME_AVAILABLE = False
    FASTAPI_RUNTIME_IMPORT_ERROR = str(exc)
    FastAPI = None
    TestClient = None
    media = None
    videos = None
    settings = None
    LocalFileStore = None
    GENERATED_IMAGE_PURPOSE = "generated_image"
    GENERATED_VIDEO_PURPOSE = "generated_video"
    MEDIA_PURPOSES = (GENERATED_IMAGE_PURPOSE, GENERATED_VIDEO_PURPOSE)

IMAGE_A_BYTES = b"png-a"
IMAGE_B_BYTES = b"png-bb"
VIDEO_BYTES = b"mp4-video"
CONTEXT_BYTES = b"ctx"


@unittest.skipUnless(
    FASTAPI_RUNTIME_AVAILABLE,
    f"FastAPI runtime dependencies unavailable: {FASTAPI_RUNTIME_IMPORT_ERROR}",
)
class MediaLibraryHttpTests(unittest.TestCase):
    """媒体库路由（挂到最小 FastAPI 应用上，用真实 LocalFileStore）。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.store = LocalFileStore(self.tmpdir.name)
        self.headers = {"Authorization": f"Bearer {settings.ADMIN_KEY}"}

        # 造 2 张图 + 1 个视频 + 1 个上下文附件（后者不应出现在媒体库）
        asyncio.run(self._seed())

        app = FastAPI()
        app.include_router(media.router, prefix="/api/admin/media")
        app.state.file_store = self.store
        self.client = TestClient(app)

    async def _seed(self) -> None:
        self.image_a = await self.store.save_bytes(
            "cat.png",
            "image/png",
            IMAGE_A_BYTES,
            GENERATED_IMAGE_PURPOSE,
            extra={"prompt": "赛博朋克猫", "size": "16:9", "model": "qwen3.8-max"},
        )
        self.image_b = await self.store.save_bytes(
            "dog.png",
            "image/png",
            IMAGE_B_BYTES,
            GENERATED_IMAGE_PURPOSE,
            extra={"prompt": "sunset dog", "size": "1:1", "model": "qwen3.8-max"},
        )
        self.video = await self.store.save_bytes(
            "clip.mp4",
            "video/mp4",
            VIDEO_BYTES,
            GENERATED_VIDEO_PURPOSE,
            extra={"prompt": "海浪", "ratio": "16:9", "duration": 5, "model": "qwen3.6-plus"},
        )
        self.context_file = await self.store.save_bytes(
            "ctx.txt", "text/plain", CONTEXT_BYTES, "context"
        )

    # ------------------------------------------------------------------
    # 列表
    # ------------------------------------------------------------------

    def test_list_returns_all_media_with_stats(self) -> None:
        response = self.client.get("/api/admin/media", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["offset"], 0)
        self.assertEqual(payload["limit"], 24)
        self.assertFalse(payload["has_more"])
        self.assertEqual(
            payload["stats"],
            {
                "image": 2,
                "video": 1,
                "bytes": len(IMAGE_A_BYTES) + len(IMAGE_B_BYTES) + len(VIDEO_BYTES),
            },
        )
        ids = {item["id"] for item in payload["items"]}
        self.assertEqual(ids, {self.image_a["id"], self.image_b["id"], self.video["id"]})
        # 上下文附件不属于媒体库
        self.assertNotIn(self.context_file["id"], ids)
        # created_at 倒序
        created = [item["created_at"] for item in payload["items"]]
        self.assertEqual(created, sorted(created, reverse=True))

    def test_list_item_shape_and_content_urls(self) -> None:
        response = self.client.get("/api/admin/media", headers=self.headers)
        items = {item["id"]: item for item in response.json()["items"]}

        image_item = items[self.image_a["id"]]
        self.assertEqual(image_item["kind"], "image")
        self.assertEqual(image_item["url"], f"/v1/images/content/{self.image_a['id']}")
        self.assertEqual(image_item["prompt"], "赛博朋克猫")
        self.assertEqual(image_item["model"], "qwen3.8-max")
        self.assertEqual(image_item["size"], len(IMAGE_A_BYTES))
        self.assertIsNone(image_item["ratio"])
        self.assertIsNone(image_item["duration"])
        self.assertEqual(
            set(image_item.keys()),
            {
                "id", "kind", "url", "filename", "content_type",
                "size", "created_at", "prompt", "ratio", "duration", "model",
            },
        )

        video_item = items[self.video["id"]]
        self.assertEqual(video_item["kind"], "video")
        self.assertEqual(video_item["url"], f"/v1/videos/content/{self.video['id']}")
        self.assertEqual(video_item["ratio"], "16:9")
        self.assertEqual(video_item["duration"], 5)

    def test_list_filters_by_kind(self) -> None:
        image_payload = self.client.get(
            "/api/admin/media", params={"kind": "images"}, headers=self.headers
        ).json()
        self.assertEqual(image_payload["total"], 2)
        self.assertTrue(all(item["kind"] == "image" for item in image_payload["items"]))
        # stats 基于 kind 过滤前的全量
        self.assertEqual(image_payload["stats"]["video"], 1)

        video_payload = self.client.get(
            "/api/admin/media", params={"kind": "video"}, headers=self.headers
        ).json()
        self.assertEqual(video_payload["total"], 1)
        self.assertEqual(video_payload["items"][0]["id"], self.video["id"])

        # 未知 kind 归一为 all
        all_payload = self.client.get(
            "/api/admin/media", params={"kind": "whatever"}, headers=self.headers
        ).json()
        self.assertEqual(all_payload["total"], 3)

    def test_list_search_matches_prompt_case_insensitive(self) -> None:
        chinese = self.client.get(
            "/api/admin/media", params={"q": "赛博"}, headers=self.headers
        ).json()
        self.assertEqual(chinese["total"], 1)
        self.assertEqual(chinese["items"][0]["id"], self.image_a["id"])

        upper = self.client.get(
            "/api/admin/media", params={"q": "SUNSET"}, headers=self.headers
        ).json()
        self.assertEqual(upper["total"], 1)
        self.assertEqual(upper["items"][0]["id"], self.image_b["id"])

        # 只有视频的 extra.ratio 参与匹配（图片的 extra 里是 size，不是 ratio）
        ratio_hit = self.client.get(
            "/api/admin/media", params={"q": "16:9"}, headers=self.headers
        ).json()
        self.assertEqual([item["id"] for item in ratio_hit["items"]], [self.video["id"]])

        nothing = self.client.get(
            "/api/admin/media", params={"q": "不存在的关键字"}, headers=self.headers
        ).json()
        self.assertEqual(nothing["total"], 0)
        self.assertEqual(nothing["items"], [])
        self.assertFalse(nothing["has_more"])

    def test_list_pagination(self) -> None:
        first = self.client.get(
            "/api/admin/media", params={"limit": 1, "offset": 0}, headers=self.headers
        ).json()
        self.assertEqual(len(first["items"]), 1)
        self.assertEqual(first["total"], 3)
        self.assertTrue(first["has_more"])

        last = self.client.get(
            "/api/admin/media", params={"limit": 1, "offset": 2}, headers=self.headers
        ).json()
        self.assertEqual(len(last["items"]), 1)
        self.assertFalse(last["has_more"])

        # limit clamp 1..100、offset 不小于 0
        clamped_high = self.client.get(
            "/api/admin/media", params={"limit": 999}, headers=self.headers
        ).json()
        self.assertEqual(clamped_high["limit"], 100)

        clamped_low = self.client.get(
            "/api/admin/media", params={"limit": 0, "offset": -5}, headers=self.headers
        ).json()
        self.assertEqual(clamped_low["limit"], 1)
        self.assertEqual(clamped_low["offset"], 0)

    def test_list_requires_admin_auth(self) -> None:
        anonymous = self.client.get("/api/admin/media")
        self.assertIn(anonymous.status_code, (401, 403))

        wrong = self.client.get(
            "/api/admin/media", headers={"Authorization": "Bearer not-the-admin-key"}
        )
        self.assertIn(wrong.status_code, (401, 403))

    # ------------------------------------------------------------------
    # 新增筛选维度：时间窗 / 体积下限 / 排序
    # ------------------------------------------------------------------

    def _backdate(self, meta: dict, days: float) -> None:
        """把某条元数据的 created_at 拨到 N 天前（模拟历史生成物）。"""
        self.store._metadata[meta["id"]]["created_at"] = time.time() - days * 86400

    def test_list_filters_by_time_window(self) -> None:
        self._backdate(self.image_a, 10)   # 10 天前
        self._backdate(self.image_b, 100)  # 100 天前
        # video 保持「刚刚」

        today = self.client.get(
            "/api/admin/media", params={"since_days": 1}, headers=self.headers
        ).json()
        self.assertEqual([item["id"] for item in today["items"]], [self.video["id"]])
        # stats 同样只统计时间窗内的数据（除类型维度外都生效）
        self.assertEqual(today["stats"], {"image": 0, "video": 1, "bytes": len(VIDEO_BYTES)})

        week = self.client.get(
            "/api/admin/media", params={"since_days": 7}, headers=self.headers
        ).json()
        self.assertEqual(week["total"], 1)

        month = self.client.get(
            "/api/admin/media", params={"since_days": 30}, headers=self.headers
        ).json()
        self.assertEqual({item["id"] for item in month["items"]}, {self.video["id"], self.image_a["id"]})

        all_time = self.client.get(
            "/api/admin/media", params={"since_days": 0}, headers=self.headers
        ).json()
        self.assertEqual(all_time["total"], 3)

        # 负数按 0（不限）处理，不能把结果清空
        negative = self.client.get(
            "/api/admin/media", params={"since_days": -5}, headers=self.headers
        ).json()
        self.assertEqual(negative["total"], 3)

    def test_list_filters_by_min_size(self) -> None:
        # 体积：image_a=5B < image_b=6B < video=9B
        big = self.client.get(
            "/api/admin/media", params={"min_size": 6}, headers=self.headers
        ).json()
        self.assertEqual({item["id"] for item in big["items"]}, {self.image_b["id"], self.video["id"]})
        self.assertEqual(big["total"], 2)
        self.assertEqual(big["stats"]["bytes"], len(IMAGE_B_BYTES) + len(VIDEO_BYTES))

        huge = self.client.get(
            "/api/admin/media", params={"min_size": 10_000}, headers=self.headers
        ).json()
        self.assertEqual(huge["items"], [])
        self.assertEqual(huge["stats"], {"image": 0, "video": 0, "bytes": 0})

        # 0 / 负数 = 不限
        for value in (0, -1):
            with self.subTest(min_size=value):
                payload = self.client.get(
                    "/api/admin/media", params={"min_size": value}, headers=self.headers
                ).json()
                self.assertEqual(payload["total"], 3)

    def test_list_sort_modes(self) -> None:
        # 造出明确的时间序：image_a 最旧 → video 最新
        self._backdate(self.image_a, 3)
        self._backdate(self.image_b, 2)
        self._backdate(self.video, 1)

        newest = self.client.get(
            "/api/admin/media", params={"sort": "newest"}, headers=self.headers
        ).json()
        self.assertEqual(
            [item["id"] for item in newest["items"]],
            [self.video["id"], self.image_b["id"], self.image_a["id"]],
        )

        oldest = self.client.get(
            "/api/admin/media", params={"sort": "oldest"}, headers=self.headers
        ).json()
        self.assertEqual(
            [item["id"] for item in oldest["items"]],
            [self.image_a["id"], self.image_b["id"], self.video["id"]],
        )

        largest = self.client.get(
            "/api/admin/media", params={"sort": "largest"}, headers=self.headers
        ).json()
        self.assertEqual(
            [item["id"] for item in largest["items"]],
            [self.video["id"], self.image_b["id"], self.image_a["id"]],
        )

        # 未知排序值退回 newest，而不是报错或返回空
        unknown = self.client.get(
            "/api/admin/media", params={"sort": "banana"}, headers=self.headers
        ).json()
        self.assertEqual([item["id"] for item in unknown["items"]],
                         [item["id"] for item in newest["items"]])

    def test_list_filters_combine_with_kind_and_query(self) -> None:
        self._backdate(self.image_b, 30)
        payload = self.client.get(
            "/api/admin/media",
            params={"kind": "image", "q": "cat", "since_days": 7, "min_size": 1},
            headers=self.headers,
        ).json()
        self.assertEqual([item["id"] for item in payload["items"]], [self.image_a["id"]])
        # stats 只忽略类型维度：关键字/时间窗照样生效（video 不匹配 "cat"）
        self.assertEqual(payload["stats"]["video"], 0)
        self.assertEqual(payload["stats"]["image"], 1)

        # 同一组条件放宽关键字后，video 回到 stats（证明类型维度确实被忽略）
        wider = self.client.get(
            "/api/admin/media",
            params={"kind": "image", "since_days": 7},
            headers=self.headers,
        ).json()
        self.assertEqual(wider["stats"]["video"], 1)
        self.assertEqual(wider["stats"]["image"], 1)

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def test_delete_single_removes_metadata_and_file(self) -> None:
        path = Path(self.video["path"])
        self.assertTrue(path.exists())

        response = self.client.delete(
            f"/api/admin/media/{self.video['id']}", headers=self.headers
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": True, "id": self.video["id"]})
        self.assertIsNone(asyncio.run(self.store.get(self.video["id"])))
        self.assertFalse(path.exists())

    def test_delete_single_rejects_missing_and_non_media(self) -> None:
        missing = self.client.delete("/api/admin/media/not-exist", headers=self.headers)
        self.assertEqual(missing.status_code, 404)
        self.assertIn("detail", missing.json())

        context_path = Path(self.context_file["path"])
        rejected = self.client.delete(
            f"/api/admin/media/{self.context_file['id']}", headers=self.headers
        )
        self.assertEqual(rejected.status_code, 404)
        self.assertIsNotNone(asyncio.run(self.store.get(self.context_file["id"])))
        self.assertTrue(context_path.exists())

    def test_delete_batch_reports_deleted_and_missing(self) -> None:
        image_path = Path(self.image_a["path"])

        response = self.client.post(
            "/api/admin/media/delete",
            json={"ids": [self.image_a["id"], "not-exist", self.context_file["id"]]},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["deleted"], 1)
        self.assertEqual(payload["ids"], [self.image_a["id"]])
        self.assertEqual(payload["missing"], ["not-exist", self.context_file["id"]])
        self.assertIsNone(asyncio.run(self.store.get(self.image_a["id"])))
        self.assertFalse(image_path.exists())
        # 未删的仍然在
        self.assertIsNotNone(asyncio.run(self.store.get(self.image_b["id"])))

    def test_delete_batch_rejects_empty_ids(self) -> None:
        response = self.client.post(
            "/api/admin/media/delete", json={"ids": []}, headers=self.headers
        )
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------
    # 清理语义
    # ------------------------------------------------------------------

    def test_cleanup_excluding_media_purposes_keeps_generated(self) -> None:
        async def _run() -> int:
            old_ts = time.time() - 7200
            self.store._metadata[self.image_a["id"]]["created_at"] = old_ts
            self.store._metadata[self.video["id"]]["created_at"] = old_ts
            self.store._metadata[self.context_file["id"]]["created_at"] = old_ts
            await self.store.save()
            return await self.store.cleanup_expired(
                3600, exclude_purpose=list(MEDIA_PURPOSES)
            )

        removed = asyncio.run(_run())

        self.assertEqual(removed, 1)
        self.assertIsNone(asyncio.run(self.store.get(self.context_file["id"])))
        self.assertIsNotNone(asyncio.run(self.store.get(self.image_a["id"])))
        self.assertIsNotNone(asyncio.run(self.store.get(self.video["id"])))

    def test_cleanup_by_purpose_removes_only_expired_media(self) -> None:
        async def _run() -> int:
            self.store._metadata[self.video["id"]]["created_at"] = time.time() - 7200
            await self.store.save()
            return await self.store.cleanup_expired(
                3600, purpose=GENERATED_VIDEO_PURPOSE
            )

        removed = asyncio.run(_run())

        self.assertEqual(removed, 1)
        self.assertIsNone(asyncio.run(self.store.get(self.video["id"])))
        self.assertIsNotNone(asyncio.run(self.store.get(self.image_a["id"])))
        self.assertFalse(Path(self.video["path"]).exists())


@unittest.skipUnless(
    FASTAPI_RUNTIME_AVAILABLE,
    f"FastAPI runtime dependencies unavailable: {FASTAPI_RUNTIME_IMPORT_ERROR}",
)
class VideoRehostTests(unittest.TestCase):
    """videos._rehost_video：回源成功落盘、失败退回 CDN 直链。"""

    def test_rehost_video_downloads_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)

            async def fake_download(url, account=None, timeout=None):
                return VIDEO_BYTES, "video/mp4"

            client = SimpleNamespace(download_url=fake_download)

            url = asyncio.run(
                videos._rehost_video(
                    store,
                    client,
                    None,
                    "https://cdn.qwenlm.ai/output/u/clip.mp4?key=k",
                    prompt="海浪",
                    ratio="16:9",
                    duration=5,
                    model="qwen3.6-plus",
                )
            )

            self.assertTrue(url.startswith("/v1/videos/content/"))
            file_id = url.rsplit("/", 1)[-1]
            meta = asyncio.run(store.get(file_id))
            self.assertIsNotNone(meta)
            self.assertEqual(meta["purpose"], GENERATED_VIDEO_PURPOSE)
            self.assertEqual(meta["content_type"], "video/mp4")
            self.assertEqual(
                meta["extra"],
                {"prompt": "海浪", "ratio": "16:9", "duration": 5, "model": "qwen3.6-plus"},
            )
            self.assertEqual(Path(meta["path"]).read_bytes(), VIDEO_BYTES)

    def test_rehost_video_download_error_returns_original_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)

            async def broken_download(url, account=None, timeout=None):
                raise RuntimeError("cdn exploded")

            client = SimpleNamespace(download_url=broken_download)
            original = "https://cdn.qwenlm.ai/output/u/clip.mp4?key=k"

            url = asyncio.run(
                videos._rehost_video(
                    store, client, None, original,
                    prompt="海浪", ratio="16:9", duration=5,
                )
            )

            self.assertEqual(url, original)
            self.assertEqual(store._metadata, {})

    def test_rehost_video_too_large_returns_original_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)

            async def big_download(url, account=None, timeout=None):
                return b"x" * 64, "video/mp4"

            client = SimpleNamespace(download_url=big_download)
            original = "https://cdn.qwenlm.ai/output/u/clip.mp4?key=k"

            with patch.object(settings, "GENERATED_VIDEO_MAX_BYTES", 8):
                url = asyncio.run(
                    videos._rehost_video(
                        store, client, None, original,
                        prompt="海浪", ratio="16:9", duration=5,
                    )
                )

            self.assertEqual(url, original)
            self.assertEqual(store._metadata, {})

    def test_rehost_video_without_store_or_downloader_is_passthrough(self) -> None:
        original = "https://cdn.qwenlm.ai/output/u/clip.mp4?key=k"

        self.assertEqual(
            asyncio.run(
                videos._rehost_video(
                    None, SimpleNamespace(), None, original,
                    prompt="海浪", ratio="16:9", duration=5,
                )
            ),
            original,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)
            # client 没有 download_url 能力
            self.assertEqual(
                asyncio.run(
                    videos._rehost_video(
                        store, SimpleNamespace(), None, original,
                        prompt="海浪", ratio="16:9", duration=5,
                    )
                ),
                original,
            )

    def test_video_content_route_serves_inline_with_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalFileStore(tmp)

            async def _seed() -> dict:
                return await store.save_bytes(
                    "clip.mp4", "video/mp4", VIDEO_BYTES, GENERATED_VIDEO_PURPOSE,
                    extra={"prompt": "海浪", "ratio": "16:9", "duration": 5},
                )

            meta = asyncio.run(_seed())

            app = FastAPI()
            app.include_router(videos.router)
            app.state.file_store = store
            client = TestClient(app)

            response = client.get(f"/v1/videos/content/{meta['id']}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, VIDEO_BYTES)
            self.assertIn("video/mp4", response.headers.get("content-type", ""))
            # 不传 filename：不应出现 attachment 头，否则 <video> 会变成下载
            self.assertNotIn("attachment", response.headers.get("content-disposition", ""))

            ranged = client.get(
                f"/v1/videos/content/{meta['id']}", headers={"Range": "bytes=0-2"}
            )
            self.assertEqual(ranged.status_code, 206)
            self.assertEqual(ranged.content, VIDEO_BYTES[:3])

            # 非视频生成物（例如上下文附件）不通过该路由暴露
            async def _seed_context() -> dict:
                return await store.save_bytes("ctx.txt", "text/plain", b"ctx", "context")

            context_meta = asyncio.run(_seed_context())
            missing = client.get(f"/v1/videos/content/{context_meta['id']}")
            self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
