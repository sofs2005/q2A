from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any

from backend.services.image_thumbnail import remove_thumbnails

log = logging.getLogger("qwen2api.file_store")


def _as_purpose_set(value: str | list[str] | tuple[str, ...] | None) -> set[str] | None:
    """把 purpose 过滤参数归一成集合；None 表示不过滤。

    调用方历史上传单个字符串（如 exclude_purpose="generated_image"），
    现在也允许传列表/元组（如 exclude_purpose=list(MEDIA_PURPOSES)）。
    """
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


class LocalFileStore:
    def __init__(self, root_dir: str, metadata_db=None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db = metadata_db
        self._metadata: dict[str, dict[str, Any]] = {}

    async def load(self):
        if self.metadata_db is None:
            return
        data = await self.metadata_db.load()
        self._metadata = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    self._metadata[item["id"]] = item

    async def save(self):
        if self.metadata_db is None:
            return
        await self.metadata_db.save(list(self._metadata.values()))

    async def save_bytes(
        self,
        filename: str,
        content_type: str,
        raw: bytes,
        purpose: str,
        owner_token: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        file_id = uuid.uuid4().hex
        suffix = Path(filename).suffix or mimetypes.guess_extension(content_type or "") or ""
        safe_name = (Path(filename).stem or "file").replace(" ", "_")
        target = self.root_dir / purpose / f"{file_id}_{safe_name}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, raw)
        meta = {
            "id": file_id,
            "path": str(target),
            "filename": f"{safe_name}{suffix}",
            "content_type": content_type or "application/octet-stream",
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "created_at": time.time(),
            "owner_token": owner_token or "",
            "purpose": purpose,
            # 业务侧附加信息（如生图 prompt / 视频比例），媒体库直接读这里
            "extra": dict(extra or {}),
        }
        self._metadata[file_id] = meta
        await self.save()
        return meta

    async def save_text(self, filename: str, text: str, content_type: str = "text/plain", purpose: str = "context", owner_token: str | None = None) -> dict:
        raw = text.encode("utf-8")
        return await self.save_bytes(filename, content_type, raw, purpose, owner_token=owner_token)

    async def get(self, file_id: str) -> dict[str, Any] | None:
        if not self._metadata and self.metadata_db is not None:
            await self.load()
        return self._metadata.get(file_id)

    async def list_metadata(
        self,
        purposes: str | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """列出元数据（按用途过滤，按 created_at 倒序，返回拷贝）。

        - purposes: 单个用途字符串 / 用途列表或元组 / None（表示全部用途）
        - 返回的是元数据拷贝，调用方可安全修改，不会污染存储
        """
        if not self._metadata and self.metadata_db is not None:
            await self.load()
        allowed = _as_purpose_set(purposes)
        items: list[dict[str, Any]] = [
            dict(meta)
            for meta in self._metadata.values()
            if allowed is None or meta.get("purpose") in allowed
        ]
        # created_at 倒序：最新的生成物排在最前，媒体库直接顺序消费
        items.sort(key=lambda meta: meta.get("created_at", 0), reverse=True)
        return items

    async def delete(self, file_id: str) -> None:
        meta = await self.get(file_id)
        if meta and meta.get("path"):
            await self.delete_path(meta["path"])
        self._metadata.pop(file_id, None)
        await self.save()
        await self._delete_thumbnails(file_id)

    async def _delete_thumbnails(self, file_id: str) -> None:
        """清掉该生成物的缩略图缓存（生图内容的 `?w=` 派生文件）。

        缩略图不在元数据里（纯缓存），所以只能按 id 前缀扫目录；失败不影响删除结果。
        """
        try:
            await asyncio.to_thread(remove_thumbnails, self.root_dir, file_id)
        except Exception as exc:  # 缓存清理失败不该让业务删除失败
            log.debug("[FileStore] thumbnail cleanup failed id=%s error=%s", file_id, exc)

    async def delete_path(self, path: str) -> None:
        target = Path(path)
        try:
            await asyncio.to_thread(target.unlink)
        except FileNotFoundError:
            pass
        remove_id = None
        for file_id, meta in self._metadata.items():
            if meta.get("path") == str(target):
                remove_id = file_id
                break
        if remove_id:
            self._metadata.pop(remove_id, None)
            await self.save()

    async def cleanup_expired(
        self,
        ttl_seconds: int,
        purpose: str | None = None,
        exclude_purpose: str | list[str] | tuple[str, ...] | None = None,
    ) -> int:
        """删除超过 TTL 的本地文件。

        - purpose: 仅清理该用途（例如 generated_image）
        - exclude_purpose: 跳过这些用途（通用清理时避开生成物，生成物有独立 TTL）；
          支持单个字符串或列表/元组
        返回删除条数。
        """
        if ttl_seconds <= 0:
            return 0
        cutoff = time.time() - ttl_seconds
        excluded = _as_purpose_set(exclude_purpose)
        expired_ids: list[str] = []
        for file_id, meta in list(self._metadata.items()):
            meta_purpose = meta.get("purpose")
            if purpose is not None and meta_purpose != purpose:
                continue
            if excluded is not None and meta_purpose in excluded:
                continue
            if meta.get("created_at", 0) < cutoff:
                expired_ids.append(file_id)
        for file_id in expired_ids:
            meta = self._metadata.get(file_id, {})
            path = meta.get("path")
            if path:
                try:
                    await asyncio.to_thread(Path(path).unlink)
                except FileNotFoundError:
                    pass
            self._metadata.pop(file_id, None)
            await self._delete_thumbnails(file_id)
        if expired_ids:
            await self.save()
        return len(expired_ids)
