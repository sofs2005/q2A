"""媒体库管理接口（后台）— 聚合展示 / 删除生成物（图片 + 视频）。

数据来源是 `LocalFileStore` 的元数据（purpose 属于 MEDIA_PURPOSES），
对外 URL 复用生成物自身的同源内容接口：
- 图片：/v1/images/content/{id}
- 视频：/v1/videos/content/{id}

量级很小（单机本地文件），过滤/分页直接在内存里做，不引入索引。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from backend.api.admin import verify_admin
from backend.services.media_purposes import (
    GENERATED_IMAGE_PURPOSE,
    GENERATED_VIDEO_PURPOSE,
    MEDIA_PURPOSES,
)

log = logging.getLogger("qwen2api.media")

# 媒体库整体要求管理凭证（与管理台一致）
router = APIRouter(dependencies=[Depends(verify_admin)])

# kind 别名归一：images→image / videos→video，其它一律 all
_KIND_IMAGE = "image"
_KIND_VIDEO = "video"
_KIND_ALL = "all"

# purpose → 对外 kind
_PURPOSE_TO_KIND = {
    GENERATED_IMAGE_PURPOSE: _KIND_IMAGE,
    GENERATED_VIDEO_PURPOSE: _KIND_VIDEO,
}

# kind → 内容 URL 前缀
_CONTENT_URL_PREFIX = {
    _KIND_IMAGE: "/v1/images/content",
    _KIND_VIDEO: "/v1/videos/content",
}

# 分页护栏
_DEFAULT_LIMIT = 24
_MIN_LIMIT = 1
_MAX_LIMIT = 100

# 排序取值
_SORT_NEWEST = "newest"
_SORT_OLDEST = "oldest"
_SORT_LARGEST = "largest"


class MediaDeleteBatchRequest(BaseModel):
    """批量删除请求体：{"ids": ["...", "..."]}"""

    ids: list[str] = []


def _normalize_kind(value: str | None) -> str:
    """把 query 参数 kind 归一成 image / video / all。"""
    candidate = str(value or "").strip().lower()
    if candidate in {"image", "images"}:
        return _KIND_IMAGE
    if candidate in {"video", "videos"}:
        return _KIND_VIDEO
    return _KIND_ALL


def _purposes_for_kind(kind: str) -> list[str]:
    """kind 对应的 purpose 过滤列表（all = 全部生成物用途）。"""
    if kind == _KIND_IMAGE:
        return [GENERATED_IMAGE_PURPOSE]
    if kind == _KIND_VIDEO:
        return [GENERATED_VIDEO_PURPOSE]
    return list(MEDIA_PURPOSES)


def _kind_of(meta: dict[str, Any]) -> str | None:
    """元数据 → kind；非生成物返回 None（媒体库不展示）。"""
    return _PURPOSE_TO_KIND.get(str(meta.get("purpose") or ""))


def _matches_query(meta: dict[str, Any], needle: str) -> bool:
    """大小写不敏感的子串匹配：extra.prompt / filename / extra.ratio。"""
    if not needle:
        return True
    extra = meta.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    haystacks = (
        extra.get("prompt"),
        meta.get("filename"),
        extra.get("ratio"),
    )
    for value in haystacks:
        if value is None:
            continue
        if needle in str(value).lower():
            return True
    return False


def _build_item(meta: dict[str, Any], kind: str) -> dict[str, Any]:
    """元数据 → 媒体库列表项（字段缺失时给出稳定的空值）。"""
    extra = meta.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    file_id = str(meta.get("id") or "")
    return {
        "id": file_id,
        "kind": kind,
        "url": f"{_CONTENT_URL_PREFIX[kind]}/{file_id}",
        "filename": meta.get("filename") or "",
        "content_type": meta.get("content_type") or "",
        "size": int(meta.get("size") or 0),
        "created_at": meta.get("created_at") or 0,
        "prompt": extra.get("prompt") or "",
        "ratio": extra.get("ratio"),
        "duration": extra.get("duration"),
        "model": extra.get("model"),
    }


def _build_stats(all_media: list[dict[str, Any]]) -> dict[str, Any]:
    """统计信息：基于 kind 过滤前的集合（其它筛选条件已生效）。

    这样「图片 x / 视频 y / 占用 z」始终描述用户当前看到的这批数据；
    只有类型维度不参与统计，否则选了「图片」就永远看不到视频数量。
    """
    image_count = 0
    video_count = 0
    total_bytes = 0
    for meta in all_media:
        kind = _kind_of(meta)
        if kind == _KIND_IMAGE:
            image_count += 1
        elif kind == _KIND_VIDEO:
            video_count += 1
        total_bytes += int(meta.get("size") or 0)
    return {"image": image_count, "video": video_count, "bytes": total_bytes}


def _normalize_sort(value: str | None) -> str:
    """排序归一：newest（默认）/ oldest / largest，未知值退回 newest。"""
    candidate = str(value or "").strip().lower()
    if candidate in {_SORT_OLDEST, _SORT_LARGEST}:
        return candidate
    return _SORT_NEWEST


def _passes_filters(meta: dict[str, Any], *, needle: str, cutoff: float | None, min_size: int) -> bool:
    """与 kind 无关的筛选：关键字 / 时间窗 / 体积下限。

    统一放在这里，stats 与 items 共用同一套判断，避免两边口径漂移。
    """
    if not _matches_query(meta, needle):
        return False
    if cutoff is not None and float(meta.get("created_at") or 0) < cutoff:
        return False
    if min_size > 0 and int(meta.get("size") or 0) < min_size:
        return False
    return True


def _sort_items(items: list[dict[str, Any]], sort: str) -> None:
    """就地排序（列表项已带 created_at / size）。"""
    if sort == _SORT_OLDEST:
        items.sort(key=lambda item: item.get("created_at") or 0)
    elif sort == _SORT_LARGEST:
        items.sort(key=lambda item: item.get("size") or 0, reverse=True)
    else:
        items.sort(key=lambda item: item.get("created_at") or 0, reverse=True)


def _get_file_store(request: Request):
    """取应用级 LocalFileStore；未初始化时返回 None。"""
    return getattr(request.app.state, "file_store", None)


@router.get("")
async def list_media(
    request: Request,
    kind: str = Query(_KIND_ALL),
    q: str = Query(""),
    limit: int = Query(_DEFAULT_LIMIT),
    offset: int = Query(0),
    since_days: int = Query(0),
    min_size: int = Query(0),
    sort: str = Query(_SORT_NEWEST),
):
    """列出媒体库条目（图片/视频）。

    筛选维度：类型 kind、关键字 q、时间窗 since_days（天，0=不限）、
    体积下限 min_size（字节，0=不限）；排序 sort = newest / oldest / largest。
    """
    file_store = _get_file_store(request)
    if file_store is None:
        raise HTTPException(status_code=503, detail="File store unavailable")

    normalized_kind = _normalize_kind(kind)
    needle = str(q or "").strip().lower()
    # limit 缺省 24，显式传 0/负数时按 1 处理（不能当成"未传"）
    raw_limit = _DEFAULT_LIMIT if limit is None else int(limit)
    safe_limit = min(max(raw_limit, _MIN_LIMIT), _MAX_LIMIT)
    raw_offset = 0 if offset is None else int(offset)
    safe_offset = max(raw_offset, 0)

    safe_since = max(int(since_days or 0), 0)
    cutoff = time.time() - safe_since * 86400 if safe_since > 0 else None
    safe_min_size = max(int(min_size or 0), 0)
    normalized_sort = _normalize_sort(sort)

    all_media = await file_store.list_metadata(purposes=list(MEDIA_PURPOSES))
    # 先过与类型无关的筛选：stats 用这批（忽略 kind），items 再叠类型条件
    base = [
        meta
        for meta in all_media
        if _passes_filters(meta, needle=needle, cutoff=cutoff, min_size=safe_min_size)
    ]
    stats = _build_stats(base)

    items: list[dict[str, Any]] = []
    for meta in base:
        item_kind = _kind_of(meta)
        if item_kind is None:
            continue
        if normalized_kind != _KIND_ALL and item_kind != normalized_kind:
            continue
        items.append(_build_item(meta, item_kind))

    _sort_items(items, normalized_sort)

    total = len(items)
    page = items[safe_offset:safe_offset + safe_limit]
    return {
        "items": page,
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "has_more": safe_offset + len(page) < total,
        "stats": stats,
    }


@router.delete("/{file_id}")
async def delete_media(request: Request, file_id: str):
    """删除单个生成物（元数据 + 磁盘文件）。"""
    file_store = _get_file_store(request)
    if file_store is None:
        raise HTTPException(status_code=503, detail="File store unavailable")
    meta = await file_store.get(file_id)
    if meta is None or meta.get("purpose") not in MEDIA_PURPOSES:
        raise HTTPException(status_code=404, detail="Media not found")
    await file_store.delete(file_id)
    log.info("[Media] deleted id=%s purpose=%s", file_id, meta.get("purpose"))
    return {"deleted": True, "id": file_id}


@router.post("/delete")
async def delete_media_batch(request: Request, payload: MediaDeleteBatchRequest):
    """批量删除生成物；返回成功/缺失两组 id。"""
    file_store = _get_file_store(request)
    if file_store is None:
        raise HTTPException(status_code=503, detail="File store unavailable")
    if not payload.ids:
        raise HTTPException(status_code=400, detail="ids is required")

    deleted: list[str] = []
    missing: list[str] = []
    for file_id in payload.ids:
        meta = await file_store.get(file_id)
        if meta is None or meta.get("purpose") not in MEDIA_PURPOSES:
            missing.append(file_id)
            continue
        await file_store.delete(file_id)
        deleted.append(file_id)

    log.info("[Media] batch delete deleted=%s missing=%s", len(deleted), len(missing))
    return {"deleted": len(deleted), "ids": deleted, "missing": missing}
