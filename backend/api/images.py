"""
图片生成接口 — 兼容 OpenAI /v1/images/generations 规范。

底层通过现有直连 HTTP 聊天能力触发千问“生成图像”模式，
不依赖浏览器运行时。

CDN 签名链接对普通浏览器常不可访问，因此生成成功后会回源下载并
转存为本地文件，对外返回同源 `/v1/images/content/{id}`。
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from backend.core.config import settings
from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.images")
router = APIRouter()

GENERATED_IMAGE_PURPOSE = "generated_image"


def _default_image_model() -> str:
    """从 env 读取默认生图模型，缺省 qwen3.8-max-preview。"""
    name = str(getattr(settings, "IMAGE_GENERATION_MODEL", "") or "").strip()
    return name or "qwen3.8-max-preview"


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        result.append(u)
    return result


def _extract_image_urls(text: str) -> list[str]:
    urls: list[str] = []

    for u in re.findall(r"!\[.*?\]\((https?://[^\s\)]+)\)", text):
        urls.append(u.rstrip(").,;"))

    for u in re.findall(r'"(?:url|image|src|imageUrl|image_url)"\s*:\s*"(https?://[^"]+)"', text):
        urls.append(u)

    cdn_pattern = (
        r"https?://(?:cdn\.qwenlm\.ai|wanx\.alicdn\.com|img\.alicdn\.com|"
        r'[^\s"<>]+\.(?:jpg|jpeg|png|webp|gif))(?:[^\s"<>]*)'
    )
    for u in re.findall(cdn_pattern, text, re.IGNORECASE):
        urls.append(u.rstrip(".,;)\"'>"))

    return _dedupe_urls(urls)


def _urls_from_image_entries(entries: Any) -> list[str]:
    urls: list[str] = []
    if not isinstance(entries, list):
        return urls
    for item in entries:
        if isinstance(item, str) and item.startswith("http"):
            urls.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("image", "url", "src", "image_url", "imageUrl"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)
                break
    return urls


def _extract_image_urls_from_events(events: list[dict]) -> list[str]:
    """从已解析的 SSE 事件中提取图片 URL。

    官网文生图关键帧：
      phase=image_gen_tool, status=finished,
      extra.image_list[].image / extra.tool_result[].image
    优先 image_list（展示用），再 tool_result，最后回退正则扫整包。
    """
    preferred: list[str] = []
    fallback: list[str] = []
    for evt in events:
        if not isinstance(evt, dict):
            continue
        extra = evt.get("extra")
        if isinstance(extra, dict):
            preferred.extend(_urls_from_image_entries(extra.get("image_list")))
            fallback.extend(_urls_from_image_entries(extra.get("tool_result")))
        # 兼容未归一化的 raw choices 结构
        choices = evt.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                nested_extra = delta.get("extra") if isinstance(delta.get("extra"), dict) else {}
                preferred.extend(_urls_from_image_entries(nested_extra.get("image_list")))
                fallback.extend(_urls_from_image_entries(nested_extra.get("tool_result")))
    ordered = _dedupe_urls(preferred + fallback)
    if ordered:
        return ordered
    # 文本 / markdown / 整包 JSON 回退
    blob = "\n".join(json.dumps(e, ensure_ascii=False) for e in events if isinstance(e, dict))
    return _extract_image_urls(blob)


def _resolve_image_model(requested: str | None) -> str:
    """解析生图上游模型。

    - 未传 / 空串 → settings.IMAGE_GENERATION_MODEL
    - 以 qwen 开头的名称 → 原样使用（便于临时指定其它 Qwen 型号）
    - 其它（含旧 dall-e 别名）→ 回退到 env 默认
    """
    default = _default_image_model()
    name = str(requested or "").strip()
    if not name:
        return default
    lower = name.lower()
    if lower.startswith("qwen"):
        return name
    return default


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _build_image_prompt(prompt: str) -> str:
    return (
        "请直接生成图片，不要只输出文字描述。"
        "如果可以生成图片，请返回可访问的图片链接或包含图片链接的结果。\n\n"
        f"用户需求：{prompt}"
    )


def _guess_filename(url: str, content_type: str) -> str:
    path = urlparse(url).path
    name = Path(path).name or "generated.png"
    if "." not in name:
        ext = mimetypes.guess_extension(content_type or "") or ".png"
        name = f"{name}{ext}"
    return name.split("?")[0]


def _public_content_url(request: Request, file_id: str) -> str:
    # 相对路径：同源前端 / 反代均可直接使用
    return f"/v1/images/content/{file_id}"


async def _rehost_urls(
    request: Request,
    client: QwenClient,
    acc: Any,
    image_urls: list[str],
    limit: int,
) -> list[dict[str, str]]:
    file_store = getattr(request.app.state, "file_store", None)
    if file_store is None or not hasattr(client, "download_url"):
        return [{"url": url, "revised_prompt": ""} for url in image_urls[:limit]]

    hosted: list[dict[str, str]] = []
    for url in image_urls[:limit]:
        try:
            raw, content_type = await client.download_url(url, account=acc)
            if not raw:
                raise RuntimeError("empty image body")
            filename = _guess_filename(url, content_type or "image/png")
            meta = await file_store.save_bytes(
                filename,
                content_type or "image/png",
                raw,
                GENERATED_IMAGE_PURPOSE,
            )
            hosted.append(
                {
                    "url": _public_content_url(request, str(meta["id"])),
                    "revised_prompt": "",
                }
            )
            log.info("[T2I] rehosted url -> file_id=%s size=%s", meta.get("id"), meta.get("size"))
        except Exception as exc:
            log.warning("[T2I] rehost failed url=%s error=%s", url[:120], exc)
            # 回退原始 CDN（可能仍不可用，但至少不吞结果）
            hosted.append({"url": url, "revised_prompt": ""})
    return hosted


@router.get("/v1/images/content/{file_id}")
@router.get("/images/content/{file_id}")
async def get_image_content(request: Request, file_id: str):
    file_store = getattr(request.app.state, "file_store", None)
    if file_store is None:
        raise HTTPException(status_code=404, detail="Image store unavailable")
    meta = await file_store.get(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if meta.get("purpose") and meta.get("purpose") != GENERATED_IMAGE_PURPOSE:
        # 仅暴露生图结果，避免误读上下文附件
        raise HTTPException(status_code=404, detail="Image not found")
    path = meta.get("path")
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Image file missing")
    media_type = meta.get("content_type") or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=meta.get("filename") or Path(path).name)


@router.post("/v1/images/generations")
@router.post("/images/generations")
async def create_image(request: Request):
    from backend.core.config import API_KEYS, settings

    client: QwenClient = request.app.state.qwen_client

    token = _get_token(request)
    if API_KEYS:
        if token != settings.ADMIN_KEY and token not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt: str = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    n: int = min(max(int(body.get("n", 1)), 1), 4)
    model = _resolve_image_model(body.get("model"))

    log.info(f"[T2I] model={model}, n={n}, prompt={prompt[:80]!r}")

    acc = None
    chat_id = None
    try:
        prompt_text = _build_image_prompt(prompt)
        events: list[dict] = []
        async for item in client.chat_stream_events_with_retry(model, prompt_text, has_custom_tools=False):
            if item.get("type") == "meta":
                acc = item.get("acc")
                chat_id = item.get("chat_id")
                continue
            if item.get("type") != "event":
                continue
            evt = item.get("event", {})
            if isinstance(evt, dict):
                events.append(evt)

        if acc is None or chat_id is None:
            raise HTTPException(status_code=500, detail="Image generation session was not created")

        image_urls = _extract_image_urls_from_events(events)
        if not image_urls:
            try:
                chats = await client.list_chats(acc.token, limit=20, account=acc)
                current_chat = next(
                    (c for c in chats if isinstance(c, dict) and str(c.get("id") or "") == str(chat_id)),
                    None,
                )
                if current_chat:
                    image_urls = _extract_image_urls(json.dumps(current_chat, ensure_ascii=False))
            except Exception as exc:
                log.debug("[T2I] current chat fallback failed chat_id=%s error=%s", chat_id, exc)
        log.info(f"[T2I] 提取到 {len(image_urls)} 张图片 URL: {image_urls}")

        if not image_urls:
            raise HTTPException(status_code=500, detail="Image generation succeeded but no URL found")

        data = await _rehost_urls(request, client, acc, image_urls, n)
        for item in data:
            if not item.get("revised_prompt"):
                item["revised_prompt"] = prompt
        return JSONResponse({"created": int(time.time()), "data": data})

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[T2I] 生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if acc is not None and getattr(acc, "inflight", 0) > 0:
            client.account_pool.release(acc)
        if acc is not None and chat_id and settings.UPSTREAM_AUTO_DELETE_ENABLED:
            asyncio.create_task(client.delete_chat(acc.token, chat_id, account=acc))
