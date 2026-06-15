"""
视频生成接口 — 兼容 OpenAI /v1/videos/generations 规范。

底层通过现有直连 HTTP 聊天能力触发千问 "生成视频" 模式（chat_type=t2v），
不依赖浏览器运行时。视频生成耗时较长，初始响应通常只返回任务 ID，
需要轮询 /api/v1/tasks/status/{id} 直到任务完成后才能取到视频 URL。
"""
import re
import time
import json
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.videos")
router = APIRouter()

DEFAULT_VIDEO_MODEL = "qwen3.6-plus"

VIDEO_MODEL_MAP = {
    "qwen3.6-plus": "qwen3.6-plus",
    "qwen3.6-plus-video": "qwen3.6-plus",
    "qwen-video": "qwen3.6-plus",
    "sora": "qwen3.6-plus",
}

VALID_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4"}

# 任务运行/成功状态判定
TASK_STATUS_RUNNING = {"running", "pending", "queued", "processing", "created"}
TASK_STATUS_SUCCESS = {"success", "succeeded", "finished", "completed"}

# 轮询参数
POLL_TIMEOUT_SECONDS = 7 * 60
POLL_INTERVAL_SECONDS = 10


def _extract_video_urls(text: str) -> list[str]:
    urls: list[str] = []

    for u in re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', text):
        urls.append(u.rstrip(").,;"))

    for u in re.findall(r'"(?:url|video|src|videoUrl|video_url)"\s*:\s*"(https?://[^"]+)"', text):
        urls.append(u)

    mp4_pattern = r'https?://[^\s"<>]+\.(?:mp4|webm|mov|m3u8)(?:[^\s"<>]*)'
    for u in re.findall(mp4_pattern, text, re.IGNORECASE):
        urls.append(u.rstrip(".,;)\"'>"))

    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if _looks_like_video_url(u) and u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _looks_like_video_url(url: str) -> bool:
    lowered = url.lower()
    return any(ext in lowered for ext in (".mp4", ".webm", ".mov", ".m3u8")) or "video" in lowered


def _extract_task_ids(text: str) -> list[str]:
    ids: list[str] = []
    for tid in re.findall(r'"task_id"\s*:\s*"([^"]+)"', text):
        ids.append(tid)
    for tid in re.findall(r'"taskId"\s*:\s*"([^"]+)"', text):
        ids.append(tid)
    seen: set[str] = set()
    result: list[str] = []
    for tid in ids:
        if tid and tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result


def _task_status_from_body(body: str) -> str:
    try:
        obj = json.loads(body)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    for key in ("task_status", "status"):
        val = obj.get(key) or (data.get(key) if data else None)
        if val:
            return str(val).strip().lower()
    return ""


def _normalize_ratio(value: str | None) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in VALID_RATIOS else "16:9"


def _resolve_video_model(requested: str | None) -> str:
    if not requested:
        return DEFAULT_VIDEO_MODEL
    return VIDEO_MODEL_MAP.get(requested, DEFAULT_VIDEO_MODEL)


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _build_video_prompt(prompt: str, duration: int, ratio: str) -> str:
    return (
        f"{prompt}\n\n"
        f"视频要求：生成 {duration} 秒视频，宽高比 {ratio}。"
        "请直接生成视频，返回可访问的视频链接。"
    )


async def _poll_video_task(client: QwenClient, acc, task_id: str) -> str:
    """轮询任务状态，返回累积的响应快照文本。超时或失败抛异常。"""
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    snapshots: list[str] = []
    last_status = ""
    while time.time() < deadline:
        res = await client.get_vision_task_status(acc.token, task_id, account=acc)
        body = res.get("body", "") or ""
        if body:
            snapshots.append(body)
        if res.get("status") == 200:
            status = _task_status_from_body(body)
            if status:
                last_status = status
            if status in TASK_STATUS_SUCCESS:
                return "\n".join(snapshots)
            if status and status not in TASK_STATUS_RUNNING:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video task failed status={status}",
                )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    raise HTTPException(
        status_code=504,
        detail=f"Video task timed out task_id={task_id} last_status={last_status or '-'}",
    )


async def _run_generation(
    client: QwenClient, model: str, prompt: str, n: int, duration: int,
    ratio: str, media_options: dict,
) -> dict:
    """完整的视频生成流程，返回最终响应体。自带账号释放与会话清理。"""
    from backend.core.config import settings

    acc = None
    chat_id = None
    try:
        prompt_text = _build_video_prompt(prompt, duration, ratio)
        event_payloads: list[str] = []
        async for item in client.chat_stream_events_with_retry(
            model, prompt_text, has_custom_tools=False,
            chat_type="t2v", media_options=media_options,
        ):
            if item.get("type") == "meta":
                acc = item.get("acc")
                chat_id = item.get("chat_id")
                continue
            if item.get("type") != "event":
                continue
            event_payloads.append(json.dumps(item.get("event", {}), ensure_ascii=False))

        if acc is None or chat_id is None:
            raise HTTPException(status_code=500, detail="Video generation session was not created")

        answer_text = "\n".join(event_payloads)
        video_urls = _extract_video_urls(answer_text)

        # 流里没有直接的 URL：尝试轮询任务
        if not video_urls:
            task_ids = _extract_task_ids(answer_text)
            if task_ids:
                log.info(f"[T2V] 开始轮询任务 task_id={task_ids[0]}")
                task_text = await _poll_video_task(client, acc, task_ids[0])
                answer_text += "\n" + task_text
                video_urls = _extract_video_urls(answer_text)

        # 仍无 URL：拉会话详情兜底
        if not video_urls:
            try:
                detail = await client.get_chat_detail(acc.token, chat_id, account=acc)
                detail_body = detail.get("body", "") or ""
                if detail_body:
                    answer_text += "\n" + detail_body
                    video_urls = _extract_video_urls(answer_text)
            except Exception as exc:
                log.debug("[T2V] chat detail fallback failed chat_id=%s error=%s", chat_id, exc)

        log.info(f"[T2V] 提取到 {len(video_urls)} 个视频 URL: {video_urls}")

        if not video_urls:
            raise HTTPException(status_code=500, detail="Video generation produced no video URL")

        data = [
            {
                "url": url,
                "revised_prompt": prompt,
                "ratio": ratio,
                "size": ratio,
                "duration": duration,
            }
            for url in video_urls[:n]
        ]
        return {"created": int(time.time()), "data": data}

    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        log.error(f"[T2V] 生成失败: {msg}")
        # 配额相关错误，标记账号视频维度受限
        if acc is not None and ("quota" in msg.lower() or "额度" in msg or "limit" in msg.lower()):
            client.account_pool.mark_video_limited(acc, error_message=msg)
        raise HTTPException(status_code=500, detail=msg)
    finally:
        if acc is not None and getattr(acc, "inflight", 0) > 0:
            client.account_pool.release(acc)
        if acc is not None and chat_id and settings.UPSTREAM_AUTO_DELETE_ENABLED:
            asyncio.create_task(client.delete_chat(acc.token, chat_id, account=acc))


@router.post("/v1/videos/generations")
@router.post("/videos/generations")
async def create_video(request: Request):
    """
    同步语义 + SSE 心跳保活：客户端发起一次请求并等待，服务端在长时间
    轮询期间持续发送心跳注释（: heartbeat）保活连接，避免被反向代理/
    负载均衡的空闲超时切断；生成完成后将结果作为单个 data 事件返回。
    """
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

    n: int = min(max(int(body.get("n", 1)), 1), 2)
    duration: int = min(max(int(body.get("duration", 5)), 1), 10)
    ratio = _normalize_ratio(body.get("ratio") or body.get("aspect_ratio") or body.get("size"))
    model = _resolve_video_model(body.get("model"))
    media_options = {"ratio": ratio, "size": ratio, "duration": duration}

    log.info(f"[T2V] model={model}, n={n}, ratio={ratio}, duration={duration}, prompt={prompt[:80]!r}")

    heartbeat_interval = settings.STREAM_HEARTBEAT_INTERVAL_SECONDS

    async def event_stream():
        gen_task = asyncio.create_task(
            _run_generation(client, model, prompt, n, duration, ratio, media_options)
        )
        try:
            while True:
                try:
                    # shield 保护生成任务不被 wait_for 超时取消，仅用于驱动心跳
                    result = await asyncio.wait_for(
                        asyncio.shield(gen_task), timeout=heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                except HTTPException as e:
                    err = {"error": {"message": str(e.detail), "code": e.status_code}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    return
                except Exception as e:
                    err = {"error": {"message": str(e)}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    return
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                return
        finally:
            if not gen_task.done():
                log.warning("[T2V] 客户端断开，取消生成任务")
                gen_task.cancel()
                try:
                    await gen_task
                except Exception:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
