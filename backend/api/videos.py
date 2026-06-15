"""
视频生成接口 — 兼容 OpenAI /v1/videos/generations 规范。

底层通过现有直连 HTTP 聊天能力触发千问 "生成视频" 模式（chat_type=t2v）。
视频生成为异步任务：聊天流很快结束（只产出 response.created），真正的
task_id 嵌在 messages[].extra.wanx.task_id，需要从会话详情读取后再轮询
GET /api/v1/tasks/status/{task_id}，直到任务完成才能取到视频 URL。

提取/轮询逻辑参考已验证的 chat.qwen.ai 网页代理实现（FreeQwenApi）：
- task_id: data.messages[].extra.wanx.task_id（优先），回退顶层 task_id
- 轮询端点: GET https://chat.qwen.ai/api/v1/tasks/status/{task_id}
- 状态字段: task_status | status；成功 completed/success/succeeded
- 视频 URL: 状态响应的 content / video_url / url（含 .mp4/.mov/.webm）
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

# 任务状态判定（大小写不敏感）
TASK_STATUS_SUCCESS = {"completed", "success", "succeeded", "finished"}
TASK_STATUS_FAILED = {"failed", "error", "cancelled", "canceled"}

# 轮询参数
POLL_TIMEOUT_SECONDS = 7 * 60
POLL_INTERVAL_SECONDS = 5

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".m3u8")


# ---------------------------------------------------------------------------
# JSON 感知的递归提取
# ---------------------------------------------------------------------------

def _find_task_id(obj) -> str | None:
    """递归查找 task_id：优先 extra.wanx.task_id，回退顶层 task_id/taskId。"""
    wanx_id: list[str] = []
    flat_id: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            wanx = o.get("wanx")
            if isinstance(wanx, dict):
                tid = wanx.get("task_id") or wanx.get("taskId")
                if tid:
                    wanx_id.append(str(tid))
            for key in ("task_id", "taskId"):
                v = o.get(key)
                if isinstance(v, str) and v:
                    flat_id.append(v)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    if wanx_id:
        return wanx_id[0]
    return flat_id[0] if flat_id else None


def _looks_like_video_url(url: str) -> bool:
    low = url.lower().split("?", 1)[0]
    return low.endswith(_VIDEO_EXTS) or "video" in url.lower()


def _find_video_url(obj) -> str | None:
    """递归收集所有 http(s) URL（含字符串内嵌/Markdown），返回首个像视频的。"""
    candidates: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for m in re.findall(r'https?://[^\s"<>\)\]]+', o):
                candidates.append(m.rstrip(".,;)\"'>"))

    walk(obj)
    for u in candidates:
        if _looks_like_video_url(u):
            return u
    return None


def _task_status_of(obj) -> str:
    """从任务状态响应中取状态字段（顶层或 data 下）。"""
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


def _parse_json(body: str):
    try:
        return json.loads(body or "")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 统一轮询：先拿 task_id/url，再轮询任务状态
# ---------------------------------------------------------------------------

async def _poll_until_ready(client: QwenClient, acc, chat_id: str,
                            task_id: str | None) -> str | None:
    """
    在超时窗口内统一轮询，直到取到视频 URL：
    - 有 task_id：轮询 GET /api/v1/tasks/status/{task_id}，从响应取 content/url
    - 无 task_id：拉会话详情 /api/v2/chats/{chat_id}，尝试发现 task_id 或直接的视频 URL
    """
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    last_status = ""
    while time.time() < deadline:
        if task_id:
            res = await client.get_vision_task_status(acc.token, task_id, account=acc)
            if res.get("status") == 200:
                obj = _parse_json(res.get("body", ""))
                status = _task_status_of(obj)
                if status:
                    last_status = status
                if status in TASK_STATUS_SUCCESS:
                    url = _find_video_url(obj)
                    if url:
                        return url
                elif status in TASK_STATUS_FAILED:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Video task failed status={status}",
                    )
        else:
            # 尚未拿到 task_id：从会话详情发现 task_id 或直接的视频 URL
            try:
                detail = await client.get_chat_detail(acc.token, chat_id, account=acc)
                obj = _parse_json(detail.get("body", ""))
                if obj is not None:
                    url = _find_video_url(obj)
                    if url:
                        return url
                    task_id = _find_task_id(obj)
                    if task_id:
                        log.info(f"[T2V] 从会话详情发现 task_id={task_id}")
                        continue  # 立即转入任务状态轮询，不等待
            except Exception as exc:
                log.debug("[T2V] chat detail poll failed chat_id=%s error=%s", chat_id, exc)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise HTTPException(
        status_code=504,
        detail=f"Video task timed out chat_id={chat_id} last_status={last_status or '-'}",
    )


# ---------------------------------------------------------------------------
# 核心生成流程（自带账号释放与会话清理）
# ---------------------------------------------------------------------------

async def _run_generation(
    client: QwenClient, model: str, prompt: str, n: int, duration: int,
    ratio: str, media_options: dict,
) -> dict:
    from backend.core.config import settings

    acc = None
    chat_id = None
    owns_release = False  # 流正常结束时由本调用方负责唯一一次 release
    try:
        prompt_text = _build_video_prompt(prompt, duration, ratio)
        events: list[dict] = []
        async for item in client.chat_stream_events_with_retry(
            model, prompt_text, has_custom_tools=False,
            chat_type="t2v", media_options=media_options,
        ):
            if item.get("type") == "meta":
                acc = item.get("acc")
                chat_id = item.get("chat_id")
                continue
            if item.get("type") == "event":
                events.append(item.get("event", {}))
        # 执行器在成功路径（return）不会 release，调用方接管
        owns_release = True

        if acc is None or chat_id is None:
            raise HTTPException(status_code=500, detail="Video generation session was not created")

        # 1) 流里可能已有直接 URL（少见）
        video_url = _find_video_url(events)
        # 2) 流里找 task_id（嵌套 extra.wanx.task_id）
        task_id = _find_task_id(events)
        log.info(f"[T2V] stream done: url={bool(video_url)} task_id={task_id or '-'} events={len(events)}")

        # 3) 统一轮询：拿 task_id/url → 轮询任务状态
        if not video_url:
            video_url = await _poll_until_ready(client, acc, chat_id, task_id)

        if not video_url:
            # 自诊断：转储事件结构，便于核对真实形状
            log.warning(
                "[T2V] no video url. events_preview=%s",
                json.dumps(events, ensure_ascii=False)[:2000],
            )
            raise HTTPException(status_code=500, detail="Video generation produced no video URL")

        log.info(f"[T2V] 视频 URL: {video_url}")
        # Qwen 单任务生成单个视频；n>1 时返回同一 URL 以兼容契约
        data = [
            {
                "url": video_url,
                "revised_prompt": prompt,
                "ratio": ratio,
                "size": ratio,
                "duration": duration,
            }
            for _ in range(n)
        ]
        return {"created": int(time.time()), "data": data}

    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        log.error(f"[T2V] 生成失败: {msg}")
        if acc is not None and ("quota" in msg.lower() or "额度" in msg or "limit" in msg.lower()):
            client.account_pool.mark_video_limited(acc, error_message=msg)
        raise HTTPException(status_code=500, detail=msg)
    finally:
        if acc is not None and owns_release:
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
