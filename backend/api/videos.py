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

引擎与规格：chat 级 model 只负责触发 t2v 通道，真正的视频引擎由
`messages[0].extra.meta.model` 决定（默认取配置 VIDEO_GENERATION_META_MODEL，
默认 happyhorse-1.1）；若请求的 `model` 直接写了引擎名（见 VIDEO_ENGINE_IDS），
则 chat 级 model 用 DEFAULT_VIDEO_MODEL 且引擎按请求值下发（别名优先）。
分辨率/时长取自请求的 resolution/duration，随
`extra.meta` 下发。响应里的 `extra.wanx.video_spec` 会被打成 `[T2V] video_spec ...`
自检日志，用于确认上述字段名是否真的被上游接受。
"""
import re
import time
import json
import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from backend.services.media_purposes import GENERATED_VIDEO_PURPOSE
from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.videos")
router = APIRouter()

DEFAULT_VIDEO_MODEL = "qwen3.8-max"

# 注意：这里的 chat 级 model **只负责触发上游 t2v 通道**，真正出视频的引擎由
# `messages[0].extra.meta.model`（配置项 VIDEO_GENERATION_META_MODEL）决定。
# 历史遗留的死模型（qwen3.6-plus 上游已删除，返回 Not_Found）统一改指 qwen3.8-max。
VIDEO_MODEL_MAP = {
    "qwen3.8-max": "qwen3.8-max",
    "qwen3.6-plus": "qwen3.8-max",
    "qwen3.6-plus-video": "qwen3.8-max",
    "qwen-video": "qwen3.8-max",
    "sora": "qwen3.8-max",
}

# 视频引擎 ID：允许客户端把 `model` 直接写成引擎名（更直观），此时 chat 级
# model 仍取 DEFAULT_VIDEO_MODEL 触发 t2v 通道，引擎则通过
# `messages[0].extra.meta.model` 显式下发（覆盖配置默认值）。
# 取值与 VIDEO_GENERATION_META_MODEL 的合法值保持一致。
VIDEO_ENGINE_IDS = {"happyhorse-1.1", "happyhorse-1.0", "wan-2.7"}

VALID_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4"}

# 官网前端可选值（2026-09 bundle）：resolutions 默认 720P、durations 默认 5 秒。
# 非法值一律回落默认并发 WARNING，**不报 400**，避免把现有客户端打挂。
DEFAULT_RESOLUTION = "720P"
VALID_RESOLUTIONS = {"720P", "1080P"}
DEFAULT_DURATION = 5
VALID_DURATIONS = {5, 10, 15}

# 上游免费号的规格限制：只有 720P + 5s 可用，其它组合会被拒绝并回
# `extra.quota_limit=true` + content「请求非法：当前视频规格仅会员可用…」。
MEMBER_ONLY_KEYWORDS = ("会员", "member")
MEMBER_ONLY_DETAIL = (
    "该视频规格需要 Qwen 会员（1080P / 10s / 15s）。"
    "免费账号只支持 720P + 5s。"
)

# 关键区分：「仅会员可用」这条上游话术有**两种**成因，处理方式完全相反。
#   1) 真的请求了会员规格（1080P / 10s / 15s）→ 规格问题，重试任何免费号都
#      必然失败，必须立即 400，绝不能拿 25 个号去白烧。
#   2) 请求的是免费规格（720P + 5s）却仍被打回 → 该账号**当天的免费视频额度
#      已用完**（免费号 1 个/天）→ 纯粹是账号额度问题，换号即可成功。
# 生产实证（2026-09）：
#   [T2V] member_only spec rejected resolution=720P duration=5 account=qw6@suzhou.gq
# 旧实现把两者混为一谈，一律 400 且不换号 —— 一个号额度用完整个请求就废了。
FREE_SPEC_RESOLUTION = "720P"
FREE_SPEC_DURATION = 5

# 免费规格被打回时最多换几个号（含首次）。全都不行再回 429。
VIDEO_QUOTA_RETRY_ATTEMPTS = 3

# 全部账号免费额度都用完时回 429：让调用方能区分「换号也没用」与「规格不对」。
VIDEO_QUOTA_EXHAUSTED_DETAIL = (
    "当前所有账号的免费视频额度已用完（免费额度 1 个/天），请稍后再试或升级会员。"
)

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


def _message_text(content: Any) -> str:
    """把上游 message.content 归一成纯文本（兼容 str 与分段列表）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for seg in content:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict):
                text = seg.get("text") or seg.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
        return " ".join(parts)
    return ""


def _find_member_only_rejection(obj) -> str | None:
    """检测上游「该规格仅会员可用」拒绝，命中时返回上游 content 文本。

    生产实测结构（免费号请求 1080P / 10s / 15s）::

        {"data": {"messages": [{"content": "请求非法：当前视频规格仅会员可用…",
                                "extra": {"update_member": true,
                                          "quota_limit": true}}]}}

    判定条件（两者同时成立才算，避免把普通额度耗尽误判成规格问题）：
    1. `messages[].extra.quota_limit` 为真；
    2. `messages[].content` 含「会员」相关字样。

    Args:
        obj: 已解析的 completions 响应体（可为 None / 任意结构）。

    Returns:
        命中的 message content；未命中返回 None。
    """
    hits: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            messages = o.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    extra = msg.get("extra")
                    if not isinstance(extra, dict) or not extra.get("quota_limit"):
                        continue
                    text = _message_text(msg.get("content")).strip()
                    low = text.lower()
                    if any(k in text or k in low for k in MEMBER_ONLY_KEYWORDS):
                        hits.append(text)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return hits[0] if hits else None


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


def _normalize_resolution(value: Any) -> str:
    """校验并归一 t2v 分辨率；非法/缺失回落 720P 并打 WARNING（不抛 400）。"""
    raw = value
    candidate = str(raw or "").strip().upper()
    if not candidate:
        return DEFAULT_RESOLUTION
    if candidate in VALID_RESOLUTIONS:
        return candidate
    log.warning(
        "[T2V] invalid resolution=%r, fallback to %s (valid: %s)",
        raw, DEFAULT_RESOLUTION, sorted(VALID_RESOLUTIONS),
    )
    return DEFAULT_RESOLUTION


def _normalize_duration(value: Any) -> int:
    """校验并归一 t2v 时长；非法/缺失回落 5 秒并打 WARNING（不抛 400）。"""
    raw = value
    if raw is None or str(raw).strip() == "":
        return DEFAULT_DURATION
    try:
        candidate = int(str(raw).strip())
    except (TypeError, ValueError):
        log.warning(
            "[T2V] invalid duration=%r, fallback to %ss (valid: %s)",
            raw, DEFAULT_DURATION, sorted(VALID_DURATIONS),
        )
        return DEFAULT_DURATION
    if candidate in VALID_DURATIONS:
        return candidate
    log.warning(
        "[T2V] invalid duration=%r, fallback to %ss (valid: %s)",
        raw, DEFAULT_DURATION, sorted(VALID_DURATIONS),
    )
    return DEFAULT_DURATION


def _extract_video_spec(obj: Any) -> dict | None:
    """从响应体里取 `extra.wanx.video_spec`（官网 2026-09 回显的上游视频规格）。"""
    found: list[dict] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            wanx = o.get("wanx")
            if isinstance(wanx, dict):
                spec = wanx.get("video_spec")
                if isinstance(spec, dict):
                    found.append(spec)
            spec = o.get("video_spec")
            if isinstance(spec, dict):
                found.append(spec)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found[0] if found else None


def _log_video_spec(
    obj: Any,
    *,
    expected_resolution: str = "",
    expected_duration: int | None = None,
) -> dict | None:
    """打印上游回显的 video_spec，用于验证 resolution/duration 字段名是否生效。

    这是当前**唯一的验证手段**：等有额度时生成一次视频，看回显的
    resolution/duration 是否等于我们传的值，即可确认字段名正确。
    旧任务格式没有该字段，只打一条 missing 日志，绝不报错。

    Args:
        obj: 已解析的 completions 响应体（可为 None / 任意结构）。
        expected_resolution: 本次请求的分辨率，用于比对并给出告警。
        expected_duration: 本次请求的时长（秒），用于比对并给出告警。

    Returns:
        命中的 video_spec，缺失时返回 None。
    """
    spec = _extract_video_spec(obj)
    if spec is None:
        log.info("[T2V] video_spec missing (old task format)")
        return None

    log.info(
        "[T2V] video_spec model=%s resolution=%s duration=%s ratio=%s watermark=%s",
        spec.get("model", "-"),
        spec.get("resolution", "-"),
        spec.get("duration", "-"),
        spec.get("ratio") or spec.get("size") or "-",
        spec.get("watermark", "-"),
    )

    # 字段名是推断值：回显若与请求不一致，说明键名不对，打 WARNING 便于定位。
    actual_resolution = str(spec.get("resolution") or "").strip().upper()
    if expected_resolution and actual_resolution and actual_resolution != expected_resolution:
        log.warning(
            "[T2V] video_spec resolution mismatch: requested=%s actual=%s",
            expected_resolution, actual_resolution,
        )
    actual_duration = str(spec.get("duration") or "").strip()
    if expected_duration is not None and actual_duration:
        try:
            if int(actual_duration) != int(expected_duration):
                log.warning(
                    "[T2V] video_spec duration mismatch: requested=%s actual=%s",
                    expected_duration, actual_duration,
                )
        except (TypeError, ValueError):
            log.warning(
                "[T2V] video_spec duration unparsable: requested=%s actual=%r",
                expected_duration, actual_duration,
            )
    return spec


def _resolve_video_model_and_engine(requested: str | None) -> tuple[str, str | None]:
    """解析请求里的 `model`，返回 (chat 级 model, 引擎覆盖值)。

    Args:
        requested: 请求体的 `model` 字段，可能是 chat 级模型别名、引擎名或空。

    Returns:
        二元组 ``(chat_model, engine_override)``：

        - `model` 命中 `VIDEO_ENGINE_IDS`（引擎名）→ chat 级 model 用默认值
          `DEFAULT_VIDEO_MODEL`，引擎覆盖值 = 请求的引擎名（别名优先于配置默认）。
        - 其它情况（含未知模型 / 空值）→ chat 级 model 走 `VIDEO_MODEL_MAP`
          解析（未知回落默认），引擎覆盖值 `None`（跟随配置默认）。
    """
    name = str(requested or "").strip()
    if name in VIDEO_ENGINE_IDS:
        return DEFAULT_VIDEO_MODEL, name
    return VIDEO_MODEL_MAP.get(name, DEFAULT_VIDEO_MODEL), None


def _resolve_video_model(requested: str | None) -> str:
    """只取 chat 级 model（保留旧签名，兼容既有调用方与单测）。"""
    return _resolve_video_model_and_engine(requested)[0]


def _build_video_media_options(
    ratio: str, duration: int, resolution: str, engine: str | None = None,
) -> dict:
    """组装下发给 payload_builder 的媒体参数。

    Args:
        ratio: 宽高比（16:9 等）。
        duration: 时长（秒）。
        resolution: 分辨率（720P / 1080P）。
        engine: 引擎覆盖值；为空时**不放入** `engine` 键（跟随配置默认）。

    Returns:
        媒体参数字典（`engine` 键仅在显式指定引擎时出现）。
    """
    options: dict[str, Any] = {
        "ratio": ratio,
        "size": ratio,
        "duration": duration,
        "resolution": resolution,
    }
    engine_name = str(engine or "").strip()
    if engine_name:
        options["engine"] = engine_name
    return options


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


def _guess_video_filename(url: str, content_type: str) -> str:
    """从 CDN URL 里取文件名；没有扩展名时按 content_type 猜，兜底 .mp4。"""
    name = Path(urlparse(url).path).name or "generated"
    if "." not in name:
        ext = mimetypes.guess_extension(content_type or "") or ".mp4"
        name = f"{name}{ext}"
    return name.split("?")[0]


async def _rehost_video(
    file_store: Any,
    client: QwenClient,
    acc: Any,
    video_url: str,
    *,
    prompt: str,
    ratio: str,
    duration: int,
    model: str = "",
    resolution: str = "",
) -> str:
    """回源下载上游视频并落盘，返回同源 `/v1/videos/content/{id}`。

    上游 CDN 直链带签名会过期，因此生成成功后尽量转存到本地。任何失败
    （无 store / 无下载能力 / 体积超限 / 下载异常）都退回原始 URL ——
    绝不能因为回源失败让生成失败。
    """
    from backend.core.config import settings

    if file_store is None or not hasattr(client, "download_url"):
        return video_url
    try:
        raw, content_type = await client.download_url(video_url, account=acc)
        if not raw:
            log.warning("[T2V] rehost skipped: empty body url=%s", video_url[:120])
            return video_url
        max_bytes = int(getattr(settings, "GENERATED_VIDEO_MAX_BYTES", 0) or 0)
        if max_bytes > 0 and len(raw) > max_bytes:
            log.warning(
                "[T2V] rehost skipped: video too large size=%s max=%s url=%s",
                len(raw), max_bytes, video_url[:120],
            )
            return video_url
        content_type = content_type or "video/mp4"
        name = _guess_video_filename(video_url, content_type)
        meta = await file_store.save_bytes(
            name,
            content_type,
            raw,
            GENERATED_VIDEO_PURPOSE,
            extra={
                "prompt": prompt,
                "ratio": ratio,
                "duration": duration,
                "model": model,
                "resolution": resolution,
            },
        )
        log.info("[T2V] rehosted url -> file_id=%s size=%s", meta.get("id"), meta.get("size"))
        return f"/v1/videos/content/{meta['id']}"
    except Exception as exc:
        log.warning("[T2V] rehost failed url=%s error=%s", video_url[:120], exc)
        return video_url


# ---------------------------------------------------------------------------
# 统一轮询：先拿 task_id/url，再轮询任务状态
# ---------------------------------------------------------------------------

async def _poll_until_ready(client: QwenClient, acc, chat_id: str,
                            task_id: str) -> str:
    """轮询 GET /api/v1/tasks/status/{task_id}，直到任务完成取到视频 URL。"""
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    last_status = ""
    while time.time() < deadline:
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
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise HTTPException(
        status_code=504,
        detail=f"Video task timed out task_id={task_id} last_status={last_status or '-'}",
    )


# ---------------------------------------------------------------------------
# 核心生成流程（自带账号释放与会话清理）
# ---------------------------------------------------------------------------

def _is_free_video_spec(resolution: str, duration: int) -> bool:
    """本次请求是否为免费号可用规格（720P + 5s）。

    这是「上游回『仅会员可用』」两种成因的唯一判别依据：
    免费规格被打回 ⇒ 账号当天额度用完（可换号重试）；
    会员规格被打回 ⇒ 规格问题（直接 400）。

    Args:
        resolution: 归一后的分辨率（如 "720P" / "1080P"）。
        duration: 归一后的时长（秒）。

    Returns:
        免费规格返回 True，否则 False。
    """
    try:
        seconds = int(duration)
    except (TypeError, ValueError):
        return False
    return str(resolution).strip().upper() == FREE_SPEC_RESOLUTION and seconds == FREE_SPEC_DURATION


def _video_blocked_emails(pool: Any) -> set:
    """取账号池中处于视频冷却期（`video_blocked_until`）的邮箱集合。

    这些账号当天的免费视频额度已用完，冷却期内不该再被选中，直接作为
    `exclude` 预置给执行器。fake pool（单测替身）可能没有该方法，做安全兜底。

    Args:
        pool: 账号池对象（可为 None）。

    Returns:
        处于视频冷却期的账号邮箱集合（永不为 None）。
    """
    getter = getattr(pool, "video_blocked_emails", None)
    if not callable(getter):
        return set()
    try:
        return set(getter() or ())
    except Exception as exc:  # 兜底：诊断能力不能反过来打断主流程
        log.warning("[T2V] collect video_blocked_emails failed: %s", exc)
        return set()


def _release_video_session(client: QwenClient, acc: Any, chat_id: str | None) -> None:
    """释放账号并异步清理上游会话（视频额度用完换号前调用）。

    Args:
        client: Qwen 客户端（用于拿到 account_pool 并发起会话清理）。
        acc: 待释放账号；为 None 时直接返回。
        chat_id: 上游会话 id；为空则跳过清理。
    """
    from backend.core.config import settings

    if acc is None:
        return
    pool = getattr(client, "account_pool", None)
    if pool is not None:
        pool.release(acc)
    if chat_id and settings.UPSTREAM_AUTO_DELETE_ENABLED:
        asyncio.create_task(client.delete_chat(acc.token, chat_id, account=acc))


async def _run_generation(
    client: QwenClient, model: str, prompt: str, n: int, duration: int,
    ratio: str, media_options: dict, file_store: Any = None,
    resolution: str = DEFAULT_RESOLUTION,
) -> dict:
    from backend.core.config import settings

    acc = None
    chat_id = None
    owns_release = False  # 成功路径由本调用方负责唯一一次 release
    try:
        prompt_text = _build_video_prompt(prompt, duration, ratio)
        # 免费规格被打回 = 该号当天免费额度用完 → 换号重试；
        # 会员规格被打回 = 真的规格问题 → 立即 400，一个号都不多试。
        is_free_spec = _is_free_video_spec(resolution, duration)
        max_attempts = VIDEO_QUOTA_RETRY_ATTEMPTS if is_free_spec else 1
        pool = getattr(client, "account_pool", None)
        # 冷却期内（video_blocked_until）的账号本次直接排除，避免撞同一个死号。
        excluded: set = _video_blocked_emails(pool)

        obj = None
        body = ""
        video_url = None
        task_id = None
        for attempt in range(1, max_attempts + 1):
            # 视频为异步任务：用非流式 completions 一次性取回响应体，
            # 从 messages[0].extra.wanx.task_id 解析 task_id 后再轮询任务状态。
            # exclude 会把本轮已判定「额度用完」的账号带给执行器，实现换号。
            try:
                res = await client.complete_once_with_retry(
                    model, prompt_text, has_custom_tools=False,
                    chat_type="t2v", media_options=media_options,
                    exclude=excluded or None,
                )
            except HTTPException:
                raise
            except Exception as exc:
                # 池子里只剩被视频冷却排除的账号 → 语义上就是「额度都用完了」
                if excluded and "no available accounts" in str(exc).lower():
                    log.warning(
                        "[T2V] no account left after excluding video-quota accounts "
                        "excluded=%s", sorted(excluded),
                    )
                    raise HTTPException(
                        status_code=429, detail=VIDEO_QUOTA_EXHAUSTED_DETAIL,
                    )
                raise
            # 执行器在成功路径不会 release，调用方接管
            owns_release = True
            acc = res.get("acc")
            chat_id = res.get("chat_id")
            body = res.get("body", "")

            if acc is None or chat_id is None:
                raise HTTPException(status_code=500, detail="Video generation session was not created")

            obj = _parse_json(body)
            # 0) 自检：把上游回显的 video_spec 打出来（字段名是否生效的唯一验证手段）
            _log_video_spec(
                obj,
                expected_resolution=resolution,
                expected_duration=duration,
            )
            # 1) 响应体里可能已有直接 URL（少见）
            video_url = _find_video_url(obj) if obj is not None else None
            # 2) 解析 task_id（嵌套 extra.wanx.task_id）
            task_id = _find_task_id(obj) if obj is not None else None
            log.info(f"[T2V] completions done: url={bool(video_url)} task_id={task_id or '-'} body_len={len(body)}")

            if video_url or task_id:
                break

            # 3) 没有 URL 也没有 task_id：先判断是不是「仅会员可用」拒绝
            member_only = (
                _find_member_only_rejection(obj) if obj is not None else None
            )
            if member_only is None:
                log.warning(
                    "[T2V] no task_id in completions body. body_preview=%s",
                    body[:2000],
                )
                raise HTTPException(status_code=500, detail="Video generation returned no task_id")

            if not is_free_spec:
                # 会员规格：确实是规格问题，直接 400，不换号（避免白烧整个账号池）
                log.warning(
                    "[T2V] member_only spec rejected resolution=%s duration=%s account=%s",
                    resolution,
                    duration,
                    getattr(acc, "email", "-"),
                )
                raise HTTPException(status_code=400, detail=MEMBER_ONLY_DETAIL)

            # 免费规格还被打回 → 该号当天免费视频额度用完，标记冷却并换号重试
            log.warning(
                "[T2V] account video quota exhausted account=%s resolution=%s duration=%s "
                "attempt=%s/%s → retry with another account",
                getattr(acc, "email", "-"),
                resolution,
                duration,
                attempt,
                max_attempts,
            )
            if pool is not None:
                # 复用现有视频限流标记（accounts.json 的 video_blocked_until，
                # 默认冷却 3600s）：冷却期内该号不再被视频请求选中。
                pool.mark_video_limited(acc, error_message=member_only)
            excluded.add(getattr(acc, "email", ""))
            # 先手动释放本轮账号与会话，再把 acc/chat_id 置空避免 finally 二次释放
            finished_acc, finished_chat = acc, chat_id
            owns_release = False
            acc = None
            chat_id = None
            _release_video_session(client, finished_acc, finished_chat)
            if attempt >= max_attempts:
                raise HTTPException(
                    status_code=429, detail=VIDEO_QUOTA_EXHAUSTED_DETAIL,
                )

        # 3) 轮询任务状态直到拿到视频 URL
        if not video_url:
            if not task_id:
                raise HTTPException(status_code=500, detail="Video generation returned no task_id")
            video_url = await _poll_until_ready(client, acc, chat_id, task_id)

        if not video_url:
            raise HTTPException(status_code=500, detail="Video generation produced no video URL")

        log.info(f"[T2V] 视频 URL: {video_url}")
        # 回源落盘只做一次：n 条结果复用同一个 URL（上游单任务只产 1 个视频）
        video_url = await _rehost_video(
            file_store, client, acc, video_url,
            prompt=prompt, ratio=ratio, duration=duration, model=model,
            resolution=resolution,
        )
        # Qwen 单任务生成单个视频；n>1 时返回同一 URL 以兼容契约
        data = [
            {
                "url": video_url,
                "revised_prompt": prompt,
                "ratio": ratio,
                "size": ratio,
                "duration": duration,
                "resolution": resolution,
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


@router.get("/v1/videos/content/{file_id}")
@router.get("/videos/content/{file_id}")
async def get_video_content(request: Request, file_id: str):
    """回源转存后的视频内容（同源播放，避免 CDN 直链过期/403）。

    不传 filename：避免 attachment 响应头让浏览器下载而不是内联播放。
    Starlette 的 FileResponse 自带 Range 支持，浏览器 <video> 拖动进度可用。
    """
    file_store = getattr(request.app.state, "file_store", None)
    if file_store is None:
        raise HTTPException(status_code=404, detail="Video not found")
    meta = await file_store.get(file_id)
    if meta is None or meta.get("purpose") != GENERATED_VIDEO_PURPOSE:
        # 仅暴露视频生成物，避免误读上下文附件/生图结果
        raise HTTPException(status_code=404, detail="Video not found")
    path = meta.get("path")
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path,
        media_type=meta.get("content_type") or "video/mp4",
        content_disposition_type="inline",
    )


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
    # 非法 duration / resolution 一律回落默认值（5 秒 / 720P）并发 WARNING，
    # 不返回 400——现有客户端可能仍在传任意秒数，不能被我们打挂。
    duration: int = _normalize_duration(body.get("duration", DEFAULT_DURATION))
    resolution: str = _normalize_resolution(body.get("resolution"))
    ratio = _normalize_ratio(body.get("ratio") or body.get("aspect_ratio") or body.get("size"))
    model, engine_override = _resolve_video_model_and_engine(body.get("model"))
    media_options = _build_video_media_options(ratio, duration, resolution, engine_override)

    log.info(
        "[T2V] model=%s, n=%s, engine=%s, ratio=%s, resolution=%s, duration=%s, prompt=%r",
        model, n, engine_override or "-", ratio, resolution, duration, prompt[:80],
    )

    heartbeat_interval = settings.STREAM_HEARTBEAT_INTERVAL_SECONDS

    async def event_stream():
        gen_task = asyncio.create_task(
            _run_generation(
                client, model, prompt, n, duration, ratio, media_options,
                file_store=getattr(request.app.state, "file_store", None),
                resolution=resolution,
            )
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
