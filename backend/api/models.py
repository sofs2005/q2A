from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.api.videos import VIDEO_ENGINE_IDS
from backend.core.config import MODEL_MAP, resolve_model, settings
from backend.core.request_logging import new_request_id, request_context
from backend.services.auth_quota import resolve_auth_context
from backend.services.qwen_client import QwenClient

router = APIRouter()

OWNER = "qwen2api"

# `/v1/models` 里 `capabilities` 的取值。OpenAI 的 model 对象没有这个字段，
# 加它是为了让客户端不必靠猜模型名就能区分对话 / 生图 / 生视频 —— 此前生图和
# 生视频的 ID 完全没上报，客户端拉完列表就只显示对话入口。
CAP_CHAT = "chat"
CAP_IMAGE = "image"
CAP_VIDEO = "video"


def _model_entry(model_id: str, capabilities: tuple[str, ...], owned_by: str = OWNER) -> dict:
    """构造一条 OpenAI 兼容的 model 记录（额外带 `capabilities`）。"""
    return {
        "id": model_id,
        "object": "model",
        "owned_by": owned_by,
        "capabilities": list(capabilities),
    }


def _capability_models() -> list[tuple[str, tuple[str, ...]]]:
    """生图 / 生视频的对外模型 ID。

    - 生图：`settings.IMAGE_MODEL_IDS` —— 上游没有独立生图模型，这些是能力
      别名，请求时解析回默认模型（`images._resolve_image_model`）；
    - 生视频：`VIDEO_ENGINE_IDS` —— 真实引擎名，`model` 可以直接写它们。

    两边都排序，保证同一份配置每次输出顺序一致（客户端会 diff 这个列表）。
    """
    pairs: list[tuple[str, tuple[str, ...]]] = [
        (str(model_id), (CAP_IMAGE,))
        for model_id in getattr(settings, "IMAGE_MODEL_IDS", ())
    ]
    pairs.extend((engine, (CAP_VIDEO,)) for engine in sorted(VIDEO_ENGINE_IDS))
    return pairs


def _capability_for(model_id: str) -> tuple[str, ...] | None:
    """查一个 ID 是不是生图 / 生视频的能力别名（大小写不敏感），不是则 None。"""
    low = str(model_id or "").strip().lower()
    if not low:
        return None
    for known, capabilities in _capability_models():
        if known.lower() == low:
            return capabilities
    return None


def _append_unique(
    data: list[dict],
    seen: set[str],
    model_id: str,
    capabilities: tuple[str, ...],
    owned_by: str = OWNER,
) -> None:
    """按 id 去重后追加 —— 上游可能把某个引擎也当对话模型返回。"""
    if model_id in seen:
        return
    seen.add(model_id)
    data.append(_model_entry(model_id, capabilities, owned_by))


def _build_model_list_payload() -> dict:
    seen: set[str] = set()
    data: list[dict] = []
    for model_id in MODEL_MAP:
        _append_unique(data, seen, model_id, (CAP_CHAT,))
    for model_id, capabilities in _capability_models():
        _append_unique(data, seen, model_id, capabilities)
    return {"object": "list", "data": data}


@router.get("/v1/models")
async def list_models(request: Request):
    req_id = new_request_id()
    with request_context(req_id=req_id, surface="models", path="/v1/models", requested_model="-", resolved_model="-"):
        app = request.app
        users_db = app.state.users_db
        client: QwenClient = app.state.qwen_client

        auth = await resolve_auth_context(request, users_db)
        token = auth.token
        upstream_models = []
        if settings.MODELS_USE_UPSTREAM:
            try:
                upstream_models = await client.list_models(token)
            except Exception:
                upstream_models = []

        if upstream_models:
            seen: set[str] = set()
            data: list[dict] = []
            for item in upstream_models:
                model_id = item.get("id") or item.get("model") or item.get("name") or str(item)
                # 上游返回的都是对话模型；生图/生视频的 ID 只由本地配置补充。
                _append_unique(data, seen, model_id, (CAP_CHAT,), item.get("owned_by", OWNER))
            for model_id in MODEL_MAP:
                _append_unique(data, seen, model_id, (CAP_CHAT,))
            for model_id, capabilities in _capability_models():
                _append_unique(data, seen, model_id, capabilities)
            return JSONResponse({"object": "list", "data": data})
        return JSONResponse(_build_model_list_payload())


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    req_id = new_request_id()
    with request_context(req_id=req_id, surface="models", path=f"/v1/models/{model_id}", requested_model=model_id, resolved_model=resolve_model(model_id)):
        resolved = resolve_model(model_id)
        capabilities = _capability_for(model_id)
        if resolved == model_id and model_id not in MODEL_MAP and capabilities is None:
            raise HTTPException(status_code=404, detail={"error": {"message": f"Model '{model_id}' not found", "type": "invalid_request_error"}})
        if capabilities is not None and CAP_IMAGE in capabilities:
            # 生图别名的真实 chat 级模型；出图引擎另由 IMAGE_GENERATION_META_MODEL 决定。
            resolved = str(getattr(settings, "IMAGE_GENERATION_MODEL", "") or resolved)
        payload: dict = {
            "id": model_id,
            "object": "model",
            "owned_by": OWNER,
            "resolved_model": resolved,
        }
        if capabilities is not None:
            payload["capabilities"] = list(capabilities)
        return JSONResponse(payload)
