import asyncio
import faulthandler
import logging
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys

# 将项目根目录加入到 sys.path，解决直接运行 main.py 时找不到 backend 模块的问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import settings, VERSION_LABEL
from backend.core.database import AsyncJsonDB
from backend.core.account_pool import AccountPool
from backend.core.session_affinity import SessionAffinityStore
from backend.core.upstream_file_cache import UpstreamFileCache
from backend.core.session_lock import SessionLockRegistry
from backend.core.browser_fingerprint import close_all_sessions
from backend.core.request_logging import configure_logging, request_context
from backend.core.diagnostics import event_loop_lag_watchdog, install_stack_dump_handler
from backend.core.spa_static_files import SPAStaticFiles
from backend.services.qwen_client import QwenClient
from backend.services.file_store import LocalFileStore
from backend.toolcore.context_offload import ContextOffloader
from backend.services.response_store import InMemoryResponseStore
from backend.services.upstream_file_uploader import UpstreamFileUploader
from backend.services.model_modalities import ModalityNotSupportedError
from backend.services.attachment_errors import (
    InlineMediaTooLargeError,
    UpstreamAttachmentRejectedError,
)
from backend.api.attachment_error_handlers import (
    inline_media_too_large_handler,
    upstream_attachment_rejected_handler,
)
import backend.api.models as models
from backend.api import admin, v1_chat, responses_api, probes, anthropic, gemini, embeddings, images, videos, files_api
from backend.services.garbage_collector import garbage_collect_chats
from backend.services.context_cleanup import context_cleanup_loop

configure_logging(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
install_stack_dump_handler(
    settings=settings,
    faulthandler_module=faulthandler,
    signal_module=signal,
    stream=sys.stderr,
)
log = logging.getLogger("qwen2api")


async def token_refresh_loop(app: FastAPI):
    """后台给账号续签短效 access_token。

    上游只认 900s 的 access_token，且该 token 只能由浏览器签发，签发一次要几秒。
    放在后台提前续，避免用户请求打到"token 刚好过期"的窗口上。
    """
    from backend.services.token_provider import ensure_fresh_token, is_token_usable

    interval = max(15.0, float(getattr(settings, "QWEN_TOKEN_REFRESH_INTERVAL_SECONDS", 60)))
    while True:
        try:
            if getattr(settings, "QWEN_BROWSER_TOKEN_ENABLED", True):
                pool = getattr(app.state, "account_pool", None)
                accounts = list(getattr(pool, "accounts", []) or [])
                changed = False
                for acc in accounts:
                    if not getattr(acc, "valid", False) or not getattr(acc, "password", ""):
                        continue
                    if is_token_usable(getattr(acc, "token", ""), margin=180):
                        continue
                    token = await ensure_fresh_token(acc)
                    if token:
                        acc.token = token
                        changed = True
                if changed and pool is not None:
                    await pool.save()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("[TokenRefresh] loop failed: %s", exc)
        await asyncio.sleep(interval)


async def chat_id_pool_loop(app: FastAPI):
    chat_pool = getattr(app.state.qwen_client, "chat_id_pool", None)
    if chat_pool is None:
        return
    while True:
        try:
            await chat_pool.cleanup(delete_all=False)
            await chat_pool.fill()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("[ChatIDPool] loop failed: %s", exc)
        await asyncio.sleep(
            max(5.0, float(getattr(settings, "CHAT_ID_PREWARM_REFILL_INTERVAL_SECONDS", 300) or 300))
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    with request_context(surface="startup"):
        log.info(f"正在启动 qwen2API {VERSION_LABEL} 企业网关...")

        # 初始化数据存储 (带锁 JSON)
        app.state.accounts_db = AsyncJsonDB(settings.ACCOUNTS_FILE, default_data=[])
        app.state.users_db = AsyncJsonDB(settings.USERS_FILE, default_data=[])
        app.state.captures_db = AsyncJsonDB(settings.CAPTURES_FILE, default_data=[])
        app.state.session_affinity_db = AsyncJsonDB(settings.CONTEXT_AFFINITY_FILE, default_data=[])
        app.state.context_cache_db = AsyncJsonDB(settings.CONTEXT_CACHE_FILE, default_data=[])
        app.state.uploaded_files_db = AsyncJsonDB(settings.UPLOADED_FILES_FILE, default_data=[])

        # 初始化组件
        app.state.account_pool = AccountPool(app.state.accounts_db, max_inflight=settings.MAX_INFLIGHT_PER_ACCOUNT)
        app.state.qwen_client = QwenClient(app.state.account_pool)
        app.state.qwen_executor = app.state.qwen_client.executor
        app.state.file_store = LocalFileStore(settings.CONTEXT_GENERATED_DIR, app.state.uploaded_files_db)
        app.state.session_affinity = SessionAffinityStore(app.state.session_affinity_db)
        app.state.upstream_file_cache = UpstreamFileCache(app.state.context_cache_db)
        app.state.context_offloader = ContextOffloader(settings)
        app.state.upstream_file_uploader = UpstreamFileUploader(app.state.qwen_client, settings)
        app.state.session_locks = SessionLockRegistry()
        app.state.response_store = InMemoryResponseStore()

        # 加载账号并启动后台清理任务
        await app.state.account_pool.load()
        await app.state.file_store.load()
        await app.state.session_affinity.load()
        await app.state.upstream_file_cache.load()
        asyncio.create_task(garbage_collect_chats(app))
        asyncio.create_task(context_cleanup_loop(app))
        app.state.chat_id_pool_task = asyncio.create_task(chat_id_pool_loop(app))
        app.state.token_refresh_task = asyncio.create_task(token_refresh_loop(app))

        app.state.event_loop_watchdog_stop = asyncio.Event()
        app.state.event_loop_watchdog_task = None
        if settings.DIAGNOSTIC_EVENT_LOOP_WATCHDOG_ENABLED:
            app.state.event_loop_watchdog_task = asyncio.create_task(
                event_loop_lag_watchdog(
                    interval_seconds=settings.DIAGNOSTIC_EVENT_LOOP_WATCHDOG_INTERVAL_SECONDS,
                    threshold_seconds=settings.DIAGNOSTIC_EVENT_LOOP_LAG_THRESHOLD_SECONDS,
                    stop_event=app.state.event_loop_watchdog_stop,
                )
            )

    yield

    with request_context(surface="shutdown"):
        log.info("正在关闭网关服务...")
        watchdog_stop = getattr(app.state, "event_loop_watchdog_stop", None)
        if watchdog_stop is not None:
            watchdog_stop.set()
        watchdog_task = getattr(app.state, "event_loop_watchdog_task", None)
        if watchdog_task is not None:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
        token_task = getattr(app.state, "token_refresh_task", None)
        if token_task is not None:
            token_task.cancel()
            try:
                await token_task
            except asyncio.CancelledError:
                pass
        try:
            from backend.services.token_provider import BrowserTokenProvider

            await BrowserTokenProvider.get_instance().close()
        except Exception:
            pass
        chat_pool_task = getattr(app.state, "chat_id_pool_task", None)
        if chat_pool_task is not None:
            chat_pool_task.cancel()
            try:
                await chat_pool_task
            except asyncio.CancelledError:
                pass
        chat_pool = getattr(getattr(app.state, "qwen_client", None), "chat_id_pool", None)
        if chat_pool is not None:
            await chat_pool.cleanup(delete_all=True)
        await close_all_sessions()

app = FastAPI(title="qwen2API Enterprise Gateway", version=VERSION_LABEL, lifespan=lifespan)


@app.exception_handler(ModalityNotSupportedError)
async def modality_not_supported_handler(request: Request, exc: ModalityNotSupportedError) -> JSONResponse:
    """多模态门控失败：按 400 返回，让调用方知道该换成支持该模态的模型。"""
    log.warning("[Modality] rejected request path=%s error=%s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "unsupported_modality",
            }
        },
    )


# 附件链路的异常处理器实现放在 backend/api/attachment_error_handlers.py
# （只依赖 fastapi.responses，便于独立测试），这里负责把它们挂到 app 上。
app.add_exception_handler(InlineMediaTooLargeError, inline_media_too_large_handler)
app.add_exception_handler(UpstreamAttachmentRejectedError, upstream_attachment_rejected_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(v1_chat.router, tags=["OpenAI Compatible"])
app.include_router(responses_api.router, tags=["Responses Compatible"])
app.include_router(models.router, tags=["Models"])
app.include_router(anthropic.router, tags=["Claude Compatible"])
app.include_router(gemini.router, tags=["Gemini Compatible"])
app.include_router(embeddings.router, tags=["Embeddings"])
app.include_router(images.router, tags=["Images"])
app.include_router(videos.router, tags=["Videos"])
app.include_router(files_api.router, tags=["Files"])
app.include_router(probes.router, tags=["Probes"])
app.include_router(admin.router, prefix="/api/admin", tags=["Dashboard Admin"])

@app.get("/api", tags=["System"])
async def root():
    return {
        "status": "qwen2API Enterprise Gateway is running",
        "docs": "/docs",
        "version": VERSION_LABEL
    }

# 托管前端构建产物
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    log.warning(f"未找到前端构建目录: {FRONTEND_DIST}，WebUI 将不可用。")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=settings.PORT, workers=1)
