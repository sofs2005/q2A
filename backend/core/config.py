import os
import json
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, Set

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    # 服务配置
    PORT: int = int(os.getenv("PORT", 8080))
    WORKERS: int = int(os.getenv("WORKERS", 3))
    ADMIN_KEY: str = os.getenv("ADMIN_KEY", "admin")

    MAX_INFLIGHT_PER_ACCOUNT: int = int(os.getenv("MAX_INFLIGHT", 1))

    # 容灾与限流
    MAX_RETRIES: int = 3
    RATE_LIMIT_COOLDOWN: int = 600
    ACCOUNT_MIN_INTERVAL_MS: int = int(os.getenv("ACCOUNT_MIN_INTERVAL_MS", 1200))
    ACCOUNT_BUSY_TIMEOUT_SECONDS: float = float(os.getenv("ACCOUNT_BUSY_TIMEOUT_SECONDS", 900))
    REQUEST_JITTER_MIN_MS: int = int(os.getenv("REQUEST_JITTER_MIN_MS", 120))
    REQUEST_JITTER_MAX_MS: int = int(os.getenv("REQUEST_JITTER_MAX_MS", 360))
    WAF_RETRY_EXTRA_COOLDOWN_SECONDS: float = float(os.getenv("WAF_RETRY_EXTRA_COOLDOWN_SECONDS", 5))
    RATE_LIMIT_BASE_COOLDOWN: int = int(os.getenv("RATE_LIMIT_BASE_COOLDOWN", 600))
    RATE_LIMIT_MAX_COOLDOWN: int = int(os.getenv("RATE_LIMIT_MAX_COOLDOWN", 3600))
    CHAT_ID_PREWARM_TARGET_PER_ACCOUNT: int = int(os.getenv("CHAT_ID_PREWARM_TARGET_PER_ACCOUNT", 5))
    CHAT_ID_PREWARM_TTL_SECONDS: int = int(os.getenv("CHAT_ID_PREWARM_TTL_SECONDS", 120))
    CHAT_ID_PREWARM_MAX_CONCURRENCY: int = int(os.getenv("CHAT_ID_PREWARM_MAX_CONCURRENCY", 16))
    CHAT_ID_PREWARM_SPREAD_SECONDS: float = float(os.getenv("CHAT_ID_PREWARM_SPREAD_SECONDS", 6))
    CHAT_ID_PREWARM_JITTER_SECONDS: float = float(os.getenv("CHAT_ID_PREWARM_JITTER_SECONDS", 1.5))
    CHAT_ID_PREWARM_MODELS: str = os.getenv("CHAT_ID_PREWARM_MODELS", "qwen3.8-max-preview,qwen3.7-plus")
    QWEN_CHAT_TRANSPORT_SEND_COOKIES: bool = os.getenv("QWEN_CHAT_TRANSPORT_SEND_COOKIES", "false").lower() in {"1", "true", "yes", "on"}
    # 默认 false：主流式走 curl_cffi（new_session 注入 UPSTREAM_PROXY + TLS 指纹）。
    # 设为 true 时回退 httpx Go-like 路径（TLS 指纹易被 WAF 识别，仅作兼容开关）。
    QWEN_CHAT_TRANSPORT_GO_LIKE_HTTP: bool = os.getenv("QWEN_CHAT_TRANSPORT_GO_LIKE_HTTP", "false").lower() in {"1", "true", "yes", "on"}

    # Captcha Solver (x5sec punish 滑块突破)
    CAPTCHA_SOLVER_ENABLED: bool = os.getenv("CAPTCHA_SOLVER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    CAPTCHA_SOLVER_TIMEOUT_MS: int = int(os.getenv("CAPTCHA_SOLVER_TIMEOUT_MS", 15000))
    CAPTCHA_BROWSER_IDLE_TIMEOUT: float = float(os.getenv("CAPTCHA_BROWSER_IDLE_TIMEOUT", 30))
    WAF_PUNISH_COOLDOWN: int = int(os.getenv("WAF_PUNISH_COOLDOWN", 1800))

    # 日志
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    QWEN_CODE_CODER_MODEL: str = os.getenv("QWEN_CODE_CODER_MODEL", "qwen3-coder-plus")
    QWEN_CODE_FORCE_CODER_FOR_TOOL_CALLS: bool = os.getenv("QWEN_CODE_FORCE_CODER_FOR_TOOL_CALLS", "true").lower() in {"1", "true", "yes", "on"}
    QWEN_CODE_FORCE_CODER_FOR_CODING_TASKS: bool = os.getenv("QWEN_CODE_FORCE_CODER_FOR_CODING_TASKS", "true").lower() in {"1", "true", "yes", "on"}
    TOOLCORE_V2_ENABLED: bool = os.getenv("TOOLCORE_V2_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    DIAGNOSTIC_STACK_DUMP_ENABLED: bool = os.getenv("DIAGNOSTIC_STACK_DUMP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    DIAGNOSTIC_EVENT_LOOP_WATCHDOG_ENABLED: bool = os.getenv("DIAGNOSTIC_EVENT_LOOP_WATCHDOG_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    DIAGNOSTIC_EVENT_LOOP_WATCHDOG_INTERVAL_SECONDS: float = float(os.getenv("DIAGNOSTIC_EVENT_LOOP_WATCHDOG_INTERVAL_SECONDS", 1.0))
    DIAGNOSTIC_EVENT_LOOP_LAG_THRESHOLD_SECONDS: float = float(os.getenv("DIAGNOSTIC_EVENT_LOOP_LAG_THRESHOLD_SECONDS", 5.0))
    DIAGNOSTIC_SLOW_STEP_SECONDS: float = float(os.getenv("DIAGNOSTIC_SLOW_STEP_SECONDS", 0.05))
    UPSTREAM_AUTO_DELETE_ENABLED: bool = os.getenv("UPSTREAM_AUTO_DELETE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    # 上游请求超时
    QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS: float = float(
        os.getenv("QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS", 60)
    )
    QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS: float = float(
        os.getenv("QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS", 600)
    )
    STREAM_HEARTBEAT_INTERVAL_SECONDS: float = float(
        os.getenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", 15)
    )
    QWEN_UPSTREAM_STREAM_TOTAL_TIMEOUT_SECONDS: float = float(
        os.getenv("QWEN_UPSTREAM_STREAM_TOTAL_TIMEOUT_SECONDS", 300)
    )
    QWEN_UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS: float = float(
        os.getenv("QWEN_UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS", 90)
    )
    QWEN_UPSTREAM_STREAM_DEDICATED_SESSION: bool = os.getenv("QWEN_UPSTREAM_STREAM_DEDICATED_SESSION", "true").lower() in {"1", "true", "yes", "on"}
    # 上游出站代理（留空=直连）。支持 socks5h://host:port、http://host:port 等，
    # 例如对接 MicroWARP（Cloudflare WARP SOCKS5）：socks5h://microwarp:1080
    UPSTREAM_PROXY: str = os.getenv("UPSTREAM_PROXY", "").strip()
    MODELS_USE_UPSTREAM: bool = os.getenv("MODELS_USE_UPSTREAM", "true").lower() in {"1", "true", "yes", "on"}
    OPENAI_JSON_SINGLEFLIGHT_ENABLED: bool = os.getenv("OPENAI_JSON_SINGLEFLIGHT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    OPENAI_JSON_SINGLEFLIGHT_WAIT_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_JSON_SINGLEFLIGHT_WAIT_TIMEOUT_SECONDS", 600))
    OPENAI_JSON_SINGLEFLIGHT_RESULT_TTL_SECONDS: float = float(os.getenv("OPENAI_JSON_SINGLEFLIGHT_RESULT_TTL_SECONDS", 120))

    # 数据文件路径
    ACCOUNTS_FILE: str = os.getenv("ACCOUNTS_FILE", str(DATA_DIR / "accounts.json"))
    USERS_FILE: str = os.getenv("USERS_FILE", str(DATA_DIR / "users.json"))
    CAPTURES_FILE: str = os.getenv("CAPTURES_FILE", str(DATA_DIR / "captures.json"))
    CONFIG_FILE: str = os.getenv("CONFIG_FILE", str(DATA_DIR / "config.json"))

    # ????? / ????
    CONTEXT_INLINE_MAX_CHARS: int = int(os.getenv("CONTEXT_INLINE_MAX_CHARS", 4000))
    CONTEXT_FORCE_FILE_MAX_CHARS: int = int(os.getenv("CONTEXT_FORCE_FILE_MAX_CHARS", 10000))
    CONTEXT_ATTACHMENT_TTL_SECONDS: int = int(os.getenv("CONTEXT_ATTACHMENT_TTL_SECONDS", 1800))
    CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS: int = int(os.getenv("CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS", 60))
    CONTEXT_GENERATED_DIR: str = os.getenv("CONTEXT_GENERATED_DIR", str(DATA_DIR / "context_files"))
    CONTEXT_CACHE_FILE: str = os.getenv("CONTEXT_CACHE_FILE", str(DATA_DIR / "context_cache.json"))
    UPLOADED_FILES_FILE: str = os.getenv("UPLOADED_FILES_FILE", str(DATA_DIR / "uploaded_files.json"))
    CONTEXT_AFFINITY_FILE: str = os.getenv("CONTEXT_AFFINITY_FILE", str(DATA_DIR / "session_affinity.json"))
    CONTEXT_ALLOWED_GENERATED_EXTS: str = os.getenv("CONTEXT_ALLOWED_GENERATED_EXTS", "txt,md,json,log")
    CONTEXT_ALLOWED_USER_EXTS: str = os.getenv("CONTEXT_ALLOWED_USER_EXTS", "txt,md,json,log,xml,yaml,yml,csv,html,css,py,js,ts,java,c,cpp,cs,php,go,rb,sh,zsh,ps1,bat,cmd,pdf,doc,docx,ppt,pptx,xls,xlsx,png,jpg,jpeg,webp,gif,tiff,bmp,svg")

    class Config:
        env_file = ".env"

API_KEYS_FILE = DATA_DIR / "api_keys.json"

def load_api_keys() -> set:
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("keys", []))
        except Exception:
            pass
    return set()

def save_api_keys(keys: set):
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump({"keys": list(keys)}, f, indent=2)

# 在内存中存储管理的 API Keys
API_KEYS = load_api_keys()

VERSION = "3.0.1"
# 对外展示用的版本标签，统一从 VERSION 派生，避免多处硬编码漂移
VERSION_LABEL = f"v{VERSION}（modified by softs2005）"

settings = Settings()

# 全局映射：仅保留用户指定的 Qwen 短别名，其余模型由上游列表提供。
MODEL_MAP = {
    "qwen-max": "qwen3.8-max-preview",
    "qwen-plus": "qwen3.7-plus",
}

def resolve_model(name: str) -> str:
    return MODEL_MAP.get(name, name)


GENERIC_QWEN_CODE_MODELS = {
    "qwen3.6-plus",
    "qwen-plus",
    "qwen-max",
    "qwen",
}


def resolve_qwen_code_model(name: str) -> str:
    return resolve_model(settings.QWEN_CODE_CODER_MODEL or name)


def _normalized_model_name(name: str | None) -> str:
    return str(name or "").strip().lower()


def _looks_like_coder_model(name: str | None) -> bool:
    normalized = _normalized_model_name(name)
    return "coder" in normalized or normalized.startswith("qwen-code")


def _is_explicit_non_coder_model(name: str | None) -> bool:
    normalized = _normalized_model_name(name)
    return any(marker in normalized for marker in ("flash", "mini", "turbo"))


def should_route_qwen_code_to_coder(
    requested_model: str,
    *,
    client_profile: str,
    tool_enabled: bool = False,
    coding_intent: bool = False,
) -> bool:
    if client_profile != "qwen_code_openai":
        return False
    if _looks_like_coder_model(requested_model):
        return False
    resolved_model = resolve_model(requested_model)
    if _looks_like_coder_model(resolved_model):
        return False
    if _is_explicit_non_coder_model(requested_model):
        return False

    if tool_enabled and settings.QWEN_CODE_FORCE_CODER_FOR_TOOL_CALLS and resolved_model in GENERIC_QWEN_CODE_MODELS:
        return True
    if coding_intent and settings.QWEN_CODE_FORCE_CODER_FOR_CODING_TASKS and resolved_model in GENERIC_QWEN_CODE_MODELS:
        return True
    return False


def resolve_request_model(
    requested_model: str,
    *,
    client_profile: str,
    tool_enabled: bool = False,
    coding_intent: bool = False,
) -> str:
    if should_route_qwen_code_to_coder(
        requested_model,
        client_profile=client_profile,
        tool_enabled=tool_enabled,
        coding_intent=coding_intent,
    ):
        return resolve_qwen_code_model(requested_model)
    return resolve_model(requested_model)
