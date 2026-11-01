import os
import json
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, Set

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_QWEN_MAX_MODEL = "qwen3.8-max"
DEFAULT_QWEN_PLUS_MODEL = "qwen3.7-plus"
# 上游 Version 头。**以实测可用性为准，不以抓包为准。**
# 0.3.10 确为官网浏览器实际发送值，但服务端对它返回 401
# （"Your session has expired, or the token is no longer valid."）—— 即该版本号
# 已被服务端下线/不认。改回 0.2.91 后请求立刻恢复，故 0.2.91 才是当前真正可用的值。
# 教训：抓包只能证明"官网在发什么"，证明不了"服务端认什么"。换版本号必须实测，
# CDN 路径反推同样不可靠（首页预渲染页也会滞后）。
# 2026-10-30 二次实测修正：0.3.27 在「短效 access_token」下可正常出流；
# 0.2.91 反而返回 Bad_Request。此前"0.3.27 被判 401"的结论是因为当时用的是
# 账号池里的长期 token —— 两个变量纠缠导致的误判。换版本号必须连同 token 类型
# 一起实测（见 backend/services/token_provider.py 的模块说明）。
DEFAULT_QWEN_WEB_VERSION = "0.3.28"
DEFAULT_QWEN_BX_VERSION = "2.5.37"

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
    WAF_RETRY_EXTRA_COOLDOWN_SECONDS: float = float(os.getenv("WAF_RETRY_EXTRA_COOLDOWN_SECONDS", 30))
    RATE_LIMIT_BASE_COOLDOWN: int = int(os.getenv("RATE_LIMIT_BASE_COOLDOWN", 600))
    RATE_LIMIT_MAX_COOLDOWN: int = int(os.getenv("RATE_LIMIT_MAX_COOLDOWN", 3600))
    # 命中 punish 后的重试退避：base * 2^attempt，上限见 MAX。固定的 2~5s 退避在
    # 风控看来就是"被拦后立刻接着打"，必须随重试次数拉长。
    WAF_RETRY_BACKOFF_BASE_SECONDS: float = float(os.getenv("WAF_RETRY_BACKOFF_BASE_SECONDS", 5))
    WAF_RETRY_BACKOFF_MAX_SECONDS: float = float(os.getenv("WAF_RETRY_BACKOFF_MAX_SECONDS", 60))
    # 预热默认值按"低强度长驻"取：单账号单模型只留 2 个、TTL 拉到 15 分钟，
    # 否则每轮预热本身就是一次自动化特征明显的请求脉冲。
    CHAT_ID_PREWARM_TARGET_PER_ACCOUNT: int = int(os.getenv("CHAT_ID_PREWARM_TARGET_PER_ACCOUNT", 2))
    CHAT_ID_PREWARM_TTL_SECONDS: int = int(os.getenv("CHAT_ID_PREWARM_TTL_SECONDS", 900))
    CHAT_ID_PREWARM_MAX_CONCURRENCY: int = int(os.getenv("CHAT_ID_PREWARM_MAX_CONCURRENCY", 4))
    CHAT_ID_PREWARM_SPREAD_SECONDS: float = float(os.getenv("CHAT_ID_PREWARM_SPREAD_SECONDS", 6))
    CHAT_ID_PREWARM_JITTER_SECONDS: float = float(os.getenv("CHAT_ID_PREWARM_JITTER_SECONDS", 1.5))
    # 预热补给轮询间隔（秒）。旧实现是 TTL//4，TTL 调大后会跟着放大成新的脉冲源，
    # 故拆成独立可配项。
    CHAT_ID_PREWARM_REFILL_INTERVAL_SECONDS: float = float(os.getenv("CHAT_ID_PREWARM_REFILL_INTERVAL_SECONDS", 300))
    # 未设置时由 ChatIDPool 跟随 qwen-max/qwen-plus 的当前目标；显式空串可关闭默认预热模型。
    CHAT_ID_PREWARM_MODELS: str | None = os.getenv("CHAT_ID_PREWARM_MODELS")
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
    # 对外短别名对应的上游模型；升级模型时优先修改环境变量。
    QWEN_MAX_MODEL: str = os.getenv("QWEN_MAX_MODEL", DEFAULT_QWEN_MAX_MODEL).strip() or DEFAULT_QWEN_MAX_MODEL
    QWEN_PLUS_MODEL: str = os.getenv("QWEN_PLUS_MODEL", DEFAULT_QWEN_PLUS_MODEL).strip() or DEFAULT_QWEN_PLUS_MODEL
    # 官网请求头版本，跟随抓包验证结果，允许部署侧独立调整。
    QWEN_WEB_VERSION: str = os.getenv("QWEN_WEB_VERSION", DEFAULT_QWEN_WEB_VERSION).strip() or DEFAULT_QWEN_WEB_VERSION
    QWEN_BX_VERSION: str = os.getenv("QWEN_BX_VERSION", DEFAULT_QWEN_BX_VERSION).strip() or DEFAULT_QWEN_BX_VERSION
    # 短效 access_token 只能由真浏览器签发（HTTP signin 只发长期 token，且 auth 域
    # 被 AWSC 挑战）。关闭后回落到纯 HTTP 长期 token 路径（上游现在会判 401）。
    QWEN_BROWSER_TOKEN_ENABLED: bool = os.getenv("QWEN_BROWSER_TOKEN_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    # 后台续期轮询间隔：token 只有 900s，提前 120s 续，60s 轮询足够
    QWEN_TOKEN_REFRESH_INTERVAL_SECONDS: float = float(os.getenv("QWEN_TOKEN_REFRESH_INTERVAL_SECONDS", 60))
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
    # 首帧/空闲超时。上游风控软拦会以 200 + 裸 JSON 返回（非 SSE），代码会立即判错，
    # 不再干等；此处只需覆盖「真·静默断流」，故默认 20s 而非 90s。
    QWEN_UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS: float = float(
        os.getenv("QWEN_UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS", 20)
    )
    # 采用上游 SSE 每帧下发的累计 usage（input_tokens/output_tokens）替代本地 tiktoken 估算。
    # 关闭后完全回落到本地估算，行为与接入前一致。
    QWEN_UPSTREAM_USAGE_ENABLED: bool = os.getenv("QWEN_UPSTREAM_USAGE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    # 上游 input_tokens 是否已包含 context_attachment 的 token。
    # 默认 false（保守继续叠加 extra_prompt_tokens，避免少扣）；实测确认后可打开。
    QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS: bool = os.getenv(
        "QWEN_UPSTREAM_USAGE_INCLUDES_ATTACHMENT_TOKENS", "false"
    ).lower() in {"1", "true", "yes", "on"}
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
    # 生图回源本地缓存 TTL（秒）。已弃用：生成物清理改由 GENERATED_MEDIA_TTL_SECONDS 统一控制
    GENERATED_IMAGE_TTL_SECONDS: int = int(os.getenv("GENERATED_IMAGE_TTL_SECONDS", 3600))
    # 生成物（图片/视频）保留秒数；0 = 永久保留（媒体库语义）
    GENERATED_MEDIA_TTL_SECONDS: int = int(os.getenv("GENERATED_MEDIA_TTL_SECONDS", 0))
    # 视频回源下载体积上限，超限退回 CDN 直链，避免打爆内存
    GENERATED_VIDEO_MAX_BYTES: int = int(os.getenv("GENERATED_VIDEO_MAX_BYTES", 268435456))
    # /v1/images/generations 默认上游模型；升级时只改 env，无需改代码
    IMAGE_GENERATION_MODEL: str = os.getenv("IMAGE_GENERATION_MODEL", DEFAULT_QWEN_MAX_MODEL).strip() or DEFAULT_QWEN_MAX_MODEL
    # t2i 请求 messages[].extra.meta.model 的取值，即真正出图的模型。
    # 默认 qwen-image-3.0-pro（官网抓包值）：留空会让上游按 chat 级 model 自行
    # 路由，出图模型不可控，故给一个明确的默认值。
    IMAGE_GENERATION_META_MODEL: str = os.getenv("IMAGE_GENERATION_META_MODEL", "qwen-image-3.0-pro").strip()
    CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS: int = int(os.getenv("CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS", 60))
    # 多模态附件：图片 / 音频 / 视频默认走 data URI 内联（实测上游 files[i].url 直接吃
    # data URI，且 /api/v2/files/getstsToken 已被阿里云 WAF 拼图验证码拦住）。
    INLINE_ATTACHMENT_ENABLED: bool = os.getenv("INLINE_ATTACHMENT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    # 单个内联附件原始字节上限（护栏）。真正的门控是 url 字符数
    # （INLINE_ATTACHMENT_MAX_URL_CHARS），因为上游限制的是 base64 后的 url 长度。
    INLINE_ATTACHMENT_MAX_BYTES: int = int(os.getenv("INLINE_ATTACHMENT_MAX_BYTES", 10485760))
    # 上游 messages[0].files[i].url 是 max_length=8192 的字符串（实测 8182 通过 /
    # 8198 被拒 "url too long"）。内联只够装约 6KB 二进制，实现里另留 92 字符边距。
    INLINE_ATTACHMENT_MAX_URL_CHARS: int = int(os.getenv("INLINE_ATTACHMENT_MAX_URL_CHARS", 8192))
    # 是否允许下载远程 http(s) URL 附件（默认关闭）。开启后网关会替调用方发起
    # 出网请求，存在 SSRF 风险（内网探测 / 云元数据凭证泄露），故默认 false；
    # 开启时也会做私网地址校验 + 每跳重定向复检。
    INLINE_ATTACHMENT_REMOTE_FETCH: bool = os.getenv("INLINE_ATTACHMENT_REMOTE_FETCH", "false").lower() in {"1", "true", "yes", "on"}
    # 模态门控：true=模型不支持该模态时直接 400；false=只丢弃该附件并 warn。
    MULTIMODAL_STRICT: bool = os.getenv("MULTIMODAL_STRICT", "true").lower() in {"1", "true", "yes", "on"}
    # 模型模态覆盖表（JSON，如 {"qwen3.8-max":["text","image","video","audio"]}）。
    MODEL_MODALITIES_OVERRIDE: str = os.getenv("MODEL_MODALITIES_OVERRIDE", "")

    # ---- 浏览器上传代理（过阿里云拼图验证码换 getstsToken 的 STS）----
    # 远程 CDP 端点（browserless / 远程 Chrome），如 http://10.6.6.4:3000。
    # 留空 = 用本进程内置的 Playwright Chromium（local 模式）；两者都不想要就设
    # UPLOAD_VIA_BROWSER=never（内联装不下的附件直接 400 attachment_too_large）。
    UPLOAD_CDP_ENDPOINT: str = os.getenv("UPLOAD_CDP_ENDPOINT", "").strip()
    # auto  = endpoint 非空走远程 CDP，为空则在进程内启动 Chromium
    # local = 强制用内置 Chromium（忽略 endpoint）
    # remote= 强制用远程 CDP（endpoint 为空则整条链路关闭）
    UPLOAD_BROWSER_MODE: str = os.getenv("UPLOAD_BROWSER_MODE", "auto").strip().lower() or "auto"
    # 浏览器上传总开关。**默认 never（关闭）**：
    #   1. 该通道的通过率尚未验证（local 模式是容器里的 Linux + SwiftShader 指纹，
    #      过拼图概率低），开了也可能是白等；
    #   2. 失败代价极高——单次要「起 Chromium + 注入登录态 + 解拼图」，几百毫秒到
    #      几十秒，还会烧账号风控分；一次拖死就是请求级超时（实测 3s → 120s）；
    #   3. 只有调用方真实附件这一条路径需要它，且内联 / HTTP 上传本来就有兜底。
    # 要用的部署显式设 UPLOAD_VIA_BROWSER=auto|always 打开。
    # auto  = 允许浏览器上传（仅用于调用方真实附件，具体走 remote / local 见上一项）
    # always= 与 auto 行为一致（上传路径优先尝试浏览器），保留语义便于运维切换
    # never = 完全不用浏览器上传（默认）
    UPLOAD_VIA_BROWSER: str = os.getenv("UPLOAD_VIA_BROWSER", "never").strip().lower() or "never"
    # 一次上传最多解几轮拼图验证码（实测 1~2 轮，留 3 轮余量）
    UPLOAD_CAPTCHA_MAX_ROUNDS: int = int(os.getenv("UPLOAD_CAPTCHA_MAX_ROUNDS", 3))
    # 等验证码滑块出现的超时（秒）
    UPLOAD_CAPTCHA_TIMEOUT_SECONDS: int = int(os.getenv("UPLOAD_CAPTCHA_TIMEOUT_SECONDS", 30))
    # 鼠标位移 ↔ 拼图位移的映射：auto（默认，优先读标定缓存）/ linear / quadratic
    UPLOAD_DRAG_MAPPING: str = os.getenv("UPLOAD_DRAG_MAPPING", "auto").strip().lower() or "auto"
    # 单次浏览器上传（连接 + 验证码 + 取 STS）的总超时（秒）
    UPLOAD_BROWSER_TIMEOUT_SECONDS: int = int(os.getenv("UPLOAD_BROWSER_TIMEOUT_SECONDS", 120))
    # 浏览器上传路径的**独立硬超时**（秒，默认 45）。整段「读字节 + 换 STS + OSS PUT」
    # 都受它约束，超时按 BrowserUploadUnavailable 处理并降级到 HTTP 上传 / 内联。
    # 必须显著小于上游请求超时：浏览器卡住时宁可降级，也不能把请求拖到客户端超时。
    UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS: float = float(os.getenv("UPLOAD_BROWSER_HARD_TIMEOUT_SECONDS", 45))
    # 拖动比例标定缓存文件（auto 模式读写）
    UPLOAD_DRAG_CALIBRATION_FILE: str = os.getenv(
        "UPLOAD_DRAG_CALIBRATION_FILE", str(DATA_DIR / "drag_calibration.json")
    )
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
# 目标模型来自 Settings，避免每次上游换代都要改代码。
MODEL_MAP = {
    "qwen-max": settings.QWEN_MAX_MODEL,
    "qwen-plus": settings.QWEN_PLUS_MODEL,
}


def resolve_model(name: str) -> str:
    """将短别名解析为上游模型，同时容忍调用方的空白和大小写差异。"""
    raw_name = str(name or "").strip()
    if not raw_name:
        return raw_name
    return MODEL_MAP.get(raw_name, MODEL_MAP.get(raw_name.lower(), raw_name))


def default_prewarm_models() -> tuple[str, ...]:
    """返回与短别名一致的默认预热模型集合。"""
    return tuple(
        dict.fromkeys(
            model
            for model in (MODEL_MAP.get("qwen-max"), MODEL_MAP.get("qwen-plus"))
            if str(model or "").strip()
        )
    )


GENERIC_QWEN_CODE_MODELS = {
    "qwen3.6-plus",
    "qwen-plus",
    "qwen-max",
    "qwen",
    settings.QWEN_MAX_MODEL,
    settings.QWEN_PLUS_MODEL,
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

    generic_models = GENERIC_QWEN_CODE_MODELS | {
        str(model).strip()
        for model in MODEL_MAP.values()
        if str(model or "").strip()
    }
    if tool_enabled and settings.QWEN_CODE_FORCE_CODER_FOR_TOOL_CALLS and resolved_model in generic_models:
        return True
    if coding_intent and settings.QWEN_CODE_FORCE_CODER_FOR_CODING_TASKS and resolved_model in generic_models:
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
