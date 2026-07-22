import time
import uuid


CUSTOM_TOOL_COMPAT_FEATURE_CONFIG = {
    "thinking_enabled": True,
    "output_schema": "phase",
    "research_mode": "normal",
    "auto_thinking": True,
    "thinking_mode": "Auto",
    "thinking_format": "summary",
    "auto_search": False,
    "code_interpreter": False,
    "plugins_enabled": False,
}

CUSTOM_TOOL_LOW_LATENCY_OVERRIDES = {
    "thinking_enabled": False,
    "auto_thinking": False,
}

# 媒体生成（视频）默认宽高比，参考上游 t2v feature_config
_DEFAULT_MEDIA_RATIO = "16:9"
_VALID_MEDIA_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4"}

# 官网带附件 completions 抓包字段子集（2026-07 softs）。
# 勿塞 enable_tools / tool_choice 等网页端没有的键——带 files 时上游校验更严。
OFFICIAL_WEB_FEATURE_CONFIG = {
    "thinking_enabled": True,
    "output_schema": "phase",
    "research_mode": "normal",
    "auto_thinking": False,
    "thinking_mode": "Thinking",
    "thinking_format": "summary",
    "auto_search": True,
}


def _normalize_media_ratio(media_options: dict | None) -> str:
    if not media_options:
        return _DEFAULT_MEDIA_RATIO
    candidate = str(
        media_options.get("ratio")
        or media_options.get("aspect_ratio")
        or media_options.get("size")
        or ""
    ).strip()
    return candidate if candidate in _VALID_MEDIA_RATIOS else _DEFAULT_MEDIA_RATIO


def build_chat_payload(
    chat_id: str,
    model: str,
    content: str,
    has_custom_tools: bool = False,
    files: list[dict] | None = None,
    chat_type: str = "t2t",
    media_options: dict | None = None,
) -> dict:
    # completions 官网抓包（2026-07 softs）用秒级 timestamp；create_chat 另用毫秒，勿混。
    ts = int(time.time())
    # 官网 messages.fid / childrenIds 为带横线 UUID，非 32 位 hex
    fid = str(uuid.uuid4())
    child_id = str(uuid.uuid4())

    is_video = chat_type == "t2v"

    if is_video:
        ratio = _normalize_media_ratio(media_options)
        feature_config = {
            "thinking_enabled": False,
            "output_schema": "phase",
            "auto_thinking": False,
            "thinking_mode": "off",
            "auto_search": False,
            "code_interpreter": False,
            "function_calling": False,
            "plugins_enabled": True,
            "video_generation": True,
            "default_aspect_ratio": ratio,
        }
        message_chat_type = "t2v"
        message_meta = {
            "subChatType": "t2v",
            "mode": "video_generation",
            "aspectRatio": ratio,
            "size": ratio,
        }
    else:
        if files:
            # 有附件：严格对齐官网网页端 feature_config 子集。
            # 旧版额外键（enable_tools / tool_choice / code_interpreter 等）在无附件时
            # 往往能过；带 files 时会触发 invalid_input。
            feature_config = dict(OFFICIAL_WEB_FEATURE_CONFIG)
            if has_custom_tools:
                # bridge 文本工具：仅关原生 function_calling，不再塞网页端没有的键
                feature_config["function_calling"] = False
        else:
            feature_config = {
                **CUSTOM_TOOL_COMPAT_FEATURE_CONFIG,
                **(CUSTOM_TOOL_LOW_LATENCY_OVERRIDES if has_custom_tools else {}),
                # bridge 文本工具：关闭原生 function_calling，避免拦截成
                # `Tool Xxx does not exists.`
                "function_calling": False,
            }
        message_chat_type = "t2t"
        message_meta = {"subChatType": "t2t"}

    payload = {
        # 视频（t2v）为异步任务：task_id 仅存在于非流式 completions 响应体的
        # messages[0].extra.wanx.task_id，故视频必须用 stream:false 取回 body。
        "stream": not is_video,
        "version": "2.1",
        "incremental_output": not is_video,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                # 与官网 completions 抓包对齐：id/model 显式占位
                "id": None,
                "fid": fid,
                "parentId": None,
                "childrenIds": [child_id],
                "role": "user",
                "content": content,
                "user_action": "chat",
                "files": files or [],
                "timestamp": ts,
                "models": [model],
                "model": "",
                "chat_type": message_chat_type,
                "feature_config": feature_config,
                "extra": {"meta": message_meta},
                "sub_chat_type": message_chat_type,
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }
    if is_video:
        payload["size"] = _normalize_media_ratio(media_options)
    return payload
