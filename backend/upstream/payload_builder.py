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
    # 与 create_chat / 浏览器 softs 对齐：上游期望毫秒时间戳
    ts = int(time.time() * 1000)
    fid = uuid.uuid4().hex
    child_id = uuid.uuid4().hex

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
        feature_config = {
            **CUSTOM_TOOL_COMPAT_FEATURE_CONFIG,
            **(CUSTOM_TOOL_LOW_LATENCY_OVERRIDES if has_custom_tools else {}),
            # Our Anthropic/OpenAI bridge relies on textual JSON/XML tool directives
            # that are parsed locally. Enabling Qwen native function_calling here causes
            # upstream interception such as `Tool Read/Bash does not exists.` for custom
            # local tools that only exist in the bridge layer.
            "function_calling": False,
            # Additional safeguards to prevent tool call interception
            "enable_tools": False,
            "enable_function_call": False,
            "tool_choice": "none",
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
                "fid": fid,
                "parentId": None,
                "childrenIds": [child_id],
                "role": "user",
                "content": content,
                "user_action": "chat",
                "files": files or [],
                "timestamp": ts,
                "models": [model],
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
