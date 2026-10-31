"""模型模态门控。

上游 `GET /api/v2/models/` 的 `info.meta.modality` 实测矩阵（2026-10-31）：

| 模型 | 支持模态 |
|---|---|
| `qwen3.8-omni-flash` / `qwen3.5-omni-plus` | text, image, video, **audio** |
| `qwen3.8-max` / `qwen3.7-plus` / `qwen3.6-plus` / `qwen3.5-plus` | text, image, video |
| `qwen3.7-max` | text |

把不支持的附件塞给上游不会得到明确报错，只会让模型「装作没收到」，
因此在网关侧提前门控：strict 模式直接拒绝，非 strict 模式只丢弃该附件。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable

log = logging.getLogger("qwen2api.model_modalities")

# 未知模型默认按「文本 + 图片 + 视频」放行：绝大多数 plus/max 系列都是这个组合，
# 只有明确已知不支持的模型（如 qwen3.7-max）才收紧。
DEFAULT_MODALITIES: frozenset[str] = frozenset({"text", "image", "video"})

SUPPORTED_MODALITIES: dict[str, set[str]] = {
    # omni 系列独有音频输入
    "qwen3.8-omni-flash": {"text", "image", "video", "audio"},
    "qwen3.5-omni-plus": {"text", "image", "video", "audio"},
    # plus / max 系列：图片 + 视频，无音频
    "qwen3.8-max": {"text", "image", "video"},
    "qwen3.7-plus": {"text", "image", "video"},
    "qwen3.6-plus": {"text", "image", "video"},
    "qwen3.5-plus": {"text", "image", "video"},
    # 纯文本模型
    "qwen3.7-max": {"text"},
}

# 常见别名 → 上表模型名；别名解析后仍查不到时走 DEFAULT_MODALITIES。
MODEL_ALIASES: dict[str, str] = {
    "qwen-max": "qwen3.8-max",
    "qwen-plus": "qwen3.7-plus",
    "qwen-omni": "qwen3.8-omni-flash",
    "qwen3.8-omni": "qwen3.8-omni-flash",
    "qwen3-omni-flash": "qwen3.8-omni-flash",
    "qwen3-omni-plus": "qwen3.5-omni-plus",
    "qwen3-max": "qwen3.7-max",
}

# 模态不被支持时推荐的替代模型（写进错误信息，方便调用方直接改 model）。
_KIND_SUGGESTIONS: dict[str, str] = {
    "image": "qwen3.8-max",
    "video": "qwen3.8-max",
    "audio": "qwen3.8-omni-flash",
    "document": "qwen3.8-max",
}

# 需要门控的模态：文档走上游上传 + parse 链路，不按 modality 判断，故不在此列。
GATED_MODALITIES: frozenset[str] = frozenset({"image", "audio", "video"})

_OVERRIDE_CACHE: dict[str, set[str]] = {}
_OVERRIDE_RAW: str | None = None


class ModalityNotSupportedError(ValueError):
    """模型不支持请求的输入模态（继承 ValueError，便于上层按 400 处理）。"""


def _truthy(value: Any, default: bool = True) -> bool:
    """解析开关型配置；None / 空串时返回默认值。"""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def strict_enabled() -> bool:
    """`MULTIMODAL_STRICT`（默认 true）：false 时只 warn + 丢弃，不报错。"""
    env_value = os.getenv("MULTIMODAL_STRICT")
    if env_value is not None:
        return _truthy(env_value, True)
    try:
        from backend.core.config import settings as runtime_settings

        return _truthy(getattr(runtime_settings, "MULTIMODAL_STRICT", None), True)
    except Exception:  # pragma: no cover - 配置模块不可用时按默认值
        return True


def _parse_overrides(raw: str) -> dict[str, set[str]]:
    """解析 `MODEL_MODALITIES_OVERRIDE`（JSON：`{"model": ["text","image"]}`）。"""
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        log.warning("[Modality] MODEL_MODALITIES_OVERRIDE is not valid JSON; ignored")
        return {}
    if not isinstance(data, dict):
        log.warning("[Modality] MODEL_MODALITIES_OVERRIDE must be a JSON object; ignored")
        return {}
    table: dict[str, set[str]] = {}
    for model, modalities in data.items():
        if not isinstance(model, str) or not model.strip():
            continue
        if isinstance(modalities, str):
            items = [part.strip() for part in modalities.split(",")]
        elif isinstance(modalities, (list, tuple, set)):
            items = [str(part).strip() for part in modalities]
        else:
            continue
        table[model.strip().lower()] = {item.lower() for item in items if item}
    return table


def _override_table() -> dict[str, set[str]]:
    """读取覆盖表；环境变量变化时自动重新解析（便于运行期调整与测试）。"""
    global _OVERRIDE_CACHE, _OVERRIDE_RAW
    raw = os.getenv("MODEL_MODALITIES_OVERRIDE", "") or ""
    if raw != _OVERRIDE_RAW:
        _OVERRIDE_CACHE = _parse_overrides(raw)
        _OVERRIDE_RAW = raw
    return _OVERRIDE_CACHE


def _normalize_model(model: str) -> str:
    """模型名归一化：去空白、小写、解析短别名。"""
    name = str(model or "").strip().lower()
    if not name:
        return ""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    try:
        from backend.core.config import resolve_model

        resolved = str(resolve_model(name) or "").strip().lower()
        if resolved and resolved != name:
            return MODEL_ALIASES.get(resolved, resolved)
    except Exception:  # pragma: no cover - 配置模块不可用时保持原名
        pass
    return name


def modalities_for(model: str) -> set[str]:
    """返回模型支持的模态集合（含环境变量覆盖与别名解析）。"""
    name = _normalize_model(model)
    overrides = _override_table()
    if name and name in overrides:
        return set(overrides[name])
    if name in SUPPORTED_MODALITIES:
        return set(SUPPORTED_MODALITIES[name])
    return set(DEFAULT_MODALITIES)


def unsupported_modalities(model: str, kinds: Iterable[str]) -> set[str]:
    """返回 `kinds` 中该模型不支持的部分（不抛错，便于调用方自行决定策略）。"""
    requested = {str(kind or "").strip().lower() for kind in kinds}
    requested = {kind for kind in requested if kind}
    if not requested:
        return set()
    return requested - modalities_for(model)


def _build_error_message(model: str, unsupported: set[str]) -> str:
    """拼装可执行的错误信息：说明哪个模态不支持 + 建议换哪个模型。"""
    ordered = [kind for kind in ("image", "video", "audio", "document") if kind in unsupported]
    ordered.extend(sorted(unsupported - set(ordered)))
    parts = []
    for kind in ordered:
        suggestion = _KIND_SUGGESTIONS.get(kind)
        if suggestion:
            parts.append(f"{kind} input (use {suggestion})")
        else:
            parts.append(f"{kind} input")
    model_name = str(model or "").strip() or "<unknown model>"
    return f"model {model_name} does not support " + ", ".join(parts)


def ensure_modalities(model: str, kinds: Iterable[str]) -> set[str]:
    """门控入口：strict 模式抛错，非 strict 模式只 warn。

    返回值是「调用方应丢弃」的模态集合（strict 模式成功时为空集）。
    """
    requested = {str(kind or "").strip().lower() for kind in kinds}
    requested = {kind for kind in requested if kind in GATED_MODALITIES}
    if not requested:
        return set()

    unsupported = requested - modalities_for(model)
    if not unsupported:
        return set()

    message = _build_error_message(model, unsupported)
    if strict_enabled():
        raise ModalityNotSupportedError(message)

    log.warning("[Modality] dropping unsupported attachments: %s", message)
    return unsupported
