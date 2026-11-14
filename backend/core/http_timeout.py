"""连接超时工具：把「建连阶段」和「读阶段」拆开设置。

背景（2026-12 生产实测）
------------------------
出口池里的出口随时有若干连不上（SOCKS5 connect 直接失败 / 黑洞不响应）。
一个出口从发起到失败要等满**整体**超时（~10s），而 `chats/new` 会挨个试出口，
于是 4 个死出口就是 ~40s 白等 —— 一次普通对话 108s 里有 21s 花在这里。

`curl_cffi`（≥0.6）的 `timeout` 支持 `(connect, read)` 元组：建连阶段按
connect 收紧（`EXIT_CONNECT_TIMEOUT_SECONDS`，默认 5s），读阶段保留原值
（大模型生成可能很慢，绝不能压到 5s）。

`curl_cffi` 不支持元组时（老版本 / 替身传输层）退化成「整体超时收紧到 connect
值」—— 慢一点的首字节会被截断，但绝不会再让死出口吃掉 10s。
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("qwen2api.http_timeout")

#: 支持 `timeout=(connect, read)` 元组的最低 curl_cffi 版本
_MIN_TUPLE_VERSION = (0, 6, 0)

#: 探测结果缓存：None 表示「还没探测过」
_SUPPORTS_TIMEOUT_TUPLE: bool | None = None


def _parse_version(raw: Any) -> tuple[int, int, int]:
    """把 `0.17.0` / `0.5.10b1` 这类版本串解析成三元组（解析失败按 0.0.0）。"""
    parts: list[int] = []
    for chunk in str(raw or "").split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def curl_cffi_supports_timeout_tuple() -> bool:
    """当前 curl_cffi 是否支持 `timeout=(connect, read)` 元组（结果缓存一次）。

    没装 curl_cffi（单测替身 / 纯 httpx 部署）时返回 False —— 调用方据此退化
    成「整体超时收紧」。
    """
    global _SUPPORTS_TIMEOUT_TUPLE
    if _SUPPORTS_TIMEOUT_TUPLE is not None:
        return _SUPPORTS_TIMEOUT_TUPLE
    try:
        import curl_cffi

        version = getattr(curl_cffi, "__version__", "")
        _SUPPORTS_TIMEOUT_TUPLE = _parse_version(version) >= _MIN_TUPLE_VERSION
    except Exception:  # pragma: no cover - 没装 curl_cffi 的部署
        _SUPPORTS_TIMEOUT_TUPLE = False
    return _SUPPORTS_TIMEOUT_TUPLE


def _as_positive(value: Any) -> float:
    """把配置值收紧成正浮点数；非法 / 非正值返回 0.0（表示「不限制」）。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0


def connect_read_timeout(
    read_timeout: Any,
    connect_timeout: Any = None,
) -> float | tuple[float, float]:
    """构造「建连收紧、读取不变」的超时值。

    用法（curl_cffi）：

    ```python
    session.get(url, timeout=connect_read_timeout(120, settings.EXIT_CONNECT_TIMEOUT_SECONDS))
    ```

    Args:
        read_timeout: 读阶段超时（秒），即原来传的整体超时。
        connect_timeout: 建连阶段超时（秒）；None / 非正数表示不额外约束。

    Returns:
        `(connect, read)` 元组（curl_cffi 支持时）；否则收窄后的单个浮点
        （`min(connect, read)`，元组不支持时整条请求都按 connect 值约束）。
    """
    read = _as_positive(read_timeout)
    connect = _as_positive(connect_timeout)
    if connect <= 0:
        # 没配 connect 超时：保持原有整体超时语义。
        return read if read > 0 else 30.0
    if read <= 0:
        read = connect
    if connect > read:
        # connect 比 read 还宽只会误导人：以 read 为准。
        connect = read
    if curl_cffi_supports_timeout_tuple():
        return (connect, read)
    log.debug(
        "[HttpTimeout] curl_cffi 不支持 timeout 元组，整体超时收紧到 %.1fs",
        connect,
    )
    return connect


def connect_read_timeout_seconds(read_timeout: Any, connect_timeout: Any = None) -> float:
    """同 :func:`connect_read_timeout`，但只返回**建连阶段**的秒数（日志/断言用）。"""
    value = connect_read_timeout(read_timeout, connect_timeout)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def reset_capability_cache() -> None:
    """清空「curl_cffi 是否支持元组」的探测缓存（单测用）。"""
    global _SUPPORTS_TIMEOUT_TUPLE
    _SUPPORTS_TIMEOUT_TUPLE = None
