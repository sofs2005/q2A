"""附件链路的 HTTP 异常处理器（FastAPI exception handlers）。

这两个处理器把 :mod:`backend.services.attachment_errors` 里的领域异常映射成对
调用方有意义的 **400**，而不是让上游的校验错误伪装成模型回答：

- :class:`~backend.services.attachment_errors.InlineMediaTooLargeError`
  → ``400 attachment_too_large``
- :class:`~backend.services.attachment_errors.UpstreamAttachmentRejectedError`
  → ``400 upstream_attachment_rejected``

**为什么单独一个模块**：处理器原先直接写在 ``backend/main.py`` 里。``main.py``
会拉起整个网关（配置单例、账号池、qwen_client …），任何只想验证「异常 → HTTP
响应」映射的测试都会被整条依赖链牵连，并被其它测试在 ``sys.modules`` 里塞的假
模块污染。把处理器放到这里后，依赖只有 ``fastapi.responses`` 与
``backend.services.attachment_errors``（纯错误类型 + 文案，无其它服务依赖），
注册则仍由 ``backend/main.py`` 完成，因此不会出现「测试绿了但线上没挂上」的假绿。
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.services.attachment_errors import (
    InlineMediaTooLargeError,
    UpstreamAttachmentRejectedError,
)

log = logging.getLogger("qwen2api")


async def inline_media_too_large_handler(
    request: Request,
    exc: InlineMediaTooLargeError,
) -> JSONResponse:
    """内联附件超出上游 url 长度上限：按 400 返回，给出可执行的处置建议。

    绝不允许把超长 url 发出去 —— 上游只会回一条 `url too long`，且以
    HTTP 200 + content 的形式返回，调用方会误以为附件已送达。

    Args:
        request: 触发异常的请求（只用于日志里的 path）。
        exc: 内联媒体超限异常。

    Returns:
        状态码 400 的 JSON 响应，``error.code == "attachment_too_large"``。
    """
    log.warning(
        "[Attachment] rejected oversized inline media path=%s filename=%s url_chars=%s limit=%s size=%s",
        request.url.path,
        exc.filename,
        exc.url_chars,
        exc.limit,
        exc.size,
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "attachment_too_large",
                "message": str(exc),
                "type": "invalid_request_error",
                "filename": exc.filename,
                "url_chars": exc.url_chars,
                "limit": exc.limit,
            }
        },
    )


async def upstream_attachment_rejected_handler(
    request: Request,
    exc: UpstreamAttachmentRejectedError,
) -> JSONResponse:
    """上游拒绝了附件载荷：按 400 返回，而不是把校验错误伪装成模型回答。

    Args:
        request: 触发异常的请求（只用于日志里的 path）。
        exc: 上游拒绝附件的异常。

    Returns:
        状态码 400 的 JSON 响应，``error.code == "upstream_attachment_rejected"``。
    """
    log.warning(
        "[Attachment] upstream rejected attachment payload path=%s upstream_code=%s",
        request.url.path,
        exc.upstream_code,
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "upstream_attachment_rejected",
                "message": str(exc),
                "type": "invalid_request_error",
                "upstream_code": exc.upstream_code,
            }
        },
    )
