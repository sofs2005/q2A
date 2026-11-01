"""上游远程文件 ref 构造（OSS 上传完成后用）。

实测依据：`docs/superpowers/specs/2026-10-31-multimodal-attachments-findings.md`
§2（`files[]` 真实结构）、§9（真人上传 2MB 视频后从 `GET /api/v2/chats/{id}` 取回的
真实 ref）。

本模块是**纯函数**集合：只做字段映射，不做 I/O、不碰 oss2、不碰网络，
因此可以被单测直接喂一份真实的 `getstsToken` JSON 来核对结构。

真实 ref 的字段数（§9 实测）：

- 顶层 **16** 个：``type / file / id / url / name / collection_name / progress /
  status / greenNet / size / error / itemId / file_type / showType / file_class /
  uploadTaskId``
- ``file`` 子对象 **13** 个：``created_at / data / filename / hash / id / user_id /
  meta / update_at / lastModified / name / webkitRelativePath / size / type``
- 视频的 ``greenNet`` 是 ``greening``（绿网校验中），图片 / 音频 / 文档是 ``success``。
"""

from __future__ import annotations

import mimetypes
import time
import uuid
from typing import Any


#: 上游 ref 顶层字段（顺序与真实抓包一致，便于人工 diff）。
REMOTE_REF_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "type",
    "file",
    "id",
    "url",
    "name",
    "collection_name",
    "progress",
    "status",
    "greenNet",
    "size",
    "error",
    "itemId",
    "file_type",
    "showType",
    "file_class",
    "uploadTaskId",
)

#: 上游 ref 的 `file` 子对象字段（浏览器 File 对象镜像）。
REMOTE_REF_FILE_FIELDS: tuple[str, ...] = (
    "created_at",
    "data",
    "filename",
    "hash",
    "id",
    "user_id",
    "meta",
    "update_at",
    "lastModified",
    "name",
    "webkitRelativePath",
    "size",
    "type",
)


def normalize_sign_region(region: str) -> str:
    """OSS 签名 region 归一化：上游给 `oss-cn-hangzhou`，oss2 v4 只认 `cn-hangzhou`。"""
    text = str(region or "").strip()
    if text.startswith("oss-"):
        return text[len("oss-"):]
    return text


def upload_filetype_from_content_type(content_type: str) -> str:
    """`getstsToken` 的 `filetype` 入参（网页端传 image / audio / video / file）。"""
    lowered = str(content_type or "").lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    return "file"


def remote_ref_type_from_content_type(content_type: str) -> str:
    """`type` / `showType` 字段取值（文档是 `file`，不是 `document`）。"""
    return upload_filetype_from_content_type(content_type)


def remote_file_class_from_content_type(content_type: str) -> str:
    """`file_class` 字段取值（图片是 `vision`，文档是 `document`）。"""
    lowered = str(content_type or "").lower()
    if lowered.startswith("image/"):
        return "vision"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    return "document"


def guess_content_type(filename: str, content_type: str = "") -> str:
    """mime 兜底：调用方没给或给了空串时按文件名猜。"""
    explicit = str(content_type or "").strip()
    if explicit:
        return explicit
    guessed = mimetypes.guess_type(str(filename or ""))[0]
    return guessed or "application/octet-stream"


def sts_file_id(sts: dict[str, Any]) -> str:
    """从 STS 响应里取 file_id。"""
    return str((sts or {}).get("file_id") or "").strip()


def sts_object_key(sts: dict[str, Any]) -> str:
    """从 STS 响应里取 OSS object key（上游叫 file_path）。"""
    return str((sts or {}).get("file_path") or "").strip()


def sts_file_url(sts: dict[str, Any], *, endpoint: str = "") -> str:
    """上传后 `files[i].url` 用的短链。

    优先用上游下发的 `file_url`（OSS 签名 URL，实测 ~375 字符）；缺失时按
    `https://<endpoint>/<file_path>` 拼一个。
    """
    payload = sts or {}
    url = str(payload.get("file_url") or "").strip()
    if url:
        return url
    object_key = sts_object_key(payload)
    host = str(endpoint or payload.get("endpoint") or "").strip()
    if not host or not object_key:
        return ""
    return f"https://{host}/{object_key.lstrip('/')}"


def sts_user_id(sts: dict[str, Any]) -> str:
    """`file.user_id`：网页端取 OSS object key 的第一段（就是上游 uid）。"""
    object_key = sts_object_key(sts)
    if "/" in object_key:
        return object_key.split("/", 1)[0]
    return ""


def build_remote_ref(
    *,
    sts: dict[str, Any],
    filename: str,
    content_type: str = "",
    size: int | None = None,
    parse_status: str = "success",
    now: float | None = None,
    endpoint: str = "",
) -> dict[str, Any]:
    """按真实抓包结构构造 `messages[0].files[i]` 的远程 ref。

    Args:
        sts: `getstsToken` 返回的 `data` 对象（含 file_id / file_path / file_url …）。
        filename: 附件名（`name` / `file.name` / `file.filename`）。
        content_type: 附件 mime；为空时按文件名猜。
        size: 原始字节数；None 时取 `sts["filesize"]`，再退回 0。
        parse_status: 文档解析状态，仅写进 `file.meta.parse_meta`。
        now: 当前时间戳（秒），便于测试注入。
        endpoint: 覆盖 `sts["endpoint"]`（DNS 回退后实际使用的 endpoint）。

    Returns:
        上游可直接消费的 ref 字典（16 个顶层字段 + 13 个 file 字段）。
    """
    payload = dict(sts or {})
    file_id = sts_file_id(payload)
    object_key = sts_object_key(payload)
    if not file_id:
        raise ValueError("sts payload missing file_id")
    if not object_key:
        raise ValueError("sts payload missing file_path")

    name = str(filename or "").strip() or object_key.rsplit("/", 1)[-1]
    mime = guess_content_type(name, content_type)
    if size is None:
        try:
            size = int(payload.get("filesize") or 0)
        except (TypeError, ValueError):
            size = 0
    size = max(0, int(size))
    now_seconds = float(now if now is not None else time.time())
    now_ms = int(now_seconds * 1000)

    ref_type = remote_ref_type_from_content_type(mime)
    file_class = remote_file_class_from_content_type(mime)
    url = sts_file_url(payload, endpoint=endpoint)

    return {
        "type": ref_type,
        "file": {
            "created_at": now_ms,
            "data": {},
            "filename": name,
            "hash": None,
            "id": file_id,
            "user_id": sts_user_id(payload),
            "meta": {
                "name": name,
                "size": size,
                "content_type": mime,
                "parse_meta": {"parse_status": str(parse_status or "success")},
            },
            "update_at": now_ms,
            "lastModified": now_ms,
            "name": name,
            "webkitRelativePath": "",
            "size": size,
            "type": mime,
        },
        "id": file_id,
        "url": url,
        "name": name,
        "collection_name": "",
        "progress": 0,
        "status": "uploaded",
        # 实测：视频在绿网校验完成前是 greening，其它类型直接 success。
        "greenNet": "greening" if ref_type == "video" else "success",
        "size": size,
        "error": "",
        "itemId": str(uuid.uuid4()),
        "file_type": mime,
        "showType": ref_type,
        "file_class": file_class,
        "uploadTaskId": str(uuid.uuid4()),
    }
