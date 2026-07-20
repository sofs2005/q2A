from __future__ import annotations

import calendar
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import oss2


_DEFAULT_UPLOAD_RATE_LIMIT_RETRY_SECONDS = 3600.0
# 官方签名 URL 常见 x-oss-expires=300；复用时提前 30s 作废，避免临界过期。
_SIGNED_URL_EXPIRY_SAFETY_SECONDS = 30.0
_OSS_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$")


class FileUploadRateLimitedError(Exception):
    """上游文件上传命中每日配额（getstsToken 返回 code=RateLimited）。

    retry_after_seconds 表示建议在多少秒内不再用该账号上传。
    """

    def __init__(self, retry_after_seconds: float, message: str = "") -> None:
        self.retry_after_seconds = max(0.0, float(retry_after_seconds or 0))
        super().__init__(message or f"file upload rate limited; retry after {self.retry_after_seconds:.0f}s")


def _upload_filetype_from_content_type(content_type: str) -> str:
    lowered = (content_type or "").lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    return "file"


def _remote_file_class_from_content_type(content_type: str) -> str:
    lowered = (content_type or "").lower()
    if lowered.startswith("image/"):
        return "vision"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    return "document"


def _remote_ref_type_from_content_type(content_type: str) -> str:
    lowered = (content_type or "").lower()
    if lowered.startswith("image/"):
        return "image"
    return "file"


def _normalize_sign_region(region: str) -> str:
    region = (region or "").strip()
    if region.startswith("oss-"):
        return region[len("oss-"):]
    return region


def signed_url_expires_at(url: str, *, now: float | None = None) -> float | None:
    """从 OSS 签名 URL 解析绝对过期时间（unix 秒）。

    官方 file_url 形如：
    ...?x-oss-date=20260720T072213Z&x-oss-expires=300&x-oss-signature=...
    过期时刻 = x-oss-date + x-oss-expires；缺 date 时用 now + expires。
    """
    text = str(url or "").strip()
    if not text:
        return None
    try:
        query = parse_qs(urlparse(text).query)
    except Exception:
        return None

    def _first(name: str) -> str:
        values = query.get(name) or query.get(name.lower()) or []
        return str(values[0]).strip() if values else ""

    expires_raw = _first("x-oss-expires")
    if not expires_raw:
        return None
    try:
        expires_seconds = float(expires_raw)
    except (TypeError, ValueError):
        return None
    if expires_seconds <= 0:
        return None

    base = now if now is not None else time.time()
    date_raw = _first("x-oss-date")
    if date_raw:
        matched = _OSS_DATE_RE.match(date_raw)
        if matched:
            year, month, day, hour, minute, second = (int(part) for part in matched.groups())
            # OSS 签名日期为 UTC。
            base = float(calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0)))
    return base + expires_seconds


def safe_signed_url_cache_expires_at(url: str, *, now: float | None = None, safety_seconds: float = _SIGNED_URL_EXPIRY_SAFETY_SECONDS) -> float | None:
    """缓存复用截止时间：签名过期前 safety_seconds 秒作废。"""
    absolute = signed_url_expires_at(url, now=now)
    if absolute is None:
        return None
    return absolute - max(0.0, float(safety_seconds or 0.0))


def _looks_like_dns_or_connect_failure(exc: Exception) -> bool:
    text = str(exc or "")
    lowered = text.lower()
    markers = (
        "nameresolutionerror",
        "temporary failure in name resolution",
        "failed to resolve",
        "max retries exceeded",
        "connectionerror",
        "newconnectionerror",
    )
    return any(marker in lowered for marker in markers)


def _build_regional_endpoint(bucketname: str, endpoint: str, region: str) -> str | None:
    endpoint = str(endpoint or "").strip()
    bucketname = str(bucketname or "").strip()
    region = str(region or "").strip()
    if not endpoint or not bucketname or not region:
        return None
    if "oss-accelerate.aliyuncs.com" not in endpoint:
        return None
    regional_host = f"oss-{region}.aliyuncs.com"
    bucket_prefix = f"{bucketname}."
    if endpoint.startswith(bucket_prefix):
        return f"{bucket_prefix}{regional_host}"
    return regional_host


class UpstreamFileUploader:
    def __init__(self, client, settings):
        self.client = client
        self.settings = settings

    @staticmethod
    def _build_bucket(auth, endpoint: str, bucketname: str, region: str):
        return oss2.Bucket(
            auth,
            f"https://{endpoint}",
            bucketname,
            region=region,
        )

    async def upload_local_file(self, acc, local_file_meta: dict[str, Any]) -> dict[str, Any]:
        filename = local_file_meta["filename"]
        file_path = local_file_meta["path"]
        content_type = local_file_meta.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        raw = Path(file_path).read_bytes()

        sts_resp = await self.client._request_json(
            "POST",
            "/api/v2/files/getstsToken",
            acc.token,
            {
                "filename": filename,
                # 官网网页端传字符串；与官方抓包保持一致。
                "filesize": str(len(raw)),
                "filetype": _upload_filetype_from_content_type(content_type),
            },
            timeout=20.0,
            account=acc,
        )
        if sts_resp.get("status") != 200:
            raise RuntimeError(f"getstsToken failed: {sts_resp.get('status')} {sts_resp.get('body', '')[:200]}")
        sts_data = json.loads(sts_resp.get("body", "{}"))
        sts = (sts_data.get("data") or {}) if isinstance(sts_data, dict) else {}
        if str(sts.get("code") or "") == "RateLimited":
            try:
                hours = float(sts.get("num"))
            except (TypeError, ValueError):
                hours = 0.0
            retry_after = hours * 3600.0 if hours > 0 else _DEFAULT_UPLOAD_RATE_LIMIT_RETRY_SECONDS
            raise FileUploadRateLimitedError(
                retry_after_seconds=retry_after,
                message=str(sts.get("details") or "file upload rate limited"),
            )
        file_id = sts.get("file_id")
        file_path_remote = sts.get("file_path", "")
        bucketname = sts.get("bucketname", "")
        endpoint = sts.get("endpoint", "")
        file_url = sts.get("file_url", "")
        region = _normalize_sign_region(sts.get("region", ""))
        access_key_id = sts.get("access_key_id", "")
        access_key_secret = sts.get("access_key_secret", "")
        security_token = sts.get("security_token", "")
        if not file_id or not file_path_remote or not bucketname or not endpoint:
            raise RuntimeError(f"getstsToken missing file data: {sts_data}")

        auth = oss2.StsAuth(access_key_id, access_key_secret, security_token, auth_version='v4')
        upload_endpoint = endpoint
        bucket = self._build_bucket(auth, upload_endpoint, bucketname, region)
        try:
            put_result = bucket.put_object(
                file_path_remote,
                raw,
                headers={"Content-Type": content_type},
            )
        except Exception as exc:
            fallback_endpoint = _build_regional_endpoint(bucketname, endpoint, region)
            if not fallback_endpoint or fallback_endpoint == upload_endpoint or not _looks_like_dns_or_connect_failure(exc):
                raise
            bucket = self._build_bucket(auth, fallback_endpoint, bucketname, region)
            put_result = bucket.put_object(
                file_path_remote,
                raw,
                headers={"Content-Type": content_type},
            )
            upload_endpoint = fallback_endpoint
        if getattr(put_result, 'status', None) not in (200, 201):
            raise RuntimeError(f"OSS put_object failed: status={getattr(put_result, 'status', None)}")

        remote_type = _remote_ref_type_from_content_type(content_type)
        parse_status = "success"
        if remote_type != "image":
            parse_resp = await self.client._request_json(
                "POST",
                "/api/v2/files/parse",
                acc.token,
                {"file_id": file_id},
                timeout=20.0,
                account=acc,
            )
            if parse_resp.get("status") != 200:
                raise RuntimeError(f"files/parse failed: {parse_resp.get('status')} {parse_resp.get('body', '')[:200]}")

            deadline = time.time() + self.settings.CONTEXT_UPLOAD_PARSE_TIMEOUT_SECONDS
            parse_status = "pending"
            while time.time() < deadline:
                status_resp = await self.client._request_json(
                    "POST",
                    "/api/v2/files/parse/status",
                    acc.token,
                    {"file_id_list": [file_id]},
                    timeout=20.0,
                    account=acc,
                )
                if status_resp.get("status") != 200:
                    raise RuntimeError(f"files/parse/status failed: {status_resp.get('status')} {status_resp.get('body', '')[:200]}")
                status_data = json.loads(status_resp.get("body", "{}"))
                rows = status_data.get("data") or []
                row = rows[0] if isinstance(rows, list) and rows else {}
                parse_status = row.get("status", "pending")
                if parse_status == "success":
                    break
                if parse_status in ("failed", "error"):
                    raise RuntimeError(f"file parse failed: {row}")
                await __import__('asyncio').sleep(1.0)

            if parse_status != "success":
                raise RuntimeError(f"file parse timeout: {file_id}")

        user_id = file_path_remote.split('/', 1)[0] if '/' in file_path_remote else ""
        now = time.time()
        now_ms = int(now * 1000)
        put_url = str(file_url or f"https://{upload_endpoint}/{file_path_remote.lstrip('/')}")
        url_expires_at = signed_url_expires_at(put_url, now=now)
        remote_ref = {
            "type": remote_type,
            "file": {
                "created_at": now_ms,
                "data": {},
                "filename": filename,
                "hash": None,
                "id": file_id,
                "user_id": user_id,
                "meta": {
                    "name": filename,
                    "size": len(raw),
                    "content_type": content_type,
                    "parse_meta": {"parse_status": parse_status},
                },
                "update_at": now_ms,
            },
            "id": file_id,
            "url": put_url,
            "name": filename,
            "collection_name": "",
            "progress": 0,
            "status": "uploaded",
            "greenNet": "success",
            "size": len(raw),
            "error": "",
            "itemId": str(uuid.uuid4()),
            "file_type": content_type,
            "showType": remote_type,
            "file_class": _remote_file_class_from_content_type(content_type),
            "uploadTaskId": str(uuid.uuid4()),
        }
        return {
            "remote_file_id": file_id,
            "remote_object_key": file_path_remote,
            "filename": filename,
            "content_type": content_type,
            "parse_status": parse_status,
            "url_expires_at": url_expires_at,
            "remote_ref": remote_ref,
        }

    async def delete_remote_file(self, acc, remote_meta: dict[str, Any]) -> bool:
        # Qwen web upload delete API has not been fully confirmed yet.
        return False
