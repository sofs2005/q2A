from __future__ import annotations

import calendar
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import oss2

from backend.services.exit_pool import ExitPool, exit_id_from_url
from backend.services.sts_ref_builder import (
    build_remote_ref,
    guess_content_type,
    normalize_sign_region,
    remote_file_class_from_content_type,
    remote_ref_type_from_content_type,
    sts_file_url,
    upload_filetype_from_content_type,
)


log = logging.getLogger("qwen2api.upstream_file_uploader")

_DEFAULT_UPLOAD_RATE_LIMIT_RETRY_SECONDS = 3600.0
# 官方签名 URL 常见 x-oss-expires=300；复用时提前 30s 作废，避免临界过期。
_SIGNED_URL_EXPIRY_SAFETY_SECONDS = 30.0
# `getstsToken` 单次请求超时（秒）。出口不稳时靠它把「挂死」变成「快速失败换出口」。
_STS_REQUEST_TIMEOUT_SECONDS = 20.0
_OSS_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$")

# 阿里云 WAF 人机验证（滑块 / 拼图）页面特征。上传接口被拦时上游返回 HTML
# 而非 JSON，必须识别出来给明确错误，不能让 json.loads 抛 JSONDecodeError。
_WAF_CHALLENGE_MARKERS = (
    "aliyun_waf",
    "aliyuncaptcha",
    "<!doctype html",
    "sceneid=",
)


class FileUploadRateLimitedError(Exception):
    """上游文件上传命中每日配额（getstsToken 返回 code=RateLimited）。

    retry_after_seconds 表示建议在多少秒内不再用该账号上传。
    """

    def __init__(self, retry_after_seconds: float, message: str = "") -> None:
        self.retry_after_seconds = max(0.0, float(retry_after_seconds or 0))
        super().__init__(message or f"file upload rate limited; retry after {self.retry_after_seconds:.0f}s")


class UpstreamWafChallengeError(Exception):
    """上游返回阿里云 WAF 人机验证页面（滑块 / 拼图），而非 JSON。

    实测 `/api/v2/files/getstsToken` 现在 100% 撞 WAF 拼图验证码，网页端也要人工
    拖拼图。抛这个异常让上层（context_attachment_manager）优雅降级为内联文本回退，
    而不是抛 JSONDecodeError 把整条请求打成 500。
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(
            message
            or "waf_challenge: upstream returned an Aliyun WAF challenge page instead of JSON"
        )


def looks_like_waf_challenge(body: str) -> bool:
    """判断响应体是否阿里云 WAF 验证页面。"""
    lowered = str(body or "").lower()
    return any(marker in lowered for marker in _WAF_CHALLENGE_MARKERS)


def _parse_json_body(resp: dict[str, Any], *, endpoint: str) -> dict[str, Any]:
    """安全解析上游响应体：WAF 页面 → UpstreamWafChallengeError，非 JSON → RuntimeError。"""
    body = str((resp or {}).get("body") or "")
    if looks_like_waf_challenge(body):
        raise UpstreamWafChallengeError(
            f"waf_challenge: {endpoint} returned an Aliyun WAF challenge page (slider/puzzle captcha); "
            "upstream file upload is blocked by anti-bot verification"
        )
    try:
        data = json.loads(body or "{}")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{endpoint} returned non-JSON body: {body[:200]}") from exc
    return data if isinstance(data, dict) else {}


def _rate_limited_error(sts: dict[str, Any]) -> FileUploadRateLimitedError:
    """`getstsToken` 返回 `code=RateLimited` 时构造统一的配额异常。"""
    try:
        hours = float(sts.get("num"))
    except (TypeError, ValueError):
        hours = 0.0
    retry_after = hours * 3600.0 if hours > 0 else _DEFAULT_UPLOAD_RATE_LIMIT_RETRY_SECONDS
    return FileUploadRateLimitedError(
        retry_after_seconds=retry_after,
        message=str(sts.get("details") or "file upload rate limited"),
    )


def _sts_payload_usable(sts: dict[str, Any]) -> bool:
    """STS 载荷是否可用（出口池据此判断「这个出口到底成没成」）。"""
    if not isinstance(sts, dict):
        return False
    for key in ("file_id", "file_path", "bucketname", "endpoint"):
        if not str(sts.get(key) or "").strip():
            return False
    return True


def _upload_filetype_from_content_type(content_type: str) -> str:
    """`getstsToken` 的 `filetype` 入参（实现见 sts_ref_builder，这里保留旧名）。"""
    return upload_filetype_from_content_type(content_type)


def _remote_file_class_from_content_type(content_type: str) -> str:
    """`file_class` 字段取值（实现见 sts_ref_builder，这里保留旧名）。"""
    return remote_file_class_from_content_type(content_type)


def _remote_ref_type_from_content_type(content_type: str) -> str:
    """`type` / `showType` 字段取值（实现见 sts_ref_builder，这里保留旧名）。"""
    return remote_ref_type_from_content_type(content_type)


def _normalize_sign_region(region: str) -> str:
    """OSS 签名 region 归一化（实现见 sts_ref_builder，这里保留旧名）。"""
    return normalize_sign_region(region)


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
    def __init__(self, client, settings, *, exit_pool=None, exit_session_factory=None):
        """构造上传器。

        Args:
            client: `QwenClient`（提供账号指纹 / WAF cookie / `_request_json`）。
            settings: 配置对象。
            exit_pool: 可选出口池（`ExitPool`）。为空时按配置
                （`UPLOAD_EXIT_POOL` / `UPLOAD_VIA_EXIT`）自动构造；池为空或
                `UPLOAD_VIA_EXIT=never` 时保持 None —— 上传链路一行代理都不碰。
            exit_session_factory: 可选 curl_cffi 会话工厂，签名
                `async (proxy_url) -> session`；单测注入假代理层用。
        """
        self.client = client
        self.settings = settings
        self.exit_pool = exit_pool if exit_pool is not None else ExitPool.from_settings(settings)
        self._exit_session_factory = exit_session_factory

    @staticmethod
    def _build_bucket(auth, endpoint: str, bucketname: str, region: str):
        return oss2.Bucket(
            auth,
            f"https://{endpoint}",
            bucketname,
            region=region,
        )

    async def upload_local_file(self, acc, local_file_meta: dict[str, Any]) -> dict[str, Any]:
        """上传本地文件到上游 OSS（**优先走出口池取 STS**）。

        链路：`getstsToken`（池启用 → 挂代理换出口；否则直连）→ :meth:`put_bytes`
        （OSS PUT + 文档 parse + ref 构造，**全部直连**）。

        出口池全失败时抛 :class:`UpstreamWafChallengeError`（语义与旧实现一致），
        由上层 `context_attachment_manager` 继续降级到浏览器 / 直连 HTTP / 内联。
        """
        return await self._upload_local_file(acc, local_file_meta, use_exit_pool=True)

    async def upload_local_file_direct(self, acc, local_file_meta: dict[str, Any]) -> dict[str, Any]:
        """直连上传（不挂任何代理）：`getstsToken` 直连 → :meth:`put_bytes`。

        保留旧语义：没有出口池、或出口池已全部失败时，这就是「现状」那条路径。
        """
        return await self._upload_local_file(acc, local_file_meta, use_exit_pool=False)

    async def _upload_local_file(
        self,
        acc,
        local_file_meta: dict[str, Any],
        *,
        use_exit_pool: bool,
    ) -> dict[str, Any]:
        """`upload_local_file` / `upload_local_file_direct` 的共同实现。"""
        filename = local_file_meta["filename"]
        file_path = local_file_meta["path"]
        content_type = local_file_meta.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        raw = Path(file_path).read_bytes()
        sts = await self.request_sts(acc, filename, raw, content_type, use_exit_pool=use_exit_pool)
        return await self.put_bytes(acc, sts, raw, filename, content_type)

    async def request_sts(
        self,
        acc,
        filename: str,
        raw: bytes,
        content_type: str,
        *,
        use_exit_pool: bool = True,
    ) -> dict[str, Any]:
        """取 `getstsToken` 的 STS 载荷；出口池启用时逐个出口重试。

        Args:
            acc: 账号对象（`token` / 浏览器指纹 / WAF cookie）。
            filename: 文件名。
            raw: 文件字节（只为算 filesize）。
            content_type: mime。
            use_exit_pool: False 时强制直连（降级路径用）。

        Returns:
            `getstsToken` 的 `data` 字典（含 file_id / file_path / 临时密钥）。

        Raises:
            UpstreamWafChallengeError: 池里所有出口都被 WAF 拦 / 连不上。
            FileUploadRateLimitedError: 账号当日上传配额用尽。
        """
        pool = self.exit_pool if use_exit_pool else None
        if pool is not None and bool(getattr(pool, "enabled", True)):
            return await self._request_sts_via_exit_pool(pool, acc, filename, raw, content_type)
        return await self._request_sts_direct(acc, filename, raw, content_type)

    @staticmethod
    def _sts_body(filename: str, raw: bytes, content_type: str) -> dict[str, Any]:
        """`getstsToken` 请求体（官网抓包口径：filesize 传字符串）。"""
        return {
            "filename": filename,
            # 官网网页端传字符串；与官方抓包保持一致。
            "filesize": str(len(raw)),
            "filetype": _upload_filetype_from_content_type(content_type),
        }

    async def _request_sts_direct(
        self,
        acc,
        filename: str,
        raw: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        """直连取 STS（**原有公共路径，行为保持不变**）。"""
        sts_resp = await self.client._request_json(
            "POST",
            "/api/v2/files/getstsToken",
            acc.token,
            self._sts_body(filename, raw, content_type),
            timeout=_STS_REQUEST_TIMEOUT_SECONDS,
            account=acc,
        )
        if sts_resp.get("status") != 200:
            if looks_like_waf_challenge(str(sts_resp.get("body") or "")):
                raise UpstreamWafChallengeError(
                    "waf_challenge: /api/v2/files/getstsToken returned an Aliyun WAF challenge page; "
                    "upstream file upload is blocked by anti-bot verification"
                )
            raise RuntimeError(f"getstsToken failed: {sts_resp.get('status')} {sts_resp.get('body', '')[:200]}")
        sts_data = _parse_json_body(sts_resp, endpoint="/api/v2/files/getstsToken")
        sts = (sts_data.get("data") or {}) if isinstance(sts_data, dict) else {}
        if str(sts.get("code") or "") == "RateLimited":
            raise _rate_limited_error(sts)
        return sts

    async def _request_sts_via_exit_pool(
        self,
        pool,
        acc,
        filename: str,
        raw: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        """出口池取 STS：轮询挑出口，换一个干净的出口 IP 绕开出口级 WAF 惩罚。

        只在这一步挂代理；拿到 STS 之后的 OSS PUT / parse / parse-status 全部
        直连（走代理反而会被 WAF 拦，也更慢）。最多遍历池里每个出口一轮。

        Args:
            pool: 出口池（`ExitPool` 或同接口替身）。
            acc: 账号对象。
            filename: 文件名（只用于日志与请求体）。
            raw: 文件字节（算 filesize / 日志体积）。
            content_type: mime。

        Returns:
            可用的 STS 字典。

        Raises:
            UpstreamWafChallengeError: 所有出口都失败（上层据此降级）。
            FileUploadRateLimitedError: 出口健康但账号配额用尽。
        """
        body = self._sts_body(filename, raw, content_type)
        tried: set[str] = set()
        last_reason = "no_exit_available"
        while True:
            proxy_url = pool.pick(exclude=tried)
            if not proxy_url:
                break
            exit_id = exit_id_from_url(proxy_url)
            tried.add(exit_id)
            started = time.monotonic()
            try:
                resp = await self._post_getsts_via_proxy(acc, body, proxy_url)
            except Exception as exc:
                # 只记异常类型：curl 的错误串可能带 host / 凭据，绝不落日志。
                reason = f"connect_failed:{type(exc).__name__}"
                last_reason = reason
                pool.mark_failure(exit_id, reason)
                log.warning("[UploadExit] exit=%s blocked reason=%s → next", exit_id, reason)
                continue

            status = int(resp.get("status") or 0)
            body_text = str(resp.get("body") or "")
            if looks_like_waf_challenge(body_text):
                last_reason = "waf"
                pool.mark_failure(exit_id, last_reason)
                log.warning(
                    "[UploadExit] exit=%s blocked reason=waf status=%s elapsed=%.1fs → next",
                    exit_id,
                    status,
                    time.monotonic() - started,
                )
                continue
            if status != 200:
                last_reason = f"http_{status}"
                pool.mark_failure(exit_id, last_reason)
                log.warning(
                    "[UploadExit] exit=%s blocked reason=%s elapsed=%.1fs → next",
                    exit_id,
                    last_reason,
                    time.monotonic() - started,
                )
                continue
            try:
                data = json.loads(body_text or "{}")
            except (json.JSONDecodeError, ValueError):
                data = None
            sts = (data.get("data") or {}) if isinstance(data, dict) else {}
            if str(sts.get("code") or "") == "RateLimited":
                # 出口本身是好的（返回了合法 JSON），只是账号配额用尽：
                # 先记账成功，再把配额异常交给上层「换号」逻辑。
                pool.mark_success(exit_id)
                raise _rate_limited_error(sts)
            if not _sts_payload_usable(sts):
                # 带上上游给的 code（如 Unauthorized / F015），排障时一眼能看出是
                # 「出口被拦」还是「账号 token 失效」。
                code_hint = str(sts.get("code") or "")[:40]
                last_reason = f"invalid_sts:{code_hint}" if code_hint else "invalid_sts"
                pool.mark_failure(exit_id, last_reason)
                log.warning("[UploadExit] exit=%s blocked reason=%s → next", exit_id, last_reason)
                continue

            pool.mark_success(exit_id)
            log.info(
                "[UploadExit] sts ok exit=%s elapsed=%.1fs file=%s size=%s",
                exit_id,
                time.monotonic() - started,
                filename,
                len(raw),
            )
            return sts

        log.warning(
            "[UploadExit] all exits failed tried=%s last_reason=%s → waf_challenge",
            sorted(tried),
            last_reason,
        )
        raise UpstreamWafChallengeError(
            "waf_challenge: all upload exits failed "
            f"(tried={len(tried)}, last={last_reason}); "
            "upstream file upload is blocked by anti-bot verification"
        )

    async def _post_getsts_via_proxy(
        self,
        acc,
        body: dict[str, Any],
        proxy_url: str,
    ) -> dict[str, Any]:
        """经指定出口 POST `getstsToken`，返回 `{status, body}`。

        走 curl_cffi + 账号浏览器指纹（`impersonate` / UA / WAF cookie 与直连
        路径一致），只把出口换成 `proxy_url`。单测可通过
        `exit_session_factory` 注入假代理层。
        """
        from curl_cffi.requests import AsyncSession

        from backend.services.auth_resolver import BASE_URL

        headers = self._sts_headers(acc)
        if self._exit_session_factory is not None:
            session = await self._exit_session_factory(proxy_url)
        else:
            from backend.core.browser_fingerprint import fingerprint_for_account

            fingerprint = fingerprint_for_account(acc)
            session = AsyncSession(
                impersonate=fingerprint.impersonate,
                timeout=_STS_REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
                proxies={"http": proxy_url, "https": proxy_url},
            )
        try:
            response = await session.post(
                f"{BASE_URL}/api/v2/files/getstsToken",
                headers=headers,
                json=body,
                timeout=_STS_REQUEST_TIMEOUT_SECONDS,
            )
            return {
                "status": int(getattr(response, "status_code", 0) or 0),
                "body": str(getattr(response, "text", "") or ""),
            }
        finally:
            close = getattr(session, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - 关闭失败不影响上传结论
                    log.debug("[UploadExit] session close failed", exc_info=True)

    def _sts_headers(self, acc) -> dict[str, str]:
        """取 STS 用的请求头：复用账号指纹 + WAF cookie，与直连路径保持一致。"""
        token = str(getattr(acc, "token", "") or "")
        builder = getattr(self.client, "_build_headers", None)
        if callable(builder):
            try:
                return dict(builder(account=acc, token=token))
            except Exception:  # pragma: no cover - 指纹构造异常时退回最小头
                log.debug("[UploadExit] _build_headers failed", exc_info=True)
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def put_bytes(
        self,
        acc,
        sts: dict[str, Any],
        raw: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """把字节 PUT 到 STS 指定的 OSS 位置，并构造上游 `files[]` ref。

        从 :meth:`upload_local_file` 拆出来的可复用部分：**只做 PUT + 文档 parse +
        ref 构造**，不再自己调 `getstsToken`。浏览器上传代理
        （`browser_upload_proxy`）过完验证码拿到 STS 后直接复用这段逻辑，
        保证两条链路产出的 ref 完全一致。

        Args:
            acc: 账号对象（文档 parse 需要 `acc.token`）。
            sts: `getstsToken` 的 `data`（含 file_id / file_path / bucketname /
                endpoint / region / 临时密钥 / file_url）。
            raw: 原始字节。
            filename: 文件名。
            content_type: mime；为空时按文件名猜。

        Returns:
            与 :meth:`upload_local_file` 相同结构的上传结果字典
            （`remote_file_id` / `remote_object_key` / `parse_status` /
            `url_expires_at` / `remote_ref` 等）。
        """
        payload = dict(sts or {})
        mime = guess_content_type(filename, str(content_type or ""))
        file_id = str(payload.get("file_id") or "").strip()
        file_path_remote = str(payload.get("file_path") or "").strip()
        bucketname = str(payload.get("bucketname") or "").strip()
        endpoint = str(payload.get("endpoint") or "").strip()
        region = _normalize_sign_region(str(payload.get("region") or ""))
        access_key_id = payload.get("access_key_id", "")
        access_key_secret = payload.get("access_key_secret", "")
        security_token = payload.get("security_token", "")
        if not file_id or not file_path_remote or not bucketname or not endpoint:
            raise RuntimeError(f"getstsToken missing file data: {payload}")

        auth = oss2.StsAuth(access_key_id, access_key_secret, security_token, auth_version='v4')
        upload_endpoint = endpoint
        bucket = self._build_bucket(auth, upload_endpoint, bucketname, region)
        try:
            put_result = bucket.put_object(
                file_path_remote,
                raw,
                headers={"Content-Type": mime},
            )
        except Exception as exc:
            fallback_endpoint = _build_regional_endpoint(bucketname, endpoint, region)
            if not fallback_endpoint or fallback_endpoint == upload_endpoint or not _looks_like_dns_or_connect_failure(exc):
                raise
            bucket = self._build_bucket(auth, fallback_endpoint, bucketname, region)
            put_result = bucket.put_object(
                file_path_remote,
                raw,
                headers={"Content-Type": mime},
            )
            upload_endpoint = fallback_endpoint
        if getattr(put_result, 'status', None) not in (200, 201):
            raise RuntimeError(f"OSS put_object failed: status={getattr(put_result, 'status', None)}")

        file_class = _remote_file_class_from_content_type(mime)
        parse_status = "success"
        # 上游只对文档解析（`parsedFileTypes=[FILE]`）：图片 / 音频 / 视频直传即可，
        # 绝不能调 /files/parse（会报错或被当作文档处理）。
        if file_class == "document":
            parse_resp = await self.client._request_json(
                "POST",
                "/api/v2/files/parse",
                acc.token,
                {"file_id": file_id},
                timeout=20.0,
                account=acc,
            )
            if parse_resp.get("status") != 200:
                if looks_like_waf_challenge(str(parse_resp.get("body") or "")):
                    raise UpstreamWafChallengeError(
                        "waf_challenge: /api/v2/files/parse returned an Aliyun WAF challenge page"
                    )
                raise RuntimeError(f"files/parse failed: {parse_resp.get('status')} {parse_resp.get('body', '')[:200]}")
            # 200 也可能返回 WAF 页面（阿里云验证码有时以 200 落地），统一走安全解析。
            _parse_json_body(parse_resp, endpoint="/api/v2/files/parse")

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
                    if looks_like_waf_challenge(str(status_resp.get("body") or "")):
                        raise UpstreamWafChallengeError(
                            "waf_challenge: /api/v2/files/parse/status returned an Aliyun WAF challenge page"
                        )
                    raise RuntimeError(f"files/parse/status failed: {status_resp.get('status')} {status_resp.get('body', '')[:200]}")
                status_data = _parse_json_body(status_resp, endpoint="/api/v2/files/parse/status")
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

        now = time.time()
        put_url = sts_file_url(payload, endpoint=upload_endpoint)
        url_expires_at = signed_url_expires_at(put_url, now=now)
        # remote_ref 字段对齐官网抓包（16 个顶层 + 13 个 file 字段），见 sts_ref_builder。
        remote_ref = build_remote_ref(
            sts=payload,
            filename=filename,
            content_type=mime,
            size=len(raw),
            parse_status=parse_status,
            now=now,
            endpoint=upload_endpoint,
        )
        return {
            "remote_file_id": file_id,
            "remote_object_key": file_path_remote,
            "filename": filename,
            "content_type": mime,
            "parse_status": parse_status,
            "url_expires_at": url_expires_at,
            "remote_ref": remote_ref,
        }

    async def delete_remote_file(self, acc, remote_meta: dict[str, Any]) -> bool:
        # Qwen web upload delete API has not been fully confirmed yet.
        return False
