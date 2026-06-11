from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from backend.core.account_pool import Account
from backend.core.browser_fingerprint import fingerprint_for_account, get_session, new_session
from backend.core.config import settings
from backend.services.auth_resolver import BASE_URL, AuthResolver
from backend.upstream.payload_builder import build_chat_payload
from backend.upstream.sse_consumer import parse_sse_chunk

log = logging.getLogger("qwen2api.client")


class QwenClient:
    def __init__(self, account_pool):
        self.account_pool = account_pool
        self.auth_resolver = AuthResolver(account_pool) if account_pool is not None else None
        self.executor = None
        from backend.upstream.qwen_executor import QwenExecutor

        self.executor = QwenExecutor(self, account_pool)

    @staticmethod
    def _build_headers(
        *,
        account: Account | None = None,
        token: str | None = None,
        cookies: str | None = None,
        referer: str | None = None,
        content_type: str | None = "application/json",
        accept: str = "application/json, text/plain, */*",
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        fingerprint = fingerprint_for_account(account)
        account_cookies = str(getattr(account, "cookies", "") or "").strip() if account is not None else ""
        effective_cookies = cookies if cookies is not None else account_cookies or None
        headers = fingerprint.build_headers(
            token=token,
            cookies=effective_cookies,
            referer=referer,
            content_type=content_type,
            accept=accept,
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    @staticmethod
    def _build_chat_clear_headers(*, account: Account | None = None, token: str | None = None, cookies: str | None = None) -> dict[str, str]:
        return QwenClient._build_headers(
            account=account,
            token=token,
            cookies=cookies,
            referer=f"{BASE_URL}/settings/chats",
            extra_headers={
                "Version": "0.2.57",
                "source": "web",
                "X-Request-Id": str(uuid.uuid4()),
                "Timezone": time.strftime("%a %b %d %Y %H:%M:%S GMT%z", time.localtime()),
            },
        )

    @staticmethod
    def _build_personalization_headers(
        *,
        account: Account | None = None,
        token: str | None = None,
        cookies: str | None = None,
    ) -> dict[str, str]:
        return QwenClient._build_headers(
            account=account,
            token=token,
            cookies=cookies,
            referer=f"{BASE_URL}/settings/personalization",
            extra_headers={
                "Version": "0.2.57",
                "source": "web",
                "X-Request-Id": str(uuid.uuid4()),
                "Timezone": time.strftime("%a %b %d %Y %H:%M:%S GMT%z", time.localtime()),
            },
        )

    @staticmethod
    def _looks_like_upstream_auth_failure(status: int, body: str) -> bool:
        body_lower = (body or "").lower()
        return (
            status in {401, 403}
            or "unauthorized" in body_lower
            or "forbidden" in body_lower
            or "token" in body_lower
            or "login" in body_lower
            or "expired" in body_lower
        )

    async def _request_raw_json(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: dict | None = None,
        timeout: float | None = None,
        account: Account | None = None,
    ) -> dict:
        request_timeout = timeout if timeout is not None else settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS
        session = await get_session(fingerprint_for_account(account))
        try:
            resp = await session.request(
                method,
                f"{BASE_URL}{path}",
                headers=headers,
                json=body,
                timeout=request_timeout,
            )
            return {"status": resp.status_code, "body": getattr(resp, "text", "")}
        except Exception as e:
            return {"status": 0, "body": str(e)}

    async def _request_json(
        self,
        method: str,
        path: str,
        token: str,
        body: dict | None = None,
        timeout: float | None = None,
        account: Account | None = None,
    ) -> dict:
        request_timeout = timeout if timeout is not None else settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS
        headers = self._build_headers(account=account, token=token)
        session = await get_session(fingerprint_for_account(account))
        try:
            resp = await session.request(
                method,
                f"{BASE_URL}{path}",
                headers=headers,
                json=body,
                timeout=request_timeout,
            )
            return {"status": resp.status_code, "body": getattr(resp, "text", "")}
        except Exception as e:
            return {"status": 0, "body": str(e)}

    async def clear_all_chats(self, account) -> dict:
        email = getattr(account, "email", "")
        cookie_value = str(getattr(account, "cookies", "") or "").strip()
        token_value = str(getattr(account, "token", "") or "").strip()

        if not cookie_value and not token_value:
            return {"email": email, "status": "skipped", "reason": "missing_credentials"}

        path = "/api/v2/chats/"

        if cookie_value:
            cookie_res = await self._request_raw_json(
                "DELETE",
                path,
                self._build_chat_clear_headers(account=account, cookies=cookie_value),
                timeout=20.0,
                account=account,
            )
            if cookie_res["status"] in (200, 204):
                return {"email": email, "status": "success", "transport": "cookie", "http_status": cookie_res["status"]}
            if not self._looks_like_upstream_auth_failure(cookie_res["status"], cookie_res["body"]):
                return {
                    "email": email,
                    "status": "failed",
                    "transport": "cookie",
                    "http_status": cookie_res["status"],
                    "error": f"HTTP {cookie_res['status']}: {cookie_res['body'][:120]}",
                }

        if token_value:
            token_res = await self._request_raw_json(
                "DELETE",
                path,
                self._build_chat_clear_headers(account=account, token=token_value),
                timeout=20.0,
                account=account,
            )
            if token_res["status"] in (200, 204):
                return {"email": email, "status": "success", "transport": "token", "http_status": token_res["status"]}
            return {
                "email": email,
                "status": "failed",
                "transport": "token",
                "http_status": token_res["status"],
                "error": f"HTTP {token_res['status']}: {token_res['body'][:120]}",
            }

        return {"email": email, "status": "skipped", "reason": "missing_credentials"}

    async def _request_personalization_raw_json(
        self,
        method: str,
        path: str,
        account,
        body: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        email = getattr(account, "email", "")
        cookie_value = str(getattr(account, "cookies", "") or "").strip()
        token_value = str(getattr(account, "token", "") or "").strip()

        if not cookie_value and not token_value:
            return {"email": email, "status": "skipped", "reason": "missing_credentials"}

        request_timeout = timeout if timeout is not None else settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS

        if cookie_value:
            cookie_res = await self._request_raw_json(
                method,
                path,
                self._build_personalization_headers(account=account, cookies=cookie_value),
                body,
                timeout=request_timeout,
                account=account,
            )
            if cookie_res["status"] in (200, 204):
                return {
                    "email": email,
                    "status": "success",
                    "transport": "cookie",
                    "http_status": cookie_res["status"],
                    "body": cookie_res["body"],
                }
            if not self._looks_like_upstream_auth_failure(cookie_res["status"], cookie_res["body"]):
                return {
                    "email": email,
                    "status": "failed",
                    "transport": "cookie",
                    "http_status": cookie_res["status"],
                    "error": f"HTTP {cookie_res['status']}: {cookie_res['body'][:120]}",
                }
            if not token_value:
                return {
                    "email": email,
                    "status": "failed",
                    "transport": "cookie",
                    "http_status": cookie_res["status"],
                    "error": f"HTTP {cookie_res['status']}: {cookie_res['body'][:120]}",
                }

        if token_value:
            token_res = await self._request_raw_json(
                method,
                path,
                self._build_personalization_headers(account=account, token=token_value, cookies=""),
                body,
                timeout=request_timeout,
                account=account,
            )
            if token_res["status"] in (200, 204):
                return {
                    "email": email,
                    "status": "success",
                    "transport": "token",
                    "http_status": token_res["status"],
                    "body": token_res["body"],
                }
            return {
                "email": email,
                "status": "failed",
                "transport": "token",
                "http_status": token_res["status"],
                "error": f"HTTP {token_res['status']}: {token_res['body'][:120]}",
            }

        return {"email": email, "status": "skipped", "reason": "missing_credentials"}

    async def clear_all_chats_for_account(self, account) -> dict:
        return await self.clear_all_chats(account)

    async def delete_chat(self, token: str, chat_id: str, account: Account | None = None):
        if not token or not chat_id:
            return
        res = await self._request_json("DELETE", f"/api/v2/chats/{chat_id}", token, timeout=20.0, account=account)
        if res["status"] in (200, 204):
            return
        raise RuntimeError(f"HTTP {res['status']}: {res['body'][:120]}")

    async def list_chats(self, token: str, limit: int = 50, account: Account | None = None) -> list[dict]:
        res = await self._request_json("GET", f"/api/v2/chats?limit={int(limit)}", token, timeout=20.0, account=account)
        if res["status"] != 200:
            return []
        try:
            data = json.loads(res.get("body", "{}"))
        except Exception:
            return []
        chats = data.get("data", [])
        return chats if isinstance(chats, list) else []

    async def get_personalization_settings(self, account) -> dict:
        return await self._request_personalization_raw_json(
            "GET",
            "/api/v2/configs/setting-config",
            account,
            timeout=20.0,
        )

    async def update_personalization_settings(self, account, payload: dict) -> dict:
        return await self._request_personalization_raw_json(
            "POST",
            "/api/v2/users/user/settings/update",
            account,
            body=payload,
            timeout=20.0,
        )

    async def verify_token(self, token: str, account: Account | None = None) -> bool:
        if not token:
            return False

        try:
            res = await self._request_json(
                "GET",
                "/api/v1/auths/",
                token,
                timeout=15,
                account=account,
            )
            if res["status"] != 200:
                return False
            try:
                data = json.loads(res.get("body", "{}"))
                return data.get("role") == "user"
            except Exception as e:
                log.warning(f"[verify_token] JSON 解析失败（可能被拦截或代理异常）: {e}, status={res['status']}, text={res['body'][:100]}")
                if "aliyun_waf" in res["body"].lower() or "<!doctype" in res["body"].lower():
                    log.info("[verify_token] 遇到 WAF 拦截页面，放行交给浏览器自动化账号流程处理。")
                    return True
                return False
        except Exception as e:
            log.warning(f"[verify_token] HTTP 请求异常: {e}")
            return False

    async def list_models(self, token: str, account: Account | None = None) -> list:
        try:
            res = await self._request_json("GET", "/api/models", token, timeout=10, account=account)
            if res["status"] != 200:
                return []
            try:
                return json.loads(res.get("body", "{}")).get("data", [])
            except Exception as e:
                log.warning(f"[list_models] JSON 解析失败: {e}, status={res['status']}, text={res['body'][:100]}")
                return []
        except Exception:
            return []

    def _build_payload(self, chat_id: str, model: str, content: str, has_custom_tools: bool = False, files: list[dict] | None = None) -> dict:
        return build_chat_payload(chat_id, model, content, has_custom_tools, files=files)

    def parse_sse_chunk(self, chunk: str) -> list[dict]:
        return parse_sse_chunk(chunk)

    async def stream(self, token: str, chat_id: str, model: str, content: str, has_custom_tools: bool = False, files: list[dict] | None = None, account: Account | None = None):
        async for event in self.executor.stream(token, chat_id, model, content, has_custom_tools, files=files, account=account):
            yield event

    async def stream_chat_once(self, token: str, chat_id: str, payload: dict, account: Account | None = None) -> AsyncIterator[dict]:
        timeout = settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS
        fingerprint = fingerprint_for_account(account)
        dedicated_session = bool(getattr(settings, "QWEN_UPSTREAM_STREAM_DEDICATED_SESSION", True))
        session = new_session(fingerprint, timeout=timeout) if dedicated_session else await get_session(fingerprint)
        headers = self._build_headers(account=account, token=token, accept="text/event-stream")
        try:
            async with session.stream(
                "POST",
                f"{BASE_URL}/api/v2/chat/completions?chat_id={chat_id}",
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    body_chunks = []
                    async for chunk in resp.aiter_content():
                        body_chunks.append(chunk)
                    body_text = b"".join(body_chunks).decode(errors="replace")[:2000]
                    yield {"status": resp.status_code, "body": body_text}
                    return

                async for chunk in resp.aiter_content():
                    decoded = chunk.decode("utf-8", errors="replace")
                    if decoded:
                        yield {"chunk": decoded}
                yield {"status": "streamed"}
        except Exception as e:
            log.error(f"[QwenClient] stream_chat_once error: {e}")
            yield {"status": 0, "body": str(e)}
        finally:
            if dedicated_session:
                close = getattr(session, "close", None)
                if close is not None:
                    await close()

    async def chat_stream_events_with_retry(
        self,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        fixed_account=None,
    ):
        async for item in self.executor.chat_stream_events_with_retry(
            model,
            content,
            has_custom_tools,
            files=files,
            fixed_account=fixed_account,
        ):
            yield item
