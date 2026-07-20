from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx

from backend.core.account_pool import Account
from backend.core.browser_fingerprint import fingerprint_for_account, get_session, new_session
from backend.core.config import settings
from backend.services.auth_resolver import BASE_URL, AuthResolver
from backend.services.chat_id_pool import ChatIDPool
from backend.services.waf_cookie_manager import WafCookieManager
from backend.services.captcha_solver import CaptchaSolver, extract_punish_url
from backend.upstream.payload_builder import build_chat_payload
from backend.upstream.sse_consumer import parse_sse_chunk

log = logging.getLogger("qwen2api.client")


class QwenClient:
    def __init__(self, account_pool):
        self.account_pool = account_pool
        self.auth_resolver = AuthResolver(account_pool) if account_pool is not None else None
        self.chat_id_pool = ChatIDPool(self, account_pool) if account_pool is not None else None
        self.executor = None
        from backend.upstream.qwen_executor import QwenExecutor

        self.executor = QwenExecutor(self, account_pool)

    @staticmethod
    def _web_client_headers() -> dict[str, str]:
        return {
            "Version": "0.2.74",
            "bx-v": "2.5.36",
            "source": "web",
            "X-Request-Id": str(uuid.uuid4()),
            "Timezone": time.strftime("%a %b %d %Y %H:%M:%S GMT%z", time.localtime()),
            "X-Accel-Buffering": "no",
        }

    @staticmethod
    def _valid_waf_cookie(account: Account | None) -> str:
        """返回账号未过期的 WAF cookie（acw_tc），过期或缺失则返回空串。"""
        if account is None:
            return ""
        waf = str(getattr(account, "waf_cookies", "") or "").strip()
        if not waf:
            return ""
        expires = float(getattr(account, "waf_cookies_expires_at", 0) or 0)
        if expires and expires <= time.time():
            return ""
        return waf

    @staticmethod
    def _extract_acw_tc(resp) -> str:
        """从上游响应提取 acw_tc（Set-Cookie）；缺失或异常返回空串。"""
        try:
            cookies = getattr(resp, "cookies", None)
            if cookies is None:
                return ""
            return str(cookies.get("acw_tc", "") or "")
        except Exception:
            return ""

    @staticmethod
    def _merge_cookie_header(*cookie_strings: str) -> str:
        """按 cookie 名合并多个 cookie 串并去重，靠后的同名值覆盖靠前的。"""
        merged: dict[str, str] = {}
        for cookie_string in cookie_strings:
            for part in str(cookie_string or "").split(";"):
                part = part.strip()
                if not part or "=" not in part:
                    continue
                name, value = part.split("=", 1)
                name = name.strip()
                if name:
                    merged[name] = value.strip()
        return "; ".join(f"{name}={value}" for name, value in merged.items())

    @staticmethod
    def _header_diagnostics(*, account: Account | None, headers: dict[str, str]) -> dict[str, Any]:
        fingerprint = fingerprint_for_account(account)
        cookie_header = str(headers.get("Cookie", "") or "")
        cookie_names = []
        for part in cookie_header.split(";"):
            name = part.strip().split("=", 1)[0].strip()
            if name:
                cookie_names.append(name)
        return {
            "email": getattr(account, "email", "") if account is not None else "",
            "fingerprint_id": getattr(account, "fingerprint_id", "") if account is not None else fingerprint.id,
            "impersonate": fingerprint.impersonate,
            "has_cookie": bool(cookie_header),
            "cookie_names": cookie_names,
            "has_authorization": bool(headers.get("Authorization")),
            "authorization_mode": "bearer" if headers.get("Authorization") else "none",
            "has_sec_ch_ua": bool(headers.get("sec-ch-ua")),
            "user_agent_family": fingerprint.browser,
            "user_agent": headers.get("User-Agent", ""),
            "version": headers.get("Version", ""),
            "bx_version": headers.get("bx-v", ""),
            "source": headers.get("source", ""),
        }

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
        base_cookies = cookies if cookies is not None else account_cookies
        # acw_tc 等 WAF cookie 始终合并注入，绕过阿里风控接入层
        effective_cookies = QwenClient._merge_cookie_header(base_cookies, QwenClient._valid_waf_cookie(account)) or None
        headers = fingerprint.build_headers(
            token=token,
            cookies=effective_cookies,
            referer=referer,
            content_type=content_type,
            accept=accept,
        )
        headers.update(QwenClient._web_client_headers())
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
        )

    @staticmethod
    def _build_chat_transport_headers(*, account: Account | None = None, token: str | None = None, accept: str = "application/json, text/plain, */*", referer: str | None = None) -> dict[str, str]:
        fingerprint = fingerprint_for_account(account)
        headers = {
            "Authorization": f"Bearer {token}" if token else "",
            "X-Request-Id": str(uuid.uuid4()),
            "User-Agent": fingerprint.user_agent,
            "Accept": accept,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": referer or f"{BASE_URL}/",
            "Origin": BASE_URL,
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        }
        # sec-ch-ua 系列为 Chromium 专属，且版本须与 TLS impersonate 层一致，
        # 否则高端风控可借 JA3/JA4 与 UA 交叉校验识破伪装；非 Chromium 指纹不发送
        if fingerprint.sec_ch_ua:
            headers["sec-ch-ua"] = fingerprint.sec_ch_ua
            headers["sec-ch-ua-mobile"] = fingerprint.sec_ch_ua_mobile
            headers["sec-ch-ua-platform"] = fingerprint.platform
        headers["sec-fetch-dest"] = "empty"
        headers["sec-fetch-mode"] = "cors"
        headers["sec-fetch-site"] = "same-origin"
        headers.update(QwenClient._web_client_headers())
        if not token:
            headers.pop("Authorization", None)
        cookie_parts: list[str] = []
        # 登录态 cookie 仍受开关控制
        if bool(getattr(settings, "QWEN_CHAT_TRANSPORT_SEND_COOKIES", False)):
            account_cookies = str(getattr(account, "cookies", "") or "").strip()
            if account_cookies:
                cookie_parts.append(account_cookies)
        # acw_tc 等 WAF cookie 不受开关限制，始终注入以绕过阿里风控
        waf_cookie = QwenClient._valid_waf_cookie(account)
        if waf_cookie:
            cookie_parts.append(waf_cookie)
        merged_cookie = QwenClient._merge_cookie_header(*cookie_parts)
        if merged_cookie:
            headers["Cookie"] = merged_cookie
        return headers

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

    @staticmethod
    def _looks_like_waf_challenge(status: int, body: str) -> bool:
        body_lower = (body or "").lower()
        if "aliyun_waf" in body_lower or "x5sec" in body_lower or "_____tmd_____" in body_lower:
            return status in {200, 403}
        if "<!doctype" in body_lower or "<html" in body_lower:
            return status in {200, 403}
        return False

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
            return {"status": resp.status_code, "body": getattr(resp, "text", ""), "acw_tc": self._extract_acw_tc(resp)}
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
        chat_transport: bool = False,
        referer: str | None = None,
        retry_waf: bool = False,
    ) -> dict:
        request_timeout = timeout if timeout is not None else settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS
        session = await get_session(fingerprint_for_account(account))

        async def _send_once() -> dict:
            headers = self._build_chat_transport_headers(account=account, token=token, referer=referer) if chat_transport else self._build_headers(account=account, token=token, referer=referer)
            log.info("[QwenClient] request_headers path=%s diag=%s", path, self._header_diagnostics(account=account, headers=headers))
            try:
                resp = await session.request(
                    method,
                    f"{BASE_URL}{path}",
                    headers=headers,
                    json=body,
                    timeout=request_timeout,
                )
                return {"status": resp.status_code, "body": getattr(resp, "text", ""), "acw_tc": self._extract_acw_tc(resp)}
            except Exception as e:
                return {"status": 0, "body": str(e)}

        if retry_waf and account is not None:
            try:
                await WafCookieManager.get_instance().get_cookies(account)
            except Exception as waf_err:
                log.warning("[QwenClient] WAF cookie refresh failed path=%s email=%s error=%s", path, getattr(account, "email", ""), waf_err)

        res = await _send_once()
        if not (retry_waf and account is not None and self._looks_like_waf_challenge(res["status"], res["body"])):
            return res

        log.warning("[QwenClient] WAF challenge detected path=%s email=%s, refreshing cookie and retrying once", path, getattr(account, "email", ""))
        waf_mgr = WafCookieManager.get_instance()
        try:
            waf_mgr.mark_expired(account)
            await waf_mgr.get_cookies(account)
        except Exception as waf_err:
            log.warning("[QwenClient] WAF retry refresh failed path=%s email=%s error=%s", path, getattr(account, "email", ""), waf_err)
            return res
        return await _send_once()

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
        res = await self._request_json("DELETE", f"/api/v2/chats/{chat_id}", token, timeout=20.0, account=account, retry_waf=True)
        if res["status"] in (200, 204) and not self._looks_like_waf_challenge(res["status"], res["body"]):
            return
        log.warning("[delete_chat] failed email=%s chat_id=%s status=%s body=%s", getattr(account, "email", ""), chat_id, res["status"], res["body"][:120])
        raise RuntimeError(f"HTTP {res['status']}: {res['body'][:120]}")

    async def list_chats(self, token: str, limit: int = 50, account: Account | None = None) -> list[dict]:
        res = await self._request_json("GET", f"/api/v2/chats?limit={int(limit)}", token, timeout=20.0, account=account, retry_waf=True)
        if res["status"] != 200 or self._looks_like_waf_challenge(res["status"], res["body"]):
            log.warning("[list_chats] failed email=%s status=%s body=%s", getattr(account, "email", ""), res["status"], res["body"][:120])
            return []
        try:
            data = json.loads(res.get("body", "{}"))
        except Exception:
            return []
        chats = data.get("data", [])
        return chats if isinstance(chats, list) else []

    async def complete_chat_once(self, token: str, chat_id: str, payload: dict, account: Account | None = None, timeout: float | None = None) -> dict:
        """非流式 completions（用于视频 t2v）：返回 {status, body}，body 为完整 JSON 文本。

        视频任务的 task_id 嵌在响应体 messages[0].extra.wanx.task_id，SSE 流不会下发，
        故必须用 stream:false 的一次性 POST 取回整个响应体。
        """
        request_timeout = timeout if timeout is not None else settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS
        return await self._request_json(
            "POST",
            f"/api/v2/chat/completions?chat_id={chat_id}",
            token,
            body=payload,
            timeout=request_timeout,
            account=account,
            chat_transport=True,
        )

    async def get_vision_task_status(self, token: str, task_id: str, account: Account | None = None, timeout: float = 30.0) -> dict:
        """轮询视频/视觉生成任务状态。返回 {status, body}。"""
        return await self._request_json(
            "GET",
            f"/api/v1/tasks/status/{task_id}",
            token,
            timeout=timeout,
            account=account,
        )

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

    @staticmethod
    def _is_punish_response(status_code: int, body_text: str, headers: dict | None = None) -> bool:
        """检测响应是否为 x5sec punish 滑块挑战页。"""
        if status_code != 200:
            return False
        body_lower = (body_text or "").lower()
        if "_____tmd_____" in body_lower:
            return True
        if "<script>" in body_text[:500] and "x5sec" in body_lower:
            return True
        if headers and headers.get("bxpunish") == "1":
            return True
        return False

    async def _handle_punish_and_retry(
        self,
        token: str,
        chat_id: str,
        payload: dict,
        account: Account | None,
        body_text: str,
        timeout: float,
    ) -> AsyncIterator[dict]:
        """检测到 x5sec punish 后尝试滑块突破并重试 HTTP completions。

        成功则 yield 重试后的正常响应；失败则标记账号冷却 1800s 并 yield 403。
        """
        punish_url = extract_punish_url(body_text)
        log.warning(
            "[stream_chat_once] x5sec punish detected chat_id=%s punish_url=%s",
            chat_id, (punish_url or "")[:120] if punish_url else "None",
        )

        solver = CaptchaSolver.get_instance()
        pass_cookies = await solver.solve_punish(
            punish_url=punish_url or "",
            account_token=token,
        )

        if pass_cookies:
            log.info(
                "[stream_chat_once] captcha solved cookies=%s, retrying HTTP completions",
                list(pass_cookies.keys()),
            )
            if account:
                waf_mgr = WafCookieManager.get_instance()
                waf_mgr.update_cookies(account, pass_cookies)

            retry_headers = self._build_chat_transport_headers(
                account=account, token=token, accept="text/event-stream",
                referer=f"{BASE_URL}/c/{chat_id}",
            )
            # 重试走 Go-like HTTP 路径（简单可靠，避免 curl_cffi session 状态问题）
            try:
                _proxy = getattr(settings, "UPSTREAM_PROXY", "") or None
                async with httpx.AsyncClient(http2=False, follow_redirects=True, timeout=timeout, proxy=_proxy) as client:
                    async with client.stream(
                        "POST",
                        f"{BASE_URL}/api/v2/chat/completions?chat_id={chat_id}",
                        headers=retry_headers,
                        json=payload,
                        timeout=timeout,
                    ) as resp2:
                        if resp2.status_code != 200:
                            body_chunks2 = []
                            async for chunk in resp2.aiter_bytes():
                                body_chunks2.append(chunk)
                            body_text2 = b"".join(body_chunks2).decode(errors="replace")[:2000]
                            still_punish = self._is_punish_response(resp2.status_code, body_text2, dict(resp2.headers))
                            if still_punish:
                                log.warning("[stream_chat_once] retry still punished after captcha solve")
                            else:
                                log.error(f"[stream_chat_once] retry non-200 status={resp2.status_code}")
                                yield {"status": resp2.status_code, "body": body_text2}
                                return
                        else:
                            log.info("[stream_chat_once] retry success after captcha solve")
                            async for chunk in resp2.aiter_bytes():
                                decoded = chunk.decode("utf-8", errors="replace")
                                if decoded:
                                    yield {"chunk": decoded}
                            yield {"status": "streamed"}
                            return
            except Exception as retry_err:
                log.error(f"[stream_chat_once] retry request failed: {retry_err}")

        # 滑块失败或重试仍被拦截：降级为 1800s 冷却
        log.warning("[stream_chat_once] captcha failed or retry still punished, cooling down 1800s")
        if account:
            waf_mgr = WafCookieManager.get_instance()
            waf_mgr.mark_expired(account)
            if self.account_pool is not None:
                self.account_pool.mark_rate_limited(
                    account, cooldown=1800,
                    error_message="x5sec punish interception, account cooled down 1800s",
                )
        yield {
            "status": 403,
            "body": '{"error":"x5sec punish interception, account cooled down 1800s"}',
        }

    async def stream_chat_once(self, token: str, chat_id: str, payload: dict, account: Account | None = None) -> AsyncIterator[dict]:
        timeout = settings.QWEN_UPSTREAM_STREAM_TIMEOUT_SECONDS

        # 构建 headers 前先刷新 WAF cookie（过期时自动 GET 首页获取 acw_tc）
        if account:
            try:
                waf_mgr = WafCookieManager.get_instance()
                await waf_mgr.get_cookies(account)
            except Exception as waf_err:
                log.warning("[stream_chat_once] WAF cookie refresh failed: %s", waf_err)

        headers = self._build_chat_transport_headers(
            account=account, token=token, accept="text/event-stream",
            referer=f"{BASE_URL}/c/{chat_id}",
        )
        if bool(getattr(settings, "QWEN_CHAT_TRANSPORT_GO_LIKE_HTTP", True)):
            log.info(
                "[QwenClient] stream_headers chat_id=%s dedicated_session=%s diag=%s",
                chat_id,
                False,
                self._header_diagnostics(account=account, headers=headers),
            )
            try:
                _proxy = getattr(settings, "UPSTREAM_PROXY", "") or None
                async with httpx.AsyncClient(http2=False, follow_redirects=True, timeout=timeout, proxy=_proxy) as client:
                    async with client.stream(
                        "POST",
                        f"{BASE_URL}/api/v2/chat/completions?chat_id={chat_id}",
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    ) as resp:
                        if resp.status_code != 200:
                            body_chunks = []
                            async for chunk in resp.aiter_bytes():
                                body_chunks.append(chunk)
                            body_text = b"".join(body_chunks).decode(errors="replace")[:2000]
                            yield {"status": resp.status_code, "body": body_text}
                            return

                        # 流式读取：先收集第一个 chunk 检测是否 punish 页
                        first_chunk = ""
                        async for chunk in resp.aiter_bytes():
                            decoded = chunk.decode("utf-8", errors="replace")
                            if decoded:
                                first_chunk += decoded
                                # 收到足够数据后检测 punish（SSE 首行通常 < 500 bytes）
                                if len(first_chunk) >= 200 or "\n\n" in first_chunk:
                                    break

                        if self._is_punish_response(200, first_chunk):
                            # 读完剩余 body 用于提取 punish URL
                            remaining = []
                            async for chunk in resp.aiter_bytes():
                                remaining.append(chunk)
                            full_body = first_chunk + b"".join(remaining).decode("utf-8", errors="replace")
                            async for item in self._handle_punish_and_retry(
                                token, chat_id, payload, account, full_body, timeout,
                            ):
                                yield item
                            return

                        # 非 punish：正常 yield 已读数据 + 继续流式
                        if first_chunk:
                            yield {"chunk": first_chunk}
                        async for chunk in resp.aiter_bytes():
                            decoded = chunk.decode("utf-8", errors="replace")
                            if decoded:
                                yield {"chunk": decoded}
                        yield {"status": "streamed"}
            except Exception as e:
                log.error(f"[QwenClient] stream_chat_once error: {e}")
                yield {"status": 0, "body": str(e)}
            return

        fingerprint = fingerprint_for_account(account)
        dedicated_session = bool(getattr(settings, "QWEN_UPSTREAM_STREAM_DEDICATED_SESSION", True))
        session = new_session(fingerprint, timeout=timeout) if dedicated_session else await get_session(fingerprint)
        log.info(
            "[QwenClient] stream_headers chat_id=%s dedicated_session=%s diag=%s",
            chat_id,
            dedicated_session,
            self._header_diagnostics(account=account, headers=headers),
        )
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

                # 流式读取：先收集第一个 chunk 检测是否 punish 页
                first_chunk = ""
                async for chunk in resp.aiter_content():
                    decoded = chunk.decode("utf-8", errors="replace")
                    if decoded:
                        first_chunk += decoded
                        if len(first_chunk) >= 200 or "\n\n" in first_chunk:
                            break

                if self._is_punish_response(200, first_chunk):
                    remaining = []
                    async for chunk in resp.aiter_content():
                        remaining.append(chunk)
                    full_body = first_chunk + b"".join(remaining).decode("utf-8", errors="replace")
                    async for item in self._handle_punish_and_retry(
                        token, chat_id, payload, account, full_body, timeout,
                    ):
                        yield item
                    return

                # 非 punish：正常 yield
                if first_chunk:
                    yield {"chunk": first_chunk}
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
        preferred_account=None,
        chat_type: str = "t2t",
        media_options: dict | None = None,
    ):
        async for item in self.executor.chat_stream_events_with_retry(
            model,
            content,
            has_custom_tools,
            files=files,
            preferred_account=preferred_account,
            chat_type=chat_type,
            media_options=media_options,
        ):
            yield item

    async def complete_once_with_retry(
        self,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        chat_type: str = "t2v",
        media_options: dict | None = None,
    ) -> dict:
        """非流式一次性 completions + 重试（用于视频 t2v）。返回 {chat_id, acc, body}。"""
        return await self.executor.complete_once_with_retry(
            model,
            content,
            has_custom_tools=has_custom_tools,
            files=files,
            chat_type=chat_type,
            media_options=media_options,
        )
