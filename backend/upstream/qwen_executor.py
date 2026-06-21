from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time

from backend.core.config import settings
from backend.core.request_logging import update_request_context
from backend.services.auth_resolver import BASE_URL, AuthResolver
from backend.upstream.payload_builder import build_chat_payload
from backend.upstream.sse_consumer import parse_sse_chunk

log = logging.getLogger("qwen2api.executor")


def _has_textual_tool_contract_marker(prompt: str) -> bool:
    return "##TOOL_CALL##" in prompt or "<|DSML|tool_calls>" in prompt


def _is_waf_blocked_body(body: str) -> bool:
    body_lower = str(body or "").lower()
    return (
        "aliyun_waf" in body_lower
        or "aliyun_waf_aa" in body_lower
        or "aliyun_waf_bb" in body_lower
        or "<!doctypehtml" in body_lower
        or "fail_sys_user_validate" in body_lower
        or "rgv587_error" in body_lower
        or "_____tmd_____" in body_lower
        or ("/punish" in body_lower and "x5secdata" in body_lower)
        or ("<script>" in body_lower and "x5sec" in body_lower)
    )


def _preview_text(value: object, limit: int = 500) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    text = re.sub(r"(x5secdata=)[^&\s\"']+", r"\1<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"(pureCaptcha=)[^&\s\"']*", r"\1<redacted>", text, flags=re.IGNORECASE)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


class QwenExecutor:
    def __init__(self, engine, account_pool):
        self.engine = engine
        self.account_pool = account_pool
        self.auth_resolver = AuthResolver(account_pool) if account_pool is not None else None

    @staticmethod
    def _resolve_account_context(account_or_token, account=None):
        if account is not None:
            return account, str(getattr(account, "token", "") or account_or_token or "")
        if hasattr(account_or_token, "token") and hasattr(account_or_token, "email"):
            return account_or_token, str(getattr(account_or_token, "token", "") or "")
        return None, str(account_or_token or "")

    @staticmethod
    def _absorb_waf_cookie(account, acw_tc: str) -> None:
        """把上游响应收割到的 acw_tc 写回账号 WAF cookie（与登录刷新同口径，TTL 1500s）。
        仅在响应真的下发了新 acw_tc 时写入；为空则跳过、保留账号原有 cookie 不动。"""
        if account is None or not acw_tc:
            return
        account.waf_cookies = f"acw_tc={acw_tc}"
        account.waf_cookies_expires_at = time.time() + 1500
        log.info(f"[Executor] {getattr(account, 'email', '')} acw_tc 预热刷新")

    async def create_chat(self, account_or_token, model: str, chat_type: str = "t2t") -> str:
        account, token = self._resolve_account_context(account_or_token)
        request_fn = getattr(self.engine, "_request_json", None) or getattr(self.engine, "api_call", None)
        if request_fn is None:
            raise Exception("request transport unavailable")

        ts = int(time.time() * 1000)
        body = {
            "title": f"api_{ts}",
            "models": [model],
            "chat_mode": "normal",
            "chat_type": chat_type,
            "timestamp": ts,
            "project_id": "",
        }

        if getattr(self.engine, "_request_json", None) is not None:
            r = await request_fn(
                "POST",
                "/api/v2/chats/new",
                token,
                body,
                timeout=settings.QWEN_UPSTREAM_REQUEST_TIMEOUT_SECONDS,
                account=account,
                chat_transport=True,
                referer=f"{BASE_URL}/c/new-chat",
            )
        else:
            r = await request_fn("POST", "/api/v2/chats/new", token, body)
        body_text = r.get("body", "")
        if r["status"] != 200:
            body_lower = body_text.lower()
            if _is_waf_blocked_body(body_text):
                raise Exception(f"waf_blocked: create_chat HTTP {r['status']}: {body_text[:200]}")
            if (
                r["status"] in (401, 403)
                or "unauthorized" in body_lower
                or "forbidden" in body_lower
                or "token" in body_lower
                or "login" in body_lower
                or "401" in body_text
                or "403" in body_text
            ):
                raise Exception(f"unauthorized: create_chat HTTP {r['status']}: {body_text[:100]}")
            if r["status"] == 429:
                raise Exception("429 Too Many Requests")
            raise Exception(f"create_chat HTTP {r['status']}: {body_text[:100]}")

        try:
            data = json.loads(body_text)
            if not data.get("success") or "id" not in data.get("data", {}):
                raise Exception("Qwen API returned error or missing id")
            # 预热/建 chat 时顺手收割 acw_tc 刷新 WAF cookie：chats/new 走 bearer 鉴权、
            # 不被 WAF 拦，故能为无密码账号（登录不了、拿不到 acw_tc）也补上风控 cookie。
            self._absorb_waf_cookie(account, r.get("acw_tc", ""))
            return data["data"]["id"]
        except Exception as e:
            body_lower = body_text.lower()
            if _is_waf_blocked_body(body_text):
                raise Exception(f"waf_blocked: create_chat returned WAF page: {body_text[:200]}")
            if any(
                kw in body_lower
                for kw in (
                    "html",
                    "login",
                    "unauthorized",
                    "activation",
                    "pending",
                    "forbidden",
                    "token",
                    "expired",
                    "invalid",
                )
            ):
                raise Exception(f"unauthorized: account issue: {body_text[:200]}")
            raise Exception(f"create_chat parse error: {e}, body={body_text[:200]}")

    async def get_chat_id(self, acc, model: str, chat_type: str = "t2t") -> tuple[str, bool]:
        chat_pool = getattr(self.engine, "chat_id_pool", None)
        if chat_pool is not None:
            await chat_pool.remember_model(model, chat_type)
            chat_id, reused = await chat_pool.take(acc.email, model, chat_type)
            if reused and chat_id:
                return chat_id, True
        if chat_type == "t2t":
            return await self.create_chat(acc, model), False
        return await self.create_chat(acc, model, chat_type=chat_type), False

    async def stream(
        self,
        account_or_token,
        chat_id: str,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        account=None,
        chat_type: str = "t2t",
        media_options: dict | None = None,
    ):
        account_obj, token = self._resolve_account_context(account_or_token, account=account)
        stream_fn = getattr(self.engine, "stream_chat_once", None) or getattr(self.engine, "fetch_chat", None)
        if stream_fn is None:
            raise Exception("stream transport unavailable")

        payload = build_chat_payload(
            chat_id, model, content, has_custom_tools, files=files,
            chat_type=chat_type, media_options=media_options,
        )
        buffer = ""
        started_at = time.perf_counter()
        first_event_logged = False
        last_chunk_time = time.perf_counter()
        chunk_count = 0
        stream_chars = 0
        parsed_event_count = 0
        last_heartbeat_at = started_at
        first_chunk_preview = ""

        feature_config = payload.get("messages", [{}])[0].get("feature_config", {})
        log.info(f"[Executor] stream start chat_id={chat_id} model={model} has_custom_tools={has_custom_tools}")
        log.info(f"[Executor] feature_config: function_calling={feature_config.get('function_calling')} auto_search={feature_config.get('auto_search')} code_interpreter={feature_config.get('code_interpreter')} plugins_enabled={feature_config.get('plugins_enabled')}")

        prompt_content = payload.get("messages", [{}])[0].get("content", "")
        if _has_textual_tool_contract_marker(prompt_content):
            log.info("[Executor] prompt contains textual tool contract markers (expected)")
        else:
            log.warning("[Executor] prompt does NOT contain textual tool contract markers - this may cause interception")
        log.info(f"[Executor] prompt preview (first 500 chars): {prompt_content[:500]}")

        try:
            if account_obj is not None and getattr(self.engine, "stream_chat_once", None) is not None:
                stream_iter = stream_fn(token, chat_id, payload, account=account_obj)
            else:
                stream_iter = stream_fn(token, chat_id, payload)
            stream_iterator = stream_iter.__aiter__()
            total_timeout = max(0.0, float(getattr(settings, "QWEN_UPSTREAM_STREAM_TOTAL_TIMEOUT_SECONDS", 0) or 0))
            idle_timeout = max(0.0, float(getattr(settings, "QWEN_UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS", 0) or 0))

            while True:
                now = time.perf_counter()
                elapsed = now - started_at
                if total_timeout > 0 and elapsed >= total_timeout:
                    raise TimeoutError(f"upstream stream total timeout after {total_timeout:.0f}s")
                wait_timeout = idle_timeout if idle_timeout > 0 else None
                if total_timeout > 0:
                    remaining_total = max(0.0, total_timeout - elapsed)
                    wait_timeout = remaining_total if wait_timeout is None else min(wait_timeout, remaining_total)
                try:
                    if wait_timeout is None:
                        chunk_result = await anext(stream_iterator)
                    else:
                        chunk_result = await asyncio.wait_for(anext(stream_iterator), timeout=wait_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    elapsed = time.perf_counter() - started_at
                    idle_time = time.perf_counter() - last_chunk_time
                    if total_timeout > 0 and elapsed >= total_timeout:
                        raise TimeoutError(f"upstream stream total timeout after {total_timeout:.0f}s") from exc
                    raise TimeoutError(f"upstream stream idle timeout after {idle_timeout:.0f}s") from exc

                last_chunk_time = time.perf_counter()

                if chunk_result.get("status") not in (None, 200, "streamed"):
                    body = chunk_result.get("body", b"")
                    if isinstance(body, bytes):
                        body = body.decode("utf-8", errors="ignore")
                    raise Exception(f"HTTP {chunk_result['status']}: {str(body)[:100]}")

                if "chunk" in chunk_result:
                    chunk = chunk_result["chunk"]
                    if _is_waf_blocked_body(chunk):
                        raise Exception(f"waf_blocked: stream validation challenge: {_preview_text(chunk)}")
                    chunk_count += 1
                    stream_chars += len(chunk)
                    if not first_chunk_preview:
                        first_chunk_preview = _preview_text(chunk)
                    now = time.perf_counter()
                    if chunk_count % 100 == 0 or now - last_heartbeat_at >= 60:
                        last_heartbeat_at = now
                        log.info(
                            "[Executor] stream heartbeat chat_id=%s chunks=%s chars=%s parsed_events=%s elapsed=%.3fs idle=%.3fs",
                            chat_id,
                            chunk_count,
                            stream_chars,
                            parsed_event_count,
                            now - started_at,
                            now - last_chunk_time,
                        )
                    buffer += chunk
                    while "\n\n" in buffer:
                        msg, buffer = buffer.split("\n\n", 1)
                        for evt in parse_sse_chunk(msg):
                            parsed_event_count += 1
                            if not first_event_logged:
                                first_event_logged = True
                                log.info(
                                    f"[Executor] first parsed event after {(time.perf_counter() - started_at):.3f}s chat_id={chat_id}"
                                )
                            yield evt
        except Exception as e:
            elapsed = time.perf_counter() - started_at
            idle_time = time.perf_counter() - last_chunk_time
            error_type = type(e).__name__
            log.error(
                f"[Executor] stream error chat_id={chat_id} error_type={error_type} "
                f"elapsed={elapsed:.3f}s idle_time={idle_time:.3f}s error={str(e)[:200]}"
            )
            raise

        if buffer:
            for evt in parse_sse_chunk(buffer):
                parsed_event_count += 1
                if not first_event_logged:
                    first_event_logged = True
                    log.info(
                        f"[Executor] first parsed event after {(time.perf_counter() - started_at):.3f}s chat_id={chat_id}"
                    )
                yield evt

        if chunk_count > 0 and parsed_event_count == 0:
            log.warning(
                "[Executor] stream_unparsed_preview chat_id=%s chunks=%s chars=%s preview=%r",
                chat_id,
                chunk_count,
                stream_chars,
                first_chunk_preview,
            )

        log.info(
            "[Executor] stream finish chat_id=%s total=%.3fs chunks=%s chars=%s parsed_events=%s",
            chat_id,
            time.perf_counter() - started_at,
            chunk_count,
            stream_chars,
            parsed_event_count,
        )

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
        exclude = set()
        # 统一走预热池路径：所有请求（含文件绑定）都从 ChatIDPool 取已预热 chat_id，
        # 不再区分 fixed_account / 普通池。preferred_account 仅作为首次尝试的首选账号，
        # WAF/限流时仍走标准失败转移（换号重试），避免原地新建裸奔 chat_id 被 WAF 拦截。
        for attempt in range(settings.MAX_RETRIES):
            update_request_context(upstream_attempt=attempt + 1)
            # 首次尝试优先使用 affinity 绑定的首选账号（保证文件可达），后续重试走池分配
            if attempt == 0 and preferred_account is not None:
                acc = preferred_account
            else:
                acc = await self.account_pool.acquire_wait(timeout=60, exclude=exclude)
                if not acc:
                    raise Exception("No available accounts in pool (all busy or rate limited)")

            try:
                log.info(f"[Executor] acquired account={acc.email} model={model} attempt={attempt + 1}")
                chat_id, reused = await self.get_chat_id(acc, model, chat_type=chat_type)
                update_request_context(chat_id=chat_id)
                log.info(f"[Executor] created chat_id={chat_id} account={acc.email} prewarmed={reused}")
                yield {"type": "meta", "chat_id": chat_id, "acc": acc}

                async for evt in self.stream(acc, chat_id, model, content, has_custom_tools, files=files, chat_type=chat_type, media_options=media_options):
                    yield {"type": "event", "event": evt}
                return

            except Exception as e:
                needs_heal = await self._classify_and_release(acc, e, exclude)
                log.warning(
                    f"[Executor] retry attempt={attempt + 1}/{settings.MAX_RETRIES} account={acc.email} error={e}"
                )
                # WAF/401 自愈完成后，如果账号仍无有效 acw_tc，强制新建 chat 收割 cookie
                # （预热池复用的 chat_id 不会触发 create_chat，导致 cookie 无法在重试链路中刷新）
                if needs_heal and acc.waf_cookies_expires_at == 0:
                    try:
                        await self.create_chat(acc, model, chat_type=chat_type)
                        log.info(f"[Executor] WAF 兜底 create_chat 完成 account={acc.email}")
                    except Exception as seed_err:
                        log.warning(f"[Executor] WAF 兜底 create_chat 失败 account={acc.email} error={seed_err}")
                # 重试前随机退避 2~5s：降低 IP 频率分，给新 cookie 生效时间
                backoff = random.uniform(2.0, 5.0)
                log.info(f"[Executor] retry backoff {backoff:.1f}s before attempt={attempt + 2}")
                await asyncio.sleep(backoff)
            except (asyncio.CancelledError, GeneratorExit):
                self.account_pool.release(acc)
                raise

        raise Exception(f"All {settings.MAX_RETRIES} attempts failed. Please check upstream accounts.")

    async def _classify_and_release(self, acc, e: Exception, exclude: set) -> bool:
        """按错误类型标记账号状态并加入排除集，最后释放账号（流式/非流式共用）。

        Returns:
            True 表示触发了 WAF/401 自愈（调用方应在重试前等待自愈完成并兜底收割 cookie）。
        """
        err_msg = str(e).lower()
        is_timeout = (
            "timeout" in err_msg
            or "timed out" in err_msg
            or "readtimeout" in err_msg
            or type(e).__name__ in ("ReadTimeout", "TimeoutError", "TimeoutException")
        )
        needs_heal = False

        if is_timeout:
            log.warning(f"[Executor] timeout detected account={acc.email} error={e}")
            exclude.add(acc.email)
        elif "429" in err_msg or "rate limit" in err_msg or "too many" in err_msg:
            self.account_pool.mark_rate_limited(acc)
            exclude.add(acc.email)
        elif "waf_blocked" in err_msg or "aliyun_waf" in err_msg:
            exclude.add(acc.email)
            # WAF 命中：标记该账号 acw_tc 失效并立即刷新，为后续请求恢复风控 cookie
            acc.waf_cookies_expires_at = 0
            if self.auth_resolver is not None:
                await self.auth_resolver.auto_heal_account(acc)
            needs_heal = True
        elif "unauthorized" in err_msg or "401" in err_msg or "403" in err_msg:
            self.account_pool.mark_invalid(acc)
            exclude.add(acc.email)
            if "activation" in err_msg or "pending" in err_msg:
                acc.activation_pending = True
            if self.auth_resolver is not None:
                await self.auth_resolver.auto_heal_account(acc)
            needs_heal = True
        else:
            exclude.add(acc.email)

        self.account_pool.release(acc)
        return needs_heal

    async def complete_once_with_retry(
        self,
        model: str,
        content: str,
        has_custom_tools: bool = False,
        files: list[dict] | None = None,
        chat_type: str = "t2v",
        media_options: dict | None = None,
    ) -> dict:
        """非流式一次性 completions + 账号重试（用于视频 t2v）。

        成功返回 {"chat_id", "acc", "body"}；body 为完整 JSON 文本。
        account 的 release 由调用方负责（与流式成功路径一致）。
        """
        complete_fn = getattr(self.engine, "complete_chat_once", None)
        if complete_fn is None:
            raise Exception("non-stream transport unavailable")

        exclude: set = set()
        for attempt in range(settings.MAX_RETRIES):
            update_request_context(upstream_attempt=attempt + 1)
            acc = await self.account_pool.acquire_wait(timeout=60, exclude=exclude)
            if not acc:
                raise Exception("No available accounts in pool (all busy or rate limited)")

            try:
                chat_id, reused = await self.get_chat_id(acc, model, chat_type=chat_type)
                update_request_context(chat_id=chat_id)
                log.info(f"[Executor] non-stream completions chat_id={chat_id} account={acc.email} attempt={attempt + 1}")
                payload = build_chat_payload(
                    chat_id, model, content, has_custom_tools, files=files,
                    chat_type=chat_type, media_options=media_options,
                )
                res = await complete_fn(acc.token, chat_id, payload, account=acc)
                status = res.get("status")
                body = res.get("body", "")
                if status != 200:
                    raise Exception(f"HTTP {status}: {str(body)[:200]}")
                if _is_waf_blocked_body(body):
                    raise Exception(f"waf_blocked: completions challenge: {_preview_text(body)}")
                return {"chat_id": chat_id, "acc": acc, "body": body}
            except (asyncio.CancelledError, GeneratorExit):
                self.account_pool.release(acc)
                raise
            except Exception as e:
                needs_heal = await self._classify_and_release(acc, e, exclude)
                log.warning(
                    f"[Executor] non-stream retry attempt={attempt + 1}/{settings.MAX_RETRIES} account={acc.email} error={e}"
                )
                # WAF/401 自愈完成后，如果账号仍无有效 acw_tc，强制新建 chat 收割 cookie
                if needs_heal and acc.waf_cookies_expires_at == 0:
                    try:
                        await self.create_chat(acc, model, chat_type=chat_type)
                        log.info(f"[Executor] non-stream WAF 兜底 create_chat 完成 account={acc.email}")
                    except Exception as seed_err:
                        log.warning(f"[Executor] non-stream WAF 兜底 create_chat 失败 account={acc.email} error={seed_err}")

        raise Exception(f"All {settings.MAX_RETRIES} attempts failed. Please check upstream accounts.")
