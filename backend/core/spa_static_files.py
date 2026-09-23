from pathlib import PurePosixPath

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException


class SPAStaticFiles(StaticFiles):
    """Serve static files while falling back to index.html for SPA routes."""

    # 这些前缀归后端 API 所有：未匹配的子路径必须 404，不能回落到前端页面。
    FALLBACK_EXCLUDED_PREFIXES = {
        "admin",
        "anthropic",
        "api",
        "assets",
        "chat",
        "embeddings",
        "healthz",
        "images",
        "messages",
        "models",
        "readyz",
        "responses",
        "v1",
        "v1beta",
        "videos",
    }

    # 例外：这些前缀本身（不含任何子路径）是前端页面路由，需要正常回落，
    # 否则直接访问 / 刷新页面会 404。子路径仍按 API 处理。
    SPA_PAGE_PATHS = {
        "images",
        "videos",
    }

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or not self._should_fallback_to_index(path, scope):
                raise

            return await super().get_response("index.html", scope)

    @staticmethod
    def _should_fallback_to_index(path: str, scope) -> bool:
        normalized_path = str(scope.get("path") or path).strip("/")
        if not normalized_path:
            return False

        route_path = PurePosixPath(normalized_path)
        # 只对无扩展名的路径做回落，避免把缺失的静态资源伪装成页面。
        if route_path.suffix != "":
            return False

        if normalized_path in SPAStaticFiles.SPA_PAGE_PATHS:
            return True

        if route_path.parts and route_path.parts[0] in SPAStaticFiles.FALLBACK_EXCLUDED_PREFIXES:
            return False

        return True
