from pathlib import PurePosixPath

from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException


class SPAStaticFiles(StaticFiles):
    """Serve static files while falling back to index.html for SPA routes."""

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
        if route_path.parts and route_path.parts[0] in SPAStaticFiles.FALLBACK_EXCLUDED_PREFIXES:
            return False

        return route_path.suffix == ""
