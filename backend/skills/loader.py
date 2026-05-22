from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from backend.skills.types import SkillDescriptor


class SkillLoader:
    def __init__(self, allowed_roots: Iterable[Path | str] = ()) -> None:
        self._allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)

    def load(self, descriptor: SkillDescriptor) -> str:
        path = Path(descriptor.location)
        resolved_path = self._resolve_location(path)
        return resolved_path.read_text(encoding="utf-8")

    def _resolve_location(self, path: Path) -> Path:
        if self._has_relative_traversal(path):
            raise ValueError(f"Unsafe skill location '{path}'")

        if self._allowed_roots:
            if path.is_absolute():
                resolved = path.resolve()
                if self._is_under_allowed_root(resolved):
                    return resolved
                raise ValueError(f"Skill location '{path}' is outside allowed roots")

            for root in self._allowed_roots:
                candidate = (root / path).resolve()
                if self._is_under_root(candidate, root):
                    return candidate
            raise ValueError(f"Skill location '{path}' is outside allowed roots")

        if path.is_absolute():
            return path.resolve()

        return (Path.cwd() / path).resolve()

    @staticmethod
    def _has_relative_traversal(path: Path) -> bool:
        return any(part == ".." for part in path.parts)

    def _is_under_allowed_root(self, path: Path) -> bool:
        return any(self._is_under_root(path, root) for root in self._allowed_roots)

    @staticmethod
    def _is_under_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True
