from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.services.client_profiles import extract_latest_user_text
from backend.skills.adapters import extract_skill_catalog_from_messages
from backend.skills.catalog import SkillCatalog, normalize_skill_catalog
from backend.skills.loader import SkillLoader
from backend.skills.resolver import SkillResolver
from backend.skills.types import SkillDescriptor

__all__ = ["build_skill_context"]

_MAX_SKILL_CONTEXT_BYTES = 256 * 1024
_ENV_SKILL_ROOTS = "QWEN2API_SKILL_ROOTS"
_LEGACY_ENV_SKILL_ROOTS = "SKILL_ALLOWED_ROOTS"
_DEFAULT_SKILL_ROOTS = (
    Path.home() / ".openclaw" / "workspace" / "skills",
    Path.home() / ".claude" / "skills",
)


def _normalized_skill_catalog(req_data: dict[str, Any], *, client_profile: str) -> SkillCatalog:
    raw_skill_catalog = req_data.get("_skill_catalog")
    if raw_skill_catalog is not None:
        return normalize_skill_catalog(raw_skill_catalog)
    return extract_skill_catalog_from_messages(req_data.get("messages", []), client_profile=client_profile)


def _trusted_skill_roots() -> tuple[Path, ...]:
    raw_roots = os.getenv(_ENV_SKILL_ROOTS) or os.getenv(_LEGACY_ENV_SKILL_ROOTS, "")
    roots: list[Path] = []
    for raw_root in raw_roots.split(os.pathsep):
        cleaned = raw_root.strip()
        if cleaned:
            roots.append(Path(cleaned).expanduser().resolve())
    roots.extend(root.expanduser().resolve() for root in _DEFAULT_SKILL_ROOTS)

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root)
    return tuple(unique_roots)


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _candidate_skill_locations(location: str, trusted_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    raw_location = Path(location).expanduser()
    if raw_location.is_absolute():
        return (raw_location.resolve(),)
    return tuple((root / raw_location).resolve() for root in trusted_roots)


def _safe_skill_location(descriptor: SkillDescriptor, trusted_roots: tuple[Path, ...]) -> tuple[Path, Path]:
    if not trusted_roots:
        raise ValueError("No trusted skill roots configured")

    for resolved_location in _candidate_skill_locations(descriptor.location, trusted_roots):
        if resolved_location.name != "SKILL.md":
            continue
        if resolved_location.parent.name != descriptor.name:
            continue

        trusted_root = next((root for root in trusted_roots if _is_under_root(resolved_location, root)), None)
        if trusted_root is None:
            continue
        if resolved_location.relative_to(trusted_root).parts != (descriptor.name, "SKILL.md"):
            continue
        try:
            size = resolved_location.stat().st_size
        except OSError:
            continue
        if size > _MAX_SKILL_CONTEXT_BYTES:
            raise ValueError(f"Skill file is too large '{descriptor.location}'")
        return resolved_location, trusted_root

    raise ValueError(f"Unsafe skill location '{descriptor.location}'")


def _skill_loader_for_descriptor(descriptor: SkillDescriptor) -> tuple[SkillLoader, SkillDescriptor]:
    resolved_location, trusted_root = _safe_skill_location(descriptor, _trusted_skill_roots())
    loader = SkillLoader([trusted_root])
    resolved_descriptor = SkillDescriptor(
        name=descriptor.name,
        location=str(resolved_location),
        description=descriptor.description,
        aliases=descriptor.aliases,
        source=descriptor.source,
    )
    return loader, resolved_descriptor


def build_skill_context(req_data: dict[str, Any], *, client_profile: str) -> str:
    try:
        catalog = _normalized_skill_catalog(req_data, client_profile=client_profile)
    except (TypeError, ValueError):
        return ""
    if len(catalog) == 0:
        return ""

    latest_user_text = extract_latest_user_text(req_data.get("messages", []), client_profile=client_profile).strip()
    if not latest_user_text:
        return ""

    plan = SkillResolver().resolve(latest_user_text, catalog)
    if plan.primary is None:
        return ""

    try:
        loader, descriptor = _skill_loader_for_descriptor(plan.primary)
        return loader.load(descriptor).strip()
    except (OSError, UnicodeError, ValueError):
        return ""
