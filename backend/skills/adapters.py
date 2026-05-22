from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from backend.services.client_profiles import OPENCLAW_OPENAI_PROFILE
from backend.skills.catalog import SkillCatalog, normalize_skill_catalog
from backend.skills.types import SkillDescriptor

_AVAILABLE_SKILLS_BLOCK_RE = re.compile(r"(?is)<available_skills\b[^>]*>(.*?)</available_skills>")
_SKILL_BLOCK_RE = re.compile(r"(?is)<skill\b[^>]*>(.*?)</skill>")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = _content_to_text(part)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        parts: list[str] = []
        text = content.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
        for key in ("content", "parts"):
            nested = content.get(key)
            if nested is None:
                continue
            nested_text = _content_to_text(nested)
            if nested_text:
                parts.append(nested_text)
        return "\n".join(parts)
    return ""


def _extract_tag_text(block: str, tag_name: str) -> str:
    match = re.search(rf"(?is)<{re.escape(tag_name)}\b[^>]*>(.*?)</{re.escape(tag_name)}>", block)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_aliases(block: str) -> tuple[str, ...]:
    aliases: list[str] = []

    for alias_match in re.finditer(r"(?is)<alias\b[^>]*>(.*?)</alias>", block):
        aliases.extend(_split_alias_text(alias_match.group(1)))

    aliases_block = _extract_tag_text(block, "aliases")
    if aliases_block:
        aliases.extend(_split_alias_text(aliases_block))

    seen: set[str] = set()
    ordered_aliases: list[str] = []
    for alias in aliases:
        alias = alias.strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        ordered_aliases.append(alias)
    return tuple(ordered_aliases)


def _split_alias_text(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n,;|]+", text) if part.strip()]


def _parse_skill_descriptor_block(block: str) -> SkillDescriptor | None:
    name = _extract_tag_text(block, "name")
    location = _extract_tag_text(block, "location")
    if not name or not location:
        return None

    description = _extract_tag_text(block, "description")
    source = _extract_tag_text(block, "source")
    aliases = _extract_aliases(block)
    return SkillDescriptor(
        name=name,
        location=location,
        description=description,
        aliases=aliases,
        source=source,
    )


def parse_skill_descriptors(content: Any) -> list[SkillDescriptor]:
    text = _content_to_text(content)
    if not text or "<available_skills" not in text.lower():
        return []

    descriptors: list[SkillDescriptor] = []
    seen_names: set[str] = set()
    for available_skills_match in _AVAILABLE_SKILLS_BLOCK_RE.finditer(text):
        available_skills_block = available_skills_match.group(1)
        for skill_match in _SKILL_BLOCK_RE.finditer(available_skills_block):
            descriptor = _parse_skill_descriptor_block(skill_match.group(1))
            if descriptor is None or descriptor.name in seen_names:
                continue
            seen_names.add(descriptor.name)
            descriptors.append(descriptor)
    return descriptors


def extract_skill_catalog_from_messages(messages: Iterable[dict[str, Any]], client_profile: str) -> SkillCatalog:
    if client_profile != OPENCLAW_OPENAI_PROFILE:
        return normalize_skill_catalog(())

    descriptors: list[SkillDescriptor] = []
    seen_names: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            continue
        for descriptor in parse_skill_descriptors(message.get("content", "")):
            if descriptor.name in seen_names:
                continue
            seen_names.add(descriptor.name)
            descriptors.append(descriptor)

    return normalize_skill_catalog(descriptors)
