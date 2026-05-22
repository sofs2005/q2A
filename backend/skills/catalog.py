from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from backend.skills.types import SkillDescriptor


class SkillCatalog:
    """Runtime-agnostic catalog for declared skills and aliases."""

    def __init__(self, skills: Iterable[SkillDescriptor]) -> None:
        self._skills: list[SkillDescriptor] = list(skills)
        self._skills_by_name: dict[str, SkillDescriptor] = {}
        self._aliases: dict[str, str] = {}

        for skill in self._skills:
            if skill.name in self._skills_by_name:
                raise ValueError(f"Duplicate skill '{skill.name}'")
            self._skills_by_name[skill.name] = skill

        for skill in self._skills:
            for alias in skill.aliases:
                if alias == skill.name:
                    continue

                conflicting_skill = self._skills_by_name.get(alias)
                if conflicting_skill is not None:
                    raise ValueError(
                        f"Alias '{alias}' for skill '{skill.name}' conflicts with skill '{conflicting_skill.name}'"
                    )

                existing_skill_name = self._aliases.get(alias)
                if existing_skill_name is not None and existing_skill_name != skill.name:
                    raise ValueError(
                        f"Alias '{alias}' for skill '{skill.name}' duplicates alias for skill '{existing_skill_name}'"
                    )

                self._aliases[alias] = skill.name

    def resolve(self, name: str) -> SkillDescriptor | None:
        exact_match = self._skills_by_name.get(name)
        if exact_match is not None:
            return exact_match
        canonical_name = self._aliases.get(name, name)
        return self._skills_by_name.get(canonical_name)

    def get(self, name: str) -> SkillDescriptor | None:
        return self.resolve(name)

    def __iter__(self) -> Iterator[SkillDescriptor]:
        return iter(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def lookup_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for skill in self._skills:
            names.append(skill.name)
            names.extend(skill.aliases)
        return tuple(names)


def normalize_skill_catalog(raw: Any) -> SkillCatalog:
    if isinstance(raw, SkillCatalog):
        return raw

    if isinstance(raw, Mapping):
        if "name" in raw and "location" in raw:
            return SkillCatalog([_descriptor_from_mapping(raw)])

        skills = []
        for name, value in raw.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"Skill '{name}' must be described by a mapping")
            descriptor_data = dict(value)
            descriptor_data.setdefault("name", name)
            skills.append(_descriptor_from_mapping(descriptor_data))
        return SkillCatalog(skills)

    return SkillCatalog(_descriptor_from_item(item) for item in raw)


def _descriptor_from_item(item: Any) -> SkillDescriptor:
    if isinstance(item, SkillDescriptor):
        return item
    if isinstance(item, Mapping):
        return _descriptor_from_mapping(item)
    raise TypeError(f"Unsupported skill descriptor type: {type(item).__name__}")


def _descriptor_from_mapping(raw: Mapping[str, Any]) -> SkillDescriptor:
    aliases = raw.get("aliases", ())
    if aliases is None:
        aliases = ()
    elif isinstance(aliases, str):
        aliases = (aliases,)
    else:
        aliases = tuple(aliases)

    try:
        name = raw["name"]
        location = raw["location"]
    except KeyError as exc:
        raise ValueError(f"Skill descriptor missing required field '{exc.args[0]}'") from exc

    return SkillDescriptor(
        name=str(name),
        location=str(location),
        description=str(raw.get("description", "")),
        aliases=tuple(str(alias) for alias in aliases),
        source=str(raw.get("source", "")),
    )
