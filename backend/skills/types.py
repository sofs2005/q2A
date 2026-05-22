from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    name: str
    location: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    source: str = ""


@dataclass(slots=True)
class ResolvedSkillPlan:
    primary: SkillDescriptor | None = None
    follow_ups: list[SkillDescriptor] = field(default_factory=list)
    reason: str = ""
