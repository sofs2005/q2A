from __future__ import annotations

import re

from backend.skills.catalog import SkillCatalog
from backend.skills.types import ResolvedSkillPlan


class SkillResolver:
    """Resolve prompt text to a skill plan using exact token-boundary matches only."""

    def resolve(self, text: str, catalog: SkillCatalog) -> ResolvedSkillPlan:
        earliest_match: tuple[int, str] | None = None

        for lookup_name in catalog.lookup_names():
            if not lookup_name:
                continue
            match = re.search(_boundary_pattern(lookup_name), text)
            if match is None:
                continue
            if earliest_match is None or match.start() < earliest_match[0]:
                earliest_match = (match.start(), lookup_name)

        if earliest_match is None:
            return ResolvedSkillPlan(reason="no matching skill")

        return ResolvedSkillPlan(
            primary=catalog.resolve(earliest_match[1]),
            reason=f"matched skill '{earliest_match[1]}'",
        )


def _boundary_pattern(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(rf"(?<![A-Za-z0-9_-]){escaped}(?![A-Za-z0-9_-])")
