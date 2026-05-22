from __future__ import annotations

import pytest

from backend.skills.catalog import SkillCatalog
from backend.skills.resolver import SkillResolver
from backend.skills.types import SkillDescriptor


def test_resolver_exact_name_match() -> None:
    catalog = SkillCatalog(
        [SkillDescriptor(name="smart-search", location="/skills/smart-search/SKILL.md")]
    )

    plan = SkillResolver().resolve("please use smart-search", catalog)

    assert plan.primary is not None
    assert plan.primary.name == "smart-search"


def test_resolver_matches_alias() -> None:
    catalog = SkillCatalog(
        [
            SkillDescriptor(
                name="smart-search",
                location="/skills/smart-search/SKILL.md",
                aliases=("search",),
            )
        ]
    )

    plan = SkillResolver().resolve("please use search", catalog)

    assert plan.primary is not None
    assert plan.primary.name == "smart-search"


def test_resolver_exact_name_priority_cannot_be_hijacked_by_alias() -> None:
    with pytest.raises(ValueError, match="conflicts with skill 'search'"):
        SkillCatalog(
            [
                SkillDescriptor(
                    name="search-plus",
                    location="/skills/search-plus/SKILL.md",
                    aliases=("search",),
                ),
                SkillDescriptor(name="search", location="/skills/search/SKILL.md"),
            ]
        )


def test_resolver_does_not_fuzzy_match_unrelated_substrings() -> None:
    catalog = SkillCatalog(
        [SkillDescriptor(name="smart-search", location="/skills/smart-search/SKILL.md")]
    )

    plan = SkillResolver().resolve("please use not-smart-searcher", catalog)

    assert plan.primary is None


def test_resolver_returns_empty_plan_when_no_match() -> None:
    catalog = SkillCatalog(
        [SkillDescriptor(name="smart-search", location="/skills/smart-search/SKILL.md")]
    )

    plan = SkillResolver().resolve("please use normal search", catalog)

    assert plan.primary is None
    assert plan.follow_ups == []
