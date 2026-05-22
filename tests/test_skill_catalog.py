from __future__ import annotations

import pytest

from backend.skills.catalog import SkillCatalog, normalize_skill_catalog
from backend.skills.types import SkillDescriptor


def test_catalog_exact_and_alias_match() -> None:
    catalog = SkillCatalog(
        [
            SkillDescriptor(
                name="smart-search",
                location="/skills/smart-search/SKILL.md",
                aliases=("search",),
            )
        ]
    )

    assert catalog.resolve("smart-search").name == "smart-search"
    assert catalog.resolve("search").name == "smart-search"


def test_catalog_rejects_alias_conflicting_with_real_skill_name() -> None:
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


def test_catalog_rejects_duplicate_alias_across_skills() -> None:
    with pytest.raises(ValueError, match="duplicates alias"):
        SkillCatalog(
            [
                SkillDescriptor(
                    name="search-plus",
                    location="/skills/search-plus/SKILL.md",
                    aliases=("search",),
                ),
                SkillDescriptor(
                    name="search-pro",
                    location="/skills/search-pro/SKILL.md",
                    aliases=("search",),
                ),
            ]
        )


def test_catalog_allows_alias_matching_own_name() -> None:
    catalog = SkillCatalog(
        [
            SkillDescriptor(
                name="search",
                location="/skills/search/SKILL.md",
                aliases=("search",),
            )
        ]
    )

    assert catalog.resolve("search").name == "search"


def test_catalog_returns_none_for_unknown_name() -> None:
    catalog = SkillCatalog([SkillDescriptor(name="smart-search", location="/skills/smart-search/SKILL.md")])

    assert catalog.resolve("missing") is None
    assert catalog.get("missing") is None


def test_normalize_skill_catalog_from_dict() -> None:
    catalog = normalize_skill_catalog(
        {
            "smart-search": {
                "location": "/skills/smart-search/SKILL.md",
                "description": "Search the web",
                "aliases": ["search"],
                "source": "test",
            }
        }
    )

    descriptor = catalog.resolve("search")
    assert descriptor is not None
    assert descriptor.name == "smart-search"
    assert descriptor.description == "Search the web"
    assert descriptor.aliases == ("search",)
    assert descriptor.source == "test"


def test_normalize_skill_catalog_from_iterable_dict_descriptors() -> None:
    catalog = normalize_skill_catalog(
        [
            {
                "name": "smart-search",
                "location": "/skills/smart-search/SKILL.md",
                "aliases": ("search",),
            }
        ]
    )

    assert catalog.resolve("smart-search") is not None
    assert catalog.resolve("search") is not None


def test_normalize_skill_catalog_returns_catalog_unchanged() -> None:
    catalog = SkillCatalog([SkillDescriptor(name="smart-search", location="/skills/smart-search/SKILL.md")])

    assert normalize_skill_catalog(catalog) is catalog
