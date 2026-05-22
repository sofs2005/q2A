from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.skills.loader import SkillLoader
from backend.skills.types import SkillDescriptor


def test_loader_rejects_path_traversal() -> None:
    loader = SkillLoader([Path("/tmp/skills")])

    with pytest.raises(ValueError):
        loader.load(SkillDescriptor(name="bad", location="../etc/passwd"))


def test_loader_reads_temp_directory_skill() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_dir = root / "smart-search"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# smart-search\nUse web fetch.", encoding="utf-8")
        loader = SkillLoader([root])

        text = loader.load(
            SkillDescriptor(name="smart-search", location=str(skill_dir / "SKILL.md"))
        )

        assert "Use web fetch." in text


def test_loader_rejects_absolute_path_outside_allowed_roots() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("outside", encoding="utf-8")
        loader = SkillLoader([root])

        with pytest.raises(ValueError):
            loader.load(SkillDescriptor(name="bad", location=str(outside)))


def test_loader_without_allowed_roots_reads_normal_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        skill_path = tmp_path / "SKILL.md"
        skill_path.write_text("relative skill", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        loader = SkillLoader([])

        text = loader.load(SkillDescriptor(name="relative", location="SKILL.md"))
        monkeypatch.undo()

        assert text == "relative skill"


def test_loader_without_allowed_roots_rejects_relative_traversal() -> None:
    loader = SkillLoader([])

    with pytest.raises(ValueError):
        loader.load(SkillDescriptor(name="bad", location="../etc/passwd"))
