# -*- coding: utf-8 -*-
"""Offline regressions for skill discovery and domain-suite packaging."""

from pathlib import Path

import pytest

from cookbooks.skills_evaluation.skill_models import SkillLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_SUITES = ("academic-eval", "arena-eval", "openjudge-core")
DOMAIN_SKILLS = sorted(skill for suite in DOMAIN_SUITES for skill in (REPO_ROOT / "skills" / suite).glob("*/SKILL.md"))
pytestmark = pytest.mark.unit


def _write_skill(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {directory.name}\ndescription: A test skill.\n---\n\nTest instructions.\n",
        encoding="utf-8",
    )
    return directory


def test_loads_flat_and_nested_skills(tmp_path: Path) -> None:
    nested = _write_skill(tmp_path / "a-suite" / "subgroup" / "nested")
    standalone = _write_skill(tmp_path / "standalone")
    (tmp_path / "empty-suite").mkdir()

    skills = SkillLoader.load_from_directory(tmp_path)

    assert [skill.directory for skill in skills] == [nested, standalone]


@pytest.mark.parametrize("single_skill", [True, False])
def test_stops_discovery_at_package_boundary(tmp_path: Path, single_skill: bool) -> None:
    package = _write_skill(tmp_path / "package")
    _write_skill(package / "references" / "example")

    skills = SkillLoader.load_from_directory(package if single_skill else tmp_path)

    assert [skill.directory for skill in skills] == [package]
    assert "references/example/SKILL.md" in {file.relative_path for file in skills[0].files}


@pytest.mark.parametrize("ignored", [".git", ".venv", "node_modules", "__pycache__"])
def test_ignores_tooling_directories_inside_suites(tmp_path: Path, ignored: str) -> None:
    real = _write_skill(tmp_path / "suite" / "real")
    _write_skill(tmp_path / "suite" / ignored / "example")

    assert [skill.directory for skill in SkillLoader.load_from_directory(tmp_path)] == [real]


def test_does_not_discover_examples_in_an_invalid_package(tmp_path: Path) -> None:
    invalid = _write_skill(tmp_path / "suite" / "invalid")
    (invalid / "SKILL.md").write_text("Not a valid skill manifest.\n", encoding="utf-8")
    _write_skill(invalid / "references" / "example")

    assert SkillLoader.load_from_directory(tmp_path) == []


def test_symlink_cycles_and_aliases_do_not_duplicate_skills(tmp_path: Path) -> None:
    real = _write_skill(tmp_path / "suite" / "real")
    (tmp_path / "suite" / "cycle").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)

    skills = SkillLoader.load_from_directory(tmp_path)

    assert len(skills) == 1
    assert skills[0].directory.resolve() == real.resolve()


def test_rejects_non_directory_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a directory"):
        SkillLoader.load_from_directory(tmp_path / "missing")


def test_repository_root_includes_every_domain_suite() -> None:
    skills = SkillLoader.load_from_directory(REPO_ROOT / "skills")
    discovered = {skill.skill_md_path for skill in skills}

    assert DOMAIN_SKILLS
    assert set(DOMAIN_SKILLS) <= discovered
    assert REPO_ROOT / "skills" / "mmx-cli" / "SKILL.md" in discovered
    assert REPO_ROOT / "skills" / "eval_pipeline" / "00-meta-eval" / "SKILL.md" in discovered


@pytest.mark.parametrize("skill_md", DOMAIN_SKILLS, ids=lambda path: str(path.relative_to(REPO_ROOT / "skills")))
def test_domain_skill_name_matches_install_directory(skill_md: Path) -> None:
    skill = SkillLoader.load_skill(skill_md.parent)

    assert skill is not None
    assert skill.manifest.name == skill_md.parent.name


def test_domain_skill_names_do_not_collide() -> None:
    skills = [SkillLoader.load_skill(path.parent) for path in DOMAIN_SKILLS]
    names = [skill.manifest.name for skill in skills]

    assert len(set(names)) == len(names)
