from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from megaplan._core.io import collect_git_diff_patch


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)


def _git_commit_all(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True, text=True)


def test_collect_git_diff_patch_includes_tracked_and_untracked_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    tracked = repo / "app.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    _git_commit_all(repo, "initial")

    tracked.write_text("value = 2\n", encoding="utf-8")
    (repo / "new_file.py").write_text("print('new file')\n", encoding="utf-8")

    patch = collect_git_diff_patch(repo)

    assert "diff --git a/app.py b/app.py" in patch
    assert "diff --git a/new_file.py b/new_file.py" in patch
    assert "+value = 2" in patch
    assert "+print('new file')" in patch


def test_collect_git_diff_patch_handles_non_repo_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "plain"
    project_dir.mkdir()

    assert collect_git_diff_patch(project_dir) == "Project directory is not a git repository."


def test_collect_git_diff_patch_handles_missing_git_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    def _raise(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr("megaplan._core.io.subprocess.run", _raise)

    assert collect_git_diff_patch(project_dir) == "git not found on PATH."


def test_collect_git_diff_patch_handles_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    def _raise(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd=["git", "diff", "HEAD"], timeout=30)

    monkeypatch.setattr("megaplan._core.io.subprocess.run", _raise)

    assert collect_git_diff_patch(project_dir) == "git diff timed out."
