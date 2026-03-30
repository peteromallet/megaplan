"""Small git helpers for MegaLoop iterations."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _run_git(project_dir: str | Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_dir),
        text=True,
        capture_output=True,
        check=check,
    )


def _normalize_pathspec(pattern: str) -> str:
    if pattern.startswith(":("):
        return pattern
    return f":(glob){pattern}"


def _changed_allowed_paths(project_dir: str | Path, allowed_changes: list[str]) -> list[str]:
    if not allowed_changes:
        return []
    status = _run_git(
        project_dir,
        ["status", "--porcelain", "--untracked-files=all", "--", *(_normalize_pathspec(item) for item in allowed_changes)],
    )
    paths: list[str] = []
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in paths:
            paths.append(path)
    return paths


def git_commit(project_dir: str | Path, message: str, allowed_changes: list[str]) -> str | None:
    changed_paths = _changed_allowed_paths(project_dir, allowed_changes)
    if not changed_paths:
        return None
    _run_git(project_dir, ["add", "--", *changed_paths])
    staged = _run_git(project_dir, ["diff", "--cached", "--name-only", "--", *changed_paths], check=False)
    if not staged.stdout.strip():
        return None
    _run_git(project_dir, ["commit", "-m", message, "--only", "--", *changed_paths])
    return git_current_sha(project_dir)


def git_revert(project_dir: str | Path, commit_sha: str) -> None:
    try:
        _run_git(project_dir, ["revert", "--no-edit", commit_sha])
    except subprocess.CalledProcessError:
        try:
            _run_git(project_dir, ["revert", "--abort"], check=False)
        except subprocess.CalledProcessError:
            pass


def git_current_sha(project_dir: str | Path) -> str:
    result = _run_git(project_dir, ["rev-parse", "HEAD"])
    return result.stdout.strip()


def parse_metric(output: str, pattern: str) -> float | None:
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        return None
    captured = next((group for group in match.groups() if group is not None), match.group(0))
    try:
        return float(captured)
    except (TypeError, ValueError):
        numeric = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(captured))
        if numeric is None:
            return None
        try:
            return float(numeric.group(0))
        except ValueError:
            return None
