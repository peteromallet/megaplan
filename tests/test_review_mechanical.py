from __future__ import annotations

import subprocess
from pathlib import Path

from megaplan.review_mechanical import run_pre_checks


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)


def _git_commit_all(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True, text=True)


def _state(project_dir: Path) -> dict[str, object]:
    return {
        "idea": "Fix the parser edge case without changing unrelated behavior.",
        "meta": {"notes": []},
        "config": {"project_dir": str(project_dir)},
    }


def test_run_pre_checks_flags_test_only_diff_and_small_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _git_commit_all(repo, "initial")

    test_file = repo / "tests" / "test_only.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("assert True\n", encoding="utf-8")

    flags = run_pre_checks(tmp_path / "plan", _state(repo), repo)
    checks = {flag["check"]: flag for flag in flags}

    assert checks["source_touch"]["severity"] == "significant"
    assert "only touches test-like files" in checks["source_touch"]["detail"]
    assert checks["diff_size_sanity"]["check"] == "diff_size_sanity"
    assert "changed_lines=" in checks["diff_size_sanity"]["detail"]


def test_diff_noise_filter_excludes_megaplan_and_caches(tmp_path: Path) -> None:
    """Ensure mechanical pre-checks ignore .megaplan/ metadata, pycache, and other workspace noise.

    This locks down the fix for the `.megaplan/` diff_size_sanity bug found
    during iteration-022-robust validation — the old implementation reported
    workspace metadata as source changes (e.g., 2912 changed lines for a
    2-line source fix). The real source file must still be accounted for.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _git_commit_all(repo, "initial")

    # Real source change: 1 line.
    (repo / "src.py").write_text("def f():\n    return 2\n", encoding="utf-8")

    # A pile of workspace noise that should be ignored.
    (repo / ".megaplan").mkdir()
    (repo / ".megaplan" / "plan.md").write_text("x\n" * 500, encoding="utf-8")
    (repo / ".megaplan" / "critique.json").write_text("{}\n" * 200, encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "src.cpython-311.pyc").write_bytes(b"\x00" * 128)
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "v").write_text("noise\n" * 100, encoding="utf-8")

    flags = run_pre_checks(tmp_path / "plan", _state(repo), repo)
    sanity = next(flag for flag in flags if flag["check"] == "diff_size_sanity")

    # With the filter working, changed_lines should reflect only src.py's
    # 1-line change, not the hundreds of lines of .megaplan/ metadata.
    assert "changed_lines=1" in sanity["detail"] or "changed_lines=2" in sanity["detail"], (
        f"Expected tiny changed-line count but got: {sanity['detail']}"
    )
    assert "files=1" in sanity["detail"], (
        f"Expected exactly 1 source file in diff but got: {sanity['detail']}"
    )
    # Explicitly assert the noise wasn't counted.
    assert "2912" not in sanity["detail"]
    assert ".megaplan" not in sanity.get("evidence_file", "")


def test_run_pre_checks_dead_guard_static_gracefully_reports_parse_failures(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _git_commit_all(repo, "initial")

    (repo / "broken.py").write_text("if True:\n    broken(\n", encoding="utf-8")

    flags = run_pre_checks(tmp_path / "plan", _state(repo), repo)

    assert any(
        flag["check"] == "dead_guard_static"
        and flag["id"].endswith("PARSE")
        and "could not be parsed as Python" in flag["detail"]
        for flag in flags
    )
