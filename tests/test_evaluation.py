"""Direct tests for megaplan.evaluation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from megaplan.evaluation import (
    PLAN_STRUCTURE_REQUIRED_STEP_ISSUE,
    _strip_fenced_blocks,
    build_gate_artifact,
    build_orchestrator_guidance,
    build_gate_signals,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    flag_weight,
    is_rubber_stamp,
    parse_plan_sections,
    reassemble_plan,
    renumber_steps,
    validate_execution_evidence,
    validate_plan_structure,
)
from megaplan.workers import _build_mock_payload


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _state(tmp_path: Path, *, iteration: int = 1, robustness: str = "standard") -> dict[str, object]:
    previous_version = iteration - 1
    return {
        "name": "plan",
        "idea": "ship it",
        "current_state": "critiqued",
        "iteration": iteration,
        "created_at": "2026-03-20T00:00:00Z",
        "config": {
            "project_dir": str(tmp_path / "project"),
            "auto_approve": False,
            "robustness": robustness,
        },
        "sessions": {},
        "plan_versions": (
            [
                {
                    "version": previous_version,
                    "file": f"plan_v{previous_version}a.md",
                    "hash": "sha256:prev",
                    "timestamp": "2026-03-19T00:00:00Z",
                },
                {
                    "version": iteration,
                    "file": f"plan_v{iteration}.md",
                    "hash": "sha256:current",
                    "timestamp": "2026-03-20T00:00:00Z",
                },
            ]
            if iteration > 1
            else [
            {
                "version": 1,
                "file": "plan_v1.md",
                "hash": "sha256:current",
                "timestamp": "2026-03-20T00:00:00Z",
            }
            ]
        ),
        "history": [],
        "meta": {
            "significant_counts": [],
            "weighted_scores": [4.0] if iteration > 1 else [],
            "plan_deltas": [33.0] if iteration > 1 else [],
            "recurring_critiques": [],
            "total_cost_usd": 0.0,
            "overrides": [],
            "notes": [],
        },
        "last_gate": {},
    }


def _scaffold(tmp_path: Path, *, iteration: int = 1, flags: list[dict[str, object]] | None = None) -> tuple[Path, dict[str, object]]:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (tmp_path / "project").mkdir()
    flags = flags or []
    _write_json(plan_dir / "faults.json", {"flags": flags})
    _write_json(
        plan_dir / f"critique_v{iteration}.json",
        {"flags": [{"concern": "same issue"}] if iteration > 1 else [], "verified_flag_ids": [], "disputed_flag_ids": []},
    )
    if iteration > 1:
        previous_version = iteration - 1
        _write_json(
            plan_dir / f"critique_v{previous_version}.json",
            {"flags": [{"concern": "same issue"}], "verified_flag_ids": [], "disputed_flag_ids": []},
        )
        (plan_dir / f"plan_v{previous_version}a.md").write_text("old plan\n", encoding="utf-8")
    (plan_dir / f"plan_v{iteration}.md").write_text("new plan with more detail\n", encoding="utf-8")
    _write_json(
        plan_dir / f"plan_v{iteration}.meta.json",
        {
            "version": iteration,
            "timestamp": "2026-03-20T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": ["criterion"],
            "questions": [],
            "assumptions": [],
        },
    )
    return plan_dir, _state(tmp_path, iteration=iteration)


def _signals(
    *,
    iteration: int = 2,
    weighted_score: float = 3.0,
    weighted_history: list[float] | None = None,
    recurring_critiques: list[str] | None = None,
    unresolved_flags: list[dict[str, object]] | None = None,
    scope_creep_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "weighted_score": weighted_score,
        "weighted_history": weighted_history if weighted_history is not None else [4.0],
        "recurring_critiques": recurring_critiques or [],
        "unresolved_flags": unresolved_flags or [],
        "scope_creep_flags": scope_creep_flags or [],
    }


# ---------------------------------------------------------------------------
# flag_weight tests
# ---------------------------------------------------------------------------


def test_flag_weight_security_highest() -> None:
    assert flag_weight({"category": "security"}) == 3.0


def test_flag_weight_correctness() -> None:
    assert flag_weight({"category": "correctness"}) == 2.0


def test_flag_weight_completeness() -> None:
    assert flag_weight({"category": "completeness"}) == 1.5


def test_flag_weight_performance() -> None:
    assert flag_weight({"category": "performance"}) == 1.0


def test_flag_weight_maintainability() -> None:
    assert flag_weight({"category": "maintainability"}) == 0.75


def test_flag_weight_other() -> None:
    assert flag_weight({"category": "other"}) == 1.0


def test_flag_weight_unknown_category() -> None:
    assert flag_weight({"category": "nonexistent"}) == 1.0


def test_flag_weight_missing_category() -> None:
    assert flag_weight({}) == 1.0


def test_flag_weight_implementation_detail_signals_reduce_weight() -> None:
    for signal in ["column", "schema", "field", "as written", "pseudocode", "seed sql", "placeholder"]:
        assert flag_weight({"category": "correctness", "concern": f"The {signal} is wrong"}) == 0.5


def test_flag_weight_security_overrides_implementation_detail() -> None:
    assert flag_weight({"category": "security", "concern": "The schema field is wrong"}) == 3.0


def test_flag_weight_empty_flag() -> None:
    assert flag_weight({}) == 1.0


# ---------------------------------------------------------------------------
# compute_plan_delta_percent tests
# ---------------------------------------------------------------------------


def test_compute_plan_delta_percent_returns_zero_for_identical_texts() -> None:
    assert compute_plan_delta_percent("same text", "same text") == 0.0


def test_compute_plan_delta_percent_returns_large_delta_for_different_text() -> None:
    delta = compute_plan_delta_percent("aaa", "zzz")
    assert delta is not None
    assert delta > 50.0


def test_compute_plan_delta_percent_returns_none_without_previous_text() -> None:
    assert compute_plan_delta_percent(None, "anything") is None


# ---------------------------------------------------------------------------
# compute_recurring_critiques tests
# ---------------------------------------------------------------------------


def test_compute_recurring_critiques_no_overlap(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    _write_json(plan_dir / "critique_v1.json", {"flags": [{"concern": "Issue A"}]})
    _write_json(plan_dir / "critique_v2.json", {"flags": [{"concern": "Issue B"}]})
    assert compute_recurring_critiques(plan_dir, 2) == []


def test_compute_recurring_critiques_iteration_less_than_2(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    assert compute_recurring_critiques(plan_dir, 1) == []
    assert compute_recurring_critiques(plan_dir, 0) == []


# ---------------------------------------------------------------------------
# plan structure validation tests
# ---------------------------------------------------------------------------


def test_strip_fenced_blocks_removes_only_fenced_content() -> None:
    text = """before
```python
inside
```
after
"""
    assert _strip_fenced_blocks(text) == "before\nafter\n"


def test_parse_plan_sections_basic() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

## Step 1: Update validation (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).

## Notes
Keep this section in place.

## Step 2: Add tests (`tests/test_evaluation.py`)
1. **Cover** the parser (`tests/test_evaluation.py:1`).
"""
    sections = parse_plan_sections(plan)

    assert [section.id for section in sections] == [None, None, "S1", None, "S2"]
    assert sections[0].heading == ""
    assert sections[0].start_line == 1
    assert sections[0].end_line == 2
    assert sections[2].heading == "## Step 1: Update validation (`megaplan/evaluation.py`)"
    assert sections[2].start_line == 6
    assert sections[2].end_line == 8
    assert sections[3].heading == "## Notes"
    assert sections[4].start_line == 12
    assert sections[4].end_line == 13


def test_parse_plan_sections_fenced() -> None:
    plan = """# Implementation Plan: Example

## Overview
```md
## Step 99: Fake step (`fake.py`)
1. **Ignore** this heading (`fake.py:1`).
```

## Step 1: Real step (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).

## Validation Order
1. Run tests.
"""
    sections = parse_plan_sections(plan)

    assert [section.heading for section in sections] == [
        "",
        "## Overview",
        "## Step 1: Real step (`megaplan/evaluation.py`)",
        "## Validation Order",
    ]
    assert [section.id for section in sections] == [None, None, "S1", None]


def test_renumber_steps() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summary.

## Step 1: First step (`a.py`)
1. **Do** the first part (`a.py:1`).

## Step 3: Third step (`c.py`)
1. **Do** the third part (`c.py:1`).
"""
    sections = parse_plan_sections(plan)
    renumbered = renumber_steps(sections)

    assert [section.id for section in renumbered if section.id is not None] == ["S1", "S2"]
    assert "## Step 2: Third step (`c.py`)" in renumbered[-1].body


def test_reassemble_roundtrip() -> None:
    plan = """# Implementation Plan: Example

## Overview
Intro text.

```python
print("## Step 42: not real")
```

## Step 1: Update validation (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).
"""
    assert reassemble_plan(parse_plan_sections(plan)) == plan


def test_validate_plan_structure_accepts_valid_plan() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

## Step 1: Update validation (`megaplan/evaluation.py`)
**Scope:** Small
1. **Add** the validator (`megaplan/evaluation.py:1`).

## Step 2: Add tests (`tests/test_evaluation.py`)
**Scope:** Small
1. **Cover** the expected plan shapes (`tests/test_evaluation.py:1`).

## Execution Order
1. Land the validator before wiring it.

## Validation Order
1. Run unit tests first.
"""
    assert validate_plan_structure(plan) == []


def test_validate_plan_structure_warns_when_overview_missing() -> None:
    plan = """# Implementation Plan: Example

## Step 1: Update validation (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).

## Validation Order
1. Run unit tests first.
"""
    issues = validate_plan_structure(plan)
    assert "Plan should include a `## Overview` section." in issues


def test_validate_plan_structure_errors_when_step_sections_missing() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

## Validation Order
1. Run unit tests first.
"""
    assert validate_plan_structure(plan) == [PLAN_STRUCTURE_REQUIRED_STEP_ISSUE]


def test_validate_plan_structure_warns_when_substeps_missing() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

## Step 1: Update validation (`megaplan/evaluation.py`)
No numbered substeps here.

## Validation Order
1. Run unit tests first.
"""
    issues = validate_plan_structure(plan)
    assert "Each step section should include at least one numbered substep." in issues


def test_validate_plan_structure_warns_when_ordering_sections_missing() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

## Step 1: Update validation (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).
"""
    issues = validate_plan_structure(plan)
    assert "Plan should include `## Execution Order` or `## Validation Order`." in issues


def test_validate_plan_structure_ignores_headings_inside_fenced_blocks() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

```md
## Step 99: Fake step (`fake.py`)
1. **Ignore** this heading (`fake.py:1`).
```

## Step 1: Real step (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).

## Validation Order
1. Run unit tests first.
"""
    assert validate_plan_structure(plan) == []


def test_validate_plan_structure_accepts_small_plan_with_single_ordering_section() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summarize the work.

## Step 1: Update validation (`megaplan/evaluation.py`)
1. **Add** the validator (`megaplan/evaluation.py:1`).

## Step 2: Add tests (`tests/test_evaluation.py`)
1. **Cover** the change (`tests/test_evaluation.py:1`).

## Validation Order
1. Run unit tests first.
"""
    assert validate_plan_structure(plan) == []


def test_strip_fenced_blocks_unclosed_fence_returns_original() -> None:
    text = """before
```python
inside code
## Step 99: Hidden
after fence
"""
    # Unclosed fence — should return original text to avoid silently losing content
    assert _strip_fenced_blocks(text) == text


def test_parse_plan_sections_unclosed_fence_still_finds_sections() -> None:
    plan = """## Overview
Summary.

```python
code without closing fence

## Step 1: Real step
1. Do the thing.

## Step 2: Another step
1. Do another thing.
"""
    sections = parse_plan_sections(plan)
    step_ids = [s.id for s in sections if s.id is not None]
    assert "S1" in step_ids
    assert "S2" in step_ids


def test_validate_plan_structure_accepts_phase_format() -> None:
    plan = """# Implementation Plan: Complex Feature

## Overview
Multi-phase integration.

## Phase 1: Foundation

### Step 1: Install dependencies (`package.json`)
**Scope:** Small
1. **Install** the required packages (`package.json:1`).

### Step 2: Create migration (`supabase/migrations/`)
**Scope:** Small
1. **Create** the database table (`supabase/migrations/001.sql:1`).

## Phase 2: Core Integration

### Step 3: Port the component (`src/components/Editor.tsx`)
**Scope:** Medium
1. **Copy** and adapt the component (`src/components/Editor.tsx:1`).

## Execution Order
1. Foundation before integration.

## Validation Order
1. Run tests after each phase.
"""
    assert validate_plan_structure(plan) == []


def test_parse_plan_sections_phase_format() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summary.

## Main Phase

### Step 1: First step (`a.py`)
1. **Do** the thing (`a.py:1`).

### Step 2: Second step (`b.py`)
1. **Do** the other thing (`b.py:1`).
"""
    sections = parse_plan_sections(plan)
    step_ids = [s.id for s in sections if s.id is not None]
    assert step_ids == ["S1", "S2"]
    # Phase header has no id
    phase_sections = [s for s in sections if "Main Phase" in s.heading]
    assert len(phase_sections) == 1
    assert phase_sections[0].id is None


def test_renumber_steps_phase_format() -> None:
    plan = """# Implementation Plan: Example

## Overview
Summary.

## Main Phase

### Step 1: First step (`a.py`)
1. **Do** the first part (`a.py:1`).

### Step 5: Skipped numbering (`c.py`)
1. **Do** the third part (`c.py:1`).
"""
    sections = parse_plan_sections(plan)
    renumbered = renumber_steps(sections)
    step_sections = [s for s in renumbered if s.id is not None]
    assert [s.id for s in step_sections] == ["S1", "S2"]
    assert "### Step 2: Skipped numbering (`c.py`)" in step_sections[1].body


def test_render_final_md_none_meta_commentary() -> None:
    from megaplan._core import render_final_md
    data = {
        "tasks": [],
        "watch_items": [],
        "sense_checks": [],
        "meta_commentary": None,
    }
    result = render_final_md(data)
    assert "None." in result  # Should not crash


def test_is_rubber_stamp_loose_rejects_generic_ack() -> None:
    assert is_rubber_stamp("Confirmed.", strict=False) is True


def test_is_rubber_stamp_loose_allows_short_specific_text() -> None:
    # Loose mode is blocklist-only (DECISION-001): short but specific text passes
    assert is_rubber_stamp("Too short", strict=False) is False


def test_is_rubber_stamp_strict_rejects_low_substance_text() -> None:
    assert is_rubber_stamp("Verified done good", strict=True) is True


def test_is_rubber_stamp_accepts_real_note_in_both_modes() -> None:
    note = (
        "Confirmed the review prompt still renders the audit fallback and checked that the "
        "settled-decision wording only changed reviewer framing."
    )
    assert is_rubber_stamp(note, strict=False) is False
    assert is_rubber_stamp(note, strict=True) is False


def test_is_rubber_stamp_allows_short_specific_ack_only_in_loose_mode() -> None:
    note = "Confirmed prompt coverage."
    assert is_rubber_stamp(note, strict=False) is False
    assert is_rubber_stamp(note, strict=True) is True


def test_validate_execution_evidence_flags_diff_mismatches_and_weak_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / ".git").mkdir(parents=True)
    (project_dir / "src").mkdir()
    (project_dir / "docs").mkdir()
    (project_dir / "src" / "existing.py").write_text("print('ok')\n", encoding="utf-8")
    (project_dir / "docs" / "new_name.py").write_text("x = 1\n", encoding="utf-8")

    finalize_data = {
        "tasks": [
            {
                "id": "T1",
                "files_changed": ["src/existing.py", "docs/new_name.py", "ghost.py"],
                "executor_notes": "Verified src/existing.py and confirmed the rename to docs/new_name.py showed up in git status.",
            }
        ],
        "sense_checks": [
            {"id": "SC1", "executor_note": "ok"},
        ],
    }

    monkeypatch.setattr(
        "megaplan.evaluation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["git", "status", "--short"],
            returncode=0,
            stdout=" M src/existing.py\n?? untracked.md\nR  old_name.py -> docs/new_name.py\nD  deleted.txt\n",
            stderr="",
        ),
    )

    result = validate_execution_evidence(finalize_data, project_dir)

    assert result["skipped"] is False
    assert result["files_in_diff"] == ["deleted.txt", "docs/new_name.py", "src/existing.py", "untracked.md"]
    assert result["files_claimed"] == ["docs/new_name.py", "ghost.py", "src/existing.py"]
    assert any("ghost.py" in finding for finding in result["findings"])
    assert any("deleted.txt" in finding and "untracked.md" in finding for finding in result["findings"])
    assert any("SC1" in finding and "perfunctory" in finding for finding in result["findings"])


def test_build_gate_artifact_passes_through_settled_decisions(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path)
    gate_payload = _build_mock_payload(
        "gate",
        state,
        plan_dir,
        settled_decisions=[
            {
                "id": "DECISION-001",
                "decision": "Reviewer must respect the softened FLAG-006 behavior.",
                "rationale": "This was approved at gate time.",
            }
        ],
    )
    artifact = build_gate_artifact(
        {
            "criteria_check": {"count": 1, "items": ["criterion"]},
            "preflight_results": {"project_dir_exists": True},
            "unresolved_flags": [],
            "warnings": [],
            "robustness": "standard",
            "signals": {"weighted_score": 0.5},
        },
        gate_payload,
        override_forced=False,
    )

    assert artifact["settled_decisions"] == gate_payload["settled_decisions"]


def test_validate_execution_evidence_skips_without_git_repo(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = validate_execution_evidence({"tasks": [], "sense_checks": []}, project_dir)

    assert result["skipped"] is True
    assert result["reason"] == "Project directory is not a git repository."


def test_validate_execution_evidence_flags_perfunctory_executor_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / ".git").mkdir(parents=True)

    finalize_data = {
        "tasks": [
            {
                "id": "T1",
                "status": "done",
                "files_changed": ["src/main.py"],
                "executor_notes": "Verified. Done.",
            },
            {
                "id": "T2",
                "status": "done",
                "files_changed": [],
                "executor_notes": "",
            },
        ],
        "sense_checks": [],
    }

    monkeypatch.setattr(
        "megaplan.evaluation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["git", "status", "--short"],
            returncode=0,
            stdout=" M src/main.py\n",
            stderr="",
        ),
    )

    result = validate_execution_evidence(finalize_data, project_dir)

    assert any("Task T1 executor_notes are perfunctory" in finding for finding in result["findings"])
    assert not any("Task T2 executor_notes are perfunctory" in finding for finding in result["findings"])


def test_validate_execution_evidence_skips_when_git_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / ".git").mkdir(parents=True)

    def _raise(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr("megaplan.evaluation.subprocess.run", _raise)

    result = validate_execution_evidence({"tasks": [], "sense_checks": []}, project_dir)

    assert result["skipped"] is True
    assert result["reason"] == "git not found on PATH."


def test_validate_execution_evidence_skips_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / ".git").mkdir(parents=True)

    def _raise(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["git", "status", "--short"], timeout=30)

    monkeypatch.setattr("megaplan.evaluation.subprocess.run", _raise)

    result = validate_execution_evidence({"tasks": [], "sense_checks": []}, project_dir)

    assert result["skipped"] is True
    assert result["reason"] == "git status timed out."


def test_validate_execution_evidence_skips_on_nonzero_git_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / ".git").mkdir(parents=True)

    monkeypatch.setattr(
        "megaplan.evaluation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["git", "status", "--short"],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        ),
    )

    result = validate_execution_evidence({"tasks": [], "sense_checks": []}, project_dir)

    assert result["skipped"] is True
    assert result["reason"] == "git status failed: fatal: not a git repository"


# ---------------------------------------------------------------------------
# build_gate_signals tests
# ---------------------------------------------------------------------------


def test_build_gate_signals_no_flags(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path, iteration=1, flags=[])
    result = build_gate_signals(plan_dir, state)
    assert result["signals"]["weighted_score"] == 0.0
    assert result["signals"]["unresolved_flags"] == []
    assert result["warnings"] == []


def test_build_gate_signals_iteration_5_warning(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path, iteration=5, flags=[])
    result = build_gate_signals(plan_dir, state)
    assert any("high iteration count" in w for w in result["warnings"])


def test_build_gate_signals_iteration_12_hard_limit_warning(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path, iteration=12, flags=[])
    result = build_gate_signals(plan_dir, state)
    assert any("hard iteration limit" in w for w in result["warnings"])


def test_build_gate_signals_resolved_flags_included(tmp_path: Path) -> None:
    flags = [
        {
            "id": "FLAG-001",
            "concern": "Was an issue",
            "category": "correctness",
            "severity_hint": "likely-significant",
            "evidence": "Fixed now",
            "status": "verified",
            "severity": "significant",
            "verified": True,
            "raised_in": "critique_v1.json",
        }
    ]
    plan_dir, state = _scaffold(tmp_path, iteration=1, flags=flags)
    result = build_gate_signals(plan_dir, state)
    assert len(result["signals"]["resolved_flags"]) == 1
    assert result["signals"]["resolved_flags"][0]["id"] == "FLAG-001"


def test_build_gate_signals_first_iteration_no_delta(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path, iteration=1, flags=[])
    result = build_gate_signals(plan_dir, state)
    assert result["signals"]["plan_delta_from_previous"] is None


# ---------------------------------------------------------------------------
# build_orchestrator_guidance tests
# ---------------------------------------------------------------------------


def test_build_orchestrator_guidance_first_iteration_follows_gate_with_hints() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ITERATE"},
        signals=_signals(
            iteration=1,
            weighted_history=[],
            recurring_critiques=["missing tests"],
            unresolved_flags=[{"id": "FLAG-001"}],
            scope_creep_flags=["FLAG-009"],
        ),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert "First iteration; follow gate recommendation: ITERATE." in guidance
    assert "Verify unresolved flags against the plan and project code before accepting." in guidance
    assert "Recurring critiques (missing tests)" in guidance
    assert "Scope creep detected" in guidance


def test_build_orchestrator_guidance_proceed_with_preflight_passed() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "PROCEED"},
        signals=_signals(),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert guidance == "Plan passed gate and preflight. Proceed to finalize."


def test_build_orchestrator_guidance_proceed_with_preflight_failure_lists_checks() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "PROCEED"},
        signals=_signals(unresolved_flags=[{"id": "FLAG-001"}]),
        preflight_passed=False,
        preflight_results={
            "project_dir_exists": True,
            "project_dir_writable": False,
            "success_criteria_present": False,
        },
        robustness="standard",
        plan_name="demo-plan",
    )
    assert "Gate says PROCEED but preflight blocked. Fix: project_dir_writable, success_criteria_present." in guidance


def test_build_orchestrator_guidance_escalate_auto_force_uses_plan_name_and_score() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ESCALATE"},
        signals=_signals(weighted_score=4.0, weighted_history=[5.0]),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="light",
        plan_name="demo-plan",
    )
    assert "Auto-force-proceed eligible." in guidance
    assert "megaplan override force-proceed --plan demo-plan" in guidance
    assert 'light robustness, score 4.0' in guidance


def test_build_orchestrator_guidance_escalate_requires_user_decision_when_not_auto_force() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ESCALATE"},
        signals=_signals(weighted_score=5.0, weighted_history=[4.0]),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert guidance == "Gate escalated. Ask the user: force-proceed, add-note, or abort."


def test_build_orchestrator_guidance_iterate_plateaued_with_recurring_critiques() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ITERATE"},
        signals=_signals(
            weighted_score=3.0,
            weighted_history=[2.0],
            recurring_critiques=["missing tests"],
        ),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert "Score plateaued with recurring critiques the loop can't fix." in guidance
    assert "megaplan override force-proceed --plan demo-plan" in guidance


def test_build_orchestrator_guidance_iterate_improving() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ITERATE"},
        signals=_signals(weighted_score=2.0, weighted_history=[3.5]),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert guidance == "Score improving (3.5 -> 2.0). Continue to revise."


def test_build_orchestrator_guidance_iterate_worsening() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ITERATE"},
        signals=_signals(weighted_score=4.5, weighted_history=[3.0]),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert guidance == "Score worsening (3.0 -> 4.5). Investigate; the loop may be diverging."


def test_build_orchestrator_guidance_iterate_fallthrough() -> None:
    guidance = build_orchestrator_guidance(
        gate_payload={"recommendation": "ITERATE"},
        signals=_signals(weighted_score=3.0, weighted_history=[3.0]),
        preflight_passed=True,
        preflight_results={"project_dir_exists": True},
        robustness="standard",
        plan_name="demo-plan",
    )
    assert guidance == "Gate recommends another iteration. Revise the plan."


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------


def test_flag_weight_preserves_low_weight_implementation_details() -> None:
    assert flag_weight({"category": "correctness", "concern": "The schema field is wrong"}) == 0.5
    assert flag_weight({"category": "security", "concern": "The schema field is wrong"}) == 3.0


def test_compute_plan_delta_percent_handles_none() -> None:
    assert compute_plan_delta_percent(None, "x") is None
    assert compute_plan_delta_percent("same", "same") == 0.0


def test_compute_recurring_critiques_detects_overlap(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    _write_json(plan_dir / "critique_v1.json", {"flags": [{"concern": "Same issue"}]})
    _write_json(plan_dir / "critique_v2.json", {"flags": [{"concern": "same issue"}]})
    assert compute_recurring_critiques(plan_dir, 2) == ["same issue"]


def test_build_gate_signals_includes_loop_summary_and_previous_version_lookup(tmp_path: Path) -> None:
    flags = [
        {
            "id": "FLAG-001",
            "concern": "Missing verification command",
            "category": "correctness",
            "severity_hint": "likely-significant",
            "evidence": "No test listed",
            "status": "open",
            "severity": "significant",
            "verified": False,
            "raised_in": "critique_v2.json",
        }
    ]
    plan_dir, state = _scaffold(tmp_path, iteration=2, flags=flags)
    result = build_gate_signals(plan_dir, state)
    assert result["robustness"] == "standard"
    assert result["signals"]["weighted_score"] == 2.0
    assert result["signals"]["plan_delta_from_previous"] is not None
    assert result["signals"]["recurring_critiques"] == ["same issue"]
    assert "Weighted score trajectory" in result["signals"]["loop_summary"]


def test_build_gate_signals_emits_scope_creep_and_high_iteration_warnings(tmp_path: Path) -> None:
    flags = [
        {
            "id": "FLAG-007",
            "concern": "Scope creep: plan now rewrites the entire app",
            "category": "other",
            "severity_hint": "likely-significant",
            "evidence": "expanded scope",
            "status": "open",
            "severity": "significant",
            "verified": False,
            "raised_in": "critique_v12.json",
        }
    ]
    plan_dir, state = _scaffold(tmp_path, iteration=12, flags=flags)
    state["config"]["robustness"] = "thorough"
    result = build_gate_signals(plan_dir, state)
    assert any("Scope creep detected" in warning for warning in result["warnings"])
    assert any("hard iteration limit reached" in warning for warning in result["warnings"])
