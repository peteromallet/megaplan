"""Direct tests for megaplan.prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

from megaplan._core import PlanState, atomic_write_json, atomic_write_text, save_flag_registry
from megaplan.prompts import create_claude_prompt, create_codex_prompt


def _state(project_dir: Path, *, iteration: int = 1) -> PlanState:
    return {
        "name": "test-plan",
        "idea": "collapse the workflow",
        "current_state": "critiqued",
        "iteration": iteration,
        "created_at": "2026-03-20T00:00:00Z",
        "config": {
            "project_dir": str(project_dir),
            "auto_approve": False,
            "robustness": "thorough",
        },
        "sessions": {},
        "plan_versions": [
            {
                "version": iteration,
                "file": f"plan_v{iteration}.md",
                "hash": "sha256:test",
                "timestamp": "2026-03-20T00:00:00Z",
            }
        ],
        "history": [],
        "meta": {
            "significant_counts": [],
            "weighted_scores": [3.5] if iteration > 1 else [],
            "plan_deltas": [42.0] if iteration > 1 else [],
            "recurring_critiques": [],
            "total_cost_usd": 0.0,
            "overrides": [],
            "notes": [],
        },
        "last_gate": {},
    }


def _scaffold(tmp_path: Path, *, iteration: int = 1) -> tuple[Path, PlanState]:
    plan_dir = tmp_path / "plan"
    project_dir = tmp_path / "project"
    plan_dir.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()
    state = _state(project_dir, iteration=iteration)

    atomic_write_text(plan_dir / f"plan_v{iteration}.md", "# Plan\nDo the thing.\n")
    atomic_write_json(
        plan_dir / f"plan_v{iteration}.meta.json",
        {
            "version": iteration,
            "timestamp": "2026-03-20T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": ["criterion"],
            "questions": ["question"],
            "assumptions": ["assumption"],
        },
    )
    atomic_write_json(
        plan_dir / f"critique_v{iteration}.json",
        {"flags": [], "verified_flag_ids": [], "disputed_flag_ids": []},
    )
    atomic_write_json(
        plan_dir / f"gate_signals_v{iteration}.json",
        {
            "robustness": "thorough",
            "signals": {
                "iteration": iteration,
                "weighted_score": 2.0,
                "weighted_history": [3.5] if iteration > 1 else [],
                "plan_delta_from_previous": 25.0,
                "recurring_critiques": ["same issue"] if iteration > 1 else [],
                "loop_summary": "Iteration summary",
                "scope_creep_flags": [],
            },
            "warnings": ["watch it"],
            "criteria_check": {"count": 1, "items": ["criterion"]},
            "preflight_results": {
                "project_dir_exists": True,
                "project_dir_writable": True,
                "success_criteria_present": True,
                "claude_available": True,
                "codex_available": True,
            },
            "unresolved_flags": [
                {
                    "id": "FLAG-001",
                    "concern": "still open",
                    "category": "correctness",
                    "severity": "significant",
                    "status": "open",
                    "evidence": "because",
                }
            ],
        },
    )
    atomic_write_json(
        plan_dir / "gate.json",
        {
            "passed": False,
            "criteria_check": {"count": 1, "items": ["criterion"]},
            "preflight_results": {
                "project_dir_exists": True,
                "project_dir_writable": True,
                "success_criteria_present": True,
                "claude_available": True,
                "codex_available": True,
            },
            "unresolved_flags": [],
            "recommendation": "ITERATE",
            "rationale": "revise it",
            "signals_assessment": "not ready",
            "warnings": [],
            "override_forced": False,
            "robustness": "thorough",
            "signals": {"loop_summary": "Iteration summary"},
        },
    )
    atomic_write_json(
        plan_dir / "execution.json",
        {"output": "done", "files_changed": [], "commands_run": [], "deviations": []},
    )
    save_flag_registry(
        plan_dir,
        {
            "flags": [
                {
                    "id": "FLAG-001",
                    "concern": "still open",
                    "category": "correctness",
                    "severity_hint": "likely-significant",
                    "evidence": "because",
                    "status": "open",
                    "severity": "significant",
                    "verified": False,
                    "raised_in": f"critique_v{iteration}.json",
                }
            ]
        },
    )
    return plan_dir, state


def test_plan_prompt_absorbs_clarification_when_missing(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path)
    prompt = create_claude_prompt("plan", state, plan_dir)
    assert "Identify ambiguities" in prompt
    assert "questions" in prompt
    assert state["idea"] in prompt


def test_plan_prompt_uses_existing_clarification_context(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path)
    state["clarification"] = {"intent_summary": "Keep it simple", "questions": ["What changes?"], "refined_idea": "Refined"}
    prompt = create_claude_prompt("plan", state, plan_dir)
    assert "Existing clarification context" in prompt
    assert "Keep it simple" in prompt


def test_revise_prompt_reads_gate_summary(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path)
    prompt = create_claude_prompt("revise", state, plan_dir)
    assert "Gate summary" in prompt
    assert "revise it" in prompt


def test_gate_prompt_includes_loop_signals_and_preflight(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path, iteration=2)
    prompt = create_codex_prompt("gate", state, plan_dir)
    assert "Gate signals" in prompt
    assert "Iteration summary" in prompt
    assert "preflight" in prompt.lower()
    assert "PROCEED, ITERATE, ESCALATE" in prompt


def test_review_prompt_includes_execution_and_gate(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path)
    prompt = create_claude_prompt("review", state, plan_dir)
    assert "Gate summary" in prompt
    assert "Execution summary" in prompt


def test_unsupported_step_raises(tmp_path: Path) -> None:
    plan_dir, state = _scaffold(tmp_path)
    with pytest.raises(Exception):
        create_claude_prompt("clarify", state, plan_dir)
