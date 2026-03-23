"""Direct tests for megaplan.evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from megaplan.evaluation import (
    build_orchestrator_guidance,
    build_gate_signals,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    flag_weight,
)


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
