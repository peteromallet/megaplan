"""Direct tests for megaplan.evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from megaplan.evaluation import (
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
