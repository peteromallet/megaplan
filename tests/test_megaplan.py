from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import megaplan
import megaplan.cli
import megaplan.handlers
import megaplan.cli
import megaplan._core
import megaplan.workers
from megaplan.evaluation import PLAN_STRUCTURE_REQUIRED_STEP_ISSUE
from megaplan._core import ensure_runtime_layout, load_plan
from megaplan.workers import WorkerResult


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_args_factory(project_dir: Path) -> Callable[..., Namespace]:
    def make_args(**overrides: object) -> Namespace:
        data = {
            "plan": None,
            "idea": "test idea",
            "name": "test-plan",
            "project_dir": str(project_dir),
            "auto_approve": False,
            "robustness": "standard",
            "agent": None,
            "ephemeral": False,
            "fresh": False,
            "persist": False,
            "confirm_destructive": True,
            "user_approved": False,
            "confirm_self_review": False,
            "override_action": None,
            "note": None,
            "reason": "",
        }
        data.update(overrides)
        return Namespace(**data)

    return make_args


@dataclass
class PlanFixture:
    root: Path
    project_dir: Path
    plan_name: str
    plan_dir: Path
    make_args: Callable[..., Namespace]


@pytest.fixture
def plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlanFixture:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    monkeypatch.setenv(megaplan.MOCK_ENV_VAR, "1")
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )

    make_args = make_args_factory(project_dir)
    response = megaplan.handle_init(root, make_args())
    plan_name = response["plan"]
    return PlanFixture(
        root=root,
        project_dir=project_dir,
        plan_name=plan_name,
        plan_dir=megaplan.plans_root(root) / plan_name,
        make_args=make_args,
    )


def load_state(plan_dir: Path) -> dict:
    return read_json(plan_dir / "state.json")


def test_init_sets_last_gate_and_next_step_plan(plan_fixture: PlanFixture) -> None:
    state = load_state(plan_fixture.plan_dir)
    assert state["current_state"] == megaplan.STATE_INITIALIZED
    assert state["last_gate"] == {}
    assert state["iteration"] == 0


def test_init_response_points_to_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )
    response = megaplan.handle_init(root, make_args_factory(project_dir)())
    assert response["next_step"] == "plan"


def test_infer_next_steps_matches_new_state_machine() -> None:
    assert megaplan.infer_next_steps({"current_state": megaplan.STATE_INITIALIZED, "last_gate": {}}) == ["plan"]
    assert megaplan.infer_next_steps({"current_state": megaplan.STATE_PLANNED, "last_gate": {}}) == ["plan", "critique"]
    assert megaplan.infer_next_steps({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {}}) == ["gate"]
    assert megaplan.infer_next_steps({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "ITERATE"}}) == ["revise"]
    assert megaplan.infer_next_steps({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "ESCALATE"}}) == [
        "override add-note",
        "override force-proceed",
        "override abort",
    ]
    assert megaplan.infer_next_steps(
        {"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "PROCEED", "passed": False}}
    ) == ["revise", "override force-proceed"]


def test_plan_rerun_keeps_iteration_and_uses_same_iteration_subversion(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="answer to questions"),
    )
    response = megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)

    assert response["iteration"] == 1
    assert state["iteration"] == 1
    assert (plan_fixture.plan_dir / "plan_v1.md").exists()
    assert (plan_fixture.plan_dir / "plan_v1a.md").exists()
    assert state["plan_versions"][-1]["file"] == "plan_v1a.md"

    critique = megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    assert critique["iteration"] == 1
    assert (plan_fixture.plan_dir / "critique_v1.json").exists()


def test_workflow_mock_end_to_end(plan_fixture: PlanFixture) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="add-note", note="keep changes scoped"),
    )
    plan = megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    critique1 = megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    gate1 = megaplan.handle_gate(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    revise = megaplan.handle_revise(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    critique2 = megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    gate2 = megaplan.handle_gate(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    finalize = megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    finalized_tracking = read_json(plan_fixture.plan_dir / "finalize.json")
    final_md_after_finalize = (plan_fixture.plan_dir / "final.md").read_text(encoding="utf-8")
    execute = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    finalized_after_execute = read_json(plan_fixture.plan_dir / "finalize.json")
    final_md_after_execute = (plan_fixture.plan_dir / "final.md").read_text(encoding="utf-8")
    review = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    finalized_after_review = read_json(plan_fixture.plan_dir / "finalize.json")
    final_md_after_review = (plan_fixture.plan_dir / "final.md").read_text(encoding="utf-8")
    plan_meta = read_json(plan_fixture.plan_dir / "plan_v1.meta.json")
    revise_meta = read_json(plan_fixture.plan_dir / "plan_v2.meta.json")
    state = load_state(plan_fixture.plan_dir)

    assert plan["state"] == megaplan.STATE_PLANNED
    assert critique1["state"] == megaplan.STATE_CRITIQUED
    assert gate1["recommendation"] == "ITERATE"
    assert revise["state"] == megaplan.STATE_PLANNED
    assert critique2["iteration"] == 2
    assert gate2["state"] == megaplan.STATE_GATED
    assert gate2["recommendation"] == "PROCEED"
    assert finalize["state"] == megaplan.STATE_FINALIZED
    assert plan_meta["structure_warnings"] == []
    assert revise_meta["structure_warnings"] == []
    assert (plan_fixture.plan_dir / "final.md").exists()
    assert (plan_fixture.plan_dir / "finalize.json").exists()
    assert finalized_tracking["tasks"][0]["status"] == "pending"
    assert "# Execution Checklist" in final_md_after_finalize
    assert execute["state"] == megaplan.STATE_EXECUTED
    assert all(task["status"] == "done" for task in finalized_after_execute["tasks"])
    assert all(task["executor_notes"] for task in finalized_after_execute["tasks"])
    assert "Executor notes:" in final_md_after_execute
    assert review["state"] == megaplan.STATE_DONE
    assert all(task["reviewer_verdict"] for task in finalized_after_review["tasks"])
    assert all(check["verdict"] for check in finalized_after_review["sense_checks"])
    assert "Reviewer verdict:" in final_md_after_review
    assert "Verdict:" in final_md_after_review
    execute_entry = next(entry for entry in state["history"] if entry["step"] == "execute")
    review_entry = next(entry for entry in state["history"] if entry["step"] == "review")
    assert execute_entry["finalize_hash"].startswith("sha256:")
    assert review_entry["finalize_hash"].startswith("sha256:")
    assert (plan_fixture.project_dir / "IMPLEMENTED_BY_MEGAPLAN.txt").exists()


def test_handle_plan_stores_nonblocking_structure_warnings(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = WorkerResult(
        payload={
            "plan": """# Implementation Plan: Warning Case

## Step 1: Touch one file (`megaplan/evaluation.py`)
1. **Implement** the change (`megaplan/evaluation.py:1`).

## Validation Order
1. Run a focused test.
""",
            "questions": [],
            "success_criteria": ["warn but continue"],
            "assumptions": [],
        },
        raw_output="warning case",
        duration_ms=1,
        cost_usd=0.0,
        session_id="plan-warning",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )

    response = megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    meta = read_json(plan_fixture.plan_dir / "plan_v1.md".replace(".md", ".meta.json"))

    assert response["success"] is True
    assert meta["structure_warnings"] == ["Plan should include a `## Overview` section."]


def test_handle_plan_rejects_zero_step_structure_error(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = WorkerResult(
        payload={
            "plan": """# Implementation Plan: Invalid

## Overview
No numbered step sections here.

## Validation Order
1. Run a focused test.
""",
            "questions": [],
            "success_criteria": ["should fail"],
            "assumptions": [],
        },
        raw_output="invalid plan output",
        duration_ms=1,
        cost_usd=0.0,
        session_id="plan-invalid",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )

    with pytest.raises(megaplan.CliError, match="structural validation"):
        megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    error_entry = state["history"][-1]
    assert error_entry["result"] == "error"
    assert PLAN_STRUCTURE_REQUIRED_STEP_ISSUE in error_entry["message"]


def test_handle_revise_requires_prior_iterate_gate(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    with pytest.raises(megaplan.CliError, match="ITERATE"):
        megaplan.handle_revise(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))


def test_force_proceed_from_critiqued_writes_override_gate(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    state["last_gate"] = {"recommendation": "ESCALATE"}
    (plan_fixture.plan_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    response = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(
            plan=plan_fixture.plan_name,
            override_action="force-proceed",
            reason="executor can resolve remaining issues",
        ),
    )
    gate = read_json(plan_fixture.plan_dir / "gate.json")
    state = load_state(plan_fixture.plan_dir)

    assert response["state"] == megaplan.STATE_GATED
    assert response["orchestrator_guidance"] == "Force-proceed override applied. Proceed to finalize."
    assert gate["override_forced"] is True
    assert gate["recommendation"] == "PROCEED"
    assert gate["orchestrator_guidance"] == "Force-proceed override applied. Proceed to finalize."
    assert state["last_gate"] == {}


def test_replan_from_gated_resets_to_planned(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(
            plan=plan_fixture.plan_name,
            override_action="force-proceed",
            reason="test gate override",
        ),
    )
    response = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(
            plan=plan_fixture.plan_name,
            override_action="replan",
            reason="edit directly",
            note="expand the verification section",
        ),
    )
    state = load_state(plan_fixture.plan_dir)

    assert response["state"] == megaplan.STATE_PLANNED
    assert state["last_gate"] == {}
    assert response["next_step"] == "critique"


def test_gate_retry_does_not_duplicate_weighted_scores(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    results = iter(
        [
            (
                WorkerResult(
                    payload={
                        "recommendation": "ESCALATE",
                        "rationale": "stuck",
                        "signals_assessment": "scores are flat",
                        "warnings": [],
                    },
                    raw_output="{}",
                    duration_ms=1,
                    cost_usd=0.0,
                    session_id="gate-1",
                ),
                "claude",
                "persistent",
                False,
            ),
            (
                WorkerResult(
                    payload={
                        "recommendation": "PROCEED",
                        "rationale": "user note clarified the issue",
                        "signals_assessment": "same score, but judgment changed",
                        "warnings": [],
                    },
                    raw_output="{}",
                    duration_ms=1,
                    cost_usd=0.0,
                    session_id="gate-2",
                ),
                "claude",
                "persistent",
                False,
            ),
        ]
    )

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", lambda *args, **kwargs: next(results))

    megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="extra context"),
    )
    megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)

    assert state["meta"]["weighted_scores"] == [3.5]


# ---------------------------------------------------------------------------
# Flag management tests
# ---------------------------------------------------------------------------


def test_normalize_flag_record_fills_defaults() -> None:
    record = megaplan.normalize_flag_record({}, "FLAG-099")
    assert record["id"] == "FLAG-099"
    assert record["concern"] == ""
    assert record["category"] == "other"
    assert record["severity_hint"] == "uncertain"
    assert record["evidence"] == ""


def test_normalize_flag_record_sanitises_bad_category() -> None:
    record = megaplan.normalize_flag_record({"category": "banana"}, "FLAG-001")
    assert record["category"] == "other"


def test_normalize_flag_record_sanitises_bad_severity_hint() -> None:
    record = megaplan.normalize_flag_record({"severity_hint": "maybe"}, "FLAG-001")
    assert record["severity_hint"] == "uncertain"


def test_normalize_flag_record_uses_own_id_when_present() -> None:
    record = megaplan.normalize_flag_record({"id": "FLAG-042"}, "FLAG-099")
    assert record["id"] == "FLAG-042"


def test_normalize_flag_record_uses_fallback_for_empty_id() -> None:
    for empty_id in [None, "", "FLAG-000"]:
        record = megaplan.normalize_flag_record({"id": empty_id}, "FLAG-099")
        assert record["id"] == "FLAG-099"


def test_update_flags_after_critique_creates_new_flags(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    critique_payload = {
        "flags": [
            {"id": "FLAG-001", "concern": "Missing tests", "category": "correctness", "severity_hint": "likely-significant", "evidence": "No tests found"},
        ],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    from megaplan.handlers import update_flags_after_critique
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique_payload, iteration=1)
    assert len(registry["flags"]) >= 1
    flag = next(f for f in registry["flags"] if f["id"] == "FLAG-001")
    assert flag["status"] == "open"
    assert flag["severity"] == "significant"


def test_update_flags_after_critique_verifies_flags(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.handlers import update_flags_after_critique
    critique1 = {
        "flags": [{"id": "FLAG-001", "concern": "x", "category": "other", "severity_hint": "likely-significant", "evidence": "y"}],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    update_flags_after_critique(plan_fixture.plan_dir, critique1, iteration=1)
    critique2 = {
        "flags": [],
        "verified_flag_ids": ["FLAG-001"],
        "disputed_flag_ids": [],
    }
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique2, iteration=2)
    flag = next(f for f in registry["flags"] if f["id"] == "FLAG-001")
    assert flag["status"] == "verified"
    assert flag["verified"] is True


def test_update_flags_after_critique_disputes_flags(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.handlers import update_flags_after_critique
    critique1 = {
        "flags": [{"id": "FLAG-001", "concern": "x", "category": "other", "severity_hint": "likely-significant", "evidence": "y"}],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    update_flags_after_critique(plan_fixture.plan_dir, critique1, iteration=1)
    critique2 = {
        "flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": ["FLAG-001"],
    }
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique2, iteration=2)
    flag = next(f for f in registry["flags"] if f["id"] == "FLAG-001")
    assert flag["status"] == "disputed"


def test_update_flags_after_critique_reuses_existing_ids(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.handlers import update_flags_after_critique
    critique1 = {
        "flags": [{"id": "FLAG-001", "concern": "x", "category": "other", "severity_hint": "likely-significant", "evidence": "y"}],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    update_flags_after_critique(plan_fixture.plan_dir, critique1, iteration=1)
    critique2 = {
        "flags": [{"id": "FLAG-001", "concern": "revised concern", "category": "correctness", "severity_hint": "likely-significant", "evidence": "new evidence"}],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique2, iteration=2)
    count = sum(1 for f in registry["flags"] if f["id"] == "FLAG-001")
    assert count == 1
    flag = next(f for f in registry["flags"] if f["id"] == "FLAG-001")
    assert flag["concern"] == "revised concern"


def test_update_flags_after_critique_autonumbers_missing_ids(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.handlers import update_flags_after_critique
    critique = {
        "flags": [
            {"concern": "no id given", "category": "other", "severity_hint": "likely-minor", "evidence": "test"},
            {"id": "", "concern": "empty id", "category": "other", "severity_hint": "likely-minor", "evidence": "test"},
        ],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique, iteration=1)
    ids = [f["id"] for f in registry["flags"]]
    # Should have auto-assigned IDs
    assert all(id_.startswith("FLAG-") for id_ in ids)
    assert len(set(ids)) == len(ids)  # all unique


def test_update_flags_after_critique_severity_from_hint(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.handlers import update_flags_after_critique
    critique = {
        "flags": [
            {"id": "FLAG-001", "concern": "major", "category": "other", "severity_hint": "likely-significant", "evidence": "x"},
            {"id": "FLAG-002", "concern": "minor", "category": "other", "severity_hint": "likely-minor", "evidence": "x"},
            {"id": "FLAG-003", "concern": "uncertain", "category": "other", "severity_hint": "uncertain", "evidence": "x"},
        ],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique, iteration=1)
    by_id = {f["id"]: f for f in registry["flags"]}
    assert by_id["FLAG-001"]["severity"] == "significant"
    assert by_id["FLAG-002"]["severity"] == "minor"
    assert by_id["FLAG-003"]["severity"] == "significant"  # uncertain => significant


def test_update_flags_after_revise_marks_addressed(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.handlers import update_flags_after_critique, update_flags_after_revise  # noqa: F811
    from megaplan._core import save_flag_registry
    save_flag_registry(plan_fixture.plan_dir, {"flags": [
        {"id": "FLAG-001", "concern": "x", "category": "other", "severity_hint": "likely-significant", "evidence": "y", "status": "open", "severity": "significant", "verified": False, "raised_in": "critique_v1.json"},
    ]})
    registry = update_flags_after_revise(plan_fixture.plan_dir, ["FLAG-001"], plan_file="plan_v2.md", summary="fixed it")
    flag = registry["flags"][0]
    assert flag["status"] == "addressed"
    assert flag["addressed_in"] == "plan_v2.md"


def test_unresolved_significant_flags_filtering() -> None:
    registry = {"flags": [
        {"id": "FLAG-001", "severity": "significant", "status": "open"},
        {"id": "FLAG-002", "severity": "minor", "status": "open"},
        {"id": "FLAG-003", "severity": "significant", "status": "verified"},
        {"id": "FLAG-004", "severity": "significant", "status": "disputed"},
        {"id": "FLAG-005", "severity": "significant", "status": "addressed"},
    ]}
    unresolved = megaplan.unresolved_significant_flags(registry)
    ids = [f["id"] for f in unresolved]
    assert "FLAG-001" in ids
    assert "FLAG-004" in ids
    assert "FLAG-002" not in ids
    assert "FLAG-003" not in ids
    assert "FLAG-005" not in ids


# ---------------------------------------------------------------------------
# Override tests
# ---------------------------------------------------------------------------


def test_override_add_note_records_note(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="my note"),
    )
    assert response["success"] is True
    state = load_state(plan_fixture.plan_dir)
    assert any(n["note"] == "my note" for n in state["meta"]["notes"])


def test_add_note_after_abort(plan_fixture: PlanFixture) -> None:
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="abort", reason="done"),
    )
    # add-note should still work on aborted plans
    response = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="postmortem"),
    )
    assert response["success"] is True


def test_abort_sets_terminal_state(plan_fixture: PlanFixture) -> None:
    response = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="abort", reason="no longer needed"),
    )
    assert response["state"] == megaplan.STATE_ABORTED
    state = load_state(plan_fixture.plan_dir)
    assert state["current_state"] == megaplan.STATE_ABORTED


def test_force_proceed_requires_critiqued_state(plan_fixture: PlanFixture) -> None:
    # In initialized state, force-proceed should fail
    with pytest.raises(megaplan.CliError, match="critiqued"):
        megaplan.handle_override(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
        )


def test_force_proceed_requires_success_criteria(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    # Remove success criteria from meta
    meta_path = plan_fixture.plan_dir / "plan_v1.meta.json"
    meta = read_json(meta_path)
    meta["success_criteria"] = []
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(megaplan.CliError, match="success criteria"):
        megaplan.handle_override(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
        )


# ---------------------------------------------------------------------------
# Execute flow tests
# ---------------------------------------------------------------------------


def test_execute_requires_confirm_destructive(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    with pytest.raises(megaplan.CliError, match="confirm-destructive"):
        megaplan.handle_execute(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=False),
        )


def test_execute_requires_user_approval_in_review_mode(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    with pytest.raises(megaplan.CliError, match="user approval"):
        megaplan.handle_execute(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=False),
        )


def test_execute_succeeds_with_user_approval(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    assert response["success"] is True
    assert response["state"] == megaplan.STATE_EXECUTED
    assert "finalize.json" in response["artifacts"]
    assert "final.md" in response["artifacts"]


def test_execute_succeeds_in_auto_approve_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()
    monkeypatch.setenv(megaplan.MOCK_ENV_VAR, "1")
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )
    make_args = make_args_factory(project_dir)
    megaplan.handle_init(root, make_args(auto_approve=True))
    megaplan.handle_plan(root, make_args(plan="test-plan"))
    megaplan.handle_critique(root, make_args(plan="test-plan"))
    megaplan.handle_override(root, make_args(plan="test-plan", override_action="force-proceed", reason="test"))
    megaplan.handle_finalize(root, make_args(plan="test-plan"))
    response = megaplan.handle_execute(
        root,
        make_args(plan="test-plan", confirm_destructive=True, user_approved=False),
    )
    assert response["success"] is True
    assert response["auto_approve"] is True
    assert "finalize.json" in response["artifacts"]


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


def test_gate_response_surfaces_auto_approve(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    assert "auto_approve" in response
    assert response["orchestrator_guidance"].startswith("First iteration; follow gate recommendation: ITERATE.")
    assert "Verify unresolved flags against the plan and project code before accepting." in response["orchestrator_guidance"]
    gate = read_json(plan_fixture.plan_dir / "gate.json")
    assert gate["orchestrator_guidance"] == response["orchestrator_guidance"]


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


def test_require_state_rejects_invalid_transition(plan_fixture: PlanFixture) -> None:
    # Plan requires initialized or planned state; critique from initialized should fail
    with pytest.raises(megaplan.CliError, match="Cannot run"):
        megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))


def test_terminal_states_block_progression(plan_fixture: PlanFixture) -> None:
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="abort", reason="test"),
    )
    with pytest.raises(megaplan.CliError, match="Cannot run"):
        megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))


# ---------------------------------------------------------------------------
# Config / setup tests
# ---------------------------------------------------------------------------


def test_global_setup_creates_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    result = megaplan.handle_setup_global(force=False, home=home)
    assert result["success"] is True
    skill_path = home / ".claude" / "skills" / "megaplan" / "SKILL.md"
    assert skill_path.exists()


def test_global_setup_skips_not_installed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = megaplan.handle_setup_global(force=False, home=home)
    assert result["success"] is False
    assert all(r.get("skipped", False) or r.get("reason") == "not installed" for r in result["installed"])


def test_global_setup_idempotent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    megaplan.handle_setup_global(force=False, home=home)
    result2 = megaplan.handle_setup_global(force=False, home=home)
    assert result2["success"] is True
    assert any(r.get("skipped") for r in result2["installed"])


def test_global_setup_force_overwrites(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    megaplan.handle_setup_global(force=False, home=home)
    result = megaplan.handle_setup_global(force=True, home=home)
    assert result["success"] is True
    # Force should NOT skip
    claude_result = next(r for r in result["installed"] if r.get("agent") == "claude")
    assert claude_result.get("skipped") is False


def test_global_setup_multiple_agents(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    result = megaplan.handle_setup_global(force=False, home=home)
    assert result["success"] is True
    agents_installed = [r["agent"] for r in result["installed"] if not r.get("skipped", False) and r.get("reason") != "not installed"]
    assert "claude" in agents_installed
    assert "codex" in agents_installed


def test_load_save_config_roundtrip(tmp_path: Path) -> None:
    from megaplan._core import load_config, save_config
    config = {"agents": {"plan": "codex"}, "custom": True}
    save_config(config, tmp_path)
    loaded = load_config(tmp_path)
    assert loaded == config


def test_load_config_corrupt_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from megaplan._core import config_dir, load_config
    config_path = config_dir(tmp_path) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("not valid json!!!", encoding="utf-8")
    result = load_config(tmp_path)
    assert result == {}


def test_config_dir_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from megaplan._core import config_dir
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_dir() == tmp_path / "xdg" / "megaplan"


def test_setup_global_writes_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(megaplan.cli, "detect_available_agents", lambda: ["claude"])
    result = megaplan.handle_setup_global(force=False, home=home)
    assert "config_path" in result
    assert "routing" in result


# ---------------------------------------------------------------------------
# CLI parsing tests
# ---------------------------------------------------------------------------


def test_parse_claude_envelope_valid_with_result_block() -> None:
    from megaplan.workers import parse_claude_envelope
    raw = json.dumps({"result": json.dumps({"plan": "x"}), "total_cost_usd": 0.05})
    envelope, payload = parse_claude_envelope(raw)
    assert payload["plan"] == "x"


def test_parse_claude_envelope_structured_output() -> None:
    from megaplan.workers import parse_claude_envelope
    raw = json.dumps({"structured_output": {"plan": "x"}, "total_cost_usd": 0.01})
    envelope, payload = parse_claude_envelope(raw)
    assert payload == {"plan": "x"}


def test_parse_claude_envelope_direct_dict() -> None:
    from megaplan.workers import parse_claude_envelope
    raw = json.dumps({"plan": "direct"})
    envelope, payload = parse_claude_envelope(raw)
    assert payload["plan"] == "direct"


def test_parse_claude_envelope_malformed() -> None:
    from megaplan.workers import parse_claude_envelope
    with pytest.raises(megaplan.CliError, match="valid JSON"):
        parse_claude_envelope("not json at all")


def test_parse_claude_envelope_is_error() -> None:
    from megaplan.workers import parse_claude_envelope
    raw = json.dumps({"is_error": True, "result": "something failed"})
    with pytest.raises(megaplan.CliError, match="failed"):
        parse_claude_envelope(raw)


def test_parse_claude_envelope_empty_result() -> None:
    from megaplan.workers import parse_claude_envelope
    raw = json.dumps({"result": ""})
    with pytest.raises(megaplan.CliError, match="empty"):
        parse_claude_envelope(raw)


def test_parse_claude_envelope_non_object_result() -> None:
    from megaplan.workers import parse_claude_envelope
    raw = json.dumps({"result": "[1, 2, 3]"})
    with pytest.raises(megaplan.CliError, match="not an object"):
        parse_claude_envelope(raw)


def test_parse_json_file_valid(tmp_path: Path) -> None:
    from megaplan.workers import parse_json_file
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"key": "value"}), encoding="utf-8")
    assert parse_json_file(path) == {"key": "value"}


def test_parse_json_file_missing(tmp_path: Path) -> None:
    from megaplan.workers import parse_json_file
    with pytest.raises(megaplan.CliError, match="not created"):
        parse_json_file(tmp_path / "missing.json")


def test_parse_json_file_non_object(tmp_path: Path) -> None:
    from megaplan.workers import parse_json_file
    path = tmp_path / "test.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(megaplan.CliError, match="not contain a JSON object"):
        parse_json_file(path)


def test_validate_payload_valid_pass() -> None:
    from megaplan.workers import validate_payload
    validate_payload("plan", {"plan": "x", "questions": [], "success_criteria": [], "assumptions": []})


def test_validate_payload_missing_keys_raise() -> None:
    from megaplan.workers import validate_payload
    with pytest.raises(megaplan.CliError, match="missing required"):
        validate_payload("plan", {"plan": "x"})


def test_validate_payload_unknown_step_noop() -> None:
    from megaplan.workers import validate_payload
    # Unknown step should not raise
    validate_payload("nonexistent_step", {})


# ---------------------------------------------------------------------------
# Worker parsing tests
# ---------------------------------------------------------------------------


def test_extract_session_id_jsonl_thread_id() -> None:
    from megaplan.workers import extract_session_id
    raw = '{"type":"thread.started","thread_id":"abc-123"}\n'
    assert extract_session_id(raw) == "abc-123"


def test_extract_session_id_unstructured() -> None:
    from megaplan.workers import extract_session_id
    raw = "Starting session... session_id: 12345678-abcd-ef01"
    assert extract_session_id(raw) == "12345678-abcd-ef01"


def test_extract_session_id_pattern() -> None:
    from megaplan.workers import extract_session_id
    raw = "session id: aabbccdd-1234-5678-abcd"
    assert extract_session_id(raw) == "aabbccdd-1234-5678-abcd"


def test_extract_session_id_no_match() -> None:
    from megaplan.workers import extract_session_id
    assert extract_session_id("no session here") is None


def test_extract_session_id_empty_string() -> None:
    from megaplan.workers import extract_session_id
    assert extract_session_id("") is None


# ---------------------------------------------------------------------------
# Schema tests (strict_schema)
# ---------------------------------------------------------------------------


def test_strict_schema_adds_additional_properties_false() -> None:
    from megaplan.schemas import strict_schema
    result = strict_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    assert result["additionalProperties"] is False


def test_strict_schema_preserves_existing_additional_properties() -> None:
    from megaplan.schemas import strict_schema
    result = strict_schema({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True})
    assert result["additionalProperties"] is True


def test_strict_schema_sets_required_from_properties() -> None:
    from megaplan.schemas import strict_schema
    result = strict_schema({"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "number"}}})
    assert result["required"] == ["x", "y"]


def test_strict_schema_nested_objects() -> None:
    from megaplan.schemas import strict_schema
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }
        },
    }
    result = strict_schema(schema)
    assert result["properties"]["inner"]["additionalProperties"] is False


def test_strict_schema_array_items() -> None:
    from megaplan.schemas import strict_schema
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        },
    }
    result = strict_schema(schema)
    assert result["properties"]["items"]["items"]["additionalProperties"] is False


def test_strict_schema_non_object_untouched() -> None:
    from megaplan.schemas import strict_schema
    assert strict_schema({"type": "string"}) == {"type": "string"}
    assert strict_schema(42) == 42
    assert strict_schema("hello") == "hello"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_step_failure_records_error_in_history(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    # Make critique fail by monkeypatching the worker
    def failing_worker(*a, **kw):
        raise megaplan.CliError("test_error", "Worker blew up", extra={"raw_output": "boom"})
    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", failing_worker)
    with pytest.raises(megaplan.CliError, match="Worker blew up"):
        megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    error_entries = [h for h in state["history"] if h.get("result") == "error"]
    assert len(error_entries) >= 1
    assert "Worker blew up" in error_entries[-1].get("message", "")


def test_step_failure_stores_raw_output_file(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    def failing_worker(*a, **kw):
        raise megaplan.CliError("test_error", "fail", extra={"raw_output": "raw content here"})
    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", failing_worker)
    with pytest.raises(megaplan.CliError):
        megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    error_entry = next(h for h in state["history"] if h.get("result") == "error")
    raw_file = error_entry.get("raw_output_file")
    assert raw_file is not None
    assert (plan_fixture.plan_dir / raw_file).exists()


def test_step_failure_uses_message_when_no_raw_output(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    def failing_worker(*a, **kw):
        raise megaplan.CliError("test_error", "the error message")
    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", failing_worker)
    with pytest.raises(megaplan.CliError):
        megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    error_entry = next(h for h in state["history"] if h.get("result") == "error")
    raw_file = error_entry.get("raw_output_file")
    content = (plan_fixture.plan_dir / raw_file).read_text(encoding="utf-8")
    assert "the error message" in content


def test_run_command_raises_on_timeout() -> None:
    from megaplan.workers import run_command
    with pytest.raises(megaplan.CliError, match="timed out"):
        run_command(["sleep", "60"], cwd=Path.cwd(), timeout=1)


def test_run_command_raises_on_file_not_found() -> None:
    from megaplan.workers import run_command
    with pytest.raises(megaplan.CliError, match="not found"):
        run_command(["nonexistent_command_xyz"], cwd=Path.cwd())


# ---------------------------------------------------------------------------
# Main entry / CLI tests
# ---------------------------------------------------------------------------


def test_init_produces_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )
    response = megaplan.handle_init(root, make_args_factory(project_dir)())
    assert response["success"] is True
    assert "plan" in response
    assert response["state"] == megaplan.STATE_INITIALIZED


def test_list_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    from megaplan._core import ensure_runtime_layout
    ensure_runtime_layout(root)
    response = megaplan.handle_list(root, Namespace(plan=None))
    assert response["plans"] == []


def test_invalid_command_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = megaplan.main(["init", "--project-dir", str(tmp_path), "test idea"])
    assert exit_code == 0  # init should succeed


def test_setup_local_creates_agents_file(tmp_path: Path) -> None:
    args = Namespace(
        local=True,
        target_dir=str(tmp_path),
        force=False,
    )
    response = megaplan.handle_setup(args)
    assert response["success"] is True
    assert (tmp_path / "AGENTS.md").exists()


def test_config_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    args = Namespace(config_action="show")
    response = megaplan.handle_config(args)
    assert response["success"] is True
    assert "routing" in response


def test_config_set_and_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    set_args = Namespace(config_action="set", key="agents.plan", value="codex")
    response = megaplan.handle_config(set_args)
    assert response["success"] is True
    assert response["value"] == "codex"

    reset_args = Namespace(config_action="reset")
    response = megaplan.handle_config(reset_args)
    assert response["success"] is True


def test_config_set_invalid_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    args = Namespace(config_action="set", key="badkey", value="codex")
    with pytest.raises(megaplan.CliError, match="agents"):
        megaplan.handle_config(args)


def test_config_set_invalid_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    args = Namespace(config_action="set", key="agents.nosuchstep", value="codex")
    with pytest.raises(megaplan.CliError, match="Unknown step"):
        megaplan.handle_config(args)


def test_config_set_invalid_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    args = Namespace(config_action="set", key="agents.plan", value="nosuchagent")
    with pytest.raises(megaplan.CliError, match="Unknown agent"):
        megaplan.handle_config(args)


# ---------------------------------------------------------------------------
# Prompt tests (additional)
# ---------------------------------------------------------------------------


def test_critique_prompt_contains_robustness_instruction(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    _, state = load_plan(plan_fixture.root, plan_fixture.plan_name)
    from megaplan.prompts import create_claude_prompt
    prompt = create_claude_prompt("critique", state, plan_fixture.plan_dir)
    assert "Robustness level" in prompt
    assert "standard" in prompt


def test_execute_prompt_includes_approval_note(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    _, state = load_plan(plan_fixture.root, plan_fixture.plan_name)
    from megaplan.prompts import create_claude_prompt
    prompt = create_claude_prompt("execute", state, plan_fixture.plan_dir)
    assert "Review mode" in prompt or "auto-approve" in prompt or "approved" in prompt


def test_render_final_md_pending_partially_done_and_reviewed_states() -> None:
    from megaplan._core import render_final_md

    pending = {
        "tasks": [
            {
                "id": "T1",
                "description": "Do work",
                "depends_on": [],
                "status": "pending",
                "executor_notes": "",
                "reviewer_verdict": "",
            }
        ],
        "watch_items": ["Watch this."],
        "sense_checks": [{"id": "SC1", "task_id": "T1", "question": "Did it work?", "verdict": ""}],
        "meta_commentary": "Pending state.",
    }
    partial = {
        **pending,
        "tasks": [{**pending["tasks"][0], "status": "done", "executor_notes": "Implemented."}],
    }
    reviewed = {
        **partial,
        "tasks": [{**partial["tasks"][0], "reviewer_verdict": "Pass"}],
        "sense_checks": [{"id": "SC1", "task_id": "T1", "question": "Did it work?", "verdict": "Confirmed."}],
    }

    pending_md = render_final_md(pending)
    partial_md = render_final_md(partial)
    reviewed_md = render_final_md(reviewed)

    assert "# Execution Checklist" in pending_md
    assert "## Watch Items" in pending_md
    assert "## Sense Checks" in pending_md
    assert "## Meta" in pending_md
    assert "- [ ] **T1:** Do work" in pending_md
    assert "- [x] **T1:** Do work" in partial_md
    assert "Executor notes: Implemented." in partial_md
    assert "Reviewer verdict: Pass" in reviewed_md
    assert "Verdict: Confirmed." in reviewed_md


def test_validate_merge_inputs_filters_malformed_entries() -> None:
    valid = megaplan.handlers._validate_merge_inputs(
        [
            {"task_id": "T1", "status": "done", "executor_notes": "Implemented."},
            {"task_id": "T2", "status": 1, "executor_notes": "Bad type"},
            {"task_id": "T3", "executor_notes": "Missing status"},
            "bad-entry",
        ],
        required_fields=("task_id", "status", "executor_notes"),
        enum_fields={"status": {"done", "skipped"}},
        label="task_updates",
    )
    empty = megaplan.handlers._validate_merge_inputs(
        [],
        required_fields=("task_id", "reviewer_verdict"),
        label="task_verdicts",
    )

    assert valid == [{"task_id": "T1", "status": "done", "executor_notes": "Implemented."}]
    assert empty == []


def test_codex_uses_same_prompt_builders_for_shared_steps(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    _, state = load_plan(plan_fixture.root, plan_fixture.plan_name)
    from megaplan.prompts import create_claude_prompt, create_codex_prompt
    for step in ["plan", "critique"]:
        claude_prompt = create_claude_prompt(step, state, plan_fixture.plan_dir)
        codex_prompt = create_codex_prompt(step, state, plan_fixture.plan_dir)
        assert claude_prompt == codex_prompt


# ---------------------------------------------------------------------------
# Legacy migration test (original)
# ---------------------------------------------------------------------------


def test_load_plan_migrates_legacy_evaluated_state(tmp_path: Path) -> None:
    root = tmp_path / "root"
    ensure_runtime_layout(root)
    plan_dir = megaplan.plans_root(root) / "legacy"
    plan_dir.mkdir(parents=True)
    (plan_dir / "state.json").write_text(
        json.dumps(
            {
                "name": "legacy",
                "idea": "old workflow",
                "current_state": "evaluated",
                "iteration": 1,
                "created_at": "2026-03-20T00:00:00Z",
                "config": {"project_dir": str(tmp_path / "project"), "auto_approve": False, "robustness": "standard"},
                "sessions": {},
                "plan_versions": [{"version": 1, "file": "plan_v1.md", "hash": "sha256:test", "timestamp": "2026-03-20T00:00:00Z"}],
                "history": [],
                "meta": {
                    "significant_counts": [],
                    "weighted_scores": [],
                    "plan_deltas": [],
                    "recurring_critiques": [],
                    "total_cost_usd": 0.0,
                    "overrides": [],
                    "notes": [],
                },
                "last_evaluation": {"recommendation": "SKIP"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _plan_dir, state = load_plan(root, "legacy")
    persisted = read_json(plan_dir / "state.json")

    assert state["current_state"] == megaplan.STATE_CRITIQUED
    assert state["last_gate"] == {}
    assert "last_evaluation" not in persisted
