from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import megaplan
import megaplan.cli
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
        megaplan.cli.shutil,
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
        megaplan.cli.shutil,
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


def test_mock_workflow_end_to_end(plan_fixture: PlanFixture) -> None:
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
    execute = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    review = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    assert plan["state"] == megaplan.STATE_PLANNED
    assert critique1["state"] == megaplan.STATE_CRITIQUED
    assert gate1["recommendation"] == "ITERATE"
    assert revise["state"] == megaplan.STATE_PLANNED
    assert critique2["iteration"] == 2
    assert gate2["state"] == megaplan.STATE_GATED
    assert gate2["recommendation"] == "PROCEED"
    assert execute["state"] == megaplan.STATE_EXECUTED
    assert review["state"] == megaplan.STATE_DONE
    assert (plan_fixture.project_dir / "IMPLEMENTED_BY_MEGAPLAN.txt").exists()


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
    assert gate["override_forced"] is True
    assert gate["recommendation"] == "PROCEED"
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

    monkeypatch.setattr(megaplan.cli, "run_step_with_worker", lambda *args, **kwargs: next(results))

    megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="extra context"),
    )
    megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)

    assert state["meta"]["weighted_scores"] == [3.5]


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
