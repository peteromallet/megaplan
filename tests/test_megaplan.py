from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import megaplan
import megaplan.cli
import megaplan.execution
import megaplan.evaluation
import megaplan.handlers
import megaplan.cli
import megaplan._core
import megaplan._core.io as io_module
import megaplan.workers
from megaplan.evaluation import PLAN_STRUCTURE_REQUIRED_STEP_ISSUE, validate_plan_structure
from megaplan._core import WORKFLOW, _ROBUSTNESS_OVERRIDES, ensure_runtime_layout, load_plan, workflow_next
from megaplan.prompts import create_claude_prompt
from megaplan.types import STATE_PREPPED
from megaplan.workers import WorkerResult, _build_mock_payload


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_main_json(
    argv: list[str],
    *,
    cwd: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, dict]:
    monkeypatch.chdir(cwd)
    exit_code = megaplan.main(argv)
    return exit_code, json.loads(capsys.readouterr().out)


def _write_lines(path: Path, count: int, *, prefix: str = "line") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{prefix}_{index}" for index in range(count)) + "\n", encoding="utf-8")


def make_args_factory(project_dir: Path) -> Callable[..., Namespace]:
    def make_args(**overrides: object) -> Namespace:
        data = {
            "plan": None,
            "idea": "test idea",
            "name": "test-plan",
            "project_dir": str(project_dir),
            "auto_approve": None,
            "robustness": None,
            "agent": None,
            "ephemeral": False,
            "fresh": False,
            "persist": False,
            "confirm_destructive": True,
            "user_approved": False,
            "confirm_self_review": False,
            "batch": None,
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


def _make_plan_fixture_with_robustness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    robustness: str,
) -> PlanFixture:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    config_path = tmp_path / "config"
    root.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    def _config_dir(home: Path | None = None) -> Path:
        del home
        return config_path

    monkeypatch.setenv(megaplan.MOCK_ENV_VAR, "1")
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )
    monkeypatch.setattr(io_module, "config_dir", _config_dir)
    monkeypatch.setattr(megaplan.cli, "config_dir", _config_dir)

    make_args = make_args_factory(project_dir)
    response = megaplan.handle_init(root, make_args(robustness=robustness))
    plan_name = response["plan"]
    return PlanFixture(
        root=root,
        project_dir=project_dir,
        plan_name=plan_name,
        plan_dir=megaplan.plans_root(root) / plan_name,
        make_args=make_args,
    )


@pytest.fixture
def plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlanFixture:
    return _make_plan_fixture_with_robustness(tmp_path, monkeypatch, robustness="standard")


def load_state(plan_dir: Path) -> dict:
    return read_json(plan_dir / "state.json")


def latest_plan_name(plan_dir: Path) -> str:
    return load_state(plan_dir)["plan_versions"][-1]["file"]


def debt_registry_path(root: Path) -> Path:
    return root / ".megaplan" / "debt.json"


def first_open_significant_flag(plan_dir: Path) -> dict:
    registry = read_json(plan_dir / "faults.json")
    return next(
        flag
        for flag in registry["flags"]
        if flag["status"] in {"open", "disputed"} and flag.get("severity") == "significant"
    )


def test_init_sets_last_gate_and_next_step_plan(plan_fixture: PlanFixture) -> None:
    state = load_state(plan_fixture.plan_dir)
    assert state["current_state"] == megaplan.STATE_INITIALIZED
    assert state["last_gate"] == {}
    assert state["iteration"] == 0


def test_init_response_points_to_next_step_by_robustness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )
    make_args = make_args_factory(project_dir)
    standard = megaplan.handle_init(root, make_args(name="standard-plan", robustness="standard"))
    light = megaplan.handle_init(root, make_args(name="light-plan", robustness="light"))
    heavy = megaplan.handle_init(root, make_args(name="heavy-plan", robustness="heavy"))
    assert standard["next_step"] == "plan"
    assert light["next_step"] == "plan"
    assert heavy["next_step"] == "prep"


_LEGACY_STATE_MACHINE_CASES = [
    ({"current_state": megaplan.STATE_INITIALIZED, "last_gate": {}}, ["plan"]),
    ({"current_state": STATE_PREPPED, "last_gate": {}}, ["plan"]),
    ({"current_state": megaplan.STATE_PLANNED, "last_gate": {}}, ["critique", "plan", "step"]),
    ({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {}}, ["gate", "step"]),
    ({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "ITERATE"}}, ["revise", "step"]),
    (
        {"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "ESCALATE"}},
        ["override add-note", "override force-proceed", "override abort", "step"],
    ),
    (
        {"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "PROCEED", "passed": False}},
        ["revise", "override force-proceed", "step"],
    ),
    ({"current_state": megaplan.STATE_GATED, "last_gate": {}}, ["finalize", "override replan", "step"]),
    ({"current_state": megaplan.STATE_FINALIZED, "last_gate": {}}, ["execute", "override replan", "step"]),
]


def test_infer_next_steps_matches_new_state_machine() -> None:
    for state, expected in _LEGACY_STATE_MACHINE_CASES:
        assert megaplan.infer_next_steps(state) == expected


def test_workflow_next_matches_legacy_partial_state_cases() -> None:
    for state, expected in _LEGACY_STATE_MACHINE_CASES:
        assert workflow_next(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"current_state": megaplan.STATE_CRITIQUED, "last_gate": {}, "config": {"robustness": "light"}},
            ["revise", "step"],
        ),
        (
            {
                "current_state": megaplan.STATE_CRITIQUED,
                "last_gate": {"recommendation": "ESCALATE"},
                "config": {"robustness": "light"},
            },
            ["revise", "step"],
        ),
        (
            {"current_state": megaplan.STATE_EXECUTED, "last_gate": {}, "config": {"robustness": "light"}},
            [],
        ),
    ],
)
def test_workflow_next_light_robustness_overrides(state: dict[str, object], expected: list[str]) -> None:
    assert workflow_next(state) == expected


def test_workflow_definition_is_complete_for_standard_flow() -> None:
    expected_states = {
        megaplan.STATE_INITIALIZED,
        megaplan.STATE_PLANNED,
        megaplan.STATE_CRITIQUED,
        megaplan.STATE_GATED,
        megaplan.STATE_FINALIZED,
        megaplan.STATE_EXECUTED,
    }

    assert expected_states.issubset(WORKFLOW)
    for state_name, transitions in WORKFLOW.items():
        assert transitions or state_name in megaplan.TERMINAL_STATES
    for robustness, overrides in _ROBUSTNESS_OVERRIDES.items():
        assert robustness in megaplan._core.ROBUSTNESS_LEVELS
        assert set(overrides).issubset(WORKFLOW)


def test_workflow_walk_matches_documented_standard_flow() -> None:
    walk = [
        ({"current_state": megaplan.STATE_INITIALIZED, "last_gate": {}}, "plan"),
        ({"current_state": megaplan.STATE_PLANNED, "last_gate": {}}, "critique"),
        ({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {}}, "gate"),
        (
            {"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "ITERATE"}},
            "revise",
        ),
        ({"current_state": megaplan.STATE_PLANNED, "last_gate": {}}, "critique"),
        (
            {
                "current_state": megaplan.STATE_CRITIQUED,
                "last_gate": {"recommendation": "PROCEED", "passed": True},
            },
            "gate",
        ),
        ({"current_state": megaplan.STATE_GATED, "last_gate": {}}, "finalize"),
        ({"current_state": megaplan.STATE_FINALIZED, "last_gate": {}}, "execute"),
        ({"current_state": megaplan.STATE_EXECUTED, "last_gate": {}}, "review"),
    ]

    actual_steps: list[str] = []
    for state, expected_step in walk:
        assert expected_step in workflow_next(state)
        actual_steps.append(expected_step)

    assert actual_steps == [
        "plan",
        "critique",
        "gate",
        "revise",
        "critique",
        "gate",
        "finalize",
        "execute",
        "review",
    ]


def test_workflow_walk_matches_documented_robust_flow() -> None:
    robust_config = {"config": {"robustness": "heavy"}}
    walk = [
        ({"current_state": megaplan.STATE_INITIALIZED, "last_gate": {}, **robust_config}, "prep"),
        ({"current_state": STATE_PREPPED, "last_gate": {}, **robust_config}, "plan"),
        ({"current_state": megaplan.STATE_PLANNED, "last_gate": {}, **robust_config}, "critique"),
        ({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {}, **robust_config}, "gate"),
        (
            {"current_state": megaplan.STATE_CRITIQUED, "last_gate": {"recommendation": "ITERATE"}, **robust_config},
            "revise",
        ),
        ({"current_state": megaplan.STATE_PLANNED, "last_gate": {}, **robust_config}, "critique"),
        (
            {
                "current_state": megaplan.STATE_CRITIQUED,
                "last_gate": {"recommendation": "PROCEED", "passed": True},
                **robust_config,
            },
            "gate",
        ),
        ({"current_state": megaplan.STATE_GATED, "last_gate": {}, **robust_config}, "finalize"),
        ({"current_state": megaplan.STATE_FINALIZED, "last_gate": {}, **robust_config}, "execute"),
        ({"current_state": megaplan.STATE_EXECUTED, "last_gate": {}, **robust_config}, "review"),
    ]

    actual_steps: list[str] = []
    for state, expected_step in walk:
        assert expected_step in workflow_next(state)
        actual_steps.append(expected_step)

    assert actual_steps == [
        "prep",
        "plan",
        "critique",
        "gate",
        "revise",
        "critique",
        "gate",
        "finalize",
        "execute",
        "review",
    ]


def test_workflow_walk_matches_documented_light_flow() -> None:
    light_config = {"config": {"robustness": "light"}}
    walk = [
        ({"current_state": megaplan.STATE_INITIALIZED, "last_gate": {}, **light_config}, "plan"),
        ({"current_state": megaplan.STATE_PLANNED, "last_gate": {}, **light_config}, "critique"),
        ({"current_state": megaplan.STATE_CRITIQUED, "last_gate": {}, **light_config}, "revise"),
        ({"current_state": megaplan.STATE_GATED, "last_gate": {}, **light_config}, "finalize"),
        ({"current_state": megaplan.STATE_FINALIZED, "last_gate": {}, **light_config}, "execute"),
    ]

    actual_steps: list[str] = []
    for state, expected_step in walk:
        assert expected_step in workflow_next(state)
        actual_steps.append(expected_step)

    assert workflow_next({"current_state": megaplan.STATE_EXECUTED, "last_gate": {}, **light_config}) == []
    assert actual_steps == ["plan", "critique", "revise", "finalize", "execute"]


def test_all_robustness_levels_route_planned_to_critique() -> None:
    """All robustness levels go directly from planned to critique."""
    for level in ("tiny", "light", "standard", "heavy"):
        state = {"current_state": megaplan.STATE_PLANNED, "last_gate": {}, "config": {"robustness": level}}
        next_steps = workflow_next(state)
        assert "critique" in next_steps, f"{level} should offer critique"
        assert "research" not in next_steps, f"{level} should not offer research"


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



def test_workflow_mock_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from megaplan.handlers import handle_prep

    plan_fixture = _make_plan_fixture_with_robustness(tmp_path, monkeypatch, robustness="heavy")
    make_args = plan_fixture.make_args
    recorded_steps: list[str] = []
    original_run_step = megaplan.workers.run_step_with_worker

    def _record(step: str, *args: object, **kwargs: object) -> tuple[WorkerResult, str, str, bool]:
        recorded_steps.append(step)
        return original_run_step(step, *args, **kwargs)

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", _record)
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="add-note", note="keep changes scoped"),
    )
    prep = handle_prep(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
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

    assert prep["state"] == STATE_PREPPED
    assert prep["next_step"] == "plan"
    assert plan["state"] == megaplan.STATE_PLANNED
    assert plan["next_step"] == "critique"
    assert critique1["state"] == megaplan.STATE_CRITIQUED
    assert gate1["recommendation"] == "ITERATE"
    assert revise["state"] == megaplan.STATE_PLANNED
    assert revise["next_step"] == "critique"
    assert critique2["iteration"] == 2
    assert gate2["state"] == megaplan.STATE_GATED
    assert gate2["recommendation"] == "PROCEED"
    assert finalize["state"] == megaplan.STATE_FINALIZED
    assert plan_meta["structure_warnings"] == []
    assert revise_meta["structure_warnings"] == []
    assert (plan_fixture.plan_dir / "final.md").exists()
    assert (plan_fixture.plan_dir / "finalize.json").exists()
    assert (plan_fixture.plan_dir / "prep.json").exists()
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
    assert recorded_steps == [
        "prep",
        "plan",
        "critique",
        "gate",
        "revise",
        "critique",
        "gate",
        "finalize",
        "execute",
        "review",
    ]
    assert (plan_fixture.project_dir / "IMPLEMENTED_BY_MEGAPLAN.txt").exists()


def test_workflow_light_robustness_single_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_fixture = _make_plan_fixture_with_robustness(tmp_path, monkeypatch, robustness="light")
    make_args = plan_fixture.make_args
    recorded_steps: list[str] = []
    original_run_step = megaplan.workers.run_step_with_worker

    def _record(step: str, *args: object, **kwargs: object) -> tuple[WorkerResult, str, str, bool]:
        recorded_steps.append(step)
        return original_run_step(step, *args, **kwargs)

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", _record)

    plan = megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    assert plan["next_step"] == "critique"

    critique = megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    assert critique["next_step"] == "revise"

    revise = megaplan.handle_revise(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    assert revise["next_step"] == "finalize"
    assert revise["state"] == megaplan.STATE_GATED

    finalize = megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    execute = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    state = load_state(plan_fixture.plan_dir)
    stored_review = read_json(plan_fixture.plan_dir / "review.json")

    assert finalize["state"] == megaplan.STATE_FINALIZED
    assert execute["state"] == megaplan.STATE_DONE
    assert execute["next_step"] is None
    assert "review.json" in execute["artifacts"]
    assert stored_review["review_verdict"] == "approved"
    assert recorded_steps == ["plan", "critique", "revise", "finalize", "execute"]
    assert [entry["step"] for entry in state["history"]] == [
        "init",
        "plan",
        "critique",
        "revise",
        "finalize",
        "execute",
    ]


def test_cli_registers_prep_command() -> None:
    parser = megaplan.cli.build_parser()
    parsed = parser.parse_args(["prep", "--plan", "demo"])

    assert parsed.command == "prep"
    assert parsed.plan == "demo"
    from megaplan.handlers import handle_prep

    assert megaplan.cli.COMMAND_HANDLERS["prep"] is handle_prep


def test_light_critique_routes_to_revise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_fixture = _make_plan_fixture_with_robustness(tmp_path, monkeypatch, robustness="light")
    make_args = plan_fixture.make_args

    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)

    assert response["next_step"] == "revise"
    assert state["last_gate"]["recommendation"] == "ITERATE"


def test_light_revise_routes_to_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_fixture = _make_plan_fixture_with_robustness(tmp_path, monkeypatch, robustness="light")
    make_args = plan_fixture.make_args

    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_revise(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)

    assert response["next_step"] == "finalize"
    assert state["current_state"] == megaplan.STATE_GATED


def test_standard_revise_routes_to_critique_and_clears_last_gate(plan_fixture: PlanFixture) -> None:
    make_args = plan_fixture.make_args

    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    gate = megaplan.handle_gate(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_revise(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)

    assert gate["recommendation"] == "ITERATE"
    assert response["state"] == megaplan.STATE_PLANNED
    assert response["next_step"] == "critique"
    assert state["current_state"] == megaplan.STATE_PLANNED
    assert state["last_gate"] == {}


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
            "success_criteria": [{"criterion": "warn but continue", "priority": "must"}],
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
            "success_criteria": [{"criterion": "should fail", "priority": "must"}],
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


def test_handle_critique_rejects_invalid_check_payload(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    worker = WorkerResult(
        payload=_build_mock_payload(
            "critique",
            load_state(plan_fixture.plan_dir),
            plan_fixture.plan_dir,
            checks=[],
        ),
        raw_output="invalid critique payload",
        duration_ms=1,
        cost_usd=0.0,
        session_id="critique-invalid",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )
    monkeypatch.setattr(
        megaplan.handlers,
        "validate_critique_checks",
        lambda payload, **kwargs: ["correctness"],
    )

    with pytest.raises(megaplan.CliError, match="Critique output failed check validation"):
        megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    assert state["history"][-1]["result"] == "error"
    assert not (plan_fixture.plan_dir / "critique_v1.json").exists()


def test_handle_critique_accepts_validated_checks(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        megaplan.handlers,
        "validate_critique_checks",
        lambda payload, **kwargs: [],
    )

    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    response = megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    assert response["success"] is True
    assert (plan_fixture.plan_dir / "critique_v1.json").exists()


def test_handle_finalize_validates_payload_shape(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )

    valid_payload = {
        "tasks": [
            {
                "id": "T1",
                "description": "Ship the change",
                "depends_on": [],
                "status": "pending",
                "executor_notes": "",
                "files_changed": [],
                "commands_run": [],
                "evidence_files": [],
                "reviewer_verdict": "",
            }
        ],
        "watch_items": [],
        "sense_checks": [
            {
                "id": "SC1",
                "task_id": "T1",
                "question": "Did it work?",
                "executor_note": "",
                "verdict": "",
            }
        ],
        "meta_commentary": "ok",
    }
    worker = WorkerResult(
        payload=valid_payload,
        raw_output="valid finalize payload",
        duration_ms=1,
        cost_usd=0.0,
        session_id="finalize-valid",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )

    response = megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    assert response["success"] is True
    assert response["state"] == megaplan.STATE_FINALIZED
    assert read_json(plan_fixture.plan_dir / "finalize.json")["tasks"][0]["status"] == "pending"


def test_handle_finalize_rejects_invalid_payload(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )

    invalid_worker = WorkerResult(
        payload={
            "tasks": [
                {
                    "id": "T1",
                    "description": "Broken finalize task",
                    "depends_on": [],
                    "status": "done",
                    "executor_notes": "",
                    "files_changed": [],
                    "commands_run": [],
                    "evidence_files": [],
                    "reviewer_verdict": "",
                }
            ],
            "watch_items": [],
            "sense_checks": [],
        },
        raw_output="invalid finalize payload",
        duration_ms=1,
        cost_usd=0.0,
        session_id="finalize-invalid",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (invalid_worker, "claude", "persistent", False),
    )

    with pytest.raises(megaplan.CliError, match="status `pending`"):
        megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    assert state["history"][-1]["result"] == "error"


def test_finalize_snapshot_remains_pending_after_execute(plan_fixture: PlanFixture) -> None:
    from megaplan._core import load_finalize_snapshot

    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )

    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    snapshot_before_execute = load_finalize_snapshot(plan_fixture.plan_dir)
    assert (plan_fixture.plan_dir / "finalize_snapshot.json").exists()
    assert all(task["status"] == "pending" for task in snapshot_before_execute["tasks"])

    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    finalize_after_execute = read_json(plan_fixture.plan_dir / "finalize.json")
    snapshot_after_execute = load_finalize_snapshot(plan_fixture.plan_dir)

    assert all(task["status"] == "done" for task in finalize_after_execute["tasks"])
    assert snapshot_after_execute == snapshot_before_execute
    assert all(task["status"] == "pending" for task in snapshot_after_execute["tasks"])


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


def test_force_proceed_registers_unresolved_flags_as_debt(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    response = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(
            plan=plan_fixture.plan_name,
            override_action="force-proceed",
            reason="executor can resolve remaining issues",
        ),
    )
    registry = read_json(debt_registry_path(plan_fixture.root))

    assert response["debt_entries_added"] >= 1
    assert len(registry["entries"]) >= 1
    assert all(entry["resolved"] is False for entry in registry["entries"])


def test_repeated_force_proceed_increments_existing_debt_instead_of_duplicating(plan_fixture: PlanFixture) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="first pass"),
    )
    first_registry = read_json(debt_registry_path(plan_fixture.root))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="replan", reason="loop back"),
    )
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="second pass"),
    )

    registry = read_json(debt_registry_path(plan_fixture.root))

    assert len(registry["entries"]) == len(first_registry["entries"])
    assert all(entry["occurrence_count"] == 2 for entry in registry["entries"])


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


def test_gate_proceed_with_accepted_tradeoffs_creates_debt(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    flag = first_open_significant_flag(plan_fixture.plan_dir)
    worker = WorkerResult(
        payload={
            "recommendation": "PROCEED",
            "rationale": "Known tradeoff accepted.",
            "signals_assessment": "Proceeding with one accepted limitation.",
            "warnings": [],
            "accepted_tradeoffs": [
                {
                    "flag_id": flag["id"],
                    "subsystem": "timeout-recovery",
                    "concern": flag["concern"],
                    "rationale": "Tracked as debt for a later redesign.",
                }
            ],
        },
        raw_output="{}",
        duration_ms=1,
        cost_usd=0.0,
        session_id="gate-debt-1",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )

    response = megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    registry = read_json(debt_registry_path(plan_fixture.root))

    assert response["debt_entries_added"] == 1
    assert len(registry["entries"]) == 1
    assert registry["entries"][0]["flag_ids"] == [flag["id"]]


def test_gate_iterate_with_empty_accepted_tradeoffs_creates_no_debt(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    worker = WorkerResult(
        payload={
            "recommendation": "ITERATE",
            "rationale": "Still needs plan work.",
            "signals_assessment": "Revisions are still needed.",
            "warnings": [],
            "accepted_tradeoffs": [],
        },
        raw_output="{}",
        duration_ms=1,
        cost_usd=0.0,
        session_id="gate-debt-2",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )

    response = megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    assert response["debt_entries_added"] == 0
    assert not debt_registry_path(plan_fixture.root).exists()


def test_gate_proceed_without_accepted_tradeoffs_registers_fallback_debt(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    flag = first_open_significant_flag(plan_fixture.plan_dir)
    worker = WorkerResult(
        payload={
            "recommendation": "PROCEED",
            "rationale": "Proceed despite the remaining planning-level debt.",
            "signals_assessment": "Proceeding with unresolved flags recorded as debt.",
            "warnings": [],
        },
        raw_output="{}",
        duration_ms=1,
        cost_usd=0.0,
        session_id="gate-debt-3",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )

    response = megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    registry = read_json(debt_registry_path(plan_fixture.root))

    assert response["debt_entries_added"] >= 1
    assert len(registry["entries"]) == response["debt_entries_added"]
    assert any(entry["flag_ids"] == [flag["id"]] for entry in registry["entries"])


def test_step_add(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    response = megaplan.handle_step(
        plan_fixture.root,
        plan_fixture.make_args(
            plan=plan_fixture.plan_name,
            step_action="add",
            after="S2",
            description="Add parser edge-case coverage",
        ),
    )
    state = load_state(plan_fixture.plan_dir)
    plan_name = latest_plan_name(plan_fixture.plan_dir)
    plan_text = (plan_fixture.plan_dir / plan_name).read_text(encoding="utf-8")
    latest_meta = read_json(plan_fixture.plan_dir / plan_name.replace(".md", ".meta.json"))
    previous_meta = read_json(plan_fixture.plan_dir / "plan_v1.meta.json")

    assert response["state"] == megaplan.STATE_PLANNED
    assert state["iteration"] == 1
    assert plan_name == "plan_v1a.md"
    assert state["last_gate"] == {}
    assert "## Step 3: Add parser edge-case coverage" in plan_text
    assert "## Step 4: Verify the behavior" in plan_text
    assert latest_meta["questions"] == previous_meta["questions"]
    assert latest_meta["success_criteria"] == previous_meta["success_criteria"]
    assert latest_meta["assumptions"] == previous_meta["assumptions"]
    assert latest_meta["step_edit"]["action"] == "add"


def test_step_add_scaffold_passes_validation(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_step(
        plan_fixture.root,
        plan_fixture.make_args(
            plan=plan_fixture.plan_name,
            step_action="add",
            after="S1",
            description="Document the handler change",
        ),
    )

    plan_name = latest_plan_name(plan_fixture.plan_dir)
    plan_text = (plan_fixture.plan_dir / plan_name).read_text(encoding="utf-8")

    assert validate_plan_structure(plan_text) == []
    assert "1. **TODO** Fill in implementation details (`path/to/file`)." in plan_text


def test_step_remove(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    response = megaplan.handle_step(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, step_action="remove", step_id="S2"),
    )
    state = load_state(plan_fixture.plan_dir)
    plan_name = latest_plan_name(plan_fixture.plan_dir)
    plan_text = (plan_fixture.plan_dir / plan_name).read_text(encoding="utf-8")

    assert response["state"] == megaplan.STATE_PLANNED
    assert plan_name == "plan_v1a.md"
    assert state["iteration"] == 1
    assert "## Step 2: Verify the behavior" in plan_text
    assert "Implement the smallest viable change" not in plan_text


def test_step_move(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    response = megaplan.handle_step(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, step_action="move", step_id="S3", after="S1"),
    )
    plan_name = latest_plan_name(plan_fixture.plan_dir)
    plan_text = (plan_fixture.plan_dir / plan_name).read_text(encoding="utf-8")
    step_two_index = plan_text.index("## Step 2: Verify the behavior")
    step_three_index = plan_text.index("## Step 3: Implement the smallest viable change")

    assert response["state"] == megaplan.STATE_PLANNED
    assert plan_name == "plan_v1a.md"
    assert step_two_index < step_three_index


def test_step_remove_last_step_rejected(plan_fixture: PlanFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = WorkerResult(
        payload={
            "plan": """# Implementation Plan: Single Step

## Overview
Keep the plan small.

## Step 1: Only step (`megaplan/handlers.py`)
1. **Implement** the change (`megaplan/handlers.py:1`).

## Validation Order
1. Run a focused test.
""",
            "questions": ["q"],
            "success_criteria": [{"criterion": "c", "priority": "must"}],
            "assumptions": ["a"],
        },
        raw_output="single-step plan",
        duration_ms=1,
        cost_usd=0.0,
        session_id="plan-single-step",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "claude", "persistent", False),
    )
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    with pytest.raises(megaplan.CliError, match="last remaining step"):
        megaplan.handle_step(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, step_action="remove", step_id="S1"),
        )


@pytest.mark.parametrize("state_name", [megaplan.STATE_DONE, megaplan.STATE_ABORTED])
def test_step_invalid_state(plan_fixture: PlanFixture, state_name: str) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    state["current_state"] = state_name
    (plan_fixture.plan_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(megaplan.CliError, match="Cannot run 'step'"):
        megaplan.handle_step(
            plan_fixture.root,
            plan_fixture.make_args(
                plan=plan_fixture.plan_name,
                step_action="add",
                after="S1",
                description="Should fail",
            ),
        )


def test_step_preserves_meta(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    meta_path = plan_fixture.plan_dir / "plan_v1.meta.json"
    meta = read_json(meta_path)
    meta["questions"] = ["What should happen next?"]
    meta["success_criteria"] = [{"criterion": "Ship the step editor.", "priority": "must"}]
    meta["assumptions"] = ["The existing plan file is valid."]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    megaplan.handle_step(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, step_action="move", step_id="S3", after="S1"),
    )

    plan_name = latest_plan_name(plan_fixture.plan_dir)
    new_meta = read_json(plan_fixture.plan_dir / plan_name.replace(".md", ".meta.json"))

    assert new_meta["questions"] == ["What should happen next?"]
    assert new_meta["success_criteria"] == [{"criterion": "Ship the step editor.", "priority": "must"}]
    assert new_meta["assumptions"] == ["The existing plan file is valid."]


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

    assert len(state["meta"]["weighted_scores"]) == 1


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
    from megaplan.flags import update_flags_after_critique
    registry = update_flags_after_critique(plan_fixture.plan_dir, critique_payload, iteration=1)
    assert len(registry["flags"]) >= 1
    flag = next(f for f in registry["flags"] if f["id"] == "FLAG-001")
    assert flag["status"] == "open"
    assert flag["severity"] == "significant"


def test_update_flags_after_critique_verifies_flags(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    from megaplan.flags import update_flags_after_critique
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
    from megaplan.flags import update_flags_after_critique
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
    from megaplan.flags import update_flags_after_critique
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
    from megaplan.flags import update_flags_after_critique
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
    from megaplan.flags import update_flags_after_critique
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
    from megaplan.flags import update_flags_after_critique, update_flags_after_revise  # noqa: F811
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
        {"id": "FLAG-006", "severity": "significant", "status": "accepted_tradeoff"},
        {"id": "FLAG-007", "severity": "significant", "status": "gate_disputed"},
    ]}
    unresolved = megaplan.unresolved_significant_flags(registry)
    ids = [f["id"] for f in unresolved]
    assert "FLAG-001" in ids
    assert "FLAG-004" in ids
    assert "FLAG-005" in ids  # "addressed" is now a blocking status
    assert "FLAG-002" not in ids
    assert "FLAG-003" not in ids
    assert "FLAG-006" not in ids  # accepted_tradeoff is a resolved state
    assert "FLAG-007" not in ids  # gate_disputed is a resolved state


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
    # Single-batch path should also write execution_batch_1.json
    assert (plan_fixture.plan_dir / "execution_batch_1.json").exists()


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


def test_step_command_help_and_parser_shape(capsys: pytest.CaptureFixture[str]) -> None:
    parser = megaplan.cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["step", "--help"])

    help_text = capsys.readouterr().out
    parsed = parser.parse_args(["step", "add", "--plan", "demo", "--after", "S3", "Add docs"])

    assert "add" in help_text
    assert "remove" in help_text
    assert "move" in help_text
    assert parsed.command == "step"
    assert parsed.step_action == "add"
    assert parsed.plan == "demo"
    assert parsed.after == "S3"
    assert parsed.description == "Add docs"
    assert megaplan.cli.COMMAND_HANDLERS["step"] is megaplan.handle_step


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


def test_strict_schema_overwrites_partial_required_arrays_recursively() -> None:
    from megaplan.schemas import strict_schema
    schema = {
        "type": "object",
        "required": ["stale_root"],
        "properties": {
            "inner": {
                "type": "object",
                "required": ["stale_inner"],
                "properties": {"child": {"type": "string"}},
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["stale_item"],
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    }
    result = strict_schema(schema)
    assert result["required"] == ["inner", "items"]
    assert result["properties"]["inner"]["required"] == ["child"]
    assert result["properties"]["items"]["items"]["required"] == ["name"]


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
    assert result["properties"]["inner"]["required"] == ["a"]


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
    assert result["properties"]["items"]["items"]["required"] == ["name"]


def test_strict_schema_non_object_untouched() -> None:
    from megaplan.schemas import strict_schema
    assert strict_schema({"type": "string"}) == {"type": "string"}
    assert strict_schema(42) == 42
    assert strict_schema("hello") == "hello"
    assert strict_schema([1, 2]) == [1, 2]


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


def test_debt_list_on_empty_registry_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    root.mkdir()

    exit_code, payload = run_main_json(["debt", "list"], cwd=root, capsys=capsys, monkeypatch=monkeypatch)

    assert exit_code == 0
    assert payload["success"] is True
    assert payload["step"] == "debt"
    assert payload["action"] == "list"
    assert payload["details"]["entries"] == []
    assert payload["details"]["by_subsystem"] == []


def test_debt_add_and_list_increment_matching_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    root.mkdir()

    exit_code, add_one = run_main_json(
        [
            "debt",
            "add",
            "--subsystem",
            "timeout-recovery",
            "--concern",
            "Timeout recovery: Retry backoff is missing",
            "--flag-ids",
            "FLAG-001",
            "--plan",
            "plan-a",
        ],
        cwd=root,
        capsys=capsys,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    assert add_one["details"]["entry"]["id"] == "DEBT-001"

    exit_code, _add_two = run_main_json(
        [
            "debt",
            "add",
            "--subsystem",
            "timeout-recovery",
            "--concern",
            "Timeout recovery: retry backoff is missing",
            "--flag-ids",
            "FLAG-002",
            "--plan",
            "plan-b",
        ],
        cwd=root,
        capsys=capsys,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0

    exit_code, payload = run_main_json(["debt", "list"], cwd=root, capsys=capsys, monkeypatch=monkeypatch)

    assert exit_code == 0
    assert len(payload["details"]["entries"]) == 1
    entry = payload["details"]["entries"][0]
    assert entry["occurrence_count"] == 2
    assert entry["flag_ids"] == ["FLAG-001", "FLAG-002"]
    assert entry["plan_ids"] == ["plan-a", "plan-b"]
    assert payload["details"]["by_subsystem"][0]["subsystem"] == "timeout-recovery"


def test_debt_resolve_hides_entry_from_default_list_but_not_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    root.mkdir()

    _exit_code, add_payload = run_main_json(
        [
            "debt",
            "add",
            "--subsystem",
            "observation",
            "--concern",
            "Observation: Missing event logging",
            "--flag-ids",
            "FLAG-010",
            "--plan",
            "plan-a",
        ],
        cwd=root,
        capsys=capsys,
        monkeypatch=monkeypatch,
    )
    debt_id = add_payload["details"]["entry"]["id"]

    exit_code, resolve_payload = run_main_json(
        ["debt", "resolve", debt_id, "--plan", "plan-b"],
        cwd=root,
        capsys=capsys,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    assert resolve_payload["details"]["entry"]["resolved"] is True
    assert resolve_payload["details"]["entry"]["resolved_by"] == "plan-b"

    exit_code, list_payload = run_main_json(["debt", "list"], cwd=root, capsys=capsys, monkeypatch=monkeypatch)
    assert exit_code == 0
    assert list_payload["details"]["entries"] == []

    exit_code, list_all_payload = run_main_json(["debt", "list", "--all"], cwd=root, capsys=capsys, monkeypatch=monkeypatch)
    assert exit_code == 0
    assert len(list_all_payload["details"]["entries"]) == 1
    assert list_all_payload["details"]["entries"][0]["resolved"] is True


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


def test_build_gate_signals_includes_debt_overlaps_when_flags_match(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    flag = first_open_significant_flag(plan_fixture.plan_dir)
    registry = {"entries": []}
    megaplan._core.add_or_increment_debt(
        registry,
        megaplan._core.extract_subsystem_tag(flag["concern"]),
        flag["concern"],
        [flag["id"]],
        plan_fixture.plan_name,
    )
    megaplan._core.save_debt_registry(plan_fixture.root, registry)
    _, state = load_plan(plan_fixture.root, plan_fixture.plan_name)

    signals = megaplan.evaluation.build_gate_signals(plan_fixture.plan_dir, state, plan_fixture.root)

    assert signals["signals"]["debt_overlaps"]
    assert signals["signals"]["debt_overlaps"][0]["flag_id"] == flag["id"]


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
                "files_changed": [],
                "commands_run": [],
                "evidence_files": [],
                "reviewer_verdict": "",
            }
        ],
        "watch_items": ["Watch this."],
        "sense_checks": [
            {"id": "SC1", "task_id": "T1", "question": "Did it work?", "executor_note": "", "verdict": ""}
        ],
        "meta_commentary": "Pending state.",
    }
    partial = {
        **pending,
        "tasks": [
            {
                **pending["tasks"][0],
                "status": "done",
                "executor_notes": "Implemented.",
                "files_changed": ["megaplan/handlers.py"],
            }
        ],
        "sense_checks": [
            {
                **pending["sense_checks"][0],
                "executor_note": "Confirmed execute evidence coverage.",
            }
        ],
    }
    reviewed = {
        **partial,
        "tasks": [
            {
                **partial["tasks"][0],
                "reviewer_verdict": "Pass",
                "evidence_files": ["megaplan/handlers.py"],
            }
        ],
        "sense_checks": [
            {
                **partial["sense_checks"][0],
                "verdict": "Confirmed.",
            }
        ],
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
    assert "Files changed:" in partial_md
    assert "Executor note: Confirmed execute evidence coverage." in partial_md
    assert "Reviewer verdict: Pass" in reviewed_md
    assert "Evidence files:" in reviewed_md
    assert "Verdict: Confirmed." in reviewed_md


def test_render_final_md_phase_marks_gaps_only_when_due() -> None:
    from megaplan._core import render_final_md

    data = {
        "tasks": [
            {
                "id": "T1",
                "description": "Do work",
                "depends_on": [],
                "status": "pending",
                "executor_notes": "",
                "files_changed": [],
                "commands_run": [],
                "evidence_files": [],
                "reviewer_verdict": "",
            },
            {
                "id": "T2",
                "description": "Ship work",
                "depends_on": ["T1"],
                "status": "done",
                "executor_notes": "",
                "files_changed": [],
                "commands_run": [],
                "evidence_files": [],
                "reviewer_verdict": "",
            },
        ],
        "watch_items": [],
        "sense_checks": [
            {"id": "SC1", "task_id": "T1", "question": "Did it work?", "executor_note": "", "verdict": ""},
            {"id": "SC2", "task_id": "T2", "question": "Was it reviewed?", "executor_note": "", "verdict": ""},
        ],
        "meta_commentary": "Status overview.",
    }

    finalize_md = render_final_md(data)
    execute_md = render_final_md(data, phase="execute")
    review_md = render_final_md(data, phase="review")

    assert "Executor notes: [MISSING]" not in finalize_md
    assert "Reviewer verdict: [PENDING]" not in finalize_md
    assert "## Coverage Gaps" not in finalize_md
    assert "Executor notes: [MISSING]" in execute_md
    assert "Reviewer verdict: [PENDING]" not in execute_md
    assert "Tasks without executor updates: 1" in execute_md
    assert "Executor notes missing: 1" in execute_md
    assert "Sense-check acknowledgments missing: 2" in execute_md
    assert "Reviewer verdict: [PENDING]" in review_md
    assert "Verdict: [PENDING]" in review_md
    assert "Reviewer verdicts pending: 2" in review_md
    assert "Sense-check verdicts pending: 2" in review_md


def test_validate_merge_inputs_filters_malformed_entries() -> None:
    valid = megaplan.merge._validate_merge_inputs(
        [
            {
                "task_id": "T1",
                "status": "done",
                "executor_notes": "Implemented.",
                "files_changed": ["megaplan/handlers.py"],
                "commands_run": ["pytest tests/test_megaplan.py"],
            },
            {"task_id": "T2", "status": 1, "executor_notes": "Bad type"},
            {"task_id": "T3", "executor_notes": "Missing status"},
            "bad-entry",
        ],
        required_fields=("task_id", "status", "executor_notes", "files_changed", "commands_run"),
        enum_fields={"status": {"done", "skipped"}},
        array_fields=("files_changed", "commands_run"),
        label="task_updates",
    )
    empty = megaplan.merge._validate_merge_inputs(
        [],
        required_fields=("task_id", "reviewer_verdict"),
        label="task_verdicts",
    )

    assert valid == [
        {
            "task_id": "T1",
            "status": "done",
            "executor_notes": "Implemented.",
            "files_changed": ["megaplan/handlers.py"],
            "commands_run": ["pytest tests/test_megaplan.py"],
        }
    ]
    assert empty == []


def test_validate_merge_inputs_rejects_empty_required_content() -> None:
    deviations: list[str] = []
    valid = megaplan.merge._validate_merge_inputs(
        [
            {"task_id": "T1", "status": "done", "executor_notes": "  "},
            {"task_id": "T2", "status": "done", "executor_notes": "\t"},
            {"task_id": "T3", "status": "skipped", "executor_notes": "Investigated and skipped."},
        ],
        required_fields=("task_id", "status", "executor_notes"),
        enum_fields={"status": {"done", "skipped"}},
        nonempty_fields={"executor_notes"},
        deviations=deviations,
        label="task_updates",
    )

    assert valid == [{"task_id": "T3", "status": "skipped", "executor_notes": "Investigated and skipped."}]
    assert deviations == [
        "Skipped task_updates[0]: 'executor_notes' must not be empty.",
        "Skipped task_updates[1]: 'executor_notes' must not be empty.",
    ]


def test_validate_merge_inputs_accepts_array_fields() -> None:
    deviations: list[str] = []
    valid = megaplan.merge._validate_merge_inputs(
        [
            {
                "task_id": "T1",
                "status": "done",
                "executor_notes": "Implemented.",
                "files_changed": ["megaplan/handlers.py"],
                "commands_run": ["pytest tests/test_megaplan.py"],
            },
            {
                "task_id": "T2",
                "status": "done",
                "executor_notes": "Bad arrays.",
                "files_changed": "megaplan/handlers.py",
                "commands_run": [],
            },
        ],
        required_fields=("task_id", "status", "executor_notes", "files_changed", "commands_run"),
        enum_fields={"status": {"done", "skipped"}},
        nonempty_fields={"executor_notes"},
        array_fields=("files_changed", "commands_run"),
        deviations=deviations,
        label="task_updates",
    )

    assert valid == [
        {
            "task_id": "T1",
            "status": "done",
            "executor_notes": "Implemented.",
            "files_changed": ["megaplan/handlers.py"],
            "commands_run": ["pytest tests/test_megaplan.py"],
        }
    ]
    assert deviations == [
        "Skipped malformed task_updates[1]: invalid field types or enum values.",
    ]


def test_validate_merge_inputs_rejects_empty_reviewer_verdict() -> None:
    deviations: list[str] = []
    valid = megaplan.merge._validate_merge_inputs(
        [
            {"task_id": "T1", "reviewer_verdict": ""},
            {"task_id": "T2", "reviewer_verdict": "   "},
            {"task_id": "T3", "reviewer_verdict": "Looks good."},
        ],
        required_fields=("task_id", "reviewer_verdict"),
        nonempty_fields={"reviewer_verdict"},
        deviations=deviations,
        label="task_verdicts",
    )

    assert valid == [{"task_id": "T3", "reviewer_verdict": "Looks good."}]
    assert deviations == [
        "Skipped task_verdicts[0]: 'reviewer_verdict' must not be empty.",
        "Skipped task_verdicts[1]: 'reviewer_verdict' must not be empty.",
    ]


def test_validate_merge_inputs_rejects_empty_sense_check_verdict() -> None:
    deviations: list[str] = []
    valid = megaplan.merge._validate_merge_inputs(
        [
            {"sense_check_id": "SC1", "verdict": ""},
            {"sense_check_id": "SC2", "verdict": "Confirmed."},
        ],
        required_fields=("sense_check_id", "verdict"),
        nonempty_fields={"verdict"},
        deviations=deviations,
        label="sense_check_verdicts",
    )

    assert valid == [{"sense_check_id": "SC2", "verdict": "Confirmed."}]
    assert deviations == [
        "Skipped sense_check_verdicts[0]: 'verdict' must not be empty.",
    ]


def test_duplicate_sense_check_verdict_dedup() -> None:
    """Two verdicts for SC1, zero for SC2 — should count 1 unique, not 2."""
    deviations: list[str] = []
    valid = megaplan.merge._validate_merge_inputs(
        [
            {"sense_check_id": "SC1", "verdict": "First pass."},
            {"sense_check_id": "SC1", "verdict": "Second pass."},
        ],
        required_fields=("sense_check_id", "verdict"),
        nonempty_fields={"verdict"},
        deviations=deviations,
        label="sense_check_verdicts",
    )

    # Both entries pass validation (last-entry-wins happens at merge time in handler)
    assert len(valid) == 2
    assert valid[0]["verdict"] == "First pass."
    assert valid[1]["verdict"] == "Second pass."


def test_is_substantive_reviewer_verdict_accepts_real_verdict() -> None:
    verdict = "Verification work is acceptable and was checked through command evidence captured in the executor notes."
    assert megaplan.handlers._is_substantive_reviewer_verdict(verdict) is True


def test_is_substantive_reviewer_verdict_rejects_short_string() -> None:
    assert megaplan.handlers._is_substantive_reviewer_verdict("Looks good.") is False


def test_is_substantive_reviewer_verdict_rejects_repeated_words() -> None:
    assert megaplan.handlers._is_substantive_reviewer_verdict("ok ok ok ok ok ok ok") is False


def test_is_substantive_reviewer_verdict_accepts_boundary_case() -> None:
    assert megaplan.handlers._is_substantive_reviewer_verdict("alpha beta beta gamma") is True


def test_execute_happy_path_tracks_all_tasks(plan_fixture: PlanFixture) -> None:
    """The default mock execute output still covers every finalized task."""
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    # Mock finalize creates 2 tasks (T1, T2). Mock execute updates both.
    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    assert response["success"] is True
    assert response["warnings"] == []
    assert "2/2 tasks tracked" in response["summary"]
    assert "2/2 sense checks acknowledged" in response["summary"]


def test_execute_timeout_recovers_partial_progress_from_finalize_json(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][0]["status"] = "done"
    finalize_data["tasks"][0]["executor_notes"] = "Verified the implementation artifact before timeout recovery."
    finalize_data["tasks"][0]["files_changed"] = ["IMPLEMENTED_BY_MEGAPLAN.txt"]
    finalize_data["tasks"][0]["commands_run"] = ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"]
    finalize_data["sense_checks"][0]["executor_note"] = "Confirmed the implementation artifact exists."
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    def timing_out_worker(*args, **kwargs):
        raise megaplan.CliError("worker_timeout", "execute timed out", extra={"session_id": "test-session", "raw_output": ""})

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", timing_out_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    state = load_state(plan_fixture.plan_dir)
    recovered = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is False
    assert response["next_step"] == "execute"
    assert response["state"] == megaplan.STATE_FINALIZED
    assert recovered["tasks"][0]["status"] == "done"
    assert state["history"][-1]["result"] == "timeout"
    assert state["sessions"][megaplan.workers.session_key_for("execute", "codex")]["id"] == "test-session"


def test_execute_timeout_reads_execution_checkpoint_json(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    checkpoint_payload = {
        "task_updates": [
            {
                "task_id": "T1",
                "status": "done",
                "executor_notes": "Recovered from execution checkpoint.",
                "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                "commands_run": ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"],
            }
        ],
        "sense_check_acknowledgments": [
            {"sense_check_id": "SC1", "executor_note": "Recovered checkpoint sense check."}
        ],
    }
    (plan_fixture.plan_dir / "execution_checkpoint.json").write_text(
        json.dumps(checkpoint_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    def timing_out_worker(*args, **kwargs):
        raise megaplan.CliError("worker_timeout", "execute timed out", extra={"session_id": "checkpoint-session", "raw_output": ""})

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", timing_out_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    recovered = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert recovered["tasks"][0]["status"] == "done"
    assert recovered["tasks"][0]["files_changed"] == ["IMPLEMENTED_BY_MEGAPLAN.txt"]
    assert recovered["sense_checks"][0]["executor_note"] == "Recovered checkpoint sense check."
    assert any(
        "Recovered timeout checkpoint from execution_checkpoint.json" in deviation
        for deviation in response["deviations"]
    )


def test_execute_timeout_resets_done_tasks_without_any_evidence(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][0]["status"] = "done"
    finalize_data["tasks"][0]["executor_notes"] = "Claimed completion without evidence."
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    def timing_out_worker(*args, **kwargs):
        raise megaplan.CliError("worker_timeout", "execute timed out", extra={"session_id": "test-session", "raw_output": ""})

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", timing_out_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    recovered = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is False
    assert recovered["tasks"][0]["status"] == "pending"
    assert "Timeout recovery reset this task to pending" in recovered["tasks"][0]["executor_notes"]
    assert any("Reset timed-out done tasks to pending" in deviation for deviation in response["deviations"])


def test_execute_reports_advisory_when_structured_output_disagrees_with_disk_checkpoint(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][0]["status"] = "done"
    finalize_data["tasks"][0]["executor_notes"] = "Checkpointed as done on disk before final structured output."
    finalize_data["tasks"][0]["files_changed"] = ["IMPLEMENTED_BY_MEGAPLAN.txt"]
    finalize_data["tasks"][0]["commands_run"] = ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"]
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    worker = WorkerResult(
        payload={
            "output": "Execution completed with structured output.",
            "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
            "commands_run": ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "skipped",
                    "executor_notes": "Verified the disk checkpoint should be downgraded because no additional work was required.",
                    "files_changed": [],
                    "commands_run": [],
                },
                {
                    "task_id": "T2",
                    "status": "done",
                    "executor_notes": "Verified the remaining task completed successfully.",
                    "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                    "commands_run": ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"],
                },
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed the checkpoint mismatch was intentional."},
                {"sense_check_id": "SC2", "executor_note": "Confirmed the remaining task completed successfully."},
            ],
        },
        raw_output="execute with mismatch",
        duration_ms=1,
        cost_usd=0.0,
        session_id="execute-mismatch",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    merged_finalize = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is True
    assert merged_finalize["tasks"][0]["status"] == "skipped"
    assert any(
        "task T1 was 'done' on disk before merge but structured output set it to 'skipped'" in deviation
        for deviation in response["deviations"]
    )


def test_review_flags_incomplete_verdicts(plan_fixture: PlanFixture) -> None:
    """When reviewer returns fewer verdicts than tasks exist, issues surface it."""
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    response = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    # Mock review provides verdicts for T1 and T2, matching mock finalize's tasks
    # So no "Incomplete review" issue expected in the happy path
    assert response["state"] == megaplan.STATE_DONE


def test_validate_merge_inputs_tracks_deviations() -> None:
    """Verify that _validate_merge_inputs populates the deviations list for malformed input."""
    deviations: list[str] = []
    megaplan.merge._validate_merge_inputs(
        [
            "not-a-dict",
            {"task_id": "T1"},  # missing required fields
            {"task_id": "T2", "status": "invalid_enum", "executor_notes": "x"},  # bad enum
        ],
        required_fields=("task_id", "status", "executor_notes"),
        enum_fields={"status": {"done", "skipped"}},
        deviations=deviations,
        label="task_updates",
    )
    assert len(deviations) == 3
    assert "expected object" in deviations[0]
    assert "missing required" in deviations[1]
    assert "invalid field" in deviations[2]


def test_validate_merge_inputs_non_list_returns_empty() -> None:
    """Non-list input returns empty with no crash."""
    assert megaplan.merge._validate_merge_inputs(
        "not-a-list",
        required_fields=("task_id",),
        label="test",
    ) == []
    assert megaplan.merge._validate_merge_inputs(
        None,
        required_fields=("task_id",),
        label="test",
    ) == []


def test_execute_deduplicates_task_updates_and_blocks_incomplete_coverage(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    worker = WorkerResult(
        payload={
            "output": "Partial execution completed.",
            "files_changed": ["src/example.py"],
            "commands_run": ["pytest -k partial"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Initial pass.",
                    "files_changed": ["src/example.py"],
                    "commands_run": ["pytest -k partial"],
                },
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Final pass.",
                    "files_changed": ["src/example.py"],
                    "commands_run": ["pytest -k partial"],
                },
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed."},
            ],
        },
        raw_output="partial execute",
        duration_ms=1,
        cost_usd=0.0,
        session_id="execute-duplicate",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    state = load_state(plan_fixture.plan_dir)
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    execute_entry = next(entry for entry in state["history"] if entry["step"] == "execute")
    final_md = (plan_fixture.plan_dir / "final.md").read_text(encoding="utf-8")

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert response["summary"] == (
        "Blocked: 1/2 tasks have no executor update; 1/2 sense checks have no executor acknowledgment. "
        "Re-run execute to complete tracking."
    )
    assert "Duplicate task_update for 'T1' — last entry wins." in response["deviations"]
    assert finalize_data["tasks"][0]["executor_notes"] == "Final pass."
    assert finalize_data["tasks"][1]["status"] == "pending"
    assert execute_entry["result"] == "blocked"
    assert (plan_fixture.plan_dir / "execution.json").exists()
    assert (plan_fixture.plan_dir / "execution_audit.json").exists()
    assert "## Coverage Gaps" in final_md
    assert "Tasks without executor updates: 1" in final_md


def test_review_blocks_incomplete_coverage_and_allows_rerun(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    first_review = WorkerResult(
        payload={
            "review_verdict": "approved",
            "criteria": [{"name": "criterion", "pass": True, "evidence": "checked"}],
            "issues": [],
            "summary": "Partial review.",
            "task_verdicts": [
                {
                    "task_id": "T1",
                    "reviewer_verdict": "Pass - partial.",
                    "evidence_files": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                },
                {
                    "task_id": "T1",
                    "reviewer_verdict": "Pass - final.",
                    "evidence_files": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                },
            ],
            "sense_check_verdicts": [
                {"sense_check_id": "SC1", "verdict": "Confirmed."},
            ],
        },
        raw_output="partial review",
        duration_ms=1,
        cost_usd=0.0,
        session_id="review-partial",
    )
    second_review = WorkerResult(
        payload={
            "review_verdict": "approved",
            "criteria": [{"name": "criterion", "pass": True, "evidence": "checked again"}],
            "issues": [],
            "summary": "Complete review.",
            "task_verdicts": [
                {
                    "task_id": "T1",
                    "reviewer_verdict": "Pass - rerun.",
                    "evidence_files": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                },
                {
                    "task_id": "T2",
                    "reviewer_verdict": "Pass - rerun with command evidence that is substantive enough for FLAG-006 softening.",
                    "evidence_files": [],
                },
            ],
            "sense_check_verdicts": [
                {"sense_check_id": "SC1", "verdict": "Confirmed on rerun."},
                {"sense_check_id": "SC2", "verdict": "Confirmed on rerun."},
            ],
        },
        raw_output="complete review",
        duration_ms=1,
        cost_usd=0.0,
        session_id="review-complete",
    )
    results = iter([first_review, second_review])
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (next(results), "codex", "persistent", False),
    )

    blocked = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    after_block = read_json(plan_fixture.plan_dir / "finalize.json")
    blocked_state = load_state(plan_fixture.plan_dir)
    blocked_entry = blocked_state["history"][-1]
    blocked_md = (plan_fixture.plan_dir / "final.md").read_text(encoding="utf-8")

    assert blocked["success"] is False
    assert blocked["state"] == megaplan.STATE_EXECUTED
    assert blocked["next_step"] == "review"
    assert blocked["summary"] == (
        "Blocked: incomplete review coverage (1/2 task verdicts, 1/2 sense checks). "
        "Re-run review to complete."
    )
    assert "Duplicate task_verdict for 'T1' — last entry wins." in blocked["issues"]
    assert after_block["tasks"][0]["reviewer_verdict"] == "Pass - final."
    assert after_block["tasks"][1]["reviewer_verdict"] == ""
    assert blocked_entry["result"] == "blocked"
    assert (plan_fixture.plan_dir / "review.json").exists()
    assert "## Coverage Gaps" in blocked_md
    assert "Reviewer verdicts pending: 1" in blocked_md
    assert "Sense-check verdicts pending: 1" in blocked_md

    completed = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    final_state = load_state(plan_fixture.plan_dir)
    final_data = read_json(plan_fixture.plan_dir / "finalize.json")

    assert completed["success"] is True
    assert completed["state"] == megaplan.STATE_DONE
    assert completed["next_step"] is None
    assert final_state["current_state"] == megaplan.STATE_DONE
    assert final_data["tasks"][0]["reviewer_verdict"] == "Pass - rerun."
    assert (
        final_data["tasks"][1]["reviewer_verdict"]
        == "Pass - rerun with command evidence that is substantive enough for FLAG-006 softening."
    )
    assert all(check["verdict"] == "Confirmed on rerun." for check in final_data["sense_checks"])


def test_execute_blocks_done_task_without_any_per_task_evidence(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    worker = WorkerResult(
        payload={
            "output": "Executed with incomplete evidence.",
            "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
            "commands_run": ["mock-run"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Implemented the main artifact.",
                    "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                    "commands_run": ["mock-run"],
                },
                {
                    "task_id": "T2",
                    "status": "done",
                    "executor_notes": "Verified the work but forgot to capture evidence.",
                    "files_changed": [],
                    "commands_run": [],
                },
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed the implementation artifact exists."},
                {"sense_check_id": "SC2", "executor_note": "Confirmed the verification task was reviewed."},
            ],
        },
        raw_output="execute missing evidence",
        duration_ms=1,
        cost_usd=0.0,
        session_id="execute-missing-evidence",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert "missing both files_changed and commands_run" in response["summary"]


def test_execute_softens_done_task_with_commands_only(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    worker = WorkerResult(
        payload={
            "output": "Executed with command-only verification evidence.",
            "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
            "commands_run": ["mock-run", "mock-verify"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Implemented the main artifact.",
                    "files_changed": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                    "commands_run": ["mock-run"],
                },
                {
                    "task_id": "T2",
                    "status": "done",
                    "executor_notes": "Verified the work using command output only.",
                    "files_changed": [],
                    "commands_run": ["mock-verify"],
                },
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed the implementation artifact exists."},
                {"sense_check_id": "SC2", "executor_note": "Confirmed the verification task is backed by command output."},
            ],
        },
        raw_output="execute softened evidence",
        duration_ms=1,
        cost_usd=0.0,
        session_id="execute-softened-evidence",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is True
    assert response["state"] == megaplan.STATE_EXECUTED
    assert any("FLAG-006 softening" in deviation for deviation in response["deviations"])
    assert finalize_data["tasks"][1]["files_changed"] == []
    assert finalize_data["tasks"][1]["commands_run"] == ["mock-verify"]


def test_execute_multi_batch_happy_path_aggregates_results(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"] = [
        {
            "id": "T1",
            "description": "First batch",
            "depends_on": [],
            "status": "pending",
            "executor_notes": "",
            "files_changed": [],
            "commands_run": [],
            "evidence_files": [],
            "reviewer_verdict": "",
        },
        {
            "id": "T2",
            "description": "Second batch",
            "depends_on": ["T1"],
            "status": "pending",
            "executor_notes": "",
            "files_changed": [],
            "commands_run": [],
            "evidence_files": [],
            "reviewer_verdict": "",
        },
    ]
    finalize_data["sense_checks"] = [
        {"id": "SC1", "task_id": "T1", "question": "Batch one?", "executor_note": "", "verdict": ""},
        {"id": "SC2", "task_id": "T2", "question": "Batch two?", "executor_note": "", "verdict": ""},
    ]
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    snapshots = iter([
        ({}, None),
        ({"batch1.py": "hash-1"}, None),
        ({"batch1.py": "hash-1"}, None),
        ({"batch1.py": "hash-1"}, None),
        ({"batch1.py": "hash-1", "batch2.py": "hash-2"}, None),
        ({"batch1.py": "hash-1", "batch2.py": "hash-2"}, None),
    ])
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: next(snapshots))

    def batched_worker(step: str, state: dict, plan_dir: Path, args: Namespace, *, root: Path, resolved=None, prompt_override: str | None = None):
        assert prompt_override is not None
        if "[T1]" in prompt_override:
            payload = {
                "output": "Batch one complete.",
                "files_changed": ["batch1.py"],
                "commands_run": ["pytest -k batch1"],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T1",
                        "status": "done",
                        "executor_notes": "Completed the first batch and verified its focused check.",
                        "files_changed": ["batch1.py"],
                        "commands_run": ["pytest -k batch1"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC1", "executor_note": "Confirmed batch one output."}
                ],
            }
            return WorkerResult(payload=payload, raw_output="batch1", duration_ms=2, cost_usd=0.1, session_id="batch-1", trace_output='{"batch":1}\n'), "codex", "persistent", False
        if "[T2]" in prompt_override:
            payload = {
                "output": "Batch two complete.",
                "files_changed": ["batch2.py"],
                "commands_run": ["pytest -k batch2"],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T2",
                        "status": "done",
                        "executor_notes": "Completed the dependent batch after T1 was persisted.",
                        "files_changed": ["batch2.py"],
                        "commands_run": ["pytest -k batch2"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC2", "executor_note": "Confirmed batch two output."}
                ],
            }
            return WorkerResult(payload=payload, raw_output="batch2", duration_ms=3, cost_usd=0.2, session_id="batch-2", trace_output='{"batch":2}\n'), "codex", "persistent", False
        raise AssertionError(f"Unexpected batch prompt: {prompt_override}")

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", batched_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    final_data = read_json(plan_fixture.plan_dir / "finalize.json")
    execution = read_json(plan_fixture.plan_dir / "execution.json")

    assert response["success"] is True
    assert response["state"] == megaplan.STATE_EXECUTED
    assert [task["status"] for task in final_data["tasks"]] == ["done", "done"]
    assert [check["executor_note"] for check in final_data["sense_checks"]] == [
        "Confirmed batch one output.",
        "Confirmed batch two output.",
    ]
    assert execution["output"].startswith("Aggregated execute batches: completed 2/2.")
    assert [item["task_id"] for item in execution["task_updates"]] == ["T1", "T2"]
    assert [item["sense_check_id"] for item in execution["sense_check_acknowledgments"]] == ["SC1", "SC2"]
    assert execution["files_changed"] == ["batch1.py", "batch2.py"]
    assert execution["commands_run"] == ["pytest -k batch1", "pytest -k batch2"]
    assert (plan_fixture.plan_dir / "execution_trace.jsonl").read_text(encoding="utf-8") == '{"batch":1}\n{"batch":2}\n'
    # T4: Per-batch artifact files should be written with correct per-batch task_updates
    batch_1 = read_json(plan_fixture.plan_dir / "execution_batch_1.json")
    batch_2 = read_json(plan_fixture.plan_dir / "execution_batch_2.json")
    assert [item["task_id"] for item in batch_1["task_updates"]] == ["T1"]
    assert [item["task_id"] for item in batch_2["task_updates"]] == ["T2"]


def test_execute_multi_batch_timeout_preserves_prior_batches(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][1]["depends_on"] = ["T1"]
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    snapshots = iter([
        ({}, None),
        ({"batch1.py": "hash-1"}, None),
        ({"batch1.py": "hash-1"}, None),
        ({"batch1.py": "hash-1"}, None),
    ])
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: next(snapshots))

    def timed_worker(step: str, state: dict, plan_dir: Path, args: Namespace, *, root: Path, resolved=None, prompt_override: str | None = None):
        assert prompt_override is not None
        if "[T1]" in prompt_override:
            payload = {
                "output": "Batch one complete.",
                "files_changed": ["batch1.py"],
                "commands_run": ["pytest -k batch1"],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T1",
                        "status": "done",
                        "executor_notes": "Completed the first batch and verified its focused check.",
                        "files_changed": ["batch1.py"],
                        "commands_run": ["pytest -k batch1"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC1", "executor_note": "Confirmed batch one output."}
                ],
            }
            return WorkerResult(payload=payload, raw_output="batch1", duration_ms=2, cost_usd=0.1, session_id="batch-1"), "codex", "persistent", False
        raise megaplan.CliError("worker_timeout", "execute timed out", extra={"session_id": "batch-2", "raw_output": "partial"})

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", timed_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    final_data = read_json(plan_fixture.plan_dir / "finalize.json")
    execution = read_json(plan_fixture.plan_dir / "execution.json")
    state = load_state(plan_fixture.plan_dir)

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert final_data["tasks"][0]["status"] == "done"
    assert final_data["tasks"][1]["status"] == "pending"
    assert final_data["sense_checks"][0]["executor_note"] == "Confirmed batch one output."
    assert final_data["sense_checks"][1]["executor_note"] == ""
    assert [item["task_id"] for item in execution["task_updates"]] == ["T1"]
    assert state["history"][-1]["result"] == "timeout"


def test_execute_rerun_with_completed_dependency_uses_single_batch_fast_path(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][0]["status"] = "done"
    finalize_data["tasks"][0]["executor_notes"] = "Already completed."
    finalize_data["tasks"][0]["files_changed"] = ["batch1.py"]
    finalize_data["tasks"][0]["commands_run"] = ["pytest -k batch1"]
    finalize_data["tasks"][1]["depends_on"] = ["T1"]
    finalize_data["sense_checks"][0]["executor_note"] = "Already acknowledged."
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    prompt_overrides: list[str | None] = []

    def rerun_worker(step: str, state: dict, plan_dir: Path, args: Namespace, *, root: Path, resolved=None, prompt_override: str | None = None):
        prompt_overrides.append(prompt_override)
        payload = {
            "output": "Rerun complete.",
            "files_changed": ["batch1.py", "batch2.py"],
            "commands_run": ["pytest -k rerun"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Kept the already-completed task intact during rerun.",
                    "files_changed": ["batch1.py"],
                    "commands_run": ["pytest -k batch1"],
                },
                {
                    "task_id": "T2",
                    "status": "done",
                    "executor_notes": "Completed the remaining dependent task.",
                    "files_changed": ["batch2.py"],
                    "commands_run": ["pytest -k rerun"],
                },
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Already acknowledged."},
                {"sense_check_id": "SC2", "executor_note": "Confirmed rerun output."},
            ],
        }
        return WorkerResult(payload=payload, raw_output="rerun", duration_ms=1, cost_usd=0.0, session_id="rerun"), "codex", "persistent", False

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", rerun_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    assert response["success"] is True
    assert prompt_overrides == [None]


def test_execute_multi_batch_observation_allows_cross_batch_reedit_and_flags_phantoms(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][1]["depends_on"] = ["T1"]
    (plan_fixture.plan_dir / "finalize.json").write_text(json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8")

    snapshots = iter([
        ({}, None),
        ({"megaplan/handlers.py": "hash-1"}, None),
        ({"megaplan/handlers.py": "hash-1"}, None),
        ({"megaplan/handlers.py": "hash-1"}, None),
        ({"megaplan/handlers.py": "hash-2"}, None),
        ({"megaplan/handlers.py": "hash-2"}, None),
    ])
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: next(snapshots))

    def observation_worker(step: str, state: dict, plan_dir: Path, args: Namespace, *, root: Path, resolved=None, prompt_override: str | None = None):
        assert prompt_override is not None
        if "[T1]" in prompt_override:
            payload = {
                "output": "Batch one complete.",
                "files_changed": ["megaplan/handlers.py"],
                "commands_run": ["pytest -k batch1"],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T1",
                        "status": "done",
                        "executor_notes": "Edited handlers.py in batch one.",
                        "files_changed": ["megaplan/handlers.py"],
                        "commands_run": ["pytest -k batch1"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC1", "executor_note": "Confirmed batch one output."}
                ],
            }
            return WorkerResult(payload=payload, raw_output="batch1", duration_ms=1, cost_usd=0.0, session_id="batch-1"), "codex", "persistent", False
        payload = {
            "output": "Batch two complete.",
            "files_changed": ["megaplan/handlers.py", "ghost.py"],
            "commands_run": ["pytest -k batch2"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T2",
                    "status": "done",
                    "executor_notes": "Re-edited handlers.py in batch two.",
                    "files_changed": ["megaplan/handlers.py", "ghost.py"],
                    "commands_run": ["pytest -k batch2"],
                }
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC2", "executor_note": "Confirmed batch two output."}
            ],
        }
        return WorkerResult(payload=payload, raw_output="batch2", duration_ms=1, cost_usd=0.0, session_id="batch-2"), "codex", "persistent", False

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", observation_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    assert response["success"] is True
    assert any("ghost.py" in deviation for deviation in response["deviations"])
    assert not any(
        "executor claimed files not observed" in deviation and "megaplan/handlers.py" in deviation
        for deviation in response["deviations"]
    )


def test_review_blocks_empty_evidence_files_without_substantive_verdict(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    worker = WorkerResult(
        payload={
            "review_verdict": "approved",
            "criteria": [{"name": "criterion", "pass": True, "evidence": "checked"}],
            "issues": [],
            "summary": "Review missing evidence files.",
            "task_verdicts": [
                {
                    "task_id": "T1",
                    "reviewer_verdict": "Pass - file backed.",
                    "evidence_files": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                },
                {
                    "task_id": "T2",
                    "reviewer_verdict": "Pass.",
                    "evidence_files": [],
                },
            ],
            "sense_check_verdicts": [
                {"sense_check_id": "SC1", "verdict": "Confirmed."},
                {"sense_check_id": "SC2", "verdict": "Confirmed."},
            ],
        },
        raw_output="review missing evidence files",
        duration_ms=1,
        cost_usd=0.0,
        session_id="review-missing-evidence-files",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    response = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_EXECUTED
    assert response["next_step"] == "review"
    assert "missing reviewer evidence_files" in response["summary"]


def test_review_softens_substantive_verdict_without_evidence_files_and_can_kick_back(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )

    worker = WorkerResult(
        payload={
            "review_verdict": "needs_rework",
            "criteria": [{"name": "criterion", "pass": False, "evidence": "Task T1 still needs follow-up edits."}],
            "issues": ["T1 implementation is incomplete and needs another execute pass."],
            "summary": "Needs rework: one task is still incomplete.",
            "task_verdicts": [
                {
                    "task_id": "T1",
                    "reviewer_verdict": "Needs more work. The main implementation is not complete yet.",
                    "evidence_files": ["IMPLEMENTED_BY_MEGAPLAN.txt"],
                },
                {
                    "task_id": "T2",
                    "reviewer_verdict": "Verification work is acceptable and was checked through command evidence captured in the executor notes.",
                    "evidence_files": [],
                },
            ],
            "sense_check_verdicts": [
                {"sense_check_id": "SC1", "verdict": "Needs another execute pass."},
                {"sense_check_id": "SC2", "verdict": "Confirmed for the verification task."},
            ],
        },
        raw_output="review needs rework",
        duration_ms=1,
        cost_usd=0.0,
        session_id="review-needs-rework",
    )
    monkeypatch.setattr(
        megaplan.workers,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    response = megaplan.handle_review(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    stored_review = read_json(plan_fixture.plan_dir / "review.json")

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert response["summary"] == "Review requested another execute pass. Re-run execute using the review findings as context."
    assert any("FLAG-006 softening" in issue for issue in response["issues"])
    assert state["current_state"] == megaplan.STATE_FINALIZED
    assert state["history"][-1]["result"] == "needs_rework"
    assert stored_review["review_verdict"] == "needs_rework"


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


# ---------------------------------------------------------------------------
# Progress command tests (T7)
# ---------------------------------------------------------------------------


def _drive_to_finalized(plan_fixture: PlanFixture) -> None:
    make_args = plan_fixture.make_args
    megaplan.handle_plan(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_critique(plan_fixture.root, make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(plan_fixture.root, make_args(plan=plan_fixture.plan_name))


def test_progress_all_pending(plan_fixture: PlanFixture) -> None:
    _drive_to_finalized(plan_fixture)
    response = megaplan.handle_progress(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name),
    )
    assert response["success"] is True
    assert response["tasks_pending"] == response["tasks_total"]
    assert response["tasks_done"] == 0
    assert response["batches_completed"] == 0
    assert response["batches_total"] >= 1


def test_progress_after_partial_execution(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _drive_to_finalized(plan_fixture)
    # Set up 2-batch plan: T1 (no deps), T2 (depends on T1)
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"] = [
        {
            "id": "T1",
            "description": "First",
            "depends_on": [],
            "status": "done",
            "executor_notes": "Done.",
            "files_changed": ["a.py"],
            "commands_run": ["pytest"],
            "evidence_files": [],
            "reviewer_verdict": "",
        },
        {
            "id": "T2",
            "description": "Second",
            "depends_on": ["T1"],
            "status": "pending",
            "executor_notes": "",
            "files_changed": [],
            "commands_run": [],
            "evidence_files": [],
            "reviewer_verdict": "",
        },
    ]
    (plan_fixture.plan_dir / "finalize.json").write_text(
        json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8"
    )
    response = megaplan.handle_progress(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name),
    )
    assert response["batches_completed"] == 1
    assert response["batches_total"] == 2
    assert response["tasks_done"] == 1
    assert response["tasks_pending"] == 1


def test_progress_after_full_execution(plan_fixture: PlanFixture) -> None:
    _drive_to_finalized(plan_fixture)
    megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    response = megaplan.handle_progress(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name),
    )
    assert response["tasks_done"] + response["tasks_skipped"] == response["tasks_total"]
    assert response["batches_completed"] == response["batches_total"]
    assert response["tasks_pending"] == 0


# ---------------------------------------------------------------------------
# Per-batch execute mode tests (T10)
# ---------------------------------------------------------------------------


def _setup_two_batch_plan(plan_fixture: PlanFixture) -> None:
    """Drive plan to finalized and set up 2-batch task structure."""
    _drive_to_finalized(plan_fixture)
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"] = [
        {
            "id": "T1",
            "description": "First batch",
            "depends_on": [],
            "status": "pending",
            "executor_notes": "",
            "files_changed": [],
            "commands_run": [],
            "evidence_files": [],
            "reviewer_verdict": "",
        },
        {
            "id": "T2",
            "description": "Second batch",
            "depends_on": ["T1"],
            "status": "pending",
            "executor_notes": "",
            "files_changed": [],
            "commands_run": [],
            "evidence_files": [],
            "reviewer_verdict": "",
        },
    ]
    finalize_data["sense_checks"] = [
        {"id": "SC1", "task_id": "T1", "question": "Batch one?", "executor_note": "", "verdict": ""},
        {"id": "SC2", "task_id": "T2", "question": "Batch two?", "executor_note": "", "verdict": ""},
    ]
    (plan_fixture.plan_dir / "finalize.json").write_text(
        json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8"
    )


def _batch_worker(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
    """Mock worker that returns batch-specific results based on prompt content."""
    assert prompt_override is not None
    if "[T1]" in prompt_override:
        payload = {
            "output": "Batch one complete.",
            "files_changed": ["batch1.py"],
            "commands_run": ["pytest -k batch1"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Completed the first batch.",
                    "files_changed": ["batch1.py"],
                    "commands_run": ["pytest -k batch1"],
                }
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed batch one."}
            ],
        }
        return WorkerResult(payload=payload, raw_output="batch1", duration_ms=2, cost_usd=0.1, session_id="batch-1"), "codex", "persistent", False
    if "[T2]" in prompt_override:
        payload = {
            "output": "Batch two complete.",
            "files_changed": ["batch2.py"],
            "commands_run": ["pytest -k batch2"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T2",
                    "status": "done",
                    "executor_notes": "Completed the second batch.",
                    "files_changed": ["batch2.py"],
                    "commands_run": ["pytest -k batch2"],
                }
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC2", "executor_note": "Confirmed batch two."}
            ],
        }
        return WorkerResult(payload=payload, raw_output="batch2", duration_ms=3, cost_usd=0.2, session_id="batch-2"), "codex", "persistent", False
    raise AssertionError(f"Unexpected batch prompt: {prompt_override}")


def test_batch_1_on_two_batch_plan_stays_finalized(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))
    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", _batch_worker)

    make_args = plan_fixture.make_args
    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )
    state = read_json(plan_fixture.plan_dir / "state.json")
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert response["batch"] == 1
    assert response["batches_total"] == 2
    assert response["batches_remaining"] == 1
    assert (plan_fixture.plan_dir / "execution_batch_1.json").exists()
    assert not (plan_fixture.plan_dir / "execution.json").exists()
    assert finalize_data["tasks"][0]["status"] == "done"
    assert finalize_data["tasks"][1]["status"] == "pending"


def test_batch_timeout_reads_execution_batch_n_json(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))

    checkpoint_payload = {
        "task_updates": [
            {
                "task_id": "T1",
                "status": "done",
                "executor_notes": "Recovered from batch checkpoint.",
                "files_changed": ["batch1.py"],
                "commands_run": ["pytest -k batch1"],
            }
        ],
        "sense_check_acknowledgments": [
            {"sense_check_id": "SC1", "executor_note": "Recovered batch checkpoint sense check."}
        ],
    }
    (plan_fixture.plan_dir / "execution_batch_1.json").write_text(
        json.dumps(checkpoint_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    def timing_out_batch_worker(*args, **kwargs):
        raise megaplan.CliError("worker_timeout", "execute timed out", extra={"session_id": "batch-timeout", "raw_output": ""})

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", timing_out_batch_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )
    recovered = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert recovered["tasks"][0]["status"] == "done"
    assert recovered["tasks"][0]["files_changed"] == ["batch1.py"]
    assert recovered["tasks"][1]["status"] == "pending"
    assert recovered["sense_checks"][0]["executor_note"] == "Recovered batch checkpoint sense check."
    assert any(
        "Recovered timeout checkpoint from execution_batch_1.json" in deviation
        for deviation in response["deviations"]
    )


def test_batch_2_after_batch_1_transitions_to_executed(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))
    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", _batch_worker)

    make_args = plan_fixture.make_args
    # Execute batch 1
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )
    # Execute batch 2
    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=2),
    )
    state = read_json(plan_fixture.plan_dir / "state.json")
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["state"] == megaplan.STATE_EXECUTED
    assert response["next_step"] == "review"
    assert response["batch"] == 2
    assert response["batches_remaining"] == 0
    assert (plan_fixture.plan_dir / "execution.json").exists()
    execution = read_json(plan_fixture.plan_dir / "execution.json")
    assert [item["task_id"] for item in execution["task_updates"]] == ["T1", "T2"]
    assert all(t["status"] == "done" for t in finalize_data["tasks"])


def test_execute_quality_advisories_flow_into_batch_artifacts_aggregate_and_next_prompt(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    batch1_path = plan_fixture.project_dir / "batch1.txt"
    batch2_path = plan_fixture.project_dir / "batch2.txt"
    _write_lines(batch1_path, 10, prefix="batch1_before")
    _write_lines(batch2_path, 10, prefix="batch2_before")

    snapshots = iter([
        ({"batch1.txt": "before-1"}, None),
        ({"batch1.txt": "after-1"}, None),
        ({"batch1.txt": "after-1"}, None),
        ({"batch1.txt": "after-1", "batch2.txt": "before-2"}, None),
        ({"batch1.txt": "after-1", "batch2.txt": "after-2"}, None),
        ({"batch1.txt": "after-1", "batch2.txt": "after-2"}, None),
    ])
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: next(snapshots))
    monkeypatch.setattr(megaplan.execution, "load_config", lambda *_: {})

    seen_prompts: list[str | None] = []

    def quality_batch_worker(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
        seen_prompts.append(prompt_override)
        assert prompt_override is not None
        if "[T1]" in prompt_override:
            _write_lines(batch1_path, 310, prefix="batch1_after")
            payload = {
                "output": "Batch one complete.",
                "files_changed": ["batch1.txt"],
                "commands_run": ["pytest -k batch1"],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T1",
                        "status": "done",
                        "executor_notes": "Completed the first batch and verified the file growth trigger.",
                        "files_changed": ["batch1.txt"],
                        "commands_run": ["pytest -k batch1"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC1", "executor_note": "Confirmed batch one output."}
                ],
            }
            return WorkerResult(payload=payload, raw_output="batch1", duration_ms=2, cost_usd=0.1, session_id="batch-1"), "codex", "persistent", False
        if "[T2]" in prompt_override:
            assert "Prior batch deviations (address if applicable):" in prompt_override
            assert "Advisory quality: batch1.txt grew by 300 lines (threshold 200)." in prompt_override
            _write_lines(batch2_path, 20, prefix="batch2_after")
            payload = {
                "output": "Batch two complete.",
                "files_changed": ["batch2.txt"],
                "commands_run": ["pytest -k batch2"],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T2",
                        "status": "done",
                        "executor_notes": "Completed the second batch after reviewing prior advisories.",
                        "files_changed": ["batch2.txt"],
                        "commands_run": ["pytest -k batch2"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC2", "executor_note": "Confirmed batch two output."}
                ],
            }
            return WorkerResult(payload=payload, raw_output="batch2", duration_ms=3, cost_usd=0.2, session_id="batch-2"), "codex", "persistent", False
        raise AssertionError(f"Unexpected batch prompt: {prompt_override}")

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", quality_batch_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    batch_1 = read_json(plan_fixture.plan_dir / "execution_batch_1.json")
    execution = read_json(plan_fixture.plan_dir / "execution.json")

    assert response["success"] is True
    assert "Advisory quality: batch1.txt grew by 300 lines (threshold 200)." in batch_1["deviations"]
    assert any(deviation.startswith("Advisory quality:") for deviation in batch_1["deviations"])
    assert "Advisory quality: batch1.txt grew by 300 lines (threshold 200)." in execution["deviations"]
    assert any(
        prompt is not None and "Prior batch deviations (address if applicable):" in prompt and "[T2]" in prompt
        for prompt in seen_prompts
    )


def test_execute_quality_config_disable_suppresses_file_growth_deviation_end_to_end(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"] = [finalize_data["tasks"][0]]
    finalize_data["sense_checks"] = [finalize_data["sense_checks"][0]]
    (plan_fixture.plan_dir / "finalize.json").write_text(
        json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8"
    )

    notes_path = plan_fixture.project_dir / "notes.txt"
    _write_lines(notes_path, 10, prefix="notes_before")

    snapshots = iter([
        ({"notes.txt": "before-1"}, None),
        ({"notes.txt": "after-1"}, None),
        ({"notes.txt": "after-1"}, None),
    ])
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: next(snapshots))
    monkeypatch.setattr(
        megaplan.execution,
        "load_config",
        lambda *_: {"quality_checks": {"file_growth": {"enabled": False}}},
    )

    def single_quality_worker(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
        _write_lines(notes_path, 310, prefix="notes_after")
        payload = {
            "output": "Single batch complete.",
            "files_changed": ["notes.txt"],
            "commands_run": ["pytest -k quality-disable"],
            "deviations": [],
            "task_updates": [
                {
                    "task_id": "T1",
                    "status": "done",
                    "executor_notes": "Completed the batch with file growth disabled in config.",
                    "files_changed": ["notes.txt"],
                    "commands_run": ["pytest -k quality-disable"],
                }
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed config-disabled batch output."}
            ],
        }
        return WorkerResult(payload=payload, raw_output="single", duration_ms=1, cost_usd=0.1, session_id="single"), "codex", "persistent", False

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", single_quality_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    batch_1 = read_json(plan_fixture.plan_dir / "execution_batch_1.json")
    execution = read_json(plan_fixture.plan_dir / "execution.json")

    assert response["success"] is True
    assert not any("notes.txt grew by" in deviation for deviation in batch_1["deviations"])
    assert not any("notes.txt grew by" in deviation for deviation in execution["deviations"])
    assert not any("notes.txt grew by" in deviation for deviation in response["deviations"])


def test_batch_out_of_range_raises(
    plan_fixture: PlanFixture,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    make_args = plan_fixture.make_args
    with pytest.raises(megaplan.CliError, match="out of range"):
        megaplan.handle_execute(
            plan_fixture.root,
            make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=3),
        )


def test_batch_2_without_batch_1_raises_prerequisites(
    plan_fixture: PlanFixture,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    make_args = plan_fixture.make_args
    with pytest.raises(megaplan.CliError, match="requires batches") as exc_info:
        megaplan.handle_execute(
            plan_fixture.root,
            make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=2),
        )
    assert exc_info.value.code == "batch_prerequisites"


def test_batch_1_on_single_batch_plan_transitions_to_executed(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    # Make T2 independent of T1 so both are in batch 1
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][1]["depends_on"] = []
    (plan_fixture.plan_dir / "finalize.json").write_text(
        json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))

    def single_batch_worker(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
        payload = {
            "output": "All tasks complete.",
            "files_changed": ["batch1.py", "batch2.py"],
            "commands_run": ["pytest"],
            "deviations": [],
            "task_updates": [
                {"task_id": "T1", "status": "done", "executor_notes": "Done T1.", "files_changed": ["batch1.py"], "commands_run": ["pytest"]},
                {"task_id": "T2", "status": "done", "executor_notes": "Done T2.", "files_changed": ["batch2.py"], "commands_run": ["pytest"]},
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed."},
                {"sense_check_id": "SC2", "executor_note": "Confirmed."},
            ],
        }
        return WorkerResult(payload=payload, raw_output="all", duration_ms=1, cost_usd=0.1, session_id="single"), "codex", "persistent", False

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", single_batch_worker)

    make_args = plan_fixture.make_args
    response = megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )

    assert response["state"] == megaplan.STATE_EXECUTED
    assert response["next_step"] == "review"
    assert (plan_fixture.plan_dir / "execution_batch_1.json").exists()
    assert (plan_fixture.plan_dir / "execution.json").exists()


def test_light_batch_1_on_single_batch_plan_transitions_to_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_fixture = _make_plan_fixture_with_robustness(tmp_path, monkeypatch, robustness="light")
    _setup_two_batch_plan(plan_fixture)
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")
    finalize_data["tasks"][1]["depends_on"] = []
    (plan_fixture.plan_dir / "finalize.json").write_text(
        json.dumps(finalize_data, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))

    def single_batch_worker(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
        payload = {
            "output": "All tasks complete.",
            "files_changed": ["batch1.py", "batch2.py"],
            "commands_run": ["pytest"],
            "deviations": [],
            "task_updates": [
                {"task_id": "T1", "status": "done", "executor_notes": "Done T1.", "files_changed": ["batch1.py"], "commands_run": ["pytest"]},
                {"task_id": "T2", "status": "done", "executor_notes": "Done T2.", "files_changed": ["batch2.py"], "commands_run": ["pytest"]},
            ],
            "sense_check_acknowledgments": [
                {"sense_check_id": "SC1", "executor_note": "Confirmed."},
                {"sense_check_id": "SC2", "executor_note": "Confirmed."},
            ],
        }
        return WorkerResult(payload=payload, raw_output="all", duration_ms=1, cost_usd=0.1, session_id="single"), "codex", "persistent", False

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", single_batch_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )
    state = load_state(plan_fixture.plan_dir)
    stored_review = read_json(plan_fixture.plan_dir / "review.json")

    assert response["state"] == megaplan.STATE_DONE
    assert response["next_step"] is None
    assert state["current_state"] == megaplan.STATE_DONE
    assert "review.json" in response["artifacts"]
    assert stored_review["review_verdict"] == "approved"
    assert (plan_fixture.plan_dir / "execution_batch_1.json").exists()
    assert (plan_fixture.plan_dir / "execution.json").exists()


def test_batch_1_incomplete_tracking_returns_blocked(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))

    def incomplete_batch_worker(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
        assert prompt_override is not None
        return WorkerResult(
            payload={
                "output": "Batch one incomplete.",
                "files_changed": [],
                "commands_run": [],
                "deviations": [],
                "task_updates": [],
                "sense_check_acknowledgments": [],
            },
            raw_output="batch incomplete",
            duration_ms=1,
            cost_usd=0.0,
            session_id="batch-incomplete",
        ), "codex", "persistent", False

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", incomplete_batch_worker)

    response = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )
    state = load_state(plan_fixture.plan_dir)
    finalize_data = read_json(plan_fixture.plan_dir / "finalize.json")

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert response["summary"] == (
        "Blocked: 1/1 tasks have no executor update; 1/1 sense checks have no executor acknowledgment. "
        "Re-run execute to complete tracking."
    )
    assert finalize_data["tasks"][0]["status"] == "pending"
    assert state["history"][-1]["result"] == "blocked"


def test_review_works_after_batch_by_batch_execution(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_two_batch_plan(plan_fixture)
    monkeypatch.setattr(megaplan.execution, "_capture_git_status_snapshot", lambda *_: ({}, None))

    original_run_step = megaplan.workers.run_step_with_worker

    def _worker_dispatch(step, state, plan_dir, args, *, root=None, resolved=None, prompt_override=None):
        if step == "execute":
            return _batch_worker(step, state, plan_dir, args, root=root, resolved=resolved, prompt_override=prompt_override)
        return original_run_step(step, state, plan_dir, args, root=root, resolved=resolved, prompt_override=prompt_override)

    monkeypatch.setattr(megaplan.workers, "run_step_with_worker", _worker_dispatch)

    make_args = plan_fixture.make_args
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=1),
    )
    megaplan.handle_execute(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True, batch=2),
    )
    # Review should work - execution.json and finalize.json are in correct shape
    review = megaplan.handle_review(
        plan_fixture.root,
        make_args(plan=plan_fixture.plan_name),
    )
    assert review["state"] == megaplan.STATE_DONE
