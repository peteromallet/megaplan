import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import megaplan
import megaplan.cli


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_args_factory(project_dir: Path):
    def make_args(**overrides) -> Namespace:
        data = {
            "plan": None,
            "idea": "test idea",
            "name": "test-plan",
            "project_dir": str(project_dir),
            "max_iterations": 3,
            "budget_usd": 25.0,
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
            "reason": None,
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
    root.mkdir()
    project_dir = tmp_path / "project"
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
    plan_dir = megaplan.plans_root(root) / plan_name
    return PlanFixture(root=root, project_dir=project_dir, plan_name=plan_name, plan_dir=plan_dir, make_args=make_args)


def eval_scaffold(
    tmp_path: Path,
    *,
    iteration: int = 1,
    current_plan_text: str = "Current plan text with substantial changes for a healthy delta.\n",
    previous_plan_text: str = "Previous draft that is materially different from the current revision.\n",
    success_criteria: list[str] | None = None,
    flags: list[dict] | None = None,
    current_critique_flags: list[dict] | None = None,
    previous_critique_flags: list[dict] | None = None,
    sig_history: list[int] | None = None,
    weighted_scores: list[float] | None = None,
    total_cost_usd: float = 0.0,
    budget_usd: float = 25.0,
    max_iterations: int = 3,
    robustness: str = "standard",
) -> tuple[Path, dict]:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True)

    success_criteria = success_criteria if success_criteria is not None else ["criterion"]
    flags = flags if flags is not None else []
    sig_history = sig_history if sig_history is not None else []
    weighted_scores = weighted_scores if weighted_scores is not None else []
    current_critique_flags = current_critique_flags if current_critique_flags is not None else []
    previous_critique_flags = previous_critique_flags if previous_critique_flags is not None else []

    (plan_dir / f"plan_v{iteration}.md").write_text(current_plan_text, encoding="utf-8")
    write_json(
        plan_dir / f"plan_v{iteration}.meta.json",
        {
            "version": iteration,
            "timestamp": "2026-03-20T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": success_criteria,
            "questions": [],
            "assumptions": [],
        },
    )

    if iteration > 1:
        (plan_dir / f"plan_v{iteration - 1}.md").write_text(previous_plan_text, encoding="utf-8")
        write_json(
            plan_dir / f"critique_v{iteration - 1}.json",
            {
                "flags": previous_critique_flags,
                "verified_flag_ids": [],
                "disputed_flag_ids": [],
            },
        )

    write_json(
        plan_dir / f"critique_v{iteration}.json",
        {
            "flags": current_critique_flags,
            "verified_flag_ids": [],
            "disputed_flag_ids": [],
        },
    )
    write_json(plan_dir / "faults.json", {"flags": flags})

    state = {
        "name": "test-plan",
        "idea": "test idea",
        "current_state": megaplan.STATE_CRITIQUED,
        "iteration": iteration,
        "config": {
            "budget_usd": budget_usd,
            "max_iterations": max_iterations,
            "project_dir": str(tmp_path / "project"),
            "auto_approve": False,
            "robustness": robustness,
        },
        "plan_versions": [{"version": iteration, "file": f"plan_v{iteration}.md", "hash": "sha256:test", "timestamp": "2026-03-20T00:00:00Z"}],
        "meta": {
            "significant_counts": sig_history,
            "weighted_scores": weighted_scores,
            "total_cost_usd": total_cost_usd,
            "plan_deltas": [],
            "recurring_critiques": [],
            "overrides": [],
            "notes": [],
        },
        "history": [],
        "sessions": {},
        "last_evaluation": {},
    }
    return plan_dir, state


def load_state(plan_dir: Path) -> dict:
    return read_json(plan_dir / "state.json")


def write_flag_registry(plan_dir: Path, flags: list[dict]) -> None:
    write_json(plan_dir / "faults.json", {"flags": flags})


def advance_to_evaluated(plan_fx: PlanFixture) -> None:
    megaplan.handle_plan(plan_fx.root, plan_fx.make_args(plan=plan_fx.plan_name))
    megaplan.handle_critique(plan_fx.root, plan_fx.make_args(plan=plan_fx.plan_name))
    megaplan.handle_evaluate(plan_fx.root, plan_fx.make_args(plan=plan_fx.plan_name))


def advance_to_gated(plan_fx: PlanFixture) -> None:
    advance_to_evaluated(plan_fx)
    megaplan.handle_override(
        plan_fx.root,
        plan_fx.make_args(plan=plan_fx.plan_name, override_action="force-proceed", reason="test gate override"),
    )


def test_init_creates_state_file(plan_fixture: PlanFixture) -> None:
    state = load_state(plan_fixture.plan_dir)
    assert (plan_fixture.plan_dir / "state.json").exists()
    assert state["current_state"] == megaplan.STATE_INITIALIZED
    assert state["iteration"] == 0


def test_init_returns_clarify_as_next_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(
        megaplan.cli.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )

    response = megaplan.handle_init(root, make_args_factory(project_dir)())
    assert response["next_step"] == "clarify"


def test_init_persists_auto_approve_and_robustness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(
        megaplan.cli.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )

    response = megaplan.handle_init(
        root,
        make_args_factory(project_dir)(auto_approve=True, robustness="thorough"),
    )
    state = load_state(megaplan.plans_root(root) / response["plan"])

    assert response["auto_approve"] is True
    assert response["robustness"] == "thorough"
    assert state["config"]["auto_approve"] is True
    assert state["config"]["robustness"] == "thorough"


def test_slugify() -> None:
    assert megaplan.slugify("Hello World!") == "hello-world"
    assert megaplan.slugify("") == "plan"
    # Word-boundary truncation
    assert megaplan.slugify("Local-First Reigh — Run Entirely On Your Own Machine") == "local-first-reigh-run"
    # Short ideas pass through unchanged
    assert megaplan.slugify("fix-bug") == "fix-bug"
    # Custom max_length
    assert megaplan.slugify("one-two-three-four-five", max_length=15) == "one-two-three"
    # Only special chars
    assert megaplan.slugify("!!!") == "plan"


def test_compute_plan_delta_percent() -> None:
    assert megaplan.compute_plan_delta_percent("same text", "same text") == 0.0
    assert megaplan.compute_plan_delta_percent(None, "current text") is None
    assert megaplan.compute_plan_delta_percent("abc", "xyz") == 100.0


def test_compute_recurring_critiques(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    write_json(
        plan_dir / "critique_v1.json",
        {
            "flags": [
                {"id": "FLAG-001", "concern": "  Same concern  ", "category": "other", "evidence": "v1"},
                {"id": "FLAG-002", "concern": "Only in first", "category": "other", "evidence": "v1"},
            ],
            "verified_flag_ids": [],
            "disputed_flag_ids": [],
        },
    )
    write_json(
        plan_dir / "critique_v2.json",
        {
            "flags": [
                {"id": "FLAG-003", "concern": "same   concern", "category": "other", "evidence": "v2"},
                {"id": "FLAG-004", "concern": "Only in second", "category": "other", "evidence": "v2"},
            ],
            "verified_flag_ids": [],
            "disputed_flag_ids": [],
        },
    )

    assert megaplan.compute_recurring_critiques(plan_dir, 2) == ["same concern"]

    write_json(
        plan_dir / "critique_v2.json",
        {
            "flags": [{"id": "FLAG-005", "concern": "Different concern", "category": "other", "evidence": "v2"}],
            "verified_flag_ids": [],
            "disputed_flag_ids": [],
        },
    )
    assert megaplan.compute_recurring_critiques(plan_dir, 2) == []


@pytest.mark.parametrize(
    ("current_state", "last_evaluation", "expected"),
    [
        (megaplan.STATE_INITIALIZED, {}, ["clarify"]),
        (megaplan.STATE_CLARIFIED, {}, ["clarify", "plan"]),
        (megaplan.STATE_PLANNED, {}, ["critique"]),
        (megaplan.STATE_CRITIQUED, {}, ["evaluate"]),
        (megaplan.STATE_GATED, {}, ["execute"]),
        (megaplan.STATE_EXECUTED, {}, ["review"]),
        (megaplan.STATE_DONE, {}, []),
        (megaplan.STATE_ABORTED, {}, []),
    ],
)
def test_infer_next_steps_non_evaluated_states(current_state: str, last_evaluation: dict, expected: list[str]) -> None:
    assert megaplan.infer_next_steps({"current_state": current_state, "last_evaluation": last_evaluation}) == expected


@pytest.mark.parametrize(
    ("recommendation", "expected"),
    [
        ("CONTINUE", ["integrate", "gate"]),
        ("SKIP", ["gate"]),
        ("ESCALATE", ["override add-note", "override force-proceed", "override abort"]),
        ("ABORT", ["override add-note", "override force-proceed", "override abort"]),
        (None, ["override add-note", "override abort"]),
    ],
)
def test_infer_next_steps_evaluated_recommendations(recommendation: str | None, expected: list[str]) -> None:
    state = {"current_state": megaplan.STATE_EVALUATED, "last_evaluation": {"recommendation": recommendation}}
    assert megaplan.infer_next_steps(state) == expected


def test_update_flags_after_critique_creates_flags(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    registry = megaplan.update_flags_after_critique(
        plan_dir,
        {
            "flags": [
                {"id": "FLAG-001", "concern": "First concern", "category": "correctness", "severity_hint": "likely-significant", "evidence": "proof"},
                {"id": "FLAG-002", "concern": "Second concern", "category": "other", "severity_hint": "likely-minor", "evidence": "proof"},
            ]
        },
        iteration=1,
    )

    assert [flag["id"] for flag in registry["flags"]] == ["FLAG-001", "FLAG-002"]
    assert all(flag["status"] == "open" for flag in registry["flags"])
    assert registry["flags"][0]["severity"] == "significant"
    assert registry["flags"][1]["severity"] == "minor"
    assert all(flag["raised_in"] == "critique_v1.json" for flag in registry["flags"])


def test_update_flags_after_critique_verifies_existing(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    write_flag_registry(
        plan_dir,
        [{"id": "FLAG-001", "concern": "Concern", "status": "open", "severity": None, "verified": False}],
    )

    registry = megaplan.update_flags_after_critique(
        plan_dir,
        {"flags": [], "verified_flag_ids": ["FLAG-001"], "disputed_flag_ids": []},
        iteration=2,
    )

    flag = registry["flags"][0]
    assert flag["status"] == "verified"
    assert flag["verified"] is True
    assert flag["verified_in"] == "critique_v2.json"


def test_update_flags_after_critique_disputes_flags(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    write_flag_registry(
        plan_dir,
        [{"id": "FLAG-001", "concern": "Concern", "status": "open", "severity": None, "verified": False}],
    )

    registry = megaplan.update_flags_after_critique(
        plan_dir,
        {"flags": [], "verified_flag_ids": [], "disputed_flag_ids": ["FLAG-001"]},
        iteration=2,
    )

    assert registry["flags"][0]["status"] == "disputed"


def test_update_flags_after_critique_reuses_ids(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    write_flag_registry(
        plan_dir,
        [{"id": "FLAG-001", "concern": "Old concern", "status": "verified", "severity": "significant", "verified": True}],
    )

    registry = megaplan.update_flags_after_critique(
        plan_dir,
        {"flags": [{"id": "FLAG-001", "concern": "Updated concern", "category": "security", "severity_hint": "likely-significant", "evidence": "new evidence"}]},
        iteration=3,
    )

    assert len(registry["flags"]) == 1
    flag = registry["flags"][0]
    assert flag["id"] == "FLAG-001"
    assert flag["concern"] == "Updated concern"
    assert flag["status"] == "open"
    assert flag["severity"] == "significant"
    assert flag["raised_in"] == "critique_v3.json"


def test_update_flags_after_critique_autonumbers(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()

    registry = megaplan.update_flags_after_critique(
        plan_dir,
        {
            "flags": [
                {"id": "", "concern": "First", "category": "other", "evidence": "one"},
                {"id": "FLAG-000", "concern": "Second", "category": "other", "evidence": "two"},
            ]
        },
        iteration=1,
    )

    assert [flag["id"] for flag in registry["flags"]] == ["FLAG-001", "FLAG-002"]


def test_update_flags_after_critique_sets_severity_from_hint(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    registry = megaplan.update_flags_after_critique(
        plan_dir,
        {
            "flags": [
                {"id": "FLAG-001", "concern": "Real risk", "category": "correctness", "severity_hint": "likely-significant", "evidence": "proof"},
                {"id": "FLAG-002", "concern": "Cosmetic", "category": "other", "severity_hint": "likely-minor", "evidence": "small"},
                {"id": "FLAG-003", "concern": "Unclear", "category": "other", "severity_hint": "uncertain", "evidence": "maybe"},
            ]
        },
        iteration=1,
    )

    by_id = {flag["id"]: flag for flag in registry["flags"]}
    assert by_id["FLAG-001"]["severity"] == "significant"
    assert by_id["FLAG-002"]["severity"] == "minor"
    assert by_id["FLAG-003"]["severity"] == "significant"  # uncertain defaults to significant


def test_update_flags_after_integrate_marks_addressed(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    write_flag_registry(
        plan_dir,
        [
            {"id": "FLAG-001", "concern": "First", "status": "open", "severity": "significant", "verified": False},
            {"id": "FLAG-002", "concern": "Second", "status": "open", "severity": "significant", "verified": False},
        ],
    )

    registry = megaplan.update_flags_after_integrate(
        plan_dir,
        ["FLAG-001"],
        plan_file="plan_v2.md",
        summary="Addressed the first issue",
    )

    by_id = {flag["id"]: flag for flag in registry["flags"]}
    assert by_id["FLAG-001"]["status"] == "addressed"
    assert by_id["FLAG-001"]["addressed_in"] == "plan_v2.md"
    assert by_id["FLAG-001"]["evidence"] == "Addressed the first issue"
    assert by_id["FLAG-002"]["status"] == "open"


def test_unresolved_significant_flags_filtering() -> None:
    registry = {
        "flags": [
            {"id": "FLAG-001", "status": "open", "severity": "significant"},
            {"id": "FLAG-002", "status": "open", "severity": "minor"},
            {"id": "FLAG-003", "status": "addressed", "severity": "significant"},
            {"id": "FLAG-004", "status": "verified", "severity": "significant"},
            {"id": "FLAG-005", "status": "disputed", "severity": "significant"},
        ]
    }

    assert [flag["id"] for flag in megaplan.unresolved_significant_flags(registry)] == ["FLAG-001", "FLAG-005"]


def test_normalize_flag_record_defaults() -> None:
    normalized = megaplan.normalize_flag_record({"id": "FLAG-001", "concern": "Concern", "category": "bogus", "evidence": "proof"}, "FLAG-999")
    assert normalized["category"] == "other"
    assert normalized["severity_hint"] == "uncertain"


def test_eval_abort_over_budget(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
        total_cost_usd=30.0,
        budget_usd=25.0,
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "ABORT"
    assert evaluation["confidence"] == "high"


def test_eval_skip_no_significant_flags(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(tmp_path, flags=[])
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "SKIP"
    assert evaluation["confidence"] == "high"


def test_eval_escalate_stagnant_with_unresolved(tmp_path: Path) -> None:
    previous = ("same plan with lots of stable context " * 8) + "A\n"
    current = ("same plan with lots of stable context " * 8) + "B\n"
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        current_plan_text=current,
        previous_plan_text=previous,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["signals"]["plan_delta_from_previous"] < 5.0
    assert evaluation["recommendation"] == "ESCALATE"


def test_eval_skip_small_delta_all_resolved(tmp_path: Path) -> None:
    previous = ("same plan with lots of stable context " * 8) + "A\n"
    current = ("same plan with lots of stable context " * 8) + "B\n"
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        current_plan_text=current,
        previous_plan_text=previous,
        flags=[{"id": "FLAG-001", "status": "addressed", "severity": "significant"}],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["signals"]["plan_delta_from_previous"] < 5.0
    assert evaluation["recommendation"] == "SKIP"


def test_eval_continue_first_iteration_significant(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=1,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "CONTINUE"
    assert evaluation["confidence"] == "high"


def test_eval_escalate_recurring_critiques(tmp_path: Path) -> None:
    concern = {"id": "FLAG-001", "concern": "Same concern", "category": "other", "evidence": "proof"}
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
        current_critique_flags=[concern],
        previous_critique_flags=[{"id": "FLAG-001", "concern": " same   concern ", "category": "other", "evidence": "proof"}],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "ESCALATE"
    assert evaluation["confidence"] == "high"


def test_eval_escalate_weighted_score_not_improving(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant", "category": "correctness", "concern": "real issue"}],
        weighted_scores=[2.0],  # current weight 2.0 >= 2.0 * 0.9 = 1.8 → stagnant
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "ESCALATE"
    assert evaluation["confidence"] == "medium"


def test_eval_skip_low_weight_improving(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant", "category": "other", "concern": "placeholder column name"}],
        weighted_scores=[4.0],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    # weight 0.5 < 2.0 threshold and < 4.0 → "good enough", executor can handle
    assert evaluation["recommendation"] == "SKIP"
    assert evaluation["confidence"] == "medium"


def test_eval_continue_weighted_score_improving_above_threshold(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[
            {"id": "FLAG-001", "status": "open", "severity": "significant", "category": "correctness", "concern": "wrong API endpoint"},
            {"id": "FLAG-002", "status": "open", "severity": "significant", "category": "completeness", "concern": "missing error handling"},
        ],
        weighted_scores=[6.0],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    # weight 2.0 + 1.5 = 3.5 >= 2.0 threshold → still CONTINUE
    assert evaluation["recommendation"] == "CONTINUE"
    assert evaluation["confidence"] == "medium"


def test_eval_thresholds_change_with_robustness(tmp_path: Path) -> None:
    light_plan_dir, light_state = eval_scaffold(
        tmp_path / "light",
        iteration=2,
        flags=[
            {"id": "FLAG-001", "status": "open", "severity": "significant", "category": "security", "concern": "security risk"},
        ],
        weighted_scores=[5.0],
        robustness="light",
    )
    light_evaluation = megaplan.build_evaluation(light_plan_dir, light_state)

    thorough_plan_dir, thorough_state = eval_scaffold(
        tmp_path / "thorough",
        iteration=2,
        flags=[
            {"id": "FLAG-001", "status": "open", "severity": "significant", "category": "performance", "concern": "slow path"},
            {"id": "FLAG-002", "status": "open", "severity": "significant", "category": "maintainability", "concern": "duplicate branch"},
        ],
        weighted_scores=[3.0],
        robustness="thorough",
    )
    thorough_evaluation = megaplan.build_evaluation(thorough_plan_dir, thorough_state)

    assert light_evaluation["robustness"] == "light"
    assert light_evaluation["recommendation"] == "SKIP"
    assert thorough_evaluation["robustness"] == "thorough"
    assert thorough_evaluation["recommendation"] == "CONTINUE"


def test_eval_surfaces_scope_creep_warning(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[
            {
                "id": "FLAG-007",
                "status": "open",
                "severity": "significant",
                "category": "maintainability",
                "concern": "Scope creep: the plan now adds a broad refactor beyond the original idea",
                "evidence": "The new steps include unrelated cleanup work.",
            },
        ],
        weighted_scores=[2.0],
    )

    evaluation = megaplan.build_evaluation(plan_dir, state)

    assert evaluation["signals"]["scope_creep_flags"] == ["FLAG-007"]
    assert "Scope creep detected" in evaluation["warnings"][0]


def test_eval_escalate_max_iterations_with_unresolved(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=3,
        max_iterations=3,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
        sig_history=[2],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "ESCALATE"
    assert evaluation["confidence"] == "high"


def test_full_mock_lifecycle(plan_fixture: PlanFixture) -> None:
    root = plan_fixture.root
    name = plan_fixture.plan_name
    make_args = plan_fixture.make_args

    clarify_result = megaplan.handle_clarify(root, make_args(plan=name))
    assert clarify_result["state"] == megaplan.STATE_CLARIFIED
    assert (plan_fixture.plan_dir / "clarify.json").exists()

    plan_result = megaplan.handle_plan(root, make_args(plan=name))
    assert plan_result["state"] == megaplan.STATE_PLANNED
    assert plan_result["iteration"] == 1
    assert (plan_fixture.plan_dir / "plan_v1.md").exists()

    critique_result = megaplan.handle_critique(root, make_args(plan=name))
    assert critique_result["state"] == megaplan.STATE_CRITIQUED
    registry = read_json(plan_fixture.plan_dir / "faults.json")
    assert len(registry["flags"]) == 2
    # Severity is now set directly from severity_hint (no triage step)
    assert all(flag["severity"] == "significant" for flag in registry["flags"])

    evaluate_result = megaplan.handle_evaluate(root, make_args(plan=name))
    assert evaluate_result["state"] == megaplan.STATE_EVALUATED
    assert evaluate_result["recommendation"] == "CONTINUE"

    integrate_result = megaplan.handle_integrate(root, make_args(plan=name))
    assert integrate_result["state"] == megaplan.STATE_PLANNED
    assert integrate_result["iteration"] == 2
    registry = read_json(plan_fixture.plan_dir / "faults.json")
    assert {flag["status"] for flag in registry["flags"]} == {"addressed"}

    megaplan.handle_critique(root, make_args(plan=name))
    registry = read_json(plan_fixture.plan_dir / "faults.json")
    assert {flag["status"] for flag in registry["flags"]} == {"verified"}

    evaluate_result = megaplan.handle_evaluate(root, make_args(plan=name))
    assert evaluate_result["recommendation"] == "SKIP"

    gate_result = megaplan.handle_gate(root, make_args(plan=name))
    assert gate_result["state"] == megaplan.STATE_GATED
    assert (plan_fixture.plan_dir / "gate.json").exists()
    assert (plan_fixture.plan_dir / "final.md").exists()

    execute_result = megaplan.handle_execute(root, make_args(plan=name, confirm_destructive=True, user_approved=True))
    assert execute_result["state"] == megaplan.STATE_EXECUTED
    assert (plan_fixture.project_dir / "IMPLEMENTED_BY_MEGAPLAN.txt").exists()

    review_result = megaplan.handle_review(root, make_args(plan=name))
    assert review_result["state"] == megaplan.STATE_DONE
    review = read_json(plan_fixture.plan_dir / "review.json")
    assert all(item["pass"] for item in review["criteria"])


def test_require_state_rejects_invalid_transition(plan_fixture: PlanFixture) -> None:
    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    assert exc_info.value.code == "invalid_transition"
    assert "clarify" in exc_info.value.valid_next


def test_clarify_transitions_to_clarified(plan_fixture: PlanFixture) -> None:
    result = megaplan.handle_clarify(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    assert result["state"] == megaplan.STATE_CLARIFIED
    assert (plan_fixture.plan_dir / "clarify.json").exists()
    assert state["current_state"] == megaplan.STATE_CLARIFIED
    assert set(state["clarification"]) == {"refined_idea", "intent_summary", "questions"}


def test_clarify_requires_initialized(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_clarify(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    assert exc_info.value.code == "invalid_transition"


def test_plan_accepts_both_initialized_and_clarified(plan_fixture: PlanFixture) -> None:
    initialized_result = megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    assert initialized_result["state"] == megaplan.STATE_PLANNED

    second_init = megaplan.handle_init(
        plan_fixture.root,
        plan_fixture.make_args(name="clarified-plan", idea="clarified idea"),
    )
    second_name = second_init["plan"]
    megaplan.handle_clarify(plan_fixture.root, plan_fixture.make_args(plan=second_name, idea="clarified idea"))
    clarified_result = megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=second_name, idea="clarified idea"))
    assert clarified_result["state"] == megaplan.STATE_PLANNED


def test_plan_prompt_includes_refined_idea(plan_fixture: PlanFixture) -> None:
    megaplan.handle_clarify(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    prompt = megaplan.cli.create_claude_prompt("plan", state, plan_fixture.plan_dir)
    assert "Refined: test idea" in prompt


def test_intent_in_critique_prompt(plan_fixture: PlanFixture) -> None:
    megaplan.handle_clarify(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    prompt = megaplan.cli.create_codex_prompt("critique", state, plan_fixture.plan_dir)
    assert "The user wants to test idea." in prompt


def test_notes_in_critique_prompt(plan_fixture: PlanFixture) -> None:
    megaplan.handle_clarify(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="use existing repository patterns"),
    )
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    state = load_state(plan_fixture.plan_dir)
    prompt = megaplan.cli.create_codex_prompt("critique", state, plan_fixture.plan_dir)
    assert "use existing repository patterns" in prompt


def test_critique_prompt_includes_robustness_guidance(plan_fixture: PlanFixture) -> None:
    state = load_state(plan_fixture.plan_dir)
    state["config"]["robustness"] = "thorough"
    megaplan.cli.save_state(plan_fixture.plan_dir, state)

    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    state = load_state(plan_fixture.plan_dir)
    prompt = megaplan.cli.create_codex_prompt("critique", state, plan_fixture.plan_dir)

    assert "Robustness level: thorough." in prompt
    assert "Be exhaustive." in prompt


def test_critique_response_surfaces_scope_creep_warning(
    plan_fixture: PlanFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))
    original_mock = megaplan.cli.mock_worker_output

    def mock_with_scope_creep(step: str, state: dict, plan_dir: Path):
        if step == "critique":
            payload = {
                "flags": [
                    {
                        "id": "FLAG-099",
                        "concern": "Scope creep: the plan adds unrelated cleanup work beyond the original idea",
                        "category": "maintainability",
                        "severity_hint": "likely-significant",
                        "evidence": "A repository-wide refactor was added without user input.",
                    }
                ],
                "verified_flag_ids": [],
                "disputed_flag_ids": [],
            }
            return megaplan.cli.WorkerResult(
                payload=payload,
                raw_output=json.dumps(payload),
                duration_ms=10,
                cost_usd=0.0,
                session_id="scope-creep",
            )
        return original_mock(step, state, plan_dir)

    monkeypatch.setattr(megaplan.cli, "mock_worker_output", mock_with_scope_creep)

    critique_result = megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    assert critique_result["scope_creep_flags"] == ["FLAG-099"]
    assert "Scope creep detected" in critique_result["warnings"][0]


def test_cannot_execute_before_gate(plan_fixture: PlanFixture) -> None:
    advance_to_evaluated(plan_fixture)

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_execute(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True))

    assert exc_info.value.code == "invalid_transition"


def test_terminal_states_block_progression_steps(plan_fixture: PlanFixture) -> None:
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="abort", reason="stop here"),
    )
    state = load_state(plan_fixture.plan_dir)
    assert megaplan.infer_next_steps(state) == []

    for action in (
        lambda: megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name)),
        lambda: megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name)),
        lambda: megaplan.handle_execute(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True)),
    ):
        with pytest.raises(megaplan.CliError):
            action()

    result = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="still documenting"),
    )
    assert result["success"] is True


def test_override_abort_sets_terminal(plan_fixture: PlanFixture) -> None:
    result = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="abort", reason="manual stop"),
    )

    state = load_state(plan_fixture.plan_dir)
    assert result["state"] == megaplan.STATE_ABORTED
    assert state["current_state"] == megaplan.STATE_ABORTED
    assert state["meta"]["overrides"][-1]["action"] == "abort"
    assert state["meta"]["overrides"][-1]["reason"] == "manual stop"


def test_override_add_note(plan_fixture: PlanFixture) -> None:
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="remember this"),
    )

    state = load_state(plan_fixture.plan_dir)
    assert state["meta"]["notes"][-1]["note"] == "remember this"
    assert state["meta"]["overrides"][-1]["action"] == "add-note"
    assert state["meta"]["overrides"][-1]["note"] == "remember this"


def test_override_add_note_after_abort(plan_fixture: PlanFixture) -> None:
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="abort", reason="stop"),
    )

    result = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="add-note", note="after abort"),
    )

    state = load_state(plan_fixture.plan_dir)
    assert result["success"] is True
    assert state["current_state"] == megaplan.STATE_ABORTED
    assert state["meta"]["notes"][-1]["note"] == "after abort"


def test_override_force_proceed_success(plan_fixture: PlanFixture) -> None:
    advance_to_evaluated(plan_fixture)

    result = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="accept risk"),
    )

    state = load_state(plan_fixture.plan_dir)
    assert result["state"] == megaplan.STATE_GATED
    assert state["current_state"] == megaplan.STATE_GATED
    assert (plan_fixture.plan_dir / "gate.json").exists()
    assert (plan_fixture.plan_dir / "final.md").exists()
    assert state["meta"]["overrides"][-1]["action"] == "force-proceed"


def test_override_force_proceed_unsafe_missing_project_dir(plan_fixture: PlanFixture) -> None:
    advance_to_evaluated(plan_fixture)
    state = load_state(plan_fixture.plan_dir)
    state["config"]["project_dir"] = str(plan_fixture.project_dir / "missing")
    write_json(plan_fixture.plan_dir / "state.json", state)

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_override(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="unsafe"),
        )

    assert exc_info.value.code == "unsafe_override"


def test_override_force_proceed_unsafe_no_success_criteria(plan_fixture: PlanFixture) -> None:
    advance_to_evaluated(plan_fixture)
    meta_path = plan_fixture.plan_dir / "plan_v1.meta.json"
    meta = read_json(meta_path)
    meta["success_criteria"] = []
    write_json(meta_path, meta)

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_override(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="unsafe"),
        )

    assert exc_info.value.code == "unsafe_override"


def test_override_force_proceed_wrong_state(plan_fixture: PlanFixture) -> None:
    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=plan_fixture.plan_name))

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_override(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="force-proceed", reason="wrong state"),
        )

    assert exc_info.value.code == "invalid_transition"


def test_override_skip(plan_fixture: PlanFixture) -> None:
    advance_to_evaluated(plan_fixture)
    result = megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, override_action="skip", reason="move on"),
    )

    state = load_state(plan_fixture.plan_dir)
    assert result["state"] == megaplan.STATE_EVALUATED
    assert state["current_state"] == megaplan.STATE_EVALUATED
    assert state["last_evaluation"]["recommendation"] == "SKIP"


def test_execute_requires_confirm_destructive(plan_fixture: PlanFixture) -> None:
    advance_to_gated(plan_fixture)

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_execute(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=False),
        )

    assert exc_info.value.code == "missing_confirmation"


def test_gate_response_surfaces_auto_approve(plan_fixture: PlanFixture) -> None:
    init_result = megaplan.handle_init(
        plan_fixture.root,
        plan_fixture.make_args(name="auto-approve-plan", idea="auto approve idea", auto_approve=True),
    )
    name = init_result["plan"]

    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))
    megaplan.handle_evaluate(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))
    megaplan.handle_integrate(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))
    megaplan.handle_evaluate(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))
    gate_result = megaplan.handle_gate(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto approve idea"))

    assert gate_result["auto_approve"] is True


def test_execute_requires_user_approval_in_review_mode(plan_fixture: PlanFixture) -> None:
    advance_to_gated(plan_fixture)

    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_execute(
            plan_fixture.root,
            plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=False),
        )

    assert exc_info.value.code == "missing_approval"


def test_execute_succeeds_with_user_approval_in_review_mode(plan_fixture: PlanFixture) -> None:
    advance_to_gated(plan_fixture)

    result = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=plan_fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    state = load_state(plan_fixture.plan_dir)

    assert result["state"] == megaplan.STATE_EXECUTED
    assert result["user_approved_gate"] is True
    assert state["meta"]["user_approved_gate"] is True


def test_execute_succeeds_without_user_approval_in_auto_approve_mode(plan_fixture: PlanFixture) -> None:
    init_result = megaplan.handle_init(
        plan_fixture.root,
        plan_fixture.make_args(name="auto-exec-plan", idea="auto execute idea", auto_approve=True),
    )
    name = init_result["plan"]

    megaplan.handle_plan(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto execute idea"))
    megaplan.handle_critique(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto execute idea"))
    megaplan.handle_evaluate(plan_fixture.root, plan_fixture.make_args(plan=name, idea="auto execute idea"))
    megaplan.handle_override(
        plan_fixture.root,
        plan_fixture.make_args(plan=name, idea="auto execute idea", override_action="force-proceed", reason="test gate"),
    )

    result = megaplan.handle_execute(
        plan_fixture.root,
        plan_fixture.make_args(plan=name, idea="auto execute idea", confirm_destructive=True, user_approved=False),
    )

    assert result["state"] == megaplan.STATE_EXECUTED
    assert result["auto_approve"] is True


def test_flag_weight_security() -> None:
    assert megaplan.flag_weight({"category": "security", "concern": "SQL injection"}) == 3.0


def test_flag_weight_implementation_detail() -> None:
    assert megaplan.flag_weight({"category": "correctness", "concern": "The column name in the pseudocode is wrong"}) == 0.5
    assert megaplan.flag_weight({"category": "correctness", "concern": "Placeholder SQL should use real table"}) == 0.5


def test_flag_weight_defaults() -> None:
    assert megaplan.flag_weight({"category": "correctness", "concern": "Architecture is wrong"}) == 2.0
    assert megaplan.flag_weight({"category": "completeness", "concern": "Missing error handling"}) == 1.5
    assert megaplan.flag_weight({"category": "maintainability", "concern": "Code duplication"}) == 0.75
    assert megaplan.flag_weight({"category": "other", "concern": "Something"}) == 1.0


def test_eval_weighted_implementation_details_dont_escalate(tmp_path: Path) -> None:
    """3 implementation-detail flags (weight 0.5 each = 1.5) should not escalate even with flat count."""
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[
            {"id": "FLAG-001", "status": "open", "severity": "significant", "category": "correctness", "concern": "column name is wrong"},
            {"id": "FLAG-002", "status": "open", "severity": "significant", "category": "correctness", "concern": "placeholder SQL"},
            {"id": "FLAG-003", "status": "open", "severity": "significant", "category": "correctness", "concern": "schema field mismatch"},
        ],
        weighted_scores=[1.5],  # same as current → but within 0.9 tolerance since 1.5 >= 1.35
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    # All flags are low-weight (0.5 each = 1.5), so override should suggest force-proceed
    assert evaluation["recommendation"] == "ESCALATE"
    assert evaluation["suggested_override"] == "force-proceed"


def test_eval_weighted_security_flags_escalate(tmp_path: Path) -> None:
    """2 security flags (weight 3.0 each = 6.0) should escalate and suggest add-note."""
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=2,
        flags=[
            {"id": "FLAG-001", "status": "open", "severity": "significant", "category": "security", "concern": "SQL injection risk"},
            {"id": "FLAG-002", "status": "open", "severity": "significant", "category": "security", "concern": "XSS vulnerability"},
        ],
        weighted_scores=[6.0],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "ESCALATE"
    assert evaluation["suggested_override"] == "add-note"


def test_eval_override_guidance_abort_on_budget(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
        total_cost_usd=30.0,
        budget_usd=25.0,
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "ABORT"
    assert evaluation["suggested_override"] == "abort"


def test_eval_no_override_on_continue(tmp_path: Path) -> None:
    plan_dir, state = eval_scaffold(
        tmp_path,
        iteration=1,
        flags=[{"id": "FLAG-001", "status": "open", "severity": "significant"}],
    )
    evaluation = megaplan.build_evaluation(plan_dir, state)
    assert evaluation["recommendation"] == "CONTINUE"
    assert "suggested_override" not in evaluation


# ── Global setup tests ──────────────────────────────────────────────


def test_global_setup_creates_files_for_detected_agents(tmp_path: Path) -> None:
    """Fresh install creates files only for agents whose config dir exists."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    # .cursor not created → should be skipped

    result = megaplan.handle_setup_global(force=False, home=home)
    assert result["success"] is True
    assert result["mode"] == "global"

    by_agent = {e["agent"]: e for e in result["installed"]}
    assert not by_agent["claude"]["skipped"]
    assert not by_agent["codex"]["skipped"]
    assert by_agent["cursor"]["skipped"]
    assert by_agent["cursor"]["reason"] == "not installed"

    assert (home / ".claude" / "skills" / "megaplan" / "SKILL.md").exists()
    assert (home / ".codex" / "skills" / "megaplan" / "SKILL.md").exists()
    assert not (home / ".cursor" / "rules" / "megaplan.mdc").exists()


def test_global_setup_skips_not_installed(tmp_path: Path) -> None:
    """Agents whose config dir doesn't exist are skipped with reason."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)

    result = megaplan.handle_setup_global(force=False, home=home)
    by_agent = {e["agent"]: e for e in result["installed"]}

    assert by_agent["codex"]["skipped"]
    assert by_agent["codex"]["reason"] == "not installed"
    assert by_agent["cursor"]["skipped"]
    assert by_agent["cursor"]["reason"] == "not installed"


def test_global_setup_idempotent(tmp_path: Path) -> None:
    """Running twice skips all files on the second run."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    (home / ".cursor").mkdir(parents=True)

    first = megaplan.handle_setup_global(force=False, home=home)
    assert all(not e["skipped"] for e in first["installed"])

    second = megaplan.handle_setup_global(force=False, home=home)
    assert all(e["skipped"] for e in second["installed"])


def test_global_setup_force_overwrites(tmp_path: Path) -> None:
    """--force overwrites even when content matches."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)

    megaplan.handle_setup_global(force=False, home=home)
    result = megaplan.handle_setup_global(force=True, home=home)

    by_agent = {e["agent"]: e for e in result["installed"]}
    assert not by_agent["claude"]["skipped"]
    assert by_agent["claude"]["existed"]


def test_global_setup_creates_child_directories(tmp_path: Path) -> None:
    """Subdirectories like skills/megaplan/ are created automatically."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)

    result = megaplan.handle_setup_global(force=False, home=home)
    assert result["success"] is True
    assert (home / ".claude" / "skills" / "megaplan" / "SKILL.md").is_file()


def test_global_setup_zero_agents_detected(tmp_path: Path) -> None:
    """If no agent dirs exist, returns success=False."""
    home = tmp_path / "home"
    home.mkdir()

    result = megaplan.handle_setup_global(force=False, home=home)
    assert result["success"] is False
    assert all(e["skipped"] for e in result["installed"])


# ── Config I/O tests ─────────────────────────────────────────────────


def test_load_config_missing_file(tmp_path: Path) -> None:
    """load_config returns {} when no config file exists."""
    home = tmp_path / "home"
    home.mkdir()
    assert megaplan.load_config(home) == {}


def test_save_and_load_config_roundtrip(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config = {"agents": {"plan": "codex", "critique": "claude"}}
    megaplan.save_config(config, home)
    loaded = megaplan.load_config(home)
    assert loaded == config


def test_load_config_corrupt_json(tmp_path: Path) -> None:
    """load_config returns {} when config.json is corrupt."""
    home = tmp_path / "home"
    config_path = home / ".config" / "megaplan" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("not valid json{{{", encoding="utf-8")
    assert megaplan.load_config(home) == {}


def test_config_dir_respects_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert megaplan.config_dir() == xdg / "megaplan"


# ── Agent resolution tests ───────────────────────────────────────────


def test_resolve_agent_cli_flag_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--agent codex for plan step overrides config and defaults."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: "/usr/bin/mock" if name == "codex" else None)
    config = {"agents": {"plan": "claude"}}
    megaplan.save_config(config, home)
    args = Namespace(agent="codex", ephemeral=False, fresh=False, persist=False, confirm_self_review=False)
    agent, _mode, _refreshed = megaplan.cli.resolve_agent_mode("plan", args, home=home)
    assert agent == "codex"


def test_resolve_agent_config_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config says plan=codex, no CLI flag → uses codex."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None)
    config = {"agents": {"plan": "codex"}}
    megaplan.save_config(config, home)
    args = Namespace(agent=None, ephemeral=False, fresh=False, persist=False, confirm_self_review=False)
    agent, _mode, _refreshed = megaplan.cli.resolve_agent_mode("plan", args, home=home)
    assert agent == "codex"


def test_resolve_agent_fallback_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When default agent is missing, falls back to available one."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: "/usr/bin/mock" if name == "claude" else None)
    args = Namespace(agent=None, ephemeral=False, fresh=False, persist=False, confirm_self_review=False)
    agent, _mode, _refreshed = megaplan.cli.resolve_agent_mode("critique", args, home=home)
    assert agent == "claude"
    assert args._agent_fallback["requested"] == "codex"
    assert args._agent_fallback["resolved"] == "claude"


def test_resolve_agent_explicit_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """--agent codex but codex not on PATH → CliError."""
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: None)
    args = Namespace(agent="codex", ephemeral=False, fresh=False, persist=False, confirm_self_review=False)
    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.cli.resolve_agent_mode("plan", args)
    assert exc_info.value.code == "agent_not_found"


def test_resolve_agent_no_agents_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing on PATH → CliError."""
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: None)
    args = Namespace(agent=None, ephemeral=False, fresh=False, persist=False, confirm_self_review=False)
    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.cli.resolve_agent_mode("plan", args)
    assert exc_info.value.code == "agent_not_found"
    assert "No supported agents" in exc_info.value.message


# ── Setup global config writing tests ────────────────────────────────


def test_setup_global_writes_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After handle_setup_global, config.json exists with correct routing."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None)

    result = megaplan.handle_setup_global(force=False, home=home)
    assert "config_path" in result
    assert "routing" in result
    config = megaplan.load_config(home)
    assert config["agents"]["plan"] == "claude"
    assert config["agents"]["critique"] == "codex"
    assert config["agents"]["execute"] == "codex"


def test_setup_global_only_claude_routes_all_to_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only claude available → all steps route to claude."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(megaplan.cli.shutil, "which", lambda name: "/usr/bin/mock" if name == "claude" else None)

    result = megaplan.handle_setup_global(force=False, home=home)
    config = megaplan.load_config(home)
    assert all(agent == "claude" for agent in config["agents"].values())
    assert result["routing"] == config["agents"]


# ── Config subcommand tests ──────────────────────────────────────────


def test_config_show_returns_effective_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    config = {"agents": {"plan": "codex"}}
    megaplan.save_config(config)
    args = Namespace(config_action="show")
    result = megaplan.handle_config(args)
    assert result["routing"]["plan"] == "codex"
    assert result["routing"]["critique"] == "codex"  # not in config, falls back to DEFAULT_AGENT_ROUTING


def test_config_set_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    args = Namespace(config_action="set", key="agents.plan", value="codex")
    result = megaplan.handle_config(args)
    assert result["success"] is True
    config = megaplan.load_config()
    assert config["agents"]["plan"] == "codex"


def test_config_set_invalid_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    args = Namespace(config_action="set", key="agents.bogus", value="claude")
    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.handle_config(args)
    assert exc_info.value.code == "invalid_args"


def test_config_reset_removes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    megaplan.save_config({"agents": {"plan": "codex"}})
    config_file = megaplan.config_dir() / "config.json"
    assert config_file.exists()
    args = Namespace(config_action="reset")
    result = megaplan.handle_config(args)
    assert result["success"] is True
    assert not config_file.exists()


# ── Safety net test ──────────────────────────────────────────────────


def test_run_command_missing_binary_raises_cli_error(tmp_path: Path) -> None:
    """run_command with a nonexistent binary raises CliError, not FileNotFoundError."""
    with pytest.raises(megaplan.CliError) as exc_info:
        megaplan.cli.run_command(
            ["nonexistent-binary-xyz", "--help"],
            cwd=tmp_path,
        )
    assert exc_info.value.code == "agent_not_found"
