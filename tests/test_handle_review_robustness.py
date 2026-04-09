from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import megaplan
import megaplan.handlers
from megaplan._core import json_dump, load_plan, read_json, save_flag_registry, save_state
from megaplan.workers import WorkerResult, _build_mock_payload


@dataclass
class PlanFixture:
    root: Path
    project_dir: Path
    plan_name: str
    plan_dir: Path
    make_args: Callable[..., Namespace]
    robustness: str


def _make_args_factory(project_dir: Path) -> Callable[..., Namespace]:
    def make_args(**overrides: object) -> Namespace:
        data = {
            "plan": None,
            "idea": "test idea",
            "name": "review-test-plan",
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


def _make_plan_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    robustness: str,
) -> PlanFixture:
    root = tmp_path / f"root-{robustness}"
    project_dir = tmp_path / f"project-{robustness}"
    root.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    monkeypatch.setenv(megaplan.MOCK_ENV_VAR, "1")
    monkeypatch.setattr(
        megaplan._core.shutil,
        "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )

    make_args = _make_args_factory(project_dir)
    response = megaplan.handle_init(root, make_args(name=f"{robustness}-plan", robustness=robustness))
    plan_name = response["plan"]
    return PlanFixture(
        root=root,
        project_dir=project_dir,
        plan_name=plan_name,
        plan_dir=megaplan.plans_root(root) / plan_name,
        make_args=make_args,
        robustness=robustness,
    )


def _advance_to_executed(fixture: PlanFixture) -> None:
    args = fixture.make_args(plan=fixture.plan_name)
    if fixture.robustness == "heavy":
        megaplan.handlers.handle_prep(fixture.root, args)
    megaplan.handle_plan(fixture.root, args)
    megaplan.handle_critique(fixture.root, args)
    megaplan.handle_override(
        fixture.root,
        fixture.make_args(plan=fixture.plan_name, override_action="force-proceed", reason="test"),
    )
    megaplan.handle_finalize(fixture.root, args)
    megaplan.handle_execute(
        fixture.root,
        fixture.make_args(plan=fixture.plan_name, confirm_destructive=True, user_approved=True),
    )


def _load_executed_plan(fixture: PlanFixture) -> tuple[Path, dict[str, object]]:
    plan_dir, state = load_plan(fixture.root, fixture.plan_name)
    state["current_state"] = megaplan.STATE_EXECUTED
    save_state(plan_dir, state)
    return plan_dir, state


def _adjacent_calls_review_checks(*, include_status: bool = True, status: str = "blocking") -> list[dict[str, object]]:
    adjacent_calls = megaplan.review_checks.get_check_by_id("adjacent_calls")
    assert adjacent_calls is not None
    finding: dict[str, object] = {
        "detail": (
            "Adjacent caller coverage still misses a sibling entry point, so the original bug remains reproducible "
            "through an alternate path."
        ),
        "flagged": True,
        "evidence_file": "pkg/module.py",
    }
    if include_status:
        finding["status"] = status
    return [
        {
            "id": adjacent_calls.id,
            "question": adjacent_calls.question,
            "findings": [finding],
        }
    ]


def test_handle_review_standard_branch_attaches_prechecks_and_updates_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="standard")
    _advance_to_executed(fixture)
    _plan_dir, state = _load_executed_plan(fixture)
    payload = _build_mock_payload("review", state, fixture.plan_dir, summary="Golden review payload.")
    pre_check_flags = [
        {
            "id": "PRECHECK-STANDARD-001",
            "check": "source_touch",
            "detail": "The diff touches a source file.",
            "severity": "minor",
        }
    ]
    call_order: list[str] = []
    update_calls: list[dict[str, object]] = []

    worker = WorkerResult(
        payload=payload,
        raw_output="review",
        duration_ms=1,
        cost_usd=0.0,
        session_id="review-standard",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    def _fail_parallel(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("standard review should not invoke heavy review helpers")

    monkeypatch.setattr(megaplan.handlers, "run_parallel_review", _fail_parallel)

    def _run_pre_checks(*args: object, **kwargs: object) -> list[dict[str, str]]:
        del args, kwargs
        call_order.append("run_pre_checks")
        return pre_check_flags

    def _update_flags_after_review(plan_dir: Path, review_payload: dict[str, object], *, iteration: int) -> dict[str, object]:
        update_calls.append(
            {
                "plan_dir": plan_dir,
                "payload": review_payload,
                "iteration": iteration,
            }
        )
        call_order.append("update_flags_after_review")
        return {"flags": []}

    monkeypatch.setattr(megaplan.handlers, "run_pre_checks", _run_pre_checks)
    monkeypatch.setattr(megaplan.handlers, "update_flags_after_review", _update_flags_after_review)
    monkeypatch.setattr(
        megaplan.handlers.worker_module,
        "run_step_with_worker",
        lambda *args, **kwargs: (call_order.append("run_step_with_worker") or (worker, "codex", "persistent", False)),
    )

    megaplan.handle_review(fixture.root, fixture.make_args(plan=fixture.plan_name))
    stored_review = read_json(fixture.plan_dir / "review.json")

    assert stored_review["pre_check_flags"] == pre_check_flags
    assert "flags" not in stored_review
    assert len(update_calls) == 1
    assert update_calls[0]["payload"] is worker.payload
    assert update_calls[0]["iteration"] == state["iteration"]
    assert call_order == ["run_pre_checks", "run_step_with_worker", "update_flags_after_review"]


def test_handle_review_light_branch_skips_prechecks_and_keeps_payload_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="light")
    _advance_to_executed(fixture)
    _plan_dir, state = _load_executed_plan(fixture)
    payload = _build_mock_payload("review", state, fixture.plan_dir, summary="Golden light review payload.")
    golden_path = tmp_path / "golden_light_review.json"
    golden_path.write_text(json_dump(payload), encoding="utf-8")
    update_call_count = 0

    worker = WorkerResult(
        payload=payload,
        raw_output="review",
        duration_ms=1,
        cost_usd=0.0,
        session_id="review-light",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    monkeypatch.setattr(
        megaplan.handlers,
        "run_pre_checks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("light review should not invoke pre-checks")),
    )

    def _update_flags_after_review(*args: object, **kwargs: object) -> None:
        nonlocal update_call_count
        del args, kwargs
        update_call_count += 1

    monkeypatch.setattr(megaplan.handlers, "update_flags_after_review", _update_flags_after_review)
    monkeypatch.setattr(
        megaplan.handlers.worker_module,
        "run_step_with_worker",
        lambda *args, **kwargs: (worker, "codex", "persistent", False),
    )

    megaplan.handle_review(fixture.root, fixture.make_args(plan=fixture.plan_name))
    stored_review = read_json(fixture.plan_dir / "review.json")

    assert stored_review.get("pre_check_flags") == []
    assert update_call_count == 0
    assert (fixture.plan_dir / "review.json").read_bytes() == golden_path.read_bytes()


def test_resolve_review_outcome_uses_standard_and_heavy_caps_separately() -> None:
    state = {
        "history": [
            {"step": "review", "result": "needs_rework"},
            {"step": "review", "result": "needs_rework"},
        ]
    }

    standard_issues: list[str] = []
    heavy_issues: list[str] = []

    standard_result = megaplan.handlers._resolve_review_outcome(
        "needs_rework",
        2,
        2,
        2,
        2,
        [],
        "standard",
        state,
        standard_issues,
    )
    heavy_result = megaplan.handlers._resolve_review_outcome(
        "needs_rework",
        2,
        2,
        2,
        2,
        [],
        "heavy",
        state,
        heavy_issues,
    )

    assert standard_result == ("needs_rework", megaplan.STATE_FINALIZED, "execute")
    assert heavy_result == ("success", megaplan.STATE_DONE, None)
    assert any("Max review rework cycles (2) reached" in issue for issue in heavy_issues)


def test_handle_review_heavy_path_merges_parallel_review_and_creates_review_rework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="heavy")
    _advance_to_executed(fixture)
    _plan_dir, state = load_plan(fixture.root, fixture.plan_name)
    coverage = megaplan.review_checks.get_check_by_id("coverage")
    assert coverage is not None

    pre_check_flags = [
        {
            "id": "PRECHECK-SOURCE_TOUCH",
            "check": "source_touch",
            "detail": "The diff touches a non-test source file.",
            "severity": "minor",
        }
    ]
    criteria_payload = _build_mock_payload("review", state, fixture.plan_dir, review_verdict="approved")
    parallel_result = WorkerResult(
        payload={
            "criteria_payload": criteria_payload,
            "checks": [
                {
                    "id": coverage.id,
                    "question": coverage.question,
                    "findings": [
                        {
                            "detail": "Coverage review found an issue example from the original bug report that the diff still does not address.",
                            "flagged": True,
                            "status": "blocking",
                            "evidence_file": "pkg/module.py",
                        }
                    ],
                }
            ],
            "verified_flag_ids": ["REVIEW-OLD-001"],
            "disputed_flag_ids": ["REVIEW-OLD-002"],
        },
        raw_output="parallel",
        duration_ms=12,
        cost_usd=0.75,
        session_id=None,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )

    monkeypatch.delenv(megaplan.MOCK_ENV_VAR, raising=False)
    monkeypatch.setattr(
        megaplan.handlers,
        "resolve_agent_mode",
        lambda *args, **kwargs: ("hermes", "persistent", True, "mock-model"),
    )
    monkeypatch.setattr(megaplan.handlers, "run_pre_checks", lambda *args, **kwargs: pre_check_flags)
    monkeypatch.setattr(megaplan.handlers, "run_parallel_review", lambda *args, **kwargs: parallel_result)
    monkeypatch.setattr(
        megaplan.handlers.worker_module,
        "run_step_with_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heavy Hermes review should not use the legacy worker")),
    )

    response = megaplan.handle_review(fixture.root, fixture.make_args(plan=fixture.plan_name))
    stored_review = read_json(fixture.plan_dir / "review.json")
    faults = read_json(fixture.plan_dir / "faults.json")

    assert response["success"] is False
    assert response["state"] == megaplan.STATE_FINALIZED
    assert response["next_step"] == "execute"
    assert stored_review["review_verdict"] == "needs_rework"
    assert stored_review["checks"][0]["id"] == "coverage"
    assert stored_review["pre_check_flags"] == pre_check_flags
    assert stored_review["verified_flag_ids"] == ["REVIEW-OLD-001"]
    assert stored_review["disputed_flag_ids"] == ["REVIEW-OLD-002"]
    assert any(item["task_id"] == "REVIEW-coverage" and item["source"] == "review_coverage" for item in stored_review["rework_items"])
    assert any(flag["id"] == "REVIEW-COVERAGE-001" for flag in faults["flags"])


def test_rework_item_synthesized_for_flagged_finding_with_empty_status() -> None:
    rework_items = megaplan.handlers._synthesize_review_rework_items(
        _adjacent_calls_review_checks(status="")
    )

    assert any(item["source"] == "review_adjacent_calls" for item in rework_items)


def test_rework_item_synthesized_for_flagged_finding_with_missing_status_key() -> None:
    rework_items = megaplan.handlers._synthesize_review_rework_items(
        _adjacent_calls_review_checks(include_status=False)
    )

    assert any(item["source"] == "review_adjacent_calls" for item in rework_items)


def test_rework_item_not_synthesized_for_significant_status() -> None:
    rework_items = megaplan.handlers._synthesize_review_rework_items(
        _adjacent_calls_review_checks(status="significant")
    )

    assert rework_items == []


def test_rework_item_not_synthesized_for_minor_status() -> None:
    rework_items = megaplan.handlers._synthesize_review_rework_items(
        _adjacent_calls_review_checks(status="minor")
    )

    assert rework_items == []


def test_rework_item_actual_does_not_duplicate_issue() -> None:
    """The rework item's `actual` field must not be a copy of `issue`.

    Historically both fields were populated from the finding's `detail`,
    which duplicated the sentence in the executor's rework prompt. The
    polish pass replaces `actual` with a templated acknowledgment.
    """
    rework_items = megaplan.handlers._synthesize_review_rework_items(
        _adjacent_calls_review_checks(status="blocking")
    )
    assert rework_items, "Expected at least one rework item for a blocking finding"
    for item in rework_items:
        assert item["issue"] != item["actual"], (
            f"issue and actual must differ, got both={item['issue']!r}"
        )
        assert "did not resolve" in item["actual"]


def test_rework_item_expected_is_actionable_per_check() -> None:
    """`expected` should be a per-check actionable directive, not the check's self-question.

    Exercises each of the 4 review checks by synthesizing a flagged finding
    via the helper and asserting `expected` carries the per-check template
    from `_EXPECTED_BY_CHECK_ID`.
    """
    per_check_markers = {
        "coverage": "every concrete failing example",
        "placement": "upstream",
        "adjacent_calls": "additional call site",
        "simplicity": "unjustified",
    }
    for check_id, marker in per_check_markers.items():
        checks = [
            {
                "id": check_id,
                "question": "Generic placeholder question for the test.",
                "findings": [
                    {
                        "flagged": True,
                        "status": "blocking",
                        "detail": f"A real concern from {check_id} that must be surfaced to the executor.",
                    }
                ],
            }
        ]
        rework_items = megaplan.handlers._synthesize_review_rework_items(checks)
        assert rework_items, f"Expected rework item for check {check_id}"
        expected = rework_items[0]["expected"]
        assert marker in expected, (
            f"Expected per-check marker {marker!r} for {check_id}, got {expected!r}"
        )
        assert "Generic placeholder question" not in expected, (
            f"Expected should not fall back to the check question when a template exists, got {expected!r}"
        )


def test_rework_item_task_id_is_scoped_by_check_id() -> None:
    """`task_id` should be `REVIEW-<check_id>`, not the bare `REVIEW` sentinel."""
    rework_items = megaplan.handlers._synthesize_review_rework_items(
        _adjacent_calls_review_checks(status="blocking")
    )
    assert rework_items
    for item in rework_items:
        assert item["task_id"] == "REVIEW-adjacent_calls", (
            f"Expected REVIEW-adjacent_calls, got {item['task_id']!r}"
        )
        # Lock down the regression — the plain sentinel must not return.
        assert item["task_id"] != "REVIEW"


def test_handle_review_heavy_iteration_two_marks_verified_review_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="heavy")
    _advance_to_executed(fixture)
    _plan_dir, state = load_plan(fixture.root, fixture.plan_name)
    state["iteration"] = 2
    state["current_state"] = megaplan.STATE_EXECUTED
    save_state(fixture.plan_dir, state)
    save_flag_registry(
        fixture.plan_dir,
        {
            "flags": [
                {
                    "id": "REVIEW-COVERAGE-001",
                    "concern": "Coverage gap is still open",
                    "category": "completeness",
                    "severity_hint": "likely-significant",
                    "evidence": "prior issue",
                    "status": "open",
                    "severity": "significant",
                    "verified": False,
                    "raised_in": "review_v1.json",
                }
            ]
        },
    )
    coverage = megaplan.review_checks.get_check_by_id("coverage")
    assert coverage is not None
    criteria_payload = _build_mock_payload("review", state, fixture.plan_dir, review_verdict="approved")
    parallel_result = WorkerResult(
        payload={
            "criteria_payload": criteria_payload,
            "checks": [
                {
                    "id": coverage.id,
                    "question": coverage.question,
                    "findings": [
                        {
                            "detail": "Confirmed the previous coverage concern is now resolved for the original issue example.",
                            "flagged": False,
                            "status": "n/a",
                        }
                    ],
                }
            ],
            "verified_flag_ids": ["REVIEW-COVERAGE-001"],
            "disputed_flag_ids": [],
        },
        raw_output="parallel",
        duration_ms=8,
        cost_usd=0.2,
        session_id=None,
        prompt_tokens=5,
        completion_tokens=6,
        total_tokens=11,
    )

    monkeypatch.delenv(megaplan.MOCK_ENV_VAR, raising=False)
    monkeypatch.setattr(
        megaplan.handlers,
        "resolve_agent_mode",
        lambda *args, **kwargs: ("hermes", "persistent", True, "mock-model"),
    )
    monkeypatch.setattr(megaplan.handlers, "run_pre_checks", lambda *args, **kwargs: [])
    monkeypatch.setattr(megaplan.handlers, "run_parallel_review", lambda *args, **kwargs: parallel_result)
    monkeypatch.setattr(
        megaplan.handlers.worker_module,
        "run_step_with_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heavy Hermes review should not use the legacy worker")),
    )

    response = megaplan.handle_review(fixture.root, fixture.make_args(plan=fixture.plan_name))
    faults = read_json(fixture.plan_dir / "faults.json")
    coverage_flag = next(flag for flag in faults["flags"] if flag["id"] == "REVIEW-COVERAGE-001")

    assert response["success"] is True
    assert response["state"] == megaplan.STATE_DONE
    assert coverage_flag["status"] == "verified"
    assert coverage_flag["verified_in"] == "review_v2.json"
