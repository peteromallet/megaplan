from __future__ import annotations

from pathlib import Path

import pytest

import megaplan
from megaplan._core import load_plan, read_json
from megaplan.workers import validate_payload
from tests.test_handle_review_robustness import PlanFixture, _advance_to_executed, _make_plan_fixture


def _artifact_names(plan_dir: Path) -> set[str]:
    return {path.name for path in plan_dir.iterdir() if path.is_file()}


def _advance_tiny_to_executed(fixture: PlanFixture) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    args = fixture.make_args(plan=fixture.plan_name)
    megaplan.handlers.handle_prep(fixture.root, args)
    megaplan.handle_plan(fixture.root, args)
    critique = megaplan.handle_critique(fixture.root, args)
    _plan_dir, post_critique_state = load_plan(fixture.root, fixture.plan_name)
    megaplan.handle_finalize(fixture.root, args)
    execute = megaplan.handle_execute(
        fixture.root,
        fixture.make_args(plan=fixture.plan_name, confirm_destructive=True, user_approved=True),
    )
    _plan_dir, final_state = load_plan(fixture.root, fixture.plan_name)
    return critique, post_critique_state, {"state": final_state, "execute": execute}


def test_tiny_robustness_matches_light_artifacts_and_stub_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny_fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="tiny")
    light_fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="light")
    standard_fixture = _make_plan_fixture(tmp_path, monkeypatch, robustness="standard")

    _plan_dir, tiny_initial_state = load_plan(tiny_fixture.root, tiny_fixture.plan_name)
    assert "prep" in megaplan.infer_next_steps(tiny_initial_state)

    tiny_critique, tiny_post_critique_state, tiny_result = _advance_tiny_to_executed(tiny_fixture)
    _advance_to_executed(light_fixture)
    _advance_to_executed(standard_fixture)

    _plan_dir, light_final_state = load_plan(light_fixture.root, light_fixture.plan_name)
    _plan_dir, standard_final_state = load_plan(standard_fixture.root, standard_fixture.plan_name)

    tiny_artifacts = _artifact_names(tiny_fixture.plan_dir)
    light_artifacts = _artifact_names(light_fixture.plan_dir)

    assert tiny_artifacts >= {
        "state.json",
        "faults.json",
        "critique_v1.json",
        "gate.json",
        "review.json",
        "finalize.json",
        "plan_v1.md",
    }
    assert tiny_artifacts == light_artifacts

    critique = read_json(tiny_fixture.plan_dir / "critique_v1.json")
    faults = read_json(tiny_fixture.plan_dir / "faults.json")
    gate = read_json(tiny_fixture.plan_dir / "gate.json")
    review = read_json(tiny_fixture.plan_dir / "review.json")

    validate_payload("critique", critique)
    validate_payload("review", review)

    assert critique == {
        "checks": [],
        "flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    assert faults == {"flags": []}
    assert gate["recommendation"] == "ITERATE"
    assert review["review_verdict"] == "approved"
    assert review["criteria"] == []
    assert review["issues"] == []
    assert review["rework_items"] == []
    assert review["task_verdicts"] == []
    assert review["sense_check_verdicts"] == []
    assert review["summary"]

    assert tiny_critique["state"] == megaplan.STATE_GATED
    assert tiny_post_critique_state["current_state"] == megaplan.STATE_GATED
    assert "finalize" in megaplan.infer_next_steps(tiny_post_critique_state)
    assert "review" not in megaplan.infer_next_steps(
        {"current_state": megaplan.STATE_EXECUTED, "last_gate": {}, "config": {"robustness": "tiny"}}
    )

    assert tiny_result["execute"]["state"] == megaplan.STATE_DONE
    assert tiny_result["state"]["current_state"] == megaplan.STATE_DONE
    assert light_final_state["current_state"] == megaplan.STATE_DONE
    assert standard_final_state["current_state"] == megaplan.STATE_EXECUTED
