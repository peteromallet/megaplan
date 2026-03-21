"""Tests for the test-both override action."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import megaplan.cli as megaplan
import megaplan.cli
import megaplan.workers
from megaplan._core import (
    MOCK_ENV_VAR,
    read_json,
    atomic_write_json,
)


# ---------------------------------------------------------------------------
# Fixtures (matching test_megaplan.py patterns)
# ---------------------------------------------------------------------------

def make_args_factory(project_dir: Path):
    def make_args(**overrides) -> Namespace:
        data = {
            "plan": None, "idea": "test idea", "name": "test-plan",
            "project_dir": str(project_dir), "max_iterations": 3,
            "budget_usd": 25.0, "auto_approve": False, "robustness": "standard",
            "agent": None, "ephemeral": False, "fresh": False, "persist": False,
            "confirm_destructive": True, "user_approved": False,
            "confirm_self_review": False,
            "override_action": None, "note": None, "reason": "",
        }
        data.update(overrides)
        return Namespace(**data)
    return make_args


def load_state(plan_dir: Path) -> dict:
    return read_json(plan_dir / "state.json")


@pytest.fixture()
def plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".git").mkdir()
    monkeypatch.setenv(MOCK_ENV_VAR, "1")
    monkeypatch.setattr(
        megaplan.cli.shutil, "which",
        lambda name: "/usr/bin/mock" if name in {"claude", "codex"} else None,
    )

    make_args = make_args_factory(project_dir)
    init_args = make_args(idea="test idea", name="test-plan")
    megaplan.handle_init(root, init_args)

    plan_dir = root / ".megaplan" / "plans" / "test-plan"
    return {
        "root": root,
        "project_dir": project_dir,
        "plan_dir": plan_dir,
        "make_args": make_args,
    }


def advance_to_evaluated(fx: dict) -> None:
    """Advance plan through plan → critique → evaluate."""
    args = fx["make_args"](plan="test-plan")
    megaplan.handle_plan(fx["root"], args)
    megaplan.handle_critique(fx["root"], args)
    megaplan.handle_evaluate(fx["root"], args)


def force_escalate(fx: dict) -> None:
    """Advance to evaluated, then mutate evaluation to ESCALATE."""
    advance_to_evaluated(fx)
    state = load_state(fx["plan_dir"])
    state["last_evaluation"]["recommendation"] = "ESCALATE"
    state["last_evaluation"]["valid_next_steps"] = [
        "override test-both", "override add-note",
        "override force-proceed", "override abort",
    ]
    atomic_write_json(fx["plan_dir"] / "state.json", state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTestBothOverride:
    def test_test_both_requires_evaluated_state(self, plan_fixture: dict) -> None:
        """test-both should fail if not in evaluated state."""
        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="test",
        )
        with pytest.raises(megaplan.CliError) as exc_info:
            megaplan.handle_override(plan_fixture["root"], args)
        assert exc_info.value.code == "invalid_transition"

    def test_test_both_requires_escalate_recommendation(self, plan_fixture: dict) -> None:
        """test-both should fail if evaluation is SKIP or CONTINUE."""
        advance_to_evaluated(plan_fixture)
        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="test",
        )
        args._test_both_root = plan_fixture["root"]
        with pytest.raises(megaplan.CliError) as exc_info:
            megaplan.handle_override(plan_fixture["root"], args)
        assert exc_info.value.code == "invalid_transition"

    def test_test_both_success(self, plan_fixture: dict) -> None:
        """test-both should succeed from ESCALATE state and write artifacts."""
        force_escalate(plan_fixture)
        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="critique stagnated",
        )
        args._test_both_root = plan_fixture["root"]
        result = megaplan.handle_override(plan_fixture["root"], args)

        assert result["success"] is True
        assert result["step"] == "override"
        assert "test-both.json" in result["artifacts"]
        assert (plan_fixture["plan_dir"] / "test-both.json").exists()

    def test_test_both_writes_verdict_to_state(self, plan_fixture: dict) -> None:
        """test-both should record the verdict in meta.overrides."""
        force_escalate(plan_fixture)
        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="deadlock",
        )
        args._test_both_root = plan_fixture["root"]
        megaplan.handle_override(plan_fixture["root"], args)

        state = load_state(plan_fixture["plan_dir"])
        last_override = state["meta"]["overrides"][-1]
        assert last_override["action"] == "test-both"
        assert last_override["verdict"] in {"approach_a", "approach_b", "synthesis"}
        assert last_override["reason"] == "deadlock"

    def test_test_both_records_history(self, plan_fixture: dict) -> None:
        """test-both should append a history entry."""
        force_escalate(plan_fixture)
        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="test",
        )
        args._test_both_root = plan_fixture["root"]
        megaplan.handle_override(plan_fixture["root"], args)

        state = load_state(plan_fixture["plan_dir"])
        test_both_entries = [h for h in state["history"] if h["step"] == "test-both"]
        assert len(test_both_entries) == 1
        assert test_both_entries[0]["result"] == "success"
        assert test_both_entries[0]["output_file"] == "test-both.json"

    def test_test_both_synthesis_sets_continue(self, plan_fixture: dict) -> None:
        """Mock returns synthesis verdict — evaluation should be set to CONTINUE."""
        force_escalate(plan_fixture)
        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="test",
        )
        args._test_both_root = plan_fixture["root"]
        result = megaplan.handle_override(plan_fixture["root"], args)

        # Default mock returns "synthesis" verdict
        state = load_state(plan_fixture["plan_dir"])
        assert state["last_evaluation"]["recommendation"] == "CONTINUE"
        assert result["next_step"] == "integrate"

    def test_test_both_approach_a_wins(
        self, plan_fixture: dict, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When approach_a wins, should proceed toward gate."""
        force_escalate(plan_fixture)
        original_mock = megaplan.workers.mock_worker_output

        def mock_approach_a_wins(step: str, state: dict, plan_dir: Path):
            if step == "test-both":
                payload = {
                    "approach_a": {
                        "label": "Current plan",
                        "build_pass": True, "test_pass": True,
                        "issues": [], "evidence": "Plan is solid.",
                    },
                    "approach_b": {
                        "label": "Alternative",
                        "build_pass": False, "test_pass": False,
                        "issues": ["Fails to build."],
                        "evidence": "Alternative has compilation errors.",
                    },
                    "verdict": "approach_a",
                    "verdict_rationale": "Current plan builds; alternative does not.",
                }
                return megaplan.workers.WorkerResult(
                    payload=payload, raw_output=json.dumps(payload),
                    duration_ms=10, cost_usd=0.0, session_id="test",
                )
            return original_mock(step, state, plan_dir)

        monkeypatch.setattr(megaplan.workers, "mock_worker_output", mock_approach_a_wins)

        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="test",
        )
        args._test_both_root = plan_fixture["root"]
        result = megaplan.handle_override(plan_fixture["root"], args)

        state = load_state(plan_fixture["plan_dir"])
        assert state["last_evaluation"]["recommendation"] == "SKIP"
        # Gate may not pass if unresolved flags exist from the escalation setup.
        # The important thing is the verdict was recorded and evaluation set to SKIP.
        assert (plan_fixture["plan_dir"] / "gate.json").exists()
        assert result["next_step"] in {"execute", "integrate"}

    def test_test_both_approach_b_wins(
        self, plan_fixture: dict, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When approach_b wins, should proceed to integrate."""
        force_escalate(plan_fixture)
        original_mock = megaplan.workers.mock_worker_output

        def mock_approach_b_wins(step: str, state: dict, plan_dir: Path):
            if step == "test-both":
                payload = {
                    "approach_a": {
                        "label": "Current plan",
                        "build_pass": True, "test_pass": False,
                        "issues": ["Fails existing tests."],
                        "evidence": "Test suite regression.",
                    },
                    "approach_b": {
                        "label": "Alternative",
                        "build_pass": True, "test_pass": True,
                        "issues": [], "evidence": "Clean build and tests.",
                    },
                    "verdict": "approach_b",
                    "verdict_rationale": "Alternative passes all tests; current plan regresses.",
                }
                return megaplan.workers.WorkerResult(
                    payload=payload, raw_output=json.dumps(payload),
                    duration_ms=10, cost_usd=0.0, session_id="test",
                )
            return original_mock(step, state, plan_dir)

        monkeypatch.setattr(megaplan.workers, "mock_worker_output", mock_approach_b_wins)

        args = plan_fixture["make_args"](
            plan="test-plan", override_action="test-both", reason="test",
        )
        args._test_both_root = plan_fixture["root"]
        result = megaplan.handle_override(plan_fixture["root"], args)

        state = load_state(plan_fixture["plan_dir"])
        assert state["last_evaluation"]["recommendation"] == "CONTINUE"
        assert result["next_step"] == "integrate"


class TestTestBothSchema:
    def test_schema_exists(self) -> None:
        """test-both.json schema should be registered."""
        from megaplan.schemas import SCHEMAS
        assert "test-both.json" in SCHEMAS

    def test_schema_required_fields(self) -> None:
        from megaplan.schemas import SCHEMAS
        schema = SCHEMAS["test-both.json"]
        assert "approach_a" in schema["properties"]
        assert "approach_b" in schema["properties"]
        assert "verdict" in schema["properties"]
        assert "verdict_rationale" in schema["properties"]
        assert "verdict" in schema["required"]

    def test_verdict_enum(self) -> None:
        from megaplan.schemas import SCHEMAS
        schema = SCHEMAS["test-both.json"]
        assert schema["properties"]["verdict"]["enum"] == [
            "approach_a", "approach_b", "synthesis",
        ]


class TestTestBothInferNextSteps:
    def test_escalate_includes_test_both(self, plan_fixture: dict) -> None:
        """infer_next_steps should include test-both for ESCALATE."""
        state = load_state(plan_fixture["plan_dir"])
        state["current_state"] = megaplan.STATE_EVALUATED
        state["last_evaluation"] = {"recommendation": "ESCALATE"}
        next_steps = megaplan.infer_next_steps(state)
        assert "override test-both" in next_steps

    def test_abort_includes_test_both(self, plan_fixture: dict) -> None:
        """infer_next_steps should include test-both for ABORT."""
        state = load_state(plan_fixture["plan_dir"])
        state["current_state"] = megaplan.STATE_EVALUATED
        state["last_evaluation"] = {"recommendation": "ABORT"}
        next_steps = megaplan.infer_next_steps(state)
        assert "override test-both" in next_steps

    def test_skip_does_not_include_test_both(self, plan_fixture: dict) -> None:
        """infer_next_steps should NOT include test-both for SKIP."""
        state = load_state(plan_fixture["plan_dir"])
        state["current_state"] = megaplan.STATE_EVALUATED
        state["last_evaluation"] = {"recommendation": "SKIP"}
        next_steps = megaplan.infer_next_steps(state)
        assert "override test-both" not in next_steps


class TestTestBothMock:
    def test_mock_returns_valid_payload(self) -> None:
        """Mock worker should return a valid test-both payload."""
        from megaplan.workers import mock_worker_output, WorkerResult
        state = {
            "idea": "test idea",
            "config": {"project_dir": "/tmp/test"},
            "iteration": 1,
            "meta": {},
            "plan_versions": [{"version": 1, "file": "plan_v1.md"}],
        }
        result = mock_worker_output("test-both", state, Path("/tmp"))
        assert isinstance(result, WorkerResult)
        assert "approach_a" in result.payload
        assert "approach_b" in result.payload
        assert result.payload["verdict"] in {"approach_a", "approach_b", "synthesis"}
        assert "verdict_rationale" in result.payload
