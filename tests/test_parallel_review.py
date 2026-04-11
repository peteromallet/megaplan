from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from megaplan._core import atomic_write_json, atomic_write_text, read_json, schemas_root
from megaplan.parallel_review import _run_check, _run_criteria_verdict, run_parallel_review
from megaplan.review_checks import checks_for_robustness
from megaplan.types import PlanState
from megaplan.workers import STEP_SCHEMA_FILENAMES, _build_mock_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


def _state(project_dir: Path, *, iteration: int = 1) -> PlanState:
    return {
        "name": "test-plan",
        "idea": "parallelize review",
        "current_state": "executed",
        "iteration": iteration,
        "created_at": "2026-04-07T00:00:00Z",
        "config": {
            "project_dir": str(project_dir),
            "auto_approve": False,
            "robustness": "superrobust",
        },
        "sessions": {},
        "plan_versions": [
            {
                "version": iteration,
                "file": f"plan_v{iteration}.md",
                "hash": "sha256:test",
                "timestamp": "2026-04-07T00:00:00Z",
            }
        ],
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
        "last_gate": {},
    }


def _scaffold(tmp_path: Path, *, iteration: int = 1) -> tuple[Path, Path, PlanState]:
    plan_dir = tmp_path / "plan"
    project_dir = tmp_path / "project"
    plan_dir.mkdir()
    project_dir.mkdir()
    (project_dir / ".git").mkdir()
    state = _state(project_dir, iteration=iteration)

    atomic_write_text(plan_dir / f"plan_v{iteration}.md", "# Plan\nDo it.\n")
    atomic_write_json(
        plan_dir / f"plan_v{iteration}.meta.json",
        {
            "version": iteration,
            "timestamp": "2026-04-07T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": [{"criterion": "criterion", "priority": "must"}],
            "questions": [],
            "assumptions": [],
        },
    )
    atomic_write_json(plan_dir / "finalize.json", _build_mock_payload("finalize", state, plan_dir))
    atomic_write_json(plan_dir / "gate.json", {"settled_decisions": []})
    atomic_write_json(plan_dir / "faults.json", {"flags": []})
    return plan_dir, project_dir, state


def _review_schema() -> dict:
    return read_json(schemas_root(REPO_ROOT) / STEP_SCHEMA_FILENAMES["review"])


def _finding(detail: str, *, flagged: bool, status: str) -> dict[str, object]:
    return {"detail": detail, "flagged": flagged, "status": status}


def _check_payload(check: object, detail: str, *, flagged: bool = False, status: str = "n/a") -> dict[str, object]:
    return {
        "id": check.id,
        "question": check.question,
        "findings": [_finding(detail, flagged=flagged, status=status)],
    }


def test_run_parallel_review_merges_check_results_in_original_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path)
    checks = checks_for_robustness("superrobust")
    criteria_payload = _build_mock_payload("review", state, plan_dir, review_verdict="approved")
    call_counts = {"criteria": 0, "checks": 0}

    def fake_run_check(index: int, check: object, **kwargs: object):
        assert kwargs["pre_check_flags"] == [{"id": "PRECHECK-1"}]
        call_counts["checks"] += 1
        if index == 0:
            time.sleep(0.05)
        return (
            index,
            _check_payload(check, f"Checked {check.id} in detail for ordered merge coverage.", flagged=False),
            [f"VERIFIED-{index + 1}"],
            [],
            0.25,
            0,
            0,
            0,
        )

    def fake_run_criteria_verdict(**kwargs: object):
        del kwargs
        call_counts["criteria"] += 1
        return criteria_payload, 0.5, 0, 0, 0

    monkeypatch.setattr("megaplan.parallel_review._run_check", fake_run_check)
    monkeypatch.setattr("megaplan.parallel_review._run_criteria_verdict", fake_run_criteria_verdict)

    result = run_parallel_review(
        state,
        plan_dir,
        root=REPO_ROOT,
        model="mock-model",
        checks=checks,
        pre_check_flags=[{"id": "PRECHECK-1"}],
    )

    assert call_counts["checks"] == len(checks)
    assert call_counts["criteria"] == 1
    assert [check["id"] for check in result.payload["checks"]] == [check.id for check in checks]
    assert result.payload["criteria_payload"] == criteria_payload
    assert len(result.payload["verified_flag_ids"]) == len(checks)


def test_run_check_uses_single_check_prompt_and_review_toolset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, project_dir, state = _scaffold(tmp_path)
    check = checks_for_robustness("standard")[0]
    schema = _review_schema()
    prompt_calls: list[tuple[str, list[dict[str, object]]]] = []
    toolset_calls: list[str] = []
    payload = {
        "checks": [
            {
                "id": check.id,
                "question": check.question,
                "guidance": check.guidance,
                "prior_findings": [],
                "findings": [
                    _finding(
                        "Checked the focused review path and confirmed the dedicated prompt builder was used for this check.",
                        flagged=False,
                        status="n/a",
                    )
                ],
            }
        ],
        "flags": [],
        "pre_check_flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }

    class FakeSessionDB:
        pass

    class FakeAIAgent:
        instances: list["FakeAIAgent"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.calls: list[tuple[str, object]] = []
            self._print_fn = None
            self.__class__.instances.append(self)

        def run_conversation(self, *, user_message: str, conversation_history: object = None) -> dict[str, object]:
            self.calls.append((user_message, conversation_history))
            return {
                "final_response": json.dumps(payload),
                "messages": [{"role": "assistant", "content": json.dumps(payload)}],
                "estimated_cost_usd": 0.42,
            }

    monkeypatch.setitem(sys.modules, "run_agent", ModuleType("run_agent"))
    monkeypatch.setitem(sys.modules, "hermes_state", ModuleType("hermes_state"))
    sys.modules["run_agent"].AIAgent = FakeAIAgent
    sys.modules["hermes_state"].SessionDB = FakeSessionDB
    monkeypatch.setattr("megaplan.parallel_review._resolve_model", lambda model: ("mock-model", {}))
    monkeypatch.setattr(
        "megaplan.parallel_review.single_check_review_prompt",
        lambda state, plan_dir, root, check, output_path, pre_check_flags: prompt_calls.append(
            (check.id, pre_check_flags)
        )
        or "focused-review-prompt",
    )
    monkeypatch.setattr(
        "megaplan.parallel_review._toolsets_for_phase",
        lambda phase: toolset_calls.append(phase) or ["file-tools"],
    )

    index, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = _run_check(
        0,
        check,
        state=state,
        plan_dir=plan_dir,
        root=tmp_path,
        model="mock-model",
        schema=schema,
        project_dir=project_dir,
        pre_check_flags=[{"id": "PRECHECK-1"}],
    )

    assert index == 0
    assert check_payload == {
        "id": check.id,
        "question": check.question,
        "findings": [
            {
                "detail": "Checked the focused review path and confirmed the dedicated prompt builder was used for this check.",
                "flagged": False,
                "status": "n/a",
            }
        ],
    }
    assert verified_ids == []
    assert disputed_ids == []
    assert cost_usd == pytest.approx(0.42)
    assert prompt_calls == [(check.id, [{"id": "PRECHECK-1"}])]
    assert toolset_calls == ["review"]
    assert FakeAIAgent.instances[0].kwargs["enabled_toolsets"] == ["file-tools"]
    assert FakeAIAgent.instances[0].calls[0][0] == "focused-review-prompt"
    assert pt == 0
    assert ct == 0
    assert tt == 0


def test_run_criteria_verdict_uses_parallel_review_prompt_and_review_toolset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, project_dir, state = _scaffold(tmp_path)
    schema = _review_schema()
    prompt_calls: list[str] = []
    toolset_calls: list[str] = []
    payload = _build_mock_payload("review", state, plan_dir, review_verdict="approved")

    class FakeSessionDB:
        pass

    class FakeAIAgent:
        instances: list["FakeAIAgent"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.calls: list[tuple[str, object]] = []
            self._print_fn = None
            self.__class__.instances.append(self)

        def run_conversation(self, *, user_message: str, conversation_history: object = None) -> dict[str, object]:
            self.calls.append((user_message, conversation_history))
            return {
                "final_response": json.dumps(payload),
                "messages": [{"role": "assistant", "content": json.dumps(payload)}],
                "estimated_cost_usd": 0.11,
            }

    monkeypatch.setitem(sys.modules, "run_agent", ModuleType("run_agent"))
    monkeypatch.setitem(sys.modules, "hermes_state", ModuleType("hermes_state"))
    sys.modules["run_agent"].AIAgent = FakeAIAgent
    sys.modules["hermes_state"].SessionDB = FakeSessionDB
    monkeypatch.setattr("megaplan.parallel_review._resolve_model", lambda model: ("mock-model", {}))
    monkeypatch.setattr(
        "megaplan.parallel_review.parallel_criteria_review_prompt",
        lambda state, plan_dir, root, output_path: prompt_calls.append(output_path.name) or "criteria-review-prompt",
    )
    monkeypatch.setattr(
        "megaplan.parallel_review._toolsets_for_phase",
        lambda phase: toolset_calls.append(phase) or ["file-tools"],
    )

    criteria_payload, cost_usd, pt, ct, tt = _run_criteria_verdict(
        state=state,
        plan_dir=plan_dir,
        root=tmp_path,
        model="mock-model",
        schema=schema,
        project_dir=project_dir,
    )

    assert criteria_payload == payload
    assert cost_usd == pytest.approx(0.11)
    assert prompt_calls == ["review_criteria_verdict.json"]
    assert toolset_calls == ["review"]
    assert FakeAIAgent.instances[0].kwargs["enabled_toolsets"] == ["file-tools"]
    assert FakeAIAgent.instances[0].calls[0][0] == "criteria-review-prompt"
    assert pt == 0
    assert ct == 0
    assert tt == 0
