from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from megaplan._core import atomic_write_json, atomic_write_text, read_json, schemas_root
from megaplan.checks import checks_for_robustness
from megaplan.hermes_worker import parse_agent_output
from megaplan.parallel_critique import _run_check, run_parallel_critique
from megaplan.prompts.critique import write_single_check_template
from megaplan.types import PlanState
from megaplan.workers import STEP_SCHEMA_FILENAMES


REPO_ROOT = Path(__file__).resolve().parents[1]


def _state(project_dir: Path, *, iteration: int = 1) -> PlanState:
    return {
        "name": "test-plan",
        "idea": "parallelize critique",
        "current_state": "planned",
        "iteration": iteration,
        "created_at": "2026-04-01T00:00:00Z",
        "config": {
            "project_dir": str(project_dir),
            "auto_approve": False,
            "robustness": "standard",
        },
        "sessions": {},
        "plan_versions": [
            {
                "version": iteration,
                "file": f"plan_v{iteration}.md",
                "hash": "sha256:test",
                "timestamp": "2026-04-01T00:00:00Z",
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
            "timestamp": "2026-04-01T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": [{"criterion": "criterion", "priority": "must"}],
            "questions": [],
            "assumptions": [],
        },
    )
    atomic_write_json(plan_dir / "faults.json", {"flags": []})
    return plan_dir, project_dir, state


def _critique_schema() -> dict:
    return read_json(schemas_root(REPO_ROOT) / STEP_SCHEMA_FILENAMES["critique"])


def _finding(detail: str, *, flagged: bool) -> dict[str, object]:
    return {"detail": detail, "flagged": flagged}


def _check_payload(check: dict[str, str], detail: str, *, flagged: bool = False) -> dict[str, object]:
    return {
        "id": check["id"],
        "question": check["question"],
        "findings": [_finding(detail, flagged=flagged)],
    }


def test_run_parallel_critique_merges_in_original_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path)
    checks = checks_for_robustness("standard")

    def fake_run_check(index: int, check: dict[str, str], **kwargs: object):
        del kwargs
        if index == 0:
            time.sleep(0.05)
        return (
            index,
            _check_payload(check, f"Checked {check['id']} in detail for ordered merge coverage.", flagged=False),
            [f"FLAG-00{index + 1}"],
            [],
            0.25, 0, 0, 0,
        )

    monkeypatch.setattr("megaplan.parallel_critique._run_check", fake_run_check)

    result = run_parallel_critique(state, plan_dir, root=REPO_ROOT, model="mock-model", checks=checks)

    assert [check["id"] for check in result.payload["checks"]] == [check["id"] for check in checks]
    assert result.payload["flags"] == []
    assert result.payload["verified_flag_ids"] == [f"FLAG-{index:03d}" for index in range(1, len(checks) + 1)]
    assert result.payload["disputed_flag_ids"] == []


def test_run_parallel_critique_disputed_flags_override_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path)
    checks = checks_for_robustness("standard")

    def fake_run_check(index: int, check: dict[str, str], **kwargs: object):
        del kwargs
        verified = ["FLAG-001"] if index == 0 else []
        disputed = ["FLAG-001"] if index == 1 else []
        return (
            index,
            _check_payload(check, f"Checked {check['id']} with explicit flag merge coverage.", flagged=index == 1),
            verified,
            disputed,
            0.1,
            0, 0, 0,
        )

    monkeypatch.setattr("megaplan.parallel_critique._run_check", fake_run_check)

    result = run_parallel_critique(state, plan_dir, root=REPO_ROOT, model="mock-model", checks=checks[:2])

    assert result.payload["disputed_flag_ids"] == ["FLAG-001"]
    assert "FLAG-001" not in result.payload["verified_flag_ids"]


def test_write_single_check_template_filters_prior_findings_to_target_check(tmp_path: Path) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path, iteration=2)
    target_check = checks_for_robustness("standard")[1]
    other_check = checks_for_robustness("standard")[2]
    atomic_write_json(
        plan_dir / "critique_v1.json",
        {
            "checks": [
                {
                    "id": target_check["id"],
                    "question": target_check["question"],
                    "findings": [
                        _finding(
                            "Checked the target check thoroughly and found a significant issue still worth tracking.",
                            flagged=True,
                        ),
                        _finding(
                            "Checked a second branch of the target check and found it behaved as expected.",
                            flagged=False,
                        ),
                    ],
                },
                {
                    "id": other_check["id"],
                    "question": other_check["question"],
                    "findings": [
                        _finding(
                            "This finding belongs to another check and must not appear in the single-check template.",
                            flagged=True,
                        )
                    ],
                },
            ],
            "flags": [],
            "verified_flag_ids": [],
            "disputed_flag_ids": [],
        },
    )
    atomic_write_json(plan_dir / "faults.json", {"flags": [{"id": target_check["id"], "status": "addressed"}]})

    output_path = write_single_check_template(plan_dir, state, target_check, "critique_check_target.json")
    payload = read_json(output_path)

    assert [check["id"] for check in payload["checks"]] == [target_check["id"]]
    assert payload["checks"][0]["prior_findings"] == [
        {
            "detail": "Checked the target check thoroughly and found a significant issue still worth tracking.",
            "flagged": True,
            "status": "addressed",
        },
        {
            "detail": "Checked a second branch of the target check and found it behaved as expected.",
            "flagged": False,
            "status": "n/a",
        },
    ]


def test_write_single_check_template_uses_unique_output_paths(tmp_path: Path) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path)
    check_a, check_b = checks_for_robustness("standard")[:2]

    path_a = write_single_check_template(plan_dir, state, check_a, "critique_check_issue_hints.json")
    path_b = write_single_check_template(plan_dir, state, check_b, "critique_check_correctness.json")

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()
    assert read_json(path_a)["checks"][0]["id"] == check_a["id"]
    assert read_json(path_b)["checks"][0]["id"] == check_b["id"]


def test_run_parallel_critique_reraises_subagent_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path)
    checks = checks_for_robustness("standard")

    def fake_run_check(index: int, check: dict[str, str], **kwargs: object):
        del check, kwargs
        if index == 1:
            raise RuntimeError("boom")
        return (
            index,
            _check_payload(checks[index], f"Checked {checks[index]['id']} before the parallel failure triggered.", flagged=False),
            [],
            [],
            0.1,
            0, 0, 0,
        )

    monkeypatch.setattr("megaplan.parallel_critique._run_check", fake_run_check)

    with pytest.raises(RuntimeError, match="boom"):
        run_parallel_critique(state, plan_dir, root=REPO_ROOT, model="mock-model", checks=checks)


def test_run_parallel_critique_does_not_mutate_session_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, _project_dir, state = _scaffold(tmp_path)
    state["sessions"] = {
        "hermes_critic": {
            "id": "persisted-session",
            "mode": "persistent",
            "created_at": "2026-04-01T00:00:00Z",
            "last_used_at": "2026-04-01T00:00:00Z",
            "refreshed": False,
        }
    }
    original_sessions = copy.deepcopy(state["sessions"])
    checks = checks_for_robustness("standard")

    def fake_run_check(index: int, check: dict[str, str], **kwargs: object):
        del kwargs
        return (
            index,
            _check_payload(check, f"Checked {check['id']} while preserving the outer session state.", flagged=False),
            [],
            [],
            0.0,
            0, 0, 0,
        )

    monkeypatch.setattr("megaplan.parallel_critique._run_check", fake_run_check)

    run_parallel_critique(state, plan_dir, root=REPO_ROOT, model="mock-model", checks=checks)

    assert state["sessions"] == original_sessions


def test_parse_agent_output_template_prompt_fallback(tmp_path: Path) -> None:
    plan_dir, project_dir, state = _scaffold(tmp_path)
    check = checks_for_robustness("standard")[0]
    output_path = write_single_check_template(plan_dir, state, check, "critique_check_issue_hints.json")
    payload = {
        "checks": [
            {
                "id": check["id"],
                "question": check["question"],
                "guidance": check["guidance"],
                "prior_findings": [],
                "findings": [
                    _finding(
                        "Checked the repository against the user notes and confirmed the resulting path is covered.",
                        flagged=False,
                    )
                ],
            }
        ],
        "flags": [],
        "verified_flag_ids": ["FLAG-001"],
        "disputed_flag_ids": [],
    }

    class FakeAgent:
        def __init__(self, followup: dict[str, object]) -> None:
            self.followup = followup
            self.calls: list[tuple[str, object]] = []

        def run_conversation(self, *, user_message: str, conversation_history: object = None) -> dict[str, object]:
            self.calls.append((user_message, conversation_history))
            return self.followup

    initial_result = {
        "final_response": "",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
            }
        ],
    }
    followup = {"final_response": json.dumps(payload), "messages": [{"role": "assistant", "content": json.dumps(payload)}]}
    agent = FakeAgent(followup)

    parsed, raw_output = parse_agent_output(
        agent,
        initial_result,
        output_path=output_path,
        schema=_critique_schema(),
        step="critique",
        project_dir=project_dir,
        plan_dir=plan_dir,
    )

    assert parsed == payload
    assert raw_output == json.dumps(payload)
    assert len(agent.calls) == 1
    assert "fill in this JSON template" in agent.calls[0][0]


def test_run_check_uses_same_parse_fallback_chain_as_hermes_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_dir, project_dir, state = _scaffold(tmp_path)
    check = checks_for_robustness("standard")[0]
    schema = _critique_schema()
    payload = {
        "checks": [
            {
                "id": check["id"],
                "question": check["question"],
                "guidance": check["guidance"],
                "prior_findings": [],
                "findings": [
                    _finding(
                        "Checked the single-check path and confirmed the summary-prompt fallback can recover JSON.",
                        flagged=False,
                    )
                ],
            }
        ],
        "flags": [],
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
            if len(self.calls) == 1:
                return {
                    "final_response": "",
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
                        }
                    ],
                    "estimated_cost_usd": 0.42,
                }
            return {
                "final_response": json.dumps(payload),
                "messages": [{"role": "assistant", "content": json.dumps(payload)}],
                "estimated_cost_usd": 0.0,
            }

    monkeypatch.setitem(sys.modules, "run_agent", ModuleType("run_agent"))
    monkeypatch.setitem(sys.modules, "hermes_state", ModuleType("hermes_state"))
    sys.modules["run_agent"].AIAgent = FakeAIAgent
    sys.modules["hermes_state"].SessionDB = FakeSessionDB

    index, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = _run_check(
        0,
        check,
        state=state,
        plan_dir=plan_dir,
        root=tmp_path,
        model="minimax:test-model",
        schema=schema,
        project_dir=project_dir,
    )

    assert index == 0
    assert check_payload == {
        "id": check["id"],
        "question": check["question"],
        "findings": [
            {
                "detail": "Checked the single-check path and confirmed the summary-prompt fallback can recover JSON.",
                "flagged": False,
            }
        ],
    }
    assert verified_ids == []
    assert disputed_ids == []
    assert cost_usd == pytest.approx(0.42)
    assert len(FakeAIAgent.instances) == 1
    assert len(FakeAIAgent.instances[0].calls) == 2
    assert "fill in this JSON template" in FakeAIAgent.instances[0].calls[1][0]
