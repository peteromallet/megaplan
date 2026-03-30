"""Direct tests for megaplan.workers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from megaplan.evaluation import validate_plan_structure
from megaplan.types import CliError
from megaplan.workers import (
    _build_mock_payload,
    extract_session_id,
    parse_claude_envelope,
    parse_json_file,
    resolve_agent_mode,
    session_key_for,
    update_session_state,
    validate_payload,
)


def _make_args(**overrides: object) -> Namespace:
    data = {
        "agent": None,
        "ephemeral": False,
        "fresh": False,
        "persist": False,
        "confirm_self_review": False,
        "hermes": None,
        "phase_model": [],
    }
    data.update(overrides)
    return Namespace(**data)


def test_parse_claude_envelope_prefers_structured_output() -> None:
    raw = json.dumps({"structured_output": {"plan": "x"}, "total_cost_usd": 0.01})
    envelope, payload = parse_claude_envelope(raw)
    assert envelope["total_cost_usd"] == 0.01
    assert payload == {"plan": "x"}


def test_parse_claude_envelope_rejects_invalid_json() -> None:
    with pytest.raises(CliError, match="valid JSON"):
        parse_claude_envelope("not json")


@pytest.mark.parametrize(
    ("step", "payload"),
    [
        ("plan", {"plan": "x", "questions": [], "success_criteria": [], "assumptions": []}),
        (
            "prep",
            {
                "task_summary": "Prepare context before planning.",
                "key_evidence": [],
                "relevant_code": [],
                "test_expectations": [],
                "constraints": [],
                "suggested_approach": "Use the brief as primary context.",
            },
        ),
        (
            "research",
            {
                "considerations": [
                    {
                        "point": "useRouter is stable in Next.js 14+",
                        "severity": "minor",
                        "detail": "No issues found with the planned usage.",
                    }
                ],
                "summary": "Verified useRouter API is current for Next.js 14.",
            },
        ),
        ("revise", {"plan": "x", "changes_summary": "y", "flags_addressed": []}),
        (
            "gate",
            {
                "recommendation": "PROCEED",
                "rationale": "ok",
                "signals_assessment": "ok",
                "warnings": [],
                "settled_decisions": [],
            },
        ),
        (
            "finalize",
            {
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
            },
        ),
        (
            "execute",
            {
                "output": "done",
                "files_changed": [],
                "commands_run": [],
                "deviations": [],
                "task_updates": [
                    {
                        "task_id": "T1",
                        "status": "done",
                        "executor_notes": "Implemented.",
                        "files_changed": ["megaplan/workers.py"],
                        "commands_run": ["pytest tests/test_workers.py"],
                    }
                ],
                "sense_check_acknowledgments": [
                    {"sense_check_id": "SC1", "executor_note": "Confirmed."}
                ],
            },
        ),
        (
            "review",
            {
                "review_verdict": "approved",
                "criteria": [],
                "issues": [],
                "summary": "ok",
                "task_verdicts": [
                    {
                        "task_id": "T1",
                        "reviewer_verdict": "Pass",
                        "evidence_files": ["megaplan/workers.py"],
                    }
                ],
                "sense_check_verdicts": [{"sense_check_id": "SC1", "verdict": "Confirmed"}],
            },
        ),
    ],
)
def test_validate_payload_accepts_current_worker_steps(step: str, payload: dict[str, object]) -> None:
    validate_payload(step, payload)


def test_validate_payload_rejects_missing_gate_key() -> None:
    with pytest.raises(CliError, match="signals_assessment"):
        validate_payload("gate", {"recommendation": "PROCEED", "rationale": "x", "warnings": []})


def test_session_key_for_matches_new_roles() -> None:
    assert session_key_for("plan", "claude") == "claude_planner"
    assert session_key_for("revise", "codex") == "codex_planner"
    assert session_key_for("research", "hermes") == "hermes_research"
    assert session_key_for("critique", "codex") == "codex_critic"
    assert session_key_for("gate", "claude") == "claude_gatekeeper"
    assert session_key_for("execute", "claude") == "claude_executor"


def test_update_session_state_preserves_created_at() -> None:
    result = update_session_state(
        "gate",
        "claude",
        "session-123",
        mode="persistent",
        refreshed=False,
        existing_sessions={"claude_gatekeeper": {"created_at": "2026-01-01T00:00:00Z"}},
    )
    assert result is not None
    key, entry = result
    assert key == "claude_gatekeeper"
    assert entry["created_at"] == "2026-01-01T00:00:00Z"


def test_extract_session_id_supports_jsonl() -> None:
    raw = '{"type":"thread.started","thread_id":"abc-123"}\n'
    assert extract_session_id(raw) == "abc-123"


def test_parse_json_file_reads_object(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert parse_json_file(path) == {"ok": True}


def test_resolve_agent_mode_uses_configured_fallback() -> None:
    with patch("megaplan.workers.shutil.which", side_effect=lambda name: None if name == "claude" else "/usr/bin/codex"):
        with patch("megaplan.workers.detect_available_agents", return_value=["codex"]):
            with patch("megaplan.workers.load_config", return_value={"agents": {"plan": "claude"}}):
                args = _make_args()
                agent, mode, refreshed, model = resolve_agent_mode("plan", args)
    assert agent == "codex"
    assert mode == "persistent"
    assert refreshed is False
    assert args._agent_fallback["requested"] == "claude"


def test_resolve_agent_mode_for_review_claude_defaults_to_fresh() -> None:
    with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
        agent, mode, refreshed, model = resolve_agent_mode("review", _make_args(agent="claude"))
    assert agent == "claude"
    assert mode == "persistent"
    assert refreshed is True


# ---------------------------------------------------------------------------
# Mock worker tests
# ---------------------------------------------------------------------------


def _mock_state(tmp_path: Path, *, iteration: int = 1) -> tuple[Path, dict]:
    import textwrap
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    state = {
        "name": "test-plan",
        "idea": "test the mock workers",
        "current_state": "critiqued",
        "iteration": iteration,
        "created_at": "2026-03-20T00:00:00Z",
        "config": {
            "project_dir": str(project_dir),
            "auto_approve": False,
            "robustness": "standard",
        },
        "sessions": {},
        "plan_versions": [
            {"version": iteration, "file": f"plan_v{iteration}.md", "hash": "sha256:test", "timestamp": "2026-03-20T00:00:00Z"}
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
    (plan_dir / f"plan_v{iteration}.md").write_text("# Plan\nDo it.\n", encoding="utf-8")
    (plan_dir / f"plan_v{iteration}.meta.json").write_text(
        json.dumps({"version": iteration, "timestamp": "2026-03-20T00:00:00Z", "hash": "sha256:test", "success_criteria": ["criterion"], "questions": [], "assumptions": []}),
        encoding="utf-8",
    )
    (plan_dir / "faults.json").write_text(json.dumps({"flags": []}), encoding="utf-8")
    (plan_dir / "gate.json").write_text(
        json.dumps(
            {
                "passed": True,
                "recommendation": "PROCEED",
                "rationale": "ok",
                "signals_assessment": "ok",
                "warnings": [],
                "settled_decisions": [],
                "criteria_check": {},
                "preflight_results": {},
                "unresolved_flags": [],
                "override_forced": False,
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "execution.json").write_text(
        json.dumps(_build_mock_payload("execute", state, plan_dir, output="done")),
        encoding="utf-8",
    )
    (plan_dir / "finalize.json").write_text(
        json.dumps(
            _build_mock_payload(
                "finalize",
                state,
                plan_dir,
                watch_items=["Watch repository assumptions."],
                tasks=[
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
                        "description": "Verify work",
                        "depends_on": ["T1"],
                        "status": "pending",
                        "executor_notes": "",
                        "files_changed": [],
                        "commands_run": [],
                        "evidence_files": [],
                        "reviewer_verdict": "",
                    },
                ],
                sense_checks=[
                    {"id": "SC1", "task_id": "T1", "question": "Did it work?", "executor_note": "", "verdict": ""},
                    {"id": "SC2", "task_id": "T2", "question": "Was it verified?", "executor_note": "", "verdict": ""},
                ],
                meta_commentary="Mock finalize output.",
            )
        ),
        encoding="utf-8",
    )
    return plan_dir, state


def test_mock_plan_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("plan", state, plan_dir)
    assert "plan" in result.payload
    assert "questions" in result.payload
    assert "success_criteria" in result.payload
    assert "assumptions" in result.payload
    assert validate_plan_structure(result.payload["plan"]) == []


def test_mock_prep_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("prep", state, plan_dir)
    assert "task_summary" in result.payload
    assert "key_evidence" in result.payload
    assert "relevant_code" in result.payload
    assert "test_expectations" in result.payload
    assert "constraints" in result.payload
    assert "suggested_approach" in result.payload


def test_build_mock_payload_execute_returns_complete_payload(tmp_path: Path) -> None:
    plan_dir, state = _mock_state(tmp_path)
    payload = _build_mock_payload(
        "execute",
        state,
        plan_dir,
        task_updates=[
            {
                "task_id": "T1",
                "status": "done",
                "executor_notes": "Verified the targeted execute payload override keeps the schema intact.",
                "files_changed": ["megaplan/workers.py"],
                "commands_run": ["pytest tests/test_workers.py -k build_mock_payload"],
            }
        ],
    )

    assert payload["output"] == "Mock execution completed successfully."
    assert payload["task_updates"][0]["task_id"] == "T1"
    assert len(payload["task_updates"]) == 1
    assert payload["sense_check_acknowledgments"]


def test_build_mock_payload_execute_scopes_batch_from_prompt_override(tmp_path: Path) -> None:
    plan_dir, state = _mock_state(tmp_path)
    payload = _build_mock_payload(
        "execute",
        state,
        plan_dir,
        prompt_override="Only produce task_updates for these tasks: [T2]",
    )

    assert [item["task_id"] for item in payload["task_updates"]] == ["T2"]
    assert [item["sense_check_id"] for item in payload["sense_check_acknowledgments"]] == ["SC2"]


def test_mock_critique_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("critique", state, plan_dir)
    assert "flags" in result.payload
    assert isinstance(result.payload["flags"], list)


def test_mock_revise_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("revise", state, plan_dir)
    assert "plan" in result.payload
    assert "changes_summary" in result.payload
    assert "flags_addressed" in result.payload
    assert validate_plan_structure(result.payload["plan"]) == []


def test_mock_gate_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("gate", state, plan_dir)
    assert "recommendation" in result.payload
    assert result.payload["recommendation"] in {"PROCEED", "ITERATE", "ESCALATE"}
    assert "rationale" in result.payload
    assert "signals_assessment" in result.payload
    assert "warnings" in result.payload
    assert "settled_decisions" in result.payload


def test_mock_finalize_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("finalize", state, plan_dir)
    assert "tasks" in result.payload
    assert "watch_items" in result.payload
    assert "sense_checks" in result.payload
    assert "meta_commentary" in result.payload
    assert isinstance(result.payload["tasks"], list)
    assert isinstance(result.payload["watch_items"], list)
    assert result.payload["tasks"][0]["status"] == "pending"
    assert result.payload["sense_checks"][0]["task_id"] == "T1"


def test_mock_execute_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("execute", state, plan_dir)
    assert "output" in result.payload
    assert "files_changed" in result.payload
    assert "commands_run" in result.payload
    assert "deviations" in result.payload
    assert "task_updates" in result.payload
    assert "sense_check_acknowledgments" in result.payload
    assert result.payload["task_updates"][0]["task_id"] == "T1"


def test_mock_review_returns_valid_payload(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    result = mock_worker_output("review", state, plan_dir)
    assert result.payload["review_verdict"] == "approved"
    assert "criteria" in result.payload
    assert "issues" in result.payload
    assert "summary" in result.payload
    assert "task_verdicts" in result.payload
    assert "sense_check_verdicts" in result.payload


def test_mock_unsupported_step_raises(tmp_path: Path) -> None:
    from megaplan.workers import mock_worker_output
    plan_dir, state = _mock_state(tmp_path)
    with pytest.raises(CliError, match="does not support"):
        mock_worker_output("nonexistent", state, plan_dir)


# ---------------------------------------------------------------------------
# Session key mapping tests
# ---------------------------------------------------------------------------


def test_session_key_for_all_steps() -> None:
    assert session_key_for("plan", "claude") == "claude_planner"
    assert session_key_for("revise", "claude") == "claude_planner"
    assert session_key_for("critique", "claude") == "claude_critic"
    assert session_key_for("gate", "claude") == "claude_gatekeeper"
    assert session_key_for("execute", "claude") == "claude_executor"
    assert session_key_for("review", "claude") == "claude_reviewer"
    assert session_key_for("plan", "codex") == "codex_planner"
    assert session_key_for("revise", "codex") == "codex_planner"
    assert session_key_for("critique", "codex") == "codex_critic"
    assert session_key_for("gate", "codex") == "codex_gatekeeper"
    assert session_key_for("execute", "codex") == "codex_executor"
    assert session_key_for("review", "codex") == "codex_reviewer"


def test_session_key_for_unknown_step_uses_step_name() -> None:
    assert session_key_for("custom", "claude") == "claude_custom"


# ---------------------------------------------------------------------------
# Schema filename mapping tests
# ---------------------------------------------------------------------------


def test_step_schema_filenames_cover_all_steps() -> None:
    from megaplan.workers import STEP_SCHEMA_FILENAMES
    required_steps = {"plan", "prep", "research", "revise", "critique", "gate", "finalize", "execute", "review"}
    assert required_steps.issubset(set(STEP_SCHEMA_FILENAMES.keys()))


def test_step_schema_filenames_reference_existing_schemas() -> None:
    from megaplan.workers import STEP_SCHEMA_FILENAMES
    from megaplan.schemas import SCHEMAS
    for step, filename in STEP_SCHEMA_FILENAMES.items():
        assert filename in SCHEMAS, f"Step '{step}' references non-existent schema '{filename}'"


# ---------------------------------------------------------------------------
# resolve_agent_mode additional tests
# ---------------------------------------------------------------------------


def test_resolve_agent_mode_cli_flag_override() -> None:
    with patch("megaplan.workers.shutil.which", return_value="/usr/bin/codex"):
        agent, mode, refreshed, model = resolve_agent_mode("plan", _make_args(agent="codex"))
    assert agent == "codex"


def test_resolve_agent_mode_config_override() -> None:
    with patch("megaplan.workers.shutil.which", return_value="/usr/bin/codex"):
        with patch("megaplan.workers.load_config", return_value={"agents": {"plan": "codex"}}):
            agent, mode, refreshed, model = resolve_agent_mode("plan", _make_args())
    assert agent == "codex"


def test_resolve_agent_mode_explicit_missing_raises() -> None:
    with patch("megaplan.workers.shutil.which", return_value=None):
        with pytest.raises(CliError, match="not found"):
            resolve_agent_mode("plan", _make_args(agent="nosuchagent"))


def test_resolve_agent_mode_no_agents_raises() -> None:
    with patch("megaplan.workers.shutil.which", return_value=None):
        with patch("megaplan.workers.load_config", return_value={}):
            with patch("megaplan.workers.detect_available_agents", return_value=[]):
                with pytest.raises(CliError, match="No supported agents"):
                    resolve_agent_mode("plan", _make_args())


def test_resolve_agent_mode_conflicting_flags_raises() -> None:
    with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
        with pytest.raises(CliError, match="Cannot combine"):
            resolve_agent_mode("plan", _make_args(fresh=True, ephemeral=True))


def test_resolve_agent_mode_ephemeral_mode() -> None:
    with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
        agent, mode, refreshed, model = resolve_agent_mode("plan", _make_args(agent="claude", ephemeral=True))
    assert mode == "ephemeral"
    assert refreshed is True


# ---------------------------------------------------------------------------
# update_session_state tests
# ---------------------------------------------------------------------------


def test_update_session_state_returns_none_for_no_session_id() -> None:
    result = update_session_state("plan", "claude", None, mode="persistent", refreshed=False)
    assert result is None


def test_update_session_state_creates_new_entry() -> None:
    result = update_session_state("plan", "claude", "sess-abc", mode="persistent", refreshed=False)
    assert result is not None
    key, entry = result
    assert key == "claude_planner"
    assert entry["id"] == "sess-abc"
    assert entry["mode"] == "persistent"


# ---------------------------------------------------------------------------
# validate_payload edge cases
# ---------------------------------------------------------------------------


def test_validate_payload_critique_requires_flags() -> None:
    with pytest.raises(CliError, match="flags"):
        validate_payload("critique", {"verified_flag_ids": [], "disputed_flag_ids": []})


def test_validate_payload_execute_requires_output() -> None:
    with pytest.raises(CliError, match="output"):
        validate_payload(
            "execute",
            {"files_changed": [], "commands_run": [], "deviations": [], "sense_check_acknowledgments": []},
        )


def test_validate_payload_review_requires_criteria() -> None:
    with pytest.raises(CliError, match="criteria"):
        validate_payload("review", {"issues": []})


# ---------------------------------------------------------------------------
# Subprocess-mocked tests for run_claude_step and run_codex_step
# ---------------------------------------------------------------------------


def test_run_claude_step_parses_structured_output(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import CommandResult, run_claude_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    plan_payload = {
        "plan": "# Plan\nDo it.",
        "questions": [],
        "success_criteria": ["criterion"],
        "assumptions": [],
    }
    claude_output = json.dumps({
        "structured_output": plan_payload,
        "total_cost_usd": 0.05,
        "session_id": "sess-abc",
    })
    fake_result = CommandResult(
        command=["claude"],
        cwd=tmp_path,
        returncode=0,
        stdout=claude_output,
        stderr="",
        duration_ms=500,
    )
    with patch("megaplan.workers.run_command", return_value=fake_result):
        result = run_claude_step("plan", state, plan_dir, root=tmp_path, fresh=True)
    assert result.payload == plan_payload
    assert result.cost_usd == 0.05
    assert result.session_id == "sess-abc"
    assert result.duration_ms == 500


def test_run_claude_step_uses_prompt_override_without_builder(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import CommandResult, run_claude_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    fake_result = CommandResult(
        command=["claude"],
        cwd=tmp_path,
        returncode=0,
        stdout=json.dumps({"structured_output": {"plan": "x", "questions": [], "success_criteria": [], "assumptions": []}}),
        stderr="",
        duration_ms=10,
    )
    with patch("megaplan.workers.create_claude_prompt", side_effect=AssertionError("builder should not run")):
        with patch("megaplan.workers.run_command", return_value=fake_result) as run_command:
            run_claude_step("plan", state, plan_dir, root=tmp_path, fresh=True, prompt_override="custom prompt")
    assert run_command.call_args.kwargs["stdin_text"] == "custom prompt"


def test_run_claude_step_raises_on_invalid_payload(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import CommandResult, run_claude_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    # Missing required keys for "plan" step
    claude_output = json.dumps({
        "structured_output": {"plan": "x"},
        "total_cost_usd": 0.0,
    })
    fake_result = CommandResult(
        command=["claude"],
        cwd=tmp_path,
        returncode=0,
        stdout=claude_output,
        stderr="",
        duration_ms=100,
    )
    with patch("megaplan.workers.run_command", return_value=fake_result):
        with pytest.raises(CliError, match="missing required keys"):
            run_claude_step("plan", state, plan_dir, root=tmp_path, fresh=True)


def test_run_claude_step_attaches_session_id_on_timeout(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import run_claude_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    state["sessions"][session_key_for("plan", "claude")] = {
        "id": "claude-session",
        "mode": "persistent",
        "created_at": "2026-03-20T00:00:00Z",
        "last_used_at": "2026-03-20T00:00:00Z",
        "refreshed": False,
    }

    timeout_error = CliError("worker_timeout", "Claude timed out", extra={"raw_output": "partial"})
    with patch("megaplan.workers.run_command", side_effect=timeout_error):
        with pytest.raises(CliError) as exc_info:
            run_claude_step("plan", state, plan_dir, root=tmp_path, fresh=False)

    assert exc_info.value.extra["session_id"] == "claude-session"


def test_run_codex_step_uses_prompt_override_without_builder(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import CommandResult, run_codex_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    output_path = tmp_path / "codex-output.json"

    def fake_named_tempfile(*args: object, **kwargs: object):
        class _TempFile:
            name = str(output_path)

            def close(self) -> None:
                return None

        return _TempFile()

    def fake_run_command(*args: object, **kwargs: object) -> CommandResult:
        output_path.write_text(
            json.dumps({"plan": "# Plan", "questions": [], "success_criteria": [], "assumptions": []}),
            encoding="utf-8",
        )
        return CommandResult(
            command=["codex"],
            cwd=tmp_path,
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=10,
        )

    with patch("megaplan.workers.create_codex_prompt", side_effect=AssertionError("builder should not run")):
        with patch("megaplan.workers.tempfile.NamedTemporaryFile", side_effect=fake_named_tempfile):
            with patch("megaplan.workers.run_command", side_effect=fake_run_command):
                run_codex_step(
                    "plan",
                    state,
                    plan_dir,
                    root=tmp_path,
                    persistent=False,
                    prompt_override="custom prompt",
                )


def test_run_step_with_worker_passes_prompt_override(tmp_path: Path) -> None:
    from megaplan.workers import run_step_with_worker

    plan_dir, state = _mock_state(tmp_path)
    payload = {"output": "done", "files_changed": [], "commands_run": [], "deviations": [], "task_updates": [], "sense_check_acknowledgments": []}
    with patch(
        "megaplan.workers.run_codex_step",
        return_value=type("Result", (), {"payload": payload, "raw_output": "", "duration_ms": 1, "cost_usd": 0.0, "session_id": "sess", "trace_output": None})(),
    ) as run_codex:
        run_step_with_worker(
            "execute",
            state,
            plan_dir,
            _make_args(agent="codex"),
            root=tmp_path,
            resolved=("codex", "persistent", False, None),
            prompt_override="custom execute prompt",
        )
    assert run_codex.call_args.kwargs["prompt_override"] == "custom execute prompt"


def test_run_codex_step_parses_output_file(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import CommandResult, run_codex_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    plan_payload = {
        "plan": "# Plan\nDo it.",
        "questions": [],
        "success_criteria": ["criterion"],
        "assumptions": [],
    }

    def fake_run_command(command: list[str], **kwargs: object) -> CommandResult:
        # Codex writes output to -o file; find the output path in the command
        output_idx = command.index("-o") + 1
        output_path = Path(command[output_idx])
        output_path.write_text(json.dumps(plan_payload), encoding="utf-8")
        return CommandResult(
            command=command,
            cwd=tmp_path,
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=300,
        )

    with patch("megaplan.workers.run_command", side_effect=fake_run_command):
        result = run_codex_step(
            "plan", state, plan_dir, root=tmp_path, persistent=False, fresh=True,
        )
    assert result.payload == plan_payload
    assert result.duration_ms == 300
    assert result.cost_usd == 0.0


def test_run_codex_step_raises_on_nonzero_exit(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import CommandResult, run_codex_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)

    def fake_run_command(command: list[str], **kwargs: object) -> CommandResult:
        return CommandResult(
            command=command,
            cwd=tmp_path,
            returncode=1,
            stdout="",
            stderr="Something went wrong",
            duration_ms=100,
        )

    with patch("megaplan.workers.run_command", side_effect=fake_run_command):
        with pytest.raises(CliError, match="failed with exit code"):
            run_codex_step(
                "plan", state, plan_dir, root=tmp_path, persistent=False, fresh=True,
            )


def test_run_codex_step_extracts_session_id_from_timeout_output(tmp_path: Path) -> None:
    from megaplan._core import ensure_runtime_layout
    from megaplan.workers import run_codex_step

    ensure_runtime_layout(tmp_path)
    plan_dir, state = _mock_state(tmp_path)
    timeout_error = CliError(
        "worker_timeout",
        "Codex timed out",
        extra={"raw_output": '{"type":"thread.started","thread_id":"codex-timeout-session"}\n'},
    )

    with patch("megaplan.workers.run_command", side_effect=timeout_error):
        with pytest.raises(CliError) as exc_info:
            run_codex_step("execute", state, plan_dir, root=tmp_path, persistent=True, fresh=True, json_trace=True)

    assert exc_info.value.extra["session_id"] == "codex-timeout-session"
