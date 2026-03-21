"""Direct tests for megaplan.workers module."""
from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from megaplan._core import CliError
from megaplan.workers import (
    parse_claude_envelope,
    parse_json_file,
    validate_payload,
    resolve_agent_mode,
    run_command,
    update_session_state,
    session_key_for,
    extract_session_id,
)


class TestParsClaudeEnvelope:
    def test_valid_envelope_with_result(self) -> None:
        raw = json.dumps({"result": json.dumps({"plan": "hello"}), "total_cost_usd": 0.01})
        envelope, payload = parse_claude_envelope(raw)
        assert payload["plan"] == "hello"
        assert envelope["total_cost_usd"] == 0.01

    def test_structured_output_preferred(self) -> None:
        raw = json.dumps({
            "result": "ignored",
            "structured_output": {"plan": "from structured"},
        })
        envelope, payload = parse_claude_envelope(raw)
        assert payload["plan"] == "from structured"

    def test_direct_dict_payload(self) -> None:
        """When the entire output is just the payload dict."""
        raw = json.dumps({"plan": "direct"})
        _envelope, payload = parse_claude_envelope(raw)
        assert payload["plan"] == "direct"

    def test_is_error_raises(self) -> None:
        raw = json.dumps({"is_error": True, "result": "something broke"})
        with pytest.raises(CliError) as exc_info:
            parse_claude_envelope(raw)
        assert exc_info.value.code == "worker_error"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(CliError) as exc_info:
            parse_claude_envelope("not json at all")
        assert exc_info.value.code == "parse_error"

    def test_empty_result_raises(self) -> None:
        raw = json.dumps({"result": ""})
        with pytest.raises(CliError) as exc_info:
            parse_claude_envelope(raw)
        assert exc_info.value.code == "parse_error"

    def test_non_object_payload_raises(self) -> None:
        raw = json.dumps({"result": json.dumps([1, 2, 3])})
        with pytest.raises(CliError) as exc_info:
            parse_claude_envelope(raw)
        assert exc_info.value.code == "parse_error"


class TestValidatePayload:
    def test_clarify_valid(self) -> None:
        validate_payload("clarify", {"questions": [], "refined_idea": "x", "intent_summary": "y"})

    def test_clarify_missing_key(self) -> None:
        with pytest.raises(CliError) as exc_info:
            validate_payload("clarify", {"questions": []})
        assert "refined_idea" in exc_info.value.message

    def test_plan_valid(self) -> None:
        validate_payload("plan", {"plan": "p", "questions": [], "success_criteria": [], "assumptions": []})

    def test_plan_missing_key(self) -> None:
        with pytest.raises(CliError) as exc_info:
            validate_payload("plan", {"plan": "p"})
        assert "questions" in exc_info.value.message

    def test_critique_valid(self) -> None:
        validate_payload("critique", {"flags": []})

    def test_critique_missing_flags(self) -> None:
        with pytest.raises(CliError):
            validate_payload("critique", {})

    def test_integrate_valid(self) -> None:
        validate_payload("integrate", {"plan": "p", "changes_summary": "s", "flags_addressed": []})

    def test_execute_valid(self) -> None:
        validate_payload("execute", {"output": "o", "files_changed": [], "commands_run": [], "deviations": []})

    def test_review_valid(self) -> None:
        validate_payload("review", {"criteria": [], "issues": []})

    def test_unknown_step_does_not_raise(self) -> None:
        """Unknown steps are silently accepted (no schema to check)."""
        validate_payload("unknown_step", {"anything": "goes"})


# ---------------------------------------------------------------------------
# resolve_agent_mode tests
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> Namespace:
    defaults = {
        "agent": None,
        "ephemeral": False,
        "fresh": False,
        "persist": False,
        "confirm_self_review": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestResolveAgentMode:
    def test_explicit_agent(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
            agent, mode, refreshed = resolve_agent_mode("plan", _make_args(agent="claude"))
        assert agent == "claude"
        assert mode == "persistent"
        assert refreshed is False

    def test_explicit_agent_not_found(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value=None):
            with pytest.raises(CliError) as exc_info:
                resolve_agent_mode("plan", _make_args(agent="claude"))
            assert exc_info.value.code == "agent_not_found"

    def test_ephemeral_mode(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
            agent, mode, refreshed = resolve_agent_mode("plan", _make_args(agent="claude", ephemeral=True))
        assert mode == "ephemeral"
        assert refreshed is True

    def test_fresh_mode(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
            agent, mode, refreshed = resolve_agent_mode("plan", _make_args(agent="claude", fresh=True))
        assert mode == "persistent"
        assert refreshed is True

    def test_conflicting_flags_raises(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
            with pytest.raises(CliError) as exc_info:
                resolve_agent_mode("plan", _make_args(agent="claude", fresh=True, ephemeral=True))
            assert exc_info.value.code == "invalid_args"

    def test_fallback_when_default_missing(self) -> None:
        def mock_which(name: str) -> str | None:
            return "/usr/bin/codex" if name == "codex" else None

        with patch("megaplan.workers.shutil.which", side_effect=mock_which):
            with patch("megaplan.workers.detect_available_agents", return_value=["codex"]):
                with patch("megaplan.workers.load_config", return_value={}):
                    args = _make_args()
                    agent, mode, refreshed = resolve_agent_mode("plan", args)
        assert agent == "codex"
        assert hasattr(args, "_agent_fallback")

    def test_no_agents_available_raises(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value=None):
            with patch("megaplan.workers.detect_available_agents", return_value=[]):
                with patch("megaplan.workers.load_config", return_value={}):
                    with pytest.raises(CliError) as exc_info:
                        resolve_agent_mode("plan", _make_args())
                    assert exc_info.value.code == "agent_not_found"

    def test_review_claude_defaults_to_fresh(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
            agent, mode, refreshed = resolve_agent_mode("review", _make_args(agent="claude"))
        assert refreshed is True

    def test_review_claude_persist_needs_confirmation(self) -> None:
        with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
            with pytest.raises(CliError) as exc_info:
                resolve_agent_mode("review", _make_args(agent="claude", persist=True))
            assert exc_info.value.code == "invalid_args"


# ---------------------------------------------------------------------------
# run_command tests
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_successful_command(self, tmp_path: Path) -> None:
        result = run_command(["echo", "hello"], cwd=tmp_path)
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert result.duration_ms >= 0

    def test_command_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(CliError) as exc_info:
            run_command(["nonexistent_binary_xyz_123"], cwd=tmp_path)
        assert exc_info.value.code == "agent_not_found"

    def test_timeout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CliError) as exc_info:
            run_command(["sleep", "10"], cwd=tmp_path, timeout=1)
        assert exc_info.value.code == "worker_timeout"


# ---------------------------------------------------------------------------
# update_session_state tests
# ---------------------------------------------------------------------------

class TestUpdateSessionState:
    def test_returns_none_for_no_session(self) -> None:
        result = update_session_state("plan", "claude", None, mode="persistent", refreshed=False)
        assert result is None

    def test_returns_key_and_entry(self) -> None:
        result = update_session_state("plan", "claude", "sess-123", mode="persistent", refreshed=False)
        assert result is not None
        key, entry = result
        assert key == "claude_planner"
        assert entry["id"] == "sess-123"
        assert entry["mode"] == "persistent"
        assert entry["refreshed"] is False

    def test_preserves_existing_created_at(self) -> None:
        existing = {"claude_planner": {"created_at": "2026-01-01T00:00:00Z"}}
        result = update_session_state(
            "plan", "claude", "sess-456",
            mode="persistent", refreshed=True,
            existing_sessions=existing,
        )
        assert result is not None
        _key, entry = result
        assert entry["created_at"] == "2026-01-01T00:00:00Z"

    def test_session_key_mapping(self) -> None:
        assert session_key_for("clarify", "claude") == "claude_planner"
        assert session_key_for("plan", "codex") == "codex_planner"
        assert session_key_for("integrate", "claude") == "claude_planner"
        assert session_key_for("critique", "codex") == "codex_critic"
        assert session_key_for("execute", "claude") == "claude_executor"
        assert session_key_for("review", "codex") == "codex_reviewer"


# ---------------------------------------------------------------------------
# parse_json_file tests
# ---------------------------------------------------------------------------

class TestParseJsonFile:
    def test_valid_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "output.json"
        path.write_text(json.dumps({"plan": "hello"}), encoding="utf-8")
        result = parse_json_file(path)
        assert result == {"plan": "hello"}

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            parse_json_file(path)
        assert exc_info.value.code == "parse_error"
        assert "not valid JSON" in exc_info.value.message

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        with pytest.raises(CliError) as exc_info:
            parse_json_file(path)
        assert exc_info.value.code == "parse_error"
        assert "not created" in exc_info.value.message

    def test_non_object_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "array.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            parse_json_file(path)
        assert exc_info.value.code == "parse_error"
        assert "not contain a JSON object" in exc_info.value.message

    def test_empty_object_returns_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("{}", encoding="utf-8")
        result = parse_json_file(path)
        assert result == {}


# ---------------------------------------------------------------------------
# extract_session_id tests
# ---------------------------------------------------------------------------

class TestExtractSessionId:
    def test_jsonl_thread_id(self) -> None:
        raw = '{"type":"thread.started","thread_id":"abc-123-def"}\n'
        assert extract_session_id(raw) == "abc-123-def"

    def test_text_pattern(self) -> None:
        raw = "Session started\nsession_id: abcd1234-5678-90ef\n"
        assert extract_session_id(raw) == "abcd1234-5678-90ef"

    def test_no_match_returns_none(self) -> None:
        assert extract_session_id("no session here") is None

    def test_empty_returns_none(self) -> None:
        assert extract_session_id("") is None
