"""Direct tests for megaplan.workers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from megaplan._core import CliError
from megaplan.workers import (
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
        ("revise", {"plan": "x", "changes_summary": "y", "flags_addressed": []}),
        ("gate", {"recommendation": "PROCEED", "rationale": "ok", "signals_assessment": "ok", "warnings": []}),
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
                agent, mode, refreshed = resolve_agent_mode("plan", args)
    assert agent == "codex"
    assert mode == "persistent"
    assert refreshed is False
    assert args._agent_fallback["requested"] == "claude"


def test_resolve_agent_mode_for_review_claude_defaults_to_fresh() -> None:
    with patch("megaplan.workers.shutil.which", return_value="/usr/bin/claude"):
        agent, mode, refreshed = resolve_agent_mode("review", _make_args(agent="claude"))
    assert agent == "claude"
    assert mode == "persistent"
    assert refreshed is True
