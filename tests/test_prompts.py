"""Direct tests for megaplan.prompts module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from megaplan._core import (
    PlanState,
    atomic_write_json,
    atomic_write_text,
    json_dump,
    save_flag_registry,
)
from megaplan.prompts import (
    create_claude_prompt,
    create_codex_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(project_dir: Path, *, iteration: int = 1) -> PlanState:
    return {
        "name": "test-plan",
        "idea": "add dark-mode support",
        "current_state": "planned",
        "iteration": iteration,
        "created_at": "2026-03-20T00:00:00Z",
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
                "timestamp": "2026-03-20T00:00:00Z",
            },
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
        "last_evaluation": {},
    }


def _scaffold(
    tmp_path: Path,
    *,
    iteration: int = 1,
    plan_text: str = "# Plan\nImplement dark mode.\n",
    success_criteria: list[str] | None = None,
    flags: list[dict] | None = None,
    clarification: dict | None = None,
    notes: list[dict] | None = None,
    gate: dict | None = None,
    execution: dict | None = None,
) -> tuple[Path, PlanState]:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    (project_dir / ".git").mkdir(exist_ok=True)

    state = _base_state(project_dir, iteration=iteration)

    if clarification:
        state["clarification"] = clarification

    if notes:
        state["meta"]["notes"] = notes

    (plan_dir / f"plan_v{iteration}.md").write_text(plan_text, encoding="utf-8")
    atomic_write_json(
        plan_dir / f"plan_v{iteration}.meta.json",
        {
            "version": iteration,
            "timestamp": "2026-03-20T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": success_criteria or ["criterion-1"],
            "questions": [],
            "assumptions": [],
        },
    )

    save_flag_registry(plan_dir, {"flags": flags or []})

    atomic_write_json(
        plan_dir / f"critique_v{iteration}.json",
        {"flags": [], "verified_flag_ids": [], "disputed_flag_ids": []},
    )

    atomic_write_json(
        plan_dir / f"evaluation_v{iteration}.json",
        {
            "recommendation": "CONTINUE",
            "confidence": "medium",
            "robustness": "standard",
            "signals": {},
            "rationale": "test",
            "valid_next_steps": ["integrate"],
        },
    )

    if gate is not None:
        atomic_write_json(plan_dir / "gate.json", gate)

    if execution is not None:
        atomic_write_json(plan_dir / "execution.json", execution)

    return plan_dir, state


# ---------------------------------------------------------------------------
# Tests: each step produces non-empty prompts
# ---------------------------------------------------------------------------

class TestPromptBuildersNonEmpty:
    """Every prompt builder must return a non-empty string for both agents."""

    @pytest.fixture(autouse=True)
    def _scaffold(self, tmp_path: Path) -> None:
        self.plan_dir, self.state = _scaffold(
            tmp_path,
            gate={"passed": True, "criteria_check": {}, "preflight_results": {}, "unresolved_flags": []},
            execution={"output": "done", "files_changed": [], "commands_run": [], "deviations": []},
            flags=[
                {
                    "id": "FLAG-001",
                    "concern": "missing tests",
                    "category": "completeness",
                    "severity_hint": "likely-significant",
                    "evidence": "no test file",
                    "status": "open",
                    "severity": "significant",
                    "verified": False,
                    "raised_in": "critique_v1.json",
                },
            ],
        )

    @pytest.mark.parametrize("step", ["clarify", "plan", "critique"])
    def test_claude_prompt_non_empty(self, step: str) -> None:
        prompt = create_claude_prompt(step, self.state, self.plan_dir)
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    @pytest.mark.parametrize("step", ["clarify", "plan", "critique"])
    def test_codex_prompt_non_empty(self, step: str) -> None:
        prompt = create_codex_prompt(step, self.state, self.plan_dir)
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_integrate_prompt_non_empty(self) -> None:
        prompt = create_claude_prompt("integrate", self.state, self.plan_dir)
        assert len(prompt) > 20

    def test_execute_prompt_non_empty(self) -> None:
        prompt = create_claude_prompt("execute", self.state, self.plan_dir)
        assert len(prompt) > 20

    def test_review_claude_prompt_non_empty(self) -> None:
        prompt = create_claude_prompt("review", self.state, self.plan_dir)
        assert len(prompt) > 20

    def test_review_codex_prompt_non_empty(self) -> None:
        prompt = create_codex_prompt("review", self.state, self.plan_dir)
        assert len(prompt) > 20


# ---------------------------------------------------------------------------
# Tests: intent threading
# ---------------------------------------------------------------------------

class TestIntentThreading:
    """Prompts that support clarification should thread the intent_summary."""

    def test_plan_prompt_includes_intent_when_clarified(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(
            tmp_path,
            clarification={
                "refined_idea": "Add dark-mode toggle to settings page",
                "intent_summary": "User wants a dark-mode preference persisted in the DB",
                "questions": [],
            },
        )
        prompt = create_claude_prompt("plan", state, plan_dir)
        assert "dark-mode toggle" in prompt
        assert "intent" in prompt.lower() or "Intent" in prompt

    def test_plan_prompt_works_without_clarification(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(tmp_path)
        prompt = create_claude_prompt("plan", state, plan_dir)
        assert state["idea"] in prompt

    def test_critique_prompt_includes_intent(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(
            tmp_path,
            clarification={
                "refined_idea": "Add dark-mode toggle",
                "intent_summary": "User wants dark-mode persisted",
                "questions": [],
            },
        )
        prompt = create_claude_prompt("critique", state, plan_dir)
        assert "dark-mode" in prompt.lower()

    def test_integrate_prompt_includes_intent(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(
            tmp_path,
            clarification={
                "refined_idea": "Add dark-mode toggle",
                "intent_summary": "User wants dark-mode persisted",
                "questions": [],
            },
            flags=[
                {
                    "id": "FLAG-001",
                    "concern": "missing",
                    "category": "completeness",
                    "severity_hint": "likely-significant",
                    "evidence": "none",
                    "status": "open",
                    "severity": "significant",
                    "verified": False,
                    "raised_in": "critique_v1.json",
                },
            ],
        )
        prompt = create_claude_prompt("integrate", state, plan_dir)
        assert "dark-mode" in prompt.lower()


# ---------------------------------------------------------------------------
# Tests: robustness instructions
# ---------------------------------------------------------------------------

class TestRobustnessInstructions:
    """Critique prompts should include robustness-specific guidance."""

    @pytest.mark.parametrize("level,keyword", [
        ("light", "pragmatic"),
        ("standard", "balanced"),
        ("thorough", "exhaustive"),
    ])
    def test_critique_robustness_instruction(self, tmp_path: Path, level: str, keyword: str) -> None:
        plan_dir, state = _scaffold(tmp_path)
        state["config"]["robustness"] = level
        prompt = create_claude_prompt("critique", state, plan_dir)
        assert keyword.lower() in prompt.lower()

    def test_execute_prompt_mentions_robustness(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(
            tmp_path,
            gate={"passed": True, "criteria_check": {}, "preflight_results": {}, "unresolved_flags": []},
        )
        state["config"]["robustness"] = "thorough"
        prompt = create_claude_prompt("execute", state, plan_dir)
        assert "thorough" in prompt.lower()


# ---------------------------------------------------------------------------
# Tests: unsupported step
# ---------------------------------------------------------------------------

class TestUnsupportedStep:
    def test_claude_unsupported_step_raises(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(tmp_path)
        from megaplan._core import CliError
        with pytest.raises(CliError):
            create_claude_prompt("nonexistent", state, plan_dir)

    def test_codex_unsupported_step_raises(self, tmp_path: Path) -> None:
        plan_dir, state = _scaffold(tmp_path)
        from megaplan._core import CliError
        with pytest.raises(CliError):
            create_codex_prompt("nonexistent", state, plan_dir)
