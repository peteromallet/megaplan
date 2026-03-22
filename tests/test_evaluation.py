"""Direct tests for megaplan.evaluation module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from megaplan.evaluation import (
    build_evaluation,
    flag_weight,
    _is_all_flags_resolved,
    _is_low_weight_trending_down,
    _is_stagnant_with_unresolved,
    _is_stagnant_all_addressed,
    _is_first_iteration_with_flags,
    _has_recurring_critiques,
    _is_score_stagnating,
    _is_score_improving,
    compute_plan_delta_percent,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _eval_scaffold(
    tmp_path: Path,
    *,
    iteration: int = 1,
    flags: list[dict] | None = None,
    weighted_scores: list[float] | None = None,
    total_cost_usd: float = 0.0,
    robustness: str = "standard",
) -> tuple[Path, dict]:
    """Set up filesystem state for build_evaluation."""
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True)
    flags = flags if flags is not None else []
    weighted_scores = weighted_scores if weighted_scores is not None else []

    plan_text = "Current plan text with substantial changes for a healthy delta.\n"
    (plan_dir / f"plan_v{iteration}.md").write_text(plan_text, encoding="utf-8")
    _write_json(
        plan_dir / f"plan_v{iteration}.meta.json",
        {
            "version": iteration,
            "timestamp": "2026-03-20T00:00:00Z",
            "hash": "sha256:test",
            "success_criteria": ["criterion"],
            "questions": [],
            "assumptions": [],
        },
    )
    _write_json(
        plan_dir / f"critique_v{iteration}.json",
        {"flags": [], "verified_flag_ids": [], "disputed_flag_ids": []},
    )
    _write_json(plan_dir / "faults.json", {"flags": flags})

    state = {
        "name": "test-plan",
        "idea": "test idea",
        "current_state": "critiqued",
        "iteration": iteration,
        "config": {
            "project_dir": str(tmp_path / "project"),
            "auto_approve": False,
            "robustness": robustness,
        },
        "plan_versions": [
            {
                "version": iteration,
                "file": f"plan_v{iteration}.md",
                "hash": "sha256:test",
                "timestamp": "2026-03-20T00:00:00Z",
            },
        ],
        "meta": {
            "significant_counts": [],
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


class TestFlagWeight:
    def test_security_flag_highest_weight(self) -> None:
        flag = {"category": "security", "concern": "SQL injection risk"}
        assert flag_weight(flag) == 3.0

    def test_correctness_flag(self) -> None:
        assert flag_weight({"category": "correctness", "concern": "logic error"}) == 2.0

    def test_completeness_flag(self) -> None:
        assert flag_weight({"category": "completeness", "concern": "missing feature"}) == 1.5

    def test_performance_flag(self) -> None:
        assert flag_weight({"category": "performance", "concern": "slow query"}) == 1.0

    def test_maintainability_flag(self) -> None:
        assert flag_weight({"category": "maintainability", "concern": "tangled code"}) == 0.75

    def test_other_flag(self) -> None:
        assert flag_weight({"category": "other", "concern": "misc"}) == 1.0

    def test_unknown_category_defaults_to_1(self) -> None:
        assert flag_weight({"category": "nonexistent", "concern": "something"}) == 1.0

    def test_missing_category_defaults_to_other(self) -> None:
        assert flag_weight({"concern": "no category"}) == 1.0

    def test_implementation_detail_signals_reduce_weight(self) -> None:
        """Flags with implementation-detail keywords get 0.5 weight."""
        for signal in ["column", "schema", "field", "as written", "pseudocode", "seed sql", "placeholder"]:
            flag = {"category": "correctness", "concern": f"The {signal} is wrong"}
            assert flag_weight(flag) == 0.5, f"Signal '{signal}' should reduce weight"

    def test_security_overrides_implementation_signal(self) -> None:
        """Security category takes priority over implementation-detail signals."""
        flag = {"category": "security", "concern": "the schema allows injection"}
        assert flag_weight(flag) == 3.0

    def test_empty_flag(self) -> None:
        assert flag_weight({}) == 1.0


class TestBuildEvaluationIntegration:
    """Integration tests for build_evaluation with filesystem state."""

    def test_first_iteration_with_flags_returns_continue(self, tmp_path: Path) -> None:
        plan_dir, state = _eval_scaffold(
            tmp_path,
            flags=[
                {
                    "id": "FLAG-001",
                    "concern": "Missing error handling",
                    "category": "correctness",
                    "severity_hint": "likely-significant",
                    "evidence": "no try/except",
                    "status": "open",
                    "severity": "significant",
                    "verified": False,
                    "raised_in": "critique_v1.json",
                },
            ],
        )
        result = build_evaluation(plan_dir, state)
        assert result["recommendation"] == "CONTINUE"
        assert result["confidence"] == "high"
        assert "signals" in result
        assert "rationale" in result
        assert "valid_next_steps" in result
        assert result["signals"]["iteration"] == 1
        assert result["signals"]["significant_flags"] == 1

    def test_no_flags_returns_skip(self, tmp_path: Path) -> None:
        plan_dir, state = _eval_scaffold(tmp_path, flags=[])
        result = build_evaluation(plan_dir, state)
        assert result["recommendation"] == "SKIP"
        assert result["valid_next_steps"] == ["gate"]

    def test_robustness_affects_result(self, tmp_path: Path) -> None:
        plan_dir, state = _eval_scaffold(tmp_path, robustness="thorough")
        result = build_evaluation(plan_dir, state)
        assert result["robustness"] == "thorough"

    def test_result_contains_required_keys(self, tmp_path: Path) -> None:
        plan_dir, state = _eval_scaffold(tmp_path)
        result = build_evaluation(plan_dir, state)
        required_keys = {"recommendation", "confidence", "robustness", "signals", "rationale", "valid_next_steps"}
        assert required_keys.issubset(result.keys())

    def test_scope_creep_flags_produce_warning(self, tmp_path: Path) -> None:
        plan_dir, state = _eval_scaffold(
            tmp_path,
            flags=[
                {
                    "id": "FLAG-001",
                    "concern": "Scope creep: plan expands beyond the original idea",
                    "category": "other",
                    "severity_hint": "likely-significant",
                    "evidence": "expanded scope",
                    "status": "open",
                    "severity": "significant",
                    "verified": False,
                    "raised_in": "critique_v1.json",
                },
            ],
        )
        result = build_evaluation(plan_dir, state)
        assert "warnings" in result
        assert any("scope creep" in w.lower() for w in result["warnings"])


# ---------------------------------------------------------------------------
# Decision-table predicate tests with explicit input/output pairs
# ---------------------------------------------------------------------------

class TestIsAllFlagsResolved:
    def test_all_resolved(self) -> None:
        assert _is_all_flags_resolved(significant_count=0, unresolved=[]) is True

    def test_significant_count_nonzero(self) -> None:
        assert _is_all_flags_resolved(significant_count=2, unresolved=[]) is False

    def test_unresolved_nonempty(self) -> None:
        assert _is_all_flags_resolved(significant_count=0, unresolved=[{"id": "FLAG-001"}]) is False

    def test_both_nonzero(self) -> None:
        assert _is_all_flags_resolved(significant_count=1, unresolved=[{"id": "FLAG-001"}]) is False


class TestIsLowWeightTrendingDown:
    def test_trending_down(self) -> None:
        assert _is_low_weight_trending_down(
            iteration=2, weighted_score=1.0, skip_threshold=2.0,
            weighted_history=[3.0],
        ) is True

    def test_first_iteration_returns_false(self) -> None:
        assert _is_low_weight_trending_down(
            iteration=1, weighted_score=1.0, skip_threshold=2.0,
            weighted_history=[3.0],
        ) is False

    def test_above_threshold_returns_false(self) -> None:
        assert _is_low_weight_trending_down(
            iteration=2, weighted_score=3.0, skip_threshold=2.0,
            weighted_history=[4.0],
        ) is False

    def test_not_improving_returns_false(self) -> None:
        assert _is_low_weight_trending_down(
            iteration=2, weighted_score=1.5, skip_threshold=2.0,
            weighted_history=[1.0],
        ) is False

    def test_empty_history_returns_false(self) -> None:
        assert _is_low_weight_trending_down(
            iteration=2, weighted_score=1.0, skip_threshold=2.0,
            weighted_history=[],
        ) is False


class TestIsStagnantWithUnresolved:
    def test_stagnant_with_flags(self) -> None:
        assert _is_stagnant_with_unresolved(plan_delta=2.0, unresolved=[{"id": "F"}]) is True

    def test_stagnant_no_flags(self) -> None:
        assert _is_stagnant_with_unresolved(plan_delta=2.0, unresolved=[]) is False

    def test_large_delta(self) -> None:
        assert _is_stagnant_with_unresolved(plan_delta=10.0, unresolved=[{"id": "F"}]) is False

    def test_none_delta(self) -> None:
        assert _is_stagnant_with_unresolved(plan_delta=None, unresolved=[{"id": "F"}]) is False


class TestIsStagnantAllAddressed:
    def test_stagnant_addressed(self) -> None:
        assert _is_stagnant_all_addressed(plan_delta=3.0, unresolved=[]) is True

    def test_stagnant_with_unresolved(self) -> None:
        assert _is_stagnant_all_addressed(plan_delta=3.0, unresolved=[{"id": "F"}]) is False

    def test_large_delta(self) -> None:
        assert _is_stagnant_all_addressed(plan_delta=10.0, unresolved=[]) is False


class TestIsFirstIterationWithFlags:
    def test_first_with_flags(self) -> None:
        assert _is_first_iteration_with_flags(iteration=1, significant_count=2) is True

    def test_first_without_flags(self) -> None:
        assert _is_first_iteration_with_flags(iteration=1, significant_count=0) is False

    def test_later_iteration(self) -> None:
        assert _is_first_iteration_with_flags(iteration=2, significant_count=5) is False


class TestHasRecurringCritiques:
    def test_has_recurring(self) -> None:
        assert _has_recurring_critiques(recurring=["concern A"]) is True

    def test_no_recurring(self) -> None:
        assert _has_recurring_critiques(recurring=[]) is False


class TestIsScoreStagnating:
    def test_stagnating(self) -> None:
        # score >= last * factor means stagnating
        assert _is_score_stagnating(
            weighted_score=9.0, weighted_history=[10.0], stagnation_factor=0.9,
        ) is True

    def test_improving(self) -> None:
        assert _is_score_stagnating(
            weighted_score=5.0, weighted_history=[10.0], stagnation_factor=0.9,
        ) is False

    def test_empty_history(self) -> None:
        assert _is_score_stagnating(
            weighted_score=5.0, weighted_history=[], stagnation_factor=0.9,
        ) is False


class TestIsScoreImproving:
    def test_improving(self) -> None:
        assert _is_score_improving(
            weighted_score=5.0, weighted_history=[10.0], stagnation_factor=0.9,
        ) is True

    def test_not_improving(self) -> None:
        assert _is_score_improving(
            weighted_score=9.5, weighted_history=[10.0], stagnation_factor=0.9,
        ) is False

    def test_empty_history(self) -> None:
        assert _is_score_improving(
            weighted_score=5.0, weighted_history=[], stagnation_factor=0.9,
        ) is False


class TestComputePlanDeltaPercent:
    def test_identical_texts(self) -> None:
        assert compute_plan_delta_percent("hello", "hello") == 0.0

    def test_completely_different(self) -> None:
        delta = compute_plan_delta_percent("aaa", "zzz")
        assert delta is not None
        assert delta > 50.0

    def test_none_previous(self) -> None:
        assert compute_plan_delta_percent(None, "any") is None
