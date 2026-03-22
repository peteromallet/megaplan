"""Evaluation logic: build_evaluation(), decision table, predicates, and scoring."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from megaplan._core import (
    PlanState,
    FlagRecord,
    EvaluationResult,
    current_iteration_artifact,
    read_json,
    normalize_text,
    load_flag_registry,
    unresolved_significant_flags,
    configured_robustness,
    scope_creep_flags,
    latest_plan_path,
    FLAG_BLOCKING_STATUSES,
    ROBUSTNESS_SKIP_THRESHOLDS,
    ROBUSTNESS_STAGNATION_FACTORS,
)


def flag_weight(flag: FlagRecord) -> float:
    """Weight a flag for evaluation scoring. Higher = more blocking."""
    category = flag.get("category", "other")
    concern = flag.get("concern", "").lower()

    if category == "security":
        return 3.0

    implementation_detail_signals = [
        "column", "schema", "field", "as written",
        "pseudocode", "seed sql", "placeholder",
    ]
    if any(signal in concern for signal in implementation_detail_signals):
        return 0.5

    weights = {
        "correctness": 2.0,
        "completeness": 1.5,
        "performance": 1.0,
        "maintainability": 0.75,
        "other": 1.0,
    }
    return weights.get(category, 1.0)


def compute_plan_delta_percent(previous_text: str | None, current_text: str) -> float | None:
    if previous_text is None:
        return None
    ratio = SequenceMatcher(None, previous_text, current_text).ratio()
    return round((1.0 - ratio) * 100.0, 2)


def compute_recurring_critiques(plan_dir: Path, iteration: int) -> list[str]:
    if iteration < 2:
        return []
    previous = read_json(current_iteration_artifact(plan_dir, "critique", iteration - 1))
    current = read_json(current_iteration_artifact(plan_dir, "critique", iteration))
    previous_concerns = {normalize_text(flag["concern"]) for flag in previous.get("flags", [])}
    current_concerns = {normalize_text(flag["concern"]) for flag in current.get("flags", [])}
    return sorted(previous_concerns.intersection(current_concerns))


# ---------------------------------------------------------------------------
# Predicate functions for the decision table
# ---------------------------------------------------------------------------

def _is_all_flags_resolved(significant_count: int, unresolved: list[Any], **_: Any) -> bool:
    """No unresolved significant flags remain."""
    return significant_count == 0 and not unresolved


def _is_low_weight_trending_down(
    iteration: int, weighted_score: float, skip_threshold: float,
    weighted_history: list[float], **_: Any,
) -> bool:
    """Past first iteration, score below threshold and improving."""
    return (
        iteration > 1
        and weighted_score < skip_threshold
        and len(weighted_history) >= 1
        and weighted_score < weighted_history[-1]
    )


def _is_stagnant_with_unresolved(
    plan_delta: float | None, unresolved: list[Any], **_: Any,
) -> bool:
    """Plan text barely changed but significant risks remain."""
    return plan_delta is not None and plan_delta < 5.0 and bool(unresolved)


def _is_stagnant_all_addressed(
    plan_delta: float | None, unresolved: list[Any], **_: Any,
) -> bool:
    """Plan text barely changed and all significant risks addressed."""
    return plan_delta is not None and plan_delta < 5.0 and not unresolved


def _is_first_iteration_with_flags(
    iteration: int, significant_count: int, **_: Any,
) -> bool:
    """First critique iteration and significant flags exist."""
    return iteration == 1 and significant_count > 0


def _has_recurring_critiques(recurring: list[Any], **_: Any) -> bool:
    """Same critique concerns repeated across iterations."""
    return bool(recurring)


def _is_score_stagnating(
    weighted_score: float, weighted_history: list[float],
    stagnation_factor: float, **_: Any,
) -> bool:
    """Weighted flag score is not improving relative to stagnation factor."""
    return (
        len(weighted_history) >= 1
        and weighted_score >= weighted_history[-1] * stagnation_factor
    )


def _is_score_improving(
    weighted_score: float, weighted_history: list[float],
    stagnation_factor: float, **_: Any,
) -> bool:
    """Weighted flag score is trending down past the stagnation factor."""
    return (
        len(weighted_history) >= 1
        and weighted_score < weighted_history[-1] * stagnation_factor
    )


# Decision table: evaluated in priority order; first match wins.
# Each entry is (predicate, recommendation, confidence, rationale_template).
# rationale_template may be a str or a callable(signals -> str) for dynamic messages.
_EVALUATION_DECISION_TABLE: list[
    tuple[
        Any,  # predicate function
        str,  # recommendation
        str,  # confidence
        str | Any,  # rationale (str or callable)
    ]
] = [
    (
        _is_all_flags_resolved,
        "SKIP", "high",
        "No unresolved significant flags remain.",
    ),
    (
        _is_low_weight_trending_down,
        "SKIP", "medium",
        lambda s: f"Remaining flags are low-weight ({s['weighted_score']}) and trending down. Executor can resolve.",
    ),
    (
        _is_stagnant_with_unresolved,
        "ESCALATE", "high",
        "Plan stagnated with unresolved significant risks.",
    ),
    (
        _is_stagnant_all_addressed,
        "SKIP", "high",
        "Plan changes are small and all significant risks appear addressed.",
    ),
    (
        _is_first_iteration_with_flags,
        "CONTINUE", "high",
        lambda s: f"First iteration still has {s['significant_count']} significant flags.",
    ),
    (
        _has_recurring_critiques,
        "ESCALATE", "high",
        "The same critique concerns repeated across iterations.",
    ),
    (
        _is_score_stagnating,
        "ESCALATE", "medium",
        "Weighted flag score is not improving.",
    ),
    (
        _is_score_improving,
        "CONTINUE", "medium",
        "Weighted flag score is trending down.",
    ),
]


def build_evaluation(plan_dir: Path, state: PlanState) -> EvaluationResult:
    iteration = state["iteration"]
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    robustness = configured_robustness(state)
    skip_threshold = ROBUSTNESS_SKIP_THRESHOLDS.get(robustness, 2.0)
    stagnation_factor = ROBUSTNESS_STAGNATION_FACTORS.get(robustness, 0.9)
    open_scope_creep = scope_creep_flags(flag_registry, statuses=FLAG_BLOCKING_STATUSES)
    significant_count = len([flag for flag in flag_registry["flags"] if flag.get("severity") == "significant" and flag.get("status") != "verified"])
    weighted_score = round(sum(flag_weight(f) for f in unresolved), 2)
    weighted_history = list(state["meta"].get("weighted_scores", []))
    latest_plan_text = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    previous_text = None
    if iteration > 1:
        previous_text = (plan_dir / f"plan_v{iteration - 1}.md").read_text(encoding="utf-8")
    plan_delta = compute_plan_delta_percent(previous_text, latest_plan_text)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    # Bundle all signals into a dict so predicates can pick what they need.
    signals = dict(
        iteration=iteration, unresolved=unresolved, significant_count=significant_count,
        weighted_score=weighted_score, weighted_history=weighted_history,
        plan_delta=plan_delta, recurring=recurring, skip_threshold=skip_threshold,
        stagnation_factor=stagnation_factor, state=state,
    )

    # Walk the decision table — first matching predicate wins.
    recommendation = "CONTINUE"
    confidence = "medium"
    rationale = "Continue refining the plan."

    for predicate, rec, conf, rationale_tmpl in _EVALUATION_DECISION_TABLE:
        if predicate(**signals):
            recommendation = rec
            confidence = conf
            rationale = rationale_tmpl(signals) if callable(rationale_tmpl) else rationale_tmpl
            break

    valid_next = ["integrate"] if recommendation == "CONTINUE" else ["gate"] if recommendation == "SKIP" else ["override add-note", "override force-proceed", "override abort"]

    result: EvaluationResult = {
        "recommendation": recommendation,
        "confidence": confidence,
        "robustness": robustness,
        "signals": {
            "iteration": iteration,
            "significant_flags": significant_count,
            "unresolved_flags": [
                {"id": f.get("id"), "concern": f.get("concern", ""), "category": f.get("category", "other")}
                for f in unresolved
            ],
            "weighted_score": weighted_score,
            "weighted_history": weighted_history,
            "plan_delta_from_previous": plan_delta,
            "recurring_critiques": recurring,
            "scope_creep_flags": [flag["id"] for flag in open_scope_creep],
        },
        "idea": state.get("idea", ""),
        "rationale": rationale,
        "valid_next_steps": valid_next,
    }
    # Build a concise loop summary so the orchestrator has full context
    # even if earlier conversation turns were compressed away.
    resolved_flags = [
        {"id": f.get("id"), "concern": f.get("concern", ""), "resolution": f.get("resolution", "")}
        for f in flag_registry["flags"] if f.get("status") == "verified"
    ]
    if resolved_flags:
        result["resolved_flags"] = resolved_flags
    if weighted_history:
        trajectory = " → ".join(str(s) for s in weighted_history) + f" → {weighted_score}"
        deltas = state["meta"].get("plan_deltas", [])
        delta_str = ", ".join(f"{d:.1f}%" for d in deltas) if deltas else "n/a"
        result["loop_summary"] = (
            f"Iteration {iteration}. Score trajectory: {trajectory}. "
            f"Plan deltas: {delta_str}. "
            f"Flags resolved: {len(resolved_flags)}. "
            f"Flags still open: {significant_count}."
        )

    if open_scope_creep:
        result.setdefault("warnings", []).append(
            "Scope creep detected: the plan appears to be expanding beyond the original idea or recorded user notes."
        )

    if iteration >= 5:
        result.setdefault("warnings", []).append(
            f"Iteration {iteration}: high iteration count."
        )

    if iteration >= 12:
        recommendation = "ESCALATE"
        result["recommendation"] = "ESCALATE"
        result["confidence"] = "high"
        result["rationale"] = f"Reached {iteration} iterations — hard limit. Proceed or abort."
        result["valid_next_steps"] = ["override add-note", "override force-proceed", "override abort"]

    if recommendation == "ESCALATE":
        if all(flag_weight(f) <= 1.0 for f in unresolved):
            result["suggested_override"] = "force-proceed"
            result["override_rationale"] = (
                "Remaining flags are implementation details (pseudocode accuracy, "
                "schema column names) that the executor will resolve by reading "
                "the actual code. Safe to proceed."
            )
        elif len(weighted_history) >= 1 and weighted_score > weighted_history[-1] * 1.5:
            result["suggested_override"] = "abort"
            result["override_rationale"] = "Weighted flag score is increasing — the plan may be fundamentally misaligned."
        else:
            result["suggested_override"] = "add-note"
            result["override_rationale"] = (
                "Significant flags remain. Add context to help the next iteration, "
                "or force-proceed if you believe the executor can handle them."
            )

    return result
