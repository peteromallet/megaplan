"""Gate-signal scoring and loop diagnostics."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from megaplan._core import (
    FLAG_BLOCKING_STATUSES,
    PlanState,
    FlagRecord,
    GateSignals,
    configured_robustness,
    current_iteration_artifact,
    latest_plan_path,
    load_flag_registry,
    normalize_text,
    read_json,
    scope_creep_flags,
    unresolved_significant_flags,
)


def flag_weight(flag: FlagRecord) -> float:
    """Weight a flag for gate context. Higher = more blocking."""
    category = flag.get("category", "other")
    concern = flag.get("concern", "").lower()

    if category == "security":
        return 3.0

    implementation_detail_signals = [
        "column",
        "schema",
        "field",
        "as written",
        "pseudocode",
        "seed sql",
        "placeholder",
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


def _previous_iteration_plan_path(plan_dir: Path, state: PlanState) -> Path | None:
    current_version = state["iteration"]
    previous_version = current_version - 1
    if previous_version < 1:
        return None
    matching = [
        record
        for record in state["plan_versions"]
        if record.get("version") == previous_version
    ]
    if not matching:
        return None
    return plan_dir / matching[-1]["file"]


def build_gate_signals(plan_dir: Path, state: PlanState) -> GateSignals:
    iteration = state["iteration"]
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    robustness = configured_robustness(state)
    open_scope_creep = scope_creep_flags(flag_registry, statuses=FLAG_BLOCKING_STATUSES)
    significant_count = len(
        [
            flag
            for flag in flag_registry["flags"]
            if flag.get("severity") == "significant" and flag.get("status") != "verified"
        ]
    )
    weighted_score = round(sum(flag_weight(flag) for flag in unresolved), 2)
    weighted_history = list(state["meta"].get("weighted_scores", []))
    latest_plan_text = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    previous_plan_path = _previous_iteration_plan_path(plan_dir, state)
    previous_text = None
    if previous_plan_path is not None and previous_plan_path.exists():
        previous_text = previous_plan_path.read_text(encoding="utf-8")
    plan_delta = compute_plan_delta_percent(previous_text, latest_plan_text)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    resolved_flags = [
        {
            "id": flag.get("id"),
            "concern": flag.get("concern", ""),
            "resolution": flag.get("evidence", ""),
        }
        for flag in flag_registry["flags"]
        if flag.get("status") == "verified"
    ]

    delta_history = state["meta"].get("plan_deltas", [])
    if weighted_history:
        trajectory = " -> ".join(str(score) for score in weighted_history) + f" -> {weighted_score}"
    else:
        trajectory = str(weighted_score)
    delta_summary = ", ".join(
        "n/a" if delta is None else f"{delta:.1f}%"
        for delta in delta_history
    ) or "n/a"
    loop_summary = (
        f"Iteration {iteration}. Weighted score trajectory: {trajectory}. "
        f"Plan deltas: {delta_summary}. "
        f"Recurring critiques: {len(recurring)}. "
        f"Resolved flags: {len(resolved_flags)}. "
        f"Open significant flags: {len(unresolved)}."
    )

    result: GateSignals = {
        "robustness": robustness,
        "signals": {
            "iteration": iteration,
            "idea": state.get("idea", ""),
            "significant_flags": significant_count,
            "unresolved_flags": [
                {
                    "id": flag.get("id"),
                    "concern": flag.get("concern", ""),
                    "category": flag.get("category", "other"),
                    "severity": flag.get("severity", "unknown"),
                    "status": flag.get("status", "unknown"),
                }
                for flag in unresolved
            ],
            "resolved_flags": resolved_flags,
            "weighted_score": weighted_score,
            "weighted_history": weighted_history,
            "plan_delta_from_previous": plan_delta,
            "recurring_critiques": recurring,
            "scope_creep_flags": [flag["id"] for flag in open_scope_creep],
            "loop_summary": loop_summary,
        },
        "warnings": [],
    }
    if open_scope_creep:
        result["warnings"].append(
            "Scope creep detected: the plan appears to be expanding beyond the original idea or recorded user notes."
        )
    if iteration >= 5:
        result["warnings"].append(f"Iteration {iteration}: high iteration count.")
    if iteration >= 12:
        result["warnings"].append(
            f"Iteration {iteration}: hard iteration limit reached. Escalation is likely warranted."
        )
    return result
