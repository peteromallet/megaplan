"""Gate-signal scoring and loop diagnostics."""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from megaplan.types import (
    FLAG_BLOCKING_STATUSES,
    FlagRecord,
    GateArtifact,
    GateCheckResult,
    GatePayload,
    GateSignals,
    PlanState,
)
from megaplan._core import (
    configured_robustness,
    current_iteration_artifact,
    latest_plan_meta_path,
    latest_plan_path,
    load_flag_registry,
    normalize_text,
    read_json,
    scope_creep_flags,
    unresolved_significant_flags,
)


PLAN_STRUCTURE_REQUIRED_STEP_ISSUE = "Plan must include at least one `## Step N:` section."


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


def _strip_fenced_blocks(text: str) -> str:
    kept_lines: list[str] = []
    inside_fence = False
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            kept_lines.append(line)
    return "".join(kept_lines)


def validate_plan_structure(plan_text: str) -> list[str]:
    issues: list[str] = []
    stripped = _strip_fenced_blocks(plan_text)

    if len(re.findall(r"(?mi)^#\s+.+$", stripped)) != 1:
        issues.append("Plan should have exactly one H1 title.")
    if not re.search(r"(?mi)^##\s+Overview\s*$", stripped):
        issues.append("Plan should include a `## Overview` section.")

    step_matches = list(re.finditer(r"(?im)^##\s+Step\s+\d+:\s+.+$", stripped))
    if not step_matches:
        issues.append(PLAN_STRUCTURE_REQUIRED_STEP_ISSUE)
        return issues

    if not (
        re.search(r"(?mi)^##\s+Execution Order\s*$", stripped)
        or re.search(r"(?mi)^##\s+Validation Order\s*$", stripped)
    ):
        issues.append("Plan should include `## Execution Order` or `## Validation Order`.")

    missing_substeps = False
    missing_file_refs = False
    for index, match in enumerate(step_matches):
        start = match.end()
        next_heading = re.search(r"(?im)^##\s+.+$", stripped[start:])
        end = start + next_heading.start() if next_heading else len(stripped)
        section = stripped[match.start():end]
        if not re.search(r"(?m)^\d+\.\s+", stripped[start:end]):
            missing_substeps = True
        if not re.search(r"`[^`]+`", section):
            missing_file_refs = True

    if missing_substeps:
        issues.append("Each `## Step N:` section should include at least one numbered substep.")
    if missing_file_refs:
        issues.append("Each `## Step N:` section should reference at least one file in backticks.")
    return issues


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
            if flag.get("severity") == "significant" and flag["status"] != "verified"
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
            "id": flag["id"],
            "concern": flag["concern"],
            "resolution": flag.get("evidence", ""),
        }
        for flag in flag_registry["flags"]
        if flag["status"] == "verified"
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
                    "id": flag["id"],
                    "concern": flag["concern"],
                    "category": flag["category"],
                    "severity": flag.get("severity", "unknown"),
                    "status": flag["status"],
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


def run_gate_checks(
    plan_dir: Path,
    state: PlanState,
    *,
    command_lookup: Callable[[str], str | None] | None = None,
) -> GateCheckResult:
    project_dir = Path(state["config"]["project_dir"])
    meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    lookup = command_lookup or (lambda name: None)
    checks = {
        "project_dir_exists": project_dir.exists(),
        "project_dir_writable": os.access(project_dir, os.W_OK),
        "success_criteria_present": bool(meta.get("success_criteria")),
        "claude_available": bool(lookup("claude")),
        "codex_available": bool(lookup("codex")),
    }
    return {
        "passed": all(checks.values()),
        "criteria_check": {
            "count": len(meta.get("success_criteria", [])),
            "items": meta.get("success_criteria", []),
        },
        "preflight_results": checks,
        "unresolved_flags": unresolved,
    }


def build_gate_artifact(
    signals: dict[str, Any],
    gate_payload: GatePayload,
    *,
    override_forced: bool,
    orchestrator_guidance: str = "",
) -> GateArtifact:
    preflight = signals["preflight_results"]
    recommendation = gate_payload["recommendation"]
    warnings = list(signals.get("warnings", [])) + list(gate_payload.get("warnings", []))
    return {
        "passed": recommendation == "PROCEED" and all(preflight.values()),
        "criteria_check": signals["criteria_check"],
        "preflight_results": preflight,
        "unresolved_flags": signals["unresolved_flags"],
        "recommendation": recommendation,
        "rationale": gate_payload["rationale"],
        "signals_assessment": gate_payload["signals_assessment"],
        "warnings": warnings,
        "override_forced": override_forced,
        "orchestrator_guidance": orchestrator_guidance,
        "robustness": signals.get("robustness"),
        "signals": signals["signals"],
    }


def build_orchestrator_guidance(
    gate_payload: GatePayload,
    signals: dict[str, Any],
    preflight_passed: bool,
    preflight_results: dict[str, bool],
    robustness: str,
    plan_name: str,
) -> str:
    """Return plain-language next-step guidance for the orchestrator."""
    recommendation = gate_payload["recommendation"]
    iteration = int(signals.get("iteration", 0))
    weighted_score = float(signals.get("weighted_score", 0.0))
    weighted_history = list(signals.get("weighted_history", []))
    recurring_critiques = list(signals.get("recurring_critiques", []))
    unresolved_flags = list(signals.get("unresolved_flags", []))
    scope_creep = list(signals.get("scope_creep_flags", []))
    previous_score = float(weighted_history[-1]) if weighted_history else None
    plateaued = previous_score is not None and weighted_score >= previous_score
    worsening = previous_score is not None and weighted_score > previous_score
    improving = previous_score is not None and weighted_score < previous_score

    if iteration == 1:
        guidance = f"First iteration; follow gate recommendation: {recommendation}."
    elif recommendation == "PROCEED" and preflight_passed:
        guidance = "Plan passed gate and preflight. Proceed to finalize."
    elif recommendation == "PROCEED":
        failing_checks = ", ".join(
            name for name, passed in preflight_results.items() if not passed
        )
        guidance = f"Gate says PROCEED but preflight blocked. Fix: {failing_checks}."
    elif recommendation == "ESCALATE" and robustness == "light" and weighted_score <= 4.0:
        guidance = (
            "Auto-force-proceed eligible. Run: "
            f'`megaplan override force-proceed --plan {plan_name} --reason "light robustness, score {weighted_score}"`'
        )
    elif recommendation == "ESCALATE":
        guidance = "Gate escalated. Ask the user: force-proceed, add-note, or abort."
    elif recommendation == "ITERATE" and plateaued and recurring_critiques:
        guidance = (
            "Score plateaued with recurring critiques the loop can't fix. Consider "
            f"force-proceeding: `megaplan override force-proceed --plan {plan_name}`"
        )
    elif recommendation == "ITERATE" and improving:
        guidance = f"Score improving ({previous_score} -> {weighted_score}). Continue to revise."
    elif recommendation == "ITERATE" and worsening:
        guidance = (
            f"Score worsening ({previous_score} -> {weighted_score}). "
            "Investigate; the loop may be diverging."
        )
    else:
        guidance = "Gate recommends another iteration. Revise the plan."

    hints: list[str] = []
    if unresolved_flags:
        hints.append("Verify unresolved flags against the plan and project code before accepting.")
    if recurring_critiques:
        critiques = ", ".join(recurring_critiques)
        hints.append(
            f"Recurring critiques ({critiques}); the loop likely can't fix these, so judge if they are real blockers."
        )
    if scope_creep:
        hints.append("Scope creep detected; compare the current plan against the original idea.")

    return " ".join([guidance, *hints]).strip()
