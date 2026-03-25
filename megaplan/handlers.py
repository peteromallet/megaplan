from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import megaplan.workers as worker_module
from megaplan.execution import (
    _capture_git_status_snapshot,
    _check_done_task_evidence,
    _resolve_execute_approval_mode,
    handle_execute_auto_loop as dispatch_execute_auto_loop,
    handle_execute_one_batch as dispatch_execute_one_batch,
)
from megaplan.merge import _merge_validated_entries, _validate_and_merge_batch, _validate_merge_inputs
from megaplan.types import (
    FLAG_BLOCKING_STATUSES,
    ROBUSTNESS_LEVELS,
    CliError,
    FlagRecord,
    FlagRegistry,
    GateCheckResult,
    HistoryEntry,
    PlanState,
    STATE_ABORTED,
    STATE_CRITIQUED,
    STATE_DONE,
    STATE_EXECUTED,
    STATE_FINALIZED,
    STATE_GATED,
    STATE_INITIALIZED,
    STATE_PLANNED,
    StepResponse,
)
from megaplan._core import (
    add_or_increment_debt,
    atomic_write_json,
    atomic_write_text,
    configured_robustness,
    current_iteration_raw_artifact,
    ensure_runtime_layout,
    extract_subsystem_tag,
    latest_plan_meta_path,
    latest_plan_path,
    load_debt_registry,
    load_flag_registry,
    load_plan,
    now_utc,
    plans_root,
    read_json,
    render_final_md,
    save_debt_registry,
    save_flag_registry,
    save_state,
    scope_creep_flags,
    sha256_file,
    sha256_text,
    slugify,
    unresolved_significant_flags,
)
from megaplan.evaluation import (
    PLAN_STRUCTURE_REQUIRED_STEP_ISSUE,
    PlanSection,
    build_gate_artifact,
    build_gate_signals,
    build_orchestrator_guidance,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    is_rubber_stamp,
    parse_plan_sections,
    reassemble_plan,
    renumber_steps,
    run_gate_checks,
    validate_execution_evidence,
    validate_plan_structure,
)
from megaplan._core import find_command, infer_next_steps, require_state
from megaplan.workers import WorkerResult, update_session_state, validate_payload


def _append_to_meta(state: PlanState, field: str, value: Any) -> None:
    state["meta"].setdefault(field, []).append(value)


def apply_session_update(
    state: PlanState,
    step: str,
    agent: str,
    session_id: str | None,
    *,
    mode: str,
    refreshed: bool,
) -> None:
    result = update_session_state(
        step,
        agent,
        session_id,
        mode=mode,
        refreshed=refreshed,
        existing_sessions=state["sessions"],
    )
    if result is not None:
        key, entry = result
        state["sessions"][key] = entry


def append_history(state: PlanState, entry: HistoryEntry) -> None:
    state["history"].append(entry)
    state["meta"].setdefault("total_cost_usd", 0.0)
    state["meta"]["total_cost_usd"] = round(
        float(state["meta"]["total_cost_usd"]) + float(entry.get("cost_usd", 0.0)),
        6,
    )


def next_flag_number(flags: list[FlagRecord]) -> int:
    highest = 0
    for flag in flags:
        match = re.fullmatch(r"FLAG-(\d+)", flag["id"])
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def make_flag_id(number: int) -> str:
    return f"FLAG-{number:03d}"


def resolve_severity(hint: str) -> str:
    if hint == "likely-significant":
        return "significant"
    if hint == "likely-minor":
        return "minor"
    return "significant"


def make_history_entry(
    step: str,
    *,
    duration_ms: int,
    cost_usd: float,
    result: str,
    worker: WorkerResult | None = None,
    agent: str | None = None,
    mode: str | None = None,
    output_file: str | None = None,
    artifact_hash: str | None = None,
    finalize_hash: str | None = None,
    raw_output_file: str | None = None,
    message: str | None = None,
    flags_count: int | None = None,
    flags_addressed: list[str] | None = None,
    recommendation: str | None = None,
    approval_mode: str | None = None,
    environment: dict[str, bool] | None = None,
) -> HistoryEntry:
    entry: HistoryEntry = {
        "step": step,
        "timestamp": now_utc(),
        "duration_ms": duration_ms,
        "cost_usd": cost_usd,
        "result": result,
    }
    if worker is not None and agent is not None and mode is not None:
        entry["session_mode"] = mode
        entry["session_id"] = worker.session_id
        entry["agent"] = agent
    if output_file is not None:
        entry["output_file"] = output_file
    if artifact_hash is not None:
        entry["artifact_hash"] = artifact_hash
    if finalize_hash is not None:
        entry["finalize_hash"] = finalize_hash
    if raw_output_file is not None:
        entry["raw_output_file"] = raw_output_file
    if message is not None:
        entry["message"] = message
    if flags_count is not None:
        entry["flags_count"] = flags_count
    if flags_addressed is not None:
        entry["flags_addressed"] = flags_addressed
    if recommendation is not None:
        entry["recommendation"] = recommendation
    if approval_mode is not None:
        entry["approval_mode"] = approval_mode
    if environment is not None:
        entry["environment"] = environment
    return entry


def attach_agent_fallback(response: StepResponse, args: argparse.Namespace) -> None:
    if hasattr(args, "_agent_fallback"):
        response["agent_fallback"] = args._agent_fallback

def _build_review_blocked_message(
    *,
    verdict_count: int,
    total_tasks: int,
    check_count: int,
    total_checks: int,
    missing_reviewer_evidence: list[str],
) -> str:
    if missing_reviewer_evidence:
        return (
            "Blocked: done tasks are missing reviewer evidence_files without a substantive reviewer_verdict ("
            + ", ".join(missing_reviewer_evidence)
            + "). Re-run review to complete."
        )
    return (
        "Blocked: incomplete review coverage "
        f"({verdict_count}/{total_tasks} task verdicts, {check_count}/{total_checks} sense checks). "
        "Re-run review to complete."
    )


def _is_substantive_reviewer_verdict(text: str) -> bool:
    return not is_rubber_stamp(text, strict=True)


def normalize_flag_record(raw_flag: dict[str, Any], fallback_id: str) -> FlagRecord:
    category = raw_flag.get("category", "other")
    if category not in {"correctness", "security", "completeness", "performance", "maintainability", "other"}:
        category = "other"
    severity_hint = raw_flag.get("severity_hint") or "uncertain"
    if severity_hint not in {"likely-significant", "likely-minor", "uncertain"}:
        severity_hint = "uncertain"
    raw_id = raw_flag.get("id")
    return {
        "id": fallback_id if raw_id in {None, "", "FLAG-000"} else raw_id,
        "concern": raw_flag.get("concern", "").strip(),
        "category": category,
        "severity_hint": severity_hint,
        "evidence": raw_flag.get("evidence", "").strip(),
    }


def update_flags_after_critique(plan_dir: Path, critique: dict[str, Any], *, iteration: int) -> FlagRegistry:
    registry = load_flag_registry(plan_dir)
    flags = registry.setdefault("flags", [])
    by_id: dict[str, FlagRecord] = {flag["id"]: flag for flag in flags}
    next_number = next_flag_number(flags)

    for verified_id in critique.get("verified_flag_ids", []):
        if verified_id in by_id:
            by_id[verified_id]["status"] = "verified"
            by_id[verified_id]["verified"] = True
            by_id[verified_id]["verified_in"] = f"critique_v{iteration}.json"

    for disputed_id in critique.get("disputed_flag_ids", []):
        if disputed_id in by_id:
            by_id[disputed_id]["status"] = "disputed"

    for raw_flag in critique.get("flags", []):
        proposed_id = raw_flag.get("id")
        if not proposed_id or proposed_id in {"", "FLAG-000"}:
            proposed_id = make_flag_id(next_number)
            next_number += 1
        normalized = normalize_flag_record(raw_flag, proposed_id)
        if normalized["id"] in by_id:
            existing = by_id[normalized["id"]]
            existing.update(normalized)
            existing["status"] = "open"
            existing["severity"] = resolve_severity(normalized.get("severity_hint", "uncertain"))
            existing["raised_in"] = f"critique_v{iteration}.json"
        else:
            severity = resolve_severity(normalized.get("severity_hint", "uncertain"))
            created: FlagRecord = {
                **normalized,
                "raised_in": f"critique_v{iteration}.json",
                "status": "open",
                "severity": severity,
                "verified": False,
            }
            flags.append(created)
            by_id[created["id"]] = created

    save_flag_registry(plan_dir, registry)
    return registry


def update_flags_after_revise(plan_dir: Path, flags_addressed: list[str], *, plan_file: str, summary: str) -> FlagRegistry:
    registry = load_flag_registry(plan_dir)
    for flag in registry["flags"]:
        if flag["id"] in flags_addressed:
            flag["status"] = "addressed"
            flag["addressed_in"] = plan_file
            flag["evidence"] = summary
    save_flag_registry(plan_dir, registry)
    return registry


def next_plan_artifact_name(plan_dir: Path, version: int) -> str:
    base = f"plan_v{version}"
    candidate = f"{base}.md"
    if not (plan_dir / candidate).exists():
        return candidate
    suffix_ord = ord("a")
    while True:
        candidate = f"{base}{chr(suffix_ord)}.md"
        if not (plan_dir / candidate).exists():
            return candidate
        suffix_ord += 1


_STEP_EDIT_ALLOWED_STATES = {STATE_PLANNED, STATE_CRITIQUED, STATE_GATED, STATE_FINALIZED}
_STEP_ID_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)


def _normalize_step_id(raw_step_id: str, *, label: str) -> str:
    match = _STEP_ID_RE.fullmatch(raw_step_id.strip())
    if match is None:
        raise CliError("invalid_args", f"{label} must use the form S<number>; got '{raw_step_id}'")
    return f"S{int(match.group(1))}"


def _step_section_index(sections: list[PlanSection], step_id: str) -> int:
    for index, section in enumerate(sections):
        if section.id == step_id:
            return index
    raise CliError("missing_step", f"Plan does not contain step '{step_id}'")


def _last_step_section_index(sections: list[PlanSection]) -> int | None:
    last_index: int | None = None
    for index, section in enumerate(sections):
        if section.id is not None:
            last_index = index
    return last_index


def _default_step_insert_index(sections: list[PlanSection]) -> int:
    last_step_index = _last_step_section_index(sections)
    if last_step_index is not None:
        return last_step_index + 1
    for index, section in enumerate(sections):
        if section.heading in {"## Execution Order", "## Validation Order"}:
            return index
    return len(sections)


def _make_step_scaffold(description: str, *, heading_level: str = "##") -> PlanSection:
    clean_description = description.strip()
    if not clean_description:
        raise CliError("invalid_args", "step add requires a non-empty description")
    body = (
        f"{heading_level} Step 0: {clean_description}\n"
        "**Scope:** Small\n"
        "1. **TODO** Fill in implementation details (`path/to/file`).\n\n"
    )
    return PlanSection(heading=f"{heading_level} Step 0: {clean_description}", body=body, id="S0", start_line=0, end_line=0)


def _detect_step_heading_level(sections: list[PlanSection]) -> str:
    """Detect whether existing steps use ## or ### headings."""
    for section in sections:
        if section.id is not None and section.heading.startswith("### "):
            return "###"
    return "##"


def _commit_step_edit(
    plan_dir: Path,
    state: PlanState,
    sections: list[PlanSection],
    *,
    action: str,
    action_summary: str,
) -> tuple[str, str, list[str]]:
    plan_text = reassemble_plan(sections)
    structure_warnings = validate_plan_structure(plan_text)
    if PLAN_STRUCTURE_REQUIRED_STEP_ISSUE in structure_warnings:
        raise CliError(
            "structure_error",
            f"Step edit failed structural validation: {PLAN_STRUCTURE_REQUIRED_STEP_ISSUE}",
            valid_next=infer_next_steps(state),
        )

    previous_meta_path = latest_plan_meta_path(plan_dir, state)
    previous_meta = read_json(previous_meta_path) if previous_meta_path.exists() else {}
    plan_filename = next_plan_artifact_name(plan_dir, state["iteration"])
    meta_filename = plan_filename.replace(".md", ".meta.json")
    meta = {
        "version": state["iteration"],
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "questions": previous_meta.get("questions", []),
        "success_criteria": previous_meta.get("success_criteria", []),
        "assumptions": previous_meta.get("assumptions", []),
        "structure_warnings": structure_warnings,
        "step_edit": {
            "action": action,
            "action_summary": action_summary,
        },
    }
    atomic_write_text(plan_dir / plan_filename, plan_text)
    atomic_write_json(plan_dir / meta_filename, meta)
    state["current_state"] = STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["last_gate"] = {}
    state["plan_versions"].append(
        {"version": state["iteration"], "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]}
    )
    append_history(
        state,
        make_history_entry(
            "step",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file=plan_filename,
            artifact_hash=meta["hash"],
            message=action_summary,
        ),
    )
    save_state(plan_dir, state)
    return plan_filename, meta_filename, structure_warnings


def _step_add(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    sections = parse_plan_sections(latest_plan_path(plan_dir, state).read_text(encoding="utf-8"))
    new_section = _make_step_scaffold(args.description, heading_level=_detect_step_heading_level(sections))
    if args.after:
        after_step_id = _normalize_step_id(args.after, label="--after")
        insert_index = _step_section_index(sections, after_step_id) + 1
        action_summary = f"Inserted new step after {after_step_id}: {args.description.strip()}"
    else:
        insert_index = _default_step_insert_index(sections)
        action_summary = f"Inserted new step at the end of the current step list: {args.description.strip()}"
    updated_sections = sections[:insert_index] + [new_section] + sections[insert_index:]
    renumbered_sections = renumber_steps(updated_sections)
    plan_filename, meta_filename, structure_warnings = _commit_step_edit(
        plan_dir,
        state,
        renumbered_sections,
        action="add",
        action_summary=action_summary,
    )
    return {
        "success": True,
        "step": "step",
        "summary": f"{action_summary}. Wrote {plan_filename} and reset the plan to planned state.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": "critique",
        "state": STATE_PLANNED,
        "warnings": structure_warnings,
    }


def _step_remove(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    step_id = _normalize_step_id(args.step_id, label="step_id")
    sections = parse_plan_sections(latest_plan_path(plan_dir, state).read_text(encoding="utf-8"))
    step_indexes = [index for index, section in enumerate(sections) if section.id is not None]
    if len(step_indexes) <= 1:
        raise CliError("invalid_step_edit", "Cannot remove the last remaining step from a plan")
    remove_index = _step_section_index(sections, step_id)
    updated_sections = sections[:remove_index] + sections[remove_index + 1 :]
    renumbered_sections = renumber_steps(updated_sections)
    action_summary = f"Removed {step_id} and renumbered remaining steps"
    plan_filename, meta_filename, structure_warnings = _commit_step_edit(
        plan_dir,
        state,
        renumbered_sections,
        action="remove",
        action_summary=action_summary,
    )
    return {
        "success": True,
        "step": "step",
        "summary": f"{action_summary}. Wrote {plan_filename} and reset the plan to planned state.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": "critique",
        "state": STATE_PLANNED,
        "warnings": structure_warnings,
    }


def _step_move(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    step_id = _normalize_step_id(args.step_id, label="step_id")
    after_step_id = _normalize_step_id(args.after, label="--after")
    if step_id == after_step_id:
        raise CliError("invalid_step_edit", f"Cannot move {step_id} after itself")

    sections = parse_plan_sections(latest_plan_path(plan_dir, state).read_text(encoding="utf-8"))
    move_index = _step_section_index(sections, step_id)
    moving_section = sections[move_index]
    remaining_sections = sections[:move_index] + sections[move_index + 1 :]
    insert_index = _step_section_index(remaining_sections, after_step_id) + 1
    updated_sections = remaining_sections[:insert_index] + [moving_section] + remaining_sections[insert_index:]
    renumbered_sections = renumber_steps(updated_sections)
    action_summary = f"Moved {step_id} after {after_step_id} and renumbered the plan"
    plan_filename, meta_filename, structure_warnings = _commit_step_edit(
        plan_dir,
        state,
        renumbered_sections,
        action="move",
        action_summary=action_summary,
    )
    return {
        "success": True,
        "step": "step",
        "summary": f"{action_summary}. Wrote {plan_filename} and reset the plan to planned state.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": "critique",
        "state": STATE_PLANNED,
        "warnings": structure_warnings,
    }


_STEP_ACTIONS: dict[str, Callable[[Path, PlanState, argparse.Namespace], StepResponse]] = {
    "add": _step_add,
    "remove": _step_remove,
    "move": _step_move,
}


def store_raw_worker_output(plan_dir: Path, step: str, iteration: int, content: str) -> str:
    filename = current_iteration_raw_artifact(plan_dir, step, iteration).name
    atomic_write_text(plan_dir / filename, content)
    return filename


def record_step_failure(
    plan_dir: Path,
    state: PlanState,
    *,
    step: str,
    iteration: int,
    error: CliError,
    duration_ms: int = 0,
) -> None:
    raw_output = str(error.extra.get("raw_output") or error.message)
    raw_name = store_raw_worker_output(plan_dir, step, iteration, raw_output)
    append_history(
        state,
        make_history_entry(
            step,
            duration_ms=duration_ms,
            cost_usd=0.0,
            result="error",
            raw_output_file=raw_name,
            message=error.message,
        ),
    )
    save_state(plan_dir, state)


def _validate_generated_plan_or_raise(
    *,
    plan_dir: Path,
    state: PlanState,
    step: str,
    iteration: int,
    worker: WorkerResult,
    plan_text: str,
) -> list[str]:
    structure_warnings = validate_plan_structure(plan_text)
    if PLAN_STRUCTURE_REQUIRED_STEP_ISSUE in structure_warnings:
        error = CliError(
            "structure_error",
            f"{step.title()} output failed structural validation: {PLAN_STRUCTURE_REQUIRED_STEP_ISSUE}",
            valid_next=infer_next_steps(state),
            extra={"raw_output": worker.raw_output},
        )
        record_step_failure(
            plan_dir,
            state,
            step=step,
            iteration=iteration,
            error=error,
            duration_ms=worker.duration_ms,
        )
        raise error
    return structure_warnings


def _store_last_gate(state: PlanState, gate_summary: dict[str, Any]) -> None:
    state["last_gate"] = {
        "recommendation": gate_summary["recommendation"],
        "rationale": gate_summary["rationale"],
        "signals_assessment": gate_summary["signals_assessment"],
        "warnings": gate_summary["warnings"],
        "settled_decisions": gate_summary.get("settled_decisions", []),
        "passed": gate_summary["passed"],
        "preflight_results": gate_summary["preflight_results"],
        "orchestrator_guidance": gate_summary["orchestrator_guidance"],
    }


def _apply_gate_outcome(state: PlanState, gate_summary: dict[str, Any], *, robustness: str) -> tuple[str, str, str]:
    result = "success"
    summary = f"Gate recommendation {gate_summary['recommendation']}: {gate_summary['rationale']}"
    if gate_summary["recommendation"] == "PROCEED" and gate_summary["passed"]:
        state["current_state"] = STATE_GATED
        state["meta"].pop("user_approved_gate", None)
        return result, "finalize", summary
    state["current_state"] = STATE_CRITIQUED
    if gate_summary["recommendation"] == "PROCEED":
        result = "blocked"
        summary = "Gate recommended PROCEED, but preflight checks are still blocking execution."
        return result, "revise", summary
    if gate_summary["recommendation"] == "ITERATE":
        return result, "revise", summary
    if gate_summary["recommendation"] == "ESCALATE":
        if robustness == "light" and gate_summary["signals"]["weighted_score"] <= 4.0:
            return result, "override force-proceed", summary
        return result, "override add-note", summary
    result = "unknown_recommendation"
    summary = f"Gate returned unknown recommendation '{gate_summary['recommendation']}'; treating as escalation."
    return result, "override add-note", summary


def _synthetic_signals_assessment(signals_artifact: dict[str, Any]) -> str:
    signals = signals_artifact["signals"]
    parts = [f"Iteration {signals.get('iteration', '?')} weighted score {signals.get('weighted_score', 0.0)}."]
    weighted_history = list(signals.get("weighted_history", []))
    if weighted_history:
        parts.append(f"Previous weighted score {weighted_history[-1]}.")
    delta = signals.get("plan_delta_from_previous")
    if delta is not None:
        parts.append(f"Plan delta from previous iteration is {delta}%.")
    recurring = list(signals.get("recurring_critiques", []))
    if recurring:
        parts.append(f"Recurring critiques: {', '.join(recurring)}.")
    unresolved = list(signals_artifact.get("unresolved_flags", []))
    parts.append(f"{len(unresolved)} unresolved significant flag(s) remain.")
    if all(signals_artifact["preflight_results"].values()):
        parts.append("Preflight is clean.")
    else:
        blocked = ", ".join(
            name for name, passed in signals_artifact["preflight_results"].items() if not passed
        )
        parts.append(f"Preflight is blocked by: {blocked}.")
    return " ".join(parts)


def _maybe_fast_forward_light_plan(
    *,
    root: Path,
    plan_dir: Path,
    state: PlanState,
    payload: dict[str, Any],
    version: int,
    plan_filename: str,
    meta_filename: str,
) -> StepResponse | None:
    if configured_robustness(state) != "light":
        return None
    if not isinstance(payload.get("self_flags"), list):
        return None
    recommendation = payload.get("gate_recommendation")
    if recommendation not in {"PROCEED", "ITERATE", "ESCALATE"}:
        return None

    critique_payload = {
        "flags": payload.get("self_flags", []),
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    critique_filename = f"critique_v{version}.json"
    atomic_write_json(plan_dir / critique_filename, critique_payload)
    registry = update_flags_after_critique(plan_dir, critique_payload, iteration=version)
    significant = len(
        [
            flag
            for flag in registry["flags"]
            if flag.get("severity") == "significant" and flag["status"] in FLAG_BLOCKING_STATUSES
        ]
    )
    _append_to_meta(state, "significant_counts", significant)
    recurring = compute_recurring_critiques(plan_dir, version)
    _append_to_meta(state, "recurring_critiques", recurring)
    state["current_state"] = STATE_CRITIQUED
    append_history(
        state,
        make_history_entry(
            "critique",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file=critique_filename,
            artifact_hash=sha256_file(plan_dir / critique_filename),
            flags_count=len(critique_payload["flags"]),
            message="Synthetic light-mode critique generated from the combined plan output.",
        ),
    )

    gate_signals = build_gate_signals(plan_dir, state, root=root)
    gate_checks = run_gate_checks(plan_dir, state, command_lookup=find_command)
    signals_artifact = {
        "robustness": gate_signals["robustness"],
        "signals": gate_signals["signals"],
        "warnings": gate_signals.get("warnings", []),
        "criteria_check": gate_checks["criteria_check"],
        "preflight_results": gate_checks["preflight_results"],
        "unresolved_flags": gate_checks["unresolved_flags"],
    }
    signals_filename = f"gate_signals_v{version}.json"
    atomic_write_json(plan_dir / signals_filename, signals_artifact)
    gate_payload = {
        "recommendation": recommendation,
        "rationale": payload.get("gate_rationale", ""),
        "signals_assessment": _synthetic_signals_assessment(signals_artifact),
        "warnings": [],
        "settled_decisions": payload.get("settled_decisions", []),
    }
    guidance = build_orchestrator_guidance(
        gate_payload=gate_payload,
        signals=signals_artifact["signals"],
        preflight_passed=all(signals_artifact["preflight_results"].values()),
        preflight_results=signals_artifact["preflight_results"],
        robustness=signals_artifact.get("robustness", "standard"),
        plan_name=state["name"],
    )
    gate_summary = build_gate_artifact(
        signals_artifact,
        gate_payload,
        override_forced=False,
        orchestrator_guidance=guidance,
    )
    atomic_write_json(plan_dir / "gate.json", gate_summary)
    _store_last_gate(state, gate_summary)
    if len(state["meta"].get("weighted_scores", [])) < version:
        _append_to_meta(state, "weighted_scores", gate_signals["signals"]["weighted_score"])
    result, next_step, gate_summary_text = _apply_gate_outcome(
        state,
        gate_summary,
        robustness=gate_signals["robustness"],
    )
    append_history(
        state,
        make_history_entry(
            "gate",
            duration_ms=0,
            cost_usd=0.0,
            result=result,
            output_file="gate.json",
            artifact_hash=sha256_file(plan_dir / "gate.json"),
            recommendation=gate_summary["recommendation"],
            message="Synthetic light-mode gate generated from the combined plan output.",
        ),
    )
    return {
        "success": gate_summary["recommendation"] != "PROCEED" or gate_summary["passed"],
        "step": "plan",
        "iteration": version,
        "summary": (
            f"Generated light robustness plan v{version} with {len(payload['questions'])} questions and "
            f"{len(payload['success_criteria'])} success criteria. {gate_summary_text}"
        ),
        "artifacts": [plan_filename, meta_filename, critique_filename, "faults.json", signals_filename, "gate.json"],
        "next_step": next_step,
        "state": state["current_state"],
        "questions": payload["questions"],
        "assumptions": payload["assumptions"],
        "success_criteria": payload["success_criteria"],
        "recommendation": gate_summary["recommendation"],
        "rationale": gate_summary["rationale"],
        "signals_assessment": gate_summary["signals_assessment"],
        "warnings": gate_summary["warnings"],
        "passed": gate_summary["passed"],
        "criteria_check": gate_summary["criteria_check"],
        "preflight_results": gate_summary["preflight_results"],
        "unresolved_flags": gate_summary["unresolved_flags"],
        "orchestrator_guidance": gate_summary["orchestrator_guidance"],
        "signals": gate_summary["signals"],
    }


def handle_step(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "step", _STEP_EDIT_ALLOWED_STATES)
    handler = _STEP_ACTIONS.get(args.step_action)
    if handler is None:
        raise CliError("invalid_args", f"Unknown step action: {args.step_action}")
    return handler(plan_dir, state, args)


def handle_init(root: Path, args: argparse.Namespace) -> StepResponse:
    ensure_runtime_layout(root)
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise CliError("invalid_project_dir", f"Project directory does not exist: {project_dir}")
    robustness = getattr(args, "robustness", "standard")
    if robustness not in ROBUSTNESS_LEVELS:
        robustness = "standard"
    auto_approve = bool(getattr(args, "auto_approve", False))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    plan_name = args.name or f"{slugify(args.idea)}-{timestamp}"
    plan_dir = plans_root(root) / plan_name
    if plan_dir.exists():
        raise CliError("duplicate_plan", f"Plan directory already exists: {plan_name}")
    plan_dir.mkdir(parents=True, exist_ok=False)

    state: PlanState = {
        "name": plan_name,
        "idea": args.idea,
        "current_state": STATE_INITIALIZED,
        "iteration": 0,
        "created_at": now_utc(),
        "config": {
            "project_dir": str(project_dir),
            "auto_approve": auto_approve,
            "robustness": robustness,
        },
        "sessions": {},
        "plan_versions": [],
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
    append_history(
        state,
        make_history_entry(
            "init",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            environment={
                "claude": bool(find_command("claude")),
                "codex": bool(find_command("codex")),
            },
        ),
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "init",
        "plan": plan_name,
        "state": STATE_INITIALIZED,
        "summary": f"Initialized plan '{plan_name}' for project {project_dir}",
        "artifacts": ["state.json"],
        "next_step": "plan",
        "auto_approve": auto_approve,
        "robustness": robustness,
    }


def handle_plan(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "plan", {STATE_INITIALIZED, STATE_PLANNED})
    rerun = state["current_state"] == STATE_PLANNED
    version = state["iteration"] if rerun else state["iteration"] + 1
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("plan", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="plan", iteration=version, error=error)
        raise
    payload = worker.payload
    plan_filename = next_plan_artifact_name(plan_dir, version)
    meta_filename = f"plan_v{version}.meta.json"
    if plan_filename != f"plan_v{version}.md":
        meta_filename = plan_filename.replace(".md", ".meta.json")
    plan_text = payload["plan"].rstrip() + "\n"
    structure_warnings = _validate_generated_plan_or_raise(
        plan_dir=plan_dir,
        state=state,
        step="plan",
        iteration=version,
        worker=worker,
        plan_text=plan_text,
    )
    atomic_write_text(plan_dir / plan_filename, plan_text)
    meta = {
        "version": version,
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "questions": payload["questions"],
        "success_criteria": payload["success_criteria"],
        "assumptions": payload["assumptions"],
        "structure_warnings": structure_warnings,
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    state["iteration"] = version
    state["current_state"] = STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["last_gate"] = {}
    state["plan_versions"].append(
        {"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]}
    )
    apply_session_update(state, "plan", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "plan",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=plan_filename,
            artifact_hash=meta["hash"],
        ),
    )
    response = _maybe_fast_forward_light_plan(
        root=root,
        plan_dir=plan_dir,
        state=state,
        payload=payload,
        version=version,
        plan_filename=plan_filename,
        meta_filename=meta_filename,
    )
    if response is None:
        response = {
            "success": True,
            "step": "plan",
            "iteration": version,
            "summary": (
                f"{'Refined' if rerun else 'Generated'} plan v{version} with "
                f"{len(payload['questions'])} questions and {len(payload['success_criteria'])} success criteria."
            ),
            "artifacts": [plan_filename, meta_filename],
            "next_step": "critique",
            "state": STATE_PLANNED,
            "questions": payload["questions"],
            "assumptions": payload["assumptions"],
            "success_criteria": payload["success_criteria"],
        }
    save_state(plan_dir, state)
    attach_agent_fallback(response, args)
    return response


def handle_critique(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "critique", {STATE_PLANNED})
    iteration = state["iteration"]
    state["last_gate"] = {}
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("critique", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="critique", iteration=iteration, error=error)
        raise
    critique_filename = f"critique_v{iteration}.json"
    atomic_write_json(plan_dir / critique_filename, worker.payload)
    registry = update_flags_after_critique(plan_dir, worker.payload, iteration=iteration)
    significant = len(
        [
            flag
            for flag in registry["flags"]
            if flag.get("severity") == "significant" and flag["status"] in FLAG_BLOCKING_STATUSES
        ]
    )
    _append_to_meta(state, "significant_counts", significant)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    _append_to_meta(state, "recurring_critiques", recurring)
    state["current_state"] = STATE_CRITIQUED
    apply_session_update(state, "critique", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "critique",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=critique_filename,
            artifact_hash=sha256_file(plan_dir / critique_filename),
            flags_count=len(worker.payload.get("flags", [])),
        ),
    )
    save_state(plan_dir, state)
    scope_flags_list = scope_creep_flags(registry, statuses=FLAG_BLOCKING_STATUSES)
    open_flags_detail = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "category": flag["category"],
            "severity": flag.get("severity", "unknown"),
        }
        for flag in registry["flags"]
        if flag["status"] == "open"
    ]
    response: StepResponse = {
        "success": True,
        "step": "critique",
        "iteration": iteration,
        "summary": f"Recorded {len(worker.payload.get('flags', []))} critique flags.",
        "artifacts": [critique_filename, "faults.json"],
        "next_step": "gate",
        "state": STATE_CRITIQUED,
        "verified_flags": worker.payload.get("verified_flag_ids", []),
        "open_flags": open_flags_detail,
        "scope_creep_flags": [flag["id"] for flag in scope_flags_list],
    }
    if scope_flags_list:
        response["warnings"] = [
            "Scope creep detected in the plan. Surface this drift to the user while continuing the loop."
        ]
    attach_agent_fallback(response, args)
    return response


def handle_revise(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "revise", {STATE_CRITIQUED})
    if state["last_gate"].get("recommendation") != "ITERATE":
        raise CliError(
            "invalid_transition",
            "Revise requires a gate recommendation of ITERATE",
            valid_next=infer_next_steps(state),
        )
    previous_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("revise", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="revise", iteration=state["iteration"] + 1, error=error)
        raise
    payload = worker.payload
    validate_payload("revise", payload)
    version = state["iteration"] + 1
    plan_filename = f"plan_v{version}.md"
    meta_filename = f"plan_v{version}.meta.json"
    plan_text = payload["plan"].rstrip() + "\n"
    structure_warnings = _validate_generated_plan_or_raise(
        plan_dir=plan_dir,
        state=state,
        step="revise",
        iteration=version,
        worker=worker,
        plan_text=plan_text,
    )
    atomic_write_text(plan_dir / plan_filename, plan_text)
    delta = compute_plan_delta_percent(previous_plan, plan_text)
    meta = {
        "version": version,
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "changes_summary": payload["changes_summary"],
        "flags_addressed": payload["flags_addressed"],
        "questions": payload.get("questions", []),
        "success_criteria": payload.get("success_criteria", []),
        "assumptions": payload.get("assumptions", []),
        "structure_warnings": structure_warnings,
        "delta_from_previous_percent": delta,
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    state["iteration"] = version
    state["current_state"] = STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["last_gate"] = {}
    state["plan_versions"].append(
        {"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]}
    )
    _append_to_meta(state, "plan_deltas", delta)
    update_flags_after_revise(
        plan_dir,
        payload["flags_addressed"],
        plan_file=plan_filename,
        summary=payload["changes_summary"],
    )
    apply_session_update(state, "revise", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "revise",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=plan_filename,
            artifact_hash=meta["hash"],
            flags_addressed=payload["flags_addressed"],
        ),
    )
    save_state(plan_dir, state)
    updated_registry = load_flag_registry(plan_dir)
    remaining = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "category": flag["category"],
        }
        for flag in updated_registry["flags"]
        if flag["status"] in FLAG_BLOCKING_STATUSES and flag.get("severity") == "significant"
    ]
    response: StepResponse = {
        "success": True,
        "step": "revise",
        "iteration": version,
        "summary": f"Updated plan to v{version}; addressed {len(payload['flags_addressed'])} flags.",
        "artifacts": [plan_filename, meta_filename, "faults.json"],
        "next_step": "critique",
        "state": STATE_PLANNED,
        "changes_summary": payload["changes_summary"],
        "flags_addressed": payload["flags_addressed"],
        "flags_remaining": remaining,
        "plan_delta_percent": delta,
    }
    attach_agent_fallback(response, args)
    return response


def handle_gate(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "gate", {STATE_CRITIQUED})
    iteration = state["iteration"]
    gate_signals = build_gate_signals(plan_dir, state, root=root)
    gate_checks = run_gate_checks(plan_dir, state, command_lookup=find_command)
    signals_artifact = {
        "robustness": gate_signals["robustness"],
        "signals": gate_signals["signals"],
        "warnings": gate_signals.get("warnings", []),
        "criteria_check": gate_checks["criteria_check"],
        "preflight_results": gate_checks["preflight_results"],
        "unresolved_flags": gate_checks["unresolved_flags"],
    }
    signals_filename = f"gate_signals_v{iteration}.json"
    atomic_write_json(plan_dir / signals_filename, signals_artifact)
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("gate", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="gate", iteration=iteration, error=error)
        raise
    guidance = build_orchestrator_guidance(
        gate_payload=worker.payload,
        signals=signals_artifact["signals"],
        preflight_passed=all(signals_artifact["preflight_results"].values()),
        preflight_results=signals_artifact["preflight_results"],
        robustness=signals_artifact.get("robustness", "standard"),
        plan_name=state["name"],
    )
    gate_summary = build_gate_artifact(
        signals_artifact,
        worker.payload,
        override_forced=False,
        orchestrator_guidance=guidance,
    )
    atomic_write_json(plan_dir / "gate.json", gate_summary)
    debt_entries_added = 0
    if gate_summary["recommendation"] == "PROCEED":
        raw_tradeoffs = worker.payload.get("accepted_tradeoffs", [])
        accepted_tradeoffs = [
            item
            for item in raw_tradeoffs
            if isinstance(item, dict)
            and isinstance(item.get("flag_id"), str)
            and isinstance(item.get("concern"), str)
        ] if isinstance(raw_tradeoffs, list) else []
        debt_registry = load_debt_registry(root)
        if accepted_tradeoffs:
            for tradeoff in accepted_tradeoffs:
                subsystem_value = tradeoff.get("subsystem")
                subsystem = (
                    subsystem_value
                    if isinstance(subsystem_value, str) and subsystem_value.strip()
                    else extract_subsystem_tag(tradeoff["concern"])
                )
                add_or_increment_debt(
                    debt_registry,
                    subsystem=subsystem,
                    concern=tradeoff["concern"],
                    flag_ids=[tradeoff["flag_id"]],
                    plan_id=state["name"],
                )
                debt_entries_added += 1
        elif gate_summary["unresolved_flags"]:
            for flag in gate_summary["unresolved_flags"]:
                if not isinstance(flag, dict):
                    continue
                flag_id = flag.get("id")
                concern = flag.get("concern")
                if not isinstance(flag_id, str) or not isinstance(concern, str):
                    continue
                add_or_increment_debt(
                    debt_registry,
                    subsystem=extract_subsystem_tag(concern),
                    concern=concern,
                    flag_ids=[flag_id],
                    plan_id=state["name"],
                )
                debt_entries_added += 1
        if debt_entries_added:
            save_debt_registry(root, debt_registry)
    _store_last_gate(state, gate_summary)
    if len(state["meta"].get("weighted_scores", [])) < iteration:
        _append_to_meta(state, "weighted_scores", gate_signals["signals"]["weighted_score"])

    artifacts = [signals_filename, "gate.json"]
    result, next_step, summary = _apply_gate_outcome(
        state,
        gate_summary,
        robustness=gate_signals["robustness"],
    )

    apply_session_update(state, "gate", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "gate",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result=result,
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="gate.json",
            artifact_hash=sha256_file(plan_dir / "gate.json"),
            recommendation=gate_summary["recommendation"],
        ),
    )
    save_state(plan_dir, state)
    response: StepResponse = {
        "success": gate_summary["recommendation"] != "PROCEED" or gate_summary["passed"],
        "step": "gate",
        "summary": summary,
        "artifacts": artifacts,
        "next_step": next_step,
        "state": state["current_state"],
        "auto_approve": bool(state["config"].get("auto_approve", False)),
        "robustness": configured_robustness(state),
        "recommendation": gate_summary["recommendation"],
        "rationale": gate_summary["rationale"],
        "signals_assessment": gate_summary["signals_assessment"],
        "warnings": gate_summary["warnings"],
        "passed": gate_summary["passed"],
        "criteria_check": gate_summary["criteria_check"],
        "preflight_results": gate_summary["preflight_results"],
        "unresolved_flags": gate_summary["unresolved_flags"],
        "orchestrator_guidance": gate_summary["orchestrator_guidance"],
        "signals": gate_summary["signals"],
        "debt_entries_added": debt_entries_added,
    }
    attach_agent_fallback(response, args)
    return response


def handle_finalize(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "finalize", {STATE_GATED})
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("finalize", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="finalize", iteration=state["iteration"], error=error)
        raise
    payload = worker.payload
    atomic_write_json(plan_dir / "finalize.json", payload)
    atomic_write_text(plan_dir / "final.md", render_final_md(payload))
    state["current_state"] = STATE_FINALIZED
    apply_session_update(state, "finalize", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "finalize",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="finalize.json",
            artifact_hash=sha256_file(plan_dir / "finalize.json"),
        ),
    )
    save_state(plan_dir, state)
    response: StepResponse = {
        "success": True,
        "step": "finalize",
        "summary": f"Finalized plan with {len(payload['tasks'])} tasks and {len(payload['watch_items'])} watch items.",
        "artifacts": ["final.md", "finalize.json"],
        "next_step": "execute",
        "state": STATE_FINALIZED,
    }
    attach_agent_fallback(response, args)
    return response


def _is_rework_reexecution(state: PlanState) -> bool:
    """Check if the last completed step was a review with needs_rework."""
    for entry in reversed(state.get("history", [])):
        if entry.get("step") == "review" and entry.get("result") == "needs_rework":
            return True
        if entry.get("step") == "execute":
            return False
    return False


def handle_execute(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "execute", {STATE_FINALIZED})
    if not args.confirm_destructive:
        raise CliError("missing_confirmation", "Execute requires --confirm-destructive")
    auto_approve = bool(state["config"].get("auto_approve", False))
    if getattr(args, "user_approved", False):
        state["meta"]["user_approved_gate"] = True
        save_state(plan_dir, state)
    if not auto_approve and not state["meta"].get("user_approved_gate", False):
        raise CliError(
            "missing_approval",
            "Execute requires explicit user approval (--user-approved) when auto-approve is not set. The orchestrator must confirm with the user at the gate checkpoint before proceeding.",
        )
    agent, mode, refreshed = worker_module.resolve_agent_mode("execute", args)
    # Force fresh session after review kickback to avoid prior-context bias
    if not refreshed and _is_rework_reexecution(state):
        refreshed = True
    if getattr(args, "batch", None) is not None:
        return dispatch_execute_one_batch(
            root=root,
            plan_dir=plan_dir,
            state=state,
            args=args,
            batch_number=args.batch,
            auto_approve=auto_approve,
            agent=agent,
            mode=mode,
            refreshed=refreshed,
            apply_session_update_fn=apply_session_update,
            append_history_fn=append_history,
            make_history_entry_fn=make_history_entry,
            save_state_fn=save_state,
            attach_agent_fallback_fn=attach_agent_fallback,
            store_raw_worker_output_fn=store_raw_worker_output,
            record_step_failure_fn=record_step_failure,
            capture_git_status_snapshot_fn=_capture_git_status_snapshot,
        )
    return dispatch_execute_auto_loop(
        root=root,
        plan_dir=plan_dir,
        state=state,
        args=args,
        auto_approve=auto_approve,
        agent=agent,
        mode=mode,
        refreshed=refreshed,
        apply_session_update_fn=apply_session_update,
        append_history_fn=append_history,
        make_history_entry_fn=make_history_entry,
        save_state_fn=save_state,
        attach_agent_fallback_fn=attach_agent_fallback,
        store_raw_worker_output_fn=store_raw_worker_output,
        record_step_failure_fn=record_step_failure,
        capture_git_status_snapshot_fn=_capture_git_status_snapshot,
    )


def handle_review(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "review", {STATE_EXECUTED})
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("review", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="review", iteration=state["iteration"], error=error)
        raise
    atomic_write_json(plan_dir / "review.json", worker.payload)
    issues = list(worker.payload.get("issues", []))
    finalize_data = read_json(plan_dir / "finalize.json")
    review_verdict = worker.payload.get("review_verdict")
    if review_verdict not in {"approved", "needs_rework"}:
        issues.append("Invalid review_verdict; expected 'approved' or 'needs_rework'.")
        review_verdict = "needs_rework"
    tasks_by_id = {task["id"]: task for task in finalize_data.get("tasks", [])}
    verdict_count, total_tasks = _validate_and_merge_batch(
        worker.payload.get("task_verdicts"),
        required_fields=("task_id", "reviewer_verdict", "evidence_files"),
        targets_by_id=tasks_by_id,
        id_field="task_id",
        merge_fields=("reviewer_verdict", "evidence_files"),
        issues=issues,
        validation_label="task_verdicts",
        merge_label="task_verdict",
        incomplete_message=(
            lambda merged_count, total: (
                f"Incomplete review: {merged_count}/{total} tasks received a reviewer verdict."
            )
        ),
        nonempty_fields={"reviewer_verdict"},
        array_fields=("evidence_files",),
    )
    sense_checks_by_id = {sense_check["id"]: sense_check for sense_check in finalize_data.get("sense_checks", [])}
    check_count, total_checks = _validate_and_merge_batch(
        worker.payload.get("sense_check_verdicts"),
        required_fields=("sense_check_id", "verdict"),
        targets_by_id=sense_checks_by_id,
        id_field="sense_check_id",
        merge_fields=("verdict",),
        issues=issues,
        validation_label="sense_check_verdicts",
        merge_label="sense_check_verdict",
        incomplete_message=(
            lambda merged_count, total: (
                f"Incomplete review: {merged_count}/{total} sense checks received a verdict."
            )
        ),
        nonempty_fields={"verdict"},
    )
    missing_reviewer_evidence = _check_done_task_evidence(
        finalize_data.get("tasks", []),
        issues=issues,
        should_classify=lambda task: bool(task.get("reviewer_verdict", "").strip()),
        has_evidence=lambda task: bool(task.get("evidence_files")),
        has_advisory_evidence=(
            lambda task: _is_substantive_reviewer_verdict(task.get("reviewer_verdict", ""))
        ),
        missing_message="Done tasks missing reviewer evidence_files without a substantive reviewer_verdict: ",
        advisory_message=(
            "Advisory: done tasks rely on substantive reviewer_verdict without evidence_files (FLAG-006 softening): "
        ),
    )

    atomic_write_json(plan_dir / "finalize.json", finalize_data)
    atomic_write_text(plan_dir / "final.md", render_final_md(finalize_data, phase="review"))
    finalize_hash = sha256_file(plan_dir / "finalize.json")
    passed = sum(1 for criterion in worker.payload.get("criteria", []) if criterion.get("pass"))
    total = len(worker.payload.get("criteria", []))
    blocked = (
        verdict_count < total_tasks
        or check_count < total_checks
        or bool(missing_reviewer_evidence)
    )
    rework_requested = not blocked and review_verdict == "needs_rework"
    if not blocked and not rework_requested:
        state["current_state"] = STATE_DONE
    elif rework_requested:
        state["current_state"] = STATE_FINALIZED
    apply_session_update(state, "review", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "review",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="blocked" if blocked else "needs_rework" if rework_requested else "success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="review.json",
            artifact_hash=sha256_file(plan_dir / "review.json"),
            finalize_hash=finalize_hash,
        ),
    )
    save_state(plan_dir, state)
    blocked_message = _build_review_blocked_message(
        verdict_count=verdict_count,
        total_tasks=total_tasks,
        check_count=check_count,
        total_checks=total_checks,
        missing_reviewer_evidence=missing_reviewer_evidence,
    )
    reroute_message = "Review requested another execute pass. Re-run execute using the review findings as context."
    response: StepResponse = {
        "success": not blocked and not rework_requested,
        "step": "review",
        "summary": (
            blocked_message
            if blocked
            else reroute_message
            if rework_requested
            else f"Review complete: {passed}/{total} success criteria passed."
        ),
        "artifacts": ["review.json", "finalize.json", "final.md"],
        "next_step": "review" if blocked else "execute" if rework_requested else None,
        "state": STATE_EXECUTED if blocked else STATE_FINALIZED if rework_requested else STATE_DONE,
        "issues": issues,
    }
    attach_agent_fallback(response, args)
    return response


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def _override_add_note(root: Path, plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    note = args.note
    _append_to_meta(state, "notes", {"timestamp": now_utc(), "note": note})
    _append_to_meta(state, "overrides", {"action": "add-note", "timestamp": now_utc(), "note": note})
    save_state(plan_dir, state)
    next_steps = infer_next_steps(state)
    return {
        "success": True,
        "step": "override",
        "summary": "Attached note to the plan.",
        "next_step": next_steps[0] if next_steps else None,
        "state": state["current_state"],
    }


def _override_abort(root: Path, plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    state["current_state"] = STATE_ABORTED
    _append_to_meta(state, "overrides", {"action": "abort", "timestamp": now_utc(), "reason": args.reason})
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": "Plan aborted.",
        "next_step": None,
        "state": STATE_ABORTED,
    }


def _override_force_proceed(root: Path, plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    if state["current_state"] != STATE_CRITIQUED:
        raise CliError(
            "invalid_transition",
            "force-proceed is only supported from critiqued state",
            valid_next=infer_next_steps(state),
        )
    gate_checks = run_gate_checks(plan_dir, state, command_lookup=find_command)
    if not gate_checks["preflight_results"]["project_dir_exists"] or not gate_checks["preflight_results"]["success_criteria_present"]:
        raise CliError("unsafe_override", "force-proceed cannot bypass missing project directory or success criteria")
    signals = build_gate_signals(plan_dir, state, root=root)
    merged_signals = {
        "robustness": signals["robustness"],
        "signals": signals["signals"],
        "warnings": signals.get("warnings", []),
        "criteria_check": gate_checks["criteria_check"],
        "preflight_results": gate_checks["preflight_results"],
        "unresolved_flags": gate_checks["unresolved_flags"],
    }
    gate = build_gate_artifact(
        merged_signals,
        {
            "recommendation": "PROCEED",
            "rationale": args.reason or "User forced execution past the gate.",
            "signals_assessment": "Forced proceed override applied by the orchestrator.",
            "warnings": signals.get("warnings", []),
        },
        override_forced=True,
        orchestrator_guidance="Force-proceed override applied. Proceed to finalize.",
    )
    atomic_write_json(plan_dir / "gate.json", gate)
    flag_registry = load_flag_registry(plan_dir)
    unresolved_flags = unresolved_significant_flags(flag_registry)
    debt_registry = load_debt_registry(root)
    for flag in unresolved_flags:
        add_or_increment_debt(
            debt_registry,
            subsystem=extract_subsystem_tag(flag["concern"]),
            concern=flag["concern"],
            flag_ids=[flag["id"]],
            plan_id=state["name"],
        )
    save_debt_registry(root, debt_registry)
    state["current_state"] = STATE_GATED
    state["meta"].pop("user_approved_gate", None)
    state["last_gate"] = {}
    _append_to_meta(state, "overrides", {"action": "force-proceed", "timestamp": now_utc(), "reason": args.reason})
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": "Force-proceeded past gate judgment into gated state.",
        "next_step": "finalize",
        "state": STATE_GATED,
        "orchestrator_guidance": gate["orchestrator_guidance"],
        "debt_entries_added": len(unresolved_flags),
    }


def _override_replan(root: Path, plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    allowed = {STATE_GATED, STATE_FINALIZED, STATE_CRITIQUED}
    if state["current_state"] not in allowed:
        raise CliError(
            "invalid_transition",
            f"replan requires state {', '.join(sorted(allowed))}, got '{state['current_state']}'",
            valid_next=infer_next_steps(state),
        )
    reason = args.reason or args.note or "Re-entering planning loop"
    plan_file = latest_plan_path(plan_dir, state)
    state["current_state"] = STATE_PLANNED
    state["last_gate"] = {}
    _append_to_meta(state, "overrides", {"action": "replan", "timestamp": now_utc(), "reason": reason})
    if args.note:
        _append_to_meta(state, "notes", {"timestamp": now_utc(), "note": args.note})
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": f"Re-entered planning loop at iteration {state['iteration']}. Reason: {reason}",
        "next_step": "critique",
        "state": STATE_PLANNED,
        "plan_file": str(plan_file),
        "message": f"Edit {plan_file.name} to incorporate your changes, then run critique.",
    }


_OVERRIDE_ACTIONS: dict[str, Callable[[Path, Path, PlanState, argparse.Namespace], StepResponse]] = {
    "add-note": _override_add_note,
    "abort": _override_abort,
    "force-proceed": _override_force_proceed,
    "replan": _override_replan,
}


def handle_override(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    action = args.override_action
    handler = _OVERRIDE_ACTIONS.get(action)
    if handler is None:
        raise CliError("invalid_override", f"Unknown override action: {action}")
    return handler(root, plan_dir, state, args)
