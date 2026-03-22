#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import re
import shutil  # noqa: F401 — tests monkeypatch megaplan.cli.shutil.which
import sys
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Callable

# Import shared utilities from _core used by cli.py handlers.
from megaplan._core import (
    PlanState, PlanConfig, PlanMeta, FlagRecord, FlagRegistry,
    HistoryEntry, StepResponse,
    STATE_INITIALIZED, STATE_CLARIFIED, STATE_PLANNED, STATE_CRITIQUED, STATE_EVALUATED,
    STATE_GATED, STATE_EXECUTED, STATE_DONE, STATE_ABORTED,
    TERMINAL_STATES, FLAG_BLOCKING_STATUSES,
    DEFAULT_AGENT_ROUTING, KNOWN_AGENTS,
    ROBUSTNESS_LEVELS,
    CliError,
    now_utc, slugify, json_dump,
    sha256_text, sha256_file,
    atomic_write_text, atomic_write_json, read_json,
    config_dir, load_config, save_config, detect_available_agents,
    ensure_runtime_layout, plans_root,
    current_iteration_raw_artifact,
    active_plan_dirs, load_plan, save_state,
    latest_plan_path, latest_plan_meta_path,
    load_flag_registry, save_flag_registry,
    unresolved_significant_flags, scope_creep_flags,
    configured_robustness,
)

# Re-export from sub-modules for backward compatibility (tests import via megaplan.cli).
from megaplan.evaluation import (  # noqa: F401
    build_evaluation,
    compute_plan_delta_percent,
    compute_recurring_critiques,
)
from megaplan.workers import (  # noqa: F401
    CommandResult,
    WorkerResult,
    run_command,
    run_step_with_worker,
    validate_payload,
    mock_worker_output,
    update_session_state,
    resolve_agent_mode,
)
from megaplan.prompts import (  # noqa: F401
    create_claude_prompt,
    create_codex_prompt,
)

__all__ = [
    # Types
    "PlanState", "PlanConfig", "PlanMeta", "FlagRecord", "StepResponse",
    # State constants
    "STATE_INITIALIZED", "STATE_CLARIFIED", "STATE_PLANNED", "STATE_CRITIQUED",
    "STATE_EVALUATED", "STATE_GATED", "STATE_EXECUTED", "STATE_DONE", "STATE_ABORTED",
    "TERMINAL_STATES",
    # Error and result types
    "CliError", "CommandResult", "WorkerResult",
    # Handlers
    "handle_init", "handle_clarify", "handle_plan", "handle_critique",
    "handle_evaluate", "handle_integrate", "handle_gate", "handle_execute",
    "handle_review", "handle_status", "handle_audit", "handle_list",
    "handle_override", "handle_setup", "handle_setup_global", "handle_config",
    # Key utilities
    "slugify", "build_evaluation", "mock_worker_output",
    "main", "cli_entry",
]


# ---------------------------------------------------------------------------
# Helpers that remain in cli.py (not needed by submodules)
# ---------------------------------------------------------------------------

def _append_to_meta(state: PlanState, field: str, value: Any) -> None:
    """Append *value* to ``state["meta"][field]``, creating intermediates as needed."""
    state["meta"].setdefault(field, []).append(value)


def render_response(response: StepResponse, *, exit_code: int = 0) -> int:
    print(json_dump(response), end="")
    return exit_code


def error_response(error: CliError) -> int:
    payload = {
        "success": False,
        "error": error.code,
        "message": error.message,
    }
    if error.valid_next:
        payload["valid_next"] = error.valid_next
    if error.extra:
        payload["details"] = error.extra
    return render_response(payload, exit_code=error.exit_code)


def apply_session_update(state: PlanState, step: str, agent: str, session_id: str | None, *, mode: str, refreshed: bool) -> None:
    """Call update_session_state and store the result on state."""
    result = update_session_state(step, agent, session_id, mode=mode, refreshed=refreshed, existing_sessions=state["sessions"])
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
        match = re.fullmatch(r"FLAG-(\d+)", str(flag.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def make_flag_id(number: int) -> str:
    return f"FLAG-{number:03d}"


def resolve_severity(hint: str) -> str:
    """Map a severity_hint to a resolved severity value.

    The fallback for 'uncertain' (and any unrecognised hint) is
    'significant' -- this is a deliberate conservative default so that
    ambiguous flags surface for human review rather than being silently
    downgraded.
    """
    if hint == "likely-significant":
        return "significant"
    if hint == "likely-minor":
        return "minor"
    # Deliberate: 'uncertain' resolves to 'significant' as a conservative default.
    return "significant"


def make_history_entry(
    step: str,
    *,
    duration_ms: int,
    cost_usd: float,
    result: str,
    worker: "WorkerResult | None" = None,
    agent: str | None = None,
    mode: str | None = None,
    output_file: str | None = None,
    artifact_hash: str | None = None,
    raw_output_file: str | None = None,
    message: str | None = None,
    flags_count: int | None = None,
    flags_addressed: list[str] | None = None,
    recommendation: str | None = None,
    approval_mode: str | None = None,
    environment: dict[str, bool] | None = None,
) -> HistoryEntry:
    """Build a history entry dict with common fields plus step-specific extras."""
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
    """Copy agent fallback info onto a response dict if present."""
    if hasattr(args, "_agent_fallback"):
        response["agent_fallback"] = args._agent_fallback


def infer_next_steps(state: PlanState) -> list[str]:
    current = state["current_state"]
    if current == STATE_INITIALIZED:
        return ["clarify"]
    if current == STATE_CLARIFIED:
        return ["clarify", "plan"]
    if current == STATE_PLANNED:
        return ["critique"]
    if current == STATE_CRITIQUED:
        return ["evaluate"]
    if current == STATE_EVALUATED:
        evaluation = state["last_evaluation"]
        recommendation = evaluation.get("recommendation")
        valid = []
        if recommendation == "CONTINUE":
            valid.append("integrate")
        if recommendation in {"SKIP", "CONTINUE"}:
            valid.append("gate")
        if recommendation in {"ESCALATE", "ABORT"}:
            valid.extend(["override replan", "override add-note", "override force-proceed", "override abort"])
        return valid or ["override add-note", "override abort"]
    if current == STATE_GATED:
        return ["execute", "override replan"]
    if current == STATE_EXECUTED:
        return ["review"]
    return []


def require_state(state: PlanState, step: str, allowed: set[str]) -> None:
    current = state["current_state"]
    if current not in allowed:
        raise CliError(
            "invalid_transition",
            f"Cannot run '{step}' while current state is '{current}'",
            valid_next=infer_next_steps(state),
            extra={"current_state": current},
        )


def find_command(name: str) -> str | None:
    return shutil.which(name)


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
    by_id = {flag["id"]: flag for flag in flags}
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
            existing["severity"] = resolve_severity(normalized["severity_hint"])
            existing["raised_in"] = f"critique_v{iteration}.json"
        else:
            severity = resolve_severity(normalized["severity_hint"])
            created = {
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


def update_flags_after_integrate(plan_dir: Path, flags_addressed: list[str], *, plan_file: str, summary: str) -> FlagRegistry:
    registry = load_flag_registry(plan_dir)
    for flag in registry["flags"]:
        if flag["id"] in flags_addressed:
            flag["status"] = "addressed"
            flag["addressed_in"] = plan_file
            flag["evidence"] = summary
    save_flag_registry(plan_dir, registry)
    return registry


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


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

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

    state = {
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
        "last_evaluation": {},
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
        "next_step": "clarify",
        "auto_approve": auto_approve,
        "robustness": robustness,
    }


def handle_clarify(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "clarify", {STATE_INITIALIZED, STATE_CLARIFIED})
    try:
        worker, agent, mode, refreshed = run_step_with_worker("clarify", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="clarify", iteration=state["iteration"], error=error)
        raise
    payload = worker.payload
    clarify_filename = "clarify.json"
    atomic_write_json(plan_dir / clarify_filename, payload)
    state["clarification"] = {
        "refined_idea": payload["refined_idea"],
        "intent_summary": payload["intent_summary"],
        "questions": payload["questions"],
    }
    state["current_state"] = STATE_CLARIFIED
    apply_session_update(state, "clarify", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "clarify",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=clarify_filename,
            artifact_hash=sha256_file(plan_dir / clarify_filename),
        ),
    )
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "clarify",
        "summary": f"Captured clarification with {len(payload['questions'])} questions.",
        "artifacts": [clarify_filename],
        "next_step": "plan",
        "state": STATE_CLARIFIED,
        "refined_idea": payload["refined_idea"],
        "intent_summary": payload["intent_summary"],
        "questions": payload["questions"],
    }
    attach_agent_fallback(response, args)
    return response


def handle_plan(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "plan", {STATE_INITIALIZED, STATE_CLARIFIED})
    try:
        worker, agent, mode, refreshed = run_step_with_worker("plan", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="plan", iteration=state["iteration"] + 1, error=error)
        raise
    payload = worker.payload
    version = state["iteration"] + 1
    plan_filename = f"plan_v{version}.md"
    meta_filename = f"plan_v{version}.meta.json"
    plan_text = payload["plan"].rstrip() + "\n"
    atomic_write_text(plan_dir / plan_filename, plan_text)
    meta = {
        "version": version,
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "questions": payload["questions"],
        "success_criteria": payload["success_criteria"],
        "assumptions": payload["assumptions"],
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    state["iteration"] = version
    state["current_state"] = STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["plan_versions"].append({"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]})
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
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "plan",
        "iteration": version,
        "summary": f"Generated plan v{version} with {len(payload['questions'])} questions and {len(payload['success_criteria'])} success criteria.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": "critique",
        "state": STATE_PLANNED,
        "questions": payload["questions"],
        "assumptions": payload["assumptions"],
        "success_criteria": payload["success_criteria"],
    }
    attach_agent_fallback(response, args)
    return response


def handle_critique(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "critique", {STATE_PLANNED})
    iteration = state["iteration"]
    try:
        worker, agent, mode, refreshed = run_step_with_worker("critique", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="critique", iteration=iteration, error=error)
        raise
    critique_filename = f"critique_v{iteration}.json"
    atomic_write_json(plan_dir / critique_filename, worker.payload)
    registry = update_flags_after_critique(plan_dir, worker.payload, iteration=iteration)
    significant = len([f for f in registry["flags"] if f.get("severity") == "significant" and f.get("status") in FLAG_BLOCKING_STATUSES])
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
        {"id": f.get("id"), "concern": f.get("concern", ""), "category": f.get("category", "other"), "severity": f.get("severity", "unknown")}
        for f in registry["flags"] if f.get("status") == "open"
    ]
    response = {
        "success": True,
        "step": "critique",
        "iteration": iteration,
        "summary": f"Recorded {len(worker.payload.get('flags', []))} critique flags.",
        "artifacts": [critique_filename, "faults.json"],
        "next_step": "evaluate",
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


def handle_evaluate(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "evaluate", {STATE_CRITIQUED})
    evaluation = build_evaluation(plan_dir, state)
    iteration = state["iteration"]
    _append_to_meta(state, "weighted_scores", evaluation["signals"]["weighted_score"])
    filename = f"evaluation_v{iteration}.json"
    atomic_write_json(plan_dir / filename, evaluation)
    state["current_state"] = STATE_EVALUATED
    state["last_evaluation"] = evaluation
    append_history(
        state,
        make_history_entry(
            "evaluate",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file=filename,
            artifact_hash=sha256_file(plan_dir / filename),
            recommendation=evaluation["recommendation"],
        ),
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "evaluate",
        "iteration": iteration,
        "summary": f"Recommendation {evaluation['recommendation']}: {evaluation['rationale']}",
        "artifacts": [filename],
        "next_step": evaluation["valid_next_steps"][0] if evaluation["valid_next_steps"] else None,
        "state": STATE_EVALUATED,
        "evaluation": evaluation,
    }


def handle_integrate(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "integrate", {STATE_EVALUATED})
    if state["last_evaluation"].get("recommendation") != "CONTINUE":
        raise CliError(
            "invalid_transition",
            "Integrate requires an evaluation recommendation of CONTINUE",
            valid_next=infer_next_steps(state),
        )
    previous_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    try:
        worker, agent, mode, refreshed = run_step_with_worker("integrate", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="integrate", iteration=state["iteration"] + 1, error=error)
        raise
    payload = worker.payload
    validate_payload("integrate", payload)
    version = state["iteration"] + 1
    plan_filename = f"plan_v{version}.md"
    meta_filename = f"plan_v{version}.meta.json"
    plan_text = payload["plan"].rstrip() + "\n"
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
        "delta_from_previous_percent": delta,
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    state["iteration"] = version
    state["current_state"] = STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["plan_versions"].append({"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]})
    _append_to_meta(state, "plan_deltas", delta)
    update_flags_after_integrate(plan_dir, payload["flags_addressed"], plan_file=plan_filename, summary=payload["changes_summary"])
    apply_session_update(state, "integrate", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "integrate",
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
    # Load updated flag registry to show remaining flags
    updated_registry = load_flag_registry(plan_dir)
    remaining = [
        {"id": f.get("id"), "concern": f.get("concern", ""), "category": f.get("category", "other")}
        for f in updated_registry["flags"]
        if f.get("status") in FLAG_BLOCKING_STATUSES and f.get("severity") == "significant"
    ]
    response = {
        "success": True,
        "step": "integrate",
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


def run_gate_checks(plan_dir: Path, state: PlanState) -> StepResponse:
    project_dir = Path(state["config"]["project_dir"])
    meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    checks = {
        "project_dir_exists": project_dir.exists(),
        "project_dir_writable": os.access(project_dir, os.W_OK),
        "success_criteria_present": bool(meta.get("success_criteria")),
        "claude_available": bool(find_command("claude")),
        "codex_available": bool(find_command("codex")),
    }
    passed = all(checks.values()) and not unresolved
    return {
        "passed": passed,
        "criteria_check": {"count": len(meta.get("success_criteria", [])), "items": meta.get("success_criteria", [])},
        "preflight_results": checks,
        "unresolved_flags": unresolved,
    }


def handle_gate(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "gate", {STATE_EVALUATED})
    recommendation = state["last_evaluation"].get("recommendation")
    if recommendation not in {"SKIP", "CONTINUE"}:
        raise CliError(
            "invalid_transition",
            f"Gate requires evaluation recommendation SKIP or CONTINUE, got {recommendation!r}",
            valid_next=infer_next_steps(state),
        )
    gate = run_gate_checks(plan_dir, state)
    atomic_write_json(plan_dir / "gate.json", gate)
    if not gate["passed"]:
        append_history(
            state,
            make_history_entry(
                "gate",
                duration_ms=0,
                cost_usd=0.0,
                result="blocked",
                output_file="gate.json",
                artifact_hash=sha256_file(plan_dir / "gate.json"),
            ),
        )
        save_state(plan_dir, state)
        return {
            "success": False,
            "step": "gate",
            "summary": "Gate blocked: unresolved flags or failed preflight checks remain.",
            "artifacts": ["gate.json"],
            "next_step": "integrate" if recommendation == "CONTINUE" else "override force-proceed",
            "state": state["current_state"],
            "auto_approve": bool(state["config"].get("auto_approve", False)),
            "robustness": configured_robustness(state),
            **gate,
        }
    final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    atomic_write_text(plan_dir / "final.md", final_plan)
    state["current_state"] = STATE_GATED
    state["meta"].pop("user_approved_gate", None)
    append_history(
        state,
        make_history_entry(
            "gate",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file="gate.json",
            artifact_hash=sha256_file(plan_dir / "gate.json"),
        ),
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "gate",
        "summary": "Gate passed. Plan is ready for execution.",
        "artifacts": ["gate.json", "final.md"],
        "next_step": "execute",
        "state": STATE_GATED,
        "auto_approve": bool(state["config"].get("auto_approve", False)),
        "robustness": configured_robustness(state),
        **gate,
    }


def handle_execute(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "execute", {STATE_GATED})
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
    try:
        worker, agent, mode, refreshed = run_step_with_worker("execute", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="execute", iteration=state["iteration"], error=error)
        raise
    atomic_write_json(plan_dir / "execution.json", worker.payload)
    if worker.trace_output is not None:
        atomic_write_text(plan_dir / "execution_trace.jsonl", worker.trace_output)
    state["current_state"] = STATE_EXECUTED
    apply_session_update(state, "execute", agent, worker.session_id, mode=mode, refreshed=refreshed)
    if auto_approve:
        approval_mode = "auto_approve"
    elif state["meta"].get("user_approved_gate", False):
        approval_mode = "user_approved"
    else:
        approval_mode = "manual"
    append_history(
        state,
        make_history_entry(
            "execute",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="execution.json",
            artifact_hash=sha256_file(plan_dir / "execution.json"),
            approval_mode=approval_mode,
        ),
    )
    save_state(plan_dir, state)
    artifacts = ["execution.json"]
    if worker.trace_output is not None:
        artifacts.append("execution_trace.jsonl")
    response = {
        "success": True,
        "step": "execute",
        "summary": worker.payload["output"],
        "artifacts": artifacts,
        "next_step": "review",
        "state": STATE_EXECUTED,
        "files_changed": worker.payload.get("files_changed", []),
        "deviations": worker.payload.get("deviations", []),
        "auto_approve": auto_approve,
        "user_approved_gate": bool(state["meta"].get("user_approved_gate", False)),
    }
    attach_agent_fallback(response, args)
    return response


def handle_review(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "review", {STATE_EXECUTED})
    try:
        worker, agent, mode, refreshed = run_step_with_worker("review", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="review", iteration=state["iteration"], error=error)
        raise
    atomic_write_json(plan_dir / "review.json", worker.payload)
    final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    atomic_write_text(plan_dir / "final.md", final_plan)
    passed = sum(1 for criterion in worker.payload.get("criteria", []) if criterion.get("pass"))
    total = len(worker.payload.get("criteria", []))
    state["current_state"] = STATE_DONE
    apply_session_update(state, "review", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "review",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="review.json",
            artifact_hash=sha256_file(plan_dir / "review.json"),
        ),
    )
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "review",
        "summary": f"Review complete: {passed}/{total} success criteria passed.",
        "artifacts": ["review.json", "final.md"],
        "next_step": None,
        "state": STATE_DONE,
        "issues": worker.payload.get("issues", []),
    }
    attach_agent_fallback(response, args)
    return response


def handle_status(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    next_steps = infer_next_steps(state)
    return {
        "success": True,
        "step": "status",
        "plan": state["name"],
        "state": state["current_state"],
        "iteration": state["iteration"],
        "summary": f"Plan '{state['name']}' is currently in state '{state['current_state']}'.",
        "next_step": next_steps[0] if next_steps else None,
        "valid_next": next_steps,
        "artifacts": sorted(path.name for path in plan_dir.iterdir() if path.is_file()),
    }


def handle_audit(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    return {
        "success": True,
        "step": "audit",
        "plan": state["name"],
        "plan_dir": str(plan_dir),
        "state": state,
    }


def handle_list(root: Path, args: argparse.Namespace) -> StepResponse:
    ensure_runtime_layout(root)
    items = []
    for plan_dir in active_plan_dirs(root):
        state = read_json(plan_dir / "state.json")
        next_steps = infer_next_steps(state)
        items.append(
            {
                "name": state["name"],
                "idea": state["idea"],
                "state": state["current_state"],
                "iteration": state["iteration"],
                "next_step": next_steps[0] if next_steps else None,
            }
        )
    return {
        "success": True,
        "step": "list",
        "summary": f"Found {len(items)} plans.",
        "plans": items,
    }


def _override_add_note(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    action = "add-note"
    note = args.note
    override_entry = {"action": action, "timestamp": now_utc(), "note": note}
    note_record = {"timestamp": now_utc(), "note": note}
    _append_to_meta(state, "notes", note_record)
    _append_to_meta(state, "overrides", override_entry)
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": "Attached note to the plan.",
        "next_step": infer_next_steps(state)[0] if infer_next_steps(state) else None,
        "state": state["current_state"],
    }


def _override_abort(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    override_entry = {"action": "abort", "timestamp": now_utc(), "reason": args.reason}
    state["current_state"] = STATE_ABORTED
    _append_to_meta(state, "overrides", override_entry)
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": "Plan aborted.",
        "next_step": None,
        "state": STATE_ABORTED,
    }


def _override_force_proceed(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    if state["current_state"] != STATE_EVALUATED:
        raise CliError(
            "invalid_transition",
            "force-proceed is only supported from evaluated state",
            valid_next=infer_next_steps(state),
        )
    gate = run_gate_checks(plan_dir, state)
    if not gate["preflight_results"]["project_dir_exists"] or not gate["preflight_results"]["success_criteria_present"]:
        raise CliError("unsafe_override", "force-proceed cannot bypass missing project directory or success criteria")
    atomic_write_json(plan_dir / "gate.json", gate)
    final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    atomic_write_text(plan_dir / "final.md", final_plan)
    state["current_state"] = STATE_GATED
    state["meta"].pop("user_approved_gate", None)
    _append_to_meta(state, "overrides", {"action": "force-proceed", "timestamp": now_utc(), "reason": args.reason})
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": "Force-proceeded past evaluation into gated state.",
        "next_step": "execute",
        "state": STATE_GATED,
    }


def _override_skip(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    if state["current_state"] != STATE_EVALUATED:
        raise CliError("invalid_transition", "skip is currently only supported from evaluated state", valid_next=infer_next_steps(state))
    evaluation = copy.deepcopy(state["last_evaluation"])
    evaluation["recommendation"] = "SKIP"
    state["last_evaluation"] = evaluation
    _append_to_meta(state, "overrides", {"action": "skip", "timestamp": now_utc(), "reason": args.reason})
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "override",
        "summary": "Marked evaluation as SKIP. Run gate next.",
        "next_step": "gate",
        "state": state["current_state"],
    }


def _override_replan(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    allowed = {STATE_GATED, STATE_EVALUATED, STATE_CRITIQUED}
    if state["current_state"] not in allowed:
        raise CliError(
            "invalid_transition",
            f"replan requires state {', '.join(sorted(allowed))}, got '{state['current_state']}'",
            valid_next=infer_next_steps(state),
        )
    reason = args.reason or args.note or "Re-entering planning loop"
    plan_file = latest_plan_path(plan_dir, state)
    state["current_state"] = STATE_PLANNED
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
        "message": f"Edit {plan_file.name} to incorporate your changes, then run critique. Or run critique directly if the note provides enough context for the loop to address.",
    }


_OVERRIDE_ACTIONS: dict[str, Callable[[Path, PlanState, argparse.Namespace], StepResponse]] = {
    "add-note": _override_add_note,
    "abort": _override_abort,
    "force-proceed": _override_force_proceed,
    "skip": _override_skip,
    "replan": _override_replan,
}


def handle_override(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    action = args.override_action
    handler = _OVERRIDE_ACTIONS.get(action)
    if handler is None:
        raise CliError("invalid_override", f"Unknown override action: {action}")
    return handler(plan_dir, state, args)


def _canonical_instructions() -> str:
    """Return the single canonical instructions file."""
    return resources.files("megaplan").joinpath("data", "instructions.md").read_text(encoding="utf-8")


_SKILL_HEADER = """\
---
name: megaplan
description: AI agent harness for coordinating Claude and GPT to make and execute extremely robust plans.
---

"""

_CURSOR_HEADER = """\
---
description: Use megaplan for high-rigor planning on complex, high-risk, or multi-stage tasks.
alwaysApply: false
---

"""


def bundled_agents_md() -> str:
    """Return instructions formatted as AGENTS.md (plain markdown)."""
    return _canonical_instructions()


def bundled_global_file(name: str) -> str:
    """Return instructions formatted for a specific agent target."""
    content = _canonical_instructions()
    if name == "skill.md":
        return _SKILL_HEADER + content
    if name == "cursor_rule.mdc":
        return _CURSOR_HEADER + content
    return content


GLOBAL_TARGETS = [
    {"agent": "claude", "detect": ".claude", "path": ".claude/skills/megaplan/SKILL.md", "data": "skill.md"},
    {"agent": "codex",  "detect": ".codex",  "path": ".codex/skills/megaplan/SKILL.md",  "data": "skill.md"},
    {"agent": "cursor", "detect": ".cursor", "path": ".cursor/rules/megaplan.mdc",       "data": "cursor_rule.mdc"},
]


def _install_owned_file(
    path: Path, content: str, *, force: bool = False
) -> dict[str, bool | str]:
    """Write a file we own, skipping if content already matches."""
    existed = path.exists()
    if existed and not force:
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return {"path": str(path), "skipped": True, "existed": True}
    atomic_write_text(path, content)
    return {"path": str(path), "skipped": False, "existed": existed}


def handle_setup_global(force: bool = False, home: Path | None = None) -> StepResponse:
    """Install megaplan config into all detected agent global dirs."""
    if home is None:
        home = Path.home()

    installed: list[dict[str, Any]] = []
    detected_count = 0

    for target in GLOBAL_TARGETS:
        agent_dir = home / target["detect"]
        if not agent_dir.is_dir():
            installed.append({
                "agent": target["agent"],
                "path": str(home / target["path"]),
                "skipped": True,
                "reason": "not installed",
            })
            continue

        detected_count += 1
        content = bundled_global_file(target["data"])
        dest = home / target["path"]
        result = _install_owned_file(dest, content, force=force)
        result["agent"] = target["agent"]
        installed.append(result)

    if detected_count == 0:
        return {
            "success": False,
            "step": "setup",
            "mode": "global",
            "summary": (
                "No supported agents detected. "
                "Create one of ~/.claude/, ~/.codex/, or ~/.cursor/ and re-run, "
                "or use 'megaplan setup' to install AGENTS.md into a specific project."
            ),
            "installed": installed,
        }

    # Write agent routing config based on what's available
    available = detect_available_agents()
    config_path = None
    routing = None
    if available:
        agents_config: dict[str, str] = {}
        for step, default in DEFAULT_AGENT_ROUTING.items():
            agents_config[step] = default if default in available else available[0]
        config = load_config(home)
        config["agents"] = agents_config
        config_path = save_config(config, home)
        routing = agents_config

    lines = []
    for install_record in installed:
        if install_record.get("reason") == "not installed":
            lines.append(f"  {install_record['agent']}: skipped (not installed)")
        elif install_record["skipped"]:
            lines.append(f"  {install_record['agent']}: up to date")
        else:
            verb = "overwrote" if install_record["existed"] else "created"
            lines.append(f"  {install_record['agent']}: {verb} {install_record['path']}")

    result_data: dict[str, Any] = {
        "success": True,
        "step": "setup",
        "mode": "global",
        "summary": "Global setup complete:\n" + "\n".join(lines),
        "installed": installed,
    }
    if config_path is not None:
        result_data["config_path"] = str(config_path)
        result_data["routing"] = routing
    return result_data


def handle_setup(args: argparse.Namespace) -> StepResponse:
    local = args.local or args.target_dir
    if not local:
        return handle_setup_global(force=args.force)
    target_dir = Path(args.target_dir).resolve() if args.target_dir else Path.cwd()
    target = target_dir / "AGENTS.md"
    content = bundled_agents_md()
    if target.exists() and not args.force:
        existing = target.read_text(encoding="utf-8")
        if "megaplan" in existing.lower():
            return {
                "success": True,
                "step": "setup",
                "summary": f"AGENTS.md already contains megaplan instructions at {target}",
                "skipped": True,
            }
        # Append to existing AGENTS.md (atomic: read-concat-write)
        combined = existing + "\n\n" + content
        atomic_write_text(target, combined)
        return {
            "success": True,
            "step": "setup",
            "summary": f"Appended megaplan instructions to existing {target}",
            "file": str(target),
        }
    atomic_write_text(target, content)
    return {
        "success": True,
        "step": "setup",
        "summary": f"Created {target}",
        "file": str(target),
    }


def handle_config(args: argparse.Namespace) -> StepResponse:
    """Handle the config subcommand: show, set, reset."""
    action = args.config_action

    if action == "show":
        config = load_config()
        effective: dict[str, str] = {}
        file_agents = config.get("agents", {})
        for step, default in DEFAULT_AGENT_ROUTING.items():
            effective[step] = file_agents.get(step, default)
        return {
            "success": True,
            "step": "config",
            "action": "show",
            "config_path": str(config_dir() / "config.json"),
            "routing": effective,
            "raw_config": config,
        }

    if action == "set":
        key = args.key
        value = args.value
        parts = key.split(".", 1)
        if len(parts) != 2 or parts[0] != "agents":
            raise CliError("invalid_args", f"Key must be 'agents.<step>', got '{key}'")
        step = parts[1]
        if step not in DEFAULT_AGENT_ROUTING:
            raise CliError("invalid_args", f"Unknown step '{step}'. Valid steps: {', '.join(DEFAULT_AGENT_ROUTING)}")
        if value not in KNOWN_AGENTS:
            raise CliError("invalid_args", f"Unknown agent '{value}'. Valid agents: {', '.join(KNOWN_AGENTS)}")
        config = load_config()
        config.setdefault("agents", {})[step] = value
        save_config(config)
        return {
            "success": True,
            "step": "config",
            "action": "set",
            "key": key,
            "value": value,
        }

    if action == "reset":
        path = config_dir() / "config.json"
        if path.exists():
            path.unlink()
        return {
            "success": True,
            "step": "config",
            "action": "reset",
            "summary": "Config file removed. Using defaults.",
        }

    raise CliError("invalid_args", f"Unknown config action: {action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Megaplan orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Install megaplan into agent configs (global by default)")
    setup_parser.add_argument("--local", action="store_true", help="Install AGENTS.md into a project instead of global agent configs")
    setup_parser.add_argument("--target-dir", help="Directory to install into (default: cwd, implies --local)")
    setup_parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--project-dir", required=True)
    init_parser.add_argument("--name")
    init_parser.add_argument("--auto-approve", action="store_true")
    init_parser.add_argument("--robustness", choices=list(ROBUSTNESS_LEVELS), default="standard")
    init_parser.add_argument("idea")

    subparsers.add_parser("list")

    for name in ["status", "audit", "evaluate", "gate"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")

    for name in ["clarify", "plan", "critique", "integrate", "execute", "review"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")
        step_parser.add_argument("--agent", choices=["claude", "codex"])
        step_parser.add_argument("--fresh", action="store_true")
        step_parser.add_argument("--persist", action="store_true")
        step_parser.add_argument("--ephemeral", action="store_true")
        if name == "execute":
            step_parser.add_argument("--confirm-destructive", action="store_true")
            step_parser.add_argument("--user-approved", action="store_true")
        if name == "review":
            step_parser.add_argument("--confirm-self-review", action="store_true")

    config_parser = subparsers.add_parser("config", help="View or edit megaplan configuration")
    config_sub = config_parser.add_subparsers(dest="config_action", required=True)
    config_sub.add_parser("show")
    set_parser = config_sub.add_parser("set")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    config_sub.add_parser("reset")

    override_parser = subparsers.add_parser("override")
    override_parser.add_argument("override_action", choices=["skip", "abort", "force-proceed", "add-note", "replan"])
    override_parser.add_argument("--plan")
    override_parser.add_argument("--reason", default="")
    override_parser.add_argument("--note")

    return parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

# Commands that take (root, args) and return a response dict.
COMMAND_HANDLERS: dict[str, Callable[..., StepResponse]] = {
    "init": handle_init,
    "clarify": handle_clarify,
    "plan": handle_plan,
    "critique": handle_critique,
    "evaluate": handle_evaluate,
    "integrate": handle_integrate,
    "gate": handle_gate,
    "execute": handle_execute,
    "review": handle_review,
    "status": handle_status,
    "audit": handle_audit,
    "list": handle_list,
    "override": handle_override,
}


def cli_entry() -> None:
    """Entry point for the `megaplan` console script."""
    sys.exit(main())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    try:
        if args.command == "setup":
            response = handle_setup(args)
            return render_response(response)
        if args.command == "config":
            response = handle_config(args)
            return render_response(response)
    except CliError as error:
        return error_response(error)
    root = Path.cwd()
    ensure_runtime_layout(root)
    try:
        handler = COMMAND_HANDLERS.get(args.command)
        if handler is None:
            raise CliError("invalid_command", f"Unknown command {args.command!r}")
        if args.command == "override" and remaining:
            if not args.note:
                args.note = " ".join(remaining)
            remaining = []
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        if args.command == "override" and args.override_action == "add-note" and not args.note:
            raise CliError("invalid_args", "override add-note requires a note")
        response = handler(root, args)
        return render_response(response)
    except CliError as error:
        return error_response(error)


if __name__ == "__main__":
    sys.exit(main())
