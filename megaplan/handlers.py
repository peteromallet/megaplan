from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import megaplan.workers as worker_module
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
    atomic_write_json,
    atomic_write_text,
    configured_robustness,
    current_iteration_raw_artifact,
    ensure_runtime_layout,
    latest_plan_path,
    load_flag_registry,
    load_plan,
    now_utc,
    plans_root,
    save_flag_registry,
    save_state,
    scope_creep_flags,
    sha256_file,
    sha256_text,
    slugify,
)
from megaplan.evaluation import (
    build_gate_artifact,
    build_gate_signals,
    build_orchestrator_guidance,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    run_gate_checks,
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
    save_state(plan_dir, state)
    response: StepResponse = {
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
    gate_signals = build_gate_signals(plan_dir, state)
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
    state["last_gate"] = {
        "recommendation": gate_summary["recommendation"],
        "rationale": gate_summary["rationale"],
        "signals_assessment": gate_summary["signals_assessment"],
        "warnings": gate_summary["warnings"],
        "passed": gate_summary["passed"],
        "preflight_results": gate_summary["preflight_results"],
        "orchestrator_guidance": gate_summary["orchestrator_guidance"],
    }
    if len(state["meta"].get("weighted_scores", [])) < iteration:
        _append_to_meta(state, "weighted_scores", gate_signals["signals"]["weighted_score"])

    result = "success"
    artifacts = [signals_filename, "gate.json"]
    summary = f"Gate recommendation {gate_summary['recommendation']}: {gate_summary['rationale']}"
    if gate_summary["recommendation"] == "PROCEED" and gate_summary["passed"]:
        state["current_state"] = STATE_GATED
        state["meta"].pop("user_approved_gate", None)
        next_step = "finalize"
    elif gate_summary["recommendation"] == "PROCEED":
        result = "blocked"
        next_step = "revise"
        summary = "Gate recommended PROCEED, but preflight checks are still blocking execution."
    elif gate_summary["recommendation"] == "ITERATE":
        next_step = "revise"
    elif gate_summary["recommendation"] == "ESCALATE":
        if gate_signals["robustness"] == "light" and gate_signals["signals"]["weighted_score"] <= 4.0:
            next_step = "override force-proceed"
        else:
            next_step = "override add-note"
    else:
        next_step = "override add-note"
        result = "unknown_recommendation"
        summary = f"Gate returned unknown recommendation '{gate_summary['recommendation']}'; treating as escalation."

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
    atomic_write_text(plan_dir / "final.md", payload["final_plan"])
    atomic_write_json(plan_dir / "finalize.json", {
        "task_count": payload["task_count"],
        "watch_items": payload["watch_items"],
        "meta_commentary": payload["meta_commentary"],
    })
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
        "summary": f"Finalized plan with {payload['task_count']} tasks and {len(payload['watch_items'])} watch items.",
        "artifacts": ["final.md", "finalize.json"],
        "next_step": "execute",
        "state": STATE_FINALIZED,
    }
    attach_agent_fallback(response, args)
    return response


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
    try:
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("execute", state, plan_dir, args, root=root)
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
    response: StepResponse = {
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
        worker, agent, mode, refreshed = worker_module.run_step_with_worker("review", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="review", iteration=state["iteration"], error=error)
        raise
    atomic_write_json(plan_dir / "review.json", worker.payload)
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
    response: StepResponse = {
        "success": True,
        "step": "review",
        "summary": f"Review complete: {passed}/{total} success criteria passed.",
        "artifacts": ["review.json"],
        "next_step": None,
        "state": STATE_DONE,
        "issues": worker.payload.get("issues", []),
    }
    attach_agent_fallback(response, args)
    return response


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def _override_add_note(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
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


def _override_abort(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
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


def _override_force_proceed(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    if state["current_state"] != STATE_CRITIQUED:
        raise CliError(
            "invalid_transition",
            "force-proceed is only supported from critiqued state",
            valid_next=infer_next_steps(state),
        )
    gate_checks = run_gate_checks(plan_dir, state, command_lookup=find_command)
    if not gate_checks["preflight_results"]["project_dir_exists"] or not gate_checks["preflight_results"]["success_criteria_present"]:
        raise CliError("unsafe_override", "force-proceed cannot bypass missing project directory or success criteria")
    signals = build_gate_signals(plan_dir, state)
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
    }


def _override_replan(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
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


_OVERRIDE_ACTIONS: dict[str, Callable[[Path, PlanState, argparse.Namespace], StepResponse]] = {
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
    return handler(plan_dir, state, args)
