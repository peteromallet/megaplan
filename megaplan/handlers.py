from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import megaplan.workers as worker_module
from megaplan.checks import checks_for_robustness, validate_critique_checks
from megaplan.execution import (
    _check_done_task_evidence,
    handle_execute_auto_loop as dispatch_execute_auto_loop,
    handle_execute_one_batch as dispatch_execute_one_batch,
)
from megaplan.flags import update_flags_after_critique, update_flags_after_gate, update_flags_after_revise
from megaplan.merge import _validate_and_merge_batch
from megaplan.merge import _validate_merge_inputs
from megaplan.parallel_critique import run_parallel_critique
from megaplan.step_edit import next_plan_artifact_name
from megaplan.types import (
    FLAG_BLOCKING_STATUSES,
    ROBUSTNESS_LEVELS,
    CliError,
    PlanState,
    STATE_ABORTED,
    STATE_CRITIQUED,
    STATE_DONE,
    STATE_EXECUTED,
    STATE_FINALIZED,
    STATE_GATED,
    STATE_INITIALIZED,
    STATE_PREPPED,
    STATE_PLANNED,
    STATE_RESEARCHED,
    StepResponse,
)
from megaplan._core import (
    add_or_increment_debt,
    append_history,
    apply_session_update,
    atomic_write_json,
    atomic_write_text,
    configured_robustness,
    ensure_runtime_layout,
    extract_subsystem_tag,
    latest_plan_path,
    load_debt_registry,
    load_flag_registry,
    load_plan,
    make_history_entry,
    now_utc,
    plans_root,
    read_json,
    record_step_failure,
    render_final_md,
    save_debt_registry,
    save_state,
    scope_creep_flags,
    sha256_file,
    sha256_text,
    slugify,
    unresolved_significant_flags,
    workflow_includes_step,
    workflow_transition,
    workflow_next,
)
from megaplan.evaluation import (
    PLAN_STRUCTURE_REQUIRED_STEP_ISSUE,
    build_gate_artifact,
    build_gate_signals,
    build_orchestrator_guidance,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    is_rubber_stamp,
    run_gate_checks,
    validate_plan_structure,
)
from megaplan.workers import resolve_agent_mode
from megaplan._core import find_command, infer_next_steps, require_state
from megaplan.workers import WorkerResult, validate_payload

log = logging.getLogger("megaplan")


def _append_to_meta(state: PlanState, field: str, value: Any) -> None:
    state["meta"].setdefault(field, []).append(value)


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


_AUTO_NEXT_STEP = object()


def _run_worker(
    step: str,
    state: PlanState,
    plan_dir: Path,
    args: argparse.Namespace,
    *,
    root: Path,
    iteration: int | None = None,
) -> tuple[WorkerResult, str, str, bool]:
    failure_iteration = state["iteration"] if iteration is None else iteration
    try:
        return worker_module.run_step_with_worker(step, state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step=step, iteration=failure_iteration, error=error)
        raise


def _finish_step(
    plan_dir: Path,
    state: PlanState,
    args: argparse.Namespace,
    *,
    step: str,
    worker: WorkerResult,
    agent: str,
    mode: str,
    refreshed: bool,
    summary: str,
    artifacts: list[str],
    output_file: str,
    artifact_hash: str,
    result: str = "success",
    success: bool = True,
    next_step: object | str | None = _AUTO_NEXT_STEP,
    response_fields: dict[str, Any] | None = None,
    history_fields: dict[str, Any] | None = None,
) -> StepResponse:
    apply_session_update(state, step, agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            step,
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result=result,
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=output_file,
            artifact_hash=artifact_hash,
            prompt_tokens=worker.prompt_tokens,
            completion_tokens=worker.completion_tokens,
            total_tokens=worker.total_tokens,
            **(history_fields or {}),
        ),
    )
    save_state(plan_dir, state)
    resolved_next = next_step
    if resolved_next is _AUTO_NEXT_STEP:
        next_steps = workflow_next(state)
        resolved_next = next_steps[0] if next_steps else None
    response: StepResponse = {
        "success": success,
        "step": step,
        "summary": summary,
        "artifacts": artifacts,
        "next_step": resolved_next,
        "state": state["current_state"],
    }
    if response_fields:
        response.update(response_fields)
    attach_agent_fallback(response, args)
    return response


def _raise_step_validation_error(
    *,
    plan_dir: Path,
    state: PlanState,
    step: str,
    iteration: int,
    worker: WorkerResult,
    code: str,
    message: str,
) -> None:
    error = CliError(code, message, valid_next=infer_next_steps(state), extra={"raw_output": worker.raw_output})
    record_step_failure(plan_dir, state, step=step, iteration=iteration, error=error, duration_ms=worker.duration_ms)
    raise error


def _write_json_artifact(plan_dir: Path, filename: str, payload: dict[str, Any]) -> str:
    atomic_write_json(plan_dir / filename, payload)
    return sha256_file(plan_dir / filename)


def _write_plan_version(
    *,
    plan_dir: Path,
    state: PlanState,
    step: str,
    version: int,
    worker: WorkerResult,
    plan_text: str,
    meta_fields: dict[str, Any],
    plan_filename: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    resolved_plan_filename = plan_filename or next_plan_artifact_name(plan_dir, version)
    meta_filename = (
        f"plan_v{version}.meta.json"
        if resolved_plan_filename == f"plan_v{version}.md"
        else resolved_plan_filename.replace(".md", ".meta.json")
    )
    structure_warnings = _validate_generated_plan_or_raise(
        plan_dir=plan_dir,
        state=state,
        step=step,
        iteration=version,
        worker=worker,
        plan_text=plan_text,
    )
    atomic_write_text(plan_dir / resolved_plan_filename, plan_text)
    meta = {
        "version": version,
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        **meta_fields,
        "structure_warnings": structure_warnings,
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    return resolved_plan_filename, meta_filename, meta


def _write_finalize_artifacts(plan_dir: Path, payload: dict[str, Any], state: PlanState) -> str:
    _ensure_verification_task(payload, state)
    _reconcile_validation_after_mutation(payload)
    atomic_write_json(plan_dir / "finalize.json", payload)
    atomic_write_json(plan_dir / "finalize_snapshot.json", payload)
    atomic_write_text(plan_dir / "final.md", render_final_md(payload))
    return sha256_file(plan_dir / "finalize.json")


def _reconcile_validation_after_mutation(payload: dict[str, Any]) -> None:
    """Ensure validation block is consistent with the (possibly mutated) task list.

    After _ensure_verification_task() may have appended a task, update the
    validation block so orphan_tasks includes any handler-injected tasks.
    """
    validation = payload.get("validation")
    if not validation or not isinstance(validation, dict):
        return
    task_ids = {t["id"] for t in payload.get("tasks", []) if isinstance(t, dict)}
    covered_ids: set[str] = set()
    for entry in validation.get("plan_steps_covered", []):
        if isinstance(entry, dict):
            for tid in entry.get("finalize_task_ids", []):
                covered_ids.add(tid)
    orphan_ids = set(validation.get("orphan_tasks", []))
    for tid in task_ids:
        if tid not in covered_ids and tid not in orphan_ids:
            orphan_ids.add(tid)
    validation["orphan_tasks"] = sorted(orphan_ids)


def _validate_finalize_payload(plan_dir: Path, state: PlanState, worker: WorkerResult) -> None:
    payload = worker.payload

    def _reject(message: str) -> None:
        _raise_step_validation_error(
            plan_dir=plan_dir, state=state, step="finalize",
            iteration=state["iteration"], worker=worker,
            code="invalid_finalize", message=message,
        )

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        _reject("Finalize output must include a non-empty `tasks` list.")
    if not isinstance(payload.get("sense_checks"), list):
        _reject("Finalize output must include a `sense_checks` list.")
    if not isinstance(payload.get("watch_items"), list):
        _reject("Finalize output must include a `watch_items` list.")
    for index, task in enumerate(tasks, start=1):
        tid = task.get("id", index) if isinstance(task, dict) else index
        if not isinstance(task, dict):
            _reject(f"Finalize task {index} must be an object.")
        if not isinstance(task.get("id"), str) or not task["id"].strip():
            _reject(f"Finalize task {index} is missing a non-empty `id`.")
        if not isinstance(task.get("description"), str) or not task["description"].strip():
            _reject(f"Finalize task {tid} is missing a non-empty `description`.")
        if task.get("status") != "pending":
            _reject(f"Finalize task {tid} must start with status `pending`.")


def _build_gate_signals_artifact(
    plan_dir: Path,
    state: PlanState,
    *,
    iteration: int,
    root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
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
    return gate_signals, signals_filename, signals_artifact


def _record_gate_debt_entries(
    root: Path,
    state: PlanState,
    gate_summary: dict[str, Any],
    worker_payload: dict[str, Any],
) -> int:
    if gate_summary["recommendation"] != "PROCEED":
        return 0

    raw_tradeoffs = worker_payload.get("accepted_tradeoffs", [])
    accepted_tradeoffs = [
        item
        for item in raw_tradeoffs
        if isinstance(item, dict)
        and isinstance(item.get("flag_id"), str)
        and isinstance(item.get("concern"), str)
    ] if isinstance(raw_tradeoffs, list) else []
    debt_registry = load_debt_registry(root)
    debt_entries_added = 0
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
    else:
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
    return debt_entries_added


def _resolve_revise_transition(state: PlanState) -> tuple[bool, Any]:
    has_gate = workflow_includes_step(configured_robustness(state), "gate")
    if has_gate and state["last_gate"].get("recommendation") != "ITERATE":
        raise CliError("invalid_transition", "Revise requires a gate recommendation of ITERATE", valid_next=infer_next_steps(state))
    revise_transition = workflow_transition(state, "revise")
    if revise_transition is None:
        raise CliError("invalid_transition", "Revise is not available from the current workflow state", valid_next=infer_next_steps(state))
    return has_gate, revise_transition


def _next_progress_step(state: PlanState) -> str | None:
    next_steps = workflow_next(state)
    return next((step for step in next_steps if step not in {"plan", "step"}), next_steps[0] if next_steps else None)


def _remaining_significant_flags(plan_dir: Path) -> list[dict[str, str]]:
    return [
        {"id": flag["id"], "concern": flag["concern"], "category": flag["category"]}
        for flag in load_flag_registry(plan_dir)["flags"]
        if flag["status"] in FLAG_BLOCKING_STATUSES and flag.get("severity") == "significant"
    ]


def _gate_response_fields(state: PlanState, gate_summary: dict[str, Any], debt_entries_added: int) -> dict[str, Any]:
    return {
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


def _apply_gate_outcome(state: PlanState, gate_summary: dict[str, Any], *, robustness: str, plan_dir: Path) -> tuple[str, str, str]:
    result = "success"
    summary = f"Gate recommendation {gate_summary['recommendation']}: {gate_summary['rationale']}"

    # Process flag resolutions when the gate recommends PROCEED.
    # The gate LLM sees all blocking flags and makes an informed decision.
    # We respect its PROCEED recommendation: explicit resolutions are persisted
    # as-is, and any remaining blocking flags are implicitly accepted as
    # tradeoffs (the gate reviewed them and still chose PROCEED).
    if gate_summary["recommendation"] == "PROCEED":
        unresolved = gate_summary.get("unresolved_flags", [])
        resolutions = gate_summary.get("flag_resolutions", [])

        # Cap: if more than 3 explicit resolutions, truncate to 3 and warn
        # (but don't override — the gate's PROCEED decision still stands).
        if len(resolutions) > 3:
            log.warning(
                "Gate provided %d flag resolutions (max 3); truncating to first 3.",
                len(resolutions),
            )
            resolutions = resolutions[:3]

        # Validate each explicit resolution
        valid_resolved_ids: set[str] = set()
        for res in resolutions:
            action = res.get("action", "")
            flag_id = res.get("flag_id", "")
            if action == "dispute":
                evidence = res.get("evidence", "").strip()
                if not evidence or is_rubber_stamp(evidence, strict=True):
                    continue  # invalid dispute — skip
            elif action == "accept_tradeoff":
                pass  # always allowed
            else:
                continue  # unknown action — skip
            valid_resolved_ids.add(flag_id)

        blocking_unresolved = [
            f for f in unresolved
            if f.get("severity") in ("significant", "likely-significant")
            and f.get("status") in FLAG_BLOCKING_STATUSES
            and f.get("id") not in valid_resolved_ids
        ]

        # Persist explicit resolutions
        if valid_resolved_ids:
            update_flags_after_gate(plan_dir, resolutions)

        # Implicitly accept remaining blocking flags as tradeoffs — the gate
        # saw them, assessed them, and still recommended PROCEED.
        if blocking_unresolved:
            implicit_resolutions = [
                {"flag_id": f["id"], "action": "accept_tradeoff"}
                for f in blocking_unresolved
            ]
            update_flags_after_gate(plan_dir, implicit_resolutions)
            flag_ids = [f.get("id", "?") for f in blocking_unresolved]
            log.info(
                "Gate recommended PROCEED with %d unresolved blocking flag(s); "
                "implicitly accepting as tradeoffs: %s",
                len(blocking_unresolved),
                ", ".join(flag_ids),
            )

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
        return result, "override add-note", summary
    result = "unknown_recommendation"
    summary = f"Gate returned unknown recommendation '{gate_summary['recommendation']}'; treating as escalation."
    return result, "override add-note", summary


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
            "agent": "hermes" if getattr(args, "hermes", None) is not None else "",
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
    next_steps = workflow_next(state)
    return {
        "success": True,
        "step": "init",
        "plan": plan_name,
        "state": STATE_INITIALIZED,
        "summary": f"Initialized plan '{plan_name}' for project {project_dir}",
        "artifacts": ["state.json"],
        "next_step": next_steps[0] if next_steps else None,
        "auto_approve": auto_approve,
        "robustness": robustness,
    }


def handle_plan(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "plan", {STATE_INITIALIZED, STATE_PREPPED, STATE_PLANNED, STATE_RESEARCHED})
    rerun = state["current_state"] in {STATE_PLANNED, STATE_RESEARCHED}
    version = state["iteration"] if rerun else state["iteration"] + 1
    worker, agent, mode, refreshed = _run_worker("plan", state, plan_dir, args, root=root, iteration=version)
    payload = worker.payload
    plan_filename, meta_filename, meta = _write_plan_version(
        plan_dir=plan_dir,
        state=state,
        step="plan",
        version=version,
        worker=worker,
        plan_text=payload["plan"].rstrip() + "\n",
        meta_fields={
            "questions": payload["questions"],
            "success_criteria": payload["success_criteria"],
            "assumptions": payload["assumptions"],
        },
    )
    state["iteration"], state["current_state"] = version, STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["last_gate"] = {}
    state["plan_versions"].append({
        "version": version, "file": plan_filename,
        "hash": meta["hash"], "timestamp": meta["timestamp"],
    })
    verb = "Refined" if rerun else "Generated"
    return _finish_step(
        plan_dir, state, args,
        step="plan",
        worker=worker, agent=agent, mode=mode, refreshed=refreshed,
        summary=f"{verb} plan v{version} with {len(payload['questions'])} questions and {len(payload['success_criteria'])} success criteria.",
        artifacts=[plan_filename, meta_filename],
        output_file=plan_filename,
        artifact_hash=meta["hash"],
        response_fields={
            "iteration": version,
            "questions": payload["questions"],
            "assumptions": payload["assumptions"],
            "success_criteria": payload["success_criteria"],
        },
    )


def handle_research(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "research", {STATE_PLANNED})
    worker, agent, mode, refreshed = _run_worker("research", state, plan_dir, args, root=root)
    research_filename = "research.json"
    artifact_hash = _write_json_artifact(plan_dir, research_filename, worker.payload)
    considerations = worker.payload.get("considerations", [])
    state["current_state"] = STATE_RESEARCHED
    return _finish_step(
        plan_dir, state, args,
        step="research",
        worker=worker, agent=agent, mode=mode, refreshed=refreshed,
        summary=f"Research complete: logged {len(considerations)} documentation consideration(s).",
        artifacts=[research_filename],
        output_file=research_filename,
        artifact_hash=artifact_hash,
        response_fields={"iteration": state["iteration"]},
    )


def handle_prep(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "prep", {STATE_INITIALIZED})
    worker, agent, mode, refreshed = _run_worker("prep", state, plan_dir, args, root=root)
    prep_filename = "prep.json"
    artifact_hash = _write_json_artifact(plan_dir, prep_filename, worker.payload)
    code_refs = len(worker.payload.get("relevant_code", []))
    test_refs = len(worker.payload.get("test_expectations", []))
    state["current_state"] = STATE_PREPPED
    return _finish_step(
        plan_dir, state, args,
        step="prep",
        worker=worker, agent=agent, mode=mode, refreshed=refreshed,
        summary=f"Prep complete: captured {code_refs} relevant code reference(s) and {test_refs} test expectation(s).",
        artifacts=[prep_filename],
        output_file=prep_filename,
        artifact_hash=artifact_hash,
        response_fields={"iteration": state["iteration"]},
    )


def handle_critique(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "critique", {STATE_PLANNED, STATE_RESEARCHED})
    iteration = state["iteration"]
    robustness = configured_robustness(state)
    active_checks = checks_for_robustness(robustness)
    expected_ids = [check["id"] for check in active_checks]
    state["last_gate"] = {}
    agent_type, mode, refreshed, model = resolve_agent_mode("critique", args)
    if len(active_checks) > 1 and agent_type == "hermes":
        try:
            worker = run_parallel_critique(state, plan_dir, root=root, model=model, checks=active_checks)
            agent, mode, refreshed = "hermes", "persistent", True
        except Exception as exc:
            print(f"[parallel-critique] Failed, falling back to sequential: {exc}", file=sys.stderr)
            worker, agent, mode, refreshed = _run_worker("critique", state, plan_dir, args, root=root)
    else:
        worker, agent, mode, refreshed = _run_worker("critique", state, plan_dir, args, root=root)
    invalid_checks = validate_critique_checks(worker.payload, expected_ids=expected_ids)
    if invalid_checks:
        _raise_step_validation_error(plan_dir=plan_dir, state=state, step="critique", iteration=iteration, worker=worker, code="invalid_critique", message="Critique output failed check validation: " + ", ".join(invalid_checks))
    critique_filename = f"critique_v{iteration}.json"
    atomic_write_json(plan_dir / critique_filename, worker.payload)
    registry = update_flags_after_critique(plan_dir, worker.payload, iteration=iteration)
    significant = len([flag for flag in registry["flags"] if flag.get("severity") == "significant" and flag["status"] in FLAG_BLOCKING_STATUSES])
    _append_to_meta(state, "significant_counts", significant)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    _append_to_meta(state, "recurring_critiques", recurring)
    state["current_state"] = STATE_CRITIQUED
    skip_gate = not workflow_includes_step(robustness, "gate")
    if skip_gate:
        minimal_gate: dict[str, Any] = {
            "recommendation": "ITERATE",
            "rationale": "Light robustness: single revision pass to incorporate critique feedback.",
            "signals_assessment": "",
            "warnings": [],
            "settled_decisions": [],
        }
        atomic_write_json(plan_dir / "gate.json", minimal_gate)
        state["last_gate"] = {"recommendation": "ITERATE"}
    scope_flags_list = scope_creep_flags(registry, statuses=FLAG_BLOCKING_STATUSES)
    open_flags_detail = [
        {"id": flag["id"], "concern": flag["concern"], "category": flag["category"], "severity": flag.get("severity", "unknown")}
        for flag in registry["flags"]
        if flag["status"] == "open"
    ]
    response_fields: dict[str, Any] = {
        "iteration": iteration,
        "checks": worker.payload.get("checks", []),
        "verified_flags": worker.payload.get("verified_flag_ids", []),
        "open_flags": open_flags_detail,
        "scope_creep_flags": [flag["id"] for flag in scope_flags_list],
    }
    if scope_flags_list:
        response_fields["warnings"] = ["Scope creep detected in the plan. Surface this drift to the user while continuing the loop."]
    return _finish_step(
        plan_dir, state, args,
        step="critique",
        worker=worker, agent=agent, mode=mode, refreshed=refreshed,
        summary=f"Recorded {len(worker.payload.get('flags', []))} critique flags.",
        artifacts=[critique_filename, "faults.json"],
        output_file=critique_filename,
        artifact_hash=sha256_file(plan_dir / critique_filename),
        response_fields=response_fields,
        history_fields={"flags_count": len(worker.payload.get("flags", []))},
    )


def handle_revise(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "revise", {STATE_CRITIQUED})
    has_gate, revise_transition = _resolve_revise_transition(state)
    previous_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    worker, agent, mode, refreshed = _run_worker("revise", state, plan_dir, args, root=root, iteration=state["iteration"] + 1)
    payload = worker.payload
    validate_payload("revise", payload)
    version = state["iteration"] + 1
    plan_text = payload["plan"].rstrip() + "\n"
    delta = compute_plan_delta_percent(previous_plan, plan_text)
    plan_filename, meta_filename, meta = _write_plan_version(
        plan_dir=plan_dir, state=state, step="revise", version=version,
        worker=worker, plan_filename=f"plan_v{version}.md", plan_text=plan_text,
        meta_fields={
            "changes_summary": payload["changes_summary"],
            "flags_addressed": payload["flags_addressed"],
            "questions": payload.get("questions", []),
            "success_criteria": payload.get("success_criteria", []),
            "assumptions": payload.get("assumptions", []),
            "delta_from_previous_percent": delta,
        },
    )
    state["iteration"], state["current_state"] = version, revise_transition.next_state
    state["meta"].pop("user_approved_gate", None)
    if has_gate:
        state["last_gate"] = {}
    state["plan_versions"].append({
        "version": version, "file": plan_filename,
        "hash": meta["hash"], "timestamp": meta["timestamp"],
    })
    _append_to_meta(state, "plan_deltas", delta)
    update_flags_after_revise(plan_dir, payload["flags_addressed"], plan_file=plan_filename, summary=payload["changes_summary"])
    next_step = _next_progress_step(state)
    remaining = _remaining_significant_flags(plan_dir)
    return _finish_step(
        plan_dir, state, args,
        step="revise",
        worker=worker, agent=agent, mode=mode, refreshed=refreshed,
        summary=f"Updated plan to v{version}; addressed {len(payload['flags_addressed'])} flags.",
        artifacts=[plan_filename, meta_filename, "faults.json"],
        output_file=plan_filename,
        artifact_hash=meta["hash"],
        next_step=next_step,
        response_fields={
            "iteration": version,
            "changes_summary": payload["changes_summary"],
            "flags_addressed": payload["flags_addressed"],
            "flags_remaining": remaining,
            "plan_delta_percent": delta,
        },
        history_fields={"flags_addressed": payload["flags_addressed"]},
    )


def handle_gate(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "gate", {STATE_CRITIQUED})
    iteration = state["iteration"]
    gate_signals, signals_filename, signals_artifact = _build_gate_signals_artifact(plan_dir, state, iteration=iteration, root=root)
    worker, agent, mode, refreshed = _run_worker("gate", state, plan_dir, args, root=root)
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
    gate_hash = _write_json_artifact(plan_dir, "gate.json", gate_summary)
    debt_entries_added = _record_gate_debt_entries(root, state, gate_summary, worker.payload)
    if len(state["meta"].get("weighted_scores", [])) < iteration:
        _append_to_meta(state, "weighted_scores", gate_signals["signals"]["weighted_score"])
    result, next_step, summary = _apply_gate_outcome(
        state,
        gate_summary,
        robustness=gate_signals["robustness"],
        plan_dir=plan_dir,
    )
    # Store last_gate AFTER _apply_gate_outcome — the outcome may override
    # the recommendation (e.g. PROCEED → ITERATE when flags are unresolved).
    _store_last_gate(state, gate_summary)
    return _finish_step(
        plan_dir,
        state,
        args,
        step="gate",
        worker=worker,
        agent=agent,
        mode=mode,
        refreshed=refreshed,
        summary=summary,
        artifacts=[signals_filename, "gate.json"],
        output_file="gate.json",
        artifact_hash=gate_hash,
        result=result,
        success=gate_summary["recommendation"] != "PROCEED" or gate_summary["passed"],
        next_step=next_step,
        response_fields=_gate_response_fields(state, gate_summary, debt_entries_added),
        history_fields={"recommendation": gate_summary["recommendation"]},
    )


def _ensure_verification_task(payload: dict, state: dict) -> None:
    """Ensure the task list ends with a test verification task.

    If the last task already looks like a verification/test task, leave it.
    Otherwise append one that depends on all other tasks.
    """
    tasks = payload.get("tasks", [])
    if not tasks:
        return

    # Check if last task is already a verification task
    last_desc = (tasks[-1].get("description") or "").lower()
    test_keywords = ("run test", "run the test", "verify", "verification", "pytest", "test suite", "run existing test")
    if any(kw in last_desc for kw in test_keywords):
        return

    # Build the verification task
    all_ids = [t["id"] for t in tasks]
    next_num = max((int(t["id"].lstrip("T")) for t in tasks if t["id"].startswith("T")), default=0) + 1
    task_id = f"T{next_num}"

    # Pull specific test IDs from the original prompt if available
    idea = state.get("idea", "") or ""
    notes = "\n".join(state.get("notes", []) or [])
    source_text = idea + "\n" + notes

    if "FAIL_TO_PASS" in source_text or "test must pass" in source_text.lower() or "verification" in source_text.lower():
        desc = (
            "Run the tests specified in the task description to verify the fix — run the full test file/module, not just individual functions. "
            "Run the project's existing test suite — do NOT create new test files. "
            "If any test fails, read the error, fix the code, and re-run until all tests pass."
        )
    else:
        desc = (
            "Run tests relevant to the changed files to verify correctness and check for regressions — run the full test file/module, not just individual functions. "
            "Find and run the project's existing test suite — do NOT create new test files. "
            "If any test fails, read the error, fix the code, and re-run until all tests pass."
        )

    verification_task = {
        "id": task_id,
        "description": desc,
        "depends_on": [all_ids[-1]],
        "status": "pending",
        "executor_notes": "",
        "files_changed": [],
        "commands_run": [],
        "evidence_files": [],
        "reviewer_verdict": "",
    }
    tasks.append(verification_task)

    # Add a sense check for it
    sense_checks = payload.get("sense_checks", [])
    sc_num = max((int(sc["id"].lstrip("SC")) for sc in sense_checks if sc["id"].startswith("SC")), default=0) + 1
    sense_checks.append({
        "id": f"SC{sc_num}",
        "task_id": task_id,
        "question": "Did the verification tests pass? Were any regressions found and fixed?",
        "executor_note": "",
        "verdict": "",
    })


def handle_finalize(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "finalize", {STATE_GATED})
    worker, agent, mode, refreshed = _run_worker("finalize", state, plan_dir, args, root=root)
    _validate_finalize_payload(plan_dir, state, worker)
    artifact_hash = _write_finalize_artifacts(plan_dir, worker.payload, state)
    state["current_state"] = STATE_FINALIZED
    return _finish_step(
        plan_dir, state, args,
        step="finalize",
        worker=worker, agent=agent, mode=mode, refreshed=refreshed,
        summary=f"Finalized plan with {len(worker.payload['tasks'])} tasks and {len(worker.payload['watch_items'])} watch items.",
        artifacts=["final.md", "finalize.json"],
        output_file="finalize.json",
        artifact_hash=artifact_hash,
        next_step="execute",
    )


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
    agent, mode, refreshed, model = worker_module.resolve_agent_mode("execute", args)
    # Force fresh session after review kickback to avoid prior-context bias
    if not refreshed and _is_rework_reexecution(state):
        refreshed = True
    if getattr(args, "batch", None) is not None:
        response = dispatch_execute_one_batch(
            root=root,
            plan_dir=plan_dir,
            state=state,
            args=args,
            batch_number=args.batch,
            auto_approve=auto_approve,
            agent=agent,
            mode=mode,
            refreshed=refreshed,
            model=model,
        )
    else:
        response = dispatch_execute_auto_loop(
            root=root,
            plan_dir=plan_dir,
            state=state,
            args=args,
            auto_approve=auto_approve,
            agent=agent,
            mode=mode,
            refreshed=refreshed,
            model=model,
        )
    if not workflow_includes_step(configured_robustness(state), "review") and response.get("state") == STATE_EXECUTED:
        state["current_state"] = STATE_DONE
        save_state(plan_dir, state)
        response["state"] = STATE_DONE
        response["next_step"] = None
    attach_agent_fallback(response, args)
    return response


def _merge_review_verdicts(
    worker_payload: dict[str, Any],
    finalize_data: dict[str, Any],
    issues: list[str],
) -> tuple[int, int, int, int, list[str]]:
    """Merge task verdicts and sense check verdicts into finalize_data.

    Returns (verdict_count, total_tasks, check_count, total_checks, missing_evidence).
    """
    tasks_by_id = {task["id"]: task for task in finalize_data.get("tasks", [])}
    verdict_count, total_tasks = _validate_and_merge_batch(
        worker_payload.get("task_verdicts"),
        required_fields=("task_id", "reviewer_verdict", "evidence_files"),
        targets_by_id=tasks_by_id,
        id_field="task_id",
        merge_fields=("reviewer_verdict", "evidence_files"),
        issues=issues,
        validation_label="task_verdicts",
        merge_label="task_verdict",
        incomplete_message=lambda merged, total: f"Incomplete review: {merged}/{total} tasks received a reviewer verdict.",
        nonempty_fields={"reviewer_verdict"},
        array_fields=("evidence_files",),
    )
    sense_checks_by_id = {sc["id"]: sc for sc in finalize_data.get("sense_checks", [])}
    check_count, total_checks = _validate_and_merge_batch(
        worker_payload.get("sense_check_verdicts"),
        required_fields=("sense_check_id", "verdict"),
        targets_by_id=sense_checks_by_id,
        id_field="sense_check_id",
        merge_fields=("verdict",),
        issues=issues,
        validation_label="sense_check_verdicts",
        merge_label="sense_check_verdict",
        incomplete_message=lambda merged, total: f"Incomplete review: {merged}/{total} sense checks received a verdict.",
        nonempty_fields={"verdict"},
    )
    missing_evidence = _check_done_task_evidence(
        finalize_data.get("tasks", []),
        issues=issues,
        should_classify=lambda task: bool(task.get("reviewer_verdict", "").strip()),
        has_evidence=lambda task: bool(task.get("evidence_files")),
        has_advisory_evidence=lambda task: _is_substantive_reviewer_verdict(task.get("reviewer_verdict", "")),
        missing_message="Done tasks missing reviewer evidence_files without a substantive reviewer_verdict: ",
        advisory_message="Advisory: done tasks rely on substantive reviewer_verdict without evidence_files (FLAG-006 softening): ",
    )
    return verdict_count, total_tasks, check_count, total_checks, missing_evidence


_MAX_REVIEW_REWORK_CYCLES = 3


def _resolve_review_outcome(
    review_verdict: str,
    verdict_count: int,
    total_tasks: int,
    check_count: int,
    total_checks: int,
    missing_evidence: list[str],
    state: PlanState,
    issues: list[str],
) -> tuple[str, str, str | None]:
    """Determine review result, next state, and next step.

    Returns (result, next_state, next_step).
    """
    blocked = (
        verdict_count < total_tasks
        or check_count < total_checks
        or bool(missing_evidence)
    )
    if blocked:
        return "blocked", STATE_EXECUTED, "review"

    rework_requested = review_verdict == "needs_rework"
    if rework_requested:
        prior_rework_count = sum(
            1 for entry in state.get("history", [])
            if entry.get("step") == "review" and entry.get("result") == "needs_rework"
        )
        if prior_rework_count >= _MAX_REVIEW_REWORK_CYCLES:
            issues.append(
                f"Max review rework cycles ({_MAX_REVIEW_REWORK_CYCLES}) reached. "
                "Force-proceeding to done despite unresolved review issues."
            )
        else:
            return "needs_rework", STATE_FINALIZED, "execute"

    return "success", STATE_DONE, None


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

    # Validate verdict
    review_verdict = worker.payload.get("review_verdict")
    if review_verdict not in {"approved", "needs_rework"}:
        issues.append("Invalid review_verdict; expected 'approved' or 'needs_rework'.")
        review_verdict = "needs_rework"

    # Merge verdicts into finalize data
    verdict_count, total_tasks, check_count, total_checks, missing_evidence = _merge_review_verdicts(
        worker.payload, finalize_data, issues,
    )

    # Save updated finalize data
    atomic_write_json(plan_dir / "finalize.json", finalize_data)
    atomic_write_text(plan_dir / "final.md", render_final_md(finalize_data, phase="review"))
    finalize_hash = sha256_file(plan_dir / "finalize.json")

    # Determine outcome
    result, next_state, next_step = _resolve_review_outcome(
        review_verdict, verdict_count, total_tasks,
        check_count, total_checks, missing_evidence,
        state, issues,
    )
    state["current_state"] = next_state

    # Record history
    apply_session_update(state, "review", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "review",
            duration_ms=worker.duration_ms, cost_usd=worker.cost_usd,
            result=result,
            worker=worker, agent=agent, mode=mode,
            output_file="review.json",
            prompt_tokens=worker.prompt_tokens,
            completion_tokens=worker.completion_tokens,
            total_tokens=worker.total_tokens,
            artifact_hash=sha256_file(plan_dir / "review.json"),
            finalize_hash=finalize_hash,
        ),
    )
    save_state(plan_dir, state)

    # Build response
    passed = sum(1 for c in worker.payload.get("criteria", []) if c.get("pass") in (True, "pass"))
    total = len(worker.payload.get("criteria", []))
    if result == "blocked":
        summary = _build_review_blocked_message(
            verdict_count=verdict_count, total_tasks=total_tasks,
            check_count=check_count, total_checks=total_checks,
            missing_reviewer_evidence=missing_evidence,
        )
    elif result == "needs_rework":
        summary = "Review requested another execute pass. Re-run execute using the review findings as context."
    else:
        summary = f"Review complete: {passed}/{total} success criteria passed."

    response: StepResponse = {
        "success": result == "success",
        "step": "review",
        "summary": summary,
        "artifacts": ["review.json", "finalize.json", "final.md"],
        "next_step": next_step,
        "state": next_state,
        "issues": issues,
        "rework_items": list(worker.payload.get("rework_items", [])),
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
    if state["current_state"] == STATE_EXECUTED:
        # Force-proceed from review loop: mark as done despite review issues
        _append_to_meta(state, "overrides", {"action": "force-proceed", "timestamp": now_utc(), "reason": args.reason})
        state["current_state"] = STATE_DONE
        save_state(plan_dir, state)
        return {
            "success": True,
            "step": "override",
            "summary": "Force-proceeded past review into done state.",
            "next_step": None,
            "state": STATE_DONE,
        }
    if state["current_state"] != STATE_CRITIQUED:
        raise CliError(
            "invalid_transition",
            "force-proceed is only supported from critiqued or executed state",
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
    next_steps = workflow_next(state)
    return {
        "success": True,
        "step": "override",
        "summary": f"Re-entered planning loop at iteration {state['iteration']}. Reason: {reason}",
        "next_step": next_steps[0] if next_steps else None,
        "state": STATE_PLANNED,
        "plan_file": str(plan_file),
        "message": f"Edit {plan_file.name} to incorporate your changes, then run the next step.",
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
