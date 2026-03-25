from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import megaplan.workers as worker_module
from megaplan._core import (
    atomic_write_json,
    atomic_write_text,
    batch_artifact_path,
    compute_global_batches,
    compute_task_batches,
    list_batch_artifacts,
    load_config,
    read_json,
    render_final_md,
    sha256_file,
)
from megaplan.evaluation import _parse_git_status_paths, validate_execution_evidence
from megaplan.merge import _validate_and_merge_batch
from megaplan.prompts import _execute_batch_prompt, _execute_prompt
from megaplan.quality import capture_before_line_counts, run_quality_checks
from megaplan.types import CliError, PlanState, STATE_EXECUTED, STATE_FINALIZED, StepResponse
from megaplan.workers import WorkerResult


@dataclass
class BatchResult:
    worker: WorkerResult
    agent: str
    mode: str
    refreshed: bool
    payload: dict[str, Any]
    batch_number: int
    batch_task_ids: list[str]
    batch_sense_check_ids: list[str]
    merged_task_count: int
    total_task_count: int
    acknowledged_sense_check_count: int
    total_sense_check_count: int
    missing_task_evidence: list[str]
    execution_audit: dict[str, Any]
    finalize_hash: str


def _check_done_task_evidence(
    tasks: list[dict[str, Any]],
    *,
    issues: list[str],
    should_classify: Callable[[dict[str, Any]], bool],
    has_evidence: Callable[[dict[str, Any]], bool],
    has_advisory_evidence: Callable[[dict[str, Any]], bool],
    missing_message: str,
    advisory_message: str,
) -> list[str]:
    missing_task_ids: list[str] = []
    advisory_task_ids: list[str] = []
    for task in tasks:
        if task.get("status") != "done" or not should_classify(task):
            continue
        if has_evidence(task):
            continue
        if has_advisory_evidence(task):
            advisory_task_ids.append(task["id"])
        else:
            missing_task_ids.append(task["id"])
    if missing_task_ids:
        issues.append(missing_message + ", ".join(missing_task_ids))
    if advisory_task_ids:
        issues.append(advisory_message + ", ".join(advisory_task_ids))
    return missing_task_ids


def _resolve_execute_approval_mode(*, auto_approve: bool, user_approved_gate: bool) -> str:
    if auto_approve:
        return "auto_approve"
    if user_approved_gate:
        return "user_approved"
    return "manual"


def _format_execute_tracking_note(
    *,
    merged_count: int,
    total_tasks: int,
    acknowledged_count: int,
    total_checks: int,
) -> str:
    tracking_bits: list[str] = []
    if total_tasks > 0:
        tracking_bits.append(f"{merged_count}/{total_tasks} tasks tracked")
    if total_checks > 0:
        tracking_bits.append(f"{acknowledged_count}/{total_checks} sense checks acknowledged")
    return f" ({', '.join(tracking_bits)})" if tracking_bits else ""


def _snapshot_task_statuses(tasks: list[dict[str, Any]]) -> dict[str, str]:
    return {
        task["id"]: str(task.get("status", ""))
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    }


def _append_execute_reconciliation_advisories(
    *,
    before_statuses: dict[str, str],
    tasks_by_id: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    for task_id, before_status in before_statuses.items():
        after_status = str(tasks_by_id.get(task_id, {}).get("status", ""))
        if before_status not in {"done", "skipped"} or after_status == before_status:
            continue
        issues.append(
            f"Advisory: task {task_id} was {before_status!r} on disk before merge but structured output set it to {after_status!r}. Structured output remains authoritative."
        )


def _normalize_execute_claimed_path(path: str, project_dir: Path | None = None) -> str:
    p = Path(path.strip())
    if project_dir is not None and p.is_absolute():
        try:
            p = p.relative_to(project_dir)
        except ValueError:
            pass
    return p.as_posix()


def _repo_path_hash(project_dir: Path, relative_path: str) -> str:
    target = project_dir / relative_path
    if not target.exists():
        return "<missing>"
    if target.is_dir():
        return "<directory>"
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _capture_git_status_snapshot(project_dir: Path) -> tuple[dict[str, str], str | None]:
    if not (project_dir / ".git").exists():
        return {}, "Project directory is not a git repository."
    try:
        process = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(project_dir),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        return {}, "git not found on PATH."
    except subprocess.TimeoutExpired:
        return {}, "git status timed out."
    if process.returncode != 0:
        return {}, f"git status failed: {process.stderr.strip() or process.stdout.strip()}"
    paths = _parse_git_status_paths(process.stdout)
    return {path: _repo_path_hash(project_dir, path) for path in paths}, None


def _observed_batch_paths(
    *,
    project_dir: Path,
    before_snapshot: dict[str, str],
    after_snapshot: dict[str, str],
) -> set[str]:
    observed: set[str] = set()
    for path in set(before_snapshot) | set(after_snapshot):
        before_hash = before_snapshot.get(path)
        after_hash = after_snapshot.get(path)
        if after_hash is None:
            after_hash = _repo_path_hash(project_dir, path)
        if before_hash is None or before_hash != after_hash:
            observed.add(path)
    return observed


def _collect_execute_claimed_paths(payload: dict[str, Any], project_dir: Path | None = None) -> set[str]:
    """Collect top-level files_changed only for git observation comparison.

    Per-task files_changed are intentionally excluded — they often include
    files the executor read/verified but didn't modify, which causes false
    phantom-claim deviations when compared against git status deltas.
    Per-task evidence is validated separately by the audit path.
    """
    return {
        _normalize_execute_claimed_path(path, project_dir)
        for path in payload.get("files_changed", [])
        if isinstance(path, str) and path.strip()
    }


def _stable_unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_aggregate_execution_payload(
    batch_payloads: list[dict[str, Any]],
    *,
    completed_batches: int,
    total_batches: int,
) -> dict[str, Any]:
    outputs = [
        f"Batch {index + 1}: {payload.get('output', '')}".strip()
        for index, payload in enumerate(batch_payloads)
    ]
    files_changed: list[str] = []
    commands_run: list[str] = []
    deviations: list[str] = []
    task_updates: list[dict[str, Any]] = []
    sense_check_acknowledgments: list[dict[str, Any]] = []
    for payload in batch_payloads:
        files_changed.extend([path for path in payload.get("files_changed", []) if isinstance(path, str)])
        commands_run.extend([command for command in payload.get("commands_run", []) if isinstance(command, str)])
        deviations.extend([issue for issue in payload.get("deviations", []) if isinstance(issue, str)])
        task_updates.extend([item for item in payload.get("task_updates", []) if isinstance(item, dict)])
        sense_check_acknowledgments.extend(
            [item for item in payload.get("sense_check_acknowledgments", []) if isinstance(item, dict)]
        )
    output = f"Aggregated execute batches: completed {completed_batches}/{total_batches}."
    if outputs:
        output = output + "\n" + "\n".join(outputs)
    return {
        "output": output,
        "files_changed": _stable_unique_strings(files_changed),
        "commands_run": _stable_unique_strings(commands_run),
        "deviations": deviations,
        "task_updates": task_updates,
        "sense_check_acknowledgments": sense_check_acknowledgments,
    }


def _active_sense_check_ids(finalize_data: dict[str, Any], active_task_ids: set[str]) -> list[str]:
    return [
        sense_check["id"]
        for sense_check in finalize_data.get("sense_checks", [])
        if isinstance(sense_check, dict)
        and isinstance(sense_check.get("id"), str)
        and sense_check.get("task_id") in active_task_ids
    ]


def _count_execute_tracking(
    finalize_data: dict[str, Any],
    *,
    active_task_ids: set[str],
    active_sense_check_ids: set[str],
) -> tuple[int, int, int, int]:
    tracked_tasks = sum(
        1
        for task in finalize_data.get("tasks", [])
        if task.get("id") in active_task_ids and task.get("status") in {"done", "skipped"}
    )
    acknowledged_checks = sum(
        1
        for sense_check in finalize_data.get("sense_checks", [])
        if sense_check.get("id") in active_sense_check_ids and str(sense_check.get("executor_note", "")).strip()
    )
    return tracked_tasks, len(active_task_ids), acknowledged_checks, len(active_sense_check_ids)


def build_blocking_reasons(
    *,
    tracked_tasks: int,
    total_tasks: int,
    acknowledged_checks: int,
    total_checks: int,
    missing_task_evidence: list[str],
    timeout_reason: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    if tracked_tasks < total_tasks:
        reasons.append(f"{total_tasks - tracked_tasks}/{total_tasks} tasks have no executor update")
    if acknowledged_checks < total_checks:
        reasons.append(f"{total_checks - acknowledged_checks}/{total_checks} sense checks have no executor acknowledgment")
    if missing_task_evidence:
        reasons.append(
            "done tasks missing both files_changed and commands_run: "
            + ", ".join(missing_task_evidence)
        )
    if timeout_reason is not None:
        reasons.append(timeout_reason)
    return reasons


def _observe_git_changes(
    *,
    project_dir: Path,
    payload: dict[str, Any],
    before_snapshot: dict[str, str],
    before_error: str | None,
    batch_number: int,
    batches_total: int,
    capture_git_status_snapshot_fn: Callable[[Path], tuple[dict[str, str], str | None]],
) -> list[str]:
    issues: list[str] = []
    if before_error is not None:
        issues.append(f"Advisory observation skip before batch {batch_number}/{batches_total}: {before_error}")
    after_snapshot, after_error = capture_git_status_snapshot_fn(project_dir)
    if after_error is not None:
        issues.append(f"Advisory observation skip after batch {batch_number}/{batches_total}: {after_error}")
    elif before_error is None:
        observed_paths = _observed_batch_paths(
            project_dir=project_dir,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
        claimed_paths = _collect_execute_claimed_paths(payload, project_dir)
        phantom_claims = sorted(claimed_paths - observed_paths)
        if phantom_claims:
            issues.append(
                "Advisory observation mismatch: executor claimed files not observed in git status/content hash delta: "
                + ", ".join(phantom_claims)
            )
        unclaimed_changes = sorted(observed_paths - claimed_paths)
        if unclaimed_changes:
            issues.append(
                "Advisory observation mismatch: git status/content hash delta found unclaimed files: "
                + ", ".join(unclaimed_changes)
            )
    return issues


def _collect_quality_deviations(
    *,
    project_dir: Path,
    before_snapshot: dict[str, str],
    before_line_counts: dict[str, int],
    quality_config: dict[str, Any],
    capture_git_status_snapshot_fn: Callable[[Path], tuple[dict[str, str], str | None]],
) -> list[str]:
    try:
        after_snapshot, after_error = capture_git_status_snapshot_fn(project_dir)
    except Exception as exc:  # pragma: no cover - defensive guard for advisory-only quality checks.
        return [f"Advisory quality: skipped quality checks because post-batch git snapshot failed: {exc}"]
    if after_error is not None:
        return [f"Advisory quality: skipped quality checks because post-batch git snapshot failed: {after_error}"]
    changed_paths = _observed_batch_paths(
        project_dir=project_dir,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )
    return run_quality_checks(
        project_dir,
        changed_paths=changed_paths,
        before_line_counts=before_line_counts,
        config=quality_config,
    )


def _merge_batch_results(
    *,
    finalize_data: dict[str, Any],
    payload: dict[str, Any],
    batch_task_ids: list[str],
    batch_sense_check_ids: list[str],
    issues: list[str],
) -> tuple[int, int, int, int]:
    batch_task_id_set = set(batch_task_ids)
    batch_sense_check_id_set = set(batch_sense_check_ids)
    pre_merge_statuses = _snapshot_task_statuses(
        [task for task in finalize_data.get("tasks", []) if task.get("id") in batch_task_id_set]
    )
    batch_tasks_by_id = {
        task["id"]: task
        for task in finalize_data.get("tasks", [])
        if task.get("id") in batch_task_id_set
    }
    merged_count, total_batch_tasks = _validate_and_merge_batch(
        payload.get("task_updates"),
        required_fields=("task_id", "status", "executor_notes", "files_changed", "commands_run"),
        targets_by_id=batch_tasks_by_id,
        id_field="task_id",
        merge_fields=("status", "executor_notes", "files_changed", "commands_run"),
        issues=issues,
        validation_label="task_updates",
        merge_label="task_update",
        incomplete_message=(
            lambda merged, total: (
                f"{total - merged}/{total} batch tasks have no executor update — tracking is incomplete."
            )
        ),
        enum_fields={"status": {"done", "skipped"}},
        nonempty_fields={"executor_notes"},
        array_fields=("files_changed", "commands_run"),
    )
    batch_sense_checks_by_id = {
        sense_check["id"]: sense_check
        for sense_check in finalize_data.get("sense_checks", [])
        if sense_check.get("id") in batch_sense_check_id_set
    }
    acknowledged_count, total_batch_checks = _validate_and_merge_batch(
        payload.get("sense_check_acknowledgments"),
        required_fields=("sense_check_id", "executor_note"),
        targets_by_id=batch_sense_checks_by_id,
        id_field="sense_check_id",
        merge_fields=("executor_note",),
        issues=issues,
        validation_label="sense_check_acknowledgments",
        merge_label="sense_check_acknowledgment",
        incomplete_message=(
            lambda merged, total: (
                f"{total - merged}/{total} batch sense checks have no executor acknowledgment — tracking is incomplete."
            )
        ),
        nonempty_fields={"executor_note"},
    )
    _append_execute_reconciliation_advisories(
        before_statuses=pre_merge_statuses,
        tasks_by_id=batch_tasks_by_id,
        issues=issues,
    )
    return merged_count, total_batch_tasks, acknowledged_count, total_batch_checks


def _run_and_merge_batch(
    *,
    root: Path,
    plan_dir: Path,
    state: PlanState,
    args: argparse.Namespace,
    agent: str,
    mode: str,
    refreshed: bool,
    prompt_override: str | None,
    batch_task_ids: list[str],
    batch_sense_check_ids: list[str],
    finalize_data: dict[str, Any],
    batch_number: int,
    batches_total: int,
    quality_config: dict[str, Any],
    capture_git_status_snapshot_fn: Callable[[Path], tuple[dict[str, str], str | None]] = _capture_git_status_snapshot,
) -> BatchResult:
    project_dir = Path(state["config"]["project_dir"])
    before_snapshot, before_error = capture_git_status_snapshot_fn(project_dir)
    before_line_counts = capture_before_line_counts(project_dir, before_snapshot.keys())
    worker, agent, mode, refreshed = worker_module.run_step_with_worker(
        "execute",
        state,
        plan_dir,
        args,
        root=root,
        resolved=(agent, mode, refreshed),
        prompt_override=prompt_override,
    )
    payload = dict(worker.payload)
    deviations = list(payload.get("deviations", []))
    batch_task_id_set = set(batch_task_ids)
    deviations.extend(
        _observe_git_changes(
            project_dir=project_dir,
            payload=payload,
            before_snapshot=before_snapshot,
            before_error=before_error,
            batch_number=batch_number,
            batches_total=batches_total,
            capture_git_status_snapshot_fn=capture_git_status_snapshot_fn,
        )
    )
    deviations.extend(
        _collect_quality_deviations(
            project_dir=project_dir,
            before_snapshot=before_snapshot,
            before_line_counts=before_line_counts,
            quality_config=quality_config,
            capture_git_status_snapshot_fn=capture_git_status_snapshot_fn,
        )
    )
    merged_count, total_batch_tasks, acknowledged_count, total_batch_checks = _merge_batch_results(
        finalize_data=finalize_data,
        payload=payload,
        batch_task_ids=batch_task_ids,
        batch_sense_check_ids=batch_sense_check_ids,
        issues=deviations,
    )
    missing_task_evidence = _check_done_task_evidence(
        finalize_data.get("tasks", []),
        issues=deviations,
        should_classify=lambda task: task.get("id") in batch_task_id_set,
        has_evidence=lambda task: bool(task.get("files_changed")),
        has_advisory_evidence=lambda task: bool(task.get("commands_run")),
        missing_message="Done tasks missing both files_changed and commands_run: ",
        advisory_message="Advisory: done tasks rely on commands_run without files_changed (FLAG-006 softening): ",
    )
    execution_audit = validate_execution_evidence(finalize_data, project_dir)
    if execution_audit["skipped"]:
        deviations.append(f"Advisory audit skip: {execution_audit['reason']}")
    for finding in execution_audit["findings"]:
        deviations.append(f"Advisory audit finding: {finding}")
    payload["deviations"] = deviations
    atomic_write_json(batch_artifact_path(plan_dir, batch_number), payload)
    atomic_write_json(plan_dir / "execution_audit.json", execution_audit)
    atomic_write_json(plan_dir / "finalize.json", finalize_data)
    atomic_write_text(plan_dir / "final.md", render_final_md(finalize_data, phase="execute"))
    return BatchResult(
        worker=worker,
        agent=agent,
        mode=mode,
        refreshed=refreshed,
        payload=payload,
        batch_number=batch_number,
        batch_task_ids=list(batch_task_ids),
        batch_sense_check_ids=list(batch_sense_check_ids),
        merged_task_count=merged_count,
        total_task_count=total_batch_tasks,
        acknowledged_sense_check_count=acknowledged_count,
        total_sense_check_count=total_batch_checks,
        missing_task_evidence=missing_task_evidence,
        execution_audit=execution_audit,
        finalize_hash=sha256_file(plan_dir / "finalize.json"),
    )


def _reset_timeout_invalid_tasks(
    finalize_data: dict[str, Any],
    *,
    execution_audit: dict[str, Any],
    issues: list[str],
) -> list[str]:
    reset_reasons: dict[str, list[str]] = {}
    missing_task_ids = _check_done_task_evidence(
        finalize_data.get("tasks", []),
        issues=issues,
        should_classify=lambda task: True,
        has_evidence=lambda task: bool(task.get("files_changed")),
        has_advisory_evidence=lambda task: bool(task.get("commands_run")),
        missing_message="Done tasks missing both files_changed and commands_run during timeout recovery: ",
        advisory_message="Advisory: done tasks rely on commands_run without files_changed during timeout recovery: ",
    )
    for task_id in missing_task_ids:
        reset_reasons.setdefault(task_id, []).append("missing both files_changed and commands_run")

    if not execution_audit.get("skipped"):
        files_in_diff = {
            _normalize_execute_claimed_path(path)
            for path in execution_audit.get("files_in_diff", [])
            if isinstance(path, str) and path.strip()
        }
        for task in finalize_data.get("tasks", []):
            if task.get("status") != "done":
                continue
            claimed_paths = [
                _normalize_execute_claimed_path(path)
                for path in task.get("files_changed", [])
                if isinstance(path, str) and path.strip()
            ]
            if claimed_paths and any(path not in files_in_diff for path in claimed_paths):
                reset_reasons.setdefault(task["id"], []).append("claimed files not present in git status")

    for task in finalize_data.get("tasks", []):
        reasons = reset_reasons.get(task.get("id"))
        if not reasons:
            continue
        note_prefix = str(task.get("executor_notes", "")).strip()
        reset_note = "Timeout recovery reset this task to pending because " + " and ".join(reasons) + "."
        task["status"] = "pending"
        task["executor_notes"] = f"{note_prefix} {reset_note}".strip()

    if reset_reasons:
        issues.append(
            "Reset timed-out done tasks to pending after evidence validation: "
            + ", ".join(sorted(reset_reasons))
        )
    return sorted(reset_reasons)


def _timeout_checkpoint_path(plan_dir: Path, *, batch_number: int | None) -> Path:
    if batch_number is None:
        return plan_dir / "execution_checkpoint.json"
    return batch_artifact_path(plan_dir, batch_number)


def _merge_timeout_checkpoint(
    *,
    finalize_data: dict[str, Any],
    checkpoint_data: dict[str, Any],
    checkpoint_name: str,
    issues: list[str],
) -> None:
    tasks_by_id = {
        task["id"]: task
        for task in finalize_data.get("tasks", [])
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    }
    merged_tasks, _ = _validate_and_merge_batch(
        checkpoint_data.get("task_updates"),
        required_fields=("task_id", "status", "executor_notes", "files_changed", "commands_run"),
        targets_by_id=tasks_by_id,
        id_field="task_id",
        merge_fields=("status", "executor_notes", "files_changed", "commands_run"),
        issues=issues,
        validation_label=f"{checkpoint_name}.task_updates",
        merge_label="checkpoint task_update",
        enum_fields={"status": {"done", "skipped"}},
        nonempty_fields={"executor_notes"},
        array_fields=("files_changed", "commands_run"),
    )
    sense_checks_by_id = {
        sense_check["id"]: sense_check
        for sense_check in finalize_data.get("sense_checks", [])
        if isinstance(sense_check, dict) and isinstance(sense_check.get("id"), str)
    }
    merged_checks, _ = _validate_and_merge_batch(
        checkpoint_data.get("sense_check_acknowledgments"),
        required_fields=("sense_check_id", "executor_note"),
        targets_by_id=sense_checks_by_id,
        id_field="sense_check_id",
        merge_fields=("executor_note",),
        issues=issues,
        validation_label=f"{checkpoint_name}.sense_check_acknowledgments",
        merge_label="checkpoint sense_check_acknowledgment",
        nonempty_fields={"executor_note"},
    )
    if merged_tasks > 0 or merged_checks > 0:
        issues.append(
            f"Recovered timeout checkpoint from {checkpoint_name}: merged {merged_tasks} task update(s) and {merged_checks} sense check acknowledgment(s)."
        )


def _recover_execute_timeout(
    *,
    plan_dir: Path,
    state: PlanState,
    error: CliError,
    agent: str,
    mode: str,
    refreshed: bool,
    auto_approve: bool,
    args: argparse.Namespace,
    batch_number: int | None,
    apply_session_update_fn: Callable[..., None],
    append_history_fn: Callable[[PlanState, dict[str, Any]], None],
    make_history_entry_fn: Callable[..., dict[str, Any]],
    save_state_fn: Callable[[Path, PlanState], None],
    attach_agent_fallback_fn: Callable[[StepResponse, argparse.Namespace], None],
    store_raw_worker_output_fn: Callable[[Path, str, int, str], str],
    persist_state: bool = True,
) -> StepResponse:
    deviations = [f"Execute timed out: {error.message}"]
    finalize_data = read_json(plan_dir / "finalize.json")
    checkpoint_path = _timeout_checkpoint_path(plan_dir, batch_number=batch_number)
    try:
        checkpoint_data = read_json(checkpoint_path)
    except FileNotFoundError:
        deviations.append(f"Advisory: timeout checkpoint {checkpoint_path.name} was not found.")
    except json.JSONDecodeError as exc:
        deviations.append(f"Advisory: timeout checkpoint {checkpoint_path.name} was not valid JSON: {exc}")
    else:
        if isinstance(checkpoint_data, dict):
            _merge_timeout_checkpoint(
                finalize_data=finalize_data,
                checkpoint_data=checkpoint_data,
                checkpoint_name=checkpoint_path.name,
                issues=deviations,
            )
        else:
            deviations.append(f"Advisory: timeout checkpoint {checkpoint_path.name} did not contain an object.")

    project_dir = Path(state["config"]["project_dir"])
    initial_audit = validate_execution_evidence(finalize_data, project_dir)
    if initial_audit["skipped"]:
        deviations.append(f"Advisory audit skip during timeout recovery: {initial_audit['reason']}")
    for finding in initial_audit["findings"]:
        deviations.append(f"Advisory audit finding during timeout recovery: {finding}")

    _reset_timeout_invalid_tasks(
        finalize_data,
        execution_audit=initial_audit,
        issues=deviations,
    )
    execution_audit = validate_execution_evidence(finalize_data, project_dir)
    atomic_write_json(plan_dir / "execution_audit.json", execution_audit)
    atomic_write_json(plan_dir / "finalize.json", finalize_data)
    atomic_write_text(plan_dir / "final.md", render_final_md(finalize_data, phase="execute"))

    finalize_hash = sha256_file(plan_dir / "finalize.json")
    raw_output = str(error.extra.get("raw_output") or error.message)
    raw_name = store_raw_worker_output_fn(plan_dir, "execute", state["iteration"], raw_output)
    session_id = error.extra.get("session_id")
    timeout_worker = WorkerResult(
        payload={},
        raw_output=raw_output,
        duration_ms=0,
        cost_usd=0.0,
        session_id=session_id if isinstance(session_id, str) else None,
    )
    if persist_state:
        apply_session_update_fn(state, "execute", agent, timeout_worker.session_id, mode=mode, refreshed=refreshed)
    user_approved_gate = bool(state["meta"].get("user_approved_gate", False))
    approval_mode = _resolve_execute_approval_mode(
        auto_approve=auto_approve,
        user_approved_gate=user_approved_gate,
    )
    if persist_state:
        append_history_fn(
            state,
            make_history_entry_fn(
                "execute",
                duration_ms=0,
                cost_usd=0.0,
                result="timeout",
                worker=timeout_worker,
                agent=agent,
                mode=mode,
                output_file="finalize.json",
                artifact_hash=finalize_hash,
                finalize_hash=finalize_hash,
                raw_output_file=raw_name,
                message=error.message,
                approval_mode=approval_mode,
            ),
        )
        save_state_fn(plan_dir, state)

    tasks = finalize_data.get("tasks", [])
    completed_tasks = [task for task in tasks if task.get("status") in {"done", "skipped"}]
    files_changed = sorted(
        {
            path
            for task in completed_tasks
            for path in task.get("files_changed", [])
            if isinstance(path, str) and path.strip()
        }
    )
    summary = (
        "Execute timed out after partial progress. "
        f"{len(completed_tasks)}/{len(tasks)} tasks remain marked done or skipped on disk. "
        "Re-run execute to finish and re-emit structured output."
    )
    response: StepResponse = {
        "success": False,
        "step": "execute",
        "summary": summary,
        "artifacts": ["execution_audit.json", "finalize.json", "final.md"],
        "next_step": "execute",
        "state": STATE_FINALIZED,
        "files_changed": files_changed,
        "deviations": deviations,
        "warnings": [summary],
        "auto_approve": auto_approve,
        "user_approved_gate": user_approved_gate,
    }
    attach_agent_fallback_fn(response, args)
    return response


def _append_trace_output(plan_dir: Path, trace_output: str | None) -> bool:
    if trace_output is None:
        return False
    trace_path = plan_dir / "execution_trace.jsonl"
    existing_trace = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
    atomic_write_text(trace_path, existing_trace + trace_output)
    return True


def handle_execute_one_batch(
    *,
    root: Path,
    plan_dir: Path,
    state: PlanState,
    args: argparse.Namespace,
    batch_number: int,
    auto_approve: bool,
    agent: str,
    mode: str,
    refreshed: bool,
    apply_session_update_fn: Callable[..., None],
    append_history_fn: Callable[[PlanState, dict[str, Any]], None],
    make_history_entry_fn: Callable[..., dict[str, Any]],
    save_state_fn: Callable[[Path, PlanState], None],
    attach_agent_fallback_fn: Callable[[StepResponse, argparse.Namespace], None],
    store_raw_worker_output_fn: Callable[[Path, str, int, str], str],
    record_step_failure_fn: Callable[..., None],
    capture_git_status_snapshot_fn: Callable[[Path], tuple[dict[str, str], str | None]] = _capture_git_status_snapshot,
) -> StepResponse:
    finalize_data = read_json(plan_dir / "finalize.json")
    global_config = load_config()
    quality_config = global_config.get("quality_checks", {})
    global_batches = compute_global_batches(finalize_data)
    batches_total = len(global_batches)

    if batch_number < 1 or batch_number > batches_total:
        raise CliError(
            "batch_out_of_range",
            f"--batch {batch_number} is out of range. Plan has {batches_total} batch(es) (1-indexed).",
        )

    tasks = finalize_data.get("tasks", [])
    completed_ids = {
        task["id"]
        for task in tasks
        if task.get("status") in {"done", "skipped"} and isinstance(task.get("id"), str)
    }
    for prior_idx in range(batch_number - 1):
        prior_batch = global_batches[prior_idx]
        missing = [task_id for task_id in prior_batch if task_id not in completed_ids]
        if missing:
            raise CliError(
                "batch_prerequisites",
                f"Batch {batch_number} requires batches 1..{batch_number - 1} to be complete. "
                f"Batch {prior_idx + 1} has incomplete tasks: {', '.join(missing)}",
            )

    batch_task_ids = global_batches[batch_number - 1]
    active_task_ids = set(batch_task_ids)
    batch_sense_check_ids = _active_sense_check_ids(finalize_data, active_task_ids)
    batch_prompt = _execute_batch_prompt(state, plan_dir, batch_task_ids, completed_ids, root=root)

    try:
        result = _run_and_merge_batch(
            root=root,
            plan_dir=plan_dir,
            state=state,
            args=args,
            agent=agent,
            mode=mode,
            refreshed=refreshed,
            prompt_override=batch_prompt,
            batch_task_ids=batch_task_ids,
            batch_sense_check_ids=batch_sense_check_ids,
            finalize_data=finalize_data,
            batch_number=batch_number,
            batches_total=batches_total,
            quality_config=quality_config,
            capture_git_status_snapshot_fn=capture_git_status_snapshot_fn,
        )
    except CliError as error:
        if error.code == "worker_timeout":
            return _recover_execute_timeout(
                plan_dir=plan_dir,
                state=state,
                error=error,
                agent=agent,
                mode=mode,
                refreshed=refreshed,
                auto_approve=auto_approve,
                args=args,
                batch_number=batch_number,
                apply_session_update_fn=apply_session_update_fn,
                append_history_fn=append_history_fn,
                make_history_entry_fn=make_history_entry_fn,
                save_state_fn=save_state_fn,
                attach_agent_fallback_fn=attach_agent_fallback_fn,
                store_raw_worker_output_fn=store_raw_worker_output_fn,
            )
        record_step_failure_fn(plan_dir, state, step="execute", iteration=state["iteration"], error=error)
        raise

    apply_session_update_fn(state, "execute", result.agent, result.worker.session_id, mode=result.mode, refreshed=result.refreshed)
    trace_written = _append_trace_output(plan_dir, result.worker.trace_output)
    blocking_reasons = build_blocking_reasons(
        tracked_tasks=result.merged_task_count,
        total_tasks=result.total_task_count,
        acknowledged_checks=result.acknowledged_sense_check_count,
        total_checks=result.total_sense_check_count,
        missing_task_evidence=result.missing_task_evidence,
    )
    blocked = bool(blocking_reasons)

    all_tasks = finalize_data.get("tasks", [])
    is_final_batch = batch_number == batches_total
    all_tracked = all(
        task.get("status") in {"done", "skipped"}
        for task in all_tasks
        if isinstance(task.get("id"), str)
    )

    if is_final_batch and all_tracked and not blocked:
        batch_payloads = [read_json(path) for path in list_batch_artifacts(plan_dir)]
        aggregate_payload = _build_aggregate_execution_payload(
            batch_payloads,
            completed_batches=len(batch_payloads),
            total_batches=batches_total,
        )
        atomic_write_json(plan_dir / "execution.json", aggregate_payload)
        state["current_state"] = STATE_EXECUTED

    user_approved_gate = bool(state["meta"].get("user_approved_gate", False))
    approval_mode = _resolve_execute_approval_mode(
        auto_approve=auto_approve,
        user_approved_gate=user_approved_gate,
    )
    append_history_fn(
        state,
        make_history_entry_fn(
            "execute",
            duration_ms=result.worker.duration_ms,
            cost_usd=result.worker.cost_usd,
            result="blocked" if blocked else "success" if (is_final_batch and all_tracked) else "partial",
            worker=result.worker,
            agent=result.agent,
            mode=result.mode,
            output_file=f"execution_batch_{batch_number}.json",
            artifact_hash=sha256_file(batch_artifact_path(plan_dir, batch_number)),
            finalize_hash=result.finalize_hash,
            approval_mode=approval_mode,
        ),
    )
    save_state_fn(plan_dir, state)

    batches_remaining = batches_total - batch_number
    tracking_note = _format_execute_tracking_note(
        merged_count=result.merged_task_count,
        total_tasks=result.total_task_count,
        acknowledged_count=result.acknowledged_sense_check_count,
        total_checks=result.total_sense_check_count,
    )
    artifacts = [f"execution_batch_{batch_number}.json", "execution_audit.json", "finalize.json", "final.md"]
    if is_final_batch and all_tracked and not blocked:
        artifacts.insert(0, "execution.json")
    if trace_written:
        artifacts.append("execution_trace.jsonl")

    if blocked:
        summary = "Blocked: " + "; ".join(blocking_reasons) + ". Re-run execute to complete tracking."
        next_step = "execute"
        response_state = STATE_FINALIZED
    elif is_final_batch and all_tracked:
        summary = result.payload.get("output", "Batch complete.") + tracking_note
        next_step = "review"
        response_state = STATE_EXECUTED
    else:
        summary = (
            f"Batch {batch_number}/{batches_total} complete.{tracking_note} "
            f"{batches_remaining} batch(es) remaining."
        )
        next_step = "execute"
        response_state = STATE_FINALIZED

    response: StepResponse = {
        "success": not blocked,
        "step": "execute",
        "summary": summary,
        "artifacts": artifacts,
        "next_step": next_step,
        "state": response_state,
        "batch": batch_number,
        "batches_total": batches_total,
        "batches_remaining": batches_remaining,
        "files_changed": result.payload.get("files_changed", []),
        "deviations": result.payload.get("deviations", []),
        "warnings": [summary] if blocked else [],
        "auto_approve": auto_approve,
        "user_approved_gate": user_approved_gate,
    }
    if next_step == "execute" and not blocked:
        response["guidance"] = f"Run --batch {batch_number + 1}"
    attach_agent_fallback_fn(response, args)
    return response


def handle_execute_auto_loop(
    *,
    root: Path,
    plan_dir: Path,
    state: PlanState,
    args: argparse.Namespace,
    auto_approve: bool,
    agent: str,
    mode: str,
    refreshed: bool,
    apply_session_update_fn: Callable[..., None],
    append_history_fn: Callable[[PlanState, dict[str, Any]], None],
    make_history_entry_fn: Callable[..., dict[str, Any]],
    save_state_fn: Callable[[Path, PlanState], None],
    attach_agent_fallback_fn: Callable[[StepResponse, argparse.Namespace], None],
    store_raw_worker_output_fn: Callable[[Path, str, int, str], str],
    record_step_failure_fn: Callable[..., None],
    capture_git_status_snapshot_fn: Callable[[Path], tuple[dict[str, str], str | None]] = _capture_git_status_snapshot,
) -> StepResponse:
    finalize_data = read_json(plan_dir / "finalize.json")
    global_config = load_config()
    quality_config = global_config.get("quality_checks", {})
    project_dir = Path(state["config"]["project_dir"])
    tasks = finalize_data.get("tasks", [])
    all_task_ids = [
        task["id"]
        for task in tasks
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    ]
    all_sense_check_ids = [
        sense_check["id"]
        for sense_check in finalize_data.get("sense_checks", [])
        if isinstance(sense_check, dict) and isinstance(sense_check.get("id"), str)
    ]
    completed_task_ids = {
        task["id"]
        for task in tasks
        if task.get("status") in {"done", "skipped"} and isinstance(task.get("id"), str)
    }
    pending_tasks = [
        task
        for task in tasks
        if task.get("status") == "pending" and isinstance(task.get("id"), str)
    ]
    pending_batches = compute_task_batches(pending_tasks, completed_ids=completed_task_ids)
    single_batch_mode = len(pending_batches) <= 1
    global_batches = compute_global_batches(finalize_data)
    global_batch_lookup = {
        tuple(batch): index + 1
        for index, batch in enumerate(global_batches)
    }
    batches_to_run = [all_task_ids] if single_batch_mode else pending_batches
    total_batches = len(batches_to_run) or 1
    active_task_ids = set(all_task_ids if single_batch_mode else [task["id"] for task in pending_tasks])
    active_sense_check_ids = set(all_sense_check_ids if single_batch_mode else _active_sense_check_ids(finalize_data, active_task_ids))

    batch_payloads: list[dict[str, Any]] = []
    trace_chunks: list[str] = []
    total_duration_ms = 0
    total_cost_usd = 0.0
    timeout_error: CliError | None = None
    latest_session_id: str | None = None
    blocking_reasons: list[str] = []
    timeout_recovery: StepResponse | None = None

    for batch_index, batch_task_ids in enumerate(batches_to_run, start=1):
        batch_prompt = None if single_batch_mode else _execute_batch_prompt(
            state,
            plan_dir,
            batch_task_ids,
            completed_task_ids,
            root=root,
        )
        batch_number_for_artifact = 1 if single_batch_mode else global_batch_lookup.get(tuple(batch_task_ids), batch_index)
        batch_sense_check_ids = all_sense_check_ids if single_batch_mode else _active_sense_check_ids(finalize_data, set(batch_task_ids))
        batches_total_for_observation = 1 if single_batch_mode else len(global_batches)
        try:
            result = _run_and_merge_batch(
                root=root,
                plan_dir=plan_dir,
                state=state,
                args=args,
                agent=agent,
                mode=mode,
                refreshed=refreshed,
                prompt_override=batch_prompt,
                batch_task_ids=batch_task_ids,
                batch_sense_check_ids=batch_sense_check_ids,
                finalize_data=finalize_data,
                batch_number=batch_number_for_artifact,
                batches_total=batches_total_for_observation,
                quality_config=quality_config,
                capture_git_status_snapshot_fn=capture_git_status_snapshot_fn,
            )
        except CliError as error:
            if error.code == "worker_timeout":
                timeout_error = error
                latest_session_id = error.extra.get("session_id") if isinstance(error.extra.get("session_id"), str) else latest_session_id
                timeout_recovery = _recover_execute_timeout(
                    plan_dir=plan_dir,
                    state=state,
                    error=error,
                    agent=agent,
                    mode=mode,
                    refreshed=refreshed,
                    auto_approve=auto_approve,
                    args=args,
                    batch_number=None if single_batch_mode else batch_number_for_artifact,
                    apply_session_update_fn=apply_session_update_fn,
                    append_history_fn=append_history_fn,
                    make_history_entry_fn=make_history_entry_fn,
                    save_state_fn=save_state_fn,
                    attach_agent_fallback_fn=attach_agent_fallback_fn,
                    store_raw_worker_output_fn=store_raw_worker_output_fn,
                    persist_state=False,
                )
                finalize_data = read_json(plan_dir / "finalize.json")
                break
            record_step_failure_fn(plan_dir, state, step="execute", iteration=state["iteration"], error=error)
            raise

        total_duration_ms += result.worker.duration_ms
        total_cost_usd += result.worker.cost_usd
        latest_session_id = result.worker.session_id
        apply_session_update_fn(state, "execute", result.agent, result.worker.session_id, mode=result.mode, refreshed=result.refreshed)
        batch_payloads.append(result.payload)
        if result.worker.trace_output is not None:
            trace_chunks.append(result.worker.trace_output)
        completed_task_ids.update(
            task_id
            for task_id in batch_task_ids
            if task_id in {
                task["id"]
                for task in finalize_data.get("tasks", [])
                if task.get("status") in {"done", "skipped"} and isinstance(task.get("id"), str)
            }
        )
        blocking_reasons = build_blocking_reasons(
            tracked_tasks=result.merged_task_count,
            total_tasks=result.total_task_count,
            acknowledged_checks=result.acknowledged_sense_check_count,
            total_checks=result.total_sense_check_count,
            missing_task_evidence=result.missing_task_evidence,
        )
        if blocking_reasons:
            agent = result.agent
            mode = result.mode
            refreshed = result.refreshed
            break
        agent = result.agent
        mode = result.mode
        refreshed = result.refreshed

    aggregate_payload = _build_aggregate_execution_payload(
        batch_payloads,
        completed_batches=len(batch_payloads),
        total_batches=total_batches,
    )
    if timeout_error is not None:
        aggregate_payload["deviations"] = list(aggregate_payload.get("deviations", []))
        aggregate_payload["deviations"].append(
            f"Execute timed out after {len(batch_payloads)}/{total_batches} completed batches: {timeout_error.message}"
        )
    if trace_chunks:
        atomic_write_text(plan_dir / "execution_trace.jsonl", "".join(trace_chunks))

    finalize_data = read_json(plan_dir / "finalize.json")
    execution_audit = validate_execution_evidence(finalize_data, project_dir)
    deviations = list(aggregate_payload.get("deviations", []))
    if timeout_recovery is not None:
        deviations.extend(
            deviation
            for deviation in timeout_recovery.get("deviations", [])
            if deviation not in deviations
        )
    if execution_audit["skipped"]:
        deviations.append(f"Advisory audit skip: {execution_audit['reason']}")
    for finding in execution_audit["findings"]:
        deviations.append(f"Advisory audit finding: {finding}")
    aggregate_payload["deviations"] = deviations
    atomic_write_json(plan_dir / "execution.json", aggregate_payload)
    atomic_write_json(plan_dir / "execution_audit.json", execution_audit)
    atomic_write_json(plan_dir / "finalize.json", finalize_data)
    atomic_write_text(plan_dir / "final.md", render_final_md(finalize_data, phase="execute"))
    finalize_hash = sha256_file(plan_dir / "finalize.json")

    tracked_tasks, total_tasks, acknowledged_checks, total_checks = _count_execute_tracking(
        finalize_data,
        active_task_ids=active_task_ids,
        active_sense_check_ids=active_sense_check_ids,
    )
    missing_task_evidence = _check_done_task_evidence(
        finalize_data.get("tasks", []),
        issues=deviations,
        should_classify=lambda task: task.get("id") in active_task_ids,
        has_evidence=lambda task: bool(task.get("files_changed")),
        has_advisory_evidence=lambda task: bool(task.get("commands_run")),
        missing_message="Done tasks missing both files_changed and commands_run: ",
        advisory_message="Advisory: done tasks rely on commands_run without files_changed (FLAG-006 softening): ",
    )
    blocking_reasons = build_blocking_reasons(
        tracked_tasks=tracked_tasks,
        total_tasks=total_tasks,
        acknowledged_checks=acknowledged_checks,
        total_checks=total_checks,
        missing_task_evidence=missing_task_evidence,
        timeout_reason=(
            f"execution timed out after {len(batch_payloads)}/{total_batches} completed batches"
            if timeout_error is not None
            else None
        ),
    )

    blocked = bool(blocking_reasons)
    if not blocked and timeout_error is None:
        state["current_state"] = STATE_EXECUTED
    if timeout_error is not None and latest_session_id is not None:
        apply_session_update_fn(state, "execute", agent, latest_session_id, mode=mode, refreshed=refreshed)
    user_approved_gate = bool(state["meta"].get("user_approved_gate", False))
    approval_mode = _resolve_execute_approval_mode(
        auto_approve=auto_approve,
        user_approved_gate=user_approved_gate,
    )
    raw_output_file: str | None = None
    result_value = "blocked" if blocked else "success"
    message: str | None = None
    if timeout_error is not None:
        result_value = "timeout"
        raw_output = str(timeout_error.extra.get("raw_output") or timeout_error.message)
        raw_output_file = store_raw_worker_output_fn(plan_dir, "execute", state["iteration"], raw_output)
        message = timeout_error.message
    append_history_fn(
        state,
        make_history_entry_fn(
            "execute",
            duration_ms=total_duration_ms,
            cost_usd=total_cost_usd,
            result=result_value,
            agent=agent,
            mode=mode,
            worker=WorkerResult(
                payload=aggregate_payload,
                raw_output="",
                duration_ms=total_duration_ms,
                cost_usd=total_cost_usd,
                session_id=latest_session_id,
                trace_output="".join(trace_chunks) if trace_chunks else None,
            ),
            output_file="execution.json",
            artifact_hash=sha256_file(plan_dir / "execution.json"),
            finalize_hash=finalize_hash,
            raw_output_file=raw_output_file,
            message=message,
            approval_mode=approval_mode,
        ),
    )
    save_state_fn(plan_dir, state)

    artifacts = ["execution.json", "execution_audit.json", "finalize.json", "final.md"]
    if trace_chunks:
        artifacts.append("execution_trace.jsonl")
    tracking_note = _format_execute_tracking_note(
        merged_count=tracked_tasks,
        total_tasks=total_tasks,
        acknowledged_count=acknowledged_checks,
        total_checks=total_checks,
    )
    if timeout_error is not None:
        summary = (
            f"Execute timed out after {len(batch_payloads)}/{total_batches} completed batches. "
            "Prior batches were persisted; re-run execute to continue."
        )
    elif blocked:
        summary = "Blocked: " + "; ".join(blocking_reasons) + ". Re-run execute to complete tracking."
    else:
        summary = aggregate_payload["output"] + tracking_note
    response: StepResponse = {
        "success": not blocked and timeout_error is None,
        "step": "execute",
        "summary": summary,
        "artifacts": artifacts,
        "next_step": "execute" if blocked or timeout_error is not None else "review",
        "state": STATE_FINALIZED if blocked or timeout_error is not None else STATE_EXECUTED,
        "files_changed": aggregate_payload.get("files_changed", []),
        "deviations": deviations,
        "warnings": [summary] if blocked or timeout_error is not None else [],
        "auto_approve": auto_approve,
        "user_approved_gate": user_approved_gate,
    }
    attach_agent_fallback_fn(response, args)
    return response
