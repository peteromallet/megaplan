"""Execute-phase prompt builders and helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from megaplan._core import (
    batch_artifact_path,
    compute_task_batches,
    configured_robustness,
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    read_json,
)
from megaplan.types import PlanState

from ._shared import _debt_watch_lines, _render_prep_block

_EXECUTE_OUTPUT_SHAPE_EXAMPLE = textwrap.dedent(
    """
    ```json
    {
      "output": "Implemented the approved plan and captured execution evidence.",
      "files_changed": ["megaplan/handlers.py", "megaplan/evaluation.py"],
      "commands_run": ["pytest tests/test_megaplan.py -k evidence"],
      "deviations": [],
      "task_updates": [
        {
          "task_id": "T6",
          "status": "done",
          "executor_notes": "Caught the empty-strings edge case while checking execution evidence: blank `commands_run` entries still leave the task uncovered, so the missing-evidence guard behaves correctly.",
          "files_changed": ["megaplan/handlers.py"],
          "commands_run": ["pytest tests/test_megaplan.py -k execute"]
        },
        {
          "task_id": "T7",
          "status": "done",
          "executor_notes": "Confirmed the happy path still records task evidence after the prompt updates by rerunning focused tests and checking the tracked task summary stayed intact.",
          "files_changed": ["megaplan/prompts.py"],
          "commands_run": ["pytest tests/test_prompts.py -k review"]
        },
        {
          "task_id": "T8",
          "status": "done",
          "executor_notes": "Kept the rubber-stamp thresholds centralized in evaluation so sense checks and reviewer verdicts share one policy entry point while still using different strictness levels.",
          "files_changed": ["megaplan/evaluation.py"],
          "commands_run": ["pytest tests/test_evaluation.py -k rubber_stamp"]
        },
        {
          "task_id": "T11",
          "status": "skipped",
          "executor_notes": "Skipped because upstream work is not ready yet; no repo changes were made for this task.",
          "files_changed": [],
          "commands_run": []
        }
      ],
      "sense_check_acknowledgments": [
        {
          "sense_check_id": "SC6",
          "executor_note": "Confirmed execute only blocks when both files_changed and commands_run are empty for a done task."
        }
      ]
    }
    ```
    """
).strip()

_EXECUTE_REQUIREMENTS_TEMPLATE = textwrap.dedent(
    """
    Requirements:
    - Implement the intent, not just the text.
    - Adapt if repository reality contradicts the plan.
    - Report deviations explicitly.
    - Do not over-engineer beyond what the plan prescribes — no str() wraps, .get() fallbacks, or try/except guards unless the plan called for them or you found a concrete reason.
    - Do NOT fix unrelated issues you encounter (e.g., dependency compatibility, Python version workarounds). Only change files directly needed for the task. If tests need updating, only update tests that are directly related to your fix.
    - If you cannot build the project from source (e.g., C extension compilation failures), report the build failure explicitly. Do NOT fall back to testing against an installed or cached package — that tests the wrong codebase and produces false positives.
    - If you cannot verify your changes (tests missing or unrunnable), treat this as high risk — re-examine your implementation with extra scrutiny instead of accepting it on faith.
    - If tests fail, read the traceback carefully. Diagnose WHY — don't just retry. Common causes: wrong function/method used, missing import, incorrect type, edge case not handled. Fix the root cause, then re-run.
    - When verifying changes, run the entire test file or module (e.g., `pytest tests/test_foo.py`), not individual test functions. Individual tests miss regressions in the same module.
    - finalize.json includes baseline_test_failures — a list of test IDs that were already failing before your changes. If a test fails and its ID appears in baseline_test_failures, it is pre-existing — do not scope-creep into fixing it. If baseline_test_failures is null, the baseline could not be captured; use your judgment but err on the side of assuming failures are regressions. You MUST still re-run the FULL test suite with your changes applied — pre-existing failures do not excuse skipping verification. Never narrow to individual test functions and stop.
    - Before declaring the work complete, write a short script (not a full test) that reproduces the exact bug or incorrect behavior described in the task. Run it to confirm the fix resolves the issue. Then delete the script so it does not appear in the final diff. If the task description is too vague to write a concrete reproduction, note this explicitly in executor_notes.
    - Output concrete files changed and commands run. `files_changed` means files you WROTE or MODIFIED — not files you read or verified. Only list files where you made actual edits.
    - Use the tasks in `finalize.json` as the execution boundary.
    - Best-effort progress checkpointing: if `{checkpoint_path}` is writable, then after each completed task read the full file, update that task's `status`, `executor_notes`, `files_changed`, and `commands_run`, and write the full file back. Do NOT write to `finalize.json` directly — the harness owns that file.
    - Best-effort sense-check checkpointing: if `{checkpoint_path}` is writable, then after each sense check acknowledgment read the full file again, update that sense check's `executor_note`, and write the full file back.
    - Always use full read-modify-write updates for `{checkpoint_path}` instead of partial edits. If the sandbox blocks writes, continue execution and rely on the structured output below.
    - Structured output remains the authoritative final summary for this step. Disk writes are progress checkpoints for timeout recovery only.
    - Return `task_updates` with one object per completed or skipped task.
    - `task_updates[].status` must be either `done` or `skipped`. Never return `pending` in execute output.
    - If a task is blocked by environment limits, missing devices, or manual-only validation that cannot happen in this session, return `status: "skipped"` and explain the remaining manual follow-up in `executor_notes` and `deviations`.
    - Return `sense_check_acknowledgments` with one object per sense check.
    - Keep `executor_notes` verification-focused: explain why your changes are correct. The diff already shows what changed; notes should cover edge cases caught, expected behaviors confirmed, or design choices made.
    - Follow this JSON shape exactly:
    {output_shape}
    """
).strip()


def _execute_review_block(plan_dir: Path) -> str:
    review_path = plan_dir / "review.json"
    if not review_path.exists():
        return "No prior `review.json` exists. Treat this as the first execution pass."
    return textwrap.dedent(
        f"""
        Previous review findings to address on this execution pass (`review.json`):
        {json_dump(read_json(review_path)).strip()}
        """
    ).strip()


def _execute_nudges(
    finalize_data: dict[str, Any], plan_dir: Path, root: Path | None
) -> str:
    nudge_lines: list[str] = []
    sense_checks = finalize_data.get("sense_checks", [])
    if sense_checks:
        nudge_lines.append(
            "Sense checks to keep in mind during execution (reviewer will verify these):"
        )
        for sense_check in sense_checks:
            nudge_lines.append(
                f"- {sense_check['id']} ({sense_check['task_id']}): {sense_check['question']}"
            )
    watch_items = finalize_data.get("watch_items", [])
    if watch_items:
        nudge_lines.append("Watch items to keep visible during execution:")
        for item in watch_items:
            nudge_lines.append(f"- {item}")
    debt_watch_items = _debt_watch_lines(plan_dir, root)
    if debt_watch_items:
        nudge_lines.append("Debt watch items (do not make these worse):")
        for item in debt_watch_items:
            nudge_lines.append(f"- {item}")
    return "\n".join(nudge_lines)


def _execute_rerun_guidance(plan_dir: Path, finalize_data: dict[str, Any]) -> str:
    tasks = finalize_data.get("tasks", [])
    done_tasks = [task for task in tasks if task.get("status") in ("done", "skipped")]
    pending_tasks = [task for task in tasks if task.get("status") == "pending"]
    if done_tasks and pending_tasks:
        done_ids = ", ".join(task["id"] for task in done_tasks)
        pending_ids = ", ".join(task["id"] for task in pending_tasks)
        return (
            f"Re-execution: {len(done_tasks)} tasks already tracked ({done_ids}). "
            f"Focus on the {len(pending_tasks)} remaining tasks ({pending_ids}). "
            "You must still return task_updates for ALL tasks (including already-tracked ones) — "
            "for previously done tasks, preserve their existing status and notes."
        )
    if done_tasks and not pending_tasks:
        review_data = (
            read_json(plan_dir / "review.json")
            if (plan_dir / "review.json").exists()
            else {}
        )
        rework_items = review_data.get("rework_items", [])
        if rework_items:
            rework_lines = []
            for item in rework_items:
                if not isinstance(item, dict):
                    continue
                task_id = item.get("task_id", "?")
                issue = item.get("issue", "")
                expected = item.get("expected", "")
                actual = item.get("actual", "")
                evidence = item.get("evidence_file", "")
                entry = f"  - [{task_id}] {issue}"
                if expected:
                    entry += f"\n    expected: {expected}"
                if actual:
                    entry += f"\n    actual: {actual}"
                if evidence:
                    entry += f"\n    evidence: {evidence}"
                rework_lines.append(entry)
            issue_list = "\n".join(rework_lines)
        else:
            review_issues = review_data.get("issues", [])
            issue_list = (
                "\n".join(f"  - {issue}" for issue in review_issues)
                if review_issues
                else "  (see review.json above for details)"
            )
        return (
            "REWORK REQUIRED: all tasks are already tracked but the reviewer kicked this back.\n"
            f"Review issues to fix:\n{issue_list}\n\n"
            "You MUST make code changes to address each issue — do not return success without modifying files. "
            "For each issue, either fix it and list the file in files_changed, or explain in deviations why no change is needed with line-level evidence. "
            "Return task_updates for all tasks with updated evidence."
        )
    return ""


def _execute_approval_note(state: PlanState) -> str:
    if state["config"].get("auto_approve"):
        return (
            "Note: User chose auto-approve mode. This execution was not manually "
            "reviewed at the gate. Exercise extra caution on destructive operations."
        )
    if state["meta"].get("user_approved_gate"):
        return "Note: User explicitly approved this plan at the gate checkpoint."
    return "Note: Review mode is enabled. Execute should only be running after explicit gate approval."


def _execute_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
    finalize_data = read_json(plan_dir / "finalize.json")
    checkpoint_path = str(plan_dir / "execution_checkpoint.json")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    gate = read_json(plan_dir / "gate.json")
    robustness = configured_robustness(state)
    prior_review_block = _execute_review_block(plan_dir)
    rerun_guidance = _execute_rerun_guidance(plan_dir, finalize_data)
    approval_note = _execute_approval_note(state)
    execution_nudges = _execute_nudges(finalize_data, plan_dir, root)
    requirements_block = _EXECUTE_REQUIREMENTS_TEMPLATE.format(
        checkpoint_path=checkpoint_path,
        output_shape=_EXECUTE_OUTPUT_SHAPE_EXAMPLE,
    )
    return textwrap.dedent(
        f"""
        Execute the approved plan in the repository.

        Project directory:
        {project_dir}

        {prep_block}

        {prep_instruction}

        {intent_and_notes_block(state)}

        Execution tracking source of truth (`finalize.json`):
        {json_dump(finalize_data).strip()}

        Absolute checkpoint path for best-effort progress checkpoints (NOT `finalize.json`):
        {checkpoint_path}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        {prior_review_block}

        {rerun_guidance}

        {approval_note}
        Robustness level: {robustness}.

        {requirements_block}

        {execution_nudges}
        """
    ).strip()


def _execute_batch_prompt(
    state: PlanState,
    plan_dir: Path,
    batch_task_ids: list[str],
    completed_task_ids: set[str] | None = None,
    root: Path | None = None,
) -> str:
    completed = set(completed_task_ids or set())
    finalize_data = read_json(plan_dir / "finalize.json")
    all_tasks = finalize_data.get("tasks", [])
    tasks_by_id = {
        task["id"]: task
        for task in all_tasks
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    }
    batch_tasks = [
        tasks_by_id[task_id] for task_id in batch_task_ids if task_id in tasks_by_id
    ]
    completed_tasks = [
        task
        for task_id, task in tasks_by_id.items()
        if task_id in completed and task_id not in set(batch_task_ids)
    ]
    batch_sense_checks = [
        sense_check
        for sense_check in finalize_data.get("sense_checks", [])
        if sense_check.get("task_id") in set(batch_task_ids)
    ]
    batch_sense_check_ids = [
        sense_check["id"]
        for sense_check in batch_sense_checks
        if isinstance(sense_check.get("id"), str)
    ]
    global_batches = compute_task_batches(all_tasks)
    batch_number = next(
        (
            index + 1
            for index, batch in enumerate(global_batches)
            if batch == batch_task_ids
        ),
        1,
    )
    batch_total = len(global_batches) or 1
    checkpoint_path = str(batch_artifact_path(plan_dir, batch_number))
    prior_batch_deviations = "None"
    if batch_number > 1:
        prior_batch_artifact = batch_artifact_path(plan_dir, batch_number - 1)
        if prior_batch_artifact.exists():
            try:
                prior_batch_payload = read_json(prior_batch_artifact)
            except (OSError, ValueError):
                prior_batch_payload = {}
            raw_deviations = prior_batch_payload.get("deviations", [])
            if isinstance(raw_deviations, list):
                deviations = [item for item in raw_deviations if isinstance(item, str)]
                if deviations:
                    prior_batch_deviations = json_dump(deviations).strip()
    approval_note = (
        "Note: User chose auto-approve mode. This execution was not manually reviewed at the gate. Exercise extra caution on destructive operations."
        if state["config"].get("auto_approve")
        else (
            "Note: User explicitly approved this plan at the gate checkpoint."
            if state["meta"].get("user_approved_gate")
            else "Note: Review mode is enabled. Execute should only be running after explicit gate approval."
        )
    )
    debt_watch_items = _debt_watch_lines(plan_dir, root)
    debt_watch_block = (
        "\n".join(
            [
                "Debt watch items (do not make these worse):",
                *[f"- {item}" for item in debt_watch_items],
            ]
        )
        if debt_watch_items
        else "Debt watch items (do not make these worse):\n- None."
    )
    return textwrap.dedent(
        f"""
        Execute the approved plan in the repository.

        Project directory:
        {Path(state["config"]["project_dir"])}

        {intent_and_notes_block(state)}

        Batch framing:
        - Execute batch {batch_number} of {batch_total}.
        - Actionable task IDs for this batch: {batch_task_ids}
        - Already completed task IDs available as dependency context: {sorted(completed)}

        Actionable tasks for this batch:
        {json_dump(batch_tasks).strip()}

        Completed task context (already satisfied, do not re-execute unless directly required by current edits):
        {json_dump(completed_tasks).strip()}

        Prior batch deviations (address if applicable):
        {prior_batch_deviations}

        Batch-scoped sense checks:
        {json_dump(batch_sense_checks).strip()}

        Full execution tracking source of truth (`finalize.json`):
        {json_dump(finalize_data).strip()}

        {debt_watch_block}

        {approval_note}
        Robustness level: {configured_robustness(state)}.

        Requirements:
        - Execute only the actionable tasks in this batch.
        - Treat completed tasks as dependency context, not new work.
        - Return structured JSON only.
        - Only produce `task_updates` for these tasks: [{", ".join(batch_task_ids)}]
        - Only produce `sense_check_acknowledgments` for these sense checks: [{", ".join(batch_sense_check_ids)}]
        - Do not include updates for tasks or sense checks outside this batch.
        - Keep `executor_notes` verification-focused.
        - Best-effort progress checkpointing: if `{checkpoint_path}` is writable, checkpoint task and sense-check updates there (not `finalize.json`). The harness owns `finalize.json`.
        - When verifying changes, run the entire test file or module, not individual test functions. Individual tests miss regressions.
        - finalize.json includes baseline_test_failures — a list of test IDs that were already failing before your changes. If a test fails and its ID appears in baseline_test_failures, it is pre-existing — do not scope-creep into fixing it. If baseline_test_failures is null, the baseline could not be captured; use your judgment but err on the side of assuming failures are regressions. You MUST still re-run the FULL test suite with your changes applied — pre-existing failures do not excuse skipping verification. Never narrow to individual test functions and stop.
        - If this batch includes the final verification task, write a short script that reproduces the exact bug described in the task, run it to confirm the fix resolves it, then delete the script.
        """
    ).strip()
