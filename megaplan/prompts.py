"""Prompt builders for each megaplan step and dispatch tables."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

from megaplan.types import (
    CliError,
    PlanState,
)
from megaplan._core import (
    collect_git_diff_summary,
    configured_robustness,
    current_iteration_artifact,
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    latest_plan_path,
    load_flag_registry,
    read_json,
    robustness_critique_instruction,
    unresolved_significant_flags,
)
from megaplan.types import FlagRegistry


PLAN_TEMPLATE = textwrap.dedent(
    """
    Plan template — simple format (adapt to the actual repo and scope):
    ````md
    # Implementation Plan: [Title]

    ## Overview
    Summarize the goal, current repository shape, and the constraints that matter.

    ## Main Phase

    ### Step 1: Audit the current behavior (`megaplan/prompts.py`)
    **Scope:** Small
    1. **Inspect** the current implementation and call out the exact insertion points (`megaplan/prompts.py:29`).

    ### Step 2: Add the first change (`megaplan/evaluation.py`)
    **Scope:** Medium
    1. **Implement** the smallest viable change with exact file references (`megaplan/evaluation.py:1`).
    2. **Capture** any tricky behavior with a short example.
       ```python
       issues = validate_plan_structure(plan_text)
       ```

    ### Step 3: Wire downstream behavior (`megaplan/handlers.py`, `megaplan/workers.py`)
    **Scope:** Medium
    1. **Update** the runtime flow in the touched files (`megaplan/handlers.py:400`, `megaplan/workers.py:199`).

    ### Step 4: Prove the change (`tests/test_evaluation.py`, `tests/test_megaplan.py`)
    **Scope:** Small
    1. **Run** the cheapest targeted checks first (`tests/test_evaluation.py:1`).
    2. **Finish** with broader verification once the wiring is in place (`tests/test_megaplan.py:1`).

    ## Execution Order
    1. Update prompts and mocks before enforcing stricter validation.
    2. Land higher-risk wiring after the validator and tests are ready.

    ## Validation Order
    1. Start with focused unit tests.
    2. Run the broader suite after the flow changes are in place.
    ````

    For complex plans, use multiple phases:
    ````md
    ## Phase 1: Foundation — Dependencies, DB, Types

    ### Step 1: Install dependencies (`package.json`)
    ...

    ### Step 2: Create database migration (`supabase/migrations/`)
    ...

    ## Phase 2: Core Integration

    ### Step 3: Port the main component (`src/components/`)
    ...
    ````

    Template guidance:
    - Simple plans: use `## Main Phase` with `### Step N:` sections underneath.
    - Complex plans: use multiple `## Phase N:` sections, each containing `### Step N:` steps. Step numbers are global (not per-phase).
    - The flat `## Step N:` format (without phases) also works for backwards compatibility.
    - Key invariants: one H1 title, one `## Overview`, numbered step sections (`### Step N:` or `## Step N:`), and at least one ordering section.
    """
).strip()


def _plan_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    clarification = state.get("clarification", {})
    if clarification:
        clarification_block = textwrap.dedent(
            f"""
            Existing clarification context:
            {json_dump(clarification).strip()}
            """
        ).strip()
    else:
        clarification_block = "No prior clarification artifact exists. Identify ambiguities, ask clarifying questions, and state your assumptions inside the plan output."
    return textwrap.dedent(
        f"""
        You are creating an implementation plan for the following idea.

        {intent_and_notes_block(state)}

        Project directory:
        {project_dir}

        {clarification_block}

        Requirements:
        - Inspect the actual repository before planning.
        - Produce a concrete implementation plan in markdown.
        - Define observable success criteria.
        - Use the `questions` field for ambiguities that would materially change implementation.
        - Use the `assumptions` field for defaults you are making so planning can proceed now.
        - Prefer cheap validation steps early.
        - If user notes answer earlier questions, incorporate them into the draft plan instead of re-asking them.

        {PLAN_TEMPLATE}
        """
    ).strip()


def _plan_light_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    clarification = state.get("clarification", {})
    if clarification:
        clarification_block = textwrap.dedent(
            f"""
            Existing clarification context:
            {json_dump(clarification).strip()}
            """
        ).strip()
    else:
        clarification_block = "No prior clarification artifact exists. Identify ambiguities, ask clarifying questions, and state your assumptions inside the plan output."
    return textwrap.dedent(
        f"""
        You are creating a light-robustness implementation plan for the following idea.

        {intent_and_notes_block(state)}

        Project directory:
        {project_dir}

        {clarification_block}

        Requirements:
        - Inspect the actual repository before planning.
        - Produce structured JSON only.
        - `plan` must be concrete markdown using the plan template below.
        - `questions`, `success_criteria`, and `assumptions` follow the normal planning rules.
        - Add `self_flags`: concrete concerns you can already see in your own plan. Use the same shape as critique flags (`id`, `concern`, `category`, `severity_hint`, `evidence`).
        - Add `gate_recommendation`: exactly one of PROCEED, ITERATE, ESCALATE.
        - Add `gate_rationale`: a compact explanation of that recommendation.
        - Add `settled_decisions`: design choices that are now settled and should carry into review without being re-litigated. Return `[]` when none.
        - Preserve quality while staying pragmatic: combine planning, self-critique, and the gate recommendation in one pass.
        - Prefer cheap validation steps early.
        - If user notes answer earlier questions, incorporate them into the draft plan instead of re-asking them.

        Example output shape:
        ```json
        {{
          "plan": "# Implementation Plan: ...",
          "questions": [],
          "success_criteria": ["..."],
          "assumptions": ["..."],
          "self_flags": [
            {{
              "id": "FLAG-001",
              "concern": "The plan still relies on an implied helper that does not exist yet.",
              "category": "correctness",
              "severity_hint": "likely-significant",
              "evidence": "No existing helper in the referenced module handles this workflow."
            }}
          ],
          "gate_recommendation": "PROCEED",
          "gate_rationale": "The plan is specific enough to execute and any remaining concerns are minor.",
          "settled_decisions": [
            {{
              "id": "DECISION-001",
              "decision": "Keep light robustness to a single plan call that also produces self-critique and gate guidance.",
              "rationale": "The user explicitly wants to collapse the plan, critique, and gate loop into one call."
            }}
          ]
        }}
        ```

        {PLAN_TEMPLATE}
        """
    ).strip()


def _revise_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    gate = read_json(plan_dir / "gate.json")
    unresolved = unresolved_significant_flags(load_flag_registry(plan_dir))
    open_flags = [
        {
            "id": flag["id"],
            "severity": flag.get("severity"),
            "status": flag["status"],
            "concern": flag["concern"],
            "evidence": flag.get("evidence"),
        }
        for flag in unresolved
    ]
    return textwrap.dedent(
        f"""
        You are revising an implementation plan after critique and gate feedback.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Current plan (markdown):
        {latest_plan}

        Current plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        Open significant flags:
        {json_dump(open_flags).strip()}

        Requirements:
        - Update the plan to address the significant issues.
        - Keep the plan readable and executable.
        - Return flags_addressed with the exact flag IDs you addressed.
        - Preserve or improve success criteria quality.
        - Verify that the plan remains aligned with the user's original intent, not just internal plan quality.
        - Remove unjustified scope growth. If critique raised scope creep, narrow the plan back to the original idea unless the broader work is strictly required.
        - Maintain the structural template: H1 title, ## Overview, phase sections with numbered step sections, ## Execution Order or ## Validation Order.

        {PLAN_TEMPLATE}
        """
    ).strip()


def _critique_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    structure_warnings = latest_meta.get("structure_warnings", [])
    flag_registry = load_flag_registry(plan_dir)
    robustness = configured_robustness(state)
    unresolved = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "status": flag["status"],
            "severity": flag.get("severity"),
        }
        for flag in flag_registry["flags"]
        if flag["status"] in {"addressed", "open", "disputed"}
    ]
    return textwrap.dedent(
        f"""
        You are an independent reviewer. Critique the plan against the actual repository.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Plan structure warnings from validator:
        {json_dump(structure_warnings).strip()}

        Existing flags:
        {json_dump(unresolved).strip()}

        Requirements:
        - Consider whether the plan is at the right level of abstraction.
        - Reuse existing flag IDs when the same concern is still open.
        - `verified_flag_ids` should list previously addressed flags that now appear resolved.
        - Focus on concrete issues that would cause real problems.
        - Robustness level: {robustness}. {robustness_critique_instruction(robustness)}
        - Verify that the plan remains aligned with the user's original intent.
        - Verify that the plan follows the expected structure: one H1 title, `## Overview`, numbered step sections (`### Step N:` under `## Phase` headers, or flat `## Step N:`) with file references and numbered substeps, plus `## Execution Order` or `## Validation Order`. Missing structure should be flagged as category `completeness` with severity_hint `likely-significant`.
        - Flag scope creep explicitly when the plan grows beyond the original idea or recorded user notes. Use the phrase "Scope creep:" in the concern.
        - Assign severity_hint carefully. Implementation details the executor will naturally resolve should usually be `likely-minor`.
        """
    ).strip()


def _gate_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    gate_signals = read_json(current_iteration_artifact(plan_dir, "gate_signals", state["iteration"]))
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    open_flags = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "category": flag["category"],
            "severity": flag.get("severity", "unknown"),
            "status": flag["status"],
            "weight": flag.get("weight"),
        }
        for flag in unresolved
    ]
    robustness = configured_robustness(state)
    return textwrap.dedent(
        f"""
        You are the gatekeeper for the megaplan workflow. Make the continuation decision directly.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate signals:
        {json_dump(gate_signals).strip()}

        Unresolved significant flags:
        {json_dump(open_flags).strip()}

        Robustness level:
        {robustness}

        Requirements:
        - Decide exactly one of: PROCEED, ITERATE, ESCALATE.
        - Use the weighted score, flag details, plan delta, recurring critiques, loop summary, and preflight results as judgment context, not as a fixed decision table.
        - PROCEED when execution should move forward now.
        - ITERATE when revising the plan is the best next move.
        - ESCALATE when the loop is stuck, churn is recurring, or user intervention is needed.
        - `signals_assessment` should summarize the score trajectory, plan delta, recurring critiques, unresolved flag weight, and preflight posture in one compact paragraph.
        - Put any cautionary notes in `warnings`.
        - Populate `settled_decisions` with design choices that are now settled and should carry into review without being re-litigated. Return `[]` when there are no such decisions.
        - Example output shape:
        ```json
        {{
          "recommendation": "PROCEED",
          "rationale": "The remaining issues are executor-level details rather than planning blockers.",
          "signals_assessment": "Weighted score is falling, plan delta is stabilizing, and preflight remains clean.",
          "warnings": ["Double-check FLAG-005 while executing."],
          "settled_decisions": [
            {{
              "id": "DECISION-001",
              "decision": "Treat FLAG-006 softening as approved gate guidance during review.",
              "rationale": "The gate already accepted this tradeoff and review should verify compliance, not reopen it."
            }}
          ]
        }}
        ```
        """
    ).strip()


def _collect_critique_summaries(plan_dir: Path, iteration: int) -> list[dict[str, object]]:
    """Gather a compact list of all critique rounds for the finalize prompt."""
    summaries: list[dict[str, object]] = []
    for i in range(1, iteration + 1):
        path = plan_dir / f"critique_v{i}.json"
        if path.exists():
            data = read_json(path)
            summaries.append({
                "iteration": i,
                "flag_count": len(data.get("flags", [])),
                "verified": data.get("verified_flag_ids", []),
            })
    return summaries


def _flag_summary(registry: FlagRegistry) -> list[dict[str, object]]:
    """Compact flag list for the finalize prompt."""
    return [
        {
            "id": f["id"],
            "concern": f["concern"],
            "status": f["status"],
            "severity": f.get("severity", "unknown"),
        }
        for f in registry["flags"]
    ]


def _finalize_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    gate = read_json(plan_dir / "gate.json")
    flag_registry = load_flag_registry(plan_dir)
    critique_history = _collect_critique_summaries(plan_dir, state["iteration"])
    return textwrap.dedent(
        f"""
        You are preparing an execution-ready briefing document from the approved plan.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        Flag registry:
        {json_dump(_flag_summary(flag_registry)).strip()}

        Critique history:
        {json_dump(critique_history).strip()}

        Requirements:
        - Produce structured JSON only.
        - `tasks` must be an ordered array of task objects. Every task object must include:
          - `id`: short stable task ID like `T1`
          - `description`: concrete work item
          - `depends_on`: array of earlier task IDs or `[]`
          - `status`: always `"pending"` at finalize time
          - `executor_notes`: always `""` at finalize time
          - `reviewer_verdict`: always `""` at finalize time
        - `watch_items` must be an array of strings covering runtime risks, critique concerns, and assumptions to keep visible during execution.
        - `sense_checks` must be an array with one verification question per task. Every sense-check object must include:
          - `id`: short stable ID like `SC1`
          - `task_id`: the related task ID
          - `question`: reviewer verification question
          - `verdict`: always `""` at finalize time
        - `meta_commentary` must be a single string with execution guidance, gotchas, or judgment calls that help the executor succeed.
        - Preserve information that strong existing artifacts already capture well: execution ordering, watch-outs, reviewer checkpoints, and practical context.
        - The structured output should be self-contained: an executor reading only `finalize.json` should have everything needed to work.
        """
    ).strip()


def _execute_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    # Codex execute often cannot write back into plan_dir during --full-auto, so
    # checkpoint instructions must stay best-effort rather than mandatory.
    finalize_path = str(plan_dir / "finalize.json")
    finalize_data = read_json(plan_dir / "finalize.json")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    robustness = configured_robustness(state)
    gate = read_json(plan_dir / "gate.json")
    review_path = plan_dir / "review.json"
    if review_path.exists():
        prior_review_block = textwrap.dedent(
            f"""
            Previous review findings to address on this execution pass (`review.json`):
            {json_dump(read_json(review_path)).strip()}
            """
        ).strip()
    else:
        prior_review_block = "No prior `review.json` exists. Treat this as the first execution pass."
    nudge_lines: list[str] = []
    sense_checks = finalize_data.get("sense_checks", [])
    if sense_checks:
        nudge_lines.append("Sense checks to keep in mind during execution (reviewer will verify these):")
        for sense_check in sense_checks:
            nudge_lines.append(f"- {sense_check['id']} ({sense_check['task_id']}): {sense_check['question']}")
    watch_items = finalize_data.get("watch_items", [])
    if watch_items:
        nudge_lines.append("Watch items to keep visible during execution:")
        for item in watch_items:
            nudge_lines.append(f"- {item}")
    execution_nudges = "\n".join(nudge_lines)
    tasks = finalize_data.get("tasks", [])
    done_tasks = [t for t in tasks if t.get("status") in ("done", "skipped")]
    pending_tasks = [t for t in tasks if t.get("status") == "pending"]
    if done_tasks and pending_tasks:
        done_ids = ", ".join(t["id"] for t in done_tasks)
        pending_ids = ", ".join(t["id"] for t in pending_tasks)
        rerun_guidance = (
            f"Re-execution: {len(done_tasks)} tasks already tracked ({done_ids}). "
            f"Focus on the {len(pending_tasks)} remaining tasks ({pending_ids}). "
            "You must still return task_updates for ALL tasks (including already-tracked ones) — "
            "for previously done tasks, preserve their existing status and notes."
        )
    elif done_tasks and not pending_tasks:
        rerun_guidance = (
            "Re-execution: all tasks are already tracked but execution was blocked or kicked back. "
            "Check the review findings above and address the specific issues raised. "
            "Return task_updates for all tasks with updated evidence where needed."
        )
    else:
        rerun_guidance = ""
    if state["config"].get("auto_approve"):
        approval_note = (
            "Note: User chose auto-approve mode. This execution was not manually "
            "reviewed at the gate. Exercise extra caution on destructive operations."
        )
    elif state["meta"].get("user_approved_gate"):
        approval_note = "Note: User explicitly approved this plan at the gate checkpoint."
    else:
        approval_note = "Note: Review mode is enabled. Execute should only be running after explicit gate approval."
    return textwrap.dedent(
        f"""
        Execute the approved plan in the repository.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Execution tracking source of truth (`finalize.json`):
        {json_dump(finalize_data).strip()}

        Absolute `finalize.json` path for best-effort progress checkpoints:
        {finalize_path}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        {prior_review_block}

        {rerun_guidance}

        {approval_note}
        Robustness level: {robustness}.

        Requirements:
        - Implement the intent, not just the text.
        - Adapt if repository reality contradicts the plan.
        - Report deviations explicitly.
        - Output concrete files changed and commands run.
        - Use the tasks in `finalize.json` as the execution boundary.
        - Best-effort progress checkpointing: if `{finalize_path}` is writable, then after each completed task read the full file, update that task's `status`, `executor_notes`, `files_changed`, and `commands_run`, and write the full file back.
        - Best-effort sense-check checkpointing: if `{finalize_path}` is writable, then after each sense check acknowledgment read the full file again, update that sense check's `executor_note`, and write the full file back.
        - Always use full read-modify-write updates for `{finalize_path}` instead of partial edits. If the sandbox blocks writes, continue execution and rely on the structured output below.
        - Structured output remains the authoritative final summary for this step. Disk writes are progress checkpoints for timeout recovery only.
        - Return `task_updates` with one object per completed or skipped task.
        - Return `sense_check_acknowledgments` with one object per sense check.
        - Keep `executor_notes` verification-focused: say what you verified was correct, what edge cases you considered, and what behavior you observed. The diff already shows what changed; the notes should explain why it is correct.
        - Follow this JSON shape exactly:
        ```json
        {{
          "output": "Implemented the approved plan and captured execution evidence.",
          "files_changed": ["megaplan/handlers.py", "megaplan/evaluation.py"],
          "commands_run": ["pytest tests/test_megaplan.py -k evidence"],
          "deviations": [],
          "task_updates": [
            {{
              "task_id": "T6",
              "status": "done",
              "executor_notes": "Verified that handle_execute now merges per-task evidence before blocking on missing execution proof. Tested that a done task with commands_run but no files_changed follows the FLAG-006 softening path. Edge case: empty strings in commands_run still count as missing evidence.",
              "files_changed": ["megaplan/handlers.py"],
              "commands_run": ["pytest tests/test_megaplan.py -k execute"]
            }},
            {{
              "task_id": "T11",
              "status": "skipped",
              "executor_notes": "Skipped because upstream work is not ready yet; no repo changes were made for this task.",
              "files_changed": [],
              "commands_run": []
            }}
          ],
          "sense_check_acknowledgments": [
            {{
              "sense_check_id": "SC6",
              "executor_note": "Confirmed execute only blocks when both files_changed and commands_run are empty for a done task."
            }}
          ]
        }}
        ```

        {execution_nudges}
        """
    ).strip()


def _settled_decisions_block(gate: dict[str, object]) -> str:
    settled_decisions = gate.get("settled_decisions", [])
    if not isinstance(settled_decisions, list) or not settled_decisions:
        return ""
    lines = ["Settled decisions (from gate approval - do not re-litigate these):"]
    for item in settled_decisions:
        if not isinstance(item, dict):
            continue
        decision_id = item.get("id", "DECISION")
        decision = item.get("decision", "")
        rationale = item.get("rationale", "")
        line = f"- {decision_id}: {decision}"
        if rationale:
            line += f" ({rationale})"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def _settled_decisions_instruction(gate: dict[str, object]) -> str:
    settled_decisions = gate.get("settled_decisions", [])
    if not isinstance(settled_decisions, list) or not settled_decisions:
        return ""
    return "- The decisions listed above were settled at the gate stage. Verify that execution honors these decisions, but do not question the decisions themselves."


def _review_robustness_instruction(robustness: str) -> list[str]:
    if robustness == "light":
        return [
            "Light robustness: trust executor evidence by default, focus on success criteria pass/fail, and do not require deep cross-referencing unless the diff or audit contradicts the claim.",
        ]
    if robustness == "thorough":
        return [
            "Thorough robustness: cross-reference every claimed file against the diff line by line before accepting the claim.",
            "Thorough robustness: verify each sense-check acknowledgment against actual code behavior, not just the prose.",
            "Thorough robustness: flag any executor note that merely describes the edit instead of explaining verification evidence.",
        ]
    return [
        "Trust executor evidence by default. Dig deeper only where the git diff, `execution_audit.json`, or vague notes make the claim ambiguous.",
    ]


def _review_claude_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    gate = read_json(plan_dir / "gate.json")
    finalize_data = read_json(plan_dir / "finalize.json")
    robustness = configured_robustness(state)
    settled_decisions_block = _settled_decisions_block(gate)
    settled_decisions_instruction = _settled_decisions_instruction(gate)
    robustness_lines = "\n".join(f"- {line}" for line in _review_robustness_instruction(robustness))
    diff_summary = collect_git_diff_summary(project_dir)
    audit_path = plan_dir / "execution_audit.json"
    if audit_path.exists():
        audit_block = textwrap.dedent(
            f"""
            Execution audit (`execution_audit.json`):
            {json_dump(read_json(audit_path)).strip()}
            """
        ).strip()
    else:
        audit_block = "Execution audit (`execution_audit.json`): not present. Skip that artifact gracefully and rely on `finalize.json`, `execution.json`, and the git diff."
    return textwrap.dedent(
        f"""
        Review the execution critically against user intent and observable success criteria.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Execution tracking state (`finalize.json`):
        {json_dump(finalize_data).strip()}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        {settled_decisions_block}

        Execution summary:
        {json_dump(execution).strip()}

        {audit_block}

        Git diff summary:
        {diff_summary}

        Requirements:
        - Judge against the success criteria, not plan elegance.
        - Be critical and call out real misses.
        {robustness_lines}
        {settled_decisions_instruction}
        - If actual implementation work is incomplete, set top-level `review_verdict` to `needs_rework` so the plan routes back to execute. Use `approved` only when the work itself is acceptable.
        - Review each task by cross-referencing the executor's per-task `files_changed` and `commands_run` against the git diff and any audit findings.
        - Review every sense check explicitly. Confirm concise executor acknowledgments when they are specific; dig deeper only when they are perfunctory or contradicted by the code.
        - Follow this JSON shape exactly:
        ```json
        {{
          "review_verdict": "approved",
          "criteria": [
            {{
              "name": "Execution evidence is auditable.",
              "pass": true,
              "evidence": "Per-task evidence in finalize.json matches the git diff and execution_audit.json reported no phantom claims."
            }}
          ],
          "issues": [],
          "summary": "Approved. Executor evidence lines up with the diff; only routine advisory findings remain.",
          "task_verdicts": [
            {{
              "task_id": "T6",
              "reviewer_verdict": "Pass. Claimed handler changes and command evidence match the repo state.",
              "evidence_files": ["megaplan/handlers.py", "megaplan/evaluation.py"]
            }}
          ],
          "sense_check_verdicts": [
            {{
              "sense_check_id": "SC6",
              "verdict": "Confirmed. The execute blocker only fires when both evidence arrays are empty."
            }}
          ]
        }}
        ```
        - When the work needs another execute pass, keep the same shape and change only `review_verdict` to `needs_rework`; make `issues`, `summary`, and task verdicts specific enough for the executor to act on directly.
        """
    ).strip()


def _review_codex_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    gate = read_json(plan_dir / "gate.json")
    finalize_data = read_json(plan_dir / "finalize.json")
    robustness = configured_robustness(state)
    settled_decisions_block = _settled_decisions_block(gate)
    settled_decisions_instruction = _settled_decisions_instruction(gate)
    robustness_lines = "\n".join(f"- {line}" for line in _review_robustness_instruction(robustness))
    diff_summary = collect_git_diff_summary(project_dir)
    audit_path = plan_dir / "execution_audit.json"
    if audit_path.exists():
        audit_block = textwrap.dedent(
            f"""
            Execution audit (`execution_audit.json`):
            {json_dump(read_json(audit_path)).strip()}
            """
        ).strip()
    else:
        audit_block = "Execution audit (`execution_audit.json`): not present. Skip that artifact gracefully and rely on `finalize.json`, `execution.json`, and the git diff."
    return textwrap.dedent(
        f"""
        Review the implementation against the success criteria.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Execution tracking state (`finalize.json`):
        {json_dump(finalize_data).strip()}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        {settled_decisions_block}

        Execution summary:
        {json_dump(execution).strip()}

        {audit_block}

        Git diff summary:
        {diff_summary}

        Requirements:
        - Be critical.
        - Verify each success criterion explicitly.
        {robustness_lines}
        {settled_decisions_instruction}
        - If actual implementation work is incomplete, set top-level `review_verdict` to `needs_rework` so the plan routes back to execute. Use `approved` only when the work itself checks out.
        - Cross-reference each task's `files_changed` and `commands_run` against the git diff and any audit findings.
        - Review every `sense_check` explicitly and treat perfunctory acknowledgments as a reason to dig deeper.
        - Follow this JSON shape exactly:
        ```json
        {{
          "review_verdict": "approved",
          "criteria": [
            {{
              "name": "Review cross-check completed",
              "pass": true,
              "evidence": "Executor evidence in finalize.json matches the diff and the audit file."
            }}
          ],
          "issues": [],
          "summary": "Approved. The executor evidence is consistent and the remaining findings are advisory only.",
          "task_verdicts": [
            {{
              "task_id": "T3",
              "reviewer_verdict": "Pass. Review prompt changes match the diff and reference the audit fallback correctly.",
              "evidence_files": ["megaplan/prompts.py"]
            }}
          ],
          "sense_check_verdicts": [
            {{
              "sense_check_id": "SC3",
              "verdict": "Confirmed. Both review prompts load execution_audit.json with a graceful fallback."
            }}
          ]
        }}
        ```
        - When the work needs another execute pass, keep the same shape and change only `review_verdict` to `needs_rework`; put the actionable gaps in `issues`, `summary`, and per-task verdicts.
        """
    ).strip()


_PromptBuilder = Callable[[PlanState, Path], str]

_CLAUDE_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": _review_claude_prompt,
}

_CODEX_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": _review_codex_prompt,
}


def create_claude_prompt(step: str, state: PlanState, plan_dir: Path) -> str:
    if step == "plan" and configured_robustness(state) == "light":
        return _plan_light_prompt(state, plan_dir)
    builder = _CLAUDE_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Claude step '{step}'")
    return builder(state, plan_dir)


def create_codex_prompt(step: str, state: PlanState, plan_dir: Path) -> str:
    if step == "plan" and configured_robustness(state) == "light":
        return _plan_light_prompt(state, plan_dir)
    builder = _CODEX_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Codex step '{step}'")
    return builder(state, plan_dir)
