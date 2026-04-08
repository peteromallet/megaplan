"""Review-phase prompt builders."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from megaplan._core import (
    collect_git_diff_patch,
    collect_git_diff_summary,
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    latest_plan_path,
    load_flag_registry,
    read_json,
)
from megaplan.types import PlanState


def _check_field(check: Any, name: str) -> Any:
    if isinstance(check, dict):
        return check.get(name)
    return getattr(check, name)


def _review_check_flag_id(check_id: str, index: int) -> str:
    stem = re.sub(r"[^A-Z0-9]+", "_", check_id.upper()).strip("_") or "CHECK"
    return f"REVIEW-{stem}-{index:03d}"


def _review_template_payload(plan_dir: Path) -> dict[str, object]:
    finalize_data = read_json(plan_dir / "finalize.json")

    task_verdicts = []
    for task in finalize_data.get("tasks", []):
        task_id = task.get("id", "")
        if task_id:
            task_verdicts.append({
                "task_id": task_id,
                "reviewer_verdict": "",
                "evidence_files": [],
            })

    sense_check_verdicts = []
    for sc in finalize_data.get("sense_checks", []):
        sc_id = sc.get("id", "")
        if sc_id:
            sense_check_verdicts.append({
                "sense_check_id": sc_id,
                "verdict": "",
            })

    criteria = []
    for crit in finalize_data.get("success_criteria", []):
        if isinstance(crit, dict) and crit.get("name"):
            criteria.append({
                "name": crit["name"],
                "priority": crit.get("priority", "must"),
                "pass": "",
                "evidence": "",
            })

    return {
        "review_verdict": "",
        "criteria": criteria,
        "issues": [],
        "rework_items": [],
        "summary": "",
        "task_verdicts": task_verdicts,
        "sense_check_verdicts": sense_check_verdicts,
    }


def _heavy_review_context(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    project_dir = Path(state["config"]["project_dir"])
    gate = read_json(plan_dir / "gate.json")
    settled_decisions = gate.get("settled_decisions", [])
    if not isinstance(settled_decisions, list):
        settled_decisions = []
    return {
        "project_dir": project_dir,
        "intent_block": intent_and_notes_block(state),
        "git_diff": collect_git_diff_patch(project_dir),
        "finalize_data": read_json(plan_dir / "finalize.json"),
        "settled_decisions": settled_decisions,
    }


def _build_review_checks_template(
    plan_dir: Path,
    state: PlanState,
    checks: tuple[Any, ...],
) -> list[dict[str, object]]:
    checks_template: list[dict[str, object]] = []
    for check in checks:
        entry: dict[str, object] = {
            "id": _check_field(check, "id"),
            "question": _check_field(check, "question"),
            "guidance": _check_field(check, "guidance") or "",
            "findings": [],
        }
        checks_template.append(entry)

    if state.get("iteration", 1) <= 1:
        return checks_template

    prior_path = plan_dir / "review.json"
    if not prior_path.exists():
        return checks_template

    prior = read_json(prior_path)
    active_check_ids = {_check_field(check, "id") for check in checks}
    prior_checks = {
        check.get("id"): check
        for check in prior.get("checks", [])
        if isinstance(check, dict) and check.get("id") in active_check_ids
    }
    registry = load_flag_registry(plan_dir)
    flag_status = {flag["id"]: flag.get("status", "open") for flag in registry.get("flags", [])}

    for entry in checks_template:
        check_id = str(entry["id"])
        prior_check = prior_checks.get(check_id)
        if not isinstance(prior_check, dict):
            continue
        prior_findings = []
        flagged_index = 0
        for finding in prior_check.get("findings", []):
            if not isinstance(finding, dict):
                continue
            flagged = bool(finding.get("flagged"))
            status = "n/a"
            if flagged:
                flagged_index += 1
                status = flag_status.get(_review_check_flag_id(check_id, flagged_index), "open")
            prior_findings.append({
                "detail": finding.get("detail", ""),
                "flagged": flagged,
                "status": finding.get("status", status),
            })
        if prior_findings:
            entry["prior_findings"] = prior_findings
    return checks_template


def _write_single_check_review_template(
    plan_dir: Path,
    state: PlanState,
    check: Any,
    filename: str,
) -> Path:
    template: dict[str, object] = {
        "checks": _build_review_checks_template(plan_dir, state, (check,)),
        "flags": [],
        "pre_check_flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }
    output_path = plan_dir / filename
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return output_path


def _write_criteria_verdict_review_template(
    plan_dir: Path,
    state: PlanState,
    filename: str,
) -> Path:
    del state
    output_path = plan_dir / filename
    output_path.write_text(json.dumps(_review_template_payload(plan_dir), indent=2), encoding="utf-8")
    return output_path


def _settled_decisions_review_block(settled_decisions: list[object]) -> str:
    if not settled_decisions:
        return "Settled decisions from gate (`gate.json`): []"
    return textwrap.dedent(
        f"""
        Settled decisions from gate (`gate.json`):
        {json_dump(settled_decisions).strip()}
        """
    ).strip()


def single_check_review_prompt(
    state: PlanState,
    plan_dir: Path,
    root: Path | None,
    check: Any,
    output_path: Path,
    pre_check_flags: list[dict[str, Any]],
) -> str:
    del root
    context = _heavy_review_context(state, plan_dir)
    check_id = _check_field(check, "id")
    question = _check_field(check, "question")
    guidance = _check_field(check, "guidance") or ""
    iteration = state.get("iteration", 1)
    iteration_context = ""
    if iteration > 1:
        iteration_context = (
            "\n\nThis is review iteration {iteration}. The template may include prior findings with their current "
            "flag status. Verify whether previously raised concerns were actually fixed before you carry them forward."
        ).format(iteration=iteration)
    return textwrap.dedent(
        f"""
        You are an independent heavy-review checker. Review one focused dimension of the executed patch against the original issue text.

        Project directory:
        {context["project_dir"]}

        {context["intent_block"]}

        Full git diff:
        {context["git_diff"]}

        Execution tracking state (`finalize.json`):
        {json_dump(context["finalize_data"]).strip()}

        {_settled_decisions_review_block(context["settled_decisions"])}

        Advisory mechanical pre-check flags (copy these verbatim into `pre_check_flags` in the output file):
        {json_dump(pre_check_flags).strip()}

        Your output template is at: {output_path}
        Read this file first. It contains exactly one check slot.

        Check ID: {check_id}
        Question: {question}
        Guidance: {guidance}

        Requirements:
        - Anchor your reasoning to the original issue text and the full diff above, not to any approved plan.
        - Investigate only this check.
        - Populate the existing `checks[0].findings` array with concrete findings. Each finding should include:
          - `detail`: a full sentence describing what you checked and what you found
          - `flagged`: `true` when the finding represents a risk, mismatch, or unresolved question
          - `status`: use `blocking`, `significant`, `minor`, or `n/a`
          - `evidence_file` when a file path makes the finding easier to act on
        - If a concern overlaps with a settled gate decision, do NOT raise it as `blocking`. Mark it `significant` and explain that the severity was downgraded because the gate already settled that concern.
        - Use `blocking` only for issue-anchored gaps that should force another revise/execute pass.
        - Use `significant` for meaningful but non-blocking concerns, including settled-decision downgrades.
        - Use `minor` for informational quality notes that do not justify rework.
        - Use `flagged: false` with `status: "n/a"` only when the finding is purely informational and poses no downside.
        - Leave `flags` empty unless you discover an additional concern that does not fit the focused check.
        - Keep `verified_flag_ids` and `disputed_flag_ids` empty unless you are explicitly confirming or disputing an existing REVIEW-* flag from a prior iteration.
        - Preserve the `pre_check_flags` list verbatim in the output file.{iteration_context}
        """
    ).strip()


def heavy_criteria_review_prompt(
    state: PlanState,
    plan_dir: Path,
    root: Path | None,
    output_path: Path,
) -> str:
    """Build the heavy-mode criteria review prompt.

    This intentionally does not wrap `_review_prompt()`. The brief literally
    asked to keep `_review_prompt()` as the heavy criteria check, but that would
    leak plan/gate/execution context that conflicts with the stronger
    issue-anchored review contract. This divergence is deliberate.
    """
    del root
    context = _heavy_review_context(state, plan_dir)
    return textwrap.dedent(
        f"""
        Review the execution against the original issue text and the finalized execution criteria.

        Project directory:
        {context["project_dir"]}

        {context["intent_block"]}

        Full git diff:
        {context["git_diff"]}

        Execution tracking state (`finalize.json`):
        {json_dump(context["finalize_data"]).strip()}

        {_settled_decisions_review_block(context["settled_decisions"])}

        Your output template is at: {output_path}
        Read the file first and write your final answer into that JSON structure.

        Requirements:
        - Use only the issue text, full git diff, `finalize.json`, and the settled decisions shown above.
        - Do not rely on any approved plan, plan metadata, gate summary, execution summary, or execution audit that are not present here.
        - Judge against the success criteria from `finalize.json`, but stay anchored to the original issue text when deciding whether the work actually solved the problem.
        - Each criterion has a `priority` (`must`, `should`, or `info`). Apply these rules:
          - `must` criteria are hard gates. A `must` criterion that fails means `needs_rework`.
          - `should` criteria are quality targets. If the spirit is met but the letter is not, mark `pass` with evidence explaining the gap. Only mark `fail` if the intent was clearly missed. A `should` failure alone does NOT require `needs_rework`.
          - `info` criteria are for human reference. Mark them `waived` with a note — do not evaluate them.
          - If a criterion cannot be verified in this context, mark it `waived` with an explanation.
        - Set `review_verdict` to `needs_rework` only when at least one `must` criterion fails or actual implementation work is incomplete. Use `approved` when all `must` criteria pass, even if some `should` criteria are flagged.
        - The settled decisions above are already approved. Verify implementation against them, but do not re-litigate them.
        - `rework_items` must be structured and directly actionable. Populate `issues` as one-line summaries derived from `rework_items`.
        - When approved, keep both `issues` and `rework_items` empty arrays.
        """
    ).strip()


def _settled_decisions_block(gate: dict[str, object]) -> str:
    settled_decisions = gate.get("settled_decisions", [])
    if not isinstance(settled_decisions, list) or not settled_decisions:
        return ""
    lines = ["Settled decisions (verify the executor implemented these correctly):"]
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
    return "- The decisions listed above were settled at the gate stage. Verify that the executor implemented each settled decision correctly. Flag deviations from these decisions, but do not question the decisions themselves."


def _write_review_template(plan_dir: Path, state: PlanState) -> Path:
    """Write a pre-populated review output template and return its path.

    Pre-fills ``task_verdicts`` and ``sense_check_verdicts`` with the actual
    task IDs and sense-check IDs from ``finalize.json`` so the model only has
    to fill in verdict text instead of inventing IDs from scratch.  This is
    the same pattern used for critique templates and fixes MiniMax-M2.7's
    tendency to return empty verdict arrays.
    """
    finalize_data = read_json(plan_dir / "finalize.json")

    task_verdicts = []
    for task in finalize_data.get("tasks", []):
        task_id = task.get("id", "")
        if task_id:
            task_verdicts.append({
                "task_id": task_id,
                "reviewer_verdict": "",
                "evidence_files": [],
            })

    sense_check_verdicts = []
    for sc in finalize_data.get("sense_checks", []):
        sc_id = sc.get("id", "")
        if sc_id:
            sense_check_verdicts.append({
                "sense_check_id": sc_id,
                "verdict": "",
            })

    # Pre-populate criteria from finalize success_criteria if available
    criteria = []
    for crit in finalize_data.get("success_criteria", []):
        if isinstance(crit, dict) and crit.get("name"):
            criteria.append({
                "name": crit["name"],
                "priority": crit.get("priority", "must"),
                "pass": "",
                "evidence": "",
            })

    template = {
        "review_verdict": "",
        "criteria": criteria,
        "issues": [],
        "rework_items": [],
        "summary": "",
        "task_verdicts": task_verdicts,
        "sense_check_verdicts": sense_check_verdicts,
    }

    output_path = plan_dir / "review_output.json"
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return output_path


def _review_prompt(
    state: PlanState,
    plan_dir: Path,
    *,
    review_intro: str,
    criteria_guidance: str,
    task_guidance: str,
    sense_check_guidance: str,
    pre_check_flags: list[dict[str, Any]] | None = None,
) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    gate = read_json(plan_dir / "gate.json")
    finalize_data = read_json(plan_dir / "finalize.json")
    settled_decisions_block = _settled_decisions_block(gate)
    settled_decisions_instruction = _settled_decisions_instruction(gate)
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
    flag_reverify_items: list[dict[str, str]] = []
    for flag in load_flag_registry(plan_dir).get("flags", []):
        if not isinstance(flag, dict):
            continue
        status = str(flag.get("status", "open"))
        if status not in {"open", "addressed", "verified", "disputed"}:
            continue
        flag_reverify_items.append(
            {
                "id": str(flag.get("id", "")),
                "concern": str(flag.get("concern", "")),
                "severity": str(flag.get("severity") or flag.get("severity_hint") or "uncertain"),
                "status": status,
            }
        )
    flag_reverify_block = ""
    if flag_reverify_items:
        flag_reverify_block = textwrap.dedent(
            f"""
            Critique flags to re-verify against the final diff:
            {json_dump(flag_reverify_items).strip()}

            For each flag above that was raised during critique, verify whether the final diff actually addresses the concern.
            A flag is resolved only if the final diff contains code that directly addresses the concern.
            Do not trust pre-execute promises or plan claims; check the diff itself.
            Add resolved flag IDs to `verified_flag_ids`.
            For any unresolved flag, add a `rework_items` entry with `task_id: "REVIEW"`, `issue`, `expected`, `actual`, `evidence_file`, `flag_id`, and `source: "review_flag_reverify"`.
            """
        ).strip()
    pre_check_block = ""
    if pre_check_flags:
        pre_check_block = textwrap.dedent(
            f"""
            Advisory mechanical pre-check flags:
            {json_dump(pre_check_flags).strip()}

            Copy this list verbatim into the output `pre_check_flags` field.
            """
        ).strip()
    extra_sections = ""
    if flag_reverify_block:
        extra_sections += f"\n\n{flag_reverify_block}"
    if pre_check_block:
        extra_sections += f"\n\n{pre_check_block}"
    return textwrap.dedent(
        f"""
        {review_intro}

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

        {settled_decisions_block}{extra_sections}

        Execution summary:
        {json_dump(execution).strip()}

        {audit_block}

        Git diff summary:
        {diff_summary}

        Requirements:
        - {criteria_guidance}
        - Trust executor evidence by default. Dig deeper only where the git diff, `execution_audit.json`, or vague notes make the claim ambiguous.
        - Each criterion has a `priority` (`must`, `should`, or `info`). Apply these rules:
          - `must` criteria are hard gates. A `must` criterion that fails means `needs_rework`.
          - `should` criteria are quality targets. If the spirit is met but the letter is not, mark `pass` with evidence explaining the gap. Only mark `fail` if the intent was clearly missed. A `should` failure alone does NOT require `needs_rework`.
          - `info` criteria are for human reference. Mark them `waived` with a note — do not evaluate them.
          - If a criterion (any priority) cannot be verified in this context (e.g., requires manual testing or runtime observation), mark it `waived` with an explanation.
        - Set `review_verdict` to `needs_rework` only when at least one `must` criterion fails or actual implementation work is incomplete. Use `approved` when all `must` criteria pass, even if some `should` criteria are flagged.
        {settled_decisions_instruction}
        - {task_guidance}
        - {sense_check_guidance}
        - Follow this JSON shape exactly:
        ```json
        {{
          "review_verdict": "approved",
          "criteria": [
            {{
              "name": "All existing tests pass",
              "priority": "must",
              "pass": "pass",
              "evidence": "Test suite ran green — 42 passed, 0 failed."
            }},
            {{
              "name": "File under ~300 lines",
              "priority": "should",
              "pass": "pass",
              "evidence": "File is 375 lines — above the target but reasonable given the component's responsibilities. Spirit met."
            }},
            {{
              "name": "Manual smoke tests pass",
              "priority": "info",
              "pass": "waived",
              "evidence": "Cannot be verified in automated review. Noted for manual QA."
            }}
          ],
          "issues": [],
          "rework_items": [],
          "summary": "Approved. All must criteria pass. The should criterion on line count is close enough given the component scope.",
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
        - `rework_items` must be an array of structured rework directives. When `review_verdict` is `needs_rework`, populate one entry per issue with:
          - `task_id`: which finalize task this issue relates to
          - `issue`: what is wrong
          - `expected`: what correct behavior looks like
          - `actual`: what was observed
          - `evidence_file` (optional): file path supporting the finding
        - `issues` must still be populated as a flat one-line-per-item summary derived from `rework_items` (for backward compatibility). When approved, both `issues` and `rework_items` should be empty arrays.
        - When the work needs another execute pass, keep the same shape and change only `review_verdict` to `needs_rework`; make `issues`, `rework_items`, `summary`, and task verdicts specific enough for the executor to act on directly.
        """
    ).strip()
