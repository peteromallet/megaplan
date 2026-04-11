"""Critique- and revise-phase prompt builders."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from megaplan.checks import checks_for_robustness
from megaplan._core import (
    configured_robustness,
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    latest_plan_path,
    load_flag_registry,
    read_json,
    robustness_critique_instruction,
    unresolved_significant_flags,
)
from megaplan.types import PlanState

from ._shared import _planning_debt_block, _render_prep_block
from .planning import PLAN_TEMPLATE


def _revise_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
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

        {prep_block}
        {prep_instruction}

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
        - Before addressing individual flags, check: does any flag suggest the plan is targeting the wrong code or the wrong root cause? If so, consider whether the plan needs a new approach rather than adjustments. Explain your reasoning.
        - Update the plan to address the significant issues.
        - Keep the plan readable and executable.
        - Return flags_addressed with the exact flag IDs you addressed.
        - Include `changes_summary` as a short plain-English summary of what changed in the revision. If there were no concrete flags, say that explicitly (for example: `No critique flags were raised; refined wording and kept the plan aligned for execution.`).
        - Preserve or improve success criteria quality. Each criterion must have a `priority` of `must`, `should`, or `info`. Promote or demote priorities if critique feedback reveals a criterion was over- or under-weighted.
        - Verify that the plan remains aligned with the user's original intent, not just internal plan quality.
        - Remove unjustified scope growth. If critique raised scope creep, narrow the plan back to the original idea unless the broader work is strictly required.
        - Maintain the structural template: H1 title, ## Overview, phase sections with numbered step sections, ## Execution Order or ## Validation Order.
        - CRITICAL: Your entire revised plan markdown (all sections) must be output as the `plan` field in the structured output. The prose response must not contain the plan text.
        - CRITICAL: Return only the structured JSON object for the schema fields `plan`, `changes_summary`, `flags_addressed`, `assumptions`, `success_criteria`, and `questions`. Do not add commentary before or after the JSON object.

        {PLAN_TEMPLATE}
        """
    ).strip()


def _critique_context(state: PlanState, plan_dir: Path, root: Path | None = None) -> dict[str, Any]:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    structure_warnings = latest_meta.get("structure_warnings", [])
    flag_registry = load_flag_registry(plan_dir)
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
    return {
        "project_dir": project_dir,
        "prep_block": prep_block,
        "prep_instruction": prep_instruction,
        "latest_plan": latest_plan,
        "latest_meta": latest_meta,
        "structure_warnings": structure_warnings,
        "unresolved": unresolved,
        "debt_block": _planning_debt_block(plan_dir, root),
        "robustness": configured_robustness(state),
    }


def _build_checks_template(
    plan_dir: Path,
    state: PlanState,
    checks: tuple[dict[str, Any], ...],
) -> list[dict[str, object]]:
    checks_template = []
    for check in checks:
        entry: dict[str, object] = {
            "id": check["id"],
            "question": check["question"],
            "guidance": check.get("guidance", ""),
            "findings": [],
        }
        checks_template.append(entry)

    iteration = state.get("iteration", 1)
    if iteration > 1:
        prior_path = plan_dir / f"critique_v{iteration - 1}.json"
        if prior_path.exists():
            prior = read_json(prior_path)
            active_check_ids = {check["id"] for check in checks}
            prior_checks = {
                c.get("id"): c for c in prior.get("checks", [])
                if isinstance(c, dict) and c.get("id") in active_check_ids
            }
            registry = load_flag_registry(plan_dir)
            flag_status = {f["id"]: f.get("status", "open") for f in registry.get("flags", [])}
            for entry in checks_template:
                cid = entry["id"]
                if cid in prior_checks:
                    pc = prior_checks[cid]
                    prior_findings = []
                    flagged_count = sum(1 for f in pc.get("findings", []) if f.get("flagged"))
                    flagged_idx = 0
                    for f in pc.get("findings", []):
                        pf: dict[str, object] = {
                            "detail": f.get("detail", ""),
                            "flagged": f.get("flagged", False),
                        }
                        if f.get("flagged"):
                            flagged_idx += 1
                            fid = cid if flagged_count == 1 else f"{cid}-{flagged_idx}"
                            pf["status"] = flag_status.get(fid, flag_status.get(cid, "open"))
                        else:
                            pf["status"] = "n/a"
                        prior_findings.append(pf)
                    entry["prior_findings"] = prior_findings
    return checks_template


def _build_critique_prompt(
    state: PlanState,
    context: dict[str, Any],
    critique_review_block: str,
) -> str:
    return textwrap.dedent(
        f"""
        You are an independent reviewer. Critique the plan against the actual repository.

        Project directory:
        {context["project_dir"]}

        {context["prep_block"]}

        {context["prep_instruction"]}

        {intent_and_notes_block(state)}

        Plan:
        {context["latest_plan"]}

        Plan metadata:
        {json_dump(context["latest_meta"]).strip()}

        Plan structure warnings from validator:
        {json_dump(context["structure_warnings"]).strip()}

        Existing flags:
        {json_dump(context["unresolved"]).strip()}

        {context["debt_block"]}

        {critique_review_block}

        Additional guidelines:
        - Robustness level: {context["robustness"]}. {robustness_critique_instruction(context["robustness"])}
        - Over-engineering: prefer the simplest approach that fully solves the problem.
        - Reuse existing flag IDs when the same concern is still open.
        - `verified_flag_ids`: list flag IDs from prior iterations that the revised plan actually resolves (e.g., if the plan was revised to fix FLAG-001, and you confirm the fix is correct, include "FLAG-001"). Only include flags you've verified — don't guess.
        - Verify that the plan follows the expected structure when validator warnings or the outline suggest drift.
        - Additional flags may use these categories: correctness, security, completeness, performance, maintainability, other.
        - Focus on concrete issues, not structural formatting.
        """
    ).strip()


def _write_critique_template(
    plan_dir: Path,
    state: PlanState,
    checks: tuple[dict[str, Any], ...],
) -> Path:
    """Write the critique output template file and return its path.

    The file serves as both guide (check questions + guidance) and output
    (findings arrays to fill in). This is the model's sole output channel.
    """
    import json

    template: dict[str, object] = {
        "checks": _build_checks_template(plan_dir, state, checks),
        "flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }

    output_path = plan_dir / "critique_output.json"
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return output_path


def write_single_check_template(
    plan_dir: Path,
    state: PlanState,
    check: dict[str, Any],
    output_name: str,
) -> Path:
    import json

    template: dict[str, object] = {
        "checks": _build_checks_template(plan_dir, state, (check,)),
        "flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
    }

    output_path = plan_dir / output_name
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return output_path


def _critique_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    context = _critique_context(state, plan_dir, root)
    active_checks = checks_for_robustness(context["robustness"])
    # Write the template file — this is both the guide and the output
    output_path = _write_critique_template(plan_dir, state, active_checks)
    iteration = state.get("iteration", 1)

    if active_checks:
        iteration_context = ""
        if iteration > 1:
            iteration_context = (
                "\n\n            This is critique iteration {iteration}. "
                "The template file includes prior findings with their status. "
                "Verify addressed flags were actually fixed, re-flag if inadequate, "
                "and check for new issues introduced by the revision."
            ).format(iteration=iteration)
        critique_review_block = textwrap.dedent(
            f"""
            Your output template is at: {output_path}
            Read this file first — it contains {len(active_checks)} checks, each with a question and guidance.
            For each check, investigate the codebase, then add your findings to the `findings` array for that check.

            Each finding needs:
            - "detail": what you specifically checked and what you found (at least a full sentence)
            - "flagged": true if this describes a difference, risk, or tension — even if you think it's justified. false only if purely informational with no possible downside.
            - Every check must end with at least one finding. Never leave a `findings` array empty. If you found no issue, add one detailed `flagged: false` finding explaining what you checked and why it appears clear.

            When in doubt, flag it — the gate can accept tradeoffs, but it can't act on findings it never sees.

            Good: {{"detail": "Checked callers of nthroot_mod in solveset.py line 1205 — passes prime moduli only, consistent with the fix.", "flagged": false}}
            Good: {{"detail": "The fix handles empty tuples but not single-element tuples which need a trailing comma.", "flagged": true}}
            Bad: {{"detail": "No issue found", "flagged": false}}  ← too brief, will be rejected
            Bad: {{"detail": "The hints suggest approach X but the plan uses Y. However Y is consistent with X's intent.", "flagged": false}}  ← a different approach than the hints IS a flag. You found a divergence — flag it. The gate decides if it's acceptable.

            After filling in checks, add any additional concerns to the `flags` array (e.g., security, performance, dependencies).
            Use the standard format (id, concern, category, severity_hint, evidence). This array can be empty.

            Workflow: read the file → investigate → read file again → add finding → write file back. Repeat for each check.{iteration_context}
        """
        ).strip()
    else:
        critique_review_block = textwrap.dedent(
            f"""
            Your output template is at: {output_path}
            Review the plan with a broad scope. Consider whether the approach is correct, whether it covers
            all the places it needs to, whether it would break callers or violate codebase conventions,
            and whether its verification strategy is adequate.

            Place any concrete concerns in the `flags` array in the template file using the standard format
            (id, concern, category, severity_hint, evidence). Leave `checks` as an empty array.

            Workflow: read the file → investigate → read file again → add findings → write file back.
        """
        ).strip()
    return _build_critique_prompt(state, context, critique_review_block)


def single_check_critique_prompt(
    state: PlanState,
    plan_dir: Path,
    root: Path | None,
    check: dict[str, Any],
    template_path: Path,
) -> str:
    context = _critique_context(state, plan_dir, root)
    iteration = state.get("iteration", 1)
    iteration_context = ""
    if iteration > 1:
        iteration_context = (
            "\n\n            This is critique iteration {iteration}. "
            "The template file includes prior findings with their status. "
            "Verify addressed flags were actually fixed, re-flag if inadequate, "
            "and check for new issues introduced by the revision."
        ).format(iteration=iteration)
    critique_review_block = textwrap.dedent(
        f"""
        Your output template is at: {template_path}
        Read this file first — it contains 1 check with a question and guidance.
        Investigate only this check, then add your findings to the `findings` array for that check.

        Check ID: {check["id"]}
        Question: {check["question"]}
        Guidance: {check.get("guidance", "")}

        Each finding needs:
        - "detail": what you specifically checked and what you found (at least a full sentence)
        - "flagged": true if this describes a difference, risk, or tension — even if you think it's justified. false only if purely informational with no possible downside.
        - This check must end with at least one finding. Never leave its `findings` array empty. If you found no issue, add one detailed `flagged: false` finding explaining what you checked and why it appears clear.

        When in doubt, flag it — the gate can accept tradeoffs, but it can't act on findings it never sees.

        Good: {{"detail": "Checked callers of nthroot_mod in solveset.py line 1205 — passes prime moduli only, consistent with the fix.", "flagged": false}}
        Good: {{"detail": "The fix handles empty tuples but not single-element tuples which need a trailing comma.", "flagged": true}}
        Bad: {{"detail": "No issue found", "flagged": false}}  ← too brief, will be rejected
        Bad: {{"detail": "The hints suggest approach X but the plan uses Y. However Y is consistent with X's intent.", "flagged": false}}  ← a different approach than the hints IS a flag. You found a divergence — flag it. The gate decides if it's acceptable.

        After filling in checks, add any additional concerns to the `flags` array (e.g., security, performance, dependencies).
        Use the standard format (id, concern, category, severity_hint, evidence). This array can be empty.

        Workflow: read the file → investigate → read file again → add finding → write file back. Repeat for this check.{iteration_context}
    """
    ).strip()
    return _build_critique_prompt(state, context, critique_review_block)
