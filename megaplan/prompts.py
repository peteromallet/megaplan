"""Prompt builders for each megaplan step and dispatch tables."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from megaplan._core import (
    PlanState,
    CliError,
    latest_plan_path,
    read_json,
    latest_plan_meta_path,
    load_flag_registry,
    unresolved_significant_flags,
    intent_and_notes_block,
    json_dump,
    current_iteration_artifact,
    configured_robustness,
    robustness_critique_instruction,
    collect_git_diff_summary,
)


def _clarify_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    notes = state["meta"].get("notes", [])
    notes_block = "\n".join(f"- {note['note']}" for note in notes) if notes else "- None"
    return textwrap.dedent(
        f"""
        You are a planning assistant. The user has proposed the following idea:

        Idea:
        {state['idea']}

        Project directory:
        {project_dir}

        User notes:
        {notes_block}

        Requirements:
        - Read the project directory to understand the codebase.
        - Restate the idea in your own words as a precise intent summary.
        - Identify ambiguities, underspecified aspects, or implicit assumptions.
        - For each ambiguity, produce a question that, if answered, would materially change the implementation plan.
        - Propose a refined version of the idea that resolves obvious ambiguities.
        - Do NOT plan the implementation - only clarify the intent.
        """
    ).strip()


def _plan_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    notes = state["meta"].get("notes", [])
    notes_block = "\n".join(f"- {note['note']}" for note in notes) if notes else "- None"
    clarification = state.get("clarification", {})
    refined = clarification.get("refined_idea", "")
    intent = clarification.get("intent_summary", "")
    if refined:
        clarify_block = textwrap.dedent(
            f"""
            Refined idea (from clarification):
            {refined}

            Intent summary:
            {intent}

            Original idea (for reference):
            {state['idea']}
            """
        ).strip()
    else:
        clarify_block = textwrap.dedent(
            f"""
            Idea:
            {state['idea']}
            """
        ).strip()
    return textwrap.dedent(
        f"""
        You are creating an implementation plan for the following idea.

        {clarify_block}

        Project directory:
        {project_dir}

        User notes:
        {notes_block}

        Requirements:
        - Inspect the actual repository before planning.
        - Produce a concrete implementation plan in markdown.
        - Define observable success criteria.
        - Call out assumptions and open questions.
        - Prefer cheap validation steps early.
        """
    ).strip()


def _integrate_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    evaluate_path = current_iteration_artifact(plan_dir, "evaluation", state["iteration"])
    evaluation = read_json(evaluate_path)
    unresolved = unresolved_significant_flags(flag_registry)
    open_flags = [
        {
            "id": flag["id"],
            "severity": flag.get("severity"),
            "status": flag.get("status"),
            "concern": flag.get("concern"),
            "evidence": flag.get("evidence"),
        }
        for flag in unresolved
    ]
    return textwrap.dedent(
        f"""
        You are updating an implementation plan based on critique and evaluation.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Current plan (markdown):
        {latest_plan}

        Current plan metadata:
        {json_dump(latest_meta).strip()}

        Evaluation:
        {json_dump(evaluation).strip()}

        Open significant flags:
        {json_dump(open_flags).strip()}

        Requirements:
        - Update the plan to address the significant issues.
        - Keep the plan readable and executable.
        - Return flags_addressed with the exact flag IDs you addressed.
        - Preserve or improve success criteria quality.
        - Verify that the plan remains aligned with the user's original intent (above), not just internal plan quality.
        - Remove unjustified scope growth. If the critique raised scope creep, narrow the plan back to the original idea unless the broader work is strictly required.
        - If a broader change is truly necessary, explain that dependency explicitly in changes_summary instead of silently expanding the plan.
        """
    ).strip()


def _critique_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    robustness = configured_robustness(state)
    unresolved = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "status": flag.get("status"),
            "severity": flag.get("severity"),
        }
        for flag in flag_registry["flags"]
        if flag.get("status") in {"addressed", "open", "disputed"}
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

        Existing flags:
        {json_dump(unresolved).strip()}

        Requirements:
        - Consider whether the plan is at the right level of abstraction. If it
          patches multiple systems for one goal, it may be too low — flag whether
          a simpler design would eliminate the problem class. If it redesigns
          architecture for a simple bug, it may be too high. Push the plan up or
          down the abstraction ladder as needed.
        - Reuse existing flag IDs when the same concern is still open.
        - verified_flag_ids should list previously addressed flags that now appear resolved.
        - Focus on concrete issues that would cause real problems.
        - Robustness level: {robustness}. {robustness_critique_instruction(robustness)}
        - Verify that the plan remains aligned with the user's original intent (above), not just internal plan quality.
        - Flag scope creep explicitly when the plan grows beyond the original idea or recorded user notes. Use the phrase "Scope creep:" in the concern so the orchestrator can surface it.
        - Do not rubber-stamp the plan.
        - Assign severity_hint carefully: "likely-significant" for issues that would
          cause real product or implementation problems. "likely-minor" for cosmetic,
          nice-to-have, issues already covered elsewhere, or implementation details
          the executor will naturally resolve by reading the actual code (e.g. exact
          line numbers, missing boilerplate, export lists).
        """
    ).strip()


def _execute_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    robustness = configured_robustness(state)
    gate = read_json(plan_dir / "gate.json")
    if state["config"].get("auto_approve"):
        approval_note = "Note: User chose auto-approve mode. This execution was not manually reviewed at the gate. Exercise extra caution on destructive operations."
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

        Approved plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        {approval_note}
        Robustness level: {robustness}.

        Requirements:
        - Implement the intent, not just the text.
        - Adapt if repository reality contradicts the plan.
        - Report deviations explicitly.
        - Output concrete files changed and commands run.
        """
    ).strip()


def _review_claude_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    gate = read_json(plan_dir / "gate.json")
    diff_summary = collect_git_diff_summary(project_dir)
    return textwrap.dedent(
        f"""
        Review the execution critically against user intent and observable success criteria.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Gate summary:
        {json_dump(gate).strip()}

        Execution summary:
        {json_dump(execution).strip()}

        Git diff summary:
        {diff_summary}

        Requirements:
        - Judge against the success criteria, not plan elegance.
        - Be critical and call out real misses.
        - If there are failures, describe them as issues.
        """
    ).strip()


def _review_codex_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    diff_summary = collect_git_diff_summary(project_dir)
    return textwrap.dedent(
        f"""
        Review the implementation against the success criteria.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Execution summary:
        {json_dump(execution).strip()}

        Git diff summary:
        {diff_summary}

        Requirements:
        - Be critical.
        - Verify each success criterion explicitly.
        - Call out any concrete gaps or regressions in issues.
        """
    ).strip()


# Step-to-builder dispatch tables per agent.
# Steps shared across agents point to the same builder function.
_CLAUDE_PROMPT_BUILDERS: dict[str, Any] = {
    "clarify": _clarify_prompt,
    "plan": _plan_prompt,
    "integrate": _integrate_prompt,
    "critique": _critique_prompt,
    "execute": _execute_prompt,
    "review": _review_claude_prompt,
}

_CODEX_PROMPT_BUILDERS: dict[str, Any] = {
    "clarify": _clarify_prompt,
    "plan": _plan_prompt,
    "integrate": _integrate_prompt,
    "critique": _critique_prompt,
    "execute": _execute_prompt,
    "review": _review_codex_prompt,
}


def create_claude_prompt(step: str, state: PlanState, plan_dir: Path) -> str:
    builder = _CLAUDE_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Claude step '{step}'")
    return builder(state, plan_dir)


def create_codex_prompt(step: str, state: PlanState, plan_dir: Path) -> str:
    builder = _CODEX_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Codex step '{step}'")
    return builder(state, plan_dir)
