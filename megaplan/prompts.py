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

        Existing flags:
        {json_dump(unresolved).strip()}

        Requirements:
        - Consider whether the plan is at the right level of abstraction.
        - Reuse existing flag IDs when the same concern is still open.
        - `verified_flag_ids` should list previously addressed flags that now appear resolved.
        - Focus on concrete issues that would cause real problems.
        - Robustness level: {robustness}. {robustness_critique_instruction(robustness)}
        - Verify that the plan remains aligned with the user's original intent.
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
        - Produce an execution-ready markdown document in the `final_plan` field.
        - Start with a task checklist: `- [ ] Task description` items, ordered by execution sequence, with dependencies or sequencing notes where relevant.
        - Each checklist item MUST be followed by an indented comment line: `  > _notes:_` — the executor is required to fill this in during execution with what they actually did, any deviations, or confirmation.
        - Follow the checklist with a "Watch Items" section listing edge cases from critique, gate warnings, and assumptions that need runtime verification.
        - After Watch Items, add a "Review Sense-Check" section with one `- [ ]` item per checklist task, worded as a verification question (e.g., "Verify: STATE_FINALIZED is exported and used in all routing tables"). The reviewer will check these off after reading the executor's notes and verifying the work. Each sense-check item must also have a `  > _verdict:_` line for the reviewer to fill in.
        - End with a "Meta" section: brief commentary to help the executor succeed — context, gotchas, or judgment calls that aren't obvious from the plan alone.
        - The document should be self-contained: an executor reading only this document should have everything they need.
        - `task_count` should equal the number of checklist items.
        - `watch_items` should list the watch items as an array of strings.
        - `meta_commentary` should contain the meta section text.
        """
    ).strip()


def _execute_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    final_md = plan_dir / "final.md"
    latest_plan = final_md.read_text(encoding="utf-8") if final_md.exists() else latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    robustness = configured_robustness(state)
    gate = read_json(plan_dir / "gate.json")
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

        Execution-ready plan:
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
        - IMPORTANT: The execution-ready plan contains a task checklist. As you complete each task, update final.md: change `- [ ]` to `- [x]` and fill in the `> _notes:_` line with what you actually did, any deviations, or confirmation that it went as planned. Every checklist item must be checked off and commented before execution is complete.
        """
    ).strip()


def _review_claude_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    gate = read_json(plan_dir / "gate.json")
    diff_summary = collect_git_diff_summary(project_dir)
    final_md = plan_dir / "final.md"
    finalized_plan = final_md.read_text(encoding="utf-8") if final_md.exists() else ""
    return textwrap.dedent(
        f"""
        Review the execution critically against user intent and observable success criteria.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Finalized plan (execution-ready document with checklist):
        {finalized_plan}

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
        - Verify the finalized plan checklist: every `- [ ]` item in the Checklist section should be `- [x]` with a filled-in `> _notes:_` comment. Flag any unchecked or uncommented items as issues.
        - Sense-check each executor comment: does the note actually describe completing the task? Does it make sense given the git diff? Flag vague, copy-pasted, or contradictory notes as issues.
        - Fill in the "Review Sense-Check" section in final.md: for each sense-check item, change `- [ ]` to `- [x]` if the work checks out or leave it unchecked if it doesn't, and fill in the `> _verdict:_` line with your assessment.
        """
    ).strip()


def _review_codex_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    execution = read_json(plan_dir / "execution.json")
    diff_summary = collect_git_diff_summary(project_dir)
    final_md = plan_dir / "final.md"
    finalized_plan = final_md.read_text(encoding="utf-8") if final_md.exists() else ""
    return textwrap.dedent(
        f"""
        Review the implementation against the success criteria.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Approved plan:
        {latest_plan}

        Finalized plan (execution-ready document with checklist):
        {finalized_plan}

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
        - Verify the finalized plan checklist: every `- [ ]` item in the Checklist section should be `- [x]` with a filled-in `> _notes:_` comment. Flag any unchecked or uncommented items as issues.
        - Sense-check each executor comment: does the note actually describe completing the task? Does it make sense given the git diff? Flag vague, copy-pasted, or contradictory notes as issues.
        - Fill in the "Review Sense-Check" section in final.md: for each sense-check item, change `- [ ]` to `- [x]` if the work checks out or leave it unchecked if it doesn't, and fill in the `> _verdict:_` line with your assessment.
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
    builder = _CLAUDE_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Claude step '{step}'")
    return builder(state, plan_dir)


def create_codex_prompt(step: str, state: PlanState, plan_dir: Path) -> str:
    builder = _CODEX_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Codex step '{step}'")
    return builder(state, plan_dir)
