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
    batch_artifact_path,
    collect_git_diff_summary,
    compute_task_batches,
    configured_robustness,
    current_iteration_artifact,
    debt_by_subsystem,
    escalated_subsystems,
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    latest_plan_path,
    load_debt_registry,
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


def _resolve_prompt_root(plan_dir: Path, root: Path | None) -> Path:
    if root is not None:
        return root
    if len(plan_dir.parents) >= 3:
        return plan_dir.parents[2]
    return plan_dir


def _grouped_debt_for_prompt(plan_dir: Path, root: Path | None) -> dict[str, list[dict[str, object]]]:
    registry = load_debt_registry(_resolve_prompt_root(plan_dir, root))
    grouped_entries = debt_by_subsystem(registry)
    return {
        subsystem: [
            {
                "id": entry["id"],
                "concern": entry["concern"],
                "occurrence_count": entry["occurrence_count"],
                "plan_ids": entry["plan_ids"],
            }
            for entry in entries
        ]
        for subsystem, entries in sorted(grouped_entries.items())
    }


def _escalated_debt_for_prompt(plan_dir: Path, root: Path | None) -> list[dict[str, object]]:
    registry = load_debt_registry(_resolve_prompt_root(plan_dir, root))
    return [
        {
            "subsystem": subsystem,
            "total_occurrences": total,
            "plan_count": len({plan_id for entry in entries for plan_id in entry["plan_ids"]}),
            "entries": [
                {
                    "id": entry["id"],
                    "concern": entry["concern"],
                    "occurrence_count": entry["occurrence_count"],
                    "plan_ids": entry["plan_ids"],
                }
                for entry in entries
            ],
        }
        for subsystem, total, entries in escalated_subsystems(registry)
    ]


def _debt_watch_lines(plan_dir: Path, root: Path | None) -> list[str]:
    lines: list[str] = []
    for subsystem, entries in sorted(_grouped_debt_for_prompt(plan_dir, root).items()):
        for entry in entries:
            lines.append(
                f"[DEBT] {subsystem}: {entry['concern']} "
                f"(flagged {entry['occurrence_count']} times across {len(entry['plan_ids'])} plans)"
            )
    return lines


def _planning_debt_block(plan_dir: Path, root: Path | None) -> str:
    return textwrap.dedent(
        f"""
        Known accepted debt grouped by subsystem:
        {json_dump(_grouped_debt_for_prompt(plan_dir, root)).strip()}

        Escalated debt subsystems:
        {json_dump(_escalated_debt_for_prompt(plan_dir, root)).strip()}

        Debt guidance:
        - These are known accepted limitations. Do not re-flag them unless the current plan makes them worse, broadens them, or fails to contain them.
        - Prefix every new concern with a subsystem tag followed by a colon, for example `Timeout recovery: retry backoff remains brittle`.
        - When a concern is recurring debt that still needs to be flagged, prefix it with `Recurring debt:` after the subsystem tag, for example `Timeout recovery: Recurring debt: retry backoff remains brittle`.
        """
    ).strip()


def _gate_debt_block(plan_dir: Path, root: Path | None) -> str:
    return textwrap.dedent(
        f"""
        Known accepted debt grouped by subsystem:
        {json_dump(_grouped_debt_for_prompt(plan_dir, root)).strip()}

        Escalated debt subsystems:
        {json_dump(_escalated_debt_for_prompt(plan_dir, root)).strip()}

        Debt guidance:
        - Treat recurring debt as decision context, not background noise.
        - If the current unresolved flags overlap an escalated subsystem, prefer recommending holistic redesign over another point fix.
        """
    ).strip()


def _finalize_debt_block(plan_dir: Path, root: Path | None) -> str:
    watch_lines = _debt_watch_lines(plan_dir, root)
    return textwrap.dedent(
        f"""
        Debt watch items (do not make these worse):
        {json_dump(watch_lines).strip()}
        """
    ).strip()


def _plan_prompt(state: PlanState, plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
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

        {prep_block}

        {prep_instruction}

        {intent_and_notes_block(state)}

        Project directory:
        {project_dir}

        {clarification_block}

        Requirements:
        - If the engineering brief above already identifies the exact file, line, and fix, trust it. Verify with at most 1-2 file reads, then produce the plan. Do NOT re-explore the codebase for information the brief already provides.
        - If the brief is absent, incomplete, or says "skip", inspect the repository yourself before planning.
        - Produce a concrete implementation plan in markdown.
        - Define observable success criteria.
        - Use the `questions` field for ambiguities that would materially change implementation.
        - Use the `assumptions` field for defaults you are making so planning can proceed now.
        - Prefer cheap validation steps early.
        - Keep the plan proportional to the task. A 1-line fix needs a 2-step plan (apply fix + run tests), not a 5-step investigation.
        - If user notes answer earlier questions, incorporate them into the draft plan instead of re-asking them.
        - Fix the problem fully. Do not limit scope just to avoid breaking existing tests — update the tests too if needed.
        - Prefer the simplest, most direct fix. No fallbacks, type conversions, or defensive wrappers without concrete evidence they are needed.
        - If the task or issue hints suggest a specific approach, follow it. Only deviate with concrete counter-evidence.

        {PLAN_TEMPLATE}
        """
    ).strip()


def _prep_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    del plan_dir
    project_dir = Path(state["config"]["project_dir"])
    project_root = root if root is not None else project_dir
    return textwrap.dedent(
        f"""
        Prepare a concise engineering brief for the task below. This brief will be the primary context for all subsequent planning and execution.

        Task:
        {state["idea"]}

        Project: {project_dir}

        First, assess: does this task need codebase investigation?

        Set "skip": true if ALL of these are true:
        - The task names the exact file(s) to change
        - The required change is clearly described
        - No ambiguity about the approach

        Set "skip": false if ANY of these are true:
        - The task doesn't say which files to change
        - Multiple approaches seem possible
        - The task references concepts, APIs, or patterns you'd need to look up in the codebase
        - The task involves more than 2-3 files
        - There are hints or references that need investigation

        If skipping, leave everything else empty. The original task description will be used directly.
        If not skipping, fill in the brief:
        1. Search the codebase (Glob, Grep, Read) for relevant files and functions.
        2. If tests exist for the affected code, read them — they reveal what the fix must actually do, which may differ from what the task description suggests.
        3. Extract evidence from the task description — hints, references, error messages.
        4. Challenge the obvious path: if the task or hints point to a specific location, verify it's actually the right place. Trace the call chain — where does data flow? Where does it go wrong? The obvious file may be a symptom, not the root cause.
        5. If the task describes a bug or incorrect behavior, seriously consider whether it is a symptom of a larger issue. Before proposing a fix, trace the root cause. Ask: why does this happen? Could the same root cause produce other failures? Is the fix a patch on one case, or does it need to address an underlying gap? If the codebase has related functionality that is also incomplete or broken, note it — a narrow fix may not be enough.
        6. If you find that a suggested fix already exists in the code, say so explicitly — this means the root cause is elsewhere.
        7. Once you identify the function, parameter, or pattern that needs fixing, grep for ALL other usages of it in the codebase. If the same parameter is passed in 3 places, all 3 may need the fix. List every call site in relevant_code — do not stop at the first one.
        8. If the code has a `NotImplementedError`, `raise`, `TODO`, or explicit skip for certain inputs, and the bug involves those inputs, the fix likely needs to implement the missing functionality — not just patch around it. Flag this in the brief so the plan knows a larger change is needed.
        9. Look for existing helper functions, utilities, or patterns in the codebase that handle similar cases. If there is existing machinery (e.g., a merge function, a validation helper, a base class method), the fix should use it rather than reinventing.
        10. Before finalizing, ask: if I change this function, are there other callers that rely on its current behavior? A function called from multiple code paths may need different fixes for different callers — or a new method instead of modifying the existing one.
        11. List all usages as a numbered list (1. file:line — description, 2. file:line — description, etc.) so none are missed.
        12. Distill into a brief that adds value beyond the raw task description.

        Brief fields:
        - skip: true if no investigation needed, false if brief has useful content.
        - task_summary: What needs to be done, in 2-3 sentences.
        - key_evidence: Facts from the task and codebase not obvious from reading the task alone.
        - relevant_code: File paths and key functions found by searching.
        - test_expectations: Tests that verify the affected behavior.
        - constraints: What must not break.
        - suggested_approach: A concrete approach grounded in what you found.
        - estimated_scope: How big is this change? (e.g., "1-line fix in one file", "changes needed in 3 files", "new function/class required", "refactor of existing subsystem"). Be honest — if this needs significant new code, say so.

        IMPORTANT: After you finish searching and reading files, you MUST output the prep.json as your final message. Do not end with a tool call — end with the JSON object as plain text.
        """
    ).strip()


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
        - Preserve or improve success criteria quality.
        - Verify that the plan remains aligned with the user's original intent, not just internal plan quality.
        - Remove unjustified scope growth. If critique raised scope creep, narrow the plan back to the original idea unless the broader work is strictly required.
        - Maintain the structural template: H1 title, ## Overview, phase sections with numbered step sections, ## Execution Order or ## Validation Order.

        {PLAN_TEMPLATE}
        """
    ).strip()


def _research_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    del root
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    package_json_path = project_dir / "package.json"
    try:
        package_json = read_json(package_json_path)
        dependency_versions = {
            **package_json.get("dependencies", {}),
            **package_json.get("devDependencies", {}),
            **package_json.get("peerDependencies", {}),
            **package_json.get("optionalDependencies", {}),
        }
        package_json_block = textwrap.dedent(
            f"""
            package.json detected at:
            {package_json_path}

            Dependency and framework version hints:
            {json_dump(dependency_versions).strip()}
            """
        ).strip()
    except FileNotFoundError:
        package_json_block = "no package.json detected"

    return textwrap.dedent(
        f"""
        You are doing targeted documentation research for the plan that was just created.

        Project directory:
        {project_dir}

        {intent_and_notes_block(state)}

        Plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        {package_json_block}

        Your job is to find things the plan might be getting wrong OR missing based on current documentation.
        You are NOT validating the plan — you are a devil's advocate looking for problems, gaps, and outdated approaches.

        You MUST do FOUR searches minimum:

        Search 1 — CHECK WHAT THE TASK ASKS FOR:
        Read the Original Task carefully. Extract every technical term, API name, directive, file convention,
        and feature name that is NOT a standard well-known API. For each extracted term, search:
        "[framework] [version] [term]"
        e.g. "next.js 16 use cache directive", "next.js 16 proxy.ts", "next.js 16 unstable_instant"
        If the task names something the plan doesn't mention, that's likely a CRITICAL gap.

        Search 2 — CHECK WHAT THE PLAN DOES:
        For each framework-specific API/pattern in the plan, search for "[framework] [version] [API]"
        e.g. "next.js 16 unstable_cache", "next.js 16 force-dynamic"
        If the docs say this API is deprecated or replaced, that's a CRITICAL consideration.

        Search 3 — CHECK WHAT THE PLAN IS MISSING:
        Search for "[framework] [version] [task type] best practices" or "checklist" or "migration guide"
        e.g. "next.js 16 app router migration checklist", "next.js 16 caching best practices"
        Compare the checklist against the plan. Every recommended step the plan omits is a consideration.

        Search 4 — CHECK FOR BREAKING CHANGES:
        Search for "[framework] [version] breaking changes" or "new features"
        e.g. "next.js 16 breaking changes", "next.js 16 new APIs"
        If the plan uses patterns that changed in this version, that's a CRITICAL consideration.

        After all searches, re-read the plan one final time and compare it against the Original Task:
        - Does the task name a specific API or pattern that the plan does NOT use? Flag as CRITICAL.
        - Is there ANY API the plan uses that the docs say should be done differently?
        - Is there ANY required config flag or file that the plan doesn't mention?
        - Is there ANY naming convention (file names, exports) that the plan gets wrong?

        Severity rules:
        - CRITICAL: docs recommend a DIFFERENT API/approach than the plan uses, OR a required config/file is missing
        - IMPORTANT: docs show a best practice the plan doesn't follow
        - MINOR: style or preference difference

        Output:
        - Each consideration must have a clear point, detail, and severity
        - The summary must list key findings — NEVER say "everything looks correct" unless you truly found zero issues
        - If you found issues, the summary should start with "Found N issues:" followed by a brief list
        """
    ).strip()


def _render_research_block(plan_dir: Path) -> tuple[str, str]:
    """Render research findings for injection into critique/review prompts.
    Returns (research_block, research_instruction)."""
    research_path = plan_dir / "research.json"
    if not research_path.exists():
        return "", ""
    research = read_json(research_path)
    considerations = research.get("considerations", [])
    severity_order = {"critical": 0, "important": 1, "minor": 2}
    considerations.sort(key=lambda c: severity_order.get(c.get("severity", "minor"), 2))
    if considerations:
        consideration_lines = []
        for c in considerations:
            severity = c.get("severity", "minor")
            point = c.get("point") or c.get("topic") or c.get("name") or ""
            detail = c.get("detail") or c.get("issue") or c.get("description") or ""
            recommendation = c.get("recommendation") or ""
            source = c.get("source") or ""
            line = f"- [{severity.upper()}] {point}"
            if detail:
                line += f"\n  {detail}"
            if recommendation:
                line += f"\n  Recommendation: {recommendation}"
            if source:
                line += f"\n  Source: {source}"
            consideration_lines.append(line)
        considerations_block = "\n".join(consideration_lines)
    else:
        considerations_block = "- No noteworthy considerations found."
    research_block = textwrap.dedent(
        f"""
        A researcher recommended you consider these points in executing the task:

        {considerations_block}
        """
    ).strip()
    research_instruction = (
        "- The research considerations above are based on current documentation searches. "
        "Any item marked CRITICAL or IMPORTANT should be flagged if the plan doesn't address it."
    )
    return research_block, research_instruction


def _render_prep_block(plan_dir: Path) -> tuple[str, str]:
    prep_path = plan_dir / "prep.json"
    if not prep_path.exists():
        return "", ""
    prep = read_json(prep_path)
    # If prep decided to skip (task was simple enough), return empty —
    # downstream phases will use the original task description as-is
    if prep.get("skip", False):
        return "", ""
    prep = read_json(prep_path)

    def _cell(value: object) -> str:
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        text = str(value).strip()
        if not text:
            return "-"
        return text.replace("|", "\\|").replace("\n", " ")

    task_summary = str(prep.get("task_summary", "")).strip() or "No task summary provided."

    evidence_items = prep.get("key_evidence", [])
    if isinstance(evidence_items, list) and evidence_items:
        evidence_lines = []
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            point = str(item.get("point", "")).strip() or "Unspecified evidence"
            source = str(item.get("source", "")).strip() or "unspecified source"
            relevance = str(item.get("relevance", "")).strip() or "unspecified relevance"
            evidence_lines.append(f"- {point} (source: {source}; relevance: {relevance})")
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "- No key evidence captured."
    else:
        evidence_block = "- No key evidence captured."

    relevant_code_items = prep.get("relevant_code", [])
    if isinstance(relevant_code_items, list) and relevant_code_items:
        code_lines = [
            "| File | Functions | Why |",
            "| --- | --- | --- |",
        ]
        for item in relevant_code_items:
            if not isinstance(item, dict):
                continue
            code_lines.append(
                f"| {_cell(item.get('file_path', ''))} | {_cell(item.get('functions', []))} | {_cell(item.get('why', ''))} |"
            )
        relevant_code_block = "\n".join(code_lines) if len(code_lines) > 2 else "- No directly relevant code captured."
    else:
        relevant_code_block = "- No directly relevant code captured."

    test_expectation_items = prep.get("test_expectations", [])
    if isinstance(test_expectation_items, list) and test_expectation_items:
        test_lines = []
        for item in test_expectation_items:
            if not isinstance(item, dict):
                continue
            test_id = str(item.get("test_id", "")).strip() or "unnamed test"
            status = str(item.get("status", "")).strip() or "unknown"
            what_it_checks = str(item.get("what_it_checks", "")).strip() or "No description provided."
            test_lines.append(f"- [{status}] {test_id}: {what_it_checks}")
        test_expectations_block = "\n".join(test_lines) if test_lines else "- No explicit test expectations captured."
    else:
        test_expectations_block = "- No explicit test expectations captured."

    constraints = prep.get("constraints", [])
    if isinstance(constraints, list) and constraints:
        constraint_lines = [f"- {str(item).strip()}" for item in constraints if str(item).strip()]
        constraints_block = "\n".join(constraint_lines) if constraint_lines else "- No explicit constraints captured."
    else:
        constraints_block = "- No explicit constraints captured."

    suggested_approach = str(prep.get("suggested_approach", "")).strip() or "No suggested approach provided."

    prep_block = textwrap.dedent(
        f"""
        Engineering brief produced from the codebase and task details:

        ### Task Summary
        {task_summary}

        ### Key Evidence
        {evidence_block}

        ### Relevant Code
        {relevant_code_block}

        ### Test Expectations
        {test_expectations_block}

        ### Constraints
        {constraints_block}

        ### Suggested Approach
        {suggested_approach}
        """
    ).strip()
    prep_instruction = "The engineering brief above was produced by analyzing the codebase. Use it as primary context."
    return prep_block, prep_instruction


def _critique_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
    research_block, research_instruction = _render_research_block(plan_dir)
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
    debt_block = _planning_debt_block(plan_dir, root)
    return textwrap.dedent(
        f"""
        You are an independent reviewer. Critique the plan against the actual repository.

        Project directory:
        {project_dir}

        {prep_block}

        {prep_instruction}

        {intent_and_notes_block(state)}

        Plan:
        {latest_plan}

        Plan metadata:
        {json_dump(latest_meta).strip()}

        Plan structure warnings from validator:
        {json_dump(structure_warnings).strip()}

        Existing flags:
        {json_dump(unresolved).strip()}

        {research_block}

        {debt_block}

        Requirements:
        - First, assess whether the plan targets the correct root cause. If the proposed fix already exists in the codebase, or if the plan contradicts evidence you find, flag this as CRITICAL — the plan needs a fundamentally different approach, not just adjustments.
        - Consider whether the plan is at the right level of abstraction.
        - Reuse existing flag IDs when the same concern is still open.
        - `verified_flag_ids` should list previously addressed flags that now appear resolved.
        - Focus on concrete issues that would cause real problems.
        - Robustness level: {robustness}. {robustness_critique_instruction(robustness)}
        - Verify that the plan remains aligned with the user's original intent.
        - Verify that the plan follows the expected structure: one H1 title, `## Overview`, numbered step sections (`### Step N:` under `## Phase` headers, or flat `## Step N:`) with file references and numbered substeps, plus `## Execution Order` or `## Validation Order`.
        - Missing required sections or step coverage (for example: no H1, no `## Overview`, or no step sections at all) should be flagged as category `completeness` with severity_hint `likely-significant`.
        - Structural formatting within steps (for example: prose instead of numbered substeps, or a missing file reference inside an otherwise actionable step) should usually be category `completeness` with severity_hint `likely-minor` because the executor can still follow the instructions.
        - Ask whether the plan is the simplest approach that solves the stated problem, whether it could use fewer steps or less machinery, and whether it introduces unnecessary complexity.
        - If the task hints suggest a specific approach and the plan deviates, flag it. The issue author often knows the correct fix.
        - If the plan limits scope to avoid breaking tests, flag as a potential under-fix.
        - Over-engineering concerns should use category `maintainability`, should usually prefix the concern with "Over-engineering:", and should scale severity_hint to the practical impact.
        - Flag scope creep explicitly when the plan grows beyond the original idea or recorded user notes. Use the phrase "Scope creep:" in the concern.
        - Assign severity_hint carefully. Implementation details the executor will naturally resolve should usually be `likely-minor`.
        {research_instruction}
        """
    ).strip()


def _gate_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
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
            "evidence": flag.get("evidence", ""),
            "category": flag["category"],
            "severity": flag.get("severity", "unknown"),
            "status": flag["status"],
            "weight": flag.get("weight"),
        }
        for flag in unresolved
    ]
    robustness = configured_robustness(state)
    debt_block = _gate_debt_block(plan_dir, root)
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

        {debt_block}

        Robustness level:
        {robustness}

        Requirements:
        - Decide exactly one of: PROCEED, ITERATE, ESCALATE.
        - Use the weighted score, flag details (including the `evidence` field — not just `concern`), plan delta, recurring critiques, loop summary, and preflight results as judgment context, not as a fixed decision table.
        - Unresolved correctness flags (wrong root cause, missing code locations, under-scoped fix) should block PROCEED unless you can explain with evidence why the flag is wrong.
        - PROCEED when execution should move forward now.
        - ITERATE when revising the plan is the best next move.
        - ESCALATE when the loop is stuck, churn is recurring, or user intervention is needed.
        - `signals_assessment` should summarize the score trajectory, plan delta, recurring critiques, unresolved flag weight, and preflight posture in one compact paragraph.
        - Put any cautionary notes in `warnings`.
        - Populate `settled_decisions` with design choices that are now settled and should carry into review without being re-litigated. Return `[]` when there are no such decisions.
        - When recommending `PROCEED` with unresolved flags, populate `accepted_tradeoffs` with one entry per accepted unresolved flag using:
          - `flag_id`: the exact flag ID
          - `subsystem`: a semantically meaningful subsystem tag like `timeout-recovery` or `execute-paths`, not the flag category
          - `concern`: the accepted limitation phrased clearly
          - `rationale`: why proceeding is still acceptable
        - When recommending `ITERATE` or `ESCALATE`, return `"accepted_tradeoffs": []`.
        - Example output shape:
        ```json
        {{
          "recommendation": "PROCEED",
          "rationale": "The remaining issues are executor-level details rather than planning blockers.",
          "signals_assessment": "Weighted score is falling, plan delta is stabilizing, and preflight remains clean.",
          "warnings": ["Double-check FLAG-005 while executing."],
          "accepted_tradeoffs": [
            {{
              "flag_id": "FLAG-005",
              "subsystem": "timeout-recovery",
              "concern": "Timeout recovery: retry backoff remains basic for this pass.",
              "rationale": "The plan contains enough guardrails to execute safely, and the remaining gap is a known tradeoff rather than a blocker."
            }}
          ],
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
            "evidence": f.get("evidence", ""),
            "status": f["status"],
            "severity": f.get("severity", "unknown"),
        }
        for f in registry["flags"]
    ]


def _finalize_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    gate = read_json(plan_dir / "gate.json")
    flag_registry = load_flag_registry(plan_dir)
    critique_history = _collect_critique_summaries(plan_dir, state["iteration"])
    debt_block = _finalize_debt_block(plan_dir, root)
    return textwrap.dedent(
        f"""
        You are preparing an execution-ready briefing document from the approved plan.

        Project directory:
        {project_dir}

        {prep_block}

        {prep_instruction}

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

        {debt_block}

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
        - Keep the task count proportional to the work. A simple 1-2 file fix should be 2 tasks: (1) apply the fix, (2) run tests. Do NOT create separate "inspect" or "read" tasks for simple changes — the executor can read and fix in one step. Only create more tasks when the work has genuinely independent stages.
        - The FINAL task MUST always be to run tests and verify the changes work. If specific test IDs or commands are mentioned in the original task, include them. Otherwise, the executor should find and run the tests most relevant to the files changed. If any test fails, read the error, fix the code, and re-run until they pass. Do NOT create new test files — run the project's existing test suite.
        """
    ).strip()


def _execute_prompt(state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    project_dir = Path(state["config"]["project_dir"])
    prep_block, prep_instruction = _render_prep_block(plan_dir)
    # Codex execute often cannot write back into plan_dir during --full-auto, so
    # checkpoint instructions must stay best-effort rather than mandatory.
    finalize_path = str(plan_dir / "finalize.json")
    checkpoint_path = str(plan_dir / "execution_checkpoint.json")
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
    debt_watch_items = _debt_watch_lines(plan_dir, root)
    if debt_watch_items:
        nudge_lines.append("Debt watch items (do not make these worse):")
        for item in debt_watch_items:
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
        review_data = read_json(plan_dir / "review.json") if (plan_dir / "review.json").exists() else {}
        review_issues = review_data.get("issues", [])
        issue_list = "\n".join(f"  - {issue}" for issue in review_issues) if review_issues else "  (see review.json above for details)"
        rerun_guidance = (
            "REWORK REQUIRED: all tasks are already tracked but the reviewer kicked this back.\n"
            f"Review issues to fix:\n{issue_list}\n\n"
            "You MUST make code changes to address each issue — do not return success without modifying files. "
            "For each issue, either fix it and list the file in files_changed, or explain in deviations why no change is needed with line-level evidence. "
            "Return task_updates for all tasks with updated evidence."
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

        Requirements:
        - Implement the intent, not just the text.
        - Adapt if repository reality contradicts the plan.
        - Report deviations explicitly.
        - Do not over-engineer beyond what the plan prescribes — no str() wraps, .get() fallbacks, or try/except guards unless the plan called for them or you found a concrete reason.
        - Do NOT modify test files. Do NOT fix unrelated issues you encounter (e.g., dependency compatibility, Python version workarounds). Only change files directly needed for the task.
        - If you cannot verify your changes (tests missing or unrunnable), treat this as high risk — re-examine your implementation with extra scrutiny instead of accepting it on faith.
        - If tests fail, read the traceback carefully. Diagnose WHY — don't just retry. Common causes: wrong function/method used, missing import, incorrect type, edge case not handled. Fix the root cause, then re-run.
        - Output concrete files changed and commands run. `files_changed` means files you WROTE or MODIFIED — not files you read or verified. Only list files where you made actual edits.
        - Use the tasks in `finalize.json` as the execution boundary.
        - Best-effort progress checkpointing: if `{checkpoint_path}` is writable, then after each completed task read the full file, update that task's `status`, `executor_notes`, `files_changed`, and `commands_run`, and write the full file back. Do NOT write to `finalize.json` directly — the harness owns that file.
        - Best-effort sense-check checkpointing: if `{checkpoint_path}` is writable, then after each sense check acknowledgment read the full file again, update that sense check's `executor_note`, and write the full file back.
        - Always use full read-modify-write updates for `{checkpoint_path}` instead of partial edits. If the sandbox blocks writes, continue execution and rely on the structured output below.
        - Structured output remains the authoritative final summary for this step. Disk writes are progress checkpoints for timeout recovery only.
        - Return `task_updates` with one object per completed or skipped task.
        - Return `sense_check_acknowledgments` with one object per sense check.
        - Keep `executor_notes` verification-focused: explain why your changes are correct. The diff already shows what changed; notes should cover edge cases caught, expected behaviors confirmed, or design choices made.
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
              "executor_notes": "Caught the empty-strings edge case while checking execution evidence: blank `commands_run` entries still leave the task uncovered, so the missing-evidence guard behaves correctly.",
              "files_changed": ["megaplan/handlers.py"],
              "commands_run": ["pytest tests/test_megaplan.py -k execute"]
            }},
            {{
              "task_id": "T7",
              "status": "done",
              "executor_notes": "Confirmed the happy path still records task evidence after the prompt updates by rerunning focused tests and checking the tracked task summary stayed intact.",
              "files_changed": ["megaplan/prompts.py"],
              "commands_run": ["pytest tests/test_prompts.py -k review"]
            }},
            {{
              "task_id": "T8",
              "status": "done",
              "executor_notes": "Kept the rubber-stamp thresholds centralized in evaluation so sense checks and reviewer verdicts share one policy entry point while still using different strictness levels.",
              "files_changed": ["megaplan/evaluation.py"],
              "commands_run": ["pytest tests/test_evaluation.py -k rubber_stamp"]
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
    batch_tasks = [tasks_by_id[task_id] for task_id in batch_task_ids if task_id in tasks_by_id]
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
    batch_sense_check_ids = [sense_check["id"] for sense_check in batch_sense_checks if isinstance(sense_check.get("id"), str)]
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
        else "Note: User explicitly approved this plan at the gate checkpoint."
        if state["meta"].get("user_approved_gate")
        else "Note: Review mode is enabled. Execute should only be running after explicit gate approval."
    )
    debt_watch_items = _debt_watch_lines(plan_dir, root)
    debt_watch_block = (
        "\n".join(["Debt watch items (do not make these worse):", *[f"- {item}" for item in debt_watch_items]])
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


def _review_robustness_instruction(robustness: str) -> list[str]:
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
    review_research_block, review_research_instruction = _render_research_block(plan_dir)
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

        {review_research_block}

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
        {review_research_instruction}
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
    review_research_block, review_research_instruction = _render_research_block(plan_dir)
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

        {review_research_block}

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
        {review_research_instruction}
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


_PromptBuilder = Callable[..., str]

_CLAUDE_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "prep": _prep_prompt,
    "research": _research_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": _review_claude_prompt,
}

_CODEX_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "prep": _prep_prompt,
    "research": _research_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": _review_codex_prompt,
}


_HERMES_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "prep": _prep_prompt,
    "research": _research_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": _review_claude_prompt,  # Hermes routes through Claude models on OpenRouter
}


def create_claude_prompt(step: str, state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    builder = _CLAUDE_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Claude step '{step}'")
    if step in {"prep", "research", "critique", "gate", "finalize", "execute"}:
        return builder(state, plan_dir, root=root)
    return builder(state, plan_dir)


def create_codex_prompt(step: str, state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    builder = _CODEX_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Codex step '{step}'")
    if step in {"prep", "research", "critique", "gate", "finalize", "execute"}:
        return builder(state, plan_dir, root=root)
    return builder(state, plan_dir)


def create_hermes_prompt(step: str, state: PlanState, plan_dir: Path, root: Path | None = None) -> str:
    builder = _HERMES_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Hermes step '{step}'")
    if step in {"prep", "research", "critique", "gate", "finalize", "execute"}:
        return builder(state, plan_dir, root=root)
    return builder(state, plan_dir)
