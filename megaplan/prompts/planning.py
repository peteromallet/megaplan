"""Planning-phase prompt builders."""

from __future__ import annotations

import textwrap
from pathlib import Path

from megaplan._core import (
    intent_and_notes_block,
    json_dump,
    latest_plan_meta_path,
    latest_plan_path,
    read_json,
)
from megaplan.types import PlanState

from ._shared import _render_prep_block

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
        - If the engineering brief suggests an approach, use it as your starting hypothesis — but before committing, consider if there's a simpler or more fundamental fix. The brief is well-researched input, not a final answer.
        - If the brief is absent, incomplete, or says "skip", inspect the repository yourself before planning.
        - Stay focused on the requested idea. If repo exploration surfaces unrelated issues or docs, ignore them and return to the task.
        - Prefer source code, tests, and directly relevant config files. Avoid `.megaplan/`, prior plan artifacts, and unrelated `docs/` or ops/deployment material unless the task explicitly depends on them.
        - Stop exploring once you have enough evidence to name the concrete touch points and validation path. Do not keep browsing after you can write the plan.
        - Produce a concrete implementation plan in markdown.
        - Define observable success criteria as objects with `criterion` (string) and `priority` (`must`, `should`, or `info`):
          - `must` — hard gate. The reviewer will block on failure. Use for correctness, functional requirements, and verifiable outcomes (e.g., "all existing tests pass", "API returns 200 for valid input"). Every `must` criterion must have a clear yes/no answer.
          - `should` — quality target. The reviewer flags but does not block. Use for subjective goals, numeric guidelines, and best-effort improvements (e.g., "file under ~300 lines", "no deeply nested conditionals", "each function has a single responsibility").
          - `info` — documented for humans, reviewer skips. Use for criteria that cannot be verified in this pipeline (e.g., "13 manual smoke tests pass", "stakeholder sign-off obtained").
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
    del root
    project_dir = Path(state["config"]["project_dir"])
    output_path = plan_dir / "prep.json"
    return textwrap.dedent(
        f"""
        Prepare a concise engineering brief for the task below. This brief will be the primary context for all subsequent planning and execution.

        Task:
        {state["idea"]}

        Project: {project_dir}
        Output file: {output_path}

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

        """
    ).strip()

