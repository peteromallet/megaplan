# Implementation Plan: Heavy-Mode Parallel Review Phase

## Overview

Add a parallel review phase for heavy robustness mode that runs 3 mechanical pre-checks and 4 focused LLM checks against the original issue text. This mirrors how `handle_critique` replaces the monolithic critique with parallel checks for heavy+hermes mode.

**Key architectural decision (revised):** For heavy mode with hermes available, parallel review checks REPLACE the monolithic review worker — identical to the critique pattern at `handlers.py:704-710`. For heavy mode without hermes (codex/claude), or for standard/light mode, the existing monolithic review runs unchanged. This resolves the file-overwrite, verdict-integration, and agent-routing issues from v1.

**Key files:**
- `megaplan/parallel_critique.py` (260 lines) — ThreadPoolExecutor pattern to extract
- `megaplan/parallel_checks.py` (NEW) — generic `run_parallel_checks` primitive
- `megaplan/review_checks.py` (NEW) — review check specs and mechanical pre-checks
- `megaplan/prompts/parallel_review.py` (NEW) — per-check prompt builders
- `megaplan/handlers.py:1105` — `handle_review` (insertion point)
- `megaplan/handlers.py:1065` — `_resolve_review_outcome` (iteration cap)
- `megaplan/types.py:287` — `DEFAULTS` (add review concurrency)
- `megaplan/check_audit.py` (NEW) — CLI tool
- `megaplan/cli.py:466` — `build_parser` (add subcommand)
- `megaplan/cli.py:602` — `COMMAND_HANDLERS` (add entry)

**Constraint summary:**
- Standard/light must be byte-identical after — the heavy branch is a pure addition
- Review checks anchor to `intent_and_notes_block`, NOT `finalize.json`
- Template-file pattern (not freeform JSON), same as critique
- Each check in its own fresh `ThreadPoolExecutor` session
- Review rework goes through existing gate loop, NOT a separate loop
- 2-iteration hard cap for parallel-review-originated rework

## Phase 1: Extract `run_parallel_checks` Primitive

### Step 1: Create `megaplan/parallel_checks.py` — generic parallel runner
**Scope:** Medium

1. **Extract** the ThreadPoolExecutor orchestration from `parallel_critique.py` into a new module `megaplan/parallel_checks.py`.

2. **Define** the generic signature with a `run_check_fn` callback (the key correction from v1 — this was missing from the concrete signature):

```python
from typing import Any, Callable
from pathlib import Path
from megaplan.types import PlanState
from megaplan.workers import WorkerResult

RunCheckFn = Callable[..., tuple]

def run_parallel_checks(
    state: PlanState,
    plan_dir: Path,
    *,
    root: Path,
    checks: tuple[dict[str, Any], ...],
    run_check_fn: RunCheckFn,
    max_concurrent: int | None = None,
    concurrency_config_key: str = "orchestration.max_critique_concurrency",
    empty_payload: dict[str, Any] | None = None,
    result_aggregator: Callable[[list[tuple | None]], dict[str, Any]] | None = None,
) -> WorkerResult:
```

3. **The primitive handles:**
   - Early return for empty `checks` tuple (using `empty_payload` or a default `{"checks": []}`)
   - `ThreadPoolExecutor` with concurrency from `get_effective()` via `concurrency_config_key`
   - Stdout/stderr swap (L203-230 pattern from `parallel_critique.py`)
   - Submitting `run_check_fn(index, check, state=state, plan_dir=plan_dir, root=root, ...)` per check
   - Collecting results via `as_completed`, preserving order by `index`
   - Cost/token aggregation
   - Result aggregation via `result_aggregator` callback (or a default that collects ordered check payloads)

4. **`run_check_fn` contract:** Each callable receives `(index, check, *, state, plan_dir, root, **kwargs)` and returns a tuple. The primitive doesn't prescribe the tuple shape — the `result_aggregator` callback interprets it. This lets critique return `(index, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt)` and review return a different shape.

5. **`result_aggregator` contract:** Receives `list[tuple | None]` (ordered by index, None means missing) and returns `dict[str, Any]` for the `WorkerResult.payload`. If not provided, a default aggregator builds `{"checks": [item[1] for item in results]}`.

6. **Add** `orchestration.max_review_concurrency` to `DEFAULTS` in `megaplan/types.py:287-293` and `_SETTABLE_NUMERIC` at L295-300, defaulting to `2` (same as critique). This resolves `all_locations-1`.

### Step 2: Refactor `megaplan/parallel_critique.py` onto the primitive
**Scope:** Small

1. **Keep** `_run_check` in `parallel_critique.py` as the critique-specific per-thread callback. This preserves the monkeypatch target at `megaplan.parallel_critique._run_check` (resolves `all_locations-7`, `callers-4`).

2. **Replace** the body of `run_parallel_critique` with a call to `run_parallel_checks`:

```python
def run_parallel_critique(state, plan_dir, *, root, model, checks, max_concurrent=None):
    from megaplan.parallel_checks import run_parallel_checks

    schema = read_json(schemas_root(root) / STEP_SCHEMA_FILENAMES["critique"])
    project_dir = Path(state["config"]["project_dir"])

    def critique_run_check(index, check, **kwargs):
        return _run_check(index, check, state=state, plan_dir=plan_dir,
                         root=root, model=model, schema=schema, project_dir=project_dir)

    return run_parallel_checks(
        state, plan_dir, root=root, checks=checks,
        run_check_fn=critique_run_check,
        max_concurrent=max_concurrent,
        concurrency_config_key="orchestration.max_critique_concurrency",
        empty_payload={"checks": [], "flags": [], "verified_flag_ids": [], "disputed_flag_ids": []},
        result_aggregator=_critique_result_aggregator,
    )
```

3. **Define** `_critique_result_aggregator(results)` that performs the existing `_merge_unique` logic for verified/disputed flag IDs (moved from `run_parallel_critique` L232-245).

4. **Verify** existing `test_parallel_critique.py` tests pass unchanged — `_run_check` stays at its current import path.

## Phase 2: Review Check Definitions and Mechanical Pre-Checks

### Step 3: Create `megaplan/review_checks.py` — check specs and mechanical pre-checks
**Scope:** Medium

1. **Define** `ReviewCheckSpec(TypedDict)` with fields: `id`, `question`, `guidance`, `category`.

2. **Define** the 4 LLM check specs with explicit guidance text:
```python
REVIEW_CHECKS: Final[tuple[ReviewCheckSpec, ...]] = (
    {
        "id": "coverage",
        "question": "Does the diff address every concrete symptom and example in the original issue?",
        "guidance": (
            "Enumerate each concrete symptom, error message, test case, or example from the issue text. "
            "For each one, verify the diff contains a change that addresses it. "
            "Flag any symptom that is not covered by any change in the diff."
        ),
        "category": "completeness",
    },
    {
        "id": "placement",
        "question": "Is the fix at the correct architectural layer?",
        "guidance": (
            "Check whether the change targets the right module, class, or function. "
            "If the issue is in library code but the fix is in CLI code (or vice versa), flag it. "
            "Check that the fix isn't patching downstream symptoms instead of the root cause."
        ),
        "category": "correctness",
    },
    {
        "id": "parity",
        "question": "Does the new code match sibling-implementation patterns in the codebase?",
        "guidance": (
            "Find similar existing patterns in the same codebase (e.g., how neighboring functions handle errors, "
            "how sibling classes implement the same interface). "
            "Flag divergences in style, error handling, naming, or structure."
        ),
        "category": "maintainability",
    },
    {
        "id": "simplicity",
        "question": "Is this the smallest fix that works?",
        "guidance": (
            "Check if the diff contains unnecessary refactoring, over-engineering, dead code, "
            "or scope creep beyond what the issue requires. "
            "The simplest correct fix is preferred. "
            "Flag any change that isn't strictly required to address the issue."
        ),
        "category": "correctness",
    },
)
```

3. **Define** `review_checks_for_robustness(robustness: str) -> tuple[ReviewCheckSpec, ...]`: returns `REVIEW_CHECKS` for heavy, empty tuple otherwise.

4. **Define** `MechanicalCheckResult(TypedDict)`: `{"id": str, "passed": bool, "detail": str}`.

5. **Implement** 3 mechanical pre-checks with concrete heuristics:

   **`source_touch(plan_dir, project_dir) -> MechanicalCheckResult`**
   - Read `execution.json`, check `files_changed` contains at least one non-test source file (file path not matching `test_*.py` or `*_test.py` or `tests/` directory).
   - Pass if at least one non-test source file is in the list.
   - Fail with detail listing only test files found.

   **`diff_size_sanity(plan_dir, project_dir) -> MechanicalCheckResult`**
   - Concrete heuristic: Run `git diff --stat` in project_dir, parse total inserted+deleted lines.
   - If diff has 0 lines: fail ("No diff detected — execution may not have made changes").
   - If diff has >500 lines: fail with warning ("Diff is {N} lines, suggesting scope creep beyond a targeted fix").
   - Otherwise: pass. This is a fixed-threshold approach (not issue-length-based), avoiding the undefined proxy from v1.

   **`dead_guard(plan_dir, project_dir) -> MechanicalCheckResult`**
   - Best-effort grep-based heuristic with explicit limitations documented:
     - Parse `git diff` output for new `if ` or `try:` lines added (lines starting with `+` containing `if ` or `try:`).
     - For each new guard, extract the enclosing function name (search backward from the diff hunk for `def `).
     - Grep the project for call sites of that function name.
     - If a function with a new guard has zero call sites in the project: fail ("New guard in function {name} has no callers — may be dead code").
     - Otherwise: pass with detail noting how many call sites found.
     - If no new guards detected: auto-pass ("No new if/try guards detected in diff").
   - Document that this is a heuristic: it checks function-level reachability, not branch-level reachability. False negatives are expected. The purpose is to catch completely unreachable guards, not subtle reachability issues.

6. **Define** `run_mechanical_prechecks(plan_dir: Path, project_dir: Path) -> list[MechanicalCheckResult]` — runs all 3 and returns results.

## Phase 3: Review Check Prompts and Templates

### Step 4: Create `megaplan/prompts/parallel_review.py` — per-check prompt builders
**Scope:** Medium

1. **Create** `single_check_review_prompt(state, plan_dir, root, check, template_path) -> str` following the exact pattern from `prompts/critique.py:single_check_critique_prompt` (L309-353).

2. **Prompt content** — anchor on `intent_and_notes_block(state)` as the PRIMARY section, with `finalize.json` available as SECONDARY context (the monolithic review prompt at `review.py:138-143` already includes both; we're emphasizing issue text more heavily):

```python
def single_check_review_prompt(state, plan_dir, root, check, template_path):
    project_dir = Path(state["config"]["project_dir"])
    intent = intent_and_notes_block(state)
    diff_summary = collect_git_diff_summary(project_dir)
    execution = read_json(plan_dir / "execution.json")
    # finalize.json as secondary context (available but not primary)
    finalize_data = read_json(plan_dir / "finalize.json")
    # ... build prompt with intent first, diff summary, execution context
```

3. **Prompt structure:**
   - `{intent_and_notes_block(state)}` — first and most prominent section
   - Git diff summary
   - Execution summary from `execution.json` (files changed, commands run)
   - Finalize data from `finalize.json` (secondary — task list for context)
   - The check's question and guidance (from the check spec)
   - Template file instructions (same pattern as critique)

4. **Create** `write_single_review_check_template(plan_dir, state, check, output_name) -> Path`:

```python
def write_single_review_check_template(plan_dir, state, check, output_name):
    template = {
        "checks": [{
            "id": check["id"],
            "question": check["question"],
            "guidance": check.get("guidance", ""),
            "findings": [],
        }],
        "rework_items": [],
    }
    output_path = plan_dir / output_name
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return output_path
```

Note: includes `guidance` in the template (resolves `callers-5` — v1 omitted it).

5. **Create** `_run_review_check` in `review_checks.py` — the review-specific per-thread callback, parallel to `parallel_critique._run_check`:

```python
def _run_review_check(
    index: int,
    check: dict[str, Any],
    *,
    state: PlanState,
    plan_dir: Path,
    root: Path,
    model: str | None,
    schema: dict[str, Any],
    project_dir: Path,
) -> tuple[int, dict[str, Any], float, int, int, int]:
    """Run a single review check in its own thread."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from megaplan.hermes_worker import _toolsets_for_phase, parse_agent_output, clean_parsed_payload
    from megaplan.prompts.parallel_review import single_check_review_prompt, write_single_review_check_template
    from megaplan.key_pool import resolve_model as _resolve_model

    output_path = write_single_review_check_template(
        plan_dir, state, check, f"review_check_{check['id']}.json"
    )
    prompt = single_check_review_prompt(state, plan_dir, root, check, output_path)
    resolved_model, agent_kwargs = _resolve_model(model)

    agent = AIAgent(
        model=resolved_model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=_toolsets_for_phase("review"),
        session_id=str(uuid.uuid4()),
        session_db=SessionDB(),
        max_tokens=8192,
        **agent_kwargs,
    )
    agent._print_fn = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)

    result = agent.run_conversation(user_message=prompt)
    payload, raw_output = parse_agent_output(
        agent, result,
        output_path=output_path,
        schema=schema,
        step="review",
        project_dir=project_dir,
        plan_dir=plan_dir,
    )
    clean_parsed_payload(payload, schema, "review")

    check_findings = payload.get("checks", [{}])[0] if payload.get("checks") else {}
    rework_items = payload.get("rework_items", [])
    cost = float(result.get("estimated_cost_usd", 0.0) or 0.0)

    return (
        index,
        {
            "check": check_findings,
            "rework_items": rework_items,
            "flagged": any(f.get("flagged", False) for f in check_findings.get("findings", [])),
        },
        cost,
        int(result.get("prompt_tokens", 0) or 0),
        int(result.get("completion_tokens", 0) or 0),
        int(result.get("total_tokens", 0) or 0),
    )
```

## Phase 4: Wire Parallel Review into `handle_review`

### Step 5: Modify `megaplan/handlers.py` — heavy-mode parallel review branch
**Scope:** Medium

**Critical architectural decision:** For heavy+hermes, parallel review REPLACES the monolithic `_run_worker("review", ...)` — identical to how `handle_critique` at L704-710 replaces monolithic critique with `run_parallel_critique`. This resolves three v1 issues at once:
- File overwrite (`all_locations-2`): no competing writes to review.json
- Verdict integration (`correctness-2`, `scope-4`): parallel results directly form the payload
- LLM seeing parallel findings (`correctness-8`): no separate monolithic LLM to be out of sync

**Implementation in `handle_review` (L1105):**

```python
def handle_review(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "review", {STATE_EXECUTED})
    robustness = configured_robustness(state)

    # Resolve agent for this step
    agent_type, mode, refreshed, model = resolve_agent_mode("review", args)

    # Heavy mode: try parallel review checks (REPLACES monolithic, like critique)
    if robustness == "heavy" and agent_type == "hermes":
        try:
            from megaplan.review_checks import (
                review_checks_for_robustness,
                run_mechanical_prechecks,
                _run_review_check,
                _review_result_aggregator,
            )
            from megaplan.parallel_checks import run_parallel_checks

            project_dir = Path(state["config"]["project_dir"])
            active_review_checks = review_checks_for_robustness(robustness)

            # 1. Mechanical pre-checks (no LLM)
            mechanical_results = run_mechanical_prechecks(plan_dir, project_dir)
            mechanical_failures = [r for r in mechanical_results if not r["passed"]]

            # 2. Parallel LLM checks
            from megaplan._core import read_json, schemas_root
            from megaplan.workers import STEP_SCHEMA_FILENAMES
            schema = read_json(schemas_root(root) / STEP_SCHEMA_FILENAMES["review"])

            def review_run_check(index, check, **kwargs):
                return _run_review_check(
                    index, check,
                    state=state, plan_dir=plan_dir, root=root,
                    model=model, schema=schema, project_dir=project_dir,
                )

            worker = run_parallel_checks(
                state, plan_dir, root=root, checks=active_review_checks,
                run_check_fn=review_run_check,
                concurrency_config_key="orchestration.max_review_concurrency",
                result_aggregator=_review_result_aggregator,
            )

            # 3. Merge mechanical pre-check results into payload
            worker.payload["mechanical_prechecks"] = mechanical_results

            # 4. If any mechanical pre-check failed, force needs_rework
            if mechanical_failures:
                worker.payload.setdefault("rework_items", []).extend([
                    {
                        "task_id": f"mech_{r['id']}",
                        "issue": r["detail"],
                        "expected": "All mechanical pre-checks pass",
                        "actual": f"Failed: {r['id']}",
                    }
                    for r in mechanical_failures
                ])
                worker.payload["review_verdict"] = "needs_rework"

            # 5. If any parallel LLM check flagged issues, force needs_rework
            flagged_checks = [
                rc for rc in worker.payload.get("review_checks", [])
                if rc.get("flagged")
            ]
            if flagged_checks and worker.payload.get("review_verdict") != "needs_rework":
                worker.payload["review_verdict"] = "needs_rework"

            # 6. Write review.json (parallel path owns this)
            atomic_write_json(plan_dir / "review.json", worker.payload)
            # Write review_checks.json for audit
            atomic_write_json(plan_dir / "review_checks.json", {
                "review_checks": worker.payload.get("review_checks", []),
                "mechanical_prechecks": mechanical_results,
            })

            # Mark that this was a parallel review for iteration tracking
            state["meta"]["parallel_review"] = True

        except Exception as exc:
            # Fall back to standard monolithic review (same pattern as critique L708-710)
            print(f"[parallel-review] Failed, falling back to sequential: {exc}", file=sys.stderr)
            worker, agent_type, mode, refreshed = _run_worker("review", state, plan_dir, args, root=root)
            atomic_write_json(plan_dir / "review.json", worker.payload)
            state["meta"]["parallel_review"] = False
    else:
        # Standard/light mode OR heavy+non-hermes: existing monolithic path unchanged
        try:
            worker, agent_type, mode, refreshed = worker_module.run_step_with_worker(
                "review", state, plan_dir, args, root=root
            )
        except CliError as error:
            record_step_failure(plan_dir, state, step="review", iteration=state["iteration"], error=error)
            raise
        atomic_write_json(plan_dir / "review.json", worker.payload)
        state["meta"]["parallel_review"] = False

    # --- Common path (both branches converge here) ---
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
    # ... (rest of handle_review unchanged)
```

**Key properties of this approach:**
- **Lazy imports:** All new modules imported only inside `if robustness == "heavy" and agent_type == "hermes"` block. Standard/light mode never imports them (resolves import guard criteria).
- **Try-except with fallback:** Mirrors critique's L708-710 fallback pattern. If parallel review fails for any reason, falls back to monolithic (resolves `callers-3`).
- **Agent routing resolved:** Only enters parallel path when `agent_type == "hermes"`. For default heavy+codex, falls through to existing monolithic path. To actually USE parallel review, users must configure `--agent hermes` or `--phase-model review=hermes:...` or set `agents.review = "hermes"` in config. This is the SAME requirement as parallel critique (resolves `issue_hints-1`, `callers-2`).
- **Verdict forcing:** Mechanical failures and flagged LLM checks directly set `review_verdict = "needs_rework"` and inject `rework_items` (resolves `correctness-2`, `correctness-3`).
- **Single write to review.json:** The parallel path writes it once; no overwrite (resolves `all_locations-2`).

### Step 6: Define `_review_result_aggregator` in `review_checks.py`
**Scope:** Small

```python
def _review_result_aggregator(
    results: list[tuple | None],
) -> dict[str, Any]:
    """Aggregate parallel review check results into a payload."""
    ordered_checks = []
    all_rework_items = []
    any_flagged = False

    for item in results:
        if item is None:
            raise CliError("worker_error", "Parallel review did not return all check results")
        _index, check_result, _cost, _pt, _ct, _tt = item
        ordered_checks.append(check_result["check"])
        all_rework_items.extend(check_result.get("rework_items", []))
        if check_result.get("flagged"):
            any_flagged = True

    return {
        "review_checks": ordered_checks,
        "rework_items": all_rework_items,
        "review_verdict": "needs_rework" if any_flagged else "approved",
        "task_verdicts": [],
        "sense_check_verdicts": [],
        "criteria": [],
        "issues": [],
        "flags": [],
    }
```

Note: The payload includes empty `task_verdicts` and `sense_check_verdicts` so `_merge_review_verdicts` doesn't crash — it will find 0/0 tasks and 0/0 checks and not flag incompleteness. The verdict is already set by the aggregator.

## Phase 5: Review Iteration Cap and Rework Flow

### Step 7: Add parallel-review iteration tracking via `state["meta"]`
**Scope:** Small

**Design decision:** Use `state["meta"]["parallel_review_rework_count"]` instead of history entries. This avoids modifying `make_history_entry` or `HistoryEntry` TypedDict (resolves `issue_hints-3`, `correctness-7`, `all_locations-6`). The counter is incremented in `handle_review` when the parallel review result is `needs_rework`.

**In `handle_review`**, after determining the outcome:

```python
# Track parallel-review-specific rework count
if result == "needs_rework" and state["meta"].get("parallel_review"):
    state["meta"]["parallel_review_rework_count"] = (
        state["meta"].get("parallel_review_rework_count", 0) + 1
    )
```

**In `_resolve_review_outcome` (L1065)**, add the parallel-review cap BEFORE the existing rework logic:

```python
def _resolve_review_outcome(
    review_verdict, verdict_count, total_tasks,
    check_count, total_checks, missing_evidence,
    state, issues,
) -> tuple[str, str, str | None]:
    blocked = (
        verdict_count < total_tasks
        or check_count < total_checks
        or bool(missing_evidence)
    )
    if blocked:
        return "blocked", STATE_EXECUTED, "review"

    rework_requested = review_verdict == "needs_rework"
    if rework_requested:
        # Parallel-review-specific 2-iteration cap
        parallel_rework_count = state.get("meta", {}).get("parallel_review_rework_count", 0)
        if parallel_rework_count >= 2:
            issues.append(
                "Parallel review rework cap (2) reached. "
                "Force-proceeding to done."
            )
            # Fall through to success/done — still goes through gate for deadlock safety
            # The existing max_review_rework_cycles cap still applies as an overall cap
            # But the parallel-review cap forces proceed regardless
        else:
            # Check overall rework cap
            max_review_rework_cycles = get_effective("execution", "max_review_rework_cycles")
            prior_rework_count = sum(
                1 for entry in state.get("history", [])
                if entry.get("step") == "review" and entry.get("result") == "needs_rework"
            )
            if prior_rework_count >= max_review_rework_cycles:
                issues.append(
                    f"Max review rework cycles ({max_review_rework_cycles}) reached. "
                    "Force-proceeding to done despite unresolved review issues."
                )
            else:
                return "needs_rework", STATE_FINALIZED, "execute"

    return "success", STATE_DONE, None
```

**Routing:** The `needs_rework` return still maps to `STATE_FINALIZED → execute` — the existing gate loop. The parallel cap forces fall-through to `success → STATE_DONE` when the cap is hit. This is the same pattern as the existing `max_review_rework_cycles` cap at L1094-1098 (resolves `issue_hints-2`, `callers-7`).

## Phase 6: `check_audit` CLI Tool

### Step 8: Create `megaplan/check_audit.py` — standalone CLI
**Scope:** Medium

1. **Define** a `run_check_audit(plan_dir: Path) -> dict` function that:
   - Reads `review.json` and/or `critique.json` from `plan_dir`
   - Reads `state.json` from `plan_dir` for history
   - Cross-references checks with outcomes

2. **Failure definition (concrete):** A "known failure" is a plan where the final `state.json` history shows:
   - Any `review` step with `result == "needs_rework"` → the review phase found problems
   - OR any `critique` step with unresolved significant flags in the flag registry
   - A check "caught" a failure if it has `flagged: true` in a finding AND the plan had a failure outcome
   - A check "missed" a failure if the plan failed but the check had no flagged findings

3. **Output format:**
```python
{
    "plan": "plan-name",
    "phases": {
        "review": {
            "checks": [
                {"check_id": "coverage", "total_runs": 5, "flagged_on_failures": 3, "catch_rate": 0.6},
                ...
            ]
        },
        "critique": {
            "checks": [...]
        }
    }
}
```

4. **Wire into CLI** — add to `megaplan/cli.py`:
   - In `build_parser()` (L466), add a new subparser:
     ```python
     check_audit_parser = subparsers.add_parser("check-audit", help="Audit per-check catch rates")
     check_audit_parser.add_argument("--plan")
     ```
   - In `COMMAND_HANDLERS` (L602), add: `"check-audit": handle_check_audit,`
   - Define `handle_check_audit(args)` that resolves the plan directory and calls `run_check_audit`.

## Phase 7: Tests

### Step 9: Create `tests/test_parallel_checks.py`
**Scope:** Small

1. **Test** the generic `run_parallel_checks` primitive with a trivial check set and fake `run_check_fn` callback.
2. **Test** that the primitive preserves check order despite parallel completion (submit checks that complete out of order, verify results are in original order).
3. **Test** empty check set returns empty payload.
4. **Test** custom `result_aggregator` is called with ordered results.

### Step 10: Create `tests/test_review_checks.py`
**Scope:** Medium

1. **Test** `review_checks_for_robustness` returns 4 checks for heavy, empty for standard/light.
2. **Test** each mechanical pre-check:
   - `source_touch`: pass when source file touched, fail when only test files, pass when both.
   - `diff_size_sanity`: pass for reasonable sizes (<500 lines), fail for 0 lines, fail for >500 lines.
   - `dead_guard`: pass when no new guards, pass when new guard in called function, fail when new guard in uncalled function.
3. **Test** `_review_result_aggregator` produces correct payload from mixed results.
4. **Test** the review check template writer produces valid JSON with `guidance` field included.

### Step 11: Create `tests/test_parallel_review.py`
**Scope:** Medium

1. **Test** `single_check_review_prompt` includes `intent_and_notes_block` content.
2. **Test** `single_check_review_prompt` includes `finalize.json` as secondary context but `intent_and_notes_block` appears first.
3. **Test** heavy+hermes `handle_review` integration: mock `run_parallel_checks`, verify `review_checks` field appears in output.
4. **Test** standard-mode `handle_review` does NOT import review checks modules (verify the else branch executes the existing `_run_worker` path).
5. **Test** mechanical pre-check failures force `review_verdict = "needs_rework"`.
6. **Test** the 2-iteration cap for parallel-review-originated rework (simulate 2 prior reworks, verify force-proceed).
7. **Test** that parallel review failure falls back to monolithic review (try-except pattern).
8. **Test** that review-originated rework flows through `needs_rework → STATE_FINALIZED → execute`.

### Step 12: Create `tests/test_check_audit.py`
**Scope:** Small

1. **Test** `run_check_audit` reads plan data and reports per-check catch rates.
2. **Test** with synthetic plan directory containing `review.json`, `critique.json`, and `state.json`.
3. **Test** edge case: plan with no failures (all checks should report 0 flagged_on_failures).
4. **Test** edge case: plan with failure but no flagged checks (catch_rate = 0.0).

## Phase 8: Final Validation

### Step 13: Run full test suite and verify behavior
**Scope:** Small

1. **Run** `pytest tests/test_parallel_checks.py` first (validates primitive).
2. **Run** `pytest tests/test_parallel_critique.py` (validates critique still works after refactor).
3. **Run** `pytest tests/test_review_checks.py` (validates check specs and mechanical checks).
4. **Run** `pytest tests/test_parallel_review.py` (validates integration).
5. **Run** `pytest tests/test_check_audit.py` (validates CLI tool).
6. **Run** full `pytest tests/` for regression.

## Execution Order
1. Phase 1 (Steps 1-2): Extract `run_parallel_checks` primitive + add `max_review_concurrency` to DEFAULTS.
2. Phase 2 (Step 3): Create review check definitions + mechanical pre-checks.
3. Phase 3 (Step 4): Create review prompt builders + `_run_review_check` callback.
4. Phase 4 (Steps 5-6): Wire into `handle_review` + result aggregator.
5. Phase 5 (Step 7): Add iteration cap via `state["meta"]`.
6. Phase 6 (Step 8): `check_audit` CLI tool + CLI wiring.
7. Phase 7 (Steps 9-12): Tests.
8. Phase 8 (Step 13): Final validation.

## Validation Order
1. `pytest tests/test_parallel_checks.py` — validates primitive extraction
2. `pytest tests/test_parallel_critique.py` — validates critique regression
3. `pytest tests/test_review_checks.py` — validates check specs
4. `pytest tests/test_parallel_review.py` — validates integration
5. `pytest tests/test_check_audit.py` — validates CLI tool
6. `pytest tests/` — full regression
