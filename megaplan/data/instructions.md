# Megaplan

Route every step through the `megaplan` CLI. Never call agents directly.

## Triage

- **Simple tasks**: skip megaplan and do the work.
- **Complex, risky, or ambiguous tasks**: use megaplan.
- **Unsure**: ask whether the user wants direct execution or the planning loop.

## Start

Ask two questions before `init`:

1. Execution mode: auto-approve or review.
2. Robustness: light, standard, or thorough.

```bash
megaplan init --project-dir "$PROJECT_DIR" [--auto-approve] [--robustness light|standard|thorough] "$IDEA"
```

Report the plan name, execution mode, robustness, current state, and next step.

## Workflow

The explicit workflow is:

1. `plan`
2. `critique`
3. `gate`
4. `revise` when gate recommends iteration
5. `execute`
6. `review`

`init` remains an implicit setup phase and is not counted in the loop.

Follow `next_step` and `valid_next` from CLI responses.

### plan

- Inspect the repository before planning.
- Produce a draft plan plus `questions`, `assumptions`, and `success_criteria`.
- The plan step itself handles clarification. If something is ambiguous, ask in `questions` and make explicit assumptions so the loop can still move forward.
- If the user answers questions later, record them with `megaplan override add-note --note "..."` and rerun `megaplan plan`.

### critique

- Surface concrete flags with concern, evidence, category, and severity.
- Reuse open flag IDs when the same issue persists.
- Call out scope creep explicitly.

### gate

Gate is an LLM judgment step, not a predicate table. It recommends `PROCEED`, `ITERATE`, or `ESCALATE`.

The gate response includes these signals — use them for your own assessment:

- **`idea`**: the original user idea — judge scope creep and intent alignment against this.
- **`weighted_score`**: unresolved flag weight. **Lower = better.** 0 = all resolved.
- **`unresolved_flags`**: `[{id, concern, category, severity}]` — what's still open.
- **`resolved_flags`**: `[{id, concern, resolution}]` — what was fixed and how.
- **`weighted_history`**: scores from previous iterations (chronological). Current score is in `weighted_score`.
- **`plan_delta_from_previous`**: % of plan text changed (0–100).
- **`recurring_critiques`**: concerns repeated across consecutive critiques — the loop is churning on these.
- **`loop_summary`**: one-line trajectory narrative. Compression-proof — use if earlier turns were truncated.
- **`warnings`**: always show to user.

**After gate, assess whether you agree with its recommendation.** The gate LLM has the plan and flags but cannot read the project code. You can. Before following the recommendation, consider:

- **Score trajectory**: improving or plateaued?
- **Flag quality**: real design concerns, or implementation details the executor will handle? Read the `unresolved_flags` — are they about pseudocode placeholders, missing column names, or actual architectural gaps?
- **Recurring critiques**: if the same concerns keep appearing, the loop can't fix them.
- **You can investigate.** Use your tools:
  - `.megaplan/plans/<name>/plan_v{N}.md` — the plan itself
  - `.megaplan/plans/<name>/faults.json` — full flag history
  - `.megaplan/plans/<name>/critique_v{N}.json` — raw critique with evidence
  - The project code — grep to verify whether flagged concerns are real
  - `megaplan status --plan <name>` / `megaplan audit --plan <name>`

**If the gate says ITERATE but you judge the remaining flags are noise**: cite the numbers, state your assessment, and run `megaplan override force-proceed --plan <name>`. Don't ask — act. The user can redirect.

**If the gate says PROCEED**: show the result and continue to execute (or pause for approval in review mode).

**If the gate says ESCALATE**: auto-force-proceed when `suggested_override == "force-proceed"` or robustness is `light` and `weighted_score <= 4.0`. Tell the user why. Otherwise present the details and ask: force-proceed, add-note, or abort?

**First iteration** (`weighted_history` empty): always follow the gate's recommendation — there's no trajectory to second-guess yet.

### revise

- Revise is the plan-editing loop step — addresses critique flags and produces a new plan version.
- After revise, show what changed: delta, flags addressed, flags remaining.
- Apply the same judgment as after gate: if the delta was small and the same flags persist, note it and consider force-proceeding.
- Then run `megaplan critique` again.

### execute

- In auto-approve mode, run `megaplan execute --confirm-destructive` after a successful gate.
- In review mode, pause at the gate checkpoint, summarize the approved plan, and wait for explicit approval before running:

```bash
megaplan execute --confirm-destructive --user-approved
```

### review

- Run `megaplan review` after execution.
- Judge success against the success criteria and user intent, not plan elegance.

### Always consider

1. **Intent alignment** — is this still solving the original ask?
2. **Abstraction level** — too low (patching multiple systems) or too high (redesigning for a simple bug)?

## Autonomy

**Keep moving.** Show results at each step but don't pause except at gate→execute in review mode.

## Overrides

Available overrides:

- `megaplan override add-note --plan <name> --note "..."`
- `megaplan override force-proceed --plan <name> --reason "..."`
- `megaplan override replan --plan <name> --reason "..." [--note "..."]`
- `megaplan override abort --plan <name> --reason "..."`

Notes:

- `force-proceed` is available from `critiqued`.
- `replan` is available from `gated` or `critiqued`.
- `add-note` is safe from any active state.
- `skip` no longer exists.

## Replan

Use `replan` when the orchestrator itself needs to edit the plan directly instead of asking the revise worker to do it.

```bash
megaplan override replan --plan <name> --reason "expanding scope" --note "Also clean up the display layer"
```

This resets to `planned` state. The response includes `plan_file`. After running replan:

1. Read the plan file with your Read tool
2. Edit it with your Edit tool to incorporate the user's changes
3. Run `megaplan critique` to review the revised plan

For small additions, `--note` alone may suffice — the critic will flag what's missing and revise will incorporate it.

## Sessions

Agents default to persistent sessions.

- `--fresh`: start a new persistent session
- `--ephemeral`: one-off call with no saved session
- `--persist`: explicit persistent mode

## Commands

```bash
megaplan status --plan <name>
megaplan audit --plan <name>
megaplan list
megaplan plan --plan <name>
megaplan critique --plan <name>
megaplan revise --plan <name>
megaplan gate --plan <name>
megaplan execute --plan <name> --confirm-destructive
megaplan review --plan <name>
megaplan override add-note --plan <name> --note "..."
megaplan override force-proceed --plan <name> --reason "..."
megaplan override replan --plan <name> --reason "..." [--note "..."]
megaplan override abort --plan <name> --reason "..."
```
