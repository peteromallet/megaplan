# Megaplan
Route every step through the `megaplan` CLI. Never call agents directly.
## Triage
Pick the right level based on the task:
- **Skip megaplan**: single-file fixes, bug fixes with clear cause, simple refactors, config changes, adding tests for existing code. Just do it.
- **Light** (default for megaplan): multi-file changes with clear scope, well-understood features, straightforward additions. One critique pass, no iteration loop.
- **Standard**: cross-cutting changes touching many subsystems, unfamiliar codebase areas, ambiguous requirements, changes with high breakage risk, or anything where the plan itself needs debate.

Default to light unless the task clearly needs standard. Do not ask the user to choose robustness — pick it yourself based on the above. Only ask execution mode (auto-approve or review) when using megaplan.
## Start
Ask execution mode (auto-approve or review) before `init`. Pick robustness yourself per the triage guidance above.
```bash
megaplan init --project-dir "$PROJECT_DIR" [--auto-approve] [--robustness light|standard] "$IDEA"
```
Report the plan name, execution mode, robustness, current state, and next step.
## Workflow
Run the loop in this order:
1. `plan`
2. `critique`
3. `gate`
4. `revise` when gate recommends iteration
5. `finalize`
6. `execute`
7. `review`
Use `next_step` and `valid_next` for CLI routing. After `gate`, follow `orchestrator_guidance` instead of manually interpreting gate signals.
At `--robustness light`, the loop is: `plan` → `critique` → `revise` → `finalize` → `execute`. No gate, no iteration, no review. One pass of external critique, one revision to incorporate it, then execute and done.
## Step Rules
- `plan`: inspect the repository first; produce the plan plus `questions`, `assumptions`, and `success_criteria`.
- `critique`: surface concrete flags with concern, evidence, category, and severity; reuse open flag IDs; call out scope creep.
- `gate`: read the response, warnings, and `orchestrator_guidance`. (Skipped at light robustness.)
- `revise`: show the delta, flags addressed, and flags remaining. At light robustness, routes to `finalize`; otherwise loops back through `critique` and `gate`.
- `review`: judge success against the success criteria and the user's intent, not plan elegance.
## Gate Principle
The gate response tells the orchestrator what to do next. Follow `orchestrator_guidance` unless you have a concrete reason to disagree after investigating the repository or plan artifacts yourself.
Investigate before disagreeing: read the current plan and critique artifacts, check the project code to verify whether a flagged issue is real, or use `megaplan status --plan <name>` / `megaplan audit --plan <name>`.
If you disagree with the guidance, explain why briefly and use an override. Do not manually reinterpret score trajectory, flag quality, or loop state when the gate already did that work for you.
## Execute
- After a successful gate, run `megaplan finalize` to produce the execution-ready briefing document.
- In auto-approve mode, run `megaplan execute --confirm-destructive` after finalize.
- In review mode, pause at the finalize-to-execute checkpoint and wait for explicit approval before running:
```bash
megaplan execute --confirm-destructive --user-approved
```
## Long-Running Execution
For plans with multiple batches, use per-batch mode to drive execution incrementally:
```bash
megaplan execute --plan <name> --confirm-destructive --user-approved --batch 1
megaplan execute --plan <name> --confirm-destructive --user-approved --batch 2
# ... continue until all batches complete
```
Between batches, poll progress:
```bash
megaplan progress --plan <name>
```
Per-batch mode uses global batch numbering (1-indexed, computed from ALL tasks). Each `--batch N` call:
- Validates that batches 1..N-1 are complete
- Executes only batch N's tasks
- Writes `execution_batch_N.json` as evidence
- On the final batch, produces aggregate `execution.json` and transitions to `executed`
Timeout recovery: re-run the same `--batch N`. The harness checks prerequisite completion and merges only untracked tasks.
Note: `progress` shows completed state only (between-batch granularity). With per-batch mode, each batch is a separate CLI call, so the orchestrator has full visibility.
## Overrides
- `megaplan override add-note --plan <name> --note "..."`
- `megaplan override force-proceed --plan <name> --reason "..."`
- `megaplan override replan --plan <name> --reason "..." [--note "..."]`
- `megaplan override abort --plan <name> --reason "..."`
`force-proceed` is available from `critiqued` (routes to finalize, not execute). `replan` is available from `gated`, `finalized`, or `critiqued`. `add-note` is safe from any active state.
## Replan
Use `replan` when the orchestrator itself needs to edit the plan directly instead of asking the revise worker to do it.
```bash
megaplan override replan --plan <name> --reason "expanding scope" --note "Also clean up the display layer"
```
After `replan`, read the returned plan file, edit it directly, then run `megaplan critique`.
## Step Editing
Use `step` when you need to insert, remove, or reorder step sections (`## Step N:` or `### Step N:`) without hand-editing the markdown.
```bash
megaplan step add --plan <name> --after S3 "Add regression coverage for the parser"
megaplan step remove --plan <name> S4
megaplan step move --plan <name> S4 --after S2
```
Each edit writes a new same-iteration plan artifact, preserves the latest plan meta questions/success criteria/assumptions, and resets the plan to `planned` so it re-enters critique.
## Sessions And Autonomy
- Agents default to persistent sessions.
- `--fresh`: start a new persistent session.
- `--ephemeral`: one-off call with no saved session.
- `--persist`: explicit persistent mode.
- Keep moving and show results at each step.
- Only pause at finalize to execute in review mode.
## Commands
```bash
megaplan status --plan <name>
megaplan progress --plan <name>
megaplan audit --plan <name>
megaplan list
megaplan plan --plan <name>
megaplan critique --plan <name>
megaplan revise --plan <name>
megaplan gate --plan <name>
megaplan finalize --plan <name>
megaplan execute --plan <name> --confirm-destructive [--batch N]
megaplan review --plan <name>
megaplan step add --plan <name> [--after S<N>] "description"
megaplan step remove --plan <name> S<N>
megaplan step move --plan <name> S<N> --after S<M>
megaplan override add-note --plan <name> --note "..."
megaplan override force-proceed --plan <name> --reason "..."
megaplan override replan --plan <name> --reason "..." [--note "..."]
megaplan override abort --plan <name> --reason "..."
```
