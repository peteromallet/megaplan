# Megaplan
Route every step through the `megaplan` CLI. Never call agents directly.
## Triage
- Simple tasks: skip megaplan and do the work.
- Complex, risky, or ambiguous tasks: use megaplan.
- Unsure: ask whether the user wants direct execution or the planning loop.
## Start
Ask two questions before `init`:
1. Execution mode: auto-approve or review.
2. Robustness: light, standard, or thorough.
```bash
megaplan init --project-dir "$PROJECT_DIR" [--auto-approve] [--robustness light|standard|thorough] "$IDEA"
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
At `--robustness light`, `plan` can collapse plan + critique + gate into one call. When that happens it writes the critique and gate artifacts itself and may route directly to `finalize`, back to `revise`, or to an override path without separate `critique` / `gate` worker calls.
## Step Rules
- `plan`: inspect the repository first; produce the plan plus `questions`, `assumptions`, and `success_criteria`.
- `plan` at light robustness: also produce `self_flags`, `gate_recommendation`, `gate_rationale`, and `settled_decisions` so the handler can fast-forward the loop.
- `critique`: surface concrete flags with concern, evidence, category, and severity; reuse open flag IDs; call out scope creep.
- `gate`: read the response, warnings, and `orchestrator_guidance`.
- `revise`: show the delta, flags addressed, and flags remaining; then loop back through `critique` and `gate`.
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
megaplan audit --plan <name>
megaplan list
megaplan plan --plan <name>
megaplan critique --plan <name>
megaplan revise --plan <name>
megaplan gate --plan <name>
megaplan finalize --plan <name>
megaplan execute --plan <name> --confirm-destructive
megaplan review --plan <name>
megaplan step add --plan <name> [--after S<N>] "description"
megaplan step remove --plan <name> S<N>
megaplan step move --plan <name> S<N> --after S<M>
megaplan override add-note --plan <name> --note "..."
megaplan override force-proceed --plan <name> --reason "..."
megaplan override replan --plan <name> --reason "..." [--note "..."]
megaplan override abort --plan <name> --reason "..."
```
