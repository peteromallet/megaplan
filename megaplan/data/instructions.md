# Megaplan

Route every step through the `megaplan` CLI — never call agents directly.

## Triage

- **Simple tasks** (single-file edits, clear bug fixes): skip megaplan, just do the work.
- **Complex tasks** (multi-file, ambiguous, high-risk, or user asks): use megaplan.
- **Unsure?** Ask: "This seems straightforward — want me to just do it, or run it through megaplan?"

## Start

Ask two questions before `init`:

1. **Execution mode**: auto-approve (proceed after gate) or review (pause for approval, default)?
2. **Robustness**: light (real failures only), standard (default), or thorough (strict)?

```bash
megaplan init --project-dir "$PROJECT_DIR" [--auto-approve] [--robustness light|standard|thorough] "$IDEA"
```

Report: plan name, execution mode, robustness, current state, next step.

## Evaluation signals

Evaluate responses include these fields in `signals`:

- **`idea`**: the original user idea — use this to judge scope creep and intent alignment.
- **`weighted_score`**: unresolved flag weight. **Lower = better.** 0 = all resolved.
- **`unresolved_flags`**: `[{id, concern, category}]` — what's still open.
- **`resolved_flags`**: `[{id, concern, resolution}]` — what was fixed and how. Shows the loop is working.
- **`weighted_history`**: scores from previous iterations (chronological). Current score is in `weighted_score`, not here.
- **`plan_delta_from_previous`**: % of plan text changed (0–100).
- **`recurring_critiques`**: concerns repeated across consecutive critiques — the loop is churning on these.
- **`loop_summary`**: one-line trajectory narrative (scores, deltas, flags resolved/open). Compression-proof — use this if earlier turns were truncated.
- **`confidence`**: `high` or `medium`. Informational only.
- **`warnings`**: always show to user.

## Workflow

Follow `next_step` / `valid_next_steps` from CLI responses.

1. **init**
2. **clarify** — show refined idea, intent summary, questions. Record user corrections with `megaplan override add-note --note "..."`. Keep moving.
3. **plan** — show questions, assumptions, success criteria. Continue unless user redirects.
4. **critique** — show flags (concern, severity, evidence). Call out scope creep explicitly.
5. **evaluate** — route on recommendation:
   - **CONTINUE**: show unresolved flags, score, robustness, recurring critiques, warnings. Assess whether to continue (see below). If yes, run `megaplan integrate`, loop to critique.
   - **SKIP**: run `megaplan gate`.
   - **ESCALATE**: see "Handling ESCALATE" below.
6. **After integrate** — show delta, flags addressed, flags remaining. Apply same judgment as step 5. If delta was small and same flags persist, note it. You can force-proceed here too.
7. **After gate** — if `auto_approve: true`, run `megaplan execute --confirm-destructive`. Otherwise summarize the plan, offer to show `final.md`, wait for explicit approval, then `megaplan execute --confirm-destructive --user-approved`.
8. **review** — finish with `megaplan review`.

### Assessing whether to continue

The evaluator says CONTINUE, but you have context it doesn't. Before integrating, consider:

- **Score trajectory**: improving (8→5→3) or plateaued (2.2→2.0→1.9)?
- **Plan delta**: did the plan meaningfully change, or barely move?
- **Flag quality**: real design concerns, or implementation details the executor will handle?
- **Recurring critiques**: same concerns repeating = the loop can't fix them.

**You can investigate.** You're not limited to the JSON:
- `.megaplan/plans/<name>/plan_v{N}.md` — the plan itself
- `.megaplan/plans/<name>/faults.json` — full flag history with statuses
- `.megaplan/plans/<name>/critique_v{N}.json` — raw critique with evidence
- The project code — grep to verify whether flagged concerns are real
- `megaplan status --plan <name>` / `megaplan audit --plan <name>`

**If continuing looks unproductive**: cite the numbers, state your recommendation, and run `megaplan override force-proceed --plan <name>`. Don't ask — act. The user can redirect.

**First iteration** (`weighted_history` empty): always continue.

### Always consider

1. **Intent alignment** — is this still solving the original ask?
2. **Abstraction level** — too low (patching multiple systems for one goal) or too high (redesigning architecture for a simple bug)?

## Autonomy

**Keep moving.** Show results at each step but don't pause for approval except at gate→execute in review mode.

- **Review mode** (default): pause only at gate→execute for user approval.
- **Auto-approve mode**: no pause — goes straight through.
- **ESCALATE** rules apply regardless of mode.

## Handling ESCALATE

Auto-force-proceed when:
- `suggested_override == "force-proceed"`, OR
- robustness is `light` and `weighted_score <= 4.0`

Tell the user why (include `override_rationale`, score, recurring critiques, warnings), then run `megaplan override force-proceed --plan <name>` → gate.

Otherwise: show evaluation details and ask the user — force-proceed, add-note, or abort?

## Re-entering the planning loop

If the user wants to expand scope or revise the plan after gate (or at any post-critique stage), use `replan`:

```bash
megaplan override replan --plan <name> --reason "expanding scope" --note "Also clean up the display layer"
```

This resets to `planned` state. The response includes `plan_file` — the path to the current plan markdown file. After running replan:

1. Read the plan file with your Read tool
2. Edit it with your Edit tool to incorporate the user's changes — add sections, expand scope, revise steps, whatever they asked for
3. Run `megaplan critique` to review the revised plan

All prior flags, plan versions, and history are preserved.

For small additions where the user gave clear direction in `--note`, you can skip the manual edit — the critic will see the note, flag that the plan doesn't cover it, and the integrator will add it on the next loop. But for substantive scope changes, edit the plan directly so the critic reviews the complete revised version.

Available from: `gated`, `evaluated`, or `critiqued` states.

## Minor edits

Use `megaplan override add-note --note "..."` for context additions. Edit plan artifacts directly for trivial fixes. Reserve the full loop for substantive changes.

## Sessions

Agents default to **persistent** sessions. Override with:
- `--fresh` — new persistent session
- `--ephemeral` — one-off, no session saved
- `--persist` — explicit persistent (default)

## Commands

```bash
megaplan status --plan <name>
megaplan audit --plan <name>
megaplan list
megaplan override add-note --plan <name> --note "..."
megaplan override force-proceed --plan <name>
megaplan override replan --plan <name> --reason "..." [--note "..."]
megaplan override abort --plan <name>
```
