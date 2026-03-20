---
name: megaplan
description: Run an idea through the megaplan planning loop with enforced sequencing and audit artifacts.
---

# Megaplan

Use this skill when the user wants a plan to go through a structured planner and critic loop before execution.

You must route every workflow step through `megaplan`. Do not call `claude` or `codex` directly for the workflow itself, because that bypasses enforcement and audit logging.

## Start

Run:

```bash
megaplan init --project-dir "$PROJECT_DIR" "$IDEA"
```

Read the JSON response and report:
- The plan name
- The current state
- The next step

## Workflow

Always follow the `next_step` or `valid_next` fields returned by the CLI.

Standard loop:
1. `megaplan plan`
2. `megaplan critique`
3. `megaplan evaluate`
4. If recommendation is `CONTINUE`, run `megaplan integrate` and loop back to critique.
5. If recommendation is `SKIP`, run `megaplan gate`.
6. If recommendation is `ESCALATE` or `ABORT`, present the result (including `suggested_override` and `override_rationale`) and use `megaplan override ...` only if the user chooses to bypass.
7. After a successful gate, run `megaplan execute --confirm-destructive`.
8. Finish with `megaplan review`.

If the user says "run autonomously" or similar, proceed through steps without
waiting for confirmation at each checkpoint. Still report results at the end.

## Session Defaults

Both agents default to **persistent** sessions. The Codex critic carries context
across iterations so it can verify its own prior feedback was addressed.

Override flags (mutually exclusive):
- `--fresh` — start a new persistent session (break continuity)
- `--ephemeral` — truly one-off call with no session saved
- `--persist` — explicit persistent (default; mainly for Claude review override)

## Reporting

After each step, summarize:
- What happened from `summary`
- Which artifacts were written
- What the next step is

If the user wants detail, inspect the artifact files under `.megaplan/plans/<plan-name>/`.

## Useful Commands

```bash
megaplan status --plan <name>
megaplan audit --plan <name>
megaplan list
megaplan override add-note --plan <name> "user context"
megaplan override abort --plan <name>
```
