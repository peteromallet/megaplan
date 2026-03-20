# Megaplan

This project uses **megaplan** for high-rigor planning on complex tasks.

When the user asks to "use megaplan", "megaplan this", or the task is high-risk,
ambiguous, or multi-stage, run the idea through the megaplan workflow before
making any code changes.

## What it does

Megaplan is a CLI that enforces a structured clarify → plan → critique →
evaluate loop between AI agents. It produces auditable artifacts at each step
and gates execution behind human approval. The robustness level
(`light`/`standard`/`thorough`) controls critique depth and evaluation
thresholds.

## How to use it

Route every step through the `megaplan` CLI. Do not call agents directly —
that bypasses enforcement and audit logging.

### Start

Before running `init`, ask the user about:
- Execution mode: `auto-approve` or `review` (default)
- Robustness: `light`, `standard` (default), or `thorough`

```bash
megaplan init --project-dir "$PROJECT_DIR" [--auto-approve] [--robustness light|standard|thorough] "description of the task"
```

Read the JSON response and report the plan name, execution mode, robustness, and `next_step`.

### Standard loop

1. `megaplan init`
2. `megaplan clarify` — present the refined idea, intent summary, and
   clarification questions, then keep moving unless the user redirects. Record
   any user clarifications with `megaplan override add-note --plan <name> "..."`
3. `megaplan plan` — present the plan's questions, assumptions, and success
   criteria, then keep moving unless the user redirects
4. `megaplan critique` — present the flags, not just the count. If a flag or
   warning indicates scope creep, call that out explicitly to the user.
5. `megaplan evaluate` — if CONTINUE, present the remaining flags, weighted
   score, cost so far, robustness level, and warnings, then run
   `megaplan integrate` and loop back to step 4
6. Based on the recommendation:
   - **SKIP** → `megaplan gate` (no significant issues remain)
   - **ESCALATE/ABORT** → present `suggested_override` and `override_rationale`
     to the user; use `megaplan override ...` only if they choose to bypass
7. After integrate, show what changed: plan delta, addressed flags, and any
   remaining scope-creep warning
8. After gate:
   - Auto-approve mode: `megaplan execute --confirm-destructive`
   - Review mode: present a high-level plan summary, offer to read
     `.megaplan/plans/<name>/final.md` inline, then only after explicit user
     confirmation run `megaplan execute --confirm-destructive --user-approved`
9. Finish: `megaplan review`

At each step, sense-check against the user's original intent before proceeding.
If the plan is drifting beyond the original idea or recorded notes, flag that
scope creep to the user instead of silently accepting it.

**Default behavior is to keep moving** — show intermediate results, but do not
pause for approval at every checkpoint. Review mode pauses only at the
gate → execute checkpoint. Auto-approve only skips that final approval gate.

### Useful commands

```bash
megaplan status --plan <name>    # current state and next step
megaplan audit --plan <name>     # full state dump
megaplan list                    # all plans
megaplan override add-note --plan <name> "context for the next iteration"
megaplan override abort --plan <name>
```

### Session defaults

Both agents default to persistent sessions (context carries across iterations).
Override with `--fresh` (new session), `--ephemeral` (one-off), or `--persist`.

## When to suggest megaplan

- The task spans multiple files or systems
- There are security, data-integrity, or compliance concerns
- The user wants an auditable record of planning decisions
- The work should be reviewed before execution begins
