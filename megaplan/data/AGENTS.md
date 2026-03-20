# Megaplan

This project uses **megaplan** for high-rigor planning on complex tasks.

When the user asks to "use megaplan", "megaplan this", or the task is high-risk,
ambiguous, or multi-stage, run the idea through the megaplan workflow before
making any code changes.

## What it does

Megaplan is a CLI that enforces a structured plan → critique → evaluate loop
between AI agents. It produces auditable artifacts at each step and gates
execution behind human approval.

## How to use it

Route every step through the `megaplan` CLI. Do not call agents directly —
that bypasses enforcement and audit logging.

### Start

```bash
megaplan init --project-dir "$PROJECT_DIR" "description of the task"
```

Read the JSON response and follow `next_step`.

### Standard loop

1. `megaplan plan` — generate an implementation plan
2. `megaplan critique` — independent review of the plan
3. `megaplan evaluate` — automated assessment of critique severity
4. Based on the recommendation:
   - **CONTINUE** → `megaplan integrate` (revise plan), then back to step 2
   - **SKIP** → `megaplan gate` (no significant issues remain)
   - **ESCALATE/ABORT** → present `suggested_override` and `override_rationale`
     to the user; use `megaplan override ...` only if they choose to bypass
5. After gate: `megaplan execute --confirm-destructive`
6. Finish: `megaplan review`

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
