---
name: megaplan
description: AI agent harness for coordinating Claude and GPT to make and execute extremely robust plans.
---

# Megaplan

Use this skill when the user wants to run an idea through the megaplan harness — a structured clarify → plan → critique → evaluate loop between AI agents with auditable artifacts and gated execution.

You must route every workflow step through `megaplan`. Do not call `claude` or `codex` directly for the workflow itself, because that bypasses enforcement and audit logging.

## Triage

Before starting, assess whether megaplan is appropriate for the task:

- **Simple tasks** (single-file edits, straightforward bug fixes, well-defined changes): Tell the user this doesn't need megaplan — just do the work directly.
- **Complex tasks** (multi-file, ambiguous, high-risk, multi-stage, or the user explicitly asks for megaplan): Proceed with the megaplan workflow below.

If in doubt, ask the user: "This seems straightforward — want me to just do it, or run it through megaplan?"

## Start

Before running `init`, ask the user two onboarding questions:

**1. Execution mode**
> Would you like to **auto-approve** the megaplan after gate, or **review** it before execution?
>
> - **Auto-approve**: run the full clarify → plan → critique → evaluate → gate loop, show every intermediate result, and proceed directly to execution after a successful gate.
> - **Review** (default): run the same loop, then pause after a successful gate to summarize the megaplan, offer to read the final artifact inline, and wait for explicit approval before execution.

**2. Robustness level**
> What level of scrutiny should the critique/evaluate loop apply?
>
> - **Light**: only flag issues that would cause real failures.
> - **Standard** (default): balanced review and thresholds.
> - **Thorough**: stricter critique and stricter evaluate thresholds.

Then run:

```bash
megaplan init --project-dir "$PROJECT_DIR" [--auto-approve] [--robustness light|standard|thorough] "$IDEA"
```

Read the JSON response and report:
- The plan name
- The execution mode
- The robustness level
- The current state
- The next step

## Workflow

Always follow the `next_step` or `valid_next` fields returned by the CLI.

Standard loop:
1. `megaplan init`
2. `megaplan clarify` — show the refined idea, intent summary, and clarification questions. If the user volunteers corrections, record them with `megaplan override add-note --note "..."`, then keep moving.
3. `megaplan plan` — show the megaplan's `questions`, `assumptions`, and success criteria, then continue unless the user redirects.
4. `megaplan critique` — show the flags (concern, severity, evidence), not just the count. If any flag or warning indicates scope creep, call that out explicitly to the user while continuing the loop.
5. `megaplan evaluate`
   - If recommendation is `CONTINUE`, show the remaining flags, weighted score, cost so far, robustness level, and any warnings, then run `megaplan integrate` and loop back to critique.
   - If recommendation is `SKIP`, run `megaplan gate`.
   - If recommendation is `ESCALATE` or `ABORT`, apply the auto-override rules in the "Handling ESCALATE and ABORT" section below.
6. After integrate, show what changed: megaplan delta, which flags were addressed, and any scope-creep warning that is still open.
7. After a successful gate:
   - If the gate response shows `"auto_approve": true`, run `megaplan execute --confirm-destructive`.
   - Otherwise, present a high-level summary of the megaplan (objectives, key steps, remaining risks), offer to read `.megaplan/plans/<plan-name>/final.md` inline, and only after explicit user confirmation run `megaplan execute --confirm-destructive --user-approved`.
8. Finish with `megaplan review`.

At every step, consider two things:

1. **Intent alignment** — is this still solving what the user asked for? If the megaplan is drifting from the original idea, flag it.
2. **Level of abstraction** — are we approaching this at the right level? If the plan patches multiple systems for one goal, we might be too low (fix the design instead). If the plan redesigns architecture for a simple bug, we might be too high (just fix the bug). Push up or down the abstraction ladder as needed, and say so when you do.

## Autonomy

**Default behavior is to keep moving** — show intermediate results after each step, but do not pause for approval at every checkpoint.

**Review mode** is the default onboarding choice. It pauses only at the gate → execute checkpoint, where you must get explicit user approval before running `megaplan execute --confirm-destructive --user-approved`.

**Auto-approve mode** only skips that final gate → execute approval. It does not bypass failed gates or user interventions. For `ESCALATE`/`ABORT`, the auto-override rules in "Handling ESCALATE and ABORT" apply regardless of execution mode.

**Robustness level** affects critique strictness and evaluation thresholds. It does not change the workflow steps themselves.

## Handling ESCALATE and ABORT

Auto-force-proceed (and tell the user why) when:
- `suggested_override` is `"force-proceed"`, OR
- Robustness is `light` and `weighted_score` < 4.0

Otherwise, present the evaluation details and ask the user what to do.

## Minor Megaplan Edits

The orchestrator may make small clarifications or corrections to the megaplan text directly (e.g. fixing a typo, adding a note the user dictated) without re-running the full critique/integrate loop. Use `megaplan override add-note --note "..."` for context additions, or edit the megaplan artifact directly for trivial fixes. Reserve the full loop for substantive changes.

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
megaplan override add-note --plan <name> --note "user context"
megaplan override abort --plan <name>
```
