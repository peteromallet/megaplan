<!-- Source of truth for Codex-specific subagent orchestration. Appended only to the Codex skill via bundled_global_file('codex_skill.md'). -->
## Subagent Mode
This appendix is Codex-specific. It adds only the orchestration delta for Codex. The base skill remains the workflow source of truth.

### Activation
- Default to subagent unless an inline override is explicit for this run or `megaplan config show` reports `"orchestration": {"mode": "inline"}`.
- Prefer subagent for long multi-phase runs where keeping the outer conversation clean matters.
- Prefer inline for small edits, quick clarifications, or when the user wants to watch each phase in the main thread.

### Tool Mapping
- `spawn_agent`: launch the autonomous megaplan runner.
- `wait_agent`: wait for either a breakpoint or completion.
- `resume_agent`: reopen the orchestrator after a breakpoint.
- `send_input`: resume after a breakpoint, or interrupt a still-running agent when the user needs an immediate change.
- `close_agent`: hard-stop a stuck orchestrator before relaunching.

### Launch
When subagent mode is active, the outer skill becomes a launcher plus breakpoint relay. Start a Codex subagent with:
- `agent_type`: `default`
- `model`: prefer `gpt-5.4` when available
- `reasoning_effort`: `high`
- `fork_context`: `false` unless the current thread contains important constraints that are not restated in the prompt
- `message`: fill the template below with `{IDEA}`, `{PROJECT_DIR}`, `{AUTO_APPROVE}`, `{AUTO_APPROVE_FLAG}`, and `{ROBUSTNESS_FLAG}`
- Expand `{AUTO_APPROVE_FLAG}` to an empty string when `raw_config.execution.auto_approve` is explicitly set; otherwise expand it to `--auto-approve` for auto-approve runs and an empty string for review runs.
- Expand `{ROBUSTNESS_FLAG}` to an empty string when `raw_config.execution.robustness` is explicitly set; otherwise expand it to `--robustness {ROBUSTNESS}`.
- After editing this source file, rerun `megaplan setup --force` so installed `SKILL.md` files pick up the refreshed appendix.

### Outer Skill Rules
- Decide inline vs subagent before starting the workflow.
- Launch once, remember the spawned agent id, then `wait_agent` for a final message that starts with either `BREAKPOINT:` or `COMPLETE:`.
- Parse only the explicit first header line when deciding whether the stop was intentional.
- If a breakpoint arrives, relay it to the user, collect the answer, then `resume_agent` and `send_input` to the same agent.
- If the user adds context while the subagent is running, default to `megaplan override add-note --plan <name> --note "..."` and let the next phase boundary pick it up.
- If the user needs an immediate redirect, add the note first, then `send_input` with `interrupt: true` telling the orchestrator to rerun `megaplan status`, read all notes, and continue from the current state.
- If the orchestrator is stuck, `close_agent`, add a note, and relaunch a new subagent with a resume prompt on the same plan.

### Agent Prompt Template
```text
You are the autonomous megaplan runner for this single run.

Project: {PROJECT_DIR}
Idea: {IDEA}
Execution mode: {AUTO_APPROVE}

Operate through the `megaplan` CLI only. Do not call workers or agents directly.
Use the same verified `<launcher>` for every CLI call in this run. Verify it with a successful harmless CLI call first; command presence alone is not enough.

Startup:
1. Run `<launcher> init --project-dir "{PROJECT_DIR}" {AUTO_APPROVE_FLAG} {ROBUSTNESS_FLAG} "{IDEA}"`.
2. Capture the returned plan name.
3. Output `PLAN_NAME: <name>` on its own line before any `BREAKPOINT:` or `COMPLETE:`.
4. Run `<launcher> status --plan <name>`.

Routing:
- Use `next_step` and `valid_next` from `<launcher> status --plan <name>` for every move.
- Trust CLI state over memory.
- If `notes_count > 0`, read the full `notes` array before acting.
- After each step, read `next_step_runtime.duration_hint` and `next_step_runtime.recommended_next_check_seconds` when present to calibrate the next status check.
- For `light`: `plan -> critique -> revise -> finalize -> execute`.
- For `standard`, `robust`, or `superrobust`: follow the base skill workflow, including `prep`, `gate`, `review`, and any revise/rework loops.
- After `gate`, follow `orchestrator_guidance` unless repository evidence proves it wrong.

Breakpoints:
- Stop only for `GATE_ESCALATE`, `GATE_BLOCKED`, `EXECUTE_APPROVAL`, `PHASE_ESCALATE`, or `EXECUTE_ESCALATE`.
- Format every breakpoint exactly as:
  `BREAKPOINT: <type>`
  `Plan: <name>`
  `State: <state>`
  `Summary: <short reason>`
  `Context: <artifacts, warnings, or the exact user decision needed>`

Safeguards:
- Retry a non-execute phase once with `--fresh` before escalating.
- If `execute` makes no forward progress for {max_execute_no_progress} attempts, stop with `BREAKPOINT: EXECUTE_ESCALATE`.
- Treat `review` returning `needs_rework` as a normal branch, not a breakpoint.

Completion:
- When the run finishes, return exactly:
  `COMPLETE: megaplan run finished`
  `Plan: <name>`
  `Final state: <state>`
  `Summary: <outcome>`
  `Artifacts: <key files or reports>`
  `Follow-up: <only if something remains>`
```
