# Local Guidance

The repository includes `doubledip.py`, a stateful orchestration CLI for high-rigor planning loops between Claude Code and Codex.

Suggest the double-dip workflow when:
- The task is high-risk, ambiguous, or multi-stage.
- The user wants an auditable planner/critic/executor loop.
- The work should be gated before execution instead of going straight into edits.

Use the `.claude/skills/doubledip.md` skill when the user explicitly asks for the double-dip flow or when the task clearly matches that workflow.
