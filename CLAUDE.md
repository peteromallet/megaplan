# Local Guidance

This repository is the source for **megaplan**, an AI agent harness for coordinating Claude and GPT to make and execute extremely robust plans.

## Development

- Source: `megaplan/cli.py`
- Tests: `tests/test_megaplan.py`
- Install in dev mode: `pip install -e .`
- Run tests: `pytest`

## Using megaplan

Suggest the megaplan workflow when:
- The task is high-risk, ambiguous, or multi-stage.
- The user wants an auditable planner/critic/executor loop.
- The work should be gated before execution instead of going straight into edits.

Use the `.claude/skills/megaplan.md` skill when the user explicitly asks for the megaplan flow or when the task clearly matches that workflow.

## For end users

After `pip install megaplan-harness`, run `megaplan setup` in a project to install
AGENTS.md — this tells any AI agent how to use megaplan automatically.
