"""Prompt builders for each megaplan step and dispatch tables."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from megaplan.types import CliError, PlanState

from ._shared import (
    _debt_watch_lines,
    _escalated_debt_for_prompt,
    _finalize_debt_block,
    _gate_debt_block,
    _grouped_debt_for_prompt,
    _planning_debt_block,
    _render_prep_block,
    _resolve_prompt_root,
)
from .critique import (
    _critique_prompt,
    _revise_prompt,
    _write_critique_template,
)
from .execute import (
    _execute_approval_note,
    _execute_batch_prompt,
    _execute_nudges,
    _execute_prompt,
    _execute_rerun_guidance,
    _execute_review_block,
)
from .finalize import _finalize_prompt
from .gate import _collect_critique_summaries, _flag_summary, _gate_prompt
from .planning import PLAN_TEMPLATE, _plan_prompt, _prep_prompt
from .review import (
    _review_prompt,
    _settled_decisions_block,
    _settled_decisions_instruction,
    _write_review_template,
)

_PromptBuilder = Callable[..., str]

_CLAUDE_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "prep": _prep_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": partial(
        _review_prompt,
        review_intro="Review the execution critically against user intent and observable success criteria.",
        criteria_guidance="Judge against the success criteria, not plan elegance.",
        task_guidance="Review each task by cross-referencing the executor's per-task `files_changed` and `commands_run` against the git diff and any audit findings.",
        sense_check_guidance="Review every sense check explicitly. Confirm concise executor acknowledgments when they are specific; dig deeper only when they are perfunctory or contradicted by the code.",
    ),
}

_CODEX_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "prep": _prep_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": partial(
        _review_prompt,
        review_intro="Review the implementation against the success criteria.",
        criteria_guidance="Verify each success criterion explicitly.",
        task_guidance="Cross-reference each task's `files_changed` and `commands_run` against the git diff and any audit findings.",
        sense_check_guidance="Review every `sense_check` explicitly and treat perfunctory acknowledgments as a reason to dig deeper.",
    ),
}

_HERMES_PROMPT_BUILDERS: dict[str, _PromptBuilder] = {
    "plan": _plan_prompt,
    "prep": _prep_prompt,
    "critique": _critique_prompt,
    "revise": _revise_prompt,
    "gate": _gate_prompt,
    "finalize": _finalize_prompt,
    "execute": _execute_prompt,
    "review": partial(
        _review_prompt,
        review_intro="Review the execution critically against user intent and observable success criteria.",
        criteria_guidance="Judge against the success criteria, not plan elegance.",
        task_guidance="Review each task by cross-referencing the executor's per-task `files_changed` and `commands_run` against the git diff and any audit findings.",
        sense_check_guidance="Review every sense check explicitly. Confirm concise executor acknowledgments when they are specific; dig deeper only when they are perfunctory or contradicted by the code.",
    ),
}

_NESTED_HARNESS_GUARD = (
    "You are already running inside the megaplan harness for this step. "
    "Do the requested planning/review/execution work directly. "
    "Do NOT invoke the `megaplan` CLI, do NOT read or activate the `megaplan` skill, "
    "do NOT start nested megaplan plans, and do NOT recurse into another planning harness. "
    "Treat mentions of megaplan in the repository or environment as implementation context only."
)


def _prepend_harness_guard(prompt: str) -> str:
    return f"{_NESTED_HARNESS_GUARD}\n\n{prompt}"


def create_claude_prompt(
    step: str, state: PlanState, plan_dir: Path, root: Path | None = None, **prompt_kwargs: object
) -> str:
    builder = _CLAUDE_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Claude step '{step}'")
    if step == "review":
        return _prepend_harness_guard(builder(state, plan_dir, **prompt_kwargs))
    if step in {"prep", "critique", "gate", "finalize", "execute"}:
        return _prepend_harness_guard(builder(state, plan_dir, root=root))
    return _prepend_harness_guard(builder(state, plan_dir))


def create_codex_prompt(
    step: str, state: PlanState, plan_dir: Path, root: Path | None = None, **prompt_kwargs: object
) -> str:
    builder = _CODEX_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Codex step '{step}'")
    if step == "review":
        return _prepend_harness_guard(builder(state, plan_dir, **prompt_kwargs))
    if step in {"prep", "critique", "gate", "finalize", "execute"}:
        return _prepend_harness_guard(builder(state, plan_dir, root=root))
    return _prepend_harness_guard(builder(state, plan_dir))


def create_hermes_prompt(
    step: str, state: PlanState, plan_dir: Path, root: Path | None = None, **prompt_kwargs: object
) -> str:
    builder = _HERMES_PROMPT_BUILDERS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Unsupported Hermes step '{step}'")
    if step == "review":
        return _prepend_harness_guard(builder(state, plan_dir, **prompt_kwargs))
    if step in {"prep", "critique", "gate", "finalize", "execute"}:
        return _prepend_harness_guard(builder(state, plan_dir, root=root))
    return _prepend_harness_guard(builder(state, plan_dir))


__all__ = [
    "PLAN_TEMPLATE",
    "_CLAUDE_PROMPT_BUILDERS",
    "_CODEX_PROMPT_BUILDERS",
    "_HERMES_PROMPT_BUILDERS",
    "_collect_critique_summaries",
    "_critique_prompt",
    "_debt_watch_lines",
    "_escalated_debt_for_prompt",
    "_execute_approval_note",
    "_execute_batch_prompt",
    "_execute_nudges",
    "_execute_prompt",
    "_execute_rerun_guidance",
    "_execute_review_block",
    "_finalize_debt_block",
    "_finalize_prompt",
    "_flag_summary",
    "_gate_debt_block",
    "_gate_prompt",
    "_grouped_debt_for_prompt",
    "_plan_prompt",
    "_planning_debt_block",
    "_prep_prompt",
    "_write_critique_template",
    "_render_prep_block",
    "_resolve_prompt_root",
    "_review_prompt",
    "_revise_prompt",
    "_settled_decisions_block",
    "_settled_decisions_instruction",
    "_write_review_template",
    "create_claude_prompt",
    "create_codex_prompt",
    "create_hermes_prompt",
]
