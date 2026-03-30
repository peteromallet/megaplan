"""Self-contained prompt builders for MegaLoop worker steps."""

from __future__ import annotations

import json
import textwrap
from typing import Any

from megaplan.loop.types import IterationResult, LoopState, Observation
from megaplan.schemas import SCHEMAS
_COMMAND_OUTPUT_LIMIT = 4000
def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
def _truncate(text: str, *, limit: int = _COMMAND_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[truncated {len(text) - limit} characters]"
def _recent_results(state: LoopState, limit: int = 5) -> list[IterationResult]:
    return state.get("results", [])[-limit:]


def _truncate_inline(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _normalize_inline(text: str) -> str:
    return " ".join(text.split())


def _format_metric(metric: float | None) -> str:
    if metric is None:
        return "-"
    return f"{metric:g}"


def _format_kill_reason(reason: str) -> str:
    if reason.startswith("break_pattern:"):
        _, pattern = reason.split(":", 1)
        return f"break pattern: {pattern}"
    if reason == "stall":
        return "metric stall"
    return reason.replace("_", " ")


def _observations_section(observations: list[Observation], observe_interval: int | None) -> str:
    if not observations:
        return ""

    interval_text = f"{observe_interval}s" if observe_interval is not None else "the configured interval"
    lines = [
        f"Process observations (sampled every {interval_text}):",
        "elapsed | metric | action | tail_snippet",
    ]
    for observation in observations:
        lines.append(
            " | ".join(
                [
                    f"{int(observation.get('elapsed_seconds', 0))}s",
                    _format_metric(observation.get("metric")),
                    str(observation.get("action", "continue")),
                    _truncate_inline(_normalize_inline(str(observation.get("tail_output", ""))), limit=200),
                ]
            )
        )
    return "\n".join(lines)


def build_loop_prompt(
    step: str,
    state: LoopState,
    command_output: str | None = None,
    observations: list[Observation] | None = None,
    observe_interval: int | None = None,
    kill_reason: str | None = None,
    is_truncated: bool | None = None,
) -> str:
    if step == "loop_plan":
        return build_plan_prompt(state)
    if step == "loop_execute":
        output = command_output if command_output is not None else state.get("last_command_output", "")
        observation_rows = observations if observations is not None else list(state.get("last_command_observations", []))
        interval = observe_interval if observe_interval is not None else state.get("spec", {}).get("observe_interval")
        prompt_kill_reason = kill_reason if kill_reason is not None else state.get("last_command_kill_reason")
        prompt_is_truncated = is_truncated if is_truncated is not None else bool(state.get("last_command_is_truncated", False))
        return build_execute_prompt(
            state,
            output,
            observations=observation_rows,
            observe_interval=int(interval) if interval is not None else None,
            kill_reason=str(prompt_kill_reason) if prompt_kill_reason else None,
            is_truncated=prompt_is_truncated,
        )
    raise ValueError(f"Unsupported loop step: {step}")
def build_plan_prompt(state: LoopState) -> str:
    spec = state.get("spec", {})
    goal = spec.get("goal", "")
    last_results = _recent_results(state)
    return textwrap.dedent(
        f"""
        You are running the planning phase of MegaLoop, a minimal iterative agent loop.

        Goal:
        {goal}

        Current loop spec (JSON):
        {_json_block(spec)}

        Recent iteration results (up to 5, newest last):
        {_json_block(last_results)}

        Known issues:
        {_json_block(spec.get("known_issues", []))}

        Tried and failed approaches:
        {_json_block(spec.get("tried_and_failed", []))}

        Update the loop spec based on the recent execution history. Keep the plan compact and practical.
        Preserve useful context, remove stale assumptions, and bias toward continuing the loop unless there is a real reason to pause.

        Return a JSON object matching this schema:
        {_json_block(SCHEMAS["loop_plan.json"])}

        Requirements:
        - Put only changed or newly learned loop-spec fields inside `spec_updates`.
        - `next_action` should say what the loop should do immediately next.
        - `reasoning` should briefly justify the update using the latest evidence.
        - Do not wrap the JSON in markdown.
        """
    ).strip()


def build_execute_prompt(
    state: LoopState,
    command_output: str,
    *,
    observations: list[Observation] | None = None,
    observe_interval: int | None = None,
    kill_reason: str | None = None,
    is_truncated: bool = False,
) -> str:
    spec = state.get("spec", {})
    goal = spec.get("goal", "")
    allowed_changes = spec.get("allowed_changes", [])
    observation_rows = observations or []
    observation_section = _observations_section(observation_rows, observe_interval)
    truncation_note = ""
    if is_truncated and kill_reason:
        last_elapsed = observation_rows[-1].get("elapsed_seconds") if observation_rows else None
        at_text = f" at {int(last_elapsed)}s" if last_elapsed is not None else ""
        truncation_note = (
            f"NOTE: Process was terminated early ({_format_kill_reason(kill_reason)}){at_text}. "
            "Output below is truncated; do not misdiagnose partial output as a different failure."
        )
    return textwrap.dedent(
        f"""
        You are running the execution diagnosis phase of MegaLoop.

        Goal:
        {goal}

        Current loop spec (JSON):
        {_json_block(spec)}

        Allowed changes:
        {_json_block(allowed_changes)}

        {observation_section}

        {truncation_note}

        Command output (truncated):
        {_truncate(command_output)}

        Diagnose the latest run and propose the smallest next fix that moves the loop forward.
        Stay grounded in the observed output and the allowed edit scope.
        Only ask to pause if the loop is genuinely stuck, risky, or already successful.

        Return a JSON object matching this schema:
        {_json_block(SCHEMAS["loop_execute.json"])}

        Requirements:
        - `diagnosis` should explain what likely happened in the latest run.
        - `fix_description` should describe the next concrete fix to attempt.
        - `files_to_change` should be a focused list of files to inspect or edit.
        - `confidence` should be a short natural-language confidence estimate.
        - `outcome` should summarize whether the loop should continue, retry, or pause.
        - Do not wrap the JSON in markdown.
        """
    ).strip()
