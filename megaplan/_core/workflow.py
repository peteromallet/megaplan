"""State machine — workflow transitions, robustness levels, step validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megaplan.types import (
    CliError,
    PlanState,
    ROBUSTNESS_LEVELS,
    STATE_ABORTED,
    STATE_CRITIQUED,
    STATE_DONE,
    STATE_EXECUTED,
    STATE_FINALIZED,
    STATE_GATED,
    STATE_INITIALIZED,
    STATE_PLANNED,
    STATE_PREPPED,
)


@dataclass(frozen=True)
class Transition:
    next_step: str
    next_state: str
    condition: str = "always"


WORKFLOW: dict[str, list[Transition]] = {
    STATE_INITIALIZED: [
        Transition("prep", STATE_PREPPED),
    ],
    STATE_PREPPED: [
        Transition("plan", STATE_PLANNED),
    ],
    STATE_PLANNED: [
        Transition("critique", STATE_CRITIQUED),
        Transition("plan", STATE_PLANNED),
    ],
    STATE_CRITIQUED: [
        Transition("gate", STATE_GATED, "gate_unset"),
        Transition("revise", STATE_PLANNED, "gate_iterate"),
        Transition("override add-note", STATE_CRITIQUED, "gate_escalate"),
        Transition("override force-proceed", STATE_GATED, "gate_escalate"),
        Transition("override abort", STATE_ABORTED, "gate_escalate"),
        Transition("revise", STATE_PLANNED, "gate_proceed_blocked"),
        Transition("override force-proceed", STATE_GATED, "gate_proceed_blocked"),
        Transition("gate", STATE_GATED, "gate_proceed"),
    ],
    STATE_GATED: [
        Transition("finalize", STATE_FINALIZED),
        Transition("override replan", STATE_PLANNED),
    ],
    STATE_FINALIZED: [
        Transition("execute", STATE_EXECUTED),
        Transition("override replan", STATE_PLANNED),
    ],
    STATE_EXECUTED: [
        # `handle_review()` may also return STATE_FINALIZED on a `needs_rework`
        # verdict. That rework loop depends on review payload semantics rather
        # than gate_* conditions, so it lives in the handler instead of here
        # because `_transition_matches()` only understands gate-based branches.
        Transition("review", STATE_DONE),
    ],
}

# Each level's *own* overrides (not inherited).  Levels inherit from the
# level below them via _ROBUSTNESS_HIERARCHY so shared transitions are
# declared once: heavy has none, standard keeps the planned->critique
# routing documented explicitly, and light adds gate/review skips.
_ROBUSTNESS_OVERRIDES: dict[str, dict[str, list[Transition]]] = {
    "heavy": {},
    "standard": {},
    "light": {
        STATE_CRITIQUED: [
            Transition("revise", STATE_GATED),
        ],
        STATE_EXECUTED: [],
    },
    "tiny": {},
}

_ROBUSTNESS_WORKFLOW_LEVELS: dict[str, tuple[str, ...]] = {
    "heavy": ("heavy",),
    "standard": ("standard",),
    "light": ("standard", "light"),
    "tiny": ("standard", "light", "tiny"),
}

_STEP_CONTEXT_STATES = {
    STATE_PLANNED,
    STATE_CRITIQUED,
    STATE_GATED,
    STATE_FINALIZED,
}


# ---------------------------------------------------------------------------
# Robustness helpers
# ---------------------------------------------------------------------------

def configured_robustness(state: PlanState) -> str:
    robustness = state["config"].get("robustness", "standard")
    if robustness not in ROBUSTNESS_LEVELS:
        return "standard"
    return robustness


def robustness_critique_instruction(robustness: str) -> str:
    if robustness == "light":
        return "Be pragmatic. Only flag issues that would cause real failures. Ignore style, minor edge cases, and issues the executor will naturally resolve."
    return "Use balanced judgment. Flag significant risks, but do not spend flags on minor polish or executor-obvious boilerplate."


# ---------------------------------------------------------------------------
# Intent / notes block for prompts
# ---------------------------------------------------------------------------

def intent_and_notes_block(state: PlanState) -> str:
    sections = []
    clarification = state.get("clarification", {})
    if clarification.get("intent_summary"):
        sections.append(f"User intent summary:\n{clarification['intent_summary']}")
        sections.append(f"Original idea:\n{state['idea']}")
    else:
        sections.append(f"Idea:\n{state['idea']}")
    notes = state["meta"].get("notes", [])
    if notes:
        notes_text = "\n".join(f"- {note['note']}" for note in notes)
        sections.append(f"User notes and answers:\n{notes_text}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Transition logic
# ---------------------------------------------------------------------------

def _normalize_workflow_robustness(robustness: Any) -> str:
    if robustness in ROBUSTNESS_LEVELS:
        return str(robustness)
    return "standard"


def _workflow_robustness_from_state(state: PlanState) -> str:
    config = state.get("config", {})
    if not isinstance(config, dict):
        return "standard"
    return _normalize_workflow_robustness(config.get("robustness", "standard"))


def _workflow_for_robustness(robustness: str) -> dict[str, list[Transition]]:
    normalized = _normalize_workflow_robustness(robustness)
    merged = dict(WORKFLOW)
    for level in _ROBUSTNESS_WORKFLOW_LEVELS.get(normalized, _ROBUSTNESS_WORKFLOW_LEVELS["standard"]):
        merged.update(_ROBUSTNESS_OVERRIDES.get(level, {}))
    return merged


def _transition_matches(state: PlanState, condition: str) -> bool:
    if condition == "always":
        return True
    gate = state.get("last_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    recommendation = gate.get("recommendation")
    if condition == "gate_unset":
        return not recommendation
    if condition == "gate_iterate":
        return recommendation == "ITERATE"
    if condition == "gate_escalate":
        return recommendation == "ESCALATE"
    if condition == "gate_proceed_blocked":
        return recommendation == "PROCEED" and not gate.get("passed", False)
    if condition == "gate_proceed":
        return recommendation == "PROCEED" and gate.get("passed", False)
    return False


def workflow_includes_step(robustness: str, step: str) -> bool:
    if step == "step":
        return True
    workflow = _workflow_for_robustness(robustness)
    return any(
        transition.next_step == step
        for transitions in workflow.values()
        for transition in transitions
    )


def workflow_transition(state: PlanState, step: str) -> Transition | None:
    current = state.get("current_state")
    if not isinstance(current, str):
        return None
    workflow = _workflow_for_robustness(_workflow_robustness_from_state(state))
    for transition in workflow.get(current, []):
        if transition.next_step == step and _transition_matches(state, transition.condition):
            return transition
    return None


def workflow_next(state: PlanState) -> list[str]:
    current = state.get("current_state")
    if not isinstance(current, str):
        return []
    workflow = _workflow_for_robustness(_workflow_robustness_from_state(state))
    next_steps = [
        transition.next_step
        for transition in workflow.get(current, [])
        if _transition_matches(state, transition.condition)
    ]
    if current in _STEP_CONTEXT_STATES:
        next_steps.append("step")
    return next_steps


infer_next_steps = workflow_next


def require_state(state: PlanState, step: str, allowed: set[str]) -> None:
    current = state["current_state"]
    if current not in allowed:
        raise CliError(
            "invalid_transition",
            f"Cannot run '{step}' while current state is '{current}'",
            valid_next=infer_next_steps(state),
            extra={"current_state": current},
        )
