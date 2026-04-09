from __future__ import annotations

import argparse
import re
from pathlib import Path

from megaplan._core import (
    append_history,
    atomic_write_json,
    atomic_write_text,
    infer_next_steps,
    latest_plan_meta_path,
    latest_plan_path,
    load_plan_locked,
    make_history_entry,
    now_utc,
    read_json,
    require_state,
    save_state,
    sha256_text,
    workflow_next,
)
from megaplan.evaluation import (
    PLAN_STRUCTURE_REQUIRED_STEP_ISSUE,
    PlanSection,
    parse_plan_sections,
    reassemble_plan,
    renumber_steps,
    validate_plan_structure,
)
from megaplan.types import (
    CliError,
    PlanState,
    STATE_CRITIQUED,
    STATE_FINALIZED,
    STATE_GATED,
    STATE_PLANNED,
    StepResponse,
)


STEP_EDIT_ALLOWED_STATES = {STATE_PLANNED, STATE_CRITIQUED, STATE_GATED, STATE_FINALIZED}
_STEP_ID_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)


def _normalize_step_id(raw_step_id: str, *, label: str) -> str:
    match = _STEP_ID_RE.fullmatch(raw_step_id.strip())
    if match is None:
        raise CliError("invalid_args", f"{label} must use the form S<number>; got '{raw_step_id}'")
    return f"S{int(match.group(1))}"


def _step_section_index(sections: list[PlanSection], step_id: str) -> int:
    for index, section in enumerate(sections):
        if section.id == step_id:
            return index
    raise CliError("missing_step", f"Plan does not contain step '{step_id}'")


def _last_step_section_index(sections: list[PlanSection]) -> int | None:
    last_index: int | None = None
    for index, section in enumerate(sections):
        if section.id is not None:
            last_index = index
    return last_index


def _default_step_insert_index(sections: list[PlanSection]) -> int:
    last_step_index = _last_step_section_index(sections)
    if last_step_index is not None:
        return last_step_index + 1
    for index, section in enumerate(sections):
        if section.heading in {"## Execution Order", "## Validation Order"}:
            return index
    return len(sections)


def _make_step_scaffold(description: str, *, heading_level: str = "##") -> PlanSection:
    clean_description = description.strip()
    if not clean_description:
        raise CliError("invalid_args", "step add requires a non-empty description")
    body = (
        f"{heading_level} Step 0: {clean_description}\n"
        "**Scope:** Small\n"
        "1. **TODO** Fill in implementation details (`path/to/file`).\n\n"
    )
    return PlanSection(
        heading=f"{heading_level} Step 0: {clean_description}",
        body=body,
        id="S0",
        start_line=0,
        end_line=0,
    )


def _detect_step_heading_level(sections: list[PlanSection]) -> str:
    for section in sections:
        if section.id is not None and section.heading.startswith("### "):
            return "###"
    return "##"


def _commit_step_edit(
    plan_dir: Path,
    state: PlanState,
    sections: list[PlanSection],
    *,
    action: str,
    action_summary: str,
) -> tuple[str, str, list[str]]:
    plan_text = reassemble_plan(sections)
    structure_warnings = validate_plan_structure(plan_text)
    if PLAN_STRUCTURE_REQUIRED_STEP_ISSUE in structure_warnings:
        raise CliError(
            "structure_error",
            f"Step edit failed structural validation: {PLAN_STRUCTURE_REQUIRED_STEP_ISSUE}",
            valid_next=infer_next_steps(state),
        )

    previous_meta_path = latest_plan_meta_path(plan_dir, state)
    previous_meta = read_json(previous_meta_path) if previous_meta_path.exists() else {}
    plan_filename = next_plan_artifact_name(plan_dir, state["iteration"])
    meta_filename = plan_filename.replace(".md", ".meta.json")
    meta = {
        "version": state["iteration"],
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "questions": previous_meta.get("questions", []),
        "success_criteria": previous_meta.get("success_criteria", []),
        "assumptions": previous_meta.get("assumptions", []),
        "structure_warnings": structure_warnings,
        "step_edit": {
            "action": action,
            "action_summary": action_summary,
        },
    }
    atomic_write_text(plan_dir / plan_filename, plan_text)
    atomic_write_json(plan_dir / meta_filename, meta)
    state["current_state"] = STATE_PLANNED
    state["meta"].pop("user_approved_gate", None)
    state["last_gate"] = {}
    state["plan_versions"].append(
        {"version": state["iteration"], "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]}
    )
    append_history(
        state,
        make_history_entry(
            "step",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file=plan_filename,
            artifact_hash=meta["hash"],
            message=action_summary,
        ),
    )
    save_state(plan_dir, state)
    return plan_filename, meta_filename, structure_warnings


def next_plan_artifact_name(plan_dir: Path, version: int) -> str:
    base = f"plan_v{version}"
    candidate = f"{base}.md"
    if not (plan_dir / candidate).exists():
        return candidate
    suffix_ord = ord("a")
    while True:
        candidate = f"{base}{chr(suffix_ord)}.md"
        if not (plan_dir / candidate).exists():
            return candidate
        suffix_ord += 1


def _step_add(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    sections = parse_plan_sections(latest_plan_path(plan_dir, state).read_text(encoding="utf-8"))
    new_section = _make_step_scaffold(args.description, heading_level=_detect_step_heading_level(sections))
    if args.after:
        after_step_id = _normalize_step_id(args.after, label="--after")
        insert_index = _step_section_index(sections, after_step_id) + 1
        action_summary = f"Inserted new step after {after_step_id}: {args.description.strip()}"
    else:
        insert_index = _default_step_insert_index(sections)
        action_summary = f"Inserted new step at the end of the current step list: {args.description.strip()}"
    updated_sections = sections[:insert_index] + [new_section] + sections[insert_index:]
    renumbered_sections = renumber_steps(updated_sections)
    plan_filename, meta_filename, structure_warnings = _commit_step_edit(
        plan_dir,
        state,
        renumbered_sections,
        action="add",
        action_summary=action_summary,
    )
    next_steps = workflow_next(state)
    return {
        "success": True,
        "step": "step",
        "summary": f"{action_summary}. Wrote {plan_filename} and reset the plan to planned state.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": next_steps[0] if next_steps else None,
        "state": STATE_PLANNED,
        "warnings": structure_warnings,
    }


def _step_remove(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    step_id = _normalize_step_id(args.step_id, label="step_id")
    sections = parse_plan_sections(latest_plan_path(plan_dir, state).read_text(encoding="utf-8"))
    step_indexes = [index for index, section in enumerate(sections) if section.id is not None]
    if len(step_indexes) <= 1:
        raise CliError("invalid_step_edit", "Cannot remove the last remaining step from a plan")
    remove_index = _step_section_index(sections, step_id)
    updated_sections = sections[:remove_index] + sections[remove_index + 1 :]
    renumbered_sections = renumber_steps(updated_sections)
    action_summary = f"Removed {step_id} and renumbered remaining steps"
    plan_filename, meta_filename, structure_warnings = _commit_step_edit(
        plan_dir,
        state,
        renumbered_sections,
        action="remove",
        action_summary=action_summary,
    )
    next_steps = workflow_next(state)
    return {
        "success": True,
        "step": "step",
        "summary": f"{action_summary}. Wrote {plan_filename} and reset the plan to planned state.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": next_steps[0] if next_steps else None,
        "state": STATE_PLANNED,
        "warnings": structure_warnings,
    }


def _step_move(plan_dir: Path, state: PlanState, args: argparse.Namespace) -> StepResponse:
    step_id = _normalize_step_id(args.step_id, label="step_id")
    after_step_id = _normalize_step_id(args.after, label="--after")
    if step_id == after_step_id:
        raise CliError("invalid_step_edit", f"Cannot move {step_id} after itself")

    sections = parse_plan_sections(latest_plan_path(plan_dir, state).read_text(encoding="utf-8"))
    move_index = _step_section_index(sections, step_id)
    moving_section = sections[move_index]
    remaining_sections = sections[:move_index] + sections[move_index + 1 :]
    insert_index = _step_section_index(remaining_sections, after_step_id) + 1
    updated_sections = remaining_sections[:insert_index] + [moving_section] + remaining_sections[insert_index:]
    renumbered_sections = renumber_steps(updated_sections)
    action_summary = f"Moved {step_id} after {after_step_id} and renumbered the plan"
    plan_filename, meta_filename, structure_warnings = _commit_step_edit(
        plan_dir,
        state,
        renumbered_sections,
        action="move",
        action_summary=action_summary,
    )
    next_steps = workflow_next(state)
    return {
        "success": True,
        "step": "step",
        "summary": f"{action_summary}. Wrote {plan_filename} and reset the plan to planned state.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": next_steps[0] if next_steps else None,
        "state": STATE_PLANNED,
        "warnings": structure_warnings,
    }


_STEP_ACTIONS = {
    "add": _step_add,
    "remove": _step_remove,
    "move": _step_move,
}


def handle_step(root: Path, args: argparse.Namespace) -> StepResponse:
    with load_plan_locked(root, args.plan, step=f"step {args.step_action}") as (plan_dir, state):
        require_state(state, "step", STEP_EDIT_ALLOWED_STATES)
        handler = _STEP_ACTIONS.get(args.step_action)
        if handler is None:
            raise CliError("invalid_args", f"Unknown step action: {args.step_action}")
        return handler(plan_dir, state, args)
