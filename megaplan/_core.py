"""Shared utilities for megaplan — I/O, config, plan resolution, flag helpers, prompt helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from megaplan.schemas import SCHEMAS, strict_schema
from megaplan.types import (
    CliError,
    DEBT_ESCALATION_THRESHOLD,
    DebtEntry,
    DebtRegistry,
    FLAG_BLOCKING_STATUSES,
    FlagRecord,
    FlagRegistry,
    KNOWN_AGENTS,
    PlanState,
    PlanVersionRecord,
    ROBUSTNESS_LEVELS,
    SCOPE_CREEP_TERMS,
    STATE_ABORTED,
    STATE_CRITIQUED,
    STATE_DONE,
    STATE_EXECUTED,
    STATE_FINALIZED,
    STATE_GATED,
    STATE_INITIALIZED,
    STATE_PREPPED,
    STATE_PLANNED,
    STATE_RESEARCHED,
    TERMINAL_STATES,
)


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str, max_length: int = 30) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) <= max_length:
        return slug or "plan"
    truncated = slug[:max_length]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > 10:
        truncated = truncated[:last_hyphen]
    return truncated or "plan"


def json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_text(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def compute_task_batches(
    tasks: list[dict[str, Any]],
    completed_ids: set[str] | None = None,
) -> list[list[str]]:
    completed = set(completed_ids or set())
    if not tasks:
        return []

    task_ids = [task["id"] for task in tasks]
    task_id_set = set(task_ids)
    remaining: dict[str, set[str]] = {}
    order_index = {task_id: index for index, task_id in enumerate(task_ids)}

    for task in tasks:
        task_id = task["id"]
        deps = task.get("depends_on", [])
        if not isinstance(deps, list):
            deps = []
        normalized_deps: set[str] = set()
        for dep in deps:
            if dep in task_id_set:
                normalized_deps.add(dep)
                continue
            if dep in completed:
                continue
            raise ValueError(f"Unknown dependency ID '{dep}' for task '{task_id}'")
        remaining[task_id] = normalized_deps

    batches: list[list[str]] = []
    satisfied = set(completed)
    unscheduled = set(task_ids)

    while unscheduled:
        ready = [
            task_id
            for task_id in unscheduled
            if remaining[task_id].issubset(satisfied)
        ]
        ready.sort(key=order_index.__getitem__)
        if not ready:
            cycle_ids = sorted(unscheduled, key=order_index.__getitem__)
            raise ValueError("Cyclic dependency graph detected among tasks: " + ", ".join(cycle_ids))
        batches.append(ready)
        satisfied.update(ready)
        unscheduled.difference_update(ready)

    return batches


def compute_global_batches(finalize_data: dict[str, Any]) -> list[list[str]]:
    tasks = finalize_data.get("tasks", [])
    return compute_task_batches(tasks)


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json_dump(data))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def render_final_md(finalize_data: dict[str, Any], *, phase: str = "finalize") -> str:
    show_execution_gaps = phase in ("execute", "review")
    show_review_gaps = phase == "review"
    tasks = finalize_data.get("tasks", [])
    sense_checks = finalize_data.get("sense_checks", [])

    lines = ["# Execution Checklist", ""]
    gap_counts: dict[str, int] = {}
    for task in tasks:
        status = task.get("status")
        checkbox = "[x]" if status == "done" else "[ ]"
        status_suffix = " (skipped)" if status == "skipped" else ""
        lines.append(f"- {checkbox} **{task['id']}:** {task['description']}{status_suffix}")
        depends_on = task.get("depends_on", [])
        if depends_on:
            lines.append(f"  Depends on: {', '.join(depends_on)}")
        executor_notes = task.get("executor_notes", "")
        if executor_notes.strip():
            lines.append(f"  Executor notes: {executor_notes}")
        elif show_execution_gaps and status != "pending":
            lines.append("  Executor notes: [MISSING]")
            gap_counts["Executor notes missing"] = gap_counts.get("Executor notes missing", 0) + 1
        files_changed = task.get("files_changed", [])
        if files_changed:
            lines.append("  Files changed:")
            for path in files_changed:
                lines.append(f"    - {path}")
        if show_execution_gaps and status == "pending":
            gap_counts["Tasks without executor updates"] = gap_counts.get("Tasks without executor updates", 0) + 1
        reviewer_verdict = task.get("reviewer_verdict", "")
        if reviewer_verdict.strip():
            lines.append(f"  Reviewer verdict: {reviewer_verdict}")
            evidence_files = task.get("evidence_files", [])
            if evidence_files:
                lines.append("  Evidence files:")
                for path in evidence_files:
                    lines.append(f"    - {path}")
        elif show_review_gaps:
            lines.append("  Reviewer verdict: [PENDING]")
            gap_counts["Reviewer verdicts pending"] = gap_counts.get("Reviewer verdicts pending", 0) + 1
        lines.append("")

    lines.extend(["## Watch Items", ""])
    watch_items = finalize_data.get("watch_items", [])
    if watch_items:
        for item in watch_items:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.extend(["## Sense Checks", ""])
    if sense_checks:
        for sense_check in sense_checks:
            lines.append(f"- **{sense_check['id']}** ({sense_check['task_id']}): {sense_check['question']}")
            executor_note = sense_check.get("executor_note", "")
            if executor_note.strip():
                lines.append(f"  Executor note: {executor_note}")
            elif show_execution_gaps:
                lines.append("  Executor note: [MISSING]")
                gap_counts["Sense-check acknowledgments missing"] = gap_counts.get("Sense-check acknowledgments missing", 0) + 1
            verdict = sense_check.get("verdict", "")
            if verdict.strip():
                lines.append(f"  Verdict: {verdict}")
            elif show_review_gaps:
                lines.append("  Verdict: [PENDING]")
                gap_counts["Sense-check verdicts pending"] = gap_counts.get("Sense-check verdicts pending", 0) + 1
            lines.append("")
    else:
        lines.extend(["- None.", ""])

    lines.extend(["## Meta", ""])
    meta_commentary = (finalize_data.get("meta_commentary") or "").strip()
    lines.append(meta_commentary or "None.")
    lines.append("")

    if gap_counts:
        lines.extend(["## Coverage Gaps", ""])
        for label, count in gap_counts.items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def config_dir(home: Path | None = None) -> Path:
    if home is None:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "megaplan"
        home = Path.home()
    return home / ".config" / "megaplan"


def load_config(home: Path | None = None) -> dict[str, Any]:
    path = config_dir(home) / "config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        import sys
        print(f"megaplan: warning: ignoring malformed config at {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_config(config: dict[str, Any], home: Path | None = None) -> Path:
    path = config_dir(home) / "config.json"
    atomic_write_json(path, config)
    return path


def detect_available_agents() -> list[str]:
    available = [a for a in KNOWN_AGENTS if a != "hermes" and shutil.which(a)]
    # Hermes is a Python library, not a CLI binary
    try:
        import run_agent  # noqa: F401
        available.append("hermes")
    except ImportError:
        pass
    return available


# ---------------------------------------------------------------------------
# Runtime layout / path helpers
# ---------------------------------------------------------------------------

def ensure_runtime_layout(root: Path) -> None:
    megaplan_rt = root / ".megaplan"
    (megaplan_rt / "plans").mkdir(parents=True, exist_ok=True)
    schemas_dir = megaplan_rt / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in SCHEMAS.items():
        atomic_write_json(schemas_dir / filename, strict_schema(schema))


def megaplan_root(root: Path) -> Path:
    return root / ".megaplan"


def plans_root(root: Path) -> Path:
    return megaplan_root(root) / "plans"


def schemas_root(root: Path) -> Path:
    return megaplan_root(root) / "schemas"


def artifact_path(plan_dir: Path, filename: str) -> Path:
    return plan_dir / filename


def batch_artifact_path(plan_dir: Path, batch_number: int) -> Path:
    return plan_dir / f"execution_batch_{batch_number}.json"


def list_batch_artifacts(plan_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"execution_batch_(\d+)\.json", path.name)
        if match is None:
            raise ValueError(f"Unexpected batch artifact filename: {path.name}")
        return (int(match.group(1)), path.name)

    return sorted(
        (
            path
            for path in plan_dir.glob("execution_batch_*.json")
            if path.is_file() and re.fullmatch(r"execution_batch_(\d+)\.json", path.name)
        ),
        key=sort_key,
    )


def current_iteration_artifact(plan_dir: Path, prefix: str, iteration: int) -> Path:
    return plan_dir / f"{prefix}_v{iteration}.json"


def current_iteration_raw_artifact(plan_dir: Path, prefix: str, iteration: int) -> Path:
    return plan_dir / f"{prefix}_v{iteration}_raw.txt"


# ---------------------------------------------------------------------------
# Plan state helpers
# ---------------------------------------------------------------------------

def active_plan_dirs(root: Path) -> list[Path]:
    if not plans_root(root).exists():
        return []
    directories: list[Path] = []
    for child in plans_root(root).iterdir():
        if child.is_dir() and (child / "state.json").exists():
            directories.append(child)
    return sorted(directories)


def resolve_plan_dir(root: Path, requested_name: str | None) -> Path:
    plan_dirs = active_plan_dirs(root)
    if requested_name:
        plan_dir = plans_root(root) / requested_name
        if not (plan_dir / "state.json").exists():
            raise CliError("missing_plan", f"Plan '{requested_name}' does not exist")
        return plan_dir
    if not plan_dirs:
        raise CliError("missing_plan", "No plans found. Run init first.")
    active = []
    for plan_dir in plan_dirs:
        state = read_json(plan_dir / "state.json")
        if state.get("current_state") not in TERMINAL_STATES:
            active.append(plan_dir)
    if len(active) == 1:
        return active[0]
    if len(plan_dirs) == 1:
        return plan_dirs[0]
    names = [path.name for path in active or plan_dirs]
    raise CliError(
        "ambiguous_plan",
        "Multiple plans exist; pass --plan explicitly",
        extra={"plans": names},
    )


def load_plan(root: Path, requested_name: str | None) -> tuple[Path, PlanState]:
    plan_dir = resolve_plan_dir(root, requested_name)
    state = read_json(plan_dir / "state.json")
    migrated = False
    if state.get("current_state") == "clarified":
        state["current_state"] = STATE_INITIALIZED
        migrated = True
    elif state.get("current_state") == "evaluated":
        state["current_state"] = STATE_CRITIQUED
        state["last_gate"] = {}
        migrated = True
    if "last_evaluation" in state:
        del state["last_evaluation"]
        migrated = True
    if "last_gate" not in state:
        state["last_gate"] = {}
        migrated = True
    if migrated:
        atomic_write_json(plan_dir / "state.json", state)
    return plan_dir, state


def save_state(plan_dir: Path, state: PlanState) -> None:
    atomic_write_json(plan_dir / "state.json", state)


def latest_plan_record(state: PlanState) -> PlanVersionRecord:
    plan_versions = state["plan_versions"]
    if not plan_versions:
        raise CliError("missing_plan_version", "No plan version exists yet")
    return plan_versions[-1]


def latest_plan_path(plan_dir: Path, state: PlanState) -> Path:
    return plan_dir / latest_plan_record(state)["file"]


def latest_plan_meta_path(plan_dir: Path, state: PlanState) -> Path:
    record = latest_plan_record(state)
    meta_name = record["file"].replace(".md", ".meta.json")
    return plan_dir / meta_name


# ---------------------------------------------------------------------------
# Flag registry
# ---------------------------------------------------------------------------

def load_flag_registry(plan_dir: Path) -> FlagRegistry:
    path = plan_dir / "faults.json"
    if path.exists():
        return read_json(path)
    return {"flags": []}


def save_flag_registry(plan_dir: Path, registry: FlagRegistry) -> None:
    atomic_write_json(plan_dir / "faults.json", registry)


def load_debt_registry(root: Path) -> DebtRegistry:
    path = megaplan_root(root) / "debt.json"
    if path.exists():
        return read_json(path)
    return {"entries": []}


def save_debt_registry(root: Path, registry: DebtRegistry) -> None:
    atomic_write_json(megaplan_root(root) / "debt.json", registry)


def next_debt_id(registry: DebtRegistry) -> str:
    max_id = 0
    for entry in registry["entries"]:
        match = re.fullmatch(r"DEBT-(\d+)", entry["id"])
        if match is None:
            continue
        max_id = max(max_id, int(match.group(1)))
    return f"DEBT-{max_id + 1:03d}"


def _normalize_subsystem_tag(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "untagged"


def extract_subsystem_tag(concern: str) -> str:
    prefix, separator, _ = concern.partition(":")
    if not separator:
        return "untagged"
    return _normalize_subsystem_tag(prefix)


def _concern_word_set(concern: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", normalize_text(concern))
        if token
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def find_matching_debt(registry: DebtRegistry, subsystem: str, concern: str) -> DebtEntry | None:
    normalized_subsystem = _normalize_subsystem_tag(subsystem)
    concern_words = _concern_word_set(concern)
    for entry in registry["entries"]:
        if entry["resolved"]:
            continue
        if entry["subsystem"] != normalized_subsystem:
            continue
        if _jaccard_similarity(_concern_word_set(entry["concern"]), concern_words) > 0.5:
            return entry
    return None


def add_or_increment_debt(
    registry: DebtRegistry,
    subsystem: str,
    concern: str,
    flag_ids: list[str],
    plan_id: str,
) -> DebtEntry:
    normalized_subsystem = _normalize_subsystem_tag(subsystem)
    normalized_concern = normalize_text(concern)
    timestamp = now_utc()
    existing = find_matching_debt(registry, normalized_subsystem, normalized_concern)
    if existing is not None:
        existing["occurrence_count"] += 1
        existing["updated_at"] = timestamp
        for flag_id in flag_ids:
            if flag_id not in existing["flag_ids"]:
                existing["flag_ids"].append(flag_id)
        if plan_id not in existing["plan_ids"]:
            existing["plan_ids"].append(plan_id)
        return existing

    entry: DebtEntry = {
        "id": next_debt_id(registry),
        "subsystem": normalized_subsystem,
        "concern": normalized_concern,
        "flag_ids": list(dict.fromkeys(flag_ids)),
        "plan_ids": [plan_id],
        "occurrence_count": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "resolved": False,
        "resolved_by": None,
        "resolved_at": None,
    }
    registry["entries"].append(entry)
    return entry


def resolve_debt(registry: DebtRegistry, debt_id: str, plan_id: str) -> DebtEntry:
    for entry in registry["entries"]:
        if entry["id"] != debt_id:
            continue
        timestamp = now_utc()
        entry["resolved"] = True
        entry["resolved_by"] = plan_id
        entry["resolved_at"] = timestamp
        entry["updated_at"] = timestamp
        return entry
    raise CliError("missing_debt", f"Debt entry '{debt_id}' does not exist")


def debt_by_subsystem(registry: DebtRegistry) -> dict[str, list[DebtEntry]]:
    grouped: dict[str, list[DebtEntry]] = {}
    for entry in registry["entries"]:
        if entry["resolved"]:
            continue
        grouped.setdefault(entry["subsystem"], []).append(entry)
    return grouped


def subsystem_occurrence_total(entries: list[DebtEntry]) -> int:
    return sum(entry["occurrence_count"] for entry in entries)


def escalated_subsystems(registry: DebtRegistry) -> list[tuple[str, int, list[DebtEntry]]]:
    escalated: list[tuple[str, int, list[DebtEntry]]] = []
    for subsystem, entries in debt_by_subsystem(registry).items():
        total = subsystem_occurrence_total(entries)
        if total >= DEBT_ESCALATION_THRESHOLD:
            escalated.append((subsystem, total, entries))
    escalated.sort(key=lambda item: (-item[1], item[0]))
    return escalated


def unresolved_significant_flags(flag_registry: FlagRegistry) -> list[FlagRecord]:
    return [
        flag
        for flag in flag_registry["flags"]
        if flag.get("severity") == "significant" and flag["status"] in FLAG_BLOCKING_STATUSES
    ]


def is_scope_creep_flag(flag: FlagRecord) -> bool:
    text = f"{flag['concern']} {flag.get('evidence', '')}".lower()
    return any(term in text for term in SCOPE_CREEP_TERMS)


def scope_creep_flags(
    flag_registry: FlagRegistry,
    *,
    statuses: set[str] | None = None,
) -> list[FlagRecord]:
    matches = []
    for flag in flag_registry["flags"]:
        if statuses is not None and flag["status"] not in statuses:
            continue
        if is_scope_creep_flag(flag):
            matches.append(flag)
    return matches


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
# Git diff summary (used by prompts)
# ---------------------------------------------------------------------------

def collect_git_diff_summary(project_dir: Path) -> str:
    if not (project_dir / ".git").exists():
        return "Project directory is not a git repository."
    try:
        process = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(project_dir),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "git not found on PATH."
    except subprocess.TimeoutExpired:
        return "git status timed out."
    if process.returncode != 0:
        return f"Unable to read git status: {process.stderr.strip() or process.stdout.strip()}"
    return process.stdout.strip() or "No git changes detected."


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

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
        Transition("review", STATE_DONE),
    ],
}

# Each level's *own* overrides (not inherited).  Levels inherit from the
# level below them via _ROBUSTNESS_HIERARCHY so shared transitions are
# declared once: heavy has none, standard keeps the planned->critique
# routing documented explicitly, and light adds gate/review skips.
_ROBUSTNESS_OVERRIDES: dict[str, dict[str, list[Transition]]] = {
    "heavy": {},
    "standard": {
        STATE_INITIALIZED: [
            Transition("plan", STATE_PLANNED),
        ],
        STATE_PLANNED: [
            Transition("critique", STATE_CRITIQUED),
            Transition("plan", STATE_PLANNED),
        ],
    },
    "light": {
        STATE_CRITIQUED: [
            Transition("revise", STATE_GATED),
        ],
        STATE_EXECUTED: [],
    },
}

# Ordered from most to least rigorous.  Each level inherits all overrides
# from every level to its right (i.e. less rigorous levels accumulate).
_ROBUSTNESS_HIERARCHY: tuple[str, ...] = ("heavy", "standard", "light")

_STEP_CONTEXT_STATES = {
    STATE_PLANNED,
    STATE_CRITIQUED,
    STATE_GATED,
    STATE_FINALIZED,
}


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
    # Compose overrides: start from the target level and accumulate every
    # level above it in the hierarchy (more rigorous levels' overrides are
    # included because less rigorous levels inherit them).
    merged: dict[str, list[Transition]] = {}
    try:
        target_index = _ROBUSTNESS_HIERARCHY.index(normalized)
    except ValueError:
        target_index = _ROBUSTNESS_HIERARCHY.index("standard")
    for level in _ROBUSTNESS_HIERARCHY[: target_index + 1]:
        merged.update(_ROBUSTNESS_OVERRIDES.get(level, {}))
    return {**WORKFLOW, **merged}


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


def find_command(name: str) -> str | None:
    return shutil.which(name)
