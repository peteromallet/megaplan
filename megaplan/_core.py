"""Shared utilities for megaplan — I/O, config, plan resolution, flag helpers, prompt helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from megaplan.schemas import SCHEMAS, strict_schema
from megaplan.types import (
    CliError,
    FLAG_BLOCKING_STATUSES,
    FlagRecord,
    FlagRegistry,
    KNOWN_AGENTS,
    PlanState,
    PlanVersionRecord,
    ROBUSTNESS_LEVELS,
    SCOPE_CREEP_TERMS,
    STATE_CRITIQUED,
    STATE_EXECUTED,
    STATE_FINALIZED,
    STATE_GATED,
    STATE_INITIALIZED,
    STATE_PLANNED,
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


def render_final_md(finalize_data: dict[str, Any]) -> str:
    lines = ["# Execution Checklist", ""]
    for task in finalize_data.get("tasks", []):
        checkbox = "[x]" if task.get("status") == "done" else "[ ]"
        status_suffix = " (skipped)" if task.get("status") == "skipped" else ""
        lines.append(f"- {checkbox} **{task['id']}:** {task['description']}{status_suffix}")
        depends_on = task.get("depends_on", [])
        if depends_on:
            lines.append(f"  Depends on: {', '.join(depends_on)}")
        executor_notes = task.get("executor_notes", "")
        if executor_notes:
            lines.append(f"  Executor notes: {executor_notes}")
        reviewer_verdict = task.get("reviewer_verdict", "")
        if reviewer_verdict:
            lines.append(f"  Reviewer verdict: {reviewer_verdict}")
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
    sense_checks = finalize_data.get("sense_checks", [])
    if sense_checks:
        for sense_check in sense_checks:
            lines.append(f"- **{sense_check['id']}** ({sense_check['task_id']}): {sense_check['question']}")
            verdict = sense_check.get("verdict", "")
            if verdict:
                lines.append(f"  Verdict: {verdict}")
            lines.append("")
    else:
        lines.extend(["- None.", ""])

    lines.extend(["## Meta", ""])
    meta_commentary = finalize_data.get("meta_commentary", "").strip()
    lines.append(meta_commentary or "None.")
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
    return [a for a in KNOWN_AGENTS if shutil.which(a)]


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
    if robustness == "thorough":
        return "Be exhaustive. Flag edge cases, missing error handling, performance concerns, and anything that could cause problems in production even if unlikely."
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

def infer_next_steps(state: PlanState) -> list[str]:
    current = state["current_state"]
    if current == STATE_INITIALIZED:
        return ["plan"]
    if current == STATE_PLANNED:
        return ["plan", "critique", "step"]
    if current == STATE_CRITIQUED:
        gate = state.get("last_gate", {})
        recommendation = gate.get("recommendation")
        if not recommendation:
            return ["gate", "step"]
        if recommendation == "ITERATE":
            return ["revise", "step"]
        if recommendation == "ESCALATE":
            return ["override add-note", "override force-proceed", "override abort", "step"]
        if recommendation == "PROCEED" and not gate.get("passed", False):
            return ["revise", "override force-proceed", "step"]
        return ["gate", "step"]
    if current == STATE_GATED:
        return ["finalize", "override replan", "step"]
    if current == STATE_FINALIZED:
        return ["execute", "override replan", "step"]
    if current == STATE_EXECUTED:
        return ["review"]
    return []


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
