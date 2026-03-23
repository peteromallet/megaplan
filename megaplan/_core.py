"""Shared utilities, types, and constants for megaplan.

This module exists to break circular dependencies.  Every symbol that
workers.py, prompts.py, or evaluation.py needs from the package lives
here so that those modules can import at the top level without pulling
in cli.py (which re-exports from them).
"""

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
from typing import Any, NotRequired, TypedDict

from megaplan.schemas import SCHEMAS, strict_schema  # noqa: F401


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

STATE_INITIALIZED = "initialized"
STATE_PLANNED = "planned"
STATE_CRITIQUED = "critiqued"
STATE_GATED = "gated"
STATE_EXECUTED = "executed"
STATE_DONE = "done"
STATE_ABORTED = "aborted"
TERMINAL_STATES = {STATE_DONE, STATE_ABORTED}


class PlanConfig(TypedDict, total=False):
    project_dir: str
    auto_approve: bool
    robustness: str
    agents: dict[str, str]


class PlanMeta(TypedDict, total=False):
    significant_counts: list[int]
    weighted_scores: list[float]
    plan_deltas: list[float | None]
    recurring_critiques: list[str]
    total_cost_usd: float
    overrides: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    user_approved_gate: bool


class PlanState(TypedDict):
    name: str
    idea: str
    current_state: str
    iteration: int
    created_at: str
    config: PlanConfig
    sessions: dict[str, SessionInfo]
    plan_versions: list[PlanVersionRecord]
    history: list[HistoryEntry]
    meta: PlanMeta
    last_gate: dict[str, Any]
    clarification: NotRequired[dict[str, Any]]


class FlagRecord(TypedDict, total=False):
    id: str
    concern: str
    category: str
    severity_hint: str
    evidence: str
    raised_in: str
    status: str
    severity: str
    verified: bool
    verified_in: str
    addressed_in: str


class SessionInfo(TypedDict, total=False):
    id: str
    mode: str
    created_at: str
    last_used_at: str
    refreshed: bool


class PlanVersionRecord(TypedDict, total=False):
    version: int
    file: str
    hash: str
    timestamp: str


class HistoryEntry(TypedDict, total=False):
    step: str
    timestamp: str
    duration_ms: int
    cost_usd: float
    result: str
    session_mode: str
    session_id: str
    agent: str
    output_file: str
    artifact_hash: str
    raw_output_file: str
    message: str
    flags_count: int
    flags_addressed: list[str]
    recommendation: str
    approval_mode: str
    environment: dict[str, bool]


class FlagRegistry(TypedDict):
    flags: list[FlagRecord]


class GateSignals(TypedDict, total=False):
    """Typed result from build_gate_signals()."""
    robustness: str
    signals: dict[str, Any]
    warnings: list[str]


class StepResponse(TypedDict, total=False):
    """Typed response dict returned by all handler functions."""
    success: bool
    step: str
    summary: str
    artifacts: list[str]
    next_step: str | None
    state: str
    # Fields returned by specific handlers
    auto_approve: bool
    robustness: str
    iteration: int
    plan: str
    plan_dir: str
    questions: list[str]
    verified_flags: list[str]
    open_flags: list[str]
    scope_creep_flags: list[str]
    warnings: list[str]
    files_changed: list[str]
    deviations: list[str]
    user_approved_gate: bool
    issues: list[str]
    valid_next: list[str]
    # Override / setup / config responses
    mode: str
    installed: list[dict[str, Any]]
    config_path: str
    routing: dict[str, str]
    raw_config: dict[str, Any]
    action: str
    key: str
    value: str
    skipped: bool
    file: str
    plans: list[dict[str, Any]]
    # Gate worker / summary fields
    recommendation: str
    signals: dict[str, Any]
    rationale: str
    gate_recommendation: str
    gate_rationale: str
    signals_assessment: str
    # Gate check fields (spread via **gate)
    passed: bool
    criteria_check: dict[str, Any]
    preflight_results: dict[str, bool]
    unresolved_flags: list[Any]
    # Error response fields
    error: str
    message: str
    details: dict[str, Any]
    # Fallback
    agent_fallback: dict[str, str]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLAG_BLOCKING_STATUSES = {"open", "disputed"}
MOCK_ENV_VAR = "MEGAPLAN_MOCK_WORKERS"

DEFAULT_AGENT_ROUTING: dict[str, str] = {
    "plan": "claude",
    "critique": "codex",
    "revise": "claude",
    "gate": "claude",
    "execute": "codex",
    "review": "codex",
}
KNOWN_AGENTS = ["claude", "codex"]
ROBUSTNESS_LEVELS = ("light", "standard", "thorough")
ROBUSTNESS_SKIP_THRESHOLDS = {"light": 4.0, "standard": 2.0, "thorough": 1.0}
ROBUSTNESS_STAGNATION_FACTORS = {"light": 0.8, "standard": 0.9, "thorough": 0.95}
SCOPE_CREEP_TERMS = (
    "scope creep",
    "out of scope",
    "beyond the original idea",
    "beyond original idea",
    "beyond user intent",
    "expanded scope",
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class CliError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        valid_next: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        exit_code: int = 1,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.valid_next = valid_next or []
        self.extra = extra or {}
        self.exit_code = exit_code


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
        if flag.get("severity") == "significant" and flag.get("status") in FLAG_BLOCKING_STATUSES
    ]


def is_scope_creep_flag(flag: FlagRecord) -> bool:
    text = f"{flag.get('concern', '')} {flag.get('evidence', '')}".lower()
    return any(term in text for term in SCOPE_CREEP_TERMS)


def scope_creep_flags(
    flag_registry: FlagRegistry,
    *,
    statuses: set[str] | None = None,
) -> list[FlagRecord]:
    matches = []
    for flag in flag_registry["flags"]:
        if statuses is not None and flag.get("status") not in statuses:
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
