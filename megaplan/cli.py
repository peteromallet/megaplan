#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import uuid
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from enum import Enum
from typing import Any, NotRequired, TypedDict

# Re-export from sub-modules so that everything remains importable from megaplan.cli
from megaplan.schemas import SCHEMAS, strict_schema  # noqa: F401
from megaplan.evaluation import (  # noqa: F401
    build_evaluation,
    flag_weight,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    ROBUSTNESS_SKIP_THRESHOLDS,
    ROBUSTNESS_STAGNATION_FACTORS,
    _EVALUATION_DECISION_TABLE,
    _is_over_budget,
    _is_all_flags_resolved,
    _is_low_weight_trending_down,
    _is_stagnant_with_unresolved,
    _is_stagnant_all_addressed,
    _is_first_iteration_with_flags,
    _has_recurring_critiques,
    _is_score_stagnating,
    _is_score_improving,
    _is_max_iterations_with_unresolved,
)
from megaplan.workers import (  # noqa: F401
    CommandResult,
    WorkerResult,
    run_command,
    run_claude_step,
    run_codex_step,
    run_step_with_worker,
    extract_session_id,
    parse_claude_envelope,
    parse_json_file,
    validate_payload,
    mock_worker_output,
    session_key_for,
    persist_session,
    resolve_agent_mode,
    WORKER_TIMEOUT_SECONDS,
)
from megaplan.prompts import (  # noqa: F401
    create_claude_prompt,
    create_codex_prompt,
    _clarify_prompt,
    _plan_prompt,
    _integrate_prompt,
    _critique_prompt,
    _execute_prompt,
    _review_claude_prompt,
    _review_codex_prompt,
    _CLAUDE_PROMPT_BUILDERS,
    _CODEX_PROMPT_BUILDERS,
)

__all__ = [
    "PlanStage", "PlanState", "PlanConfig", "PlanMeta", "FlagRecord",
    "STATE_INITIALIZED", "STATE_CLARIFIED", "STATE_PLANNED", "STATE_CRITIQUED", "STATE_EVALUATED",
    "STATE_GATED", "STATE_EXECUTED", "STATE_DONE", "STATE_ABORTED",
    "TERMINAL_STATES", "FLAG_BLOCKING_STATUSES", "MOCK_ENV_VAR",
    "CliError", "CommandResult", "WorkerResult",
    "slugify", "compute_plan_delta_percent", "normalize_flag_record",
    "unresolved_significant_flags", "flag_weight",
    "update_flags_after_critique", "update_flags_after_integrate",
    "compute_recurring_critiques", "build_evaluation",
    "infer_next_steps", "require_state",
    "handle_init", "handle_clarify", "handle_plan", "handle_critique", "handle_evaluate",
    "handle_integrate", "handle_gate", "handle_execute", "handle_review",
    "handle_status", "handle_audit", "handle_list", "handle_override",
    "handle_setup", "handle_setup_global", "handle_config",
    "load_flag_registry", "save_flag_registry",
    "load_config", "save_config", "config_dir", "detect_available_agents",
    "DEFAULT_AGENT_ROUTING",
    "plans_root", "main", "cli_entry",
]


class PlanStage(str, Enum):
    INITIALIZED = "initialized"
    CLARIFIED = "clarified"
    PLANNED = "planned"
    CRITIQUED = "critiqued"
    EVALUATED = "evaluated"
    GATED = "gated"
    EXECUTED = "executed"
    DONE = "done"
    ABORTED = "aborted"


# Backward-compatible aliases
STATE_INITIALIZED = PlanStage.INITIALIZED
STATE_CLARIFIED = PlanStage.CLARIFIED
STATE_PLANNED = PlanStage.PLANNED
STATE_CRITIQUED = PlanStage.CRITIQUED
STATE_EVALUATED = PlanStage.EVALUATED
STATE_GATED = PlanStage.GATED
STATE_EXECUTED = PlanStage.EXECUTED
STATE_DONE = PlanStage.DONE
STATE_ABORTED = PlanStage.ABORTED
TERMINAL_STATES = {STATE_DONE, STATE_ABORTED}


class PlanConfig(TypedDict):
    max_iterations: int
    budget_usd: float
    project_dir: str
    auto_approve: bool
    robustness: str


class PlanMeta(TypedDict, total=False):
    significant_counts: list[int]
    weighted_scores: list[float]
    plan_deltas: list[float | None]
    recurring_critiques: list[str]
    total_cost_usd: float
    overrides: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    user_approved_gate: bool


class PlanState(TypedDict, total=False):
    name: str
    idea: str
    current_state: str
    iteration: int
    created_at: str
    config: PlanConfig
    sessions: dict[str, Any]
    plan_versions: list[dict[str, Any]]
    history: list[dict[str, Any]]
    meta: PlanMeta
    last_evaluation: dict[str, Any]
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
FLAG_BLOCKING_STATUSES = {"open", "disputed"}
MOCK_ENV_VAR = "MEGAPLAN_MOCK_WORKERS"

DEFAULT_AGENT_ROUTING: dict[str, str] = {
    "clarify": "claude",
    "plan": "claude",
    "critique": "codex",
    "integrate": "claude",
    "execute": "codex",
    "review": "codex",
}
KNOWN_AGENTS = ["claude", "codex"]
ROBUSTNESS_LEVELS = ("light", "standard", "thorough")
SCOPE_CREEP_TERMS = (
    "scope creep",
    "out of scope",
    "beyond the original idea",
    "beyond original idea",
    "beyond user intent",
    "expanded scope",
)


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
    except (json.JSONDecodeError, ValueError):
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


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False) + "\n"


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


def sha256_text(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def ensure_runtime_layout(root: Path) -> None:
    megaplan_root = root / ".megaplan"
    (megaplan_root / "plans").mkdir(parents=True, exist_ok=True)
    schemas_dir = megaplan_root / "schemas"
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


def append_history(state: dict[str, Any], entry: dict[str, Any]) -> None:
    state.setdefault("history", []).append(entry)
    state.setdefault("meta", {}).setdefault("total_cost_usd", 0.0)
    state["meta"]["total_cost_usd"] = round(
        float(state["meta"]["total_cost_usd"]) + float(entry.get("cost_usd", 0.0)),
        6,
    )


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


def load_plan(root: Path, requested_name: str | None) -> tuple[Path, dict[str, Any]]:
    plan_dir = resolve_plan_dir(root, requested_name)
    return plan_dir, read_json(plan_dir / "state.json")


def save_state(plan_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(plan_dir / "state.json", state)


def render_response(data: dict[str, Any], *, exit_code: int = 0) -> int:
    print(json_dump(data), end="")
    return exit_code


def error_response(error: CliError) -> int:
    payload = {
        "success": False,
        "error": error.code,
        "message": error.message,
    }
    if error.valid_next:
        payload["valid_next"] = error.valid_next
    if error.extra:
        payload["details"] = error.extra
    return render_response(payload, exit_code=error.exit_code)


def latest_plan_record(state: dict[str, Any]) -> dict[str, Any]:
    plan_versions = state.get("plan_versions", [])
    if not plan_versions:
        raise CliError("missing_plan_version", "No plan version exists yet")
    return plan_versions[-1]


def latest_plan_path(plan_dir: Path, state: dict[str, Any]) -> Path:
    return plan_dir / latest_plan_record(state)["file"]


def latest_plan_meta_path(plan_dir: Path, state: dict[str, Any]) -> Path:
    record = latest_plan_record(state)
    meta_name = record["file"].replace(".md", ".meta.json")
    return plan_dir / meta_name


def load_flag_registry(plan_dir: Path) -> dict[str, Any]:
    path = plan_dir / "faults.json"
    if path.exists():
        return read_json(path)
    return {"flags": []}


def save_flag_registry(plan_dir: Path, data: dict[str, Any]) -> None:
    atomic_write_json(plan_dir / "faults.json", data)


def next_flag_number(flags: list[dict[str, Any]]) -> int:
    highest = 0
    for flag in flags:
        match = re.fullmatch(r"FLAG-(\d+)", str(flag.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def make_flag_id(number: int) -> str:
    return f"FLAG-{number:03d}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_severity(hint: str) -> str:
    """Map a severity_hint to a resolved severity value."""
    if hint == "likely-significant":
        return "significant"
    if hint == "likely-minor":
        return "minor"
    return "significant"


def make_history_entry(
    step: str,
    *,
    duration_ms: int,
    cost_usd: float,
    result: str,
    worker: "WorkerResult | None" = None,
    agent: str | None = None,
    mode: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a history entry dict with common fields plus step-specific extras."""
    entry: dict[str, Any] = {
        "step": step,
        "timestamp": now_utc(),
        "duration_ms": duration_ms,
        "cost_usd": cost_usd,
        "result": result,
    }
    if worker is not None and agent is not None and mode is not None:
        entry["session_mode"] = mode
        entry["session_id"] = worker.session_id
        entry["agent"] = agent
    entry.update(extra)
    return entry


def attach_agent_fallback(response: dict[str, Any], args: argparse.Namespace) -> None:
    """Copy agent fallback info onto a response dict if present."""
    if hasattr(args, "_agent_fallback"):
        response["agent_fallback"] = args._agent_fallback


def unresolved_significant_flags(flag_registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        flag
        for flag in flag_registry.get("flags", [])
        if flag.get("severity") == "significant" and flag.get("status") in FLAG_BLOCKING_STATUSES
    ]


def current_iteration_artifact(plan_dir: Path, prefix: str, iteration: int) -> Path:
    return plan_dir / f"{prefix}_v{iteration}.json"


def current_iteration_raw_artifact(plan_dir: Path, prefix: str, iteration: int) -> Path:
    return plan_dir / f"{prefix}_v{iteration}_raw.txt"


def infer_next_steps(state: dict[str, Any]) -> list[str]:
    current = state.get("current_state")
    if current == STATE_INITIALIZED:
        return ["clarify"]
    if current == STATE_CLARIFIED:
        return ["clarify", "plan"]
    if current == STATE_PLANNED:
        return ["critique"]
    if current == STATE_CRITIQUED:
        return ["evaluate"]
    if current == STATE_EVALUATED:
        evaluation = state.get("last_evaluation", {})
        recommendation = evaluation.get("recommendation")
        valid = []
        if recommendation == "CONTINUE":
            valid.append("integrate")
        if recommendation in {"SKIP", "CONTINUE"}:
            valid.append("gate")
        if recommendation in {"ESCALATE", "ABORT"}:
            valid.extend(["override add-note", "override force-proceed", "override abort"])
        return valid or ["override add-note", "override abort"]
    if current == STATE_GATED:
        return ["execute"]
    if current == STATE_EXECUTED:
        return ["review"]
    return []


def require_state(state: dict[str, Any], step: str, allowed: set[str]) -> None:
    current = state.get("current_state")
    if current not in allowed:
        raise CliError(
            "invalid_transition",
            f"Cannot run '{step}' while current state is '{current}'",
            valid_next=infer_next_steps(state),
            extra={"current_state": current},
        )


def find_command(name: str) -> str | None:
    return shutil.which(name)


def intent_and_notes_block(state: dict[str, Any]) -> str:
    sections = []
    clarification = state.get("clarification", {})
    if clarification.get("intent_summary"):
        sections.append(f"User intent summary:\n{clarification['intent_summary']}")
        sections.append(f"Original idea:\n{state['idea']}")
    else:
        sections.append(f"Idea:\n{state['idea']}")
    notes = state.get("meta", {}).get("notes", [])
    if notes:
        notes_text = "\n".join(f"- {note['note']}" for note in notes)
        sections.append(f"User notes and answers:\n{notes_text}")
    return "\n\n".join(sections)


def configured_robustness(state: dict[str, Any]) -> str:
    robustness = state.get("config", {}).get("robustness", "standard")
    if robustness not in ROBUSTNESS_LEVELS:
        return "standard"
    return robustness


def robustness_critique_instruction(robustness: str) -> str:
    if robustness == "light":
        return "Be pragmatic. Only flag issues that would cause real failures. Ignore style, minor edge cases, and issues the executor will naturally resolve."
    if robustness == "thorough":
        return "Be exhaustive. Flag edge cases, missing error handling, performance concerns, and anything that could cause problems in production even if unlikely."
    return "Use balanced judgment. Flag significant risks, but do not spend flags on minor polish or executor-obvious boilerplate."


def is_scope_creep_flag(flag: dict[str, Any]) -> bool:
    text = f"{flag.get('concern', '')} {flag.get('evidence', '')}".lower()
    return any(term in text for term in SCOPE_CREEP_TERMS)


def scope_creep_flags(
    flag_registry: dict[str, Any],
    *,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    matches = []
    for flag in flag_registry.get("flags", []):
        if statuses is not None and flag.get("status") not in statuses:
            continue
        if is_scope_creep_flag(flag):
            matches.append(flag)
    return matches


def collect_git_diff_summary(project_dir: Path) -> str:
    if not (project_dir / ".git").exists():
        return "Project directory is not a git repository."
    diff = run_command(["git", "status", "--short"], cwd=project_dir)
    if diff.returncode != 0:
        return f"Unable to read git status: {diff.stderr.strip() or diff.stdout.strip()}"
    return diff.stdout.strip() or "No git changes detected."


def normalize_flag_record(item: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    category = item.get("category", "other")
    if category not in {"correctness", "security", "completeness", "performance", "maintainability", "other"}:
        category = "other"
    severity_hint = item.get("severity_hint") or "uncertain"
    if severity_hint not in {"likely-significant", "likely-minor", "uncertain"}:
        severity_hint = "uncertain"
    raw_id = item.get("id")
    return {
        "id": fallback_id if raw_id in {None, "", "FLAG-000"} else raw_id,
        "concern": item.get("concern", "").strip(),
        "category": category,
        "severity_hint": severity_hint,
        "evidence": item.get("evidence", "").strip(),
    }


def update_flags_after_critique(plan_dir: Path, critique: dict[str, Any], *, iteration: int) -> dict[str, Any]:
    registry = load_flag_registry(plan_dir)
    flags = registry.setdefault("flags", [])
    by_id = {flag["id"]: flag for flag in flags}
    next_number = next_flag_number(flags)

    for verified_id in critique.get("verified_flag_ids", []):
        if verified_id in by_id:
            by_id[verified_id]["status"] = "verified"
            by_id[verified_id]["verified"] = True
            by_id[verified_id]["verified_in"] = f"critique_v{iteration}.json"

    for disputed_id in critique.get("disputed_flag_ids", []):
        if disputed_id in by_id:
            by_id[disputed_id]["status"] = "disputed"

    for raw_flag in critique.get("flags", []):
        proposed_id = raw_flag.get("id")
        if not proposed_id or proposed_id in {"", "FLAG-000"}:
            proposed_id = make_flag_id(next_number)
            next_number += 1
        normalized = normalize_flag_record(raw_flag, proposed_id)
        if normalized["id"] in by_id:
            existing = by_id[normalized["id"]]
            existing.update(normalized)
            existing["status"] = "open"
            existing["severity"] = resolve_severity(normalized["severity_hint"])
            existing["raised_in"] = f"critique_v{iteration}.json"
        else:
            severity = resolve_severity(normalized["severity_hint"])
            created = {
                **normalized,
                "raised_in": f"critique_v{iteration}.json",
                "status": "open",
                "severity": severity,
                "verified": False,
            }
            flags.append(created)
            by_id[created["id"]] = created

    save_flag_registry(plan_dir, registry)
    return registry


def update_flags_after_integrate(plan_dir: Path, flags_addressed: list[str], *, plan_file: str, summary: str) -> dict[str, Any]:
    registry = load_flag_registry(plan_dir)
    for flag in registry.get("flags", []):
        if flag["id"] in flags_addressed:
            flag["status"] = "addressed"
            flag["addressed_in"] = plan_file
            flag["evidence"] = summary
    save_flag_registry(plan_dir, registry)
    return registry


def store_raw_worker_output(plan_dir: Path, step: str, iteration: int, content: str) -> str:
    filename = current_iteration_raw_artifact(plan_dir, step, iteration).name
    atomic_write_text(plan_dir / filename, content)
    return filename


def record_step_failure(
    plan_dir: Path,
    state: dict[str, Any],
    *,
    step: str,
    iteration: int,
    error: CliError,
    duration_ms: int = 0,
) -> None:
    raw_output = str(error.extra.get("raw_output") or error.message)
    raw_name = store_raw_worker_output(plan_dir, step, iteration, raw_output)
    append_history(
        state,
        make_history_entry(
            step,
            duration_ms=duration_ms,
            cost_usd=0.0,
            result="error",
            raw_output_file=raw_name,
            message=error.message,
        ),
    )
    save_state(plan_dir, state)


def handle_init(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_layout(root)
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise CliError("invalid_project_dir", f"Project directory does not exist: {project_dir}")
    robustness = getattr(args, "robustness", "standard")
    if robustness not in ROBUSTNESS_LEVELS:
        robustness = "standard"
    auto_approve = bool(getattr(args, "auto_approve", False))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    plan_name = args.name or f"{slugify(args.idea)}-{timestamp}"
    plan_dir = plans_root(root) / plan_name
    if plan_dir.exists():
        raise CliError("duplicate_plan", f"Plan directory already exists: {plan_name}")
    plan_dir.mkdir(parents=True, exist_ok=False)

    state = {
        "name": plan_name,
        "idea": args.idea,
        "current_state": STATE_INITIALIZED,
        "iteration": 0,
        "created_at": now_utc(),
        "config": {
            "max_iterations": args.max_iterations,
            "budget_usd": args.budget_usd,
            "project_dir": str(project_dir),
            "auto_approve": auto_approve,
            "robustness": robustness,
        },
        "sessions": {},
        "plan_versions": [],
        "history": [],
        "meta": {
            "significant_counts": [],
            "weighted_scores": [],
            "plan_deltas": [],
            "recurring_critiques": [],
            "total_cost_usd": 0.0,
            "overrides": [],
            "notes": [],
        },
        "last_evaluation": {},
    }
    append_history(
        state,
        make_history_entry(
            "init",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            environment={
                "claude": bool(find_command("claude")),
                "codex": bool(find_command("codex")),
            },
        ),
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "init",
        "plan": plan_name,
        "state": STATE_INITIALIZED,
        "summary": f"Initialized plan '{plan_name}' for project {project_dir}",
        "artifacts": ["state.json"],
        "next_step": "clarify",
        "auto_approve": auto_approve,
        "robustness": robustness,
    }


def handle_clarify(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "clarify", {STATE_INITIALIZED, STATE_CLARIFIED})
    try:
        worker, agent, mode, refreshed = run_step_with_worker("clarify", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="clarify", iteration=state["iteration"], error=error)
        raise
    payload = worker.payload
    clarify_filename = "clarify.json"
    atomic_write_json(plan_dir / clarify_filename, payload)
    state["clarification"] = {
        "refined_idea": payload["refined_idea"],
        "intent_summary": payload["intent_summary"],
        "questions": payload["questions"],
    }
    state["current_state"] = STATE_CLARIFIED
    persist_session(state, "clarify", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "clarify",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=clarify_filename,
            artifact_hash=sha256_file(plan_dir / clarify_filename),
        ),
    )
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "clarify",
        "summary": f"Captured clarification with {len(payload['questions'])} questions.",
        "artifacts": [clarify_filename],
        "next_step": "plan",
        "state": STATE_CLARIFIED,
        "questions": payload["questions"],
    }
    attach_agent_fallback(response, args)
    return response


def handle_plan(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "plan", {STATE_INITIALIZED, STATE_CLARIFIED})
    try:
        worker, agent, mode, refreshed = run_step_with_worker("plan", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="plan", iteration=state["iteration"] + 1, error=error)
        raise
    payload = worker.payload
    version = state["iteration"] + 1
    plan_filename = f"plan_v{version}.md"
    meta_filename = f"plan_v{version}.meta.json"
    plan_text = payload["plan"].rstrip() + "\n"
    atomic_write_text(plan_dir / plan_filename, plan_text)
    meta = {
        "version": version,
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "questions": payload["questions"],
        "success_criteria": payload["success_criteria"],
        "assumptions": payload["assumptions"],
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    state["iteration"] = version
    state["current_state"] = STATE_PLANNED
    state.setdefault("meta", {}).pop("user_approved_gate", None)
    state.setdefault("plan_versions", []).append({"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]})
    persist_session(state, "plan", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "plan",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=plan_filename,
            artifact_hash=meta["hash"],
        ),
    )
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "plan",
        "iteration": version,
        "summary": f"Generated plan v{version} with {len(payload['questions'])} questions and {len(payload['success_criteria'])} success criteria.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": "critique",
        "state": STATE_PLANNED,
    }
    attach_agent_fallback(response, args)
    return response


def handle_critique(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "critique", {STATE_PLANNED})
    iteration = state["iteration"]
    try:
        worker, agent, mode, refreshed = run_step_with_worker("critique", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="critique", iteration=iteration, error=error)
        raise
    critique_filename = f"critique_v{iteration}.json"
    atomic_write_json(plan_dir / critique_filename, worker.payload)
    registry = update_flags_after_critique(plan_dir, worker.payload, iteration=iteration)
    significant = len([f for f in registry.get("flags", []) if f.get("severity") == "significant" and f.get("status") in FLAG_BLOCKING_STATUSES])
    state.setdefault("meta", {}).setdefault("significant_counts", []).append(significant)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    state.setdefault("meta", {}).setdefault("recurring_critiques", []).append(recurring)
    state["current_state"] = STATE_CRITIQUED
    persist_session(state, "critique", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "critique",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=critique_filename,
            artifact_hash=sha256_file(plan_dir / critique_filename),
            flags_count=len(worker.payload.get("flags", [])),
        ),
    )
    save_state(plan_dir, state)
    scope_flags = scope_creep_flags(registry, statuses=FLAG_BLOCKING_STATUSES)
    response = {
        "success": True,
        "step": "critique",
        "iteration": iteration,
        "summary": f"Recorded {len(worker.payload.get('flags', []))} critique flags.",
        "artifacts": [critique_filename, "faults.json"],
        "next_step": "evaluate",
        "state": STATE_CRITIQUED,
        "verified_flags": worker.payload.get("verified_flag_ids", []),
        "open_flags": [flag["id"] for flag in registry.get("flags", []) if flag.get("status") == "open"],
        "scope_creep_flags": [flag["id"] for flag in scope_flags],
    }
    if scope_flags:
        response["warnings"] = [
            "Scope creep detected in the plan. Surface this drift to the user while continuing the loop."
        ]
    attach_agent_fallback(response, args)
    return response


def handle_evaluate(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "evaluate", {STATE_CRITIQUED})
    evaluation = build_evaluation(plan_dir, state)
    iteration = state["iteration"]
    state.setdefault("meta", {}).setdefault("weighted_scores", []).append(evaluation["signals"]["weighted_score"])
    filename = f"evaluation_v{iteration}.json"
    atomic_write_json(plan_dir / filename, evaluation)
    state["current_state"] = STATE_EVALUATED
    state["last_evaluation"] = evaluation
    append_history(
        state,
        make_history_entry(
            "evaluate",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file=filename,
            artifact_hash=sha256_file(plan_dir / filename),
            recommendation=evaluation["recommendation"],
        ),
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "evaluate",
        "iteration": iteration,
        "summary": f"Recommendation {evaluation['recommendation']}: {evaluation['rationale']}",
        "artifacts": [filename],
        "next_step": evaluation["valid_next_steps"][0] if evaluation["valid_next_steps"] else None,
        "state": STATE_EVALUATED,
        **evaluation,
    }


def handle_integrate(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "integrate", {STATE_EVALUATED})
    if state.get("last_evaluation", {}).get("recommendation") != "CONTINUE":
        raise CliError(
            "invalid_transition",
            "Integrate requires an evaluation recommendation of CONTINUE",
            valid_next=infer_next_steps(state),
        )
    previous_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    try:
        worker, agent, mode, refreshed = run_step_with_worker("integrate", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="integrate", iteration=state["iteration"] + 1, error=error)
        raise
    payload = worker.payload
    validate_payload("integrate", payload)
    version = state["iteration"] + 1
    plan_filename = f"plan_v{version}.md"
    meta_filename = f"plan_v{version}.meta.json"
    plan_text = payload["plan"].rstrip() + "\n"
    atomic_write_text(plan_dir / plan_filename, plan_text)
    delta = compute_plan_delta_percent(previous_plan, plan_text)
    meta = {
        "version": version,
        "timestamp": now_utc(),
        "hash": sha256_text(plan_text),
        "changes_summary": payload["changes_summary"],
        "flags_addressed": payload["flags_addressed"],
        "questions": payload.get("questions", []),
        "success_criteria": payload.get("success_criteria", []),
        "assumptions": payload.get("assumptions", []),
        "delta_from_previous_percent": delta,
    }
    atomic_write_json(plan_dir / meta_filename, meta)
    state["iteration"] = version
    state["current_state"] = STATE_PLANNED
    state.setdefault("meta", {}).pop("user_approved_gate", None)
    state.setdefault("plan_versions", []).append({"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]})
    state.setdefault("meta", {}).setdefault("plan_deltas", []).append(delta)
    update_flags_after_integrate(plan_dir, payload["flags_addressed"], plan_file=plan_filename, summary=payload["changes_summary"])
    persist_session(state, "integrate", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "integrate",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file=plan_filename,
            artifact_hash=meta["hash"],
            flags_addressed=payload["flags_addressed"],
        ),
    )
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "integrate",
        "iteration": version,
        "summary": f"Updated plan to v{version}; addressed {len(payload['flags_addressed'])} flags.",
        "artifacts": [plan_filename, meta_filename, "faults.json"],
        "next_step": "critique",
        "state": STATE_PLANNED,
    }
    attach_agent_fallback(response, args)
    return response


def run_gate_checks(plan_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    project_dir = Path(state["config"]["project_dir"])
    meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    checks = {
        "project_dir_exists": project_dir.exists(),
        "project_dir_writable": os.access(project_dir, os.W_OK),
        "success_criteria_present": bool(meta.get("success_criteria")),
        "claude_available": bool(find_command("claude")),
        "codex_available": bool(find_command("codex")),
    }
    passed = all(checks.values()) and not unresolved
    return {
        "passed": passed,
        "criteria_check": {"count": len(meta.get("success_criteria", [])), "items": meta.get("success_criteria", [])},
        "preflight_results": checks,
        "unresolved_flags": unresolved,
    }


def handle_gate(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "gate", {STATE_EVALUATED})
    recommendation = state.get("last_evaluation", {}).get("recommendation")
    if recommendation not in {"SKIP", "CONTINUE"}:
        raise CliError(
            "invalid_transition",
            f"Gate requires evaluation recommendation SKIP or CONTINUE, got {recommendation!r}",
            valid_next=infer_next_steps(state),
        )
    gate = run_gate_checks(plan_dir, state)
    atomic_write_json(plan_dir / "gate.json", gate)
    if not gate["passed"]:
        append_history(
            state,
            make_history_entry(
                "gate",
                duration_ms=0,
                cost_usd=0.0,
                result="blocked",
                output_file="gate.json",
                artifact_hash=sha256_file(plan_dir / "gate.json"),
            ),
        )
        save_state(plan_dir, state)
        return {
            "success": False,
            "step": "gate",
            "summary": "Gate blocked: unresolved flags or failed preflight checks remain.",
            "artifacts": ["gate.json"],
            "next_step": "integrate" if recommendation == "CONTINUE" else "override force-proceed",
            "state": state["current_state"],
            "auto_approve": bool(state.get("config", {}).get("auto_approve", False)),
            "robustness": configured_robustness(state),
            **gate,
        }
    final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    atomic_write_text(plan_dir / "final.md", final_plan)
    state["current_state"] = STATE_GATED
    state.setdefault("meta", {}).pop("user_approved_gate", None)
    append_history(
        state,
        make_history_entry(
            "gate",
            duration_ms=0,
            cost_usd=0.0,
            result="success",
            output_file="gate.json",
            artifact_hash=sha256_file(plan_dir / "gate.json"),
        ),
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "gate",
        "summary": "Gate passed. Plan is ready for execution.",
        "artifacts": ["gate.json", "final.md"],
        "next_step": "execute",
        "state": STATE_GATED,
        "auto_approve": bool(state.get("config", {}).get("auto_approve", False)),
        "robustness": configured_robustness(state),
        **gate,
    }


def handle_execute(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "execute", {STATE_GATED})
    if not args.confirm_destructive:
        raise CliError("missing_confirmation", "Execute requires --confirm-destructive")
    auto_approve = bool(state.get("config", {}).get("auto_approve", False))
    if getattr(args, "user_approved", False):
        state.setdefault("meta", {})["user_approved_gate"] = True
        save_state(plan_dir, state)
    if not auto_approve and not state.get("meta", {}).get("user_approved_gate", False):
        raise CliError(
            "missing_approval",
            "Execute requires explicit user approval (--user-approved) when auto-approve is not set. The orchestrator must confirm with the user at the gate checkpoint before proceeding.",
        )
    try:
        worker, agent, mode, refreshed = run_step_with_worker("execute", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="execute", iteration=state["iteration"], error=error)
        raise
    atomic_write_json(plan_dir / "execution.json", worker.payload)
    if worker.trace_output is not None:
        atomic_write_text(plan_dir / "execution_trace.jsonl", worker.trace_output)
    state["current_state"] = STATE_EXECUTED
    persist_session(state, "execute", agent, worker.session_id, mode=mode, refreshed=refreshed)
    if auto_approve:
        approval_mode = "auto_approve"
    elif state.get("meta", {}).get("user_approved_gate", False):
        approval_mode = "user_approved"
    else:
        approval_mode = "manual"
    append_history(
        state,
        make_history_entry(
            "execute",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="execution.json",
            artifact_hash=sha256_file(plan_dir / "execution.json"),
            approval_mode=approval_mode,
        ),
    )
    save_state(plan_dir, state)
    artifacts = ["execution.json"]
    if worker.trace_output is not None:
        artifacts.append("execution_trace.jsonl")
    response = {
        "success": True,
        "step": "execute",
        "summary": worker.payload["output"],
        "artifacts": artifacts,
        "next_step": "review",
        "state": STATE_EXECUTED,
        "files_changed": worker.payload.get("files_changed", []),
        "deviations": worker.payload.get("deviations", []),
        "auto_approve": auto_approve,
        "user_approved_gate": bool(state.get("meta", {}).get("user_approved_gate", False)),
    }
    attach_agent_fallback(response, args)
    return response


def handle_review(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "review", {STATE_EXECUTED})
    try:
        worker, agent, mode, refreshed = run_step_with_worker("review", state, plan_dir, args, root=root)
    except CliError as error:
        record_step_failure(plan_dir, state, step="review", iteration=state["iteration"], error=error)
        raise
    atomic_write_json(plan_dir / "review.json", worker.payload)
    final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    atomic_write_text(plan_dir / "final.md", final_plan)
    passed = sum(1 for criterion in worker.payload.get("criteria", []) if criterion.get("pass"))
    total = len(worker.payload.get("criteria", []))
    state["current_state"] = STATE_DONE
    persist_session(state, "review", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        make_history_entry(
            "review",
            duration_ms=worker.duration_ms,
            cost_usd=worker.cost_usd,
            result="success",
            worker=worker,
            agent=agent,
            mode=mode,
            output_file="review.json",
            artifact_hash=sha256_file(plan_dir / "review.json"),
        ),
    )
    save_state(plan_dir, state)
    response = {
        "success": True,
        "step": "review",
        "summary": f"Review complete: {passed}/{total} success criteria passed.",
        "artifacts": ["review.json", "final.md"],
        "next_step": None,
        "state": STATE_DONE,
        "issues": worker.payload.get("issues", []),
    }
    attach_agent_fallback(response, args)
    return response


def handle_status(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    next_steps = infer_next_steps(state)
    return {
        "success": True,
        "step": "status",
        "plan": state["name"],
        "state": state["current_state"],
        "iteration": state["iteration"],
        "summary": f"Plan '{state['name']}' is currently in state '{state['current_state']}'.",
        "next_step": next_steps[0] if next_steps else None,
        "valid_next": next_steps,
        "artifacts": sorted(path.name for path in plan_dir.iterdir() if path.is_file()),
    }


def handle_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    return {
        "success": True,
        "step": "audit",
        "plan": state["name"],
        "plan_dir": str(plan_dir),
        "state": state,
    }


def handle_list(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_layout(root)
    items = []
    for plan_dir in active_plan_dirs(root):
        state = read_json(plan_dir / "state.json")
        next_steps = infer_next_steps(state)
        items.append(
            {
                "name": state["name"],
                "idea": state["idea"],
                "state": state["current_state"],
                "iteration": state["iteration"],
                "next_step": next_steps[0] if next_steps else None,
            }
        )
    return {
        "success": True,
        "step": "list",
        "summary": f"Found {len(items)} plans.",
        "plans": items,
    }


def handle_override(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    action = args.override_action
    note = args.note
    if action == "add-note":
        override_entry = {"action": action, "timestamp": now_utc(), "note": note}
        entry = {"timestamp": now_utc(), "note": note}
        state.setdefault("meta", {}).setdefault("notes", []).append(entry)
        state.setdefault("meta", {}).setdefault("overrides", []).append(override_entry)
        save_state(plan_dir, state)
        return {
            "success": True,
            "step": "override",
            "summary": "Attached note to the plan.",
            "next_step": infer_next_steps(state)[0] if infer_next_steps(state) else None,
            "state": state["current_state"],
        }
    if action == "abort":
        override_entry = {"action": action, "timestamp": now_utc(), "reason": args.reason}
        state["current_state"] = STATE_ABORTED
        state.setdefault("meta", {}).setdefault("overrides", []).append(override_entry)
        save_state(plan_dir, state)
        return {
            "success": True,
            "step": "override",
            "summary": "Plan aborted.",
            "next_step": None,
            "state": STATE_ABORTED,
        }
    if action == "force-proceed":
        if state["current_state"] == STATE_EVALUATED:
            gate = run_gate_checks(plan_dir, state)
            if not gate["preflight_results"]["project_dir_exists"] or not gate["preflight_results"]["success_criteria_present"]:
                raise CliError("unsafe_override", "force-proceed cannot bypass missing project directory or success criteria")
            atomic_write_json(plan_dir / "gate.json", gate)
            final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
            atomic_write_text(plan_dir / "final.md", final_plan)
            state["current_state"] = STATE_GATED
            state.setdefault("meta", {}).pop("user_approved_gate", None)
            state.setdefault("meta", {}).setdefault("overrides", []).append({"action": action, "timestamp": now_utc(), "reason": args.reason})
            save_state(plan_dir, state)
            return {
                "success": True,
                "step": "override",
                "summary": "Force-proceeded past evaluation into gated state.",
                "next_step": "execute",
                "state": STATE_GATED,
            }
        raise CliError(
            "invalid_transition",
            "force-proceed is only supported from evaluated state",
            valid_next=infer_next_steps(state),
        )
    if action == "skip":
        if state["current_state"] == STATE_EVALUATED:
            evaluation = copy.deepcopy(state.get("last_evaluation", {}))
            evaluation["recommendation"] = "SKIP"
            state["last_evaluation"] = evaluation
            state.setdefault("meta", {}).setdefault("overrides", []).append({"action": action, "timestamp": now_utc(), "reason": args.reason})
            save_state(plan_dir, state)
            return {
                "success": True,
                "step": "override",
                "summary": "Marked evaluation as SKIP. Run gate next.",
                "next_step": "gate",
                "state": state["current_state"],
            }
        raise CliError("invalid_transition", "skip is currently only supported from evaluated state", valid_next=infer_next_steps(state))
    raise CliError("invalid_override", f"Unknown override action: {action}")


def bundled_agents_md() -> str:
    """Return the contents of the bundled AGENTS.md file."""
    return resources.files("megaplan").joinpath("data", "AGENTS.md").read_text(encoding="utf-8")


def bundled_global_file(name: str) -> str:
    """Return the contents of a bundled global config file."""
    return resources.files("megaplan").joinpath("data", "global", name).read_text(encoding="utf-8")


GLOBAL_TARGETS = [
    {"agent": "claude", "detect": ".claude", "path": ".claude/skills/megaplan/SKILL.md", "data": "skill.md"},
    {"agent": "codex",  "detect": ".codex",  "path": ".codex/skills/megaplan/SKILL.md",  "data": "skill.md"},
    {"agent": "cursor", "detect": ".cursor", "path": ".cursor/rules/megaplan.mdc",       "data": "cursor_rule.mdc"},
]


def _install_owned_file(
    path: Path, content: str, *, force: bool = False
) -> dict[str, Any]:
    """Write a file we own, skipping if content already matches."""
    existed = path.exists()
    if existed and not force:
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return {"path": str(path), "skipped": True, "existed": True}
    atomic_write_text(path, content)
    return {"path": str(path), "skipped": False, "existed": existed}


def handle_setup_global(force: bool = False, home: Path | None = None) -> dict[str, Any]:
    """Install megaplan config into all detected agent global dirs."""
    if home is None:
        home = Path.home()

    installed: list[dict[str, Any]] = []
    detected_count = 0

    for target in GLOBAL_TARGETS:
        agent_dir = home / target["detect"]
        if not agent_dir.is_dir():
            installed.append({
                "agent": target["agent"],
                "path": str(home / target["path"]),
                "skipped": True,
                "reason": "not installed",
            })
            continue

        detected_count += 1
        content = bundled_global_file(target["data"])
        dest = home / target["path"]
        result = _install_owned_file(dest, content, force=force)
        result["agent"] = target["agent"]
        installed.append(result)

    if detected_count == 0:
        return {
            "success": False,
            "step": "setup",
            "mode": "global",
            "summary": (
                "No supported agents detected. "
                "Create one of ~/.claude/, ~/.codex/, or ~/.cursor/ and re-run, "
                "or use 'megaplan setup' to install AGENTS.md into a specific project."
            ),
            "installed": installed,
        }

    # Write agent routing config based on what's available
    available = detect_available_agents()
    config_path = None
    routing = None
    if available:
        agents_config: dict[str, str] = {}
        for step, default in DEFAULT_AGENT_ROUTING.items():
            agents_config[step] = default if default in available else available[0]
        config = load_config(home)
        config["agents"] = agents_config
        config_path = save_config(config, home)
        routing = agents_config

    lines = []
    for entry in installed:
        if entry.get("reason") == "not installed":
            lines.append(f"  {entry['agent']}: skipped (not installed)")
        elif entry["skipped"]:
            lines.append(f"  {entry['agent']}: up to date")
        else:
            verb = "overwrote" if entry["existed"] else "created"
            lines.append(f"  {entry['agent']}: {verb} {entry['path']}")

    result: dict[str, Any] = {
        "success": True,
        "step": "setup",
        "mode": "global",
        "summary": "Global setup complete:\n" + "\n".join(lines),
        "installed": installed,
    }
    if config_path is not None:
        result["config_path"] = str(config_path)
        result["routing"] = routing
    return result


def handle_setup(args: argparse.Namespace) -> dict[str, Any]:
    local = args.local or args.target_dir
    if not local:
        return handle_setup_global(force=args.force)
    target_dir = Path(args.target_dir).resolve() if args.target_dir else Path.cwd()
    target = target_dir / "AGENTS.md"
    content = bundled_agents_md()
    if target.exists() and not args.force:
        existing = target.read_text(encoding="utf-8")
        if "megaplan" in existing.lower():
            return {
                "success": True,
                "step": "setup",
                "summary": f"AGENTS.md already contains megaplan instructions at {target}",
                "skipped": True,
            }
        # Append to existing AGENTS.md (atomic: read-concat-write)
        combined = existing + "\n\n" + content
        atomic_write_text(target, combined)
        return {
            "success": True,
            "step": "setup",
            "summary": f"Appended megaplan instructions to existing {target}",
            "file": str(target),
        }
    atomic_write_text(target, content)
    return {
        "success": True,
        "step": "setup",
        "summary": f"Created {target}",
        "file": str(target),
    }


def handle_config(args: argparse.Namespace) -> dict[str, Any]:
    """Handle the config subcommand: show, set, reset."""
    action = args.config_action

    if action == "show":
        config = load_config()
        effective: dict[str, str] = {}
        file_agents = config.get("agents", {})
        for step, default in DEFAULT_AGENT_ROUTING.items():
            effective[step] = file_agents.get(step, default)
        return {
            "success": True,
            "step": "config",
            "action": "show",
            "config_path": str(config_dir() / "config.json"),
            "routing": effective,
            "raw_config": config,
        }

    if action == "set":
        key = args.key
        value = args.value
        parts = key.split(".", 1)
        if len(parts) != 2 or parts[0] != "agents":
            raise CliError("invalid_args", f"Key must be 'agents.<step>', got '{key}'")
        step = parts[1]
        if step not in DEFAULT_AGENT_ROUTING:
            raise CliError("invalid_args", f"Unknown step '{step}'. Valid steps: {', '.join(DEFAULT_AGENT_ROUTING)}")
        if value not in KNOWN_AGENTS:
            raise CliError("invalid_args", f"Unknown agent '{value}'. Valid agents: {', '.join(KNOWN_AGENTS)}")
        config = load_config()
        config.setdefault("agents", {})[step] = value
        save_config(config)
        return {
            "success": True,
            "step": "config",
            "action": "set",
            "key": key,
            "value": value,
        }

    if action == "reset":
        path = config_dir() / "config.json"
        if path.exists():
            path.unlink()
        return {
            "success": True,
            "step": "config",
            "action": "reset",
            "summary": "Config file removed. Using defaults.",
        }

    raise CliError("invalid_args", f"Unknown config action: {action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Megaplan orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Install megaplan into agent configs (global by default)")
    setup_parser.add_argument("--local", action="store_true", help="Install AGENTS.md into a project instead of global agent configs")
    setup_parser.add_argument("--target-dir", help="Directory to install into (default: cwd, implies --local)")
    setup_parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--project-dir", required=True)
    init_parser.add_argument("--name")
    init_parser.add_argument("--max-iterations", type=int, default=3)
    init_parser.add_argument("--budget-usd", type=float, default=25.0)
    init_parser.add_argument("--auto-approve", action="store_true")
    init_parser.add_argument("--robustness", choices=list(ROBUSTNESS_LEVELS), default="standard")
    init_parser.add_argument("idea")

    subparsers.add_parser("list")

    for name in ["status", "audit", "evaluate", "gate"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")

    for name in ["clarify", "plan", "critique", "integrate", "execute", "review"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")
        step_parser.add_argument("--agent", choices=["claude", "codex"])
        step_parser.add_argument("--fresh", action="store_true")
        step_parser.add_argument("--persist", action="store_true")
        step_parser.add_argument("--ephemeral", action="store_true")
        if name == "execute":
            step_parser.add_argument("--confirm-destructive", action="store_true")
            step_parser.add_argument("--user-approved", action="store_true")
        if name == "review":
            step_parser.add_argument("--confirm-self-review", action="store_true")

    config_parser = subparsers.add_parser("config", help="View or edit megaplan configuration")
    config_sub = config_parser.add_subparsers(dest="config_action", required=True)
    config_sub.add_parser("show")
    set_parser = config_sub.add_parser("set")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    config_sub.add_parser("reset")

    override_parser = subparsers.add_parser("override")
    override_parser.add_argument("override_action", choices=["skip", "abort", "force-proceed", "add-note"])
    override_parser.add_argument("--plan")
    override_parser.add_argument("--reason", default="")
    override_parser.add_argument("note", nargs="?")

    return parser


def cli_entry() -> None:
    """Entry point for the `megaplan` console script."""
    sys.exit(main())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            response = handle_setup(args)
            return render_response(response)
        if args.command == "config":
            response = handle_config(args)
            return render_response(response)
    except CliError as error:
        return error_response(error)
    root = Path.cwd()
    ensure_runtime_layout(root)
    try:
        if args.command == "init":
            response = handle_init(root, args)
        elif args.command == "clarify":
            response = handle_clarify(root, args)
        elif args.command == "plan":
            response = handle_plan(root, args)
        elif args.command == "critique":
            response = handle_critique(root, args)
        elif args.command == "evaluate":
            response = handle_evaluate(root, args)
        elif args.command == "integrate":
            response = handle_integrate(root, args)
        elif args.command == "gate":
            response = handle_gate(root, args)
        elif args.command == "execute":
            response = handle_execute(root, args)
        elif args.command == "review":
            response = handle_review(root, args)
        elif args.command == "status":
            response = handle_status(root, args)
        elif args.command == "audit":
            response = handle_audit(root, args)
        elif args.command == "list":
            response = handle_list(root, args)
        elif args.command == "override":
            if args.override_action == "add-note" and not args.note:
                raise CliError("invalid_args", "override add-note requires a note")
            response = handle_override(root, args)
        else:
            raise CliError("invalid_command", f"Unknown command {args.command!r}")
        return render_response(response)
    except CliError as error:
        return error_response(error)


if __name__ == "__main__":
    sys.exit(main())
