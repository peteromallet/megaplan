"""Plan state management — load, save, history, sessions, failure recording."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import fcntl

from megaplan.types import (
    ActiveStep,
    CliError,
    HistoryEntry,
    PlanState,
    PlanVersionRecord,
    TERMINAL_STATES,
)
from .phase_runtime import DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS, phase_stale_seconds

from .io import (
    atomic_write_json,
    atomic_write_text,
    current_iteration_raw_artifact,
    now_utc,
    plans_root,
    read_json,
)

if TYPE_CHECKING:
    from megaplan.workers import WorkerResult


DEFAULT_ACTIVE_STEP_STALE_SECONDS = DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS


# ---------------------------------------------------------------------------
# Plan resolution
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
    return load_plan_from_dir(plan_dir)


def load_plan_from_dir(plan_dir: Path) -> tuple[Path, PlanState]:
    state = read_json(plan_dir / "state.json")
    migrated = False
    if state.get("current_state") == "clarified":
        state["current_state"] = "initialized"
        migrated = True
    elif state.get("current_state") == "evaluated":
        state["current_state"] = "critiqued"
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


def _parse_utc_timestamp(timestamp: str | None) -> datetime | None:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def active_step_is_stale(
    active_step: ActiveStep | None,
    *,
    configured_timeout_seconds: int = DEFAULT_ACTIVE_STEP_STALE_SECONDS,
) -> bool:
    if not isinstance(active_step, dict):
        return False
    step = active_step.get("step")
    if not isinstance(step, str) or not step:
        return False
    started_at = _parse_utc_timestamp(active_step.get("started_at"))
    if started_at is None:
        return False
    age_seconds = max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))
    return age_seconds >= phase_stale_seconds(
        step,
        configured_timeout_seconds=configured_timeout_seconds,
    )


def plan_lock_path(plan_dir: Path) -> Path:
    return plan_dir / ".plan.lock"


def plan_lock_is_held(plan_dir: Path) -> bool:
    lock_path = plan_lock_path(plan_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return False


def _build_plan_locked_details(plan_dir: Path, *, step: str) -> dict[str, object]:
    state_path = plan_dir / "state.json"
    details: dict[str, object] = {"plan": plan_dir.name, "step": step}
    if not state_path.exists():
        return details
    try:
        state = read_json(state_path)
    except Exception:
        return details
    if not isinstance(state, dict):
        return details
    active_step = state.get("active_step")
    if isinstance(active_step, dict):
        details["active_step"] = dict(active_step)
    return details


@contextmanager
def plan_lock(plan_dir: Path, *, step: str) -> Iterator[None]:
    lock_path = plan_lock_path(plan_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            details = _build_plan_locked_details(plan_dir, step=step)
            active_step = details.get("active_step")
            if isinstance(active_step, dict):
                message = (
                    f"Cannot run '{step}' because plan '{plan_dir.name}' already has an active "
                    f"'{active_step.get('step')}' step via {active_step.get('agent')}."
                )
            else:
                message = f"Cannot run '{step}' because plan '{plan_dir.name}' is locked by another process."
            raise CliError("plan_locked", message, extra=details) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def load_plan_locked(root: Path, requested_name: str | None, *, step: str) -> Iterator[tuple[Path, PlanState]]:
    plan_dir = resolve_plan_dir(root, requested_name)
    with plan_lock(plan_dir, step=step):
        yield load_plan_from_dir(plan_dir)


def save_state(plan_dir: Path, state: PlanState) -> None:
    atomic_write_json(plan_dir / "state.json", state)


def apply_session_update(
    state: PlanState,
    step: str,
    agent: str,
    session_id: str | None,
    *,
    mode: str,
    refreshed: bool,
) -> None:
    from megaplan.workers import update_session_state

    result = update_session_state(
        step,
        agent,
        session_id,
        mode=mode,
        refreshed=refreshed,
        existing_sessions=state["sessions"],
    )
    if result is not None:
        key, entry = result
        state["sessions"][key] = entry


def set_active_step(
    state: PlanState,
    *,
    step: str,
    agent: str,
    mode: str,
    model: str | None = None,
    run_id: str | None = None,
) -> str:
    resolved_run_id = run_id or str(uuid.uuid4())
    active_step: ActiveStep = {
        "step": step,
        "agent": agent,
        "mode": mode,
        "run_id": resolved_run_id,
        "started_at": now_utc(),
    }
    if model:
        active_step["model"] = model
    if mode == "persistent":
        from megaplan.workers import session_key_for

        session = state.get("sessions", {}).get(session_key_for(step, agent, model), {})
        session_id = session.get("id")
        if isinstance(session_id, str) and session_id:
            active_step["session_id"] = session_id
    state["active_step"] = active_step
    return resolved_run_id


def clear_active_step(state: PlanState, *, run_id: str | None = None) -> None:
    active_step = state.get("active_step")
    if run_id is not None and isinstance(active_step, dict) and active_step.get("run_id") != run_id:
        return
    state.pop("active_step", None)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def append_history(state: PlanState, entry: HistoryEntry) -> None:
    state["history"].append(entry)
    state["meta"].setdefault("total_cost_usd", 0.0)
    state["meta"]["total_cost_usd"] = round(
        float(state["meta"]["total_cost_usd"]) + float(entry.get("cost_usd", 0.0)),
        6,
    )


def make_history_entry(
    step: str,
    *,
    duration_ms: int,
    cost_usd: float,
    result: str,
    worker: WorkerResult | None = None,
    agent: str | None = None,
    mode: str | None = None,
    output_file: str | None = None,
    artifact_hash: str | None = None,
    finalize_hash: str | None = None,
    raw_output_file: str | None = None,
    message: str | None = None,
    flags_count: int | None = None,
    flags_addressed: list[str] | None = None,
    recommendation: str | None = None,
    approval_mode: str | None = None,
    environment: dict[str, bool] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> HistoryEntry:
    entry: HistoryEntry = {
        "step": step,
        "timestamp": now_utc(),
        "duration_ms": duration_ms,
        "cost_usd": cost_usd,
        "result": result,
    }
    if total_tokens > 0:
        entry["prompt_tokens"] = prompt_tokens
        entry["completion_tokens"] = completion_tokens
        entry["total_tokens"] = total_tokens
    if worker is not None and agent is not None and mode is not None:
        entry["session_mode"] = mode
        entry["session_id"] = worker.session_id
        entry["agent"] = agent
    if output_file is not None:
        entry["output_file"] = output_file
    if artifact_hash is not None:
        entry["artifact_hash"] = artifact_hash
    if finalize_hash is not None:
        entry["finalize_hash"] = finalize_hash
    if raw_output_file is not None:
        entry["raw_output_file"] = raw_output_file
    if message is not None:
        entry["message"] = message
    if flags_count is not None:
        entry["flags_count"] = flags_count
    if flags_addressed is not None:
        entry["flags_addressed"] = flags_addressed
    if recommendation is not None:
        entry["recommendation"] = recommendation
    if approval_mode is not None:
        entry["approval_mode"] = approval_mode
    if environment is not None:
        entry["environment"] = environment
    return entry


def store_raw_worker_output(plan_dir: Path, step: str, iteration: int, content: str) -> str:
    filename = current_iteration_raw_artifact(plan_dir, step, iteration).name
    atomic_write_text(plan_dir / filename, content)
    return filename


def record_step_failure(
    plan_dir: Path,
    state: PlanState,
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


# ---------------------------------------------------------------------------
# Plan version helpers
# ---------------------------------------------------------------------------

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
