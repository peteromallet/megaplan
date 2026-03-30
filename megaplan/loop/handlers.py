"""CLI handlers for MegaLoop."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from megaplan._core import ensure_runtime_layout, slugify
from megaplan.loop.engine import init_loop, load_loop_state, run_loop, run_plan_phase, save_loop_state
from megaplan.types import CliError, StepResponse
def _project_dir(root: Path, args: argparse.Namespace) -> Path:
    return Path(getattr(args, "project_dir", None) or root).expanduser().resolve()
def _loop_dir(project_dir: Path, name: str) -> Path:
    return project_dir / ".megaplan" / "loops" / name
def _load_loop(project_dir: Path, name: str) -> tuple[Path, dict[str, Any]]:
    plan_dir = _loop_dir(project_dir, name)
    if not (plan_dir / "state.json").exists():
        raise CliError("missing_loop", f"Loop '{name}' does not exist in {project_dir}")
    return plan_dir, load_loop_state(project_dir, name)
def _status_details(state: dict[str, Any]) -> dict[str, Any]:
    spec = state.get("spec", {})
    results = state.get("results", [])
    last_result = results[-1] if results else {}
    best_result = spec.get("current_best") or state.get("current_best", {})
    observations = last_result.get("observations") if results else state.get("last_command_observations", [])
    last_observation = observations[-1] if observations else {}
    kill_reason = last_result.get("kill_reason") if results else state.get("last_command_kill_reason")
    return {
        "iteration": state.get("iteration", 0),
        "phase": state.get("phase", "plan"),
        "status": state.get("status", "unknown"),
        "best_result": {
            "iteration": best_result.get("iteration"),
            "outcome": best_result.get("outcome"),
            "returncode": best_result.get("returncode"),
            "metric": best_result.get("metric"),
        },
        "last_outcome": last_result.get("outcome"),
        "pause_requested": bool(state.get("pause_requested")),
        "pause_reason": state.get("pause_reason"),
        "monitoring": {
            "active": bool(spec.get("observe_interval")),
            "observe_interval": spec.get("observe_interval"),
            "observe_break_patterns": spec.get("observe_break_patterns", []),
            "last_action": last_observation.get("action"),
            "kill_reason": kill_reason,
        },
        "recent_history": [
            {
                "iteration": item.get("iteration"),
                "outcome": item.get("outcome"),
                "returncode": item.get("returncode"),
                "metric": item.get("metric"),
                "reverted": item.get("reverted", False),
            }
            for item in results[-3:]
        ],
    }
def handle_loop_init(root: Path, args: argparse.Namespace) -> StepResponse:
    ensure_runtime_layout(root)
    project_dir = _project_dir(root, args)
    if not project_dir.exists() or not project_dir.is_dir():
        raise CliError("invalid_project_dir", f"Project directory does not exist: {project_dir}")

    goal = str(getattr(args, "goal_option", None) or getattr(args, "goal", "")).strip()
    command = str(getattr(args, "command", "")).strip()
    if not goal:
        raise CliError("invalid_args", "Loop goal is required")
    if not command:
        raise CliError("invalid_args", "Loop command is required")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = getattr(args, "name", None) or f"{slugify(goal)}-{timestamp}"
    plan_dir = _loop_dir(project_dir, name)
    if plan_dir.exists():
        raise CliError("duplicate_loop", f"Loop directory already exists: {name}")

    state = init_loop(name, goal, command, str(project_dir), args)
    run_plan_phase(state, plan_dir, args, root)
    save_loop_state(plan_dir, state)
    return {
        "success": True,
        "step": "loop-init",
        "plan": name,
        "plan_dir": str(plan_dir),
        "state": str(state.get("status", "running")),
        "iteration": int(state.get("iteration", 0)),
        "summary": f"Initialized loop '{name}' and produced the first planning spec.",
        "artifacts": ["state.json", "loop_plan_v1.json", "loop_plan_v1_raw.txt"],
        "next_step": "loop-run",
        "details": _status_details(state),
    }
def handle_loop_run(root: Path, args: argparse.Namespace) -> StepResponse:
    project_dir = _project_dir(root, args)
    plan_dir, state = _load_loop(project_dir, args.name)
    final_state = run_loop(state, args, root)
    return {
        "success": True,
        "step": "loop-run",
        "plan": args.name,
        "plan_dir": str(plan_dir),
        "state": str(final_state.get("status", "unknown")),
        "iteration": int(final_state.get("iteration", 0)),
        "summary": f"Loop '{args.name}' finished with status {final_state.get('status', 'unknown')}.",
        "artifacts": ["state.json"],
        "next_step": None,
        "details": _status_details(final_state),
    }
def handle_loop_status(root: Path, args: argparse.Namespace) -> StepResponse:
    project_dir = _project_dir(root, args)
    plan_dir, state = _load_loop(project_dir, args.name)
    return {
        "success": True,
        "step": "loop-status",
        "plan": args.name,
        "plan_dir": str(plan_dir),
        "state": str(state.get("status", "unknown")),
        "iteration": int(state.get("iteration", 0)),
        "summary": (
            f"Loop '{args.name}' is {state.get('status', 'unknown')} "
            f"at iteration {state.get('iteration', 0)} in phase {state.get('phase', 'plan')}."
        ),
        "artifacts": ["state.json"],
        "next_step": None,
        "details": _status_details(state),
    }
def handle_loop_pause(root: Path, args: argparse.Namespace) -> StepResponse:
    project_dir = _project_dir(root, args)
    plan_dir, state = _load_loop(project_dir, args.name)
    was_running = state.get("status") == "running"
    state["pause_requested"] = True
    state["pause_reason"] = str(getattr(args, "reason", "") or "Paused by user.")
    if not was_running:
        state["status"] = "paused"
    save_loop_state(plan_dir, state)
    return {
        "success": True,
        "step": "loop-pause",
        "plan": args.name,
        "plan_dir": str(plan_dir),
        "state": "pause_requested" if was_running else "paused",
        "iteration": int(state.get("iteration", 0)),
        "summary": (
            f"Loop '{args.name}' will pause after the current iteration completes."
            if was_running
            else f"Loop '{args.name}' paused."
        ),
        "artifacts": ["state.json"],
        "next_step": None,
        "details": _status_details(state),
    }
