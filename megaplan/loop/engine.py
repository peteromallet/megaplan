"""Core MegaLoop engine."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from megaplan._core import (
    atomic_write_json,
    atomic_write_text,
    current_iteration_artifact,
    current_iteration_raw_artifact,
    ensure_runtime_layout,
    now_utc,
    read_json,
)
from megaplan.loop.git import git_commit, git_revert, parse_metric
from megaplan.loop.prompts import build_loop_prompt
from megaplan.loop.types import IterationResult, LoopSpec, LoopState, Observation
from megaplan.workers import WorkerResult, resolve_agent_mode, run_step_with_worker, update_session_state
_DEFAULT_ALLOWED_CHANGES = ["."]
_COMMAND_OUTPUT_LIMIT = 12000
_DEFAULT_TIME_BUDGET_SECONDS = 300
_MONITORED_OUTPUT_BUFFER_LIMIT = 200_000
def _loop_dir(project_dir: str | Path, name: str) -> Path:
    return Path(project_dir) / ".megaplan" / "loops" / name
def _state_path(project_dir: str | Path, name: str) -> Path:
    return _loop_dir(project_dir, name) / "state.json"
def _normalized_args(args: argparse.Namespace | None) -> argparse.Namespace:
    values = vars(args).copy() if args is not None else {}
    normalized = argparse.Namespace(**values)
    defaults = {
        "agent": None,
        "phase_model": [],
        "hermes": None,
        "ephemeral": False,
        "fresh": False,
        "persist": False,
        "confirm_self_review": False,
        "iterations": None,
        "time_budget": None,
        "time_budget_seconds": None,
    }
    for key, value in defaults.items():
        if not hasattr(normalized, key):
            setattr(normalized, key, value)
    return normalized
def _string_list(value: Any, *, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default or [])
def _merge_value(existing: Any, update: Any) -> Any:
    if isinstance(existing, list) and isinstance(update, list):
        merged: list[Any] = []
        for item in [*existing, *update]:
            if item not in merged:
                merged.append(item)
        return merged
    return update
def _merge_spec(spec: LoopSpec, updates: dict[str, Any]) -> LoopSpec:
    merged: LoopSpec = dict(spec)
    for key, value in updates.items():
        if value in (None, "", []):
            continue
        merged[key] = _merge_value(merged.get(key), value)
    return merged
def _time_budget_seconds(state: LoopState, args: argparse.Namespace | None) -> int:
    normalized = _normalized_args(args)
    budget = (
        getattr(normalized, "time_budget_seconds", None)
        or getattr(normalized, "time_budget", None)
        or state.get("time_budget_seconds")
        or _DEFAULT_TIME_BUDGET_SECONDS
    )
    return max(int(budget), 1)
def _truncate_output(text: str, *, limit: int = _COMMAND_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[truncated {len(text) - limit} characters]"
def _command_output(stdout: str, stderr: str) -> str:
    parts: list[str] = []
    if stdout:
        parts.append("STDOUT:\n" + stdout)
    if stderr:
        parts.append("STDERR:\n" + stderr)
    return _truncate_output("\n\n".join(parts).strip())
def _iterations_from_args(args: argparse.Namespace | None) -> int:
    normalized = _normalized_args(args)
    configured = getattr(normalized, "iterations", None) or 3
    return max(int(configured), 0)
def _run_user_command(command: str, *, cwd: Path, timeout: int) -> dict[str, Any]:
    shell = shutil.which("zsh") or "/bin/sh"
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            executable=shell,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
            "output": _command_output(result.stdout, result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        timeout_note = f"\n\nTIMEOUT: command exceeded {timeout} seconds."
        return {
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            "output": _truncate_output((_command_output(stdout, stderr) + timeout_note).strip()),
        }


def _append_bounded_chunk(
    buffer: deque[str],
    *,
    buffer_size: list[int],
    chunk: str,
    buffer_limit: int,
) -> None:
    buffer.append(chunk)
    buffer_size[0] += len(chunk)
    while buffer_size[0] > buffer_limit and buffer:
        removed = buffer.popleft()
        buffer_size[0] -= len(removed)


def _reader_thread(
    stream: Any,
    *,
    stream_buffer: deque[str],
    stream_buffer_size: list[int],
    output_buffer: deque[str],
    output_buffer_size: list[int],
    buffer_limit: int,
    lock: threading.Lock,
) -> None:
    try:
        for chunk in iter(stream.readline, ""):
            if not chunk:
                break
            with lock:
                _append_bounded_chunk(
                    stream_buffer,
                    buffer_size=stream_buffer_size,
                    chunk=chunk,
                    buffer_limit=buffer_limit,
                )
                _append_bounded_chunk(
                    output_buffer,
                    buffer_size=output_buffer_size,
                    chunk=chunk,
                    buffer_limit=buffer_limit,
                )
    finally:
        stream.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _metric_values_for_output(spec: LoopSpec, output: str) -> list[float]:
    pattern = spec.get("metric_pattern")
    if not pattern:
        return []

    metrics: list[float] = []
    for match in re.finditer(pattern, output, re.MULTILINE):
        captured = next((group for group in match.groups() if group is not None), match.group(0))
        try:
            metrics.append(float(captured))
            continue
        except (TypeError, ValueError):
            numeric = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(captured))
            if numeric is None:
                continue
            try:
                metrics.append(float(numeric.group(0)))
            except ValueError:
                continue
    return metrics


def _take_observation(output: str, elapsed_seconds: int, spec: LoopSpec) -> Observation:
    tail_output = output[-4000:]
    metric = _metric_for_output(spec, output)
    action = "continue"

    for pattern in spec.get("observe_break_patterns", []):
        if re.search(pattern, tail_output, re.MULTILINE):
            action = "break_pattern"
            break

    stall_window = int(spec.get("observe_metric_stall", 0) or 0)
    if action == "continue" and stall_window > 1:
        metrics = _metric_values_for_output(spec, output)
        recent_metrics = metrics[-stall_window:]
        if len(recent_metrics) == stall_window and all(
            abs(value - recent_metrics[0]) <= 1e-9 for value in recent_metrics[1:]
        ):
            action = "stall"

    return {
        "elapsed_seconds": elapsed_seconds,
        "tail_output": tail_output,
        "metric": metric,
        "action": action,
    }


def _run_monitored_command(
    command: str,
    *,
    cwd: Path,
    timeout: int,
    observe_interval: int,
    spec: LoopSpec,
) -> dict[str, Any]:
    shell = shutil.which("zsh") or "/bin/sh"
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        shell=True,
        executable=shell,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to capture process output for monitored command")

    stdout_buffer: deque[str] = deque()
    stderr_buffer: deque[str] = deque()
    output_buffer: deque[str] = deque()
    stdout_buffer_size = [0]
    stderr_buffer_size = [0]
    output_buffer_size = [0]
    observations: list[Observation] = []
    lock = threading.Lock()
    reader_threads = [
        threading.Thread(
            target=_reader_thread,
            kwargs={
                "stream": process.stdout,
                "stream_buffer": stdout_buffer,
                "stream_buffer_size": stdout_buffer_size,
                "output_buffer": output_buffer,
                "output_buffer_size": output_buffer_size,
                "buffer_limit": _MONITORED_OUTPUT_BUFFER_LIMIT,
                "lock": lock,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_reader_thread,
            kwargs={
                "stream": process.stderr,
                "stream_buffer": stderr_buffer,
                "stream_buffer_size": stderr_buffer_size,
                "output_buffer": output_buffer,
                "output_buffer_size": output_buffer_size,
                "buffer_limit": _MONITORED_OUTPUT_BUFFER_LIMIT,
                "lock": lock,
            },
            daemon=True,
        ),
    ]
    for thread in reader_threads:
        thread.start()

    observation_fn: Callable[[str, int, LoopSpec], Observation] = _take_observation
    started_at = time.monotonic()
    deadline = started_at + timeout
    timed_out = False
    try:
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_process(process)
                break

            time.sleep(min(max(observe_interval, 1), remaining))
            if process.poll() is not None:
                break

            elapsed_seconds = max(int(time.monotonic() - started_at), 0)
            with lock:
                snapshot_output = "".join(output_buffer)
            observation = observation_fn(snapshot_output, elapsed_seconds, spec)
            observations.append(observation)
            if observation.get("action") != "continue" and process.poll() is None:
                _terminate_process(process)
                break
    finally:
        if process.poll() is None:
            _terminate_process(process)
        for thread in reader_threads:
            thread.join()

    stdout = "".join(stdout_buffer)
    stderr = "".join(stderr_buffer)
    output = _command_output(stdout, stderr)
    if timed_out:
        output = _truncate_output((output + f"\n\nTIMEOUT: command exceeded {timeout} seconds.").strip())
        returncode = 124
    else:
        returncode = process.returncode
    return {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "output": output,
        "observations": observations,
    }


def _metric_direction(spec: LoopSpec) -> str:
    text = " ".join(
        [
            str(spec.get("optimization_strategy", "")),
            " ".join(spec.get("success_criteria", [])),
        ]
    ).lower()
    if any(token in text for token in ("min", "lower", "reduce", "smaller", "decrease", "less")):
        return "min"
    return "max"
def _metric_for_output(spec: LoopSpec, output: str) -> float | None:
    pattern = spec.get("metric_pattern")
    if not pattern:
        return None
    return parse_metric(output, pattern)


def _is_worse(candidate: dict[str, Any], baseline: dict[str, Any], spec: LoopSpec) -> bool:
    if baseline["returncode"] == 0 and candidate["returncode"] != 0:
        return True
    baseline_metric = baseline.get("metric")
    candidate_metric = candidate.get("metric")
    if baseline_metric is not None and candidate_metric is not None:
        if _metric_direction(spec) == "min":
            return candidate_metric > baseline_metric
        return candidate_metric < baseline_metric
    return False


def _is_better(candidate: IterationResult, current_best: IterationResult | None, spec: LoopSpec) -> bool:
    if current_best is None:
        return True
    best_returncode = current_best.get("returncode", 1)
    candidate_returncode = candidate.get("returncode", 1)
    if candidate_returncode == 0 and best_returncode != 0:
        return True
    best_metric = current_best.get("metric")
    candidate_metric = candidate.get("metric")
    if best_metric is not None and candidate_metric is not None:
        if _metric_direction(spec) == "min":
            return candidate_metric < best_metric
        return candidate_metric > best_metric
    return False


def _observation_kill_reason(observations: list[Observation], spec: LoopSpec) -> str | None:
    if not observations:
        return None

    final_observation = observations[-1]
    action = final_observation.get("action", "continue")
    if action == "continue":
        return None
    if action == "break_pattern":
        tail_output = final_observation.get("tail_output", "")
        for pattern in spec.get("observe_break_patterns", []):
            if re.search(pattern, tail_output, re.MULTILINE):
                return f"break_pattern:{pattern}"
    return str(action)


def save_loop_state(plan_dir: Path, state: LoopState) -> None:
    state_path = plan_dir / "state.json"
    if state_path.exists():
        persisted = read_json(state_path)
        if persisted.get("pause_requested"):
            state["pause_requested"] = True
            state["pause_reason"] = str(state.get("pause_reason") or persisted.get("pause_reason") or "Pause requested by user.")
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)


def load_loop_state(project_dir: str | Path, name: str) -> LoopState:
    return read_json(_state_path(project_dir, name))


def init_loop(name: str, goal: str, command: str, project_dir: str, args: argparse.Namespace | None) -> LoopState:
    normalized = _normalized_args(args)
    allowed_changes = _string_list(getattr(normalized, "allowed_changes", None), default=_DEFAULT_ALLOWED_CHANGES)
    spec: LoopSpec = {
        "goal": goal,
        "command": command,
        "success_criteria": _string_list(getattr(normalized, "success_criteria", None), default=["Command exits with status 0."]),
        "allowed_changes": allowed_changes,
        "optimization_strategy": str(getattr(normalized, "optimization_strategy", "") or "Prefer the smallest fix that improves the command result."),
        "bug_finding_approach": str(getattr(normalized, "bug_finding_approach", "") or "Use the latest command output as the primary source of truth."),
        "philosophy": str(getattr(normalized, "philosophy", "") or "Keep iterating until the goal is met or there is a real reason to pause."),
        "known_issues": [],
        "tried_and_failed": [],
    }
    metric_pattern = getattr(normalized, "metric_pattern", None)
    if metric_pattern:
        spec["metric_pattern"] = str(metric_pattern)
    observe_interval = getattr(normalized, "observe_interval", None)
    if observe_interval is not None:
        spec["observe_interval"] = max(int(observe_interval), 1)
    observe_break_patterns = _string_list(getattr(normalized, "observe_break_patterns", None))
    if observe_break_patterns:
        spec["observe_break_patterns"] = observe_break_patterns
    config = {"project_dir": str(Path(project_dir).resolve()), "robustness": getattr(normalized, "robustness", "robust")}
    if getattr(normalized, "agent", None):
        config["agents"] = {
            "loop_plan": normalized.agent,
            "loop_execute": normalized.agent,
        }
    now = now_utc()
    state: LoopState = {
        "name": name,
        "spec": spec,
        "phase": "plan",
        "status": "running",
        "iteration": 0,
        "config": config,
        "sessions": {},
        "created_at": now,
        "updated_at": now,
        "max_iterations": _iterations_from_args(normalized),
        "results": [],
        "pause_requested": False,
    }
    plan_dir = _loop_dir(config["project_dir"], name)
    save_loop_state(plan_dir, state)
    return state


def _apply_requested_pause(state: LoopState) -> bool:
    if state.get("pause_requested") and state.get("status") == "running":
        state["status"] = "paused"
        state["phase"] = "plan"
        state["pause_reason"] = str(state.get("pause_reason") or "Pause requested by user.")
        return True
    return False


def run_loop_worker(
    step: str,
    state: LoopState,
    plan_dir: Path,
    args: argparse.Namespace | None,
    root: Path,
    *,
    command_output: str | None = None,
    observations: list[Observation] | None = None,
    observe_interval: int | None = None,
    kill_reason: str | None = None,
    is_truncated: bool | None = None,
) -> WorkerResult:
    ensure_runtime_layout(root)
    normalized = _normalized_args(args)
    prompt = build_loop_prompt(
        step,
        state,
        command_output=command_output,
        observations=observations,
        observe_interval=observe_interval,
        kill_reason=kill_reason,
        is_truncated=is_truncated,
    )
    shim_state = {
        "config": state["config"],
        "sessions": state["sessions"],
    }
    resolved = resolve_agent_mode(step, normalized)
    worker, agent, mode, refreshed = run_step_with_worker(
        step,
        shim_state,  # type: ignore[arg-type]
        plan_dir,
        normalized,
        root=root,
        resolved=resolved,
        prompt_override=prompt,
    )
    session_update = update_session_state(
        step,
        agent,
        worker.session_id,
        mode=mode,
        refreshed=refreshed,
        model=resolved[3],
        existing_sessions=state["sessions"],
    )
    if session_update is not None:
        key, entry = session_update
        state["sessions"][key] = entry
    return worker


def run_plan_phase(state: LoopState, plan_dir: Path, args: argparse.Namespace | None, root: Path) -> LoopState:
    state["phase"] = "plan"
    worker = run_loop_worker("loop_plan", state, plan_dir, args, root)
    payload = worker.payload
    spec_updates = payload.get("spec_updates", {})
    if isinstance(spec_updates, dict):
        state["spec"] = _merge_spec(state.get("spec", {}), spec_updates)
    state["last_plan"] = payload
    iteration = state["iteration"] + 1
    atomic_write_json(current_iteration_artifact(plan_dir, "loop_plan", iteration), payload)
    atomic_write_text(current_iteration_raw_artifact(plan_dir, "loop_plan", iteration), worker.raw_output)
    state["phase"] = "execute"
    state["updated_at"] = now_utc()
    return state


def run_execute_phase(state: LoopState, plan_dir: Path, args: argparse.Namespace | None, root: Path) -> LoopState:
    state["phase"] = "execute"
    project_dir = Path(state["config"]["project_dir"])
    spec = state.get("spec", {})
    command = str(spec.get("command", "")).strip()
    if not command:
        raise ValueError("Loop spec is missing a command")

    observe_interval = int(spec.get("observe_interval", 0) or 0)
    timeout = _time_budget_seconds(state, args)

    if observe_interval > 0:
        baseline = _run_monitored_command(
            command,
            cwd=project_dir,
            timeout=timeout,
            observe_interval=observe_interval,
            spec=spec,
        )
    else:
        baseline = _run_user_command(command, cwd=project_dir, timeout=timeout)
    baseline["metric"] = _metric_for_output(spec, baseline["output"])
    baseline_observations = list(baseline.get("observations", []))
    baseline_kill_reason = _observation_kill_reason(baseline_observations, spec)
    state["last_command_output"] = baseline["output"]
    state["last_command_observations"] = baseline_observations
    state["last_command_kill_reason"] = baseline_kill_reason
    state["last_command_is_truncated"] = baseline_kill_reason is not None

    worker = run_loop_worker(
        "loop_execute",
        state,
        plan_dir,
        args,
        root,
        command_output=baseline["output"],
        observations=baseline_observations,
        observe_interval=observe_interval if observe_interval > 0 else None,
        kill_reason=baseline_kill_reason,
        is_truncated=baseline_kill_reason is not None,
    )
    payload = worker.payload
    allowed_changes = _string_list(spec.get("allowed_changes"), default=_DEFAULT_ALLOWED_CHANGES)
    commit_sha = git_commit(project_dir, f"megaloop: iteration {state['iteration'] + 1}", allowed_changes)

    final_command = baseline
    reverted = False
    if commit_sha is not None:
        if observe_interval > 0:
            candidate = _run_monitored_command(
                command,
                cwd=project_dir,
                timeout=timeout,
                observe_interval=observe_interval,
                spec=spec,
            )
        else:
            candidate = _run_user_command(command, cwd=project_dir, timeout=timeout)
        candidate["metric"] = _metric_for_output(spec, candidate["output"])
        if _is_worse(candidate, baseline, spec):
            git_revert(project_dir, commit_sha)
            reverted = True
        else:
            final_command = candidate

    final_observations = list(final_command.get("observations", []))
    final_kill_reason = _observation_kill_reason(final_observations, spec)

    iteration = state["iteration"] + 1
    result: IterationResult = {
        "iteration": iteration,
        "phase": "execute",
        "outcome": str(payload.get("outcome", "continue")),
        "diagnosis": str(payload.get("diagnosis", "")),
        "fix_description": str(payload.get("fix_description", "")),
        "files_to_change": _string_list(payload.get("files_to_change")),
        "confidence": str(payload.get("confidence", "")),
        "should_pause": bool(payload.get("should_pause", False)),
        "returncode": int(final_command["returncode"]),
        "command_output": str(final_command["output"]),
        "metric": final_command.get("metric"),
        "commit_sha": commit_sha or "",
        "reverted": reverted,
        "observations": final_observations,
    }
    if "reasoning" in payload:
        result["reasoning"] = str(payload["reasoning"])
    if final_kill_reason is not None:
        result["kill_reason"] = final_kill_reason
        result["is_truncated"] = True

    execution_artifact = {
        "iteration": iteration,
        "baseline": {
            "returncode": baseline["returncode"],
            "timed_out": baseline["timed_out"],
            "metric": baseline.get("metric"),
            "output": baseline["output"],
            "observations": baseline_observations,
            "kill_reason": baseline_kill_reason,
            "is_truncated": baseline_kill_reason is not None,
        },
        "worker": payload,
        "final": {
            "returncode": final_command["returncode"],
            "timed_out": final_command["timed_out"],
            "metric": final_command.get("metric"),
            "output": final_command["output"],
            "observations": final_observations,
            "kill_reason": final_kill_reason,
            "is_truncated": final_kill_reason is not None,
        },
        "commit_sha": commit_sha,
        "reverted": reverted,
    }
    atomic_write_json(current_iteration_artifact(plan_dir, "loop_execute", iteration), execution_artifact)
    atomic_write_text(current_iteration_raw_artifact(plan_dir, "loop_execute", iteration), worker.raw_output)

    state.setdefault("results", []).append(result)
    state["iteration"] = iteration
    state["last_command_output"] = final_command["output"]
    state["last_command_observations"] = final_observations
    state["last_command_kill_reason"] = final_kill_reason
    state["last_command_is_truncated"] = final_kill_reason is not None
    state["goal_met"] = result["returncode"] == 0
    state["agent_requested_pause"] = result["should_pause"]
    if result["should_pause"]:
        state["status"] = "paused"
        state["pause_reason"] = str(payload.get("outcome") or "Loop worker requested pause")
        state["pause_requested"] = True
    elif state["goal_met"]:
        state["status"] = "done"
    else:
        state["status"] = "running"
    if _is_better(result, state.get("current_best"), spec):
        state["current_best"] = result
        best_update: dict[str, Any] = {"current_best": result}
        if result.get("command_output"):
            best_update["best_result_summary"] = result["command_output"][:500]
        state["spec"] = _merge_spec(state.get("spec", {}), best_update)
    state["phase"] = "plan" if state["status"] == "running" else state["phase"]
    state["updated_at"] = now_utc()
    return state


def should_continue(state: LoopState) -> bool:
    if state.get("status") == "paused":
        return False
    if state.get("goal_met"):
        return False
    if state.get("agent_requested_pause"):
        return False
    max_iterations = int(state.get("max_iterations", 0) or 0)
    if max_iterations and int(state.get("iteration", 0)) >= max_iterations:
        return False
    return True


def run_loop(state: LoopState, args: argparse.Namespace | None, root: Path | None = None) -> LoopState:
    project_root = Path(root or state["config"]["project_dir"]).resolve()
    plan_dir = _loop_dir(state["config"]["project_dir"], state["name"])
    override_iterations = getattr(_normalized_args(args), "iterations", None)
    if override_iterations is not None:
        state["max_iterations"] = int(state.get("iteration", 0)) + max(int(override_iterations), 0)
    _apply_requested_pause(state)
    save_loop_state(plan_dir, state)

    while should_continue(state):
        if state.get("phase") != "execute":
            run_plan_phase(state, plan_dir, args, project_root)
            save_loop_state(plan_dir, state)
            if not should_continue(state):
                break
        run_execute_phase(state, plan_dir, args, project_root)
        save_loop_state(plan_dir, state)
        if _apply_requested_pause(state):
            save_loop_state(plan_dir, state)
            break

    if state.get("status") == "running" and state.get("goal_met"):
        state["status"] = "done"
    elif state.get("status") == "running" and int(state.get("iteration", 0)) >= int(state.get("max_iterations", 0) or 0):
        state["status"] = "completed"
    save_loop_state(plan_dir, state)
    return state
