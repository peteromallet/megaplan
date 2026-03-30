from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from megaplan.cli import build_parser
from megaplan.loop.engine import (
    _MONITORED_OUTPUT_BUFFER_LIMIT,
    _run_monitored_command,
    _take_observation,
    init_loop,
    load_loop_state,
    run_execute_phase,
    run_loop,
    run_loop_worker,
    save_loop_state,
    should_continue,
)
from megaplan.loop.git import git_commit, git_current_sha, git_revert, parse_metric
from megaplan.loop.handlers import handle_loop_init, handle_loop_pause, handle_loop_run, handle_loop_status
from megaplan.loop.prompts import build_execute_prompt, build_loop_prompt, build_plan_prompt
from megaplan.workers import WorkerResult


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)


def _loop_args(**overrides: object) -> argparse.Namespace:
    base = {
        "goal": "ship loop",
        "command": "python app.py",
        "project_dir": None,
        "name": "demo-loop",
        "iterations": 2,
        "time_budget_seconds": 5,
        "time_budget": None,
        "agent": None,
        "phase_model": [],
        "hermes": None,
        "ephemeral": True,
        "fresh": False,
        "persist": False,
        "confirm_self_review": False,
        "allowed_changes": ["app.py"],
        "observe_interval": None,
        "observe_break_patterns": None,
        "reason": "",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _python_command(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(textwrap.dedent(script))}"


def test_init_loop_creates_persisted_state(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    state = init_loop("demo", "ship loop", "python app.py", str(project_dir), _loop_args())

    assert state["name"] == "demo"
    assert state["phase"] == "plan"
    assert state["status"] == "running"
    assert state["iteration"] == 0
    assert state["spec"]["goal"] == "ship loop"
    assert state["spec"]["command"] == "python app.py"
    assert state["spec"]["allowed_changes"] == ["app.py"]
    assert (project_dir / ".megaplan" / "loops" / "demo" / "state.json").exists()


def test_loop_init_parser_uses_continue_biased_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["loop-init", "--project-dir", "/tmp/project", "--command", "pytest -x", "make tests pass"])

    assert args.iterations == 3


def test_loop_init_accepts_monitoring_flags_and_status_reports_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    (project_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    _git_init(project_dir)
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    monkeypatch.setenv("MEGAPLAN_MOCK_WORKERS", "1")
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))

    parser = build_parser()
    args = parser.parse_args(
        [
            "loop-init",
            "--project-dir",
            str(project_dir),
            "--command",
            "python app.py",
            "--name",
            "demo-loop",
            "--goal",
            "train model",
            "--observe-interval",
            "30",
            "--observe-break-patterns",
            "CUDA OOM,NaN loss",
        ]
    )

    init_response = handle_loop_init(root, args)
    state = load_loop_state(project_dir, "demo-loop")

    assert state["spec"]["observe_interval"] == 30
    assert state["spec"]["observe_break_patterns"] == ["CUDA OOM", "NaN loss"]
    assert init_response["details"]["monitoring"]["active"] is True
    assert init_response["details"]["monitoring"]["observe_interval"] == 30

    state["results"] = [
        {
            "iteration": 1,
            "phase": "execute",
            "outcome": "continue",
            "returncode": 1,
            "command_output": "partial output",
            "observations": [
                {
                    "elapsed_seconds": 60,
                    "tail_output": "CUDA OOM",
                    "metric": None,
                    "action": "break_pattern",
                }
            ],
            "kill_reason": "break_pattern:CUDA OOM",
        }
    ]
    save_loop_state(project_dir / ".megaplan" / "loops" / "demo-loop", state)

    status_response = handle_loop_status(root, argparse.Namespace(project_dir=str(project_dir), name="demo-loop"))

    assert status_response["details"]["monitoring"]["active"] is True
    assert status_response["details"]["monitoring"]["last_action"] == "break_pattern"
    assert status_response["details"]["monitoring"]["kill_reason"] == "break_pattern:CUDA OOM"


def test_should_continue_stops_for_budget_goal_and_pause() -> None:
    running = {
        "status": "running",
        "iteration": 0,
        "max_iterations": 2,
        "goal_met": False,
        "agent_requested_pause": False,
    }
    budget_exhausted = dict(running, iteration=2)
    goal_met = dict(running, goal_met=True)
    paused = dict(running, status="paused")

    assert should_continue(running) is True
    assert should_continue(budget_exhausted) is False
    assert should_continue(goal_met) is False
    assert should_continue(paused) is False


def test_git_commit_and_revert_round_trip(tmp_path: Path) -> None:
    _git_init(tmp_path)
    target = tmp_path / "app.py"
    target.write_text("print('v1')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)
    before = git_current_sha(tmp_path)

    target.write_text("print('v2')\n", encoding="utf-8")
    commit_sha = git_commit(tmp_path, "loop fix", ["app.py"])

    assert commit_sha is not None
    assert git_current_sha(tmp_path) != before
    assert target.read_text(encoding="utf-8") == "print('v2')\n"

    git_revert(tmp_path, commit_sha)

    assert target.read_text(encoding="utf-8") == "print('v1')\n"


@pytest.mark.parametrize(
    ("output", "pattern", "expected"),
    [
        ("score=1.25", r"score=(\d+\.\d+)", 1.25),
        ("val_bpb: 0.987", r"val_bpb:\s*(\d+\.\d+)", 0.987),
        ("metric 2e-3", r"metric\s+([0-9eE\.-]+)", 0.002),
    ],
)
def test_parse_metric_extracts_float(output: str, pattern: str, expected: float) -> None:
    assert parse_metric(output, pattern) == pytest.approx(expected)


def test_loop_prompts_include_expected_sections() -> None:
    state = {
        "name": "demo",
        "phase": "plan",
        "status": "running",
        "iteration": 1,
        "config": {"project_dir": "."},
        "sessions": {},
        "spec": {
            "goal": "ship loop",
            "allowed_changes": ["app.py"],
            "known_issues": ["tests failing"],
            "tried_and_failed": ["blind retry"],
        },
        "results": [{"iteration": 1, "phase": "execute", "outcome": "failed", "command_output": "traceback"}],
    }

    plan_prompt = build_plan_prompt(state)
    execute_prompt = build_execute_prompt(state, "A" * 4500)

    assert "spec_updates" in plan_prompt
    assert "next_action" in plan_prompt
    assert "reasoning" in plan_prompt
    assert "diagnosis" in execute_prompt
    assert "should_pause" in execute_prompt
    assert "[truncated " in execute_prompt


def test_loop_execute_prompt_includes_observations_and_truncation_note() -> None:
    observations = [
        {
            "elapsed_seconds": 30,
            "tail_output": "epoch 1 loss=0.9",
            "metric": 0.9,
            "action": "continue",
        },
        {
            "elapsed_seconds": 60,
            "tail_output": "CUDA OOM detected in training loop",
            "metric": 0.9,
            "action": "break_pattern",
        },
    ]
    state = {
        "name": "demo",
        "phase": "execute",
        "status": "running",
        "iteration": 1,
        "config": {"project_dir": "."},
        "sessions": {},
        "spec": {
            "goal": "ship loop",
            "allowed_changes": ["app.py"],
            "observe_interval": 30,
        },
        "results": [],
        "last_command_output": "partial output",
        "last_command_observations": observations,
        "last_command_kill_reason": "break_pattern:CUDA OOM",
        "last_command_is_truncated": True,
    }

    prompt = build_loop_prompt("loop_execute", state)

    assert "Process observations (sampled every 30s):" in prompt
    assert "elapsed | metric | action | tail_snippet" in prompt
    assert "30s | 0.9 | continue | epoch 1 loss=0.9" in prompt
    assert "60s | 0.9 | break_pattern | CUDA OOM detected in training loop" in prompt
    assert "NOTE: Process was terminated early (break pattern: CUDA OOM) at 60s." in prompt
    assert "Output below is truncated; do not misdiagnose partial output as a different failure." in prompt


def test_run_monitored_command_collects_periodic_observations(tmp_path: Path) -> None:
    command = _python_command(
        """
        import time

        for index in range(5):
            print(f"tick={index}", flush=True)
            time.sleep(1)
        """
    )

    result = _run_monitored_command(command, cwd=tmp_path, timeout=8, observe_interval=2, spec={})

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert 1 <= len(result["observations"]) <= 3
    assert all(obs["action"] == "continue" for obs in result["observations"])
    elapsed = [obs["elapsed_seconds"] for obs in result["observations"]]
    assert elapsed == sorted(elapsed)
    assert elapsed[0] in range(1, 4)
    assert elapsed[-1] in range(3, 6)


def test_run_monitored_command_stops_on_break_pattern(tmp_path: Path) -> None:
    command = _python_command(
        """
        import time

        print("starting", flush=True)
        time.sleep(1.1)
        print("FATAL ERROR", flush=True)
        time.sleep(5)
        print("should-not-print", flush=True)
        """
    )

    result = _run_monitored_command(
        command,
        cwd=tmp_path,
        timeout=8,
        observe_interval=1,
        spec={"observe_break_patterns": ["FATAL ERROR"]},
    )

    assert result["timed_out"] is False
    assert result["observations"]
    assert result["observations"][-1]["action"] == "break_pattern"
    assert "FATAL ERROR" in result["stdout"]
    assert "should-not-print" not in result["stdout"]


def test_run_monitored_command_stops_on_metric_stall(tmp_path: Path) -> None:
    command = _python_command(
        """
        import time

        for _ in range(4):
            print("metric=0.9", flush=True)
            time.sleep(1)
        """
    )

    result = _run_monitored_command(
        command,
        cwd=tmp_path,
        timeout=8,
        observe_interval=1,
        spec={
            "metric_pattern": r"metric=([0-9.]+)",
            "optimization_strategy": "maximize accuracy",
            "observe_metric_stall": 2,
        },
    )

    assert result["timed_out"] is False
    assert result["observations"]
    assert result["observations"][-1]["action"] == "stall"
    assert len(result["observations"]) >= 2


def test_take_observation_only_stalls_on_flat_metrics() -> None:
    max_spec = {
        "metric_pattern": r"metric=([0-9.]+)",
        "optimization_strategy": "maximize accuracy",
        "observe_metric_stall": 2,
    }
    min_spec = {
        "metric_pattern": r"metric=([0-9.]+)",
        "optimization_strategy": "minimize loss",
        "observe_metric_stall": 2,
    }

    flat = _take_observation("metric=0.9\nmetric=0.9\n", 10, max_spec)
    regression = _take_observation("metric=0.9\nmetric=0.85\n", 10, max_spec)
    improvement = _take_observation("metric=0.9\nmetric=0.85\n", 10, min_spec)

    assert flat["action"] == "stall"
    assert regression["action"] == "continue"
    assert improvement["action"] == "continue"


def test_run_monitored_command_caps_stream_buffers(tmp_path: Path) -> None:
    command = _python_command(
        """
        import sys

        for index in range(4):
            sys.stdout.write(f"out{index}:" + ("x" * 60000) + "\\n")
            sys.stdout.flush()
            sys.stderr.write(f"err{index}:" + ("y" * 60000) + "\\n")
            sys.stderr.flush()
        """
    )

    result = _run_monitored_command(command, cwd=tmp_path, timeout=5, observe_interval=1, spec={})

    assert result["returncode"] == 0
    assert len(result["stdout"]) <= _MONITORED_OUTPUT_BUFFER_LIMIT
    assert len(result["stderr"]) <= _MONITORED_OUTPUT_BUFFER_LIMIT
    assert "out0:" not in result["stdout"]
    assert "out3:" in result["stdout"]
    assert "err0:" not in result["stderr"]
    assert "err3:" in result["stderr"]


def test_run_monitored_command_respects_timeout(tmp_path: Path) -> None:
    command = _python_command(
        """
        import time

        print("sleeping", flush=True)
        time.sleep(5)
        """
    )

    result = _run_monitored_command(command, cwd=tmp_path, timeout=1, observe_interval=1, spec={})

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "TIMEOUT: command exceeded 1 seconds." in result["output"]


def test_run_monitored_command_handles_fast_process_without_observations(tmp_path: Path) -> None:
    command = _python_command("""print("done", flush=True)""")

    result = _run_monitored_command(command, cwd=tmp_path, timeout=5, observe_interval=1, spec={})

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["observations"] == []
    assert "done" in result["stdout"]


def test_run_monitored_command_joins_reader_threads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    command = _python_command(
        """
        import time

        print("thread-check", flush=True)
        time.sleep(1.2)
        print("done", flush=True)
        """
    )
    created_threads: list[threading.Thread] = []
    original_thread = threading.Thread

    def tracking_thread(*args, **kwargs):
        thread = original_thread(*args, **kwargs)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr("megaplan.loop.engine.threading.Thread", tracking_thread)

    result = _run_monitored_command(command, cwd=tmp_path, timeout=5, observe_interval=1, spec={})

    assert result["returncode"] == 0
    assert len(created_threads) == 2
    assert all(not thread.is_alive() for thread in created_threads)


def test_run_loop_worker_uses_config_sessions_shim_and_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    state = init_loop("demo", "ship loop", "python app.py", str(project_dir), _loop_args())
    seen: dict[str, object] = {}

    def fake_resolve_agent_mode(step: str, args: argparse.Namespace):
        return ("codex", "ephemeral", True, None)

    def fake_run_step_with_worker(step: str, shim_state: dict[str, object], plan_dir: Path, args: argparse.Namespace, **kwargs):
        seen["step"] = step
        seen["keys"] = sorted(shim_state.keys())
        seen["prompt_override"] = kwargs.get("prompt_override")
        return (
            WorkerResult(
                payload={"spec_updates": {}, "next_action": "continue", "reasoning": "ok"},
                raw_output="{}",
                duration_ms=1,
                cost_usd=0.0,
                session_id="sess-1",
            ),
            "codex",
            "ephemeral",
            True,
        )

    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", fake_resolve_agent_mode)
    monkeypatch.setattr("megaplan.loop.engine.run_step_with_worker", fake_run_step_with_worker)

    worker = run_loop_worker("loop_plan", state, project_dir / ".megaplan" / "loops" / "demo", _loop_args(), project_dir)

    assert worker.session_id == "sess-1"
    assert seen["step"] == "loop_plan"
    assert seen["keys"] == ["config", "sessions"]
    assert isinstance(seen["prompt_override"], str)
    assert "spec_updates" in str(seen["prompt_override"])
    assert "codex_loop_plan" in state["sessions"]


def test_loop_handlers_return_step_responses_and_init_runs_first_plan_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    _git_init(project_dir)
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    monkeypatch.setenv("MEGAPLAN_MOCK_WORKERS", "1")
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    args = _loop_args(project_dir=str(project_dir))

    init_response = handle_loop_init(root, args)
    status_response = handle_loop_status(root, argparse.Namespace(project_dir=str(project_dir), name="demo-loop"))
    pause_response = handle_loop_pause(root, argparse.Namespace(project_dir=str(project_dir), name="demo-loop", reason="manual"))
    run_response = handle_loop_run(root, argparse.Namespace(project_dir=str(project_dir), name="demo-loop", iterations=0))

    assert init_response["step"] == "loop-init"
    assert status_response["step"] == "loop-status"
    assert pause_response["step"] == "loop-pause"
    assert run_response["step"] == "loop-run"
    assert init_response["details"]["phase"] == "execute"
    assert status_response["details"]["phase"] == "execute"
    assert pause_response["state"] == "pause_requested"
    assert pause_response["details"]["pause_requested"] is True
    assert run_response["state"] == "paused"


def test_run_loop_applies_iteration_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _git_init(project_dir)
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    monkeypatch.setenv("MEGAPLAN_MOCK_WORKERS", "1")
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    state = init_loop("demo", "observe failures", "python app.py", str(project_dir), _loop_args(iterations=1))

    final_state = run_loop(state, _loop_args(project_dir=str(project_dir), iterations=2), project_dir)

    assert final_state["iteration"] == 2
    assert final_state["max_iterations"] == 2


def test_pause_request_stops_after_current_iteration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _git_init(project_dir)
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    monkeypatch.setenv("MEGAPLAN_MOCK_WORKERS", "1")
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    state = init_loop("demo", "observe failures", "python app.py", str(project_dir), _loop_args(iterations=3))
    original_git_commit = git_commit
    pause_sent = {"value": False}

    def pause_during_execute(repo: Path, message: str, allowed_changes: list[str]):
        if not pause_sent["value"]:
            pause_sent["value"] = True
            handle_loop_pause(project_dir, argparse.Namespace(project_dir=str(project_dir), name="demo", reason="manual"))
        return original_git_commit(repo, message, allowed_changes)

    monkeypatch.setattr("megaplan.loop.engine.git_commit", pause_during_execute)

    final_state = run_loop(state, _loop_args(project_dir=str(project_dir), iterations=3), project_dir)

    assert final_state["iteration"] == 1
    assert final_state["status"] == "paused"
    assert final_state["pause_requested"] is True


def test_loop_status_reports_recent_history_and_current_best(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _git_init(project_dir)
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    monkeypatch.setenv("MEGAPLAN_MOCK_WORKERS", "1")
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    state = init_loop("demo", "observe failures", "python app.py", str(project_dir), _loop_args(iterations=2))

    final_state = run_loop(state, _loop_args(project_dir=str(project_dir), iterations=2), project_dir)
    status_response = handle_loop_status(project_dir, argparse.Namespace(project_dir=str(project_dir), name="demo"))

    assert len(status_response["details"]["recent_history"]) == 2
    assert status_response["details"]["best_result"]["iteration"] == final_state["spec"]["current_best"]["iteration"]
    assert final_state["spec"]["current_best"]["returncode"] == final_state["current_best"]["returncode"]


def test_mock_worker_loop_runs_two_iterations_and_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _git_init(project_dir)
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    monkeypatch.setenv("MEGAPLAN_MOCK_WORKERS", "1")
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    args = _loop_args(project_dir=str(project_dir), iterations=2)
    state = init_loop("demo", "observe failures", "python app.py", str(project_dir), args)

    final_state = run_loop(state, args, project_dir)

    assert final_state["iteration"] == 2
    assert final_state["status"] == "completed"
    assert len(final_state["results"]) == 2
    assert (project_dir / ".megaplan" / "loops" / "demo" / "loop_plan_v2.json").exists()
    assert (project_dir / ".megaplan" / "loops" / "demo" / "loop_execute_v2.json").exists()


def test_monitored_execute_phase_records_observations_artifacts_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _git_init(project_dir)
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    args = _loop_args(project_dir=str(project_dir), iterations=1, observe_interval=30)
    state = init_loop("demo", "observe failures", "python app.py", str(project_dir), args)
    state["spec"]["observe_interval"] = 30
    state["spec"]["observe_break_patterns"] = ["CUDA OOM"]
    plan_dir = project_dir / ".megaplan" / "loops" / "demo"
    captured: dict[str, str] = {}
    expected_observations = [
        {
            "elapsed_seconds": 30,
            "tail_output": "epoch 1 loss=0.9",
            "metric": 0.9,
            "action": "continue",
        },
        {
            "elapsed_seconds": 60,
            "tail_output": "CUDA OOM detected in training loop",
            "metric": 0.9,
            "action": "break_pattern",
        },
    ]

    def fake_monitored(command: str, *, cwd: Path, timeout: int, observe_interval: int, spec: dict[str, object]):
        assert command == "python app.py"
        assert cwd == project_dir
        assert timeout == 5
        assert observe_interval == 30
        assert spec["observe_break_patterns"] == ["CUDA OOM"]
        return {
            "returncode": -15,
            "stdout": "CUDA OOM\n",
            "stderr": "",
            "timed_out": False,
            "output": "STDOUT:\nCUDA OOM\n",
            "observations": expected_observations,
        }

    def fake_run_step_with_worker(step: str, shim_state: dict[str, object], worker_plan_dir: Path, worker_args: argparse.Namespace, **kwargs):
        assert step == "loop_execute"
        assert worker_plan_dir == plan_dir
        assert worker_args.project_dir == str(project_dir)
        assert worker_args.observe_interval == 30
        assert shim_state["config"]["project_dir"] == str(project_dir)
        captured["prompt"] = str(kwargs.get("prompt_override", ""))
        return (
            WorkerResult(
                payload={
                    "outcome": "continue",
                    "diagnosis": "baseline failed under monitoring",
                    "fix_description": "noop",
                    "files_to_change": [],
                    "confidence": "high",
                    "should_pause": False,
                },
                raw_output="{}",
                duration_ms=1,
                cost_usd=0.0,
                session_id="sess-monitor",
            ),
            "codex",
            "ephemeral",
            True,
        )

    monkeypatch.setattr("megaplan.loop.engine._run_monitored_command", fake_monitored)
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    monkeypatch.setattr("megaplan.loop.engine.run_step_with_worker", fake_run_step_with_worker)
    monkeypatch.setattr("megaplan.loop.engine.git_commit", lambda *args, **kwargs: None)

    final_state = run_execute_phase(state, plan_dir, args, project_dir)
    latest_result = final_state["results"][-1]
    artifact = json.loads((plan_dir / "loop_execute_v1.json").read_text(encoding="utf-8"))

    assert latest_result["observations"] == expected_observations
    assert latest_result["kill_reason"] == "break_pattern:CUDA OOM"
    assert latest_result["is_truncated"] is True
    assert artifact["baseline"]["observations"] == expected_observations
    assert artifact["final"]["observations"] == expected_observations
    assert "Process observations (sampled every 30s):" in captured["prompt"]
    assert "60s | 0.9 | break_pattern | CUDA OOM detected in training loop" in captured["prompt"]
    assert "NOTE: Process was terminated early (break pattern: CUDA OOM) at 60s." in captured["prompt"]
    assert "Output below is truncated; do not misdiagnose partial output as a different failure." in captured["prompt"]


def test_candidate_success_clears_baseline_kill_metadata_in_results_and_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _git_init(project_dir)
    (project_dir / "app.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=project_dir, check=True, capture_output=True, text=True)

    args = _loop_args(project_dir=str(project_dir), iterations=1, observe_interval=30)
    state = init_loop("demo", "observe failures", "python app.py", str(project_dir), args)
    state["spec"]["observe_interval"] = 30
    state["spec"]["observe_break_patterns"] = ["CUDA OOM"]
    plan_dir = project_dir / ".megaplan" / "loops" / "demo"

    baseline = {
        "returncode": -15,
        "stdout": "CUDA OOM\n",
        "stderr": "",
        "timed_out": False,
        "output": "STDOUT:\nCUDA OOM\n",
        "observations": [
            {
                "elapsed_seconds": 30,
                "tail_output": "CUDA OOM",
                "metric": None,
                "action": "break_pattern",
            }
        ],
    }
    candidate = {
        "returncode": 0,
        "stdout": "recovered\n",
        "stderr": "",
        "timed_out": False,
        "output": "STDOUT:\nrecovered\n",
        "observations": [
            {
                "elapsed_seconds": 30,
                "tail_output": "metric=0.8",
                "metric": 0.8,
                "action": "continue",
            }
        ],
    }
    calls = {"count": 0}

    def fake_monitored(command: str, *, cwd: Path, timeout: int, observe_interval: int, spec: dict[str, object]):
        assert command == "python app.py"
        assert cwd == project_dir
        calls["count"] += 1
        return baseline if calls["count"] == 1 else candidate

    def fake_run_step_with_worker(step: str, shim_state: dict[str, object], worker_plan_dir: Path, worker_args: argparse.Namespace, **kwargs):
        return (
            WorkerResult(
                payload={
                    "outcome": "continue",
                    "diagnosis": "candidate fixed it",
                    "fix_description": "noop",
                    "files_to_change": [],
                    "confidence": "high",
                    "should_pause": False,
                },
                raw_output="{}",
                duration_ms=1,
                cost_usd=0.0,
                session_id="sess-candidate",
            ),
            "codex",
            "ephemeral",
            True,
        )

    monkeypatch.setattr("megaplan.loop.engine._run_monitored_command", fake_monitored)
    monkeypatch.setattr("megaplan.loop.engine.resolve_agent_mode", lambda step, args: ("codex", "ephemeral", True, None))
    monkeypatch.setattr("megaplan.loop.engine.run_step_with_worker", fake_run_step_with_worker)
    monkeypatch.setattr("megaplan.loop.engine.git_commit", lambda *args, **kwargs: "commit-sha")

    final_state = run_execute_phase(state, plan_dir, args, project_dir)
    save_loop_state(plan_dir, final_state)
    latest_result = final_state["results"][-1]
    artifact = json.loads((plan_dir / "loop_execute_v1.json").read_text(encoding="utf-8"))
    status_response = handle_loop_status(project_dir, argparse.Namespace(project_dir=str(project_dir), name="demo"))

    assert calls["count"] == 2
    assert latest_result["returncode"] == 0
    assert latest_result["observations"] == candidate["observations"]
    assert "kill_reason" not in latest_result
    assert "is_truncated" not in latest_result
    assert final_state["last_command_kill_reason"] is None
    assert final_state["last_command_is_truncated"] is False
    assert artifact["baseline"]["kill_reason"] == "break_pattern:CUDA OOM"
    assert artifact["baseline"]["is_truncated"] is True
    assert artifact["final"]["kill_reason"] is None
    assert artifact["final"]["is_truncated"] is False
    assert status_response["details"]["monitoring"]["last_action"] == "continue"
    assert status_response["details"]["monitoring"]["kill_reason"] is None
