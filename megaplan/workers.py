"""Worker orchestration: running Claude and Codex steps."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megaplan.schemas import SCHEMAS  # noqa: F401


WORKER_TIMEOUT_SECONDS = 3600


@dataclass
class CommandResult:
    command: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass
class WorkerResult:
    payload: dict[str, Any]
    raw_output: str
    duration_ms: int
    cost_usd: float
    session_id: str | None = None
    trace_output: str | None = None


def run_command(
    command: list[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    timeout: int | None = WORKER_TIMEOUT_SECONDS,
) -> CommandResult:
    # Import here to avoid circular dependency
    from megaplan.cli import CliError

    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise CliError(
            "agent_not_found",
            f"Command not found: {command[0]}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CliError(
            "worker_timeout",
            f"Command timed out after {timeout}s: {' '.join(command[:3])}...",
            extra={"raw_output": str(exc.stdout or "") + str(exc.stderr or "")},
        ) from exc
    return CommandResult(
        command=command,
        cwd=cwd,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def extract_session_id(raw: str) -> str | None:
    # Try structured JSONL first (codex --json emits {"type":"thread.started","thread_id":"..."})
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("thread_id"):
                return str(obj["thread_id"])
        except (json.JSONDecodeError, ValueError):
            continue
    # Fallback: unstructured text pattern
    match = re.search(r"\bsession[_ ]id[: ]+([0-9a-fA-F-]{8,})", raw)
    return match.group(1) if match else None


def parse_claude_envelope(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from megaplan.cli import CliError

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError("parse_error", f"Claude output was not valid JSON: {exc}", extra={"raw_output": raw}) from exc
    if isinstance(envelope, dict) and envelope.get("is_error"):
        message = envelope.get("result") or envelope.get("message") or "Claude returned an error"
        raise CliError("worker_error", f"Claude step failed: {message}", extra={"raw_output": raw})
    # When using --json-schema, structured output lives in "structured_output"
    # rather than "result" (which may be empty).
    payload: Any = envelope
    if isinstance(envelope, dict):
        if "structured_output" in envelope and isinstance(envelope["structured_output"], dict):
            payload = envelope["structured_output"]
        elif "result" in envelope:
            payload = envelope["result"]
    if isinstance(payload, str):
        if not payload.strip():
            raise CliError("parse_error", "Claude returned empty result (check structured_output field)", extra={"raw_output": raw})
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CliError("parse_error", f"Claude result payload was not valid JSON: {exc}", extra={"raw_output": raw}) from exc
    if not isinstance(payload, dict):
        raise CliError("parse_error", "Claude result payload was not an object", extra={"raw_output": raw})
    return envelope, payload


def parse_json_file(path: Path) -> dict[str, Any]:
    from megaplan.cli import CliError, read_json

    try:
        payload = read_json(path)
    except FileNotFoundError as exc:
        raise CliError("parse_error", f"Output file {path.name} was not created") from exc
    except json.JSONDecodeError as exc:
        raise CliError("parse_error", f"Output file {path.name} was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliError("parse_error", f"Output file {path.name} did not contain a JSON object")
    return payload


def validate_payload(step: str, payload: dict[str, Any]) -> None:
    from megaplan.cli import CliError

    def require_keys(keys: list[str]) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise CliError("parse_error", f"{step} output missing required keys: {', '.join(missing)}")

    if step == "clarify":
        require_keys(["questions", "refined_idea", "intent_summary"])
    elif step == "plan":
        require_keys(["plan", "questions", "success_criteria", "assumptions"])
    elif step == "integrate":
        require_keys(["plan", "changes_summary", "flags_addressed"])
    elif step == "critique":
        require_keys(["flags"])
    elif step == "execute":
        require_keys(["output", "files_changed", "commands_run", "deviations"])
    elif step == "review":
        require_keys(["criteria", "issues"])


def mock_worker_output(step: str, state: dict[str, Any], plan_dir: Path) -> WorkerResult:
    from megaplan.cli import (
        json_dump, read_json, latest_plan_meta_path,
    )

    iteration = state["iteration"] or 1
    if step == "clarify":
        payload = {
            "questions": [
                {
                    "question": "Should the feature be behind a flag?",
                    "context": "No feature flags exist in the repo currently.",
                },
            ],
            "refined_idea": f"Refined: {state['idea']}",
            "intent_summary": f"The user wants to {state['idea']}.",
        }
        return WorkerResult(payload=payload, raw_output=json_dump(payload), duration_ms=10, cost_usd=0.0, session_id=str(uuid.uuid4()))

    if step == "plan":
        payload = {
            "plan": textwrap.dedent(
                f"""
                # Implementation Plan

                ## Goal
                Implement: {state['idea']}

                ## Steps
                1. Inspect the repository and identify the touch points.
                2. Implement the feature with tests or local verification hooks.
                3. Validate success criteria before finishing.

                ## Risks
                - Repository reality may differ from the initial assumption.
                - Missing verification would block execution.
                """
            ).strip(),
            "questions": ["Are there existing patterns in the repo that should be preserved?"],
            "success_criteria": [
                "A concrete implementation path exists.",
                "Verification is defined before execution.",
            ],
            "assumptions": ["The project directory is writable."],
        }
        return WorkerResult(payload=payload, raw_output=json_dump(payload), duration_ms=10, cost_usd=0.0, session_id=str(uuid.uuid4()))

    if step == "critique":
        if iteration == 1:
            payload = {
                "flags": [
                    {
                        "id": "FLAG-001",
                        "concern": "The plan does not name the files or modules it expects to touch.",
                        "category": "completeness",
                        "severity_hint": "likely-significant",
                        "evidence": "Execution could drift because there is no repo-specific scope.",
                    },
                    {
                        "id": "FLAG-002",
                        "concern": "The plan does not define an observable verification command.",
                        "category": "correctness",
                        "severity_hint": "likely-significant",
                        "evidence": "Success cannot be demonstrated without a concrete check.",
                    },
                ],
                "verified_flag_ids": [],
                "disputed_flag_ids": [],
            }
        else:
            payload = {"flags": [], "verified_flag_ids": ["FLAG-001", "FLAG-002"], "disputed_flag_ids": []}
        return WorkerResult(payload=payload, raw_output=json_dump(payload), duration_ms=10, cost_usd=0.0, session_id=str(uuid.uuid4()))

    if step == "integrate":
        payload = {
            "plan": textwrap.dedent(
                f"""
                # Implementation Plan

                ## Goal
                Implement: {state['idea']}

                ## Concrete Scope
                1. Inspect the repository and identify the exact files to touch before editing.
                2. Implement the change in the smallest viable slice.
                3. Run a concrete verification command and capture the result.

                ## Verification
                - Run a repo-specific smoke test or command before closing the task.

                ## Risks
                - If the repo shape differs from expectations, adapt and record the deviation.
                """
            ).strip(),
            "changes_summary": "Added explicit repo-scoping and verification steps.",
            "flags_addressed": ["FLAG-001", "FLAG-002"],
            "assumptions": ["The repository contains enough context for implementation."],
            "success_criteria": [
                "The plan identifies exact touch points before editing.",
                "A concrete verification command is defined.",
            ],
            "questions": [],
        }
        return WorkerResult(payload=payload, raw_output=json_dump(payload), duration_ms=10, cost_usd=0.0, session_id=str(uuid.uuid4()))

    if step == "execute":
        target = Path(state["config"]["project_dir"]) / "IMPLEMENTED_BY_MEGAPLAN.txt"
        target.write_text("mock execution completed\n", encoding="utf-8")
        payload = {
            "output": "Mock execution completed successfully.",
            "files_changed": [str(target.relative_to(Path(state["config"]["project_dir"])))],
            "commands_run": ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"],
            "deviations": [],
        }
        return WorkerResult(payload=payload, raw_output=json_dump(payload), duration_ms=10, cost_usd=0.0, session_id=str(uuid.uuid4()), trace_output='{"event":"mock-execute"}\n')

    if step == "review":
        meta = read_json(latest_plan_meta_path(plan_dir, state))
        criteria = [
            {"name": criterion, "pass": True, "evidence": "Mock execution and artifacts satisfy the criterion."}
            for criterion in meta.get("success_criteria", [])
        ]
        payload = {"criteria": criteria, "issues": [], "summary": "Mock review passed."}
        return WorkerResult(payload=payload, raw_output=json_dump(payload), duration_ms=10, cost_usd=0.0, session_id=str(uuid.uuid4()))

    from megaplan.cli import CliError
    raise CliError("unsupported_step", f"Mock worker does not support '{step}'")


def session_key_for(step: str, agent: str) -> str:
    if step in {"clarify", "plan", "integrate"}:
        return f"{agent}_planner"
    if step == "critique":
        return f"{agent}_critic"
    if step == "execute":
        return f"{agent}_executor"
    if step == "review":
        return f"{agent}_reviewer"
    return f"{agent}_{step}"


def persist_session(state: dict[str, Any], step: str, agent: str, session_id: str | None, *, mode: str, refreshed: bool) -> None:
    from megaplan.cli import now_utc

    if not session_id:
        return
    key = session_key_for(step, agent)
    state.setdefault("sessions", {})[key] = {
        "id": session_id,
        "mode": mode,
        "created_at": state.setdefault("sessions", {}).get(key, {}).get("created_at", now_utc()),
        "last_used_at": now_utc(),
        "refreshed": refreshed,
    }


def run_claude_step(step: str, state: dict[str, Any], plan_dir: Path, *, root: Path, fresh: bool) -> WorkerResult:
    from megaplan.cli import (
        CliError, MOCK_ENV_VAR, read_json, schemas_root,
    )
    from megaplan.prompts import create_claude_prompt
    import megaplan.cli as _cli

    if os.getenv(MOCK_ENV_VAR) == "1":
        return _cli.mock_worker_output(step, state, plan_dir)
    project_dir = Path(state["config"]["project_dir"])
    schema_name = {
        "clarify": "clarify.json",
        "plan": "plan.json",
        "integrate": "integrate.json",
        "critique": "critique.json",
        "execute": "execution.json",
        "review": "review.json",
    }[step]
    schema_text = json.dumps(read_json(schemas_root(root) / schema_name))
    session_key = session_key_for(step, "claude")
    session = state.setdefault("sessions", {}).get(session_key, {})
    session_id = session.get("id")
    command = ["claude", "-p", "--output-format", "json", "--json-schema", schema_text, "--add-dir", str(project_dir)]
    if session_id and not fresh:
        command.extend(["--resume", session_id])
    else:
        session_id = str(uuid.uuid4())
        command.extend(["--session-id", session_id])
    prompt = create_claude_prompt(step, state, plan_dir)
    result = run_command(command, cwd=project_dir, stdin_text=prompt)
    raw = result.stdout or result.stderr
    envelope, payload = parse_claude_envelope(raw)
    try:
        validate_payload(step, payload)
    except CliError as error:
        raise CliError(error.code, error.message, extra={"raw_output": raw}) from error
    return WorkerResult(
        payload=payload,
        raw_output=raw,
        duration_ms=result.duration_ms,
        cost_usd=float(envelope.get("total_cost_usd", 0.0) or 0.0),
        session_id=str(envelope.get("session_id") or session_id),
    )


def run_codex_step(
    step: str,
    state: dict[str, Any],
    plan_dir: Path,
    *,
    root: Path,
    persistent: bool,
    fresh: bool = False,
    json_trace: bool = False,
) -> WorkerResult:
    from megaplan.cli import (
        CliError, MOCK_ENV_VAR, schemas_root,
    )
    from megaplan.prompts import create_codex_prompt
    import megaplan.cli as _cli

    if os.getenv(MOCK_ENV_VAR) == "1":
        return _cli.mock_worker_output(step, state, plan_dir)
    project_dir = Path(state["config"]["project_dir"])
    schema_file = schemas_root(root) / {
        "clarify": "clarify.json",
        "plan": "plan.json",
        "integrate": "integrate.json",
        "critique": "critique.json",
        "execute": "execution.json",
        "review": "review.json",
    }[step]
    session_key = session_key_for(step, "codex")
    session = state.setdefault("sessions", {}).get(session_key, {})
    out_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    out_handle.close()
    output_path = Path(out_handle.name)
    prompt = create_codex_prompt(step, state, plan_dir)

    if persistent and session.get("id") and not fresh:
        # codex exec resume does not support --output-schema; we rely on
        # validate_payload() after parsing the output file instead.
        command = ["codex", "exec", "resume"]
        if json_trace:
            command.append("--json")
        command.extend([
            "--skip-git-repo-check",
            "-o", str(output_path),
            str(session["id"]), "-",
        ])
    else:
        command = ["codex", "exec", "--skip-git-repo-check", "-C", str(project_dir), "-o", str(output_path)]
        if not persistent:
            command.append("--ephemeral")
        if step == "execute":
            command.append("--full-auto")
        if json_trace:
            command.append("--json")
        command.extend(["--output-schema", str(schema_file), "-"])

    result = run_command(command, cwd=Path.cwd(), stdin_text=prompt)
    raw = result.stdout + result.stderr
    if result.returncode != 0 and (not output_path.exists() or not output_path.read_text(encoding="utf-8").strip()):
        raise CliError("worker_error", f"Codex step failed with exit code {result.returncode}", extra={"raw_output": raw})
    try:
        payload = parse_json_file(output_path)
    except CliError as error:
        raise CliError(error.code, error.message, extra={"raw_output": raw}) from error
    try:
        validate_payload(step, payload)
    except CliError as error:
        raise CliError(error.code, error.message, extra={"raw_output": raw}) from error
    session_id = session.get("id") if persistent else None
    if persistent and not session_id:
        session_id = extract_session_id(raw)
        if not session_id:
            raise CliError(
                "worker_error",
                f"Could not determine Codex session id for persistent {step} step",
                extra={"raw_output": raw},
            )
    trace_output = raw if json_trace else None
    return WorkerResult(
        payload=payload,
        raw_output=raw,
        duration_ms=result.duration_ms,
        cost_usd=0.0,
        session_id=session_id,
        trace_output=trace_output,
    )


def resolve_agent_mode(step: str, args: argparse.Namespace, *, home: Path | None = None) -> tuple[str, str, bool]:
    """Returns (agent, mode, refreshed).

    Both agents default to persistent sessions.  Use --fresh to start a new
    persistent session (break continuity) or --ephemeral for a truly one-off
    call with no session saved.
    """
    from megaplan.cli import (
        CliError, DEFAULT_AGENT_ROUTING, KNOWN_AGENTS,
        load_config, detect_available_agents,
    )

    explicit = args.agent
    if explicit:
        if not shutil.which(explicit):
            raise CliError("agent_not_found", f"Agent '{explicit}' not found on PATH")
        agent = explicit
    else:
        config = load_config(home)
        agent = config.get("agents", {}).get(step) or DEFAULT_AGENT_ROUTING[step]
        if not shutil.which(agent):
            available = detect_available_agents()
            if not available:
                raise CliError(
                    "agent_not_found",
                    "No supported agents found on PATH. Install claude or codex.",
                )
            fallback = available[0]
            args._agent_fallback = {
                "requested": agent,
                "resolved": fallback,
                "reason": f"{agent} not found on PATH",
            }
            agent = fallback
    ephemeral = getattr(args, "ephemeral", False)
    fresh = getattr(args, "fresh", False)
    persist = getattr(args, "persist", False)
    conflicting = sum([fresh, persist, ephemeral])
    if conflicting > 1:
        raise CliError("invalid_args", "Cannot combine --fresh, --persist, and --ephemeral")
    if ephemeral:
        return agent, "ephemeral", True
    refreshed = fresh
    # Review with Claude: default to fresh to avoid self-bias (principle #5)
    if step == "review" and agent == "claude":
        if persist and not getattr(args, "confirm_self_review", False):
            raise CliError("invalid_args", "Claude review requires --confirm-self-review when using --persist")
        if not persist:
            refreshed = True
    return agent, "persistent", refreshed


def run_step_with_worker(step: str, state: dict[str, Any], plan_dir: Path, args: argparse.Namespace, *, root: Path) -> tuple[WorkerResult, str, str, bool]:
    agent, mode, refreshed = resolve_agent_mode(step, args)
    if agent == "claude":
        worker = run_claude_step(step, state, plan_dir, root=root, fresh=refreshed)
    else:
        worker = run_codex_step(step, state, plan_dir, root=root, persistent=(mode == "persistent"), fresh=refreshed, json_trace=(step == "execute"))
    return worker, agent, mode, refreshed
