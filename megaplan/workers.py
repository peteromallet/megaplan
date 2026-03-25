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
from typing import Any, Callable

from megaplan.schemas import SCHEMAS
from megaplan.types import (
    CliError,
    DEFAULT_AGENT_ROUTING,
    MOCK_ENV_VAR,
    PlanState,
    SessionInfo,
)
from megaplan._core import (
    detect_available_agents,
    json_dump,
    latest_plan_meta_path,
    load_config,
    now_utc,
    read_json,
    schemas_root,
)
from megaplan.prompts import create_claude_prompt, create_codex_prompt


WORKER_TIMEOUT_SECONDS = 7200

# Shared mapping from step name to schema filename, used by both
# run_claude_step and run_codex_step.
STEP_SCHEMA_FILENAMES: dict[str, str] = {
    "plan": "plan.json",
    "revise": "revise.json",
    "critique": "critique.json",
    "gate": "gate.json",
    "finalize": "finalize.json",
    "execute": "execution.json",
    "review": "review.json",
}

# Derive required keys per step from SCHEMAS so they aren't duplicated.
_STEP_REQUIRED_KEYS: dict[str, list[str]] = {
    step: SCHEMAS[filename].get("required", [])
    for step, filename in STEP_SCHEMA_FILENAMES.items()
}


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


_CODEX_ERROR_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern_substring, error_code, human_message)
    ("rate limit", "rate_limit", "Codex hit a rate limit"),
    ("rate_limit", "rate_limit", "Codex hit a rate limit"),
    ("429", "rate_limit", "Codex hit a rate limit (HTTP 429)"),
    ("quota", "quota_exceeded", "Codex quota exceeded"),
    ("context length", "context_overflow", "Prompt exceeded Codex context length"),
    ("context_length", "context_overflow", "Prompt exceeded Codex context length"),
    ("maximum context", "context_overflow", "Prompt exceeded Codex context length"),
    ("too many tokens", "context_overflow", "Prompt exceeded Codex context length"),
    ("timed out", "worker_timeout", "Codex request timed out"),
    ("timeout", "worker_timeout", "Codex request timed out"),
    ("connection error", "connection_error", "Codex could not connect to the API"),
    ("connection refused", "connection_error", "Codex could not connect to the API"),
    ("internal server error", "api_error", "Codex API returned an internal error"),
    ("500", "api_error", "Codex API returned an internal error (HTTP 500)"),
    ("502", "api_error", "Codex API returned a gateway error (HTTP 502)"),
    ("503", "api_error", "Codex API service unavailable (HTTP 503)"),
    ("model not found", "model_error", "Codex model not found or unavailable"),
    ("permission denied", "permission_error", "Codex permission denied"),
    ("authentication", "auth_error", "Codex authentication failed"),
    ("unauthorized", "auth_error", "Codex authentication failed"),
]


def _diagnose_codex_failure(raw: str, returncode: int) -> tuple[str, str]:
    """Parse Codex stderr/stdout for known error patterns. Returns (error_code, message)."""
    lower = raw.lower()
    for pattern, code, message in _CODEX_ERROR_PATTERNS:
        if pattern in lower:
            return code, f"{message}. Try --agent claude to use a different backend."
    return "worker_error", (
        f"Codex step failed with exit code {returncode} (no recognized error pattern in output). "
        "Try --agent claude to use a different backend."
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
    required = _STEP_REQUIRED_KEYS.get(step)
    if required is None:
        return
    missing = [key for key in required if key not in payload]
    if missing:
        raise CliError("parse_error", f"{step} output missing required keys: {', '.join(missing)}")


def _mock_result(
    payload: dict[str, Any],
    *,
    trace_output: str | None = None,
) -> WorkerResult:
    return WorkerResult(
        payload=payload,
        raw_output=json_dump(payload),
        duration_ms=10,
        cost_usd=0.0,
        session_id=str(uuid.uuid4()),
        trace_output=trace_output,
    )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
            continue
        merged[key] = value
    return merged


def _default_mock_plan_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "plan": textwrap.dedent(
            f"""
            # Implementation Plan: Mock Planning Pass

            ## Overview
            Produce a concrete plan for: {state['idea']}. Keep the scope grounded in the repository and define validation before execution.

            ## Step 1: Inspect the current flow (`megaplan/workers.py`)
            **Scope:** Small
            1. **Inspect** the planner and prompt touch points before editing (`megaplan/workers.py:199`, `megaplan/prompts.py:29`).

            ## Step 2: Implement the smallest viable change (`megaplan/handlers.py`)
            **Scope:** Medium
            1. **Update** the narrowest set of files required to implement the idea (`megaplan/handlers.py:400`).
            2. **Capture** any non-obvious behavior with a short example.
               ```python
               result = "keep the plan structure consistent"
               ```

            ## Step 3: Verify the behavior (`tests/test_megaplan.py`)
            **Scope:** Small
            1. **Run** focused checks that prove the change works (`tests/test_megaplan.py:1`).

            ## Execution Order
            1. Inspect before editing so the plan stays repo-specific.
            2. Implement before expanding verification.

            ## Validation Order
            1. Run targeted tests first.
            2. Run broader checks after the core change lands.
            """
        ).strip(),
        "questions": ["Are there existing patterns in the repo that should be preserved?"],
        "success_criteria": [
            "A concrete implementation path exists.",
            "Verification is defined before execution.",
        ],
        "assumptions": ["The project directory is writable."],
    }
    if state["config"].get("robustness") == "light":
        payload.update(
            {
                "self_flags": [
                    {
                        "id": "FLAG-101",
                        "concern": "The mock plan assumes the existing repo structure stays stable while wiring the light-path checks.",
                        "category": "maintainability",
                        "severity_hint": "likely-minor",
                        "evidence": "The plan intentionally stays narrow and leans on current handler boundaries.",
                    }
                ],
                "gate_recommendation": "PROCEED",
                "gate_rationale": "The combined plan is concrete enough to execute and the remaining self-flag is minor.",
                "settled_decisions": [],
            }
        )
    return payload


def _default_mock_critique_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    iteration = state["iteration"] or 1
    if iteration == 1:
        return {
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
    return {"flags": [], "verified_flag_ids": ["FLAG-001", "FLAG-002"], "disputed_flag_ids": []}


def _default_mock_revise_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    return {
        "plan": textwrap.dedent(
            f"""
            # Implementation Plan: Mock Revision Pass

            ## Overview
            Refine the plan for: {state['idea']}. Tighten file-level scope and keep validation explicit.

            ## Step 1: Reconfirm file scope (`megaplan/handlers.py`)
            **Scope:** Small
            1. **Inspect** the exact edit points before changing the plan (`megaplan/handlers.py:540`).

            ## Step 2: Tighten the implementation slice (`megaplan/workers.py`)
            **Scope:** Medium
            1. **Limit** the plan to the smallest coherent change set (`megaplan/workers.py:256`).
            2. **Illustrate** the intended shape when it helps reviewers.
               ```python
               changes_summary = "Added explicit scope and verification details."
               ```

            ## Step 3: Reconfirm verification (`tests/test_workers.py`)
            **Scope:** Small
            1. **Run** a concrete verification command and record the expected proof point (`tests/test_workers.py:251`).

            ## Execution Order
            1. Re-scope the plan before adjusting implementation details.
            2. Re-run validation after the plan is tightened.

            ## Validation Order
            1. Start with the focused worker and handler tests.
            2. End with the broader suite if the focused checks pass.
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


def _default_mock_gate_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    recommendation = "ITERATE" if state["iteration"] == 1 else "PROCEED"
    return {
        "recommendation": recommendation,
        "rationale": (
            "First critique cycle still needs another pass."
            if recommendation == "ITERATE"
            else "Signals are strong enough to move into execution."
        ),
        "signals_assessment": (
            "Iteration 1 still carries unresolved significant flags and should revise."
            if recommendation == "ITERATE"
            else "Weighted score and loop trajectory support proceeding."
        ),
        "warnings": [],
        "settled_decisions": [],
    }


def _default_mock_finalize_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "T1",
                "description": f"Implement: {state['idea']}",
                "depends_on": [],
                "status": "pending",
                "executor_notes": "",
                "files_changed": [],
                "commands_run": [],
                "evidence_files": [],
                "reviewer_verdict": "",
            },
            {
                "id": "T2",
                "description": "Verify success criteria",
                "depends_on": [],
                "status": "pending",
                "executor_notes": "",
                "files_changed": [],
                "commands_run": [],
                "evidence_files": [],
                "reviewer_verdict": "",
            },
        ],
        "watch_items": ["Ensure repository state matches plan assumptions"],
        "sense_checks": [
            {
                "id": "SC1",
                "task_id": "T1",
                "question": "Verify implementation matches the stated idea.",
                "executor_note": "",
                "verdict": "",
            },
            {
                "id": "SC2",
                "task_id": "T2",
                "question": "Verify success criteria were actually checked.",
                "executor_note": "",
                "verdict": "",
            },
        ],
        "meta_commentary": "This is a mock finalize output.",
    }


def _task_ids_from_prompt_override(prompt_override: str | None) -> set[str] | None:
    if prompt_override is None:
        return None
    match = re.search(r"Only produce `?task_updates`? for these tasks:\s*\[([^\]]*)\]", prompt_override)
    if match is None:
        return None
    task_ids = {item.strip() for item in match.group(1).split(",") if item.strip()}
    return task_ids


def _default_mock_execute_payload(
    state: PlanState,
    plan_dir: Path,
    *,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    target = Path(state["config"]["project_dir"]) / "IMPLEMENTED_BY_MEGAPLAN.txt"
    relative_target = str(target.relative_to(Path(state["config"]["project_dir"])))
    payload = {
        "output": "Mock execution completed successfully.",
        "files_changed": [relative_target],
        "commands_run": ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"],
        "deviations": [],
        "task_updates": [
            {
                "task_id": "T1",
                "status": "done",
                "executor_notes": "Implemented via mock worker output and wrote IMPLEMENTED_BY_MEGAPLAN.txt.",
                "files_changed": [relative_target],
                "commands_run": ["mock-write IMPLEMENTED_BY_MEGAPLAN.txt"],
            },
            {
                "task_id": "T2",
                "status": "done",
                "executor_notes": "Verified success criteria via mock worker output and command checks.",
                "files_changed": [],
                "commands_run": ["mock-verify success criteria"],
            },
        ],
        "sense_check_acknowledgments": [
            {
                "sense_check_id": "SC1",
                "executor_note": "Confirmed the implementation artifact was written for the main task.",
            },
            {
                "sense_check_id": "SC2",
                "executor_note": "Confirmed the verification-only task is backed by command evidence.",
            },
        ],
    }
    batch_task_ids = _task_ids_from_prompt_override(prompt_override)
    if batch_task_ids is None:
        return payload
    payload["task_updates"] = [
        task_update
        for task_update in payload["task_updates"]
        if task_update["task_id"] in batch_task_ids
    ]
    payload["sense_check_acknowledgments"] = [
        acknowledgment
        for acknowledgment in payload["sense_check_acknowledgments"]
        if acknowledgment["sense_check_id"] in {
            f"SC{task_id[1:]}"
            for task_id in batch_task_ids
            if task_id.startswith("T")
        }
    ]
    return payload


def _default_mock_review_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    meta = read_json(latest_plan_meta_path(plan_dir, state))
    criteria = [
        {"name": criterion, "pass": True, "evidence": "Mock execution and artifacts satisfy the criterion."}
        for criterion in meta.get("success_criteria", [])
    ]
    return {
        "review_verdict": "approved",
        "criteria": criteria,
        "issues": [],
        "summary": "Mock review passed.",
        "task_verdicts": [
            {
                "task_id": "T1",
                "reviewer_verdict": "Pass - mock verified with file-backed implementation evidence.",
                "evidence_files": [str((Path(state["config"]["project_dir"]) / "IMPLEMENTED_BY_MEGAPLAN.txt").relative_to(Path(state["config"]["project_dir"])))],
            },
            {
                "task_id": "T2",
                "reviewer_verdict": "Pass - verification task was reviewed via command evidence and executor notes rather than a changed file.",
                "evidence_files": [],
            },
        ],
        "sense_check_verdicts": [
            {"sense_check_id": "SC1", "verdict": "Confirmed."},
            {"sense_check_id": "SC2", "verdict": "Confirmed."},
        ],
    }


_MockPayloadBuilder = Callable[[PlanState, Path], dict[str, Any]]

_MOCK_DEFAULTS: dict[str, _MockPayloadBuilder] = {
    "plan": _default_mock_plan_payload,
    "critique": _default_mock_critique_payload,
    "revise": _default_mock_revise_payload,
    "gate": _default_mock_gate_payload,
    "finalize": _default_mock_finalize_payload,
    "execute": _default_mock_execute_payload,
    "review": _default_mock_review_payload,
}


def _build_mock_payload(step: str, state: PlanState, plan_dir: Path, **overrides: Any) -> dict[str, Any]:
    builder = _MOCK_DEFAULTS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Mock worker does not support '{step}'")
    prompt_override = overrides.pop("prompt_override", None)
    if step == "execute":
        return _deep_merge(_default_mock_execute_payload(state, plan_dir, prompt_override=prompt_override), overrides)
    return _deep_merge(builder(state, plan_dir), overrides)


def _mock_plan(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("plan", state, plan_dir))


def _mock_critique(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("critique", state, plan_dir))


def _mock_revise(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("revise", state, plan_dir))


def _mock_gate(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("gate", state, plan_dir))


def _mock_finalize(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("finalize", state, plan_dir))


def _mock_execute(state: PlanState, plan_dir: Path, *, prompt_override: str | None = None) -> WorkerResult:
    target = Path(state["config"]["project_dir"]) / "IMPLEMENTED_BY_MEGAPLAN.txt"
    target.write_text("mock execution completed\n", encoding="utf-8")
    return _mock_result(
        _build_mock_payload("execute", state, plan_dir, prompt_override=prompt_override),
        trace_output='{"event":"mock-execute"}\n',
    )


def _mock_review(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("review", state, plan_dir))


_MockHandler = Callable[..., WorkerResult]

_MOCK_DISPATCH: dict[str, _MockHandler] = {
    "plan": _mock_plan,
    "critique": _mock_critique,
    "revise": _mock_revise,
    "gate": _mock_gate,
    "finalize": _mock_finalize,
    "execute": _mock_execute,
    "review": _mock_review,
}


def mock_worker_output(
    step: str,
    state: PlanState,
    plan_dir: Path,
    *,
    prompt_override: str | None = None,
) -> WorkerResult:
    handler = _MOCK_DISPATCH.get(step)
    if handler is None:
        raise CliError("unsupported_step", f"Mock worker does not support '{step}'")
    if step == "execute":
        return handler(state, plan_dir, prompt_override=prompt_override)
    return handler(state, plan_dir)


def session_key_for(step: str, agent: str) -> str:
    if step in {"plan", "revise"}:
        return f"{agent}_planner"
    if step == "critique":
        return f"{agent}_critic"
    if step == "gate":
        return f"{agent}_gatekeeper"
    if step == "finalize":
        return f"{agent}_finalizer"
    if step == "execute":
        return f"{agent}_executor"
    if step == "review":
        return f"{agent}_reviewer"
    return f"{agent}_{step}"


def update_session_state(step: str, agent: str, session_id: str | None, *, mode: str, refreshed: bool, existing_sessions: dict[str, Any] | None = None) -> tuple[str, SessionInfo] | None:
    """Build a session entry for the given step.

    Returns ``(key, entry)`` so the caller can store it on the state dict,
    or ``None`` when there is no session_id to record.
    """
    if not session_id:
        return None
    key = session_key_for(step, agent)
    if existing_sessions is None:
        existing_sessions = {}
    entry = {
        "id": session_id,
        "mode": mode,
        "created_at": existing_sessions.get(key, {}).get("created_at", now_utc()),
        "last_used_at": now_utc(),
        "refreshed": refreshed,
    }
    return key, entry


def run_claude_step(
    step: str,
    state: PlanState,
    plan_dir: Path,
    *,
    root: Path,
    fresh: bool,
    prompt_override: str | None = None,
) -> WorkerResult:
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir, prompt_override=prompt_override)
    project_dir = Path(state["config"]["project_dir"])
    schema_name = STEP_SCHEMA_FILENAMES[step]
    schema_text = json.dumps(read_json(schemas_root(root) / schema_name))
    session_key = session_key_for(step, "claude")
    session = state["sessions"].get(session_key, {})
    session_id = session.get("id")
    command = ["claude", "-p", "--output-format", "json", "--json-schema", schema_text, "--add-dir", str(project_dir)]
    if step == "execute":
        command.extend(["--permission-mode", "bypassPermissions"])
    if session_id and not fresh:
        command.extend(["--resume", session_id])
    else:
        session_id = str(uuid.uuid4())
        command.extend(["--session-id", session_id])
    prompt = prompt_override if prompt_override is not None else create_claude_prompt(step, state, plan_dir)
    try:
        result = run_command(command, cwd=project_dir, stdin_text=prompt)
    except CliError as error:
        if error.code == "worker_timeout":
            error.extra["session_id"] = session_id
        raise
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
    state: PlanState,
    plan_dir: Path,
    *,
    root: Path,
    persistent: bool,
    fresh: bool = False,
    json_trace: bool = False,
    prompt_override: str | None = None,
) -> WorkerResult:
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir, prompt_override=prompt_override)
    project_dir = Path(state["config"]["project_dir"])
    schema_file = schemas_root(root) / STEP_SCHEMA_FILENAMES[step]
    session_key = session_key_for(step, "codex")
    session = state["sessions"].get(session_key, {})
    out_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    out_handle.close()
    output_path = Path(out_handle.name)
    prompt = prompt_override if prompt_override is not None else create_codex_prompt(step, state, plan_dir)

    if persistent and session.get("id") and not fresh:
        # codex exec resume does not support --output-schema; we rely on
        # validate_payload() after parsing the output file instead.
        command = ["codex", "exec", "resume"]
        if step == "execute":
            command.append("--full-auto")
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

    try:
        result = run_command(command, cwd=Path.cwd(), stdin_text=prompt)
    except CliError as error:
        if error.code == "worker_timeout":
            timeout_session_id = session.get("id") if persistent else None
            if timeout_session_id is None:
                timeout_session_id = extract_session_id(error.extra.get("raw_output", ""))
            if timeout_session_id is not None:
                error.extra["session_id"] = timeout_session_id
        raise
    raw = result.stdout + result.stderr
    if result.returncode != 0 and (not output_path.exists() or not output_path.read_text(encoding="utf-8").strip()):
        error_code, error_message = _diagnose_codex_failure(raw, result.returncode)
        raise CliError(error_code, error_message, extra={"raw_output": raw})
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


def run_step_with_worker(
    step: str,
    state: PlanState,
    plan_dir: Path,
    args: argparse.Namespace,
    *,
    root: Path,
    resolved: tuple[str, str, bool] | None = None,
    prompt_override: str | None = None,
) -> tuple[WorkerResult, str, str, bool]:
    agent, mode, refreshed = resolved or resolve_agent_mode(step, args)
    if agent == "claude":
        worker = run_claude_step(step, state, plan_dir, root=root, fresh=refreshed, prompt_override=prompt_override)
    else:
        worker = run_codex_step(
            step,
            state,
            plan_dir,
            root=root,
            persistent=(mode == "persistent"),
            fresh=refreshed,
            json_trace=(step == "execute"),
            prompt_override=prompt_override,
        )
    return worker, agent, mode, refreshed
