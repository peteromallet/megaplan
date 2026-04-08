"""Worker orchestration: running Claude and Codex steps."""

from __future__ import annotations

import argparse
import hashlib
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

from megaplan.checks import build_empty_template, checks_for_robustness
from megaplan.schemas import SCHEMAS
from megaplan.types import (
    CliError,
    DEFAULT_AGENT_ROUTING,
    MOCK_ENV_VAR,
    PlanState,
    SessionInfo,
    parse_agent_spec,
)
from megaplan._core import (
    configured_robustness,
    detect_available_agents,
    get_effective,
    json_dump,
    latest_plan_meta_path,
    load_config,
    now_utc,
    read_json,
    schemas_root,
)
from megaplan.prompts import create_claude_prompt, create_codex_prompt


WORKER_TIMEOUT_SECONDS = 7200
_EXECUTE_STEPS = {"execute", "loop_execute"}

# Shared mapping from step name to schema filename, used by both
# run_claude_step and run_codex_step.
STEP_SCHEMA_FILENAMES: dict[str, str] = {
    "plan": "plan.json",
    "prep": "prep.json",
    "revise": "revise.json",
    "critique": "critique.json",
    "gate": "gate.json",
    "finalize": "finalize.json",
    "execute": "execution.json",
    "loop_plan": "loop_plan.json",
    "loop_execute": "loop_execute.json",
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def run_command(
    command: list[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    timeout: int | None = None,
) -> CommandResult:
    started = time.monotonic()
    timeout = timeout or get_effective("execution", "worker_timeout_seconds")
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


def _extract_json_from_raw(raw: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from raw agent output.

    Handles the common case where a sandbox blocks file writes and the agent
    dumps the JSON into stdout/stderr wrapped in error text or markdown fences.
    """
    # Strategy 1: look for ```json ... ``` fenced blocks
    fenced = re.findall(r"```json\s*\n(.*?)```", raw, re.DOTALL)
    for block in fenced:
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    # Strategy 2: find the largest { ... } substring that parses as JSON
    # (greedy match from first { to last })
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidate = raw[brace_start : brace_end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


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
            {"criterion": "A concrete implementation path exists.", "priority": "must"},
            {"criterion": "Verification is defined before execution.", "priority": "should"},
        ],
        "assumptions": ["The project directory is writable."],
    }
    return payload


def _default_mock_prep_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    del plan_dir
    return {
        "task_summary": str(state.get("idea", "")).strip() or "Prepare a concise engineering brief for the requested task.",
        "key_evidence": [],
        "relevant_code": [],
        "test_expectations": [],
        "constraints": [],
        "suggested_approach": "Inspect the code paths named in the task, read nearby tests first when they exist, then carry the distilled brief into planning.",
    }


def _loop_goal(state: dict[str, Any]) -> str:
    return str(state.get("idea", state.get("spec", {}).get("goal", "")))


def _default_mock_loop_plan_payload(state: dict[str, Any], plan_dir: Path) -> dict[str, Any]:
    spec = state.get("spec", {})
    goal = _loop_goal(state)
    return {
        "spec_updates": {
            "known_issues": spec.get("known_issues", []),
            "tried_and_failed": spec.get("tried_and_failed", []),
            "best_result_summary": f"Most recent mock planning pass for: {goal}",
        },
        "next_action": "Run the project command, inspect the failures, and prepare the next minimal fix.",
        "reasoning": "The loop spec is initialized and ready for an execution pass based on the current goal and retained context.",
    }


def _default_mock_loop_execute_payload(
    state: dict[str, Any],
    plan_dir: Path,
    *,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    spec = state.get("spec", {})
    goal = _loop_goal(state)
    return {
        "diagnosis": f"Mock execution diagnosis for goal: {goal}",
        "fix_description": "Inspect the command failure, update the smallest relevant file, and rerun the command.",
        "files_to_change": list(spec.get("allowed_changes", []))[:3],
        "confidence": "medium",
        "outcome": "continue",
        "should_pause": False,
    }


def _default_mock_critique_payload(state: PlanState, plan_dir: Path) -> dict[str, Any]:
    iteration = state["iteration"] or 1
    del plan_dir
    active_checks = checks_for_robustness(configured_robustness(state))
    checks = build_empty_template(active_checks)
    if iteration == 1:
        return {
            "checks": [
                {
                    **check,
                    "findings": [
                        {
                            "detail": "Mock critique found a concrete repository issue that should be addressed before proceeding.",
                            "flagged": True,
                        }
                    ],
                }
                for check in checks
            ],
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
    return {
        "checks": [
            {
                **check,
                "findings": [
                    {
                        "detail": "Mock critique verified the revised plan against the repository context and found no remaining issue.",
                        "flagged": False,
                    }
                ],
            }
            for check in checks
        ],
        "flags": [],
        "verified_flag_ids": [*(check["id"] for check in checks), "FLAG-001", "FLAG-002"],
        "disputed_flag_ids": [],
    }



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
            {"criterion": "The plan identifies exact touch points before editing.", "priority": "must"},
            {"criterion": "A concrete verification command is defined.", "priority": "should"},
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
        "validation": {
            "plan_steps_covered": [
                {"plan_step_summary": f"Implement: {state['idea']}", "finalize_task_ids": ["T1"]},
                {"plan_step_summary": "Verify success criteria", "finalize_task_ids": ["T2"]},
            ],
            "orphan_tasks": [],
            "completeness_notes": "All plan steps mapped to tasks.",
            "coverage_complete": True,
        },
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
    criteria = []
    for entry in meta.get("success_criteria", []):
        if isinstance(entry, dict):
            name = entry.get("criterion", str(entry))
            priority = entry.get("priority", "must")
        else:
            name = str(entry)
            priority = "must"
        criteria.append({"name": name, "priority": priority, "pass": "pass", "evidence": "Mock execution and artifacts satisfy the criterion."})
    return {
        "review_verdict": "approved",
        "criteria": criteria,
        "issues": [],
        "rework_items": [],
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


_MockPayloadBuilder = Callable[[dict[str, Any], Path], dict[str, Any]]

_MOCK_DEFAULTS: dict[str, _MockPayloadBuilder] = {
    "plan": _default_mock_plan_payload,
    "prep": _default_mock_prep_payload,
    "loop_plan": _default_mock_loop_plan_payload,
    "critique": _default_mock_critique_payload,
    "revise": _default_mock_revise_payload,
    "gate": _default_mock_gate_payload,
    "finalize": _default_mock_finalize_payload,
    "execute": _default_mock_execute_payload,
    "loop_execute": _default_mock_loop_execute_payload,
    "review": _default_mock_review_payload,
}


def _build_mock_payload(step: str, state: dict[str, Any], plan_dir: Path, **overrides: Any) -> dict[str, Any]:
    builder = _MOCK_DEFAULTS.get(step)
    if builder is None:
        raise CliError("unsupported_step", f"Mock worker does not support '{step}'")
    prompt_override = overrides.pop("prompt_override", None)
    if step in _EXECUTE_STEPS:
        if step == "loop_execute":
            return _deep_merge(_default_mock_loop_execute_payload(state, plan_dir, prompt_override=prompt_override), overrides)
        return _deep_merge(_default_mock_execute_payload(state, plan_dir, prompt_override=prompt_override), overrides)
    return _deep_merge(builder(state, plan_dir), overrides)


def _mock_plan(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("plan", state, plan_dir))


def _mock_prep(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("prep", state, plan_dir))


def _mock_loop_plan(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("loop_plan", state, plan_dir))



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


def _mock_loop_execute(state: PlanState, plan_dir: Path, *, prompt_override: str | None = None) -> WorkerResult:
    return _mock_result(
        _build_mock_payload("loop_execute", state, plan_dir, prompt_override=prompt_override),
        trace_output='{"event":"mock-loop-execute"}\n',
    )


def _mock_review(state: PlanState, plan_dir: Path) -> WorkerResult:
    return _mock_result(_build_mock_payload("review", state, plan_dir))


_MockHandler = Callable[..., WorkerResult]

_MOCK_DISPATCH: dict[str, _MockHandler] = {
    "plan": _mock_plan,
    "prep": _mock_prep,
    "loop_plan": _mock_loop_plan,
    "critique": _mock_critique,
    "revise": _mock_revise,
    "gate": _mock_gate,
    "finalize": _mock_finalize,
    "execute": _mock_execute,
    "loop_execute": _mock_loop_execute,
    "review": _mock_review,
}


def mock_worker_output(
    step: str,
    state: PlanState,
    plan_dir: Path,
    *,
    prompt_override: str | None = None,
    prompt_kwargs: dict[str, Any] | None = None,
) -> WorkerResult:
    del prompt_kwargs
    handler = _MOCK_DISPATCH.get(step)
    if handler is None:
        raise CliError("unsupported_step", f"Mock worker does not support '{step}'")
    if step in _EXECUTE_STEPS:
        return handler(state, plan_dir, prompt_override=prompt_override)
    return handler(state, plan_dir)


def session_key_for(step: str, agent: str, model: str | None = None) -> str:
    if step in {"plan", "revise"}:
        key = f"{agent}_planner"
    elif step == "critique":
        key = f"{agent}_critic"
    elif step == "gate":
        key = f"{agent}_gatekeeper"
    elif step == "finalize":
        key = f"{agent}_finalizer"
    elif step == "execute":
        key = f"{agent}_executor"
    elif step == "review":
        key = f"{agent}_reviewer"
    else:
        key = f"{agent}_{step}"
    if model:
        key += f"_{hashlib.sha256(model.encode()).hexdigest()[:8]}"
    return key


def update_session_state(step: str, agent: str, session_id: str | None, *, mode: str, refreshed: bool, model: str | None = None, existing_sessions: dict[str, Any] | None = None) -> tuple[str, SessionInfo] | None:
    """Build a session entry for the given step.

    Returns ``(key, entry)`` so the caller can store it on the state dict,
    or ``None`` when there is no session_id to record.
    """
    if not session_id:
        return None
    key = session_key_for(step, agent, model=model)
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
    prompt_kwargs: dict[str, Any] | None = None,
) -> WorkerResult:
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir, prompt_override=prompt_override, prompt_kwargs=prompt_kwargs)
    project_dir = Path(state["config"]["project_dir"])
    schema_name = STEP_SCHEMA_FILENAMES[step]
    schema_text = json.dumps(read_json(schemas_root(root) / schema_name))
    session_key = session_key_for(step, "claude")
    session = state["sessions"].get(session_key, {})
    session_id = session.get("id")
    command = ["claude", "-p", "--output-format", "json", "--json-schema", schema_text, "--add-dir", str(project_dir)]
    if step in _EXECUTE_STEPS:
        command.extend(["--permission-mode", "bypassPermissions"])
    if session_id and not fresh:
        command.extend(["--resume", session_id])
    else:
        session_id = str(uuid.uuid4())
        command.extend(["--session-id", session_id])
    prompt = prompt_override if prompt_override is not None else create_claude_prompt(
        step,
        state,
        plan_dir,
        root=root,
        **(prompt_kwargs or {}),
    )
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
    prompt_kwargs: dict[str, Any] | None = None,
) -> WorkerResult:
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir, prompt_override=prompt_override, prompt_kwargs=prompt_kwargs)
    project_dir = Path(state["config"]["project_dir"])
    schema_file = schemas_root(root) / STEP_SCHEMA_FILENAMES[step]
    session_key = session_key_for(step, "codex")
    session = state["sessions"].get(session_key, {})
    out_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    out_handle.close()
    output_path = Path(out_handle.name)
    prompt = prompt_override if prompt_override is not None else create_codex_prompt(
        step,
        state,
        plan_dir,
        root=root,
        **(prompt_kwargs or {}),
    )

    if persistent and session.get("id") and not fresh:
        # codex exec resume does not support --output-schema; we rely on
        # validate_payload() after parsing the output file instead.
        command = ["codex", "exec", "resume"]
        if step in _EXECUTE_STEPS:
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
        if step in _EXECUTE_STEPS:
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
    payload = None
    try:
        payload = parse_json_file(output_path)
    except CliError:
        pass
    # Fallback 1: when resuming a persistent session, --output-schema is not
    # supported so Codex may write to the plan-dir output file instead of
    # the temp file.  Try the known output path before giving up.
    if payload is None:
        fallback_names = {
            "critique": "critique_output.json",
        }
        fallback_name = fallback_names.get(step, f"{step}_output.json")
        fallback_path = plan_dir / fallback_name
        if fallback_path != output_path and fallback_path.exists():
            try:
                payload = parse_json_file(fallback_path)
            except CliError:
                pass
    # Fallback 2: sandbox may block file writes.  The agent dumps JSON
    # into stdout/stderr wrapped in error text (e.g. "Read-only sandbox
    # prevented writing ... ```json\n{...}\n```").  Extract it.
    if payload is None:
        payload = _extract_json_from_raw(raw)
    if payload is None:
        raise CliError("parse_error", f"Output file {output_path.name} was not valid JSON and no fallback found", extra={"raw_output": raw})
    # Fallback 3: the output file may have been written successfully but contain
    # a wrapper/error payload instead of the actual structured output.  If the
    # raw output contains a better payload (e.g. with populated findings),
    # prefer it.
    raw_payload = _extract_json_from_raw(raw)
    if raw_payload is not None and step == "critique":
        # Check if the raw version has more content (e.g. populated findings
        # vs empty template that the file-based parse returned).
        raw_checks = raw_payload.get("checks", [])
        file_checks = payload.get("checks", [])
        raw_findings = sum(len(c.get("findings", [])) for c in raw_checks if isinstance(c, dict))
        file_findings = sum(len(c.get("findings", [])) for c in file_checks if isinstance(c, dict))
        if raw_findings > file_findings:
            payload = raw_payload
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


def _is_agent_available(agent: str) -> bool:
    """Check if an agent is available (CLI binary or importable for hermes)."""
    if agent == "hermes":
        try:
            import run_agent  # noqa: F401
            return True
        except ImportError:
            return False
    return bool(shutil.which(agent))


def resolve_agent_mode(step: str, args: argparse.Namespace, *, home: Path | None = None) -> tuple[str, str, bool, str | None]:
    """Returns (agent, mode, refreshed, model).

    Both agents default to persistent sessions.  Use --fresh to start a new
    persistent session (break continuity) or --ephemeral for a truly one-off
    call with no session saved.

    The model is extracted from compound agent specs (e.g. 'hermes:openai/gpt-5')
    or from --phase-model / --hermes CLI flags. None means use agent default.
    """
    model = None

    # Check --phase-model overrides first (highest priority)
    phase_models = getattr(args, "phase_model", None) or []
    for pm in phase_models:
        if "=" in pm:
            pm_step, pm_spec = pm.split("=", 1)
            if pm_step == step:
                agent, model = parse_agent_spec(pm_spec)
                break
    else:
        # Check --hermes flag
        hermes_flag = getattr(args, "hermes", None)
        if hermes_flag is not None:
            agent = "hermes"
            if isinstance(hermes_flag, str) and hermes_flag:
                model = hermes_flag
        else:
            # Check explicit --agent flag
            explicit = args.agent
            if explicit:
                agent, model = parse_agent_spec(explicit)
            else:
                # Fall back to config / defaults
                config = load_config(home)
                spec = config.get("agents", {}).get(step) or DEFAULT_AGENT_ROUTING[step]
                agent, model = parse_agent_spec(spec)

    # Validate agent availability
    explicit_agent = args.agent  # was an explicit --agent flag used?
    if not _is_agent_available(agent):
        # If explicitly requested (via --agent), fail immediately
        if explicit_agent and not any(pm.startswith(f"{step}=") for pm in (getattr(args, "phase_model", None) or [])):
            if agent == "hermes":
                from megaplan.hermes_worker import check_hermes_available
                ok, msg = check_hermes_available()
                raise CliError("agent_not_found", msg if not ok else f"Agent '{agent}' not found")
            raise CliError("agent_not_found", f"Agent '{agent}' not found on PATH")
        # For hermes via --hermes flag, give a specific error
        if getattr(args, "hermes", None) is not None or agent == "hermes":
            from megaplan.hermes_worker import check_hermes_available
            ok, msg = check_hermes_available()
            if not ok:
                raise CliError("agent_not_found", msg)
        # Try fallback
        available = detect_available_agents()
        if not available:
            raise CliError(
                "agent_not_found",
                "No supported agents found. Install claude, codex, or hermes-agent.",
            )
        fallback = available[0]
        args._agent_fallback = {
            "requested": agent,
            "resolved": fallback,
            "reason": f"{agent} not available",
        }
        agent = fallback
        model = None  # Reset model when falling back

    ephemeral = getattr(args, "ephemeral", False)
    fresh = getattr(args, "fresh", False)
    persist = getattr(args, "persist", False)
    conflicting = sum([fresh, persist, ephemeral])
    if conflicting > 1:
        raise CliError("invalid_args", "Cannot combine --fresh, --persist, and --ephemeral")
    if ephemeral:
        return agent, "ephemeral", True, model
    refreshed = fresh
    # Review with Claude: default to fresh to avoid self-bias (principle #5)
    if step == "review" and agent == "claude":
        if persist and not getattr(args, "confirm_self_review", False):
            raise CliError("invalid_args", "Claude review requires --confirm-self-review when using --persist")
        if not persist:
            refreshed = True
    return agent, "persistent", refreshed, model


def run_step_with_worker(
    step: str,
    state: PlanState,
    plan_dir: Path,
    args: argparse.Namespace,
    *,
    root: Path,
    resolved: tuple[str, str, bool, str | None] | None = None,
    prompt_override: str | None = None,
    prompt_kwargs: dict[str, Any] | None = None,
) -> tuple[WorkerResult, str, str, bool]:
    agent, mode, refreshed, model = resolved or resolve_agent_mode(step, args)
    if agent == "hermes":
        # Deferred import to avoid circular import (hermes_worker imports from workers)
        from megaplan.hermes_worker import run_hermes_step
        worker = run_hermes_step(
            step,
            state,
            plan_dir,
            root=root,
            fresh=refreshed,
            model=model,
            prompt_override=prompt_override,
        )
    elif agent == "claude":
        worker = run_claude_step(
            step,
            state,
            plan_dir,
            root=root,
            fresh=refreshed,
            prompt_override=prompt_override,
            prompt_kwargs=prompt_kwargs,
        )
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
            prompt_kwargs=prompt_kwargs,
        )
    return worker, agent, mode, refreshed
