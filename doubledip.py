#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


STATE_INITIALIZED = "initialized"
STATE_PLANNED = "planned"
STATE_CRITIQUED = "critiqued"
STATE_EVALUATED = "evaluated"
STATE_GATED = "gated"
STATE_EXECUTED = "executed"
STATE_DONE = "done"
STATE_ABORTED = "aborted"
TERMINAL_STATES = {STATE_DONE, STATE_ABORTED}
FLAG_BLOCKING_STATUSES = {"open", "disputed"}
MOCK_ENV_VAR = "DOUBLEDIP_MOCK_WORKERS"
WORKER_TIMEOUT_SECONDS = 600

SCHEMAS: dict[str, dict[str, Any]] = {
    "plan.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plan", "questions", "success_criteria", "assumptions"],
    },
    "integrate.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "changes_summary": {"type": "string"},
            "flags_addressed": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plan", "changes_summary", "flags_addressed"],
    },
    "critique.json": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "concern": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "correctness",
                                "security",
                                "completeness",
                                "performance",
                                "maintainability",
                                "other",
                            ],
                        },
                        "severity_hint": {
                            "type": "string",
                            "enum": ["likely-significant", "likely-minor", "uncertain"],
                        },
                        "evidence": {"type": "string"},
                    },
                    "required": ["id", "concern", "category", "severity_hint", "evidence"],
                },
            },
            "verified_flag_ids": {"type": "array", "items": {"type": "string"}},
            "disputed_flag_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["flags"],
    },
    "execution.json": {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "commands_run": {"type": "array", "items": {"type": "string"}},
            "deviations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["output", "files_changed", "commands_run", "deviations"],
    },
    "review.json": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "pass": {"type": "boolean"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["name", "pass", "evidence"],
                },
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["criteria", "issues"],
    },
}


def strict_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        updated = {key: strict_schema(value) for key, value in schema.items()}
        if updated.get("type") == "object":
            updated.setdefault("additionalProperties", False)
            if "properties" in updated:
                updated["required"] = list(updated["properties"].keys())
        return updated
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    return schema


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
    doubledip_root = root / ".doubledip"
    (doubledip_root / "plans").mkdir(parents=True, exist_ok=True)
    schemas_dir = doubledip_root / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in SCHEMAS.items():
        atomic_write_json(schemas_dir / filename, strict_schema(schema))


def doubledip_root(root: Path) -> Path:
    return root / ".doubledip"


def plans_root(root: Path) -> Path:
    return doubledip_root(root) / "plans"


def schemas_root(root: Path) -> Path:
    return doubledip_root(root) / "schemas"


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
    payload.update(error.extra)
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
    path = plan_dir / "flags.json"
    if path.exists():
        return read_json(path)
    return {"flags": []}


def save_flag_registry(plan_dir: Path, data: dict[str, Any]) -> None:
    atomic_write_json(plan_dir / "flags.json", data)


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


def compute_plan_delta_percent(previous_text: str | None, current_text: str) -> float | None:
    if previous_text is None:
        return None
    ratio = SequenceMatcher(None, previous_text, current_text).ratio()
    return round((1.0 - ratio) * 100.0, 2)


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
        return ["plan"]
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


def ensure_command(name: str) -> str | None:
    return shutil.which(name)


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
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        raise CliError(
            "worker_timeout",
            f"Command timed out after {timeout}s: {' '.join(command[:3])}...",
            extra={"raw_output": (exc.stdout or "") + (exc.stderr or "")},
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
    def require_keys(keys: list[str]) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise CliError("parse_error", f"{step} output missing required keys: {', '.join(missing)}")

    if step == "plan":
        require_keys(["plan", "questions", "success_criteria", "assumptions"])
    elif step == "integrate":
        require_keys(["plan", "changes_summary", "flags_addressed"])
    elif step == "critique":
        require_keys(["flags"])
    elif step == "execute":
        require_keys(["output", "files_changed", "commands_run", "deviations"])
    elif step == "review":
        require_keys(["criteria", "issues"])


def create_claude_prompt(step: str, state: dict[str, Any], plan_dir: Path) -> str:
    project_dir = Path(state["config"]["project_dir"])
    notes = state.get("meta", {}).get("notes", [])
    notes_block = "\n".join(f"- {note['note']}" for note in notes) if notes else "- None"
    latest_plan = ""
    latest_meta: dict[str, Any] = {}
    if state.get("plan_versions"):
        latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
        latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)

    if step == "plan":
        return textwrap.dedent(
            f"""
            You are creating an implementation plan for the following idea.

            Idea:
            {state['idea']}

            Project directory:
            {project_dir}

            User notes:
            {notes_block}

            Requirements:
            - Inspect the actual repository before planning.
            - Produce a concrete implementation plan in markdown.
            - Define observable success criteria.
            - Call out assumptions and open questions.
            - Prefer cheap validation steps early.
            """
        ).strip()

    if step == "integrate":
        evaluate_path = current_iteration_artifact(plan_dir, "evaluation", state["iteration"])
        evaluation = read_json(evaluate_path)
        unresolved = unresolved_significant_flags(flag_registry)
        open_flags = [
            {
                "id": flag["id"],
                "severity": flag.get("severity"),
                "status": flag.get("status"),
                "concern": flag.get("concern"),
                "evidence": flag.get("evidence"),
            }
            for flag in unresolved
        ]
        return textwrap.dedent(
            f"""
            You are updating an implementation plan based on critique and evaluation.

            Project directory:
            {project_dir}

            Current plan (markdown):
            {latest_plan}

            Current plan metadata:
            {json_dump(latest_meta).strip()}

            Evaluation:
            {json_dump(evaluation).strip()}

            Open significant flags:
            {json_dump(open_flags).strip()}

            Requirements:
            - Update the plan to address the significant issues.
            - Keep the plan readable and executable.
            - Return flags_addressed with the exact flag IDs you addressed.
            - Preserve or improve success criteria quality.
            """
        ).strip()

    if step == "review":
        execution = read_json(plan_dir / "execution.json")
        gate = read_json(plan_dir / "gate.json")
        diff_summary = collect_git_diff_summary(project_dir)
        return textwrap.dedent(
            f"""
            Review the execution critically against user intent and observable success criteria.

            Project directory:
            {project_dir}

            Approved plan:
            {latest_plan}

            Plan metadata:
            {json_dump(latest_meta).strip()}

            Gate summary:
            {json_dump(gate).strip()}

            Execution summary:
            {json_dump(execution).strip()}

            Git diff summary:
            {diff_summary}

            Requirements:
            - Judge against the success criteria, not plan elegance.
            - Be critical and call out real misses.
            - If there are failures, describe them as issues.
            """
        ).strip()

    if step in {"critique", "execute"}:
        return create_codex_prompt(step, state, plan_dir)

    raise CliError("unsupported_step", f"Unsupported Claude step '{step}'")


def create_codex_prompt(step: str, state: dict[str, Any], plan_dir: Path) -> str:
    if step in {"plan", "integrate"}:
        return create_claude_prompt(step, state, plan_dir)
    project_dir = Path(state["config"]["project_dir"])
    latest_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    latest_meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)

    if step == "critique":
        unresolved = [
            {
                "id": flag["id"],
                "concern": flag["concern"],
                "status": flag.get("status"),
                "severity": flag.get("severity"),
            }
            for flag in flag_registry.get("flags", [])
            if flag.get("status") in {"addressed", "open", "disputed"}
        ]
        return textwrap.dedent(
            f"""
            You are an independent reviewer. Critique the plan against the actual repository.

            Project directory:
            {project_dir}

            Plan:
            {latest_plan}

            Plan metadata:
            {json_dump(latest_meta).strip()}

            Existing flags:
            {json_dump(unresolved).strip()}

            Requirements:
            - Reuse existing flag IDs when the same concern is still open.
            - verified_flag_ids should list previously addressed flags that now appear resolved.
            - Focus on concrete issues that would cause real problems.
            - Do not rubber-stamp the plan.
            - Assign severity_hint carefully: "likely-significant" for issues that would
              cause real product or implementation problems. "likely-minor" for cosmetic,
              nice-to-have, or issues already covered elsewhere.
            """
        ).strip()

    if step == "execute":
        gate = read_json(plan_dir / "gate.json")
        return textwrap.dedent(
            f"""
            Execute the approved plan in the repository.

            Project directory:
            {project_dir}

            Approved plan:
            {latest_plan}

            Plan metadata:
            {json_dump(latest_meta).strip()}

            Gate summary:
            {json_dump(gate).strip()}

            Requirements:
            - Implement the intent, not just the text.
            - Adapt if repository reality contradicts the plan.
            - Report deviations explicitly.
            - Output concrete files changed and commands run.
            """
        ).strip()

    if step == "review":
        execution = read_json(plan_dir / "execution.json")
        diff_summary = collect_git_diff_summary(project_dir)
        return textwrap.dedent(
            f"""
            Review the implementation against the success criteria.

            Project directory:
            {project_dir}

            Approved plan:
            {latest_plan}

            Plan metadata:
            {json_dump(latest_meta).strip()}

            Execution summary:
            {json_dump(execution).strip()}

            Git diff summary:
            {diff_summary}

            Requirements:
            - Be critical.
            - Verify each success criterion explicitly.
            - Call out any concrete gaps or regressions in issues.
            """
        ).strip()

    raise CliError("unsupported_step", f"Unsupported Codex step '{step}'")


def collect_git_diff_summary(project_dir: Path) -> str:
    if not (project_dir / ".git").exists():
        return "Project directory is not a git repository."
    diff = run_command(["git", "status", "--short"], cwd=project_dir)
    if diff.returncode != 0:
        return f"Unable to read git status: {diff.stderr.strip() or diff.stdout.strip()}"
    return diff.stdout.strip() or "No git changes detected."


def mock_worker_output(step: str, state: dict[str, Any], plan_dir: Path) -> WorkerResult:
    iteration = state["iteration"] or 1
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
        target = Path(state["config"]["project_dir"]) / "IMPLEMENTED_BY_DOUBLEDIP.txt"
        target.write_text("mock execution completed\n", encoding="utf-8")
        payload = {
            "output": "Mock execution completed successfully.",
            "files_changed": [str(target.relative_to(Path(state["config"]["project_dir"])))],
            "commands_run": ["mock-write IMPLEMENTED_BY_DOUBLEDIP.txt"],
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

    raise CliError("unsupported_step", f"Mock worker does not support '{step}'")


def run_claude_step(step: str, state: dict[str, Any], plan_dir: Path, *, root: Path, fresh: bool) -> WorkerResult:
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir)
    project_dir = Path(state["config"]["project_dir"])
    schema_name = {
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
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir)
    project_dir = Path(state["config"]["project_dir"])
    schema_file = schemas_root(root) / {
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


def session_key_for(step: str, agent: str) -> str:
    if step in {"plan", "integrate"}:
        return f"{agent}_planner"
    if step == "critique":
        return f"{agent}_critic"
    if step == "execute":
        return f"{agent}_executor"
    if step == "review":
        return f"{agent}_reviewer"
    return f"{agent}_{step}"


def persist_session(state: dict[str, Any], step: str, agent: str, session_id: str | None, *, mode: str, refreshed: bool) -> None:
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
            existing["severity"] = "significant" if normalized["severity_hint"] == "likely-significant" else "minor" if normalized["severity_hint"] == "likely-minor" else "significant"
            existing["raised_in"] = f"critique_v{iteration}.json"
        else:
            severity = "significant" if normalized["severity_hint"] == "likely-significant" else "minor" if normalized["severity_hint"] == "likely-minor" else "significant"
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


def compute_recurring_critiques(plan_dir: Path, iteration: int) -> list[str]:
    if iteration < 2:
        return []
    previous = read_json(current_iteration_artifact(plan_dir, "critique", iteration - 1))
    current = read_json(current_iteration_artifact(plan_dir, "critique", iteration))
    previous_concerns = {normalize_text(item["concern"]) for item in previous.get("flags", [])}
    current_concerns = {normalize_text(item["concern"]) for item in current.get("flags", [])}
    return sorted(previous_concerns.intersection(current_concerns))


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
        {
            "step": step,
            "timestamp": now_utc(),
            "duration_ms": duration_ms,
            "cost_usd": 0.0,
            "result": "error",
            "raw_output_file": raw_name,
            "message": error.message,
        },
    )
    save_state(plan_dir, state)


def resolve_agent_mode(step: str, args: argparse.Namespace) -> tuple[str, str, bool]:
    """Returns (agent, mode, refreshed).

    Both agents default to persistent sessions.  Use --fresh to start a new
    persistent session (break continuity) or --ephemeral for a truly one-off
    call with no session saved.
    """
    default_agent = "claude" if step in {"plan", "integrate"} else "codex"
    agent = args.agent or default_agent
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


def handle_init(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_layout(root)
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise CliError("invalid_project_dir", f"Project directory does not exist: {project_dir}")
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
        {
            "step": "init",
            "timestamp": now_utc(),
            "duration_ms": 0,
            "cost_usd": 0.0,
            "result": "success",
            "environment": {
                "claude": bool(ensure_command("claude")),
                "codex": bool(ensure_command("codex")),
            },
        },
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "init",
        "plan": plan_name,
        "state": STATE_INITIALIZED,
        "summary": f"Initialized plan '{plan_name}' for project {project_dir}",
        "artifacts": ["state.json"],
        "next_step": "plan",
    }


def handle_plan(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "plan", {STATE_INITIALIZED})
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
    state.setdefault("plan_versions", []).append({"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]})
    persist_session(state, "plan", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        {
            "step": "plan",
            "timestamp": now_utc(),
            "duration_ms": worker.duration_ms,
            "cost_usd": worker.cost_usd,
            "result": "success",
            "output_file": plan_filename,
            "artifact_hash": meta["hash"],
            "session_mode": mode,
            "session_id": worker.session_id,
            "agent": agent,
        },
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "plan",
        "iteration": version,
        "summary": f"Generated plan v{version} with {len(payload['questions'])} questions and {len(payload['success_criteria'])} success criteria.",
        "artifacts": [plan_filename, meta_filename],
        "next_step": "critique",
        "state": STATE_PLANNED,
    }


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
        {
            "step": "critique",
            "timestamp": now_utc(),
            "duration_ms": worker.duration_ms,
            "cost_usd": worker.cost_usd,
            "result": "success",
            "output_file": critique_filename,
            "artifact_hash": sha256_file(plan_dir / critique_filename),
            "flags_count": len(worker.payload.get("flags", [])),
            "session_mode": mode,
            "session_id": worker.session_id,
            "agent": agent,
        },
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "critique",
        "iteration": iteration,
        "summary": f"Recorded {len(worker.payload.get('flags', []))} critique flags.",
        "artifacts": [critique_filename, "flags.json"],
        "next_step": "evaluate",
        "state": STATE_CRITIQUED,
        "verified_flags": worker.payload.get("verified_flag_ids", []),
        "open_flags": [flag["id"] for flag in registry.get("flags", []) if flag.get("status") == "open"],
    }


def flag_weight(flag: dict[str, Any]) -> float:
    """Weight a flag for evaluation scoring. Higher = more blocking."""
    category = flag.get("category", "other")
    concern = flag.get("concern", "").lower()

    if category == "security":
        return 3.0

    implementation_detail_signals = [
        "column", "schema", "field", "as written",
        "pseudocode", "seed sql", "placeholder",
    ]
    if any(signal in concern for signal in implementation_detail_signals):
        return 0.5

    weights = {
        "correctness": 2.0,
        "completeness": 1.5,
        "performance": 1.0,
        "maintainability": 0.75,
        "other": 1.0,
    }
    return weights.get(category, 1.0)


def build_evaluation(plan_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    iteration = state["iteration"]
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    significant_count = len([flag for flag in flag_registry.get("flags", []) if flag.get("severity") == "significant" and flag.get("status") != "verified"])
    weighted_score = round(sum(flag_weight(f) for f in unresolved), 2)
    weighted_history = state.get("meta", {}).get("weighted_scores", [])
    latest_plan_text = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    previous_text = None
    if iteration > 1:
        previous_text = (plan_dir / f"plan_v{iteration - 1}.md").read_text(encoding="utf-8")
    plan_delta = compute_plan_delta_percent(previous_text, latest_plan_text)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    budget = float(state["config"].get("budget_usd", 25.0))
    total_cost = float(state.get("meta", {}).get("total_cost_usd", 0.0))
    recommendation = "CONTINUE"
    confidence = "medium"
    rationale = "Continue refining the plan."

    if total_cost > budget:
        recommendation = "ABORT"
        confidence = "high"
        rationale = f"Cost ${total_cost:.3f} exceeded configured budget ${budget:.3f}."
    elif significant_count == 0 and not unresolved:
        recommendation = "SKIP"
        confidence = "high"
        rationale = "No unresolved significant flags remain."
    elif plan_delta is not None and plan_delta < 5.0:
        if unresolved:
            recommendation = "ESCALATE"
            confidence = "high"
            rationale = "Plan stagnated with unresolved significant risks."
        else:
            recommendation = "SKIP"
            confidence = "high"
            rationale = "Plan changes are small and all significant risks appear addressed."
    elif iteration == 1 and significant_count > 0:
        recommendation = "CONTINUE"
        confidence = "high"
        rationale = f"First iteration still has {significant_count} significant flags."
    elif recurring:
        recommendation = "ESCALATE"
        confidence = "high"
        rationale = "The same critique concerns repeated across iterations."
    elif len(weighted_history) >= 1 and weighted_score >= weighted_history[-1] * 0.9:
        recommendation = "ESCALATE"
        confidence = "medium"
        rationale = "Weighted flag score is not improving."
    elif len(weighted_history) >= 1 and weighted_score < weighted_history[-1] * 0.9:
        recommendation = "CONTINUE"
        confidence = "medium"
        rationale = "Weighted flag score is trending down."
    elif iteration >= int(state["config"].get("max_iterations", 3)) and unresolved:
        recommendation = "ESCALATE"
        confidence = "high"
        rationale = "Reached max iterations with unresolved significant risks."

    valid_next = ["integrate"] if recommendation == "CONTINUE" else ["gate"] if recommendation == "SKIP" else ["override add-note", "override force-proceed", "override abort"]

    result: dict[str, Any] = {
        "recommendation": recommendation,
        "confidence": confidence,
        "signals": {
            "iteration": iteration,
            "max_iterations": state["config"].get("max_iterations"),
            "significant_flags": significant_count,
            "weighted_score": weighted_score,
            "weighted_history": weighted_history,
            "plan_delta_from_previous": plan_delta,
            "recurring_critiques": recurring,
            "cost_so_far_usd": total_cost,
        },
        "rationale": rationale,
        "valid_next_steps": valid_next,
    }

    if recommendation in ("ESCALATE", "ABORT"):
        if recommendation == "ABORT":
            result["suggested_override"] = "abort"
            result["override_rationale"] = "Budget exceeded. Abort or increase budget."
        elif all(flag_weight(f) <= 1.0 for f in unresolved):
            result["suggested_override"] = "force-proceed"
            result["override_rationale"] = (
                "Remaining flags are implementation details (pseudocode accuracy, "
                "schema column names) that the executor will resolve by reading "
                "the actual code. Safe to proceed."
            )
        elif len(weighted_history) >= 1 and weighted_score > weighted_history[-1] * 1.5:
            result["suggested_override"] = "abort"
            result["override_rationale"] = "Weighted flag score is increasing — the plan may be fundamentally misaligned."
        else:
            result["suggested_override"] = "add-note"
            result["override_rationale"] = (
                "Significant flags remain. Add context to help the next iteration, "
                "or force-proceed if you believe the executor can handle them."
            )

    return result


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
        {
            "step": "evaluate",
            "timestamp": now_utc(),
            "duration_ms": 0,
            "cost_usd": 0.0,
            "result": "success",
            "output_file": filename,
            "artifact_hash": sha256_file(plan_dir / filename),
            "recommendation": evaluation["recommendation"],
        },
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
    state.setdefault("plan_versions", []).append({"version": version, "file": plan_filename, "hash": meta["hash"], "timestamp": meta["timestamp"]})
    state.setdefault("meta", {}).setdefault("plan_deltas", []).append(delta)
    update_flags_after_integrate(plan_dir, payload["flags_addressed"], plan_file=plan_filename, summary=payload["changes_summary"])
    persist_session(state, "integrate", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        {
            "step": "integrate",
            "timestamp": now_utc(),
            "duration_ms": worker.duration_ms,
            "cost_usd": worker.cost_usd,
            "result": "success",
            "output_file": plan_filename,
            "artifact_hash": meta["hash"],
            "session_mode": mode,
            "session_id": worker.session_id,
            "agent": agent,
            "flags_addressed": payload["flags_addressed"],
        },
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "integrate",
        "iteration": version,
        "summary": f"Updated plan to v{version}; addressed {len(payload['flags_addressed'])} flags.",
        "artifacts": [plan_filename, meta_filename, "flags.json"],
        "next_step": "critique",
        "state": STATE_PLANNED,
    }


def run_gate_checks(plan_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    project_dir = Path(state["config"]["project_dir"])
    meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    checks = {
        "project_dir_exists": project_dir.exists(),
        "project_dir_writable": os.access(project_dir, os.W_OK),
        "success_criteria_present": bool(meta.get("success_criteria")),
        "claude_available": bool(ensure_command("claude")),
        "codex_available": bool(ensure_command("codex")),
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
            {
                "step": "gate",
                "timestamp": now_utc(),
                "duration_ms": 0,
                "cost_usd": 0.0,
                "result": "blocked",
                "output_file": "gate.json",
                "artifact_hash": sha256_file(plan_dir / "gate.json"),
            },
        )
        save_state(plan_dir, state)
        return {
            "success": False,
            "step": "gate",
            "summary": "Gate blocked: unresolved flags or failed preflight checks remain.",
            "artifacts": ["gate.json"],
            "next_step": "integrate" if recommendation == "CONTINUE" else "override force-proceed",
            "state": state["current_state"],
            **gate,
        }
    final_plan = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    atomic_write_text(plan_dir / "plan_final.md", final_plan)
    state["current_state"] = STATE_GATED
    append_history(
        state,
        {
            "step": "gate",
            "timestamp": now_utc(),
            "duration_ms": 0,
            "cost_usd": 0.0,
            "result": "success",
            "output_file": "gate.json",
            "artifact_hash": sha256_file(plan_dir / "gate.json"),
        },
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "gate",
        "summary": "Gate passed. Plan is ready for execution.",
        "artifacts": ["gate.json", "plan_final.md"],
        "next_step": "execute",
        "state": STATE_GATED,
        **gate,
    }


def handle_execute(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    require_state(state, "execute", {STATE_GATED})
    if not args.confirm_destructive:
        raise CliError("missing_confirmation", "Execute requires --confirm-destructive")
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
    append_history(
        state,
        {
            "step": "execute",
            "timestamp": now_utc(),
            "duration_ms": worker.duration_ms,
            "cost_usd": worker.cost_usd,
            "result": "success",
            "output_file": "execution.json",
            "artifact_hash": sha256_file(plan_dir / "execution.json"),
            "session_mode": mode,
            "session_id": worker.session_id,
            "agent": agent,
        },
    )
    save_state(plan_dir, state)
    artifacts = ["execution.json"]
    if worker.trace_output is not None:
        artifacts.append("execution_trace.jsonl")
    return {
        "success": True,
        "step": "execute",
        "summary": worker.payload["output"],
        "artifacts": artifacts,
        "next_step": "review",
        "state": STATE_EXECUTED,
        "files_changed": worker.payload.get("files_changed", []),
        "deviations": worker.payload.get("deviations", []),
    }


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
    atomic_write_text(plan_dir / "plan_final.md", final_plan)
    passed = sum(1 for criterion in worker.payload.get("criteria", []) if criterion.get("pass"))
    total = len(worker.payload.get("criteria", []))
    state["current_state"] = STATE_DONE
    persist_session(state, "review", agent, worker.session_id, mode=mode, refreshed=refreshed)
    append_history(
        state,
        {
            "step": "review",
            "timestamp": now_utc(),
            "duration_ms": worker.duration_ms,
            "cost_usd": worker.cost_usd,
            "result": "success",
            "output_file": "review.json",
            "artifact_hash": sha256_file(plan_dir / "review.json"),
            "session_mode": mode,
            "session_id": worker.session_id,
            "agent": agent,
        },
    )
    save_state(plan_dir, state)
    return {
        "success": True,
        "step": "review",
        "summary": f"Review complete: {passed}/{total} success criteria passed.",
        "artifacts": ["review.json", "plan_final.md"],
        "next_step": None,
        "state": STATE_DONE,
        "issues": worker.payload.get("issues", []),
    }


def handle_status(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    plan_dir, state = load_plan(root, args.plan)
    return {
        "success": True,
        "step": "status",
        "plan": state["name"],
        "state": state["current_state"],
        "iteration": state["iteration"],
        "summary": f"Plan '{state['name']}' is currently in state '{state['current_state']}'.",
        "next_step": infer_next_steps(state)[0] if infer_next_steps(state) else None,
        "valid_next": infer_next_steps(state),
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
        items.append(
            {
                "name": state["name"],
                "idea": state["idea"],
                "state": state["current_state"],
                "iteration": state["iteration"],
                "next_step": infer_next_steps(state)[0] if infer_next_steps(state) else None,
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
            atomic_write_text(plan_dir / "plan_final.md", final_plan)
            state["current_state"] = STATE_GATED
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Double-dip orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--project-dir", required=True)
    init_parser.add_argument("--name")
    init_parser.add_argument("--max-iterations", type=int, default=3)
    init_parser.add_argument("--budget-usd", type=float, default=25.0)
    init_parser.add_argument("idea")

    subparsers.add_parser("list")

    for name in ["status", "audit", "evaluate", "gate"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")

    for name in ["plan", "critique", "integrate", "execute", "review"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")
        step_parser.add_argument("--agent", choices=["claude", "codex"])
        step_parser.add_argument("--fresh", action="store_true")
        step_parser.add_argument("--persist", action="store_true")
        step_parser.add_argument("--ephemeral", action="store_true")
        if name == "execute":
            step_parser.add_argument("--confirm-destructive", action="store_true")
        if name == "review":
            step_parser.add_argument("--confirm-self-review", action="store_true")

    override_parser = subparsers.add_parser("override")
    override_parser.add_argument("override_action", choices=["skip", "abort", "force-proceed", "add-note"])
    override_parser.add_argument("--plan")
    override_parser.add_argument("--reason", default="")
    override_parser.add_argument("note", nargs="?")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path.cwd()
    ensure_runtime_layout(root)
    try:
        if args.command == "init":
            response = handle_init(root, args)
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
