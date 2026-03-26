"""Hermes Agent worker for megaplan — runs phases via AIAgent with OpenRouter."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from megaplan.types import CliError, MOCK_ENV_VAR, PlanState
from megaplan.workers import (
    STEP_SCHEMA_FILENAMES,
    WorkerResult,
    mock_worker_output,
    session_key_for,
    validate_payload,
)
from megaplan._core import read_json, schemas_root
from megaplan.prompts import create_hermes_prompt


def check_hermes_available() -> tuple[bool, str]:
    """Check if Hermes Agent is importable and has API credentials."""
    try:
        from run_agent import AIAgent  # noqa: F401
    except ImportError:
        return (False, "hermes-agent not installed. Install with: pip install hermes-agent")

    # Check for API key — Hermes stores keys in ~/.hermes/.env, loaded via dotenv.
    # After dotenv load, the key is available as an env var.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try loading from Hermes .env file directly
        try:
            from hermes_cli.config import get_env_path
            env_path = get_env_path()
            if env_path and env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip("'\"")
                        break
        except (ImportError, Exception):
            pass

    if not api_key:
        return (False, "OPENROUTER_API_KEY not set. Configure via env var or ~/.hermes/.env")
    return (True, "")


def _toolsets_for_phase(phase: str) -> list[str] | None:
    """Return toolsets for a given megaplan phase.

    Execute phase gets full terminal + file access.
    Planning phases get file access (for repo inspection).
    """
    if phase == "execute":
        return ["terminal", "file"]
    return ["file"]


def run_hermes_step(
    step: str,
    state: PlanState,
    plan_dir: Path,
    *,
    root: Path,
    fresh: bool,
    model: str | None = None,
    prompt_override: str | None = None,
) -> WorkerResult:
    """Run a megaplan phase using Hermes Agent via OpenRouter.

    Structured output is enforced via the prompt (megaplan prompts already
    embed the JSON schema). The final response is parsed and validated.
    """
    if os.getenv(MOCK_ENV_VAR) == "1":
        return mock_worker_output(step, state, plan_dir, prompt_override=prompt_override)

    from run_agent import AIAgent
    from hermes_state import SessionDB

    project_dir = Path(state["config"]["project_dir"])

    # Session management
    session_key = session_key_for(step, "hermes", model=model)
    session = state["sessions"].get(session_key, {})
    session_id = session.get("id") if not fresh else None

    # Reload conversation history for session continuity
    conversation_history = None
    if session_id:
        try:
            db = SessionDB()
            conversation_history = db.get_messages_as_conversation(session_id)
        except Exception:
            conversation_history = None

    # Generate new session ID if needed
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    # Build prompt — megaplan prompts already embed the JSON schema
    prompt = prompt_override if prompt_override is not None else create_hermes_prompt(
        step, state, plan_dir, root=root
    )

    # Instantiate AIAgent
    agent = AIAgent(
        model=model or "anthropic/claude-opus-4.6",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=_toolsets_for_phase(step),
        session_id=session_id,
        session_db=SessionDB(),
    )

    # Run
    started = time.monotonic()
    try:
        result = agent.run_conversation(
            user_message=prompt,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        raise CliError(
            "worker_error",
            f"Hermes worker failed for step '{step}': {exc}",
            extra={"session_id": session_id},
        ) from exc
    elapsed_ms = int((time.monotonic() - started) * 1000)

    # Parse structured output from the response
    raw_output = result.get("final_response", "") or ""
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CliError(
            "worker_parse_error",
            f"Hermes worker returned invalid JSON for step '{step}': {exc}",
            extra={"raw_output": raw_output},
        ) from exc

    try:
        validate_payload(step, payload)
    except CliError as error:
        raise CliError(error.code, error.message, extra={"raw_output": raw_output}) from error

    cost_usd = result.get("estimated_cost_usd", 0.0) or 0.0

    return WorkerResult(
        payload=payload,
        raw_output=raw_output,
        duration_ms=elapsed_ms,
        cost_usd=float(cost_usd),
        session_id=session_id,
    )
