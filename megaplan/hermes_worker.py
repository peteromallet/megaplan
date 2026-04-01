"""Hermes Agent worker for megaplan — runs phases via AIAgent with OpenRouter."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from megaplan.checks import build_empty_template, checks_for_robustness
from megaplan.types import CliError, MOCK_ENV_VAR, PlanState
from megaplan.workers import (
    STEP_SCHEMA_FILENAMES,
    WorkerResult,
    mock_worker_output,
    session_key_for,
    validate_payload,
)
from megaplan._core import configured_robustness, read_json, schemas_root
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

    Execute phase gets full terminal + file + web access.
    Plan, critique, and revise get file + web (verify APIs against docs).
    Gate, finalize, review get file only (judgment, not research).
    """
    if phase == "execute":
        return ["terminal", "file", "web"]
    if phase in ("plan", "prep", "critique", "revise", "research"):
        return ["file", "web"]
    return ["file"]


_TEMPLATE_FILE_PHASES = {"critique", "finalize", "review", "prep"}


def _build_output_template(step: str, schema: dict, state: PlanState) -> str:
    if step != "critique":
        return _schema_template(schema)

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    critique_checks = build_empty_template(
        checks_for_robustness(configured_robustness(state))
    )
    template: dict[str, object] = {}
    for key in required:
        if key == "checks":
            template[key] = critique_checks
            continue
        prop = properties.get(key, {})
        ptype = prop.get("type", "string") if isinstance(prop, dict) else "string"
        if ptype == "array":
            template[key] = []
        elif ptype == "object":
            template[key] = {}
        elif ptype == "boolean":
            template[key] = False
        elif ptype in ("number", "integer"):
            template[key] = 0
        else:
            template[key] = ""
    return json.dumps(template, indent=2)


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
    schema_name = STEP_SCHEMA_FILENAMES[step]
    schema = read_json(schemas_root(root) / schema_name)
    output_path: Path | None = None

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

    # Build prompt — megaplan prompts embed the JSON schema, but some models
    # ignore formatting instructions buried in long prompts.  Append a clear
    # reminder so the final response is valid JSON, not markdown.
    prompt = prompt_override if prompt_override is not None else create_hermes_prompt(
        step, state, plan_dir, root=root
    )
    # Add web search guidance for phases that have it
    if step in ("plan", "critique", "revise"):
        prompt += (
            "\n\nWEB SEARCH: You have web_search and web_extract tools. "
            "If the task involves a framework API you're not certain about — "
            "for example a specific Next.js feature, a particular import path, "
            "or a config flag that might have changed between versions — "
            "search for the current documentation before committing to an approach. "
            "Your training data may be outdated for newer framework features."
        )
    elif step == "execute":
        prompt += (
            "\n\nWEB SEARCH: You have web_search available. "
            "If you encounter an API you're not sure about while coding, "
            "search before writing — a quick lookup is cheaper than a build failure."
            "\n\nIMPORTANT: Do NOT rename, modify, or delete EVAL.ts or any test files. "
            "They are used for scoring after execution and must remain unchanged."
        )

    if step in _TEMPLATE_FILE_PHASES:
        output_path = plan_dir / f"{step}_output.json"
        output_path.write_text(_build_output_template(step, schema, state), encoding="utf-8")
        prompt += (
            f"\n\nIMPORTANT: A template file has been written to {output_path}. "
            "Fill in this file with your findings using the file write tool as you work. "
            "Update the file after each major finding. When done, ensure the file contains "
            "your complete output."
        )
    else:
        template = _schema_template(schema)
        prompt += (
            "\n\nIMPORTANT: Your final response MUST be a single valid JSON object. "
            "Do NOT use markdown. Do NOT wrap in code fences. Output ONLY raw JSON "
            "matching this template:\n\n" + template
        )

    # Redirect stdout to stderr before creating the agent.  Hermes's
    # KawaiiSpinner captures sys.stdout at __init__ time and writes
    # directly to it (bypassing _print_fn).  By swapping stdout to
    # stderr here, all spinner/progress output flows to stderr while
    # megaplan keeps stdout clean for structured JSON results.
    import sys
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Resolve model provider — support direct API providers via prefix
    # e.g. "zhipu:glm-5.1" → base_url=Zhipu API, model="glm-5.1"
    agent_kwargs: dict = {}
    resolved_model = model or "anthropic/claude-opus-4.6"
    if resolved_model.startswith("zhipu:"):
        resolved_model = resolved_model[len("zhipu:"):]
        agent_kwargs["base_url"] = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        agent_kwargs["api_key"] = os.environ.get("ZHIPU_API_KEY", "")
    elif resolved_model.startswith("minimax:"):
        resolved_model = resolved_model[len("minimax:"):]
        agent_kwargs["base_url"] = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        agent_kwargs["api_key"] = os.environ.get("MINIMAX_API_KEY", "")

    # Instantiate AIAgent
    agent = AIAgent(
        model=resolved_model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=_toolsets_for_phase(step),
        session_id=session_id,
        session_db=SessionDB(),
        # Cap output tokens to prevent repetition loops (Qwen generates 330K+
        # of repeated text without a limit). 8192 is plenty for any megaplan
        # phase response. Execute gets more since it may include verbose output.
        max_tokens=16384 if step == "execute" else 8192,
        **agent_kwargs,
    )
    agent._print_fn = lambda *a, **kw: print(*a, **kw, file=sys.stderr)
    # Don't set response_format when tools are enabled — many models
    # (Qwen, GLM-5) hang or produce garbage when both are active.
    # The JSON template in the prompt is sufficient; _parse_json_response
    # handles code fences and markdown wrapping.
    toolsets = _toolsets_for_phase(step)
    if not toolsets:
        agent.set_response_format(schema, name=f"megaplan_{step}")

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
    finally:
        sys.stdout = real_stdout
    elapsed_ms = int((time.monotonic() - started) * 1000)

    # Parse structured output from the response.
    # Try: final_response → reasoning field → assistant content → summary call.
    raw_output = result.get("final_response", "") or ""
    messages = result.get("messages", [])

    # If final_response is empty and the model used tools, the agent loop exited
    # after tool calls without giving the model a chance to output JSON.
    # Make one more API call with the template to force structured output.
    if not raw_output.strip() and messages and any(m.get("tool_calls") for m in messages if m.get("role") == "assistant"):
        try:
            template = _schema_template(schema)
            summary_prompt = (
                "You have finished investigating. Now fill in this JSON template with your findings "
                "and output it as your response. Output ONLY the raw JSON, nothing else.\n\n"
                + template
            )
            summary_result = agent.run_conversation(
                user_message=summary_prompt,
                conversation_history=messages,
            )
            raw_output = summary_result.get("final_response", "") or ""
            messages = summary_result.get("messages", messages)
            if raw_output.strip():
                print(f"[hermes-worker] Got JSON from template prompt ({len(raw_output)} chars)", file=sys.stderr)
        except Exception as exc:
            print(f"[hermes-worker] Template prompt failed: {exc}", file=sys.stderr)

    # For template-file phases, check the template file FIRST — we told the
    # model to write there, so it's the primary output path.
    payload = None
    if output_path and output_path.exists():
        try:
            candidate_payload = json.loads(output_path.read_text(encoding="utf-8"))
            # Only use if the model actually filled something in (not just the empty template)
            if isinstance(candidate_payload, dict):
                has_content = any(
                    v for k, v in candidate_payload.items()
                    if isinstance(v, list) and v  # non-empty arrays
                    or isinstance(v, str) and v.strip()  # non-empty strings
                )
                if has_content:
                    payload = candidate_payload
                    print(f"[hermes-worker] Read JSON from template file: {output_path}", file=sys.stderr)
        except (json.JSONDecodeError, OSError):
            pass

    # Try parsing the final text response
    if payload is None:
        payload = _parse_json_response(raw_output)

    # Fallback: some models (GLM-5) put JSON in reasoning/think tags
    # instead of content. Just grab it from there.
    if payload is None and messages:
        payload = _extract_json_from_reasoning(messages)
        if payload is not None:
            print(f"[hermes-worker] Extracted JSON from reasoning tags", file=sys.stderr)

    # Fallback: check all assistant message content fields (not just final_response)
    # The model may have output JSON in an earlier message before making more tool calls
    if payload is None and messages:
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                payload = _parse_json_response(content)
                if payload is not None:
                    print(f"[hermes-worker] Extracted JSON from assistant message content", file=sys.stderr)
                    break

    # Fallback: for execute phase, reconstruct from tool calls + git diff
    if payload is None and step == "execute":
        payload = _reconstruct_execute_payload(messages, project_dir, plan_dir)
        if payload is not None:
            print(f"[hermes-worker] Reconstructed execute payload from tool calls", file=sys.stderr)

    # Fallback: the model may have written the JSON to a different file location
    if payload is None:
        schema_filename = STEP_SCHEMA_FILENAMES.get(step, f"{step}.json")
        for candidate in [
            plan_dir / f"{step}_output.json",  # template file path
            project_dir / schema_filename,
            plan_dir / schema_filename,
            project_dir / f"{step}.json",
        ]:
            if candidate.exists() and candidate != output_path:  # skip if already checked
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    print(f"[hermes-worker] Read JSON from file written by model: {candidate}", file=sys.stderr)
                    break
                except (json.JSONDecodeError, OSError):
                    pass

    if payload is None:
        raise CliError(
            "worker_parse_error",
            f"Hermes worker returned invalid JSON for step '{step}': "
            f"could not extract JSON from response ({len(raw_output)} chars)",
            extra={"raw_output": raw_output},
        )

    # Fill in missing required fields with safe defaults before validation.
    # Models often omit empty arrays/strings that megaplan requires.
    _fill_schema_defaults(payload, schema)

    # Normalize field aliases in nested arrays (e.g. critique flags use
    # "summary" instead of "concern", "detail" instead of "evidence").
    _normalize_nested_aliases(payload, schema)

    try:
        validate_payload(step, payload)
    except CliError:
        # For execute, try reconstructed payload if validation fails
        if step == "execute":
            reconstructed = _reconstruct_execute_payload(messages, project_dir, plan_dir)
            if reconstructed is not None:
                try:
                    validate_payload(step, reconstructed)
                    payload = reconstructed
                    print(f"[hermes-worker] Using reconstructed payload (original failed validation)", file=sys.stderr)
                    error = None
                except CliError:
                    pass
        if error is not None:
            raise CliError(error.code, error.message, extra={"raw_output": raw_output}) from error

    cost_usd = result.get("estimated_cost_usd", 0.0) or 0.0

    return WorkerResult(
        payload=payload,
        raw_output=raw_output,
        duration_ms=elapsed_ms,
        cost_usd=float(cost_usd),
        session_id=session_id,
    )


def _extract_json_from_reasoning(messages: list) -> dict | None:
    """Extract JSON from the last assistant message's reasoning field.

    Some models (GLM-5) wrap their entire response in think/reasoning tags,
    so the content field is empty but reasoning contains valid JSON.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for field in ("reasoning", "reasoning_content"):
            text = msg.get(field)
            if isinstance(text, str) and text.strip():
                result = _parse_json_response(text)
                if result is not None:
                    return result
        # Also check reasoning_details (list of dicts with "content")
        details = msg.get("reasoning_details")
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict):
                    text = item.get("content", "")
                    if isinstance(text, str) and text.strip():
                        result = _parse_json_response(text)
                        if result is not None:
                            return result
    return None


def _reconstruct_execute_payload(
    messages: list,
    project_dir: Path,
    plan_dir: Path,
) -> dict | None:
    """Reconstruct an execute phase response from tool calls and git state.

    When the model did the work via tools but couldn't produce the JSON
    report (e.g., response trapped in think tags, or timeout), build the
    response from what actually happened.
    """
    import subprocess

    # Collect tool calls from messages
    tool_calls = []
    files_changed = set()
    commands_run = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}

            tool_calls.append({"name": name, "args": args})

            if name in ("write_file", "patch", "edit_file", "apply_patch"):
                path = args.get("path", "")
                if isinstance(path, str) and path:
                    # Make relative to project dir
                    try:
                        rel = str(Path(path).relative_to(project_dir))
                    except ValueError:
                        rel = path
                    files_changed.add(rel)
            elif name in ("terminal", "shell"):
                cmd = args.get("command", "")
                if isinstance(cmd, str) and cmd:
                    commands_run.append(cmd)

    # Also check git diff for files changed
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=project_dir,
            capture_output=True, text=True, timeout=10, check=False,
        )
        if diff_result.returncode == 0:
            for line in diff_result.stdout.splitlines():
                if line.strip():
                    files_changed.add(line.strip())
    except Exception:
        pass

    # Check for untracked files too
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_dir,
            capture_output=True, text=True, timeout=10, check=False,
        )
        if status_result.returncode == 0:
            for line in status_result.stdout.splitlines():
                if line.startswith("?? ") or line.startswith("A  ") or line.startswith("M  "):
                    fname = line[3:].strip()
                    if fname and not fname.startswith(".megaplan/"):
                        files_changed.add(fname)
    except Exception:
        pass

    if not tool_calls and not files_changed:
        return None  # Nothing happened, can't reconstruct

    # Try to read checkpoint file for task updates
    task_updates = []
    checkpoint_files = sorted(plan_dir.glob("execution_batch_*.json"), reverse=True)
    for cp_file in checkpoint_files:
        try:
            cp_data = json.loads(cp_file.read_text(encoding="utf-8"))
            updates = cp_data.get("task_updates", [])
            if isinstance(updates, list):
                task_updates.extend(updates)
        except Exception:
            pass

    files_list = sorted(files_changed)
    return {
        "output": f"[Reconstructed from tool calls] Made {len(tool_calls)} tool calls, changed {len(files_list)} files.",
        "files_changed": files_list,
        "commands_run": commands_run,
        "deviations": ["Execute response reconstructed from tool calls — model failed to produce JSON report."],
        "task_updates": task_updates,
        "sense_check_acknowledgments": [],
    }


def _fill_schema_defaults(payload: dict, schema: dict) -> None:
    """Fill missing required fields with safe defaults based on schema types.

    Models often omit empty arrays, empty strings, or optional-sounding fields
    that the schema marks as required. Rather than rejecting the response,
    fill them with type-appropriate defaults.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    for field in required:
        if field in payload:
            continue
        prop = properties.get(field, {})
        ptype = prop.get("type", "string")
        if ptype == "array":
            payload[field] = []
        elif ptype == "object":
            payload[field] = {}
        elif ptype == "boolean":
            payload[field] = False
        elif ptype in ("number", "integer"):
            payload[field] = 0
        else:
            payload[field] = ""


def _normalize_nested_aliases(payload: dict, schema: dict) -> None:
    """Normalize field aliases in nested array items.

    Models often use synonyms for required fields (e.g. "summary" instead of
    "concern", "detail" instead of "evidence"). This applies the alias mapping
    from merge._FIELD_ALIASES to nested objects in arrays.
    """
    from megaplan.merge import _FIELD_ALIASES

    properties = schema.get("properties", {})
    for field, prop in properties.items():
        if prop.get("type") != "array" or field not in payload:
            continue
        items_schema = prop.get("items", {})
        if items_schema.get("type") != "object":
            continue
        required = items_schema.get("required", [])
        items = payload[field]
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for req_field in required:
                if req_field in item and item[req_field]:
                    continue  # Already has a non-empty value
                aliases = _FIELD_ALIASES.get(req_field, ())
                for alias in aliases:
                    if alias in item and item[alias]:
                        item[req_field] = item[alias]
                        break


def _schema_template(schema: dict) -> str:
    """Generate a JSON template from a schema showing required keys with placeholder values."""
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return "{}"
    template = {}
    for key, prop in props.items():
        if not isinstance(prop, dict):
            template[key] = "..."
            continue
        ptype = prop.get("type", "string")
        if ptype == "string":
            desc = prop.get("description", "")
            template[key] = f"<{desc}>" if desc else "..."
        elif ptype == "array":
            items = prop.get("items", {})
            if isinstance(items, dict) and items.get("type") == "string":
                template[key] = ["..."]
            else:
                template[key] = []
        elif ptype == "boolean":
            template[key] = True
        elif ptype in ("number", "integer"):
            template[key] = 0
        elif ptype == "object":
            template[key] = {}
        else:
            template[key] = "..."
    return json.dumps(template, indent=2)


def _parse_json_response(text: str) -> dict | None:
    """Extract a JSON object from a model response.

    Tries in order:
    1. Direct JSON parse
    2. Repair common JSON issues (escaped newlines in structural positions)
    3. Extract from ```json ... ``` code block
    4. Find first { ... } JSON object in the text

    Each step also tries the repaired version.
    """
    text = text.strip()
    if not text:
        return None

    for candidate in [text, _repair_json(text)]:
        # Direct parse
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Extract from code block
        import re
        code_block = re.search(r'```(?:json)?\s*\n(.*?)\n```', candidate, re.DOTALL)
        if code_block:
            block_text = code_block.group(1)
            for block_candidate in [block_text, _repair_json(block_text)]:
                try:
                    parsed = json.loads(block_candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass

        # Find first JSON object
        decoder = json.JSONDecoder()
        for i, ch in enumerate(candidate):
            if ch != '{':
                continue
            try:
                parsed, end = decoder.raw_decode(candidate[i:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


def _repair_json(text: str) -> str:
    """Fix common JSON issues from LLM output.

    Models sometimes mix escaped and literal newlines, or produce
    backslash-n outside of strings where real whitespace is needed.
    """
    # Replace literal \n that appear outside of JSON strings with actual newlines.
    # This handles the case where the model outputs [\n    "item"] instead of
    # [\n    "item"] — the \n is structural whitespace, not string content.
    result = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue
        if ch == '\\' and in_string:
            escape = True
            result.append(ch)
            i += 1
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        # Outside a string, replace \n with actual newline
        if not in_string and ch == '\\' and i + 1 < len(text) and text[i + 1] == 'n':
            result.append('\n')
            i += 2
            continue
        result.append(ch)
        i += 1
    return ''.join(result)
