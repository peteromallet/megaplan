"""Parallel Hermes critique runner."""

from __future__ import annotations

import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from megaplan._core import get_effective, read_json, schemas_root
from megaplan.hermes_worker import _toolsets_for_phase, clean_parsed_payload, parse_agent_output
from megaplan.prompts.critique import single_check_critique_prompt, write_single_check_template
from megaplan.types import CliError, PlanState
from megaplan.workers import STEP_SCHEMA_FILENAMES, WorkerResult


from megaplan.key_pool import (
    _load_hermes_env,
    _get_api_credential,
    resolve_model as _resolve_model,
    acquire_key,
    report_429,
)


def _merge_unique(groups: list[list[str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def _run_check(
    index: int,
    check: dict[str, Any],
    *,
    state: PlanState,
    plan_dir: Path,
    root: Path,
    model: str | None,
    schema: dict[str, Any],
    project_dir: Path,
) -> tuple[int, dict[str, Any], list[str], list[str], float]:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    output_path = write_single_check_template(plan_dir, state, check, f"critique_check_{check['id']}.json")
    prompt = single_check_critique_prompt(state, plan_dir, root, check, output_path)
    resolved_model, agent_kwargs = _resolve_model(model)

    _model_lower = (resolved_model or "").lower()
    _reasoning_families = ("qwen/qwen3", "deepseek/deepseek-r1")
    _reasoning_off = (
        {"enabled": False}
        if any(_model_lower.startswith(prefix) for prefix in _reasoning_families)
        else None
    )

    def _make_agent(m: str, kw: dict) -> "AIAgent":
        a = AIAgent(
            model=m,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=_toolsets_for_phase("critique"),
            session_id=str(uuid.uuid4()),
            session_db=SessionDB(),
            max_tokens=8192,
            reasoning_config=_reasoning_off,
            **kw,
        )
        a._print_fn = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)
        return a

    def _failure_reason(exc: Exception) -> str:
        if isinstance(exc, CliError):
            return exc.message
        return str(exc) or exc.__class__.__name__

    def _run_attempt(current_agent, current_output_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], float]:
        current_result = current_agent.run_conversation(user_message=prompt)
        payload, raw_output = parse_agent_output(
            current_agent,
            current_result,
            output_path=current_output_path,
            schema=schema,
            step="critique",
            project_dir=project_dir,
            plan_dir=plan_dir,
        )
        clean_parsed_payload(payload, schema, "critique")
        payload_checks = payload.get("checks")
        if not isinstance(payload_checks, list) or len(payload_checks) != 1 or not isinstance(payload_checks[0], dict):
            raise CliError(
                "worker_parse_error",
                f"Parallel critique output for check '{check['id']}' did not contain exactly one check",
                extra={"raw_output": raw_output},
            )
        verified = payload.get("verified_flag_ids", [])
        disputed = payload.get("disputed_flag_ids", [])
        return (
            current_result,
            payload_checks[0],
            verified if isinstance(verified, list) else [],
            disputed if isinstance(disputed, list) else [],
            float(current_result.get("estimated_cost_usd", 0.0) or 0.0),
            int(current_result.get("prompt_tokens", 0) or 0),
            int(current_result.get("completion_tokens", 0) or 0),
            int(current_result.get("total_tokens", 0) or 0),
        )

    agent = _make_agent(resolved_model, agent_kwargs)
    try:
        _result, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = _run_attempt(agent, output_path)
    except Exception as exc:
        # Report 429 to key pool so it cools down this key
        exc_str = str(exc)
        if "429" in exc_str:
            if model and model.startswith("minimax:"):
                report_429("minimax", agent_kwargs.get("api_key", ""), cooldown_secs=60)
            elif model and model.startswith("zhipu:"):
                cooldown = 3600 if "Limit Exhausted" in exc_str else 120
                report_429("zhipu", agent_kwargs.get("api_key", ""), cooldown_secs=cooldown)
        # Fallback to OpenRouter if primary MiniMax API fails (429, timeout, etc.)
        if model and model.startswith("minimax:"):
            or_key = acquire_key("openrouter")
            if or_key:
                from megaplan.key_pool import minimax_openrouter_model
                fallback_model = minimax_openrouter_model(model[len("minimax:"):])
                fallback_kwargs = {"base_url": "https://openrouter.ai/api/v1", "api_key": or_key}
                if isinstance(exc, CliError):
                    print(
                        f"[parallel-critique] MiniMax returned bad content ({_failure_reason(exc)}), falling back to OpenRouter",
                        file=sys.stderr,
                    )
                else:
                    print(f"[parallel-critique] Primary MiniMax failed ({exc}), falling back to OpenRouter", file=sys.stderr)
                # Re-write template since the previous agent may have corrupted it
                output_path = write_single_check_template(plan_dir, state, check, f"critique_check_{check['id']}.json")
                agent = _make_agent(fallback_model, fallback_kwargs)
                try:
                    _result, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = _run_attempt(agent, output_path)
                except Exception as fallback_exc:
                    raise CliError(
                        "worker_error",
                        (
                            f"Parallel critique failed for check '{check['id']}' "
                            f"(both MiniMax and OpenRouter): primary={_failure_reason(exc)}; "
                            f"fallback={_failure_reason(fallback_exc)}"
                        ),
                        extra={"check_id": check["id"]},
                    ) from fallback_exc
            else:
                raise
        else:
            raise
    return (
        index,
        check_payload,
        verified_ids,
        disputed_ids,
        cost_usd,
        pt,
        ct,
        tt,
    )


def run_parallel_critique(
    state: PlanState,
    plan_dir: Path,
    *,
    root: Path,
    model: str | None,
    checks: tuple[dict[str, Any], ...],
    max_concurrent: int | None = None,
) -> WorkerResult:
    started = time.monotonic()
    if not checks:
        return WorkerResult(
            payload={"checks": [], "flags": [], "verified_flag_ids": [], "disputed_flag_ids": []},
            raw_output="parallel",
            duration_ms=0,
            cost_usd=0.0,
            session_id=None,
        )

    schema = read_json(schemas_root(root) / STEP_SCHEMA_FILENAMES["critique"])
    project_dir = Path(state["config"]["project_dir"])
    results: list[tuple[dict[str, Any], list[str], list[str]] | None] = [None] * len(checks)
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        concurrency = min(max_concurrent or get_effective("orchestration", "max_critique_concurrency"), len(checks))
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _run_check,
                    index,
                    check,
                    state=state,
                    plan_dir=plan_dir,
                    root=root,
                    model=model,
                    schema=schema,
                    project_dir=project_dir,
                )
                for index, check in enumerate(checks)
            ]
            for future in as_completed(futures):
                index, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = future.result()
                results[index] = (check_payload, verified_ids, disputed_ids)
                total_cost += cost_usd
                total_prompt_tokens += pt
                total_completion_tokens += ct
                total_tokens += tt
    finally:
        sys.stdout = real_stdout

    ordered_checks: list[dict[str, Any]] = []
    verified_groups: list[list[str]] = []
    disputed_groups: list[list[str]] = []
    for item in results:
        if item is None:
            raise CliError("worker_error", "Parallel critique did not return all check results")
        check_payload, verified_ids, disputed_ids = item
        ordered_checks.append(check_payload)
        verified_groups.append(verified_ids)
        disputed_groups.append(disputed_ids)

    disputed_flag_ids = _merge_unique(disputed_groups)
    disputed_set = set(disputed_flag_ids)
    verified_flag_ids = [flag_id for flag_id in _merge_unique(verified_groups) if flag_id not in disputed_set]
    return WorkerResult(
        payload={
            "checks": ordered_checks,
            "flags": [],
            "verified_flag_ids": verified_flag_ids,
            "disputed_flag_ids": disputed_flag_ids,
        },
        raw_output="parallel",
        duration_ms=int((time.monotonic() - started) * 1000),
        cost_usd=total_cost,
        session_id=None,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )
