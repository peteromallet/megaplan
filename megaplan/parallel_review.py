"""Parallel Hermes review runner.

The preloaded-template-ID convention is the preferred way to get structured
output from focused review agents: write the exact slot shape first, then have
each agent fill that file instead of inventing IDs in free-form JSON.

This module intentionally mirrors `megaplan.parallel_critique` so the two phase
runners remain easy to compare and later extract into a shared utility.
"""

from __future__ import annotations

import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from megaplan._core import get_effective, read_json, schemas_root
from megaplan.hermes_worker import _toolsets_for_phase, clean_parsed_payload, parse_agent_output
from megaplan.prompts.review import (
    _write_criteria_verdict_review_template,
    _write_single_check_review_template,
    heavy_criteria_review_prompt,
    single_check_review_prompt,
)
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


def _clean_review_check_payload(payload: dict[str, Any]) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return
    for check in checks:
        if not isinstance(check, dict):
            continue
        check.pop("guidance", None)
        check.pop("prior_findings", None)


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, CliError):
        return exc.message
    return str(exc) or exc.__class__.__name__


def _run_check(
    index: int,
    check: Any,
    *,
    state: PlanState,
    plan_dir: Path,
    root: Path,
    model: str | None,
    schema: dict[str, Any],
    project_dir: Path,
    pre_check_flags: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], list[str], list[str], float, int, int, int]:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    check_id = check["id"] if isinstance(check, dict) else getattr(check, "id")
    output_path = _write_single_check_review_template(plan_dir, state, check, f"review_check_{check_id}.json")
    prompt = single_check_review_prompt(state, plan_dir, root, check, output_path, pre_check_flags)
    resolved_model, agent_kwargs = _resolve_model(model)

    _model_lower = (resolved_model or "").lower()
    _reasoning_families = ("qwen/qwen3", "deepseek/deepseek-r1")
    _reasoning_off = (
        {"enabled": False}
        if any(_model_lower.startswith(prefix) for prefix in _reasoning_families)
        else None
    )

    def _make_agent(m: str, kw: dict) -> "AIAgent":
        agent = AIAgent(
            model=m,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=_toolsets_for_phase("review"),
            session_id=str(uuid.uuid4()),
            session_db=SessionDB(),
            max_tokens=8192,
            reasoning_config=_reasoning_off,
            **kw,
        )
        agent._print_fn = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)
        return agent

    def _run_attempt(current_agent, current_output_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], float, int, int, int]:
        current_result = current_agent.run_conversation(user_message=prompt)
        payload, raw_output = parse_agent_output(
            current_agent,
            current_result,
            output_path=current_output_path,
            schema=schema,
            step="review",
            project_dir=project_dir,
            plan_dir=plan_dir,
        )
        clean_parsed_payload(payload, schema, "review")
        _clean_review_check_payload(payload)
        payload_checks = payload.get("checks")
        if not isinstance(payload_checks, list) or len(payload_checks) != 1 or not isinstance(payload_checks[0], dict):
            raise CliError(
                "worker_parse_error",
                f"Parallel review output for check '{check_id}' did not contain exactly one check",
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
        exc_str = str(exc)
        if "429" in exc_str:
            if model and model.startswith("minimax:"):
                report_429("minimax", agent_kwargs.get("api_key", ""), cooldown_secs=60)
            elif model and model.startswith("zhipu:"):
                cooldown = 3600 if "Limit Exhausted" in exc_str else 120
                report_429("zhipu", agent_kwargs.get("api_key", ""), cooldown_secs=cooldown)
        if model and model.startswith("minimax:"):
            or_key = acquire_key("openrouter")
            if or_key:
                from megaplan.key_pool import minimax_openrouter_model

                fallback_model = minimax_openrouter_model(model[len("minimax:"):])
                fallback_kwargs = {"base_url": "https://openrouter.ai/api/v1", "api_key": or_key}
                if isinstance(exc, CliError):
                    print(
                        f"[parallel-review] MiniMax returned bad content ({_failure_reason(exc)}), falling back to OpenRouter",
                        file=sys.stderr,
                    )
                else:
                    print(f"[parallel-review] Primary MiniMax failed ({exc}), falling back to OpenRouter", file=sys.stderr)
                output_path = _write_single_check_review_template(plan_dir, state, check, f"review_check_{check_id}.json")
                agent = _make_agent(fallback_model, fallback_kwargs)
                try:
                    _result, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = _run_attempt(agent, output_path)
                except Exception as fallback_exc:
                    raise CliError(
                        "worker_error",
                        (
                            f"Parallel review failed for check '{check_id}' "
                            f"(both MiniMax and OpenRouter): primary={_failure_reason(exc)}; "
                            f"fallback={_failure_reason(fallback_exc)}"
                        ),
                        extra={"check_id": check_id},
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


def _run_criteria_verdict(
    *,
    state: PlanState,
    plan_dir: Path,
    root: Path,
    model: str | None,
    schema: dict[str, Any],
    project_dir: Path,
) -> tuple[dict[str, Any], float, int, int, int]:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    output_path = _write_criteria_verdict_review_template(plan_dir, state, "review_criteria_verdict.json")
    prompt = heavy_criteria_review_prompt(state, plan_dir, root, output_path)
    resolved_model, agent_kwargs = _resolve_model(model)

    _model_lower = (resolved_model or "").lower()
    _reasoning_families = ("qwen/qwen3", "deepseek/deepseek-r1")
    _reasoning_off = (
        {"enabled": False}
        if any(_model_lower.startswith(prefix) for prefix in _reasoning_families)
        else None
    )

    def _make_agent(m: str, kw: dict) -> "AIAgent":
        agent = AIAgent(
            model=m,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=_toolsets_for_phase("review"),
            session_id=str(uuid.uuid4()),
            session_db=SessionDB(),
            max_tokens=8192,
            reasoning_config=_reasoning_off,
            **kw,
        )
        agent._print_fn = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)
        return agent

    def _run_attempt(current_agent, current_output_path: Path) -> tuple[dict[str, Any], dict[str, Any], float, int, int, int]:
        current_result = current_agent.run_conversation(user_message=prompt)
        payload, _raw_output = parse_agent_output(
            current_agent,
            current_result,
            output_path=current_output_path,
            schema=schema,
            step="review",
            project_dir=project_dir,
            plan_dir=plan_dir,
        )
        clean_parsed_payload(payload, schema, "review")
        return (
            current_result,
            payload,
            float(current_result.get("estimated_cost_usd", 0.0) or 0.0),
            int(current_result.get("prompt_tokens", 0) or 0),
            int(current_result.get("completion_tokens", 0) or 0),
            int(current_result.get("total_tokens", 0) or 0),
        )

    agent = _make_agent(resolved_model, agent_kwargs)
    try:
        _result, payload, cost_usd, pt, ct, tt = _run_attempt(agent, output_path)
    except Exception as exc:
        exc_str = str(exc)
        if "429" in exc_str:
            if model and model.startswith("minimax:"):
                report_429("minimax", agent_kwargs.get("api_key", ""), cooldown_secs=60)
            elif model and model.startswith("zhipu:"):
                cooldown = 3600 if "Limit Exhausted" in exc_str else 120
                report_429("zhipu", agent_kwargs.get("api_key", ""), cooldown_secs=cooldown)
        if model and model.startswith("minimax:"):
            or_key = acquire_key("openrouter")
            if or_key:
                from megaplan.key_pool import minimax_openrouter_model

                fallback_model = minimax_openrouter_model(model[len("minimax:"):])
                fallback_kwargs = {"base_url": "https://openrouter.ai/api/v1", "api_key": or_key}
                if isinstance(exc, CliError):
                    print(
                        f"[parallel-review] MiniMax returned bad criteria content ({_failure_reason(exc)}), falling back to OpenRouter",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[parallel-review] Primary MiniMax failed for criteria verdict ({exc}), falling back to OpenRouter",
                        file=sys.stderr,
                    )
                output_path = _write_criteria_verdict_review_template(plan_dir, state, "review_criteria_verdict.json")
                agent = _make_agent(fallback_model, fallback_kwargs)
                try:
                    _result, payload, cost_usd, pt, ct, tt = _run_attempt(agent, output_path)
                except Exception as fallback_exc:
                    raise CliError(
                        "worker_error",
                        (
                            "Parallel review criteria verdict failed "
                            f"(both MiniMax and OpenRouter): primary={_failure_reason(exc)}; "
                            f"fallback={_failure_reason(fallback_exc)}"
                        ),
                    ) from fallback_exc
            else:
                raise
        else:
            raise
    return payload, cost_usd, pt, ct, tt


def run_parallel_review(
    state: PlanState,
    plan_dir: Path,
    *,
    root: Path,
    model: str | None,
    checks: tuple[Any, ...],
    pre_check_flags: list[dict[str, Any]],
    max_concurrent: int | None = None,
) -> WorkerResult:
    started = time.monotonic()
    schema = read_json(schemas_root(root) / STEP_SCHEMA_FILENAMES["review"])
    project_dir = Path(state["config"]["project_dir"])
    results: list[tuple[dict[str, Any], list[str], list[str]] | None] = [None] * len(checks)
    criteria_payload: dict[str, Any] | None = None
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        total_futures = len(checks) + 1
        concurrency = min(
            max_concurrent or get_effective("orchestration", "max_critique_concurrency"),
            total_futures,
        )
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
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
                    pre_check_flags=pre_check_flags,
                ): ("check", index)
                for index, check in enumerate(checks)
            }
            future_map[
                executor.submit(
                    _run_criteria_verdict,
                    state=state,
                    plan_dir=plan_dir,
                    root=root,
                    model=model,
                    schema=schema,
                    project_dir=project_dir,
                )
            ] = ("criteria", None)
            for future in as_completed(future_map):
                kind, _index = future_map[future]
                result = future.result()
                if kind == "criteria":
                    criteria_payload, cost_usd, pt, ct, tt = result
                else:
                    index, check_payload, verified_ids, disputed_ids, cost_usd, pt, ct, tt = result
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
            raise CliError("worker_error", "Parallel review did not return all check results")
        check_payload, verified_ids, disputed_ids = item
        ordered_checks.append(check_payload)
        verified_groups.append(verified_ids)
        disputed_groups.append(disputed_ids)

    if criteria_payload is None:
        raise CliError("worker_error", "Parallel review did not return a criteria verdict payload")

    disputed_flag_ids = _merge_unique(disputed_groups)
    disputed_set = set(disputed_flag_ids)
    verified_flag_ids = [flag_id for flag_id in _merge_unique(verified_groups) if flag_id not in disputed_set]
    return WorkerResult(
        payload={
            "checks": ordered_checks,
            "verified_flag_ids": verified_flag_ids,
            "disputed_flag_ids": disputed_flag_ids,
            "criteria_payload": criteria_payload,
        },
        raw_output="parallel",
        duration_ms=int((time.monotonic() - started) * 1000),
        cost_usd=total_cost,
        session_id=None,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )
