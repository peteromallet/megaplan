"""Centralized per-phase runtime policy for patience, polling, and timeouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS = 900


@dataclass(frozen=True)
class PhaseRuntimePolicy:
    expected_min_seconds: int
    expected_max_seconds: int | None
    recommended_next_check_seconds: int
    escalation_threshold_seconds: int | None
    timeout_cap_seconds: int | None
    artifact_mode: str = "completion_only"


@dataclass(frozen=True)
class ResolvedPhaseRuntime:
    expected_duration_seconds: dict[str, int]
    recommended_next_check_seconds: int
    escalation_threshold_seconds: int
    timeout_budget_seconds: int
    artifact_mode: str


PHASE_RUNTIME_POLICY: dict[str, PhaseRuntimePolicy] = {
    "prep": PhaseRuntimePolicy(
        expected_min_seconds=30,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=60,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "plan": PhaseRuntimePolicy(
        expected_min_seconds=60,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=120,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "critique": PhaseRuntimePolicy(
        expected_min_seconds=60,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=120,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "revise": PhaseRuntimePolicy(
        expected_min_seconds=60,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=120,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "gate": PhaseRuntimePolicy(
        expected_min_seconds=30,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=60,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "finalize": PhaseRuntimePolicy(
        expected_min_seconds=60,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=120,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "execute": PhaseRuntimePolicy(
        expected_min_seconds=300,
        expected_max_seconds=None,
        recommended_next_check_seconds=300,
        escalation_threshold_seconds=None,
        timeout_cap_seconds=None,
    ),
    "review": PhaseRuntimePolicy(
        expected_min_seconds=60,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=120,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "loop_plan": PhaseRuntimePolicy(
        expected_min_seconds=60,
        expected_max_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        recommended_next_check_seconds=120,
        escalation_threshold_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
        timeout_cap_seconds=DEFAULT_NON_EXECUTE_TIMEOUT_CAP_SECONDS,
    ),
    "loop_execute": PhaseRuntimePolicy(
        expected_min_seconds=300,
        expected_max_seconds=None,
        recommended_next_check_seconds=300,
        escalation_threshold_seconds=None,
        timeout_cap_seconds=None,
    ),
}


def phase_runtime_policy(step: str) -> PhaseRuntimePolicy:
    try:
        return PHASE_RUNTIME_POLICY[step]
    except KeyError as exc:
        raise KeyError(f"Unknown phase runtime step: {step}") from exc


def humanize_seconds(seconds: int) -> str:
    total_seconds = max(0, int(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, remainder = divmod(total_seconds, 60)
    if total_seconds < 3600:
        if remainder == 0:
            return f"{minutes}m"
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h {minutes}m"


def resolve_phase_runtime(step: str, *, configured_timeout_seconds: int) -> ResolvedPhaseRuntime:
    policy = phase_runtime_policy(step)
    timeout_budget_seconds = configured_timeout_seconds
    if policy.timeout_cap_seconds is not None:
        timeout_budget_seconds = min(timeout_budget_seconds, policy.timeout_cap_seconds)
    expected_max_seconds = timeout_budget_seconds
    if policy.expected_max_seconds is not None:
        expected_max_seconds = min(expected_max_seconds, policy.expected_max_seconds)
    escalation_threshold_seconds = timeout_budget_seconds
    if policy.escalation_threshold_seconds is not None:
        escalation_threshold_seconds = min(
            escalation_threshold_seconds,
            policy.escalation_threshold_seconds,
        )
    return ResolvedPhaseRuntime(
        expected_duration_seconds={
            "min": policy.expected_min_seconds,
            "max": max(policy.expected_min_seconds, expected_max_seconds),
        },
        recommended_next_check_seconds=policy.recommended_next_check_seconds,
        escalation_threshold_seconds=max(policy.expected_min_seconds, escalation_threshold_seconds),
        timeout_budget_seconds=max(policy.expected_min_seconds, timeout_budget_seconds),
        artifact_mode=policy.artifact_mode,
    )


def format_duration_hint(step: str, *, configured_timeout_seconds: int) -> str:
    policy = phase_runtime_policy(step)
    resolved = resolve_phase_runtime(step, configured_timeout_seconds=configured_timeout_seconds)
    if policy.expected_max_seconds is None:
        return (
            f"Expected minimum duration: {humanize_seconds(policy.expected_min_seconds)} "
            "(depends on task count)."
        )
    return (
        "Expected duration: "
        f"{humanize_seconds(resolved.expected_duration_seconds['min'])}-"
        f"{humanize_seconds(resolved.expected_duration_seconds['max'])}."
    )


def build_next_step_runtime(
    step: Any,
    *,
    configured_timeout_seconds: int,
) -> dict[str, Any] | None:
    if not isinstance(step, str) or step not in PHASE_RUNTIME_POLICY:
        return None
    resolved = resolve_phase_runtime(step, configured_timeout_seconds=configured_timeout_seconds)
    return {
        "expected_duration_seconds": resolved.expected_duration_seconds,
        "recommended_next_check_seconds": resolved.recommended_next_check_seconds,
        "duration_hint": format_duration_hint(
            step,
            configured_timeout_seconds=configured_timeout_seconds,
        ),
    }


def phase_timeout_seconds(step: str, *, configured_timeout_seconds: int) -> int:
    return resolve_phase_runtime(
        step,
        configured_timeout_seconds=configured_timeout_seconds,
    ).timeout_budget_seconds


def phase_stale_seconds(step: str, *, configured_timeout_seconds: int) -> int:
    return resolve_phase_runtime(
        step,
        configured_timeout_seconds=configured_timeout_seconds,
    ).escalation_threshold_seconds


def build_phase_observability(
    step: str,
    *,
    configured_timeout_seconds: int,
    age_seconds: int | None = None,
    lock_held: bool = False,
) -> dict[str, Any]:
    resolved = resolve_phase_runtime(step, configured_timeout_seconds=configured_timeout_seconds)
    payload: dict[str, Any] = asdict(resolved)
    stale = False
    if age_seconds is not None:
        stale = age_seconds >= resolved.escalation_threshold_seconds
        payload["age_seconds"] = age_seconds
        payload["stale"] = stale
    if age_seconds is None or not stale:
        payload["health"] = "healthy"
        payload["recommended_action"] = "wait"
        payload["recommended_action_reason"] = "The active step is within its expected runtime window."
        return payload
    if lock_held:
        payload["health"] = "slow"
        payload["recommended_action"] = "wait"
        payload["recommended_action_reason"] = (
            "The active step has exceeded its expected runtime window, but the plan lock is still held."
        )
        return payload
    payload["health"] = "stale"
    if step in {"execute", "loop_execute"}:
        payload["recommended_action"] = "rerun_execute"
        payload["recommended_action_reason"] = (
            "The active execute step is stale and no process holds the plan lock."
        )
    else:
        payload["recommended_action"] = "rerun_same_step"
        payload["recommended_action_reason"] = (
            "The active step is stale and no process holds the plan lock."
        )
    return payload
