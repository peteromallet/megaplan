"""Type definitions, constants, and exceptions for megaplan."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

STATE_INITIALIZED = "initialized"
STATE_PREPPED = "prepped"
STATE_PLANNED = "planned"
STATE_CRITIQUED = "critiqued"
STATE_GATED = "gated"
STATE_FINALIZED = "finalized"
STATE_EXECUTED = "executed"
STATE_DONE = "done"
STATE_ABORTED = "aborted"
TERMINAL_STATES = {STATE_DONE, STATE_ABORTED}


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class PlanConfig(TypedDict, total=False):
    project_dir: str
    auto_approve: bool
    robustness: str
    agents: dict[str, str]


class PlanMeta(TypedDict, total=False):
    significant_counts: list[int]
    weighted_scores: list[float]
    plan_deltas: list[float | None]
    recurring_critiques: list[str]
    total_cost_usd: float
    overrides: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    user_approved_gate: bool


class SessionInfo(TypedDict, total=False):
    id: str
    mode: str
    created_at: str
    last_used_at: str
    refreshed: bool


class ActiveStep(TypedDict, total=False):
    step: str
    agent: str
    mode: str
    model: str
    run_id: str
    session_id: str
    started_at: str


class PlanVersionRecord(TypedDict, total=False):
    version: int
    file: str
    hash: str
    timestamp: str


class HistoryEntry(TypedDict, total=False):
    step: str
    timestamp: str
    duration_ms: int
    cost_usd: float
    result: str
    session_mode: str
    session_id: str
    agent: str
    output_file: str
    artifact_hash: str
    finalize_hash: str
    raw_output_file: str
    message: str
    flags_count: int
    flags_addressed: list[str]
    recommendation: str
    approval_mode: str
    environment: dict[str, bool]


class ClarificationRecord(TypedDict, total=False):
    refined_idea: str
    intent_summary: str
    questions: list[str]


class LastGateRecord(TypedDict, total=False):
    recommendation: str
    rationale: str
    signals_assessment: str
    warnings: list[str]
    settled_decisions: list["SettledDecision"]
    passed: bool
    preflight_results: dict[str, bool]
    orchestrator_guidance: str


class PlanState(TypedDict):
    name: str
    idea: str
    current_state: str
    iteration: int
    created_at: str
    config: PlanConfig
    sessions: dict[str, SessionInfo]
    plan_versions: list[PlanVersionRecord]
    history: list[HistoryEntry]
    meta: PlanMeta
    last_gate: LastGateRecord
    active_step: NotRequired[ActiveStep]
    clarification: NotRequired[ClarificationRecord]


class _FlagRecordRequired(TypedDict):
    id: str
    concern: str
    category: str
    status: str


class FlagRecord(_FlagRecordRequired, total=False):
    severity_hint: str
    evidence: str
    raised_in: str
    severity: str
    verified: bool
    verified_in: str
    addressed_in: str


class FlagRegistry(TypedDict):
    flags: list[FlagRecord]


class GateCheckResult(TypedDict):
    passed: bool
    criteria_check: dict[str, Any]
    preflight_results: dict[str, bool]
    unresolved_flags: list[FlagRecord]


class SettledDecision(TypedDict, total=False):
    id: str
    decision: str
    rationale: str


class GatePayload(TypedDict):
    recommendation: str
    rationale: str
    signals_assessment: str
    warnings: list[str]
    settled_decisions: list[SettledDecision]


class GateArtifact(TypedDict, total=False):
    passed: bool
    criteria_check: dict[str, Any]
    preflight_results: dict[str, bool]
    unresolved_flags: list[FlagRecord]
    recommendation: str
    rationale: str
    signals_assessment: str
    warnings: list[str]
    settled_decisions: list[SettledDecision]
    override_forced: bool
    orchestrator_guidance: str
    robustness: str
    signals: dict[str, Any]


class GateSignals(TypedDict, total=False):
    robustness: str
    signals: dict[str, Any]
    warnings: list[str]


class StepResponse(TypedDict, total=False):
    success: bool
    step: str
    summary: str
    artifacts: list[str]
    next_step: str | None
    state: str
    auto_approve: bool
    robustness: str
    iteration: int
    plan: str
    plan_dir: str
    questions: list[str]
    verified_flags: list[str]
    open_flags: list[str]
    scope_creep_flags: list[str]
    warnings: list[str]
    files_changed: list[str]
    deviations: list[str]
    user_approved_gate: bool
    issues: list[str]
    valid_next: list[str]
    mode: str
    installed: list[dict[str, Any]]
    config_path: str
    routing: dict[str, str]
    raw_config: dict[str, Any]
    action: str
    key: str
    value: str
    skipped: bool
    file: str
    plans: list[dict[str, Any]]
    recommendation: str
    signals: dict[str, Any]
    rationale: str
    signals_assessment: str
    orchestrator_guidance: str
    passed: bool
    criteria_check: dict[str, Any]
    preflight_results: dict[str, bool]
    unresolved_flags: list[Any]
    error: str
    message: str
    details: dict[str, Any]
    agent_fallback: dict[str, str]


class DebtEntry(TypedDict):
    id: str
    subsystem: str
    concern: str
    flag_ids: list[str]
    plan_ids: list[str]
    occurrence_count: int
    created_at: str
    updated_at: str
    resolved: bool
    resolved_by: str | None
    resolved_at: str | None


class DebtRegistry(TypedDict):
    entries: list[DebtEntry]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLAG_BLOCKING_STATUSES = {"open", "disputed", "addressed"}
FLAG_VALID_STATUSES = {
    "open", "addressed", "disputed", "verified",
    "accepted_tradeoff", "gate_disputed",
}
DEBT_ESCALATION_THRESHOLD = 3
MOCK_ENV_VAR = "MEGAPLAN_MOCK_WORKERS"

DEFAULT_AGENT_ROUTING: dict[str, str] = {
    "plan": "claude",
    "prep": "claude",
    "critique": "codex",
    "revise": "claude",
    "gate": "claude",
    "finalize": "claude",
    "execute": "codex",
    "loop_plan": "claude",
    "loop_execute": "codex",
    "review": "codex",
}
KNOWN_AGENTS = ["claude", "codex", "hermes"]
ROBUSTNESS_LEVELS = ("tiny", "light", "standard", "heavy")
def parse_agent_spec(spec: str) -> tuple[str, str | None]:
    """Parse 'hermes:model/name' → ('hermes', 'model/name') or 'claude' → ('claude', None)."""
    if ":" in spec:
        agent, model = spec.split(":", 1)
        return agent, model
    return spec, None


SCOPE_CREEP_TERMS = (
    "scope creep",
    "out of scope",
    "beyond the original idea",
    "beyond original idea",
    "beyond user intent",
    "expanded scope",
)

DEFAULTS = {
    "execution.auto_approve": False,
    "execution.robustness": "standard",
    "execution.worker_timeout_seconds": 7200,
    "execution.max_review_rework_cycles": 3,
    "execution.max_heavy_review_rework_cycles": 2,
    "execution.max_execute_no_progress": 3,
    "orchestration.max_critique_concurrency": 2,
    "orchestration.mode": "subagent",
}

_SETTABLE_BOOL = {
    "execution.auto_approve",
}

_SETTABLE_ENUM = {
    "execution.robustness": ROBUSTNESS_LEVELS,
}

_SETTABLE_NUMERIC = {
    "execution.worker_timeout_seconds",
    "execution.max_review_rework_cycles",
    "execution.max_heavy_review_rework_cycles",
    "execution.max_execute_no_progress",
    "orchestration.max_critique_concurrency",
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

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
