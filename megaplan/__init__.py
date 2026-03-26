"""Megaplan — stateful orchestration CLI for high-rigor planning loops."""

from megaplan.types import (
    PlanState, PlanConfig, PlanMeta, FlagRecord, StepResponse,
    STATE_INITIALIZED, STATE_PLANNED, STATE_CRITIQUED,
    STATE_GATED, STATE_FINALIZED, STATE_EXECUTED, STATE_DONE, STATE_ABORTED,
    TERMINAL_STATES, MOCK_ENV_VAR,
    CliError,
)
from megaplan._core import (
    slugify,
    config_dir, load_config, save_config,
    plans_root,
    unresolved_significant_flags,
)
from megaplan.workers import CommandResult, WorkerResult, mock_worker_output
from megaplan.evaluation import (
    build_orchestrator_guidance,
    build_gate_signals,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    flag_weight,
)
from megaplan.handlers import (
    handle_init,
    handle_plan,
    handle_critique,
    handle_revise,
    handle_gate,
    handle_finalize,
    handle_execute,
    handle_review,
    handle_step,
    normalize_flag_record,
    update_flags_after_critique,
    update_flags_after_revise,
)
from megaplan.handlers import handle_override
from megaplan.cli import handle_setup, handle_setup_global, handle_config
from megaplan._core import infer_next_steps, workflow_includes_step, workflow_next
from megaplan.cli import handle_status, handle_audit, handle_progress, handle_list, main, cli_entry

__version__ = "0.1.0"

__all__ = [
    # Types
    "PlanState", "PlanConfig", "PlanMeta", "FlagRecord", "StepResponse",
    # State constants
    "STATE_INITIALIZED", "STATE_PLANNED", "STATE_CRITIQUED",
    "STATE_GATED", "STATE_FINALIZED", "STATE_EXECUTED", "STATE_DONE", "STATE_ABORTED",
    "TERMINAL_STATES", "MOCK_ENV_VAR",
    # Error and result types
    "CliError", "CommandResult", "WorkerResult",
    # Handlers
    "handle_init", "handle_plan", "handle_critique",
    "handle_revise", "handle_gate", "handle_finalize", "handle_execute",
    "handle_review", "handle_step", "handle_status", "handle_audit", "handle_progress", "handle_list",
    "handle_override", "handle_setup", "handle_setup_global", "handle_config",
    # Key utilities
    "slugify", "build_gate_signals", "mock_worker_output",
    "build_orchestrator_guidance",
    "compute_plan_delta_percent", "compute_recurring_critiques", "flag_weight",
    "infer_next_steps", "workflow_includes_step", "workflow_next", "normalize_flag_record",
    "update_flags_after_critique", "update_flags_after_revise",
    "unresolved_significant_flags",
    "config_dir", "load_config", "save_config", "plans_root",
    "main", "cli_entry",
]
