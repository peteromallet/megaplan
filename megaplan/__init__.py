"""Megaplan — stateful orchestration CLI for high-rigor planning loops."""

from megaplan._core import (
    PlanState, PlanConfig, PlanMeta, FlagRecord, StepResponse,
    STATE_INITIALIZED, STATE_PLANNED, STATE_CRITIQUED,
    STATE_GATED, STATE_EXECUTED, STATE_DONE, STATE_ABORTED,
    TERMINAL_STATES, MOCK_ENV_VAR,
    CliError,
    slugify,
    config_dir, load_config, save_config,
    plans_root,
    unresolved_significant_flags,
)
from megaplan.workers import CommandResult, WorkerResult, mock_worker_output
from megaplan.evaluation import (
    build_gate_signals,
    compute_plan_delta_percent,
    compute_recurring_critiques,
    flag_weight,
)
from megaplan.cli import (
    handle_init, handle_plan, handle_critique,
    handle_revise, handle_gate, handle_execute,
    handle_review, handle_status, handle_audit, handle_list,
    handle_override, handle_setup, handle_setup_global, handle_config,
    infer_next_steps, normalize_flag_record,
    update_flags_after_critique, update_flags_after_revise,
    main, cli_entry,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "PlanState", "PlanConfig", "PlanMeta", "FlagRecord", "StepResponse",
    # State constants
    "STATE_INITIALIZED", "STATE_PLANNED", "STATE_CRITIQUED",
    "STATE_GATED", "STATE_EXECUTED", "STATE_DONE", "STATE_ABORTED",
    "TERMINAL_STATES", "MOCK_ENV_VAR",
    # Error and result types
    "CliError", "CommandResult", "WorkerResult",
    # Handlers
    "handle_init", "handle_plan", "handle_critique",
    "handle_revise", "handle_gate", "handle_execute",
    "handle_review", "handle_status", "handle_audit", "handle_list",
    "handle_override", "handle_setup", "handle_setup_global", "handle_config",
    # Key utilities
    "slugify", "build_gate_signals", "mock_worker_output",
    "compute_plan_delta_percent", "compute_recurring_critiques", "flag_weight",
    "infer_next_steps", "normalize_flag_record",
    "update_flags_after_critique", "update_flags_after_revise",
    "unresolved_significant_flags",
    "config_dir", "load_config", "save_config", "plans_root",
    "main", "cli_entry",
]
