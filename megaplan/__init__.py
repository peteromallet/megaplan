"""Megaplan — stateful orchestration CLI for high-rigor planning loops."""

from megaplan.cli import *  # noqa: F401,F403

__version__ = "0.1.0"

__all__ = [
    # Types
    "PlanState", "PlanConfig", "PlanMeta", "FlagRecord",
    # State constants
    "STATE_INITIALIZED", "STATE_CLARIFIED", "STATE_PLANNED", "STATE_CRITIQUED",
    "STATE_EVALUATED", "STATE_GATED", "STATE_EXECUTED", "STATE_DONE", "STATE_ABORTED",
    "TERMINAL_STATES",
    # Error and result types
    "CliError", "CommandResult", "WorkerResult",
    # Handlers
    "handle_init", "handle_clarify", "handle_plan", "handle_critique",
    "handle_evaluate", "handle_integrate", "handle_gate", "handle_execute",
    "handle_review", "handle_status", "handle_audit", "handle_list",
    "handle_override", "handle_setup", "handle_setup_global", "handle_config",
    # Key utilities
    "slugify", "build_evaluation", "main", "cli_entry",
]
