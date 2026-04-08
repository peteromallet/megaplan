"""Core infrastructure for megaplan — re-exports from submodules.

All public names are available via ``from megaplan._core import <name>``
so existing callers don't need to change.
"""

# Re-export shutil so monkeypatches like `megaplan._core.shutil.which` still work.
import shutil  # noqa: F401

# -- io.py: pure utilities, atomic I/O, paths, config -----------------------
from .io import (
    artifact_path,
    atomic_write_json,
    atomic_write_text,
    batch_artifact_path,
    collect_git_diff_patch,
    collect_git_diff_summary,
    compute_global_batches,
    compute_task_batches,
    config_dir,
    current_iteration_artifact,
    current_iteration_raw_artifact,
    detect_available_agents,
    ensure_runtime_layout,
    find_command,
    get_effective,
    json_dump,
    list_batch_artifacts,
    load_config,
    load_finalize_snapshot,
    megaplan_root,
    normalize_text,
    now_utc,
    plans_root,
    read_json,
    render_final_md,
    save_config,
    schemas_root,
    sha256_file,
    sha256_text,
    slugify,
)

# -- state.py: plan state, history, sessions ---------------------------------
from .state import (
    active_plan_dirs,
    append_history,
    apply_session_update,
    latest_plan_meta_path,
    latest_plan_path,
    latest_plan_record,
    load_plan,
    make_history_entry,
    record_step_failure,
    resolve_plan_dir,
    save_state,
    store_raw_worker_output,
)

# -- registries.py: flag + debt registries -----------------------------------
from .registries import (
    add_or_increment_debt,
    debt_by_subsystem,
    escalated_subsystems,
    extract_subsystem_tag,
    find_matching_debt,
    is_scope_creep_flag,
    load_debt_registry,
    load_flag_registry,
    next_debt_id,
    resolve_debt,
    save_debt_registry,
    save_flag_registry,
    scope_creep_flags,
    subsystem_occurrence_total,
    unresolved_significant_flags,
)

# -- workflow.py: state machine, transitions ---------------------------------
from megaplan.types import ROBUSTNESS_LEVELS  # noqa: F401 — accessed by tests via megaplan._core

from .workflow import (
    WORKFLOW,
    Transition,
    _ROBUSTNESS_OVERRIDES,
    configured_robustness,
    infer_next_steps,
    intent_and_notes_block,
    require_state,
    robustness_critique_instruction,
    workflow_includes_step,
    workflow_next,
    workflow_transition,
)

__all__ = [
    # io
    "artifact_path",
    "atomic_write_json",
    "atomic_write_text",
    "batch_artifact_path",
    "collect_git_diff_patch",
    "collect_git_diff_summary",
    "compute_global_batches",
    "compute_task_batches",
    "config_dir",
    "current_iteration_artifact",
    "current_iteration_raw_artifact",
    "detect_available_agents",
    "ensure_runtime_layout",
    "find_command",
    "get_effective",
    "json_dump",
    "list_batch_artifacts",
    "load_config",
    "load_finalize_snapshot",
    "megaplan_root",
    "normalize_text",
    "now_utc",
    "plans_root",
    "read_json",
    "render_final_md",
    "save_config",
    "schemas_root",
    "sha256_file",
    "sha256_text",
    "slugify",
    # state
    "active_plan_dirs",
    "append_history",
    "apply_session_update",
    "latest_plan_meta_path",
    "latest_plan_path",
    "latest_plan_record",
    "load_plan",
    "make_history_entry",
    "record_step_failure",
    "resolve_plan_dir",
    "save_state",
    "store_raw_worker_output",
    # registries
    "add_or_increment_debt",
    "debt_by_subsystem",
    "escalated_subsystems",
    "extract_subsystem_tag",
    "find_matching_debt",
    "is_scope_creep_flag",
    "load_debt_registry",
    "load_flag_registry",
    "next_debt_id",
    "resolve_debt",
    "save_debt_registry",
    "save_flag_registry",
    "scope_creep_flags",
    "subsystem_occurrence_total",
    "unresolved_significant_flags",
    # workflow
    "WORKFLOW",
    "Transition",
    "_ROBUSTNESS_OVERRIDES",
    "configured_robustness",
    "infer_next_steps",
    "intent_and_notes_block",
    "require_state",
    "robustness_critique_instruction",
    "workflow_includes_step",
    "workflow_next",
    "workflow_transition",
]
