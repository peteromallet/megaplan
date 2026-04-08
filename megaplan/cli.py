#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from megaplan.types import (
    CliError,
    DEFAULT_AGENT_ROUTING,
    DEFAULTS,
    KNOWN_AGENTS,
    ROBUSTNESS_LEVELS,
    StepResponse,
    _SETTABLE_BOOL,
    _SETTABLE_ENUM,
    _SETTABLE_NUMERIC,
)
from megaplan._core import (
    active_plan_dirs,
    add_or_increment_debt,
    atomic_write_text,
    compute_global_batches,
    config_dir,
    detect_available_agents,
    escalated_subsystems,
    ensure_runtime_layout,
    get_effective,
    infer_next_steps,
    json_dump,
    load_config,
    load_debt_registry,
    load_plan,
    read_json,
    resolve_debt,
    save_debt_registry,
    save_config,
    subsystem_occurrence_total,
)
from megaplan.handlers import (
    handle_critique,
    handle_execute,
    handle_finalize,
    handle_gate,
    handle_init,
    handle_override,
    handle_plan,
    handle_prep,
    handle_review,
    handle_revise,
)
from megaplan.loop.handlers import (
    handle_loop_init,
    handle_loop_pause,
    handle_loop_run,
    handle_loop_status,
)
from megaplan.step_edit import handle_step



def render_response(response: StepResponse, *, exit_code: int = 0) -> int:
    print(json_dump(response), end="")
    return exit_code


def error_response(error: CliError) -> int:
    payload: StepResponse = {
        "success": False,
        "error": error.code,
        "message": error.message,
    }
    if error.valid_next:
        payload["valid_next"] = error.valid_next
    if error.extra:
        payload["details"] = error.extra
    return render_response(payload, exit_code=error.exit_code)


def handle_status(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    next_steps = infer_next_steps(state)
    return {
        "success": True,
        "step": "status",
        "plan": state["name"],
        "state": state["current_state"],
        "iteration": state["iteration"],
        "summary": f"Plan '{state['name']}' is currently in state '{state['current_state']}'.",
        "next_step": next_steps[0] if next_steps else None,
        "valid_next": next_steps,
        "artifacts": sorted(path.name for path in plan_dir.iterdir() if path.is_file()),
    }


def handle_audit(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    return {
        "success": True,
        "step": "audit",
        "plan": state["name"],
        "plan_dir": str(plan_dir),
        "state": state,
    }


def handle_progress(root: Path, args: argparse.Namespace) -> StepResponse:
    plan_dir, state = load_plan(root, args.plan)
    finalize_path = plan_dir / "finalize.json"
    if not finalize_path.exists():
        return {
            "success": True,
            "step": "progress",
            "plan": state["name"],
            "summary": "No finalize.json yet — plan has not been finalized.",
            "tasks_total": 0,
            "tasks_done": 0,
            "tasks_skipped": 0,
            "tasks_pending": 0,
            "batches_total": 0,
            "batches_completed": 0,
            "tasks": [],
        }
    finalize_data = read_json(finalize_path)
    global_batches = compute_global_batches(finalize_data)
    tasks = finalize_data.get("tasks", [])
    task_id_to_batch: dict[str, int] = {}
    for batch_idx, batch_ids in enumerate(global_batches, start=1):
        for task_id in batch_ids:
            task_id_to_batch[task_id] = batch_idx
    tasks_done = sum(1 for t in tasks if t.get("status") == "done")
    tasks_skipped = sum(1 for t in tasks if t.get("status") == "skipped")
    tasks_pending = sum(1 for t in tasks if t.get("status") == "pending")
    tasks_total = len(tasks)
    # A batch is complete when ALL its task IDs have status done or skipped.
    completed_ids = {
        t["id"] for t in tasks if t.get("status") in {"done", "skipped"} and isinstance(t.get("id"), str)
    }
    batches_completed = sum(
        1
        for batch_ids in global_batches
        if all(tid in completed_ids for tid in batch_ids)
    )
    task_status_list = [
        {
            "id": t.get("id", ""),
            "status": t.get("status", "pending"),
            "batch": task_id_to_batch.get(t.get("id", ""), 0),
        }
        for t in tasks
    ]
    return {
        "success": True,
        "step": "progress",
        "plan": state["name"],
        "summary": (
            f"Execution progress: {tasks_done + tasks_skipped}/{tasks_total} tasks tracked, "
            f"{batches_completed}/{len(global_batches)} batches completed. "
            "Progress reflects the last finalize.json write (between-batch granularity)."
        ),
        "tasks_total": tasks_total,
        "tasks_done": tasks_done,
        "tasks_skipped": tasks_skipped,
        "tasks_pending": tasks_pending,
        "batches_total": len(global_batches),
        "batches_completed": batches_completed,
        "tasks": task_status_list,
    }


def handle_list(root: Path, args: argparse.Namespace) -> StepResponse:
    ensure_runtime_layout(root)
    items = []
    for plan_dir in active_plan_dirs(root):
        state = read_json(plan_dir / "state.json")
        next_steps = infer_next_steps(state)
        items.append(
            {
                "name": state["name"],
                "idea": state["idea"],
                "state": state["current_state"],
                "iteration": state["iteration"],
                "next_step": next_steps[0] if next_steps else None,
            }
        )
    return {
        "success": True,
        "step": "list",
        "summary": f"Found {len(items)} plans.",
        "plans": items,
    }


def handle_debt(root: Path, args: argparse.Namespace) -> StepResponse:
    ensure_runtime_layout(root)
    action = args.debt_action
    registry = load_debt_registry(root)
    default_plan_id = getattr(args, "plan", None) or "manual"

    if action == "list":
        entries = registry["entries"] if args.all else [entry for entry in registry["entries"] if not entry["resolved"]]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            grouped.setdefault(entry["subsystem"], []).append(entry)
        escalated = {
            subsystem: total
            for subsystem, total, _entries in escalated_subsystems(registry)
        }
        by_subsystem = [
            {
                "subsystem": subsystem,
                "escalated": subsystem in escalated,
                "total_occurrences": subsystem_occurrence_total(entries_for_subsystem)
                if not args.all
                else sum(entry["occurrence_count"] for entry in entries_for_subsystem if not entry["resolved"]),
                "entries": entries_for_subsystem,
            }
            for subsystem, entries_for_subsystem in sorted(grouped.items())
        ]
        return {
            "success": True,
            "step": "debt",
            "action": "list",
            "summary": f"Found {len(entries)} debt entries across {len(by_subsystem)} subsystem groups.",
            "details": {
                "entries": entries,
                "by_subsystem": by_subsystem,
                "escalated_subsystems": [
                    {"subsystem": subsystem, "total_occurrences": total}
                    for subsystem, total in sorted(escalated.items())
                ],
            },
        }

    if action == "add":
        flag_ids = [
            flag_id.strip()
            for flag_id in (args.flag_ids or "").split(",")
            if flag_id.strip()
        ]
        entry = add_or_increment_debt(
            registry,
            subsystem=args.subsystem,
            concern=args.concern,
            flag_ids=flag_ids,
            plan_id=default_plan_id,
        )
        save_debt_registry(root, registry)
        return {
            "success": True,
            "step": "debt",
            "action": "add",
            "summary": f"Tracked debt entry {entry['id']} for subsystem '{entry['subsystem']}'.",
            "details": {"entry": entry},
        }

    if action == "resolve":
        entry = resolve_debt(registry, args.debt_id, default_plan_id)
        save_debt_registry(root, registry)
        return {
            "success": True,
            "step": "debt",
            "action": "resolve",
            "summary": f"Resolved debt entry {entry['id']}.",
            "details": {"entry": entry},
        }

    raise CliError("invalid_args", f"Unknown debt action: {action}")


# ---------------------------------------------------------------------------
# Setup and config
# ---------------------------------------------------------------------------

def _canonical_instructions() -> str:
    return resources.files("megaplan").joinpath("data", "instructions.md").read_text(encoding="utf-8")


_SKILL_HEADER = """\
---
name: megaplan
description: AI agent harness for coordinating Claude and GPT to make and execute extremely robust plans.
---

"""

_CURSOR_HEADER = """\
---
description: Use megaplan for high-rigor planning on complex, high-risk, or multi-stage tasks.
alwaysApply: false
---

"""


def bundled_agents_md() -> str:
    return _canonical_instructions()


def _claude_subagent_appendix() -> str:
    content = resources.files("megaplan").joinpath("data", "claude_subagent_appendix.md").read_text(encoding="utf-8")
    content = content.replace(
        "{max_execute_no_progress}",
        str(get_effective("execution", "max_execute_no_progress")),
    )
    content = content.replace(
        "{max_review_rework_cycles}",
        str(get_effective("execution", "max_review_rework_cycles")),
    )
    return content


def bundled_global_file(name: str) -> str:
    content = _canonical_instructions()
    if name == "claude_skill.md":
        return _SKILL_HEADER + content + "\n\n" + _claude_subagent_appendix()
    if name == "skill.md":
        return _SKILL_HEADER + content
    if name == "cursor_rule.mdc":
        return _CURSOR_HEADER + content
    return content


_GLOBAL_TARGETS = [
    {"agent": "claude", "detect": ".claude", "path": ".claude/skills/megaplan/SKILL.md", "data": "claude_skill.md"},
    {"agent": "codex", "detect": ".codex", "path": ".codex/skills/megaplan/SKILL.md", "data": "skill.md"},
    {"agent": "cursor", "detect": ".cursor", "path": ".cursor/rules/megaplan.mdc", "data": "cursor_rule.mdc"},
]


def _install_owned_file(path: Path, content: str, *, force: bool = False) -> dict[str, bool | str]:
    existed = path.exists()
    if existed and not force:
        if path.read_text(encoding="utf-8") == content:
            return {"path": str(path), "skipped": True, "existed": True}
    atomic_write_text(path, content)
    return {"path": str(path), "skipped": False, "existed": existed}


def handle_setup_global(force: bool = False, home: Path | None = None) -> StepResponse:
    if home is None:
        home = Path.home()
    installed: list[dict[str, Any]] = []
    detected_count = 0
    for target in _GLOBAL_TARGETS:
        agent_dir = home / target["detect"]
        if not agent_dir.is_dir():
            installed.append({"agent": target["agent"], "path": str(home / target["path"]), "skipped": True, "reason": "not installed"})
            continue
        detected_count += 1
        result = _install_owned_file(home / target["path"], bundled_global_file(target["data"]), force=force)
        result["agent"] = target["agent"]
        installed.append(result)
    if detected_count == 0:
        return {
            "success": False, "step": "setup", "mode": "global",
            "summary": "No supported agents detected. Create one of ~/.claude/, ~/.codex/, or ~/.cursor/ and re-run.",
            "installed": installed,
        }
    available = detect_available_agents()
    config_path = None
    routing = None
    if available:
        agents_config = {step: (default if default in available else available[0]) for step, default in DEFAULT_AGENT_ROUTING.items()}
        config = load_config(home)
        config["agents"] = agents_config
        config_path = save_config(config, home)
        routing = agents_config
    lines = []
    for rec in installed:
        if rec.get("reason") == "not installed":
            lines.append(f"  {rec['agent']}: skipped (not installed)")
        elif rec["skipped"]:
            lines.append(f"  {rec['agent']}: up to date")
        else:
            lines.append(f"  {rec['agent']}: {'overwrote' if rec['existed'] else 'created'} {rec['path']}")
    result_data: dict[str, Any] = {"success": True, "step": "setup", "mode": "global", "summary": "Global setup complete:\n" + "\n".join(lines), "installed": installed}
    if config_path is not None:
        result_data["config_path"] = str(config_path)
        result_data["routing"] = routing
    return result_data


def handle_setup(args: argparse.Namespace) -> StepResponse:
    local = args.local or args.target_dir
    if not local:
        return handle_setup_global(force=args.force)
    target_dir = Path(args.target_dir).resolve() if args.target_dir else Path.cwd()
    target = target_dir / "AGENTS.md"
    content = bundled_agents_md()
    if target.exists() and not args.force:
        existing = target.read_text(encoding="utf-8")
        if "megaplan" in existing.lower():
            return {"success": True, "step": "setup", "summary": f"AGENTS.md already contains megaplan instructions at {target}", "skipped": True}
        atomic_write_text(target, existing + "\n\n" + content)
        return {"success": True, "step": "setup", "summary": f"Appended megaplan instructions to existing {target}", "file": str(target)}
    atomic_write_text(target, content)
    return {"success": True, "step": "setup", "summary": f"Created {target}", "file": str(target)}


def handle_config(args: argparse.Namespace) -> StepResponse:
    action = args.config_action
    if action == "show":
        config = load_config()
        effective_routing = {step: config.get("agents", {}).get(step, default) for step, default in DEFAULT_AGENT_ROUTING.items()}
        effective_settings = {
            dot_key: get_effective(section, setting)
            for dot_key in sorted(DEFAULTS)
            for section, setting in [dot_key.split(".", 1)]
        }
        return {
            "success": True,
            "step": "config",
            "action": "show",
            "config_path": str(config_dir() / "config.json"),
            "routing": effective_routing,
            "effective_settings": effective_settings,
            "raw_config": config,
        }
    if action == "set":
        key, value = args.key, args.value
        parts = key.split(".", 1)
        config = load_config()
        valid_keys = [
            *(f"agents.{step}" for step in DEFAULT_AGENT_ROUTING),
            "orchestration.mode",
            *sorted(_SETTABLE_BOOL),
            *sorted(_SETTABLE_ENUM),
            *sorted(_SETTABLE_NUMERIC),
        ]
        if len(parts) != 2:
            raise CliError(
                "invalid_args",
                f"Unknown config key '{key}'. Valid keys: {', '.join(valid_keys)}",
            )
        section, setting = parts
        normalized_value = value.strip().lower()
        if section == "agents":
            if setting not in DEFAULT_AGENT_ROUTING:
                raise CliError("invalid_args", f"Unknown step '{setting}'. Valid steps: {', '.join(DEFAULT_AGENT_ROUTING)}")
            if value not in KNOWN_AGENTS:
                raise CliError("invalid_args", f"Unknown agent '{value}'. Valid agents: {', '.join(KNOWN_AGENTS)}")
            config.setdefault("agents", {})[setting] = value
        elif key == "orchestration.mode":
            if value not in {"inline", "subagent"}:
                raise CliError("invalid_args", "orchestration.mode must be 'inline' or 'subagent'")
            config.setdefault("orchestration", {})["mode"] = value
        elif key in _SETTABLE_BOOL:
            if normalized_value in {"true", "1", "yes", "on"}:
                parsed_value = True
            elif normalized_value in {"false", "0", "no", "off"}:
                parsed_value = False
            else:
                raise CliError(
                    "invalid_args",
                    f"{key} must be one of: true, false, 1, 0, yes, no, on, off",
                )
            config.setdefault(section, {})[setting] = parsed_value
        elif key in _SETTABLE_ENUM:
            allowed_values = _SETTABLE_ENUM[key]
            if value not in allowed_values:
                raise CliError(
                    "invalid_args",
                    f"{key} must be one of: {', '.join(allowed_values)}",
                )
            config.setdefault(section, {})[setting] = value
        elif key in _SETTABLE_NUMERIC:
            try:
                parsed_value = int(value)
            except ValueError as exc:
                raise CliError("invalid_args", f"{key} must be an integer, got '{value}'") from exc
            config.setdefault(section, {})[setting] = parsed_value
        else:
            raise CliError(
                "invalid_args",
                f"Unknown config key '{key}'. Valid keys: {', '.join(valid_keys)}",
            )
        save_config(config)
        return {"success": True, "step": "config", "action": "set", "key": key, "value": config[section][setting]}
    if action == "reset":
        path = config_dir() / "config.json"
        if path.exists():
            path.unlink()
        return {"success": True, "step": "config", "action": "reset", "summary": "Config file removed. Using defaults."}
    raise CliError("invalid_args", f"Unknown config action: {action}")


# ---------------------------------------------------------------------------
# Parser and dispatch
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Megaplan orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Install megaplan into agent configs (global by default)")
    setup_parser.add_argument("--local", action="store_true", help="Install AGENTS.md into a project instead of global agent configs")
    setup_parser.add_argument("--target-dir", help="Directory to install into (default: cwd, implies --local)")
    setup_parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--project-dir", required=True)
    init_parser.add_argument("--name")
    init_parser.add_argument("--auto-approve", action="store_true", default=None)
    init_parser.add_argument("--robustness", choices=list(ROBUSTNESS_LEVELS), default=None)
    init_parser.add_argument("--hermes", nargs="?", const="", default=None,
                             help="Use Hermes agent for all phases. Optional: specify default model")
    init_parser.add_argument("--phase-model", action="append", default=[],
                             help="Per-phase model override: --phase-model critique=hermes:openai/gpt-5")
    init_parser.add_argument("idea")

    subparsers.add_parser("list")

    for name in ["status", "audit", "progress"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")

    for name in ["plan", "prep", "critique", "revise", "gate", "finalize", "execute", "review"]:
        step_parser = subparsers.add_parser(name)
        step_parser.add_argument("--plan")
        step_parser.add_argument("--agent", choices=["claude", "codex", "hermes"])
        step_parser.add_argument("--hermes", nargs="?", const="", default=None,
                                 help="Use Hermes agent for all phases. Optional: specify default model (e.g. --hermes anthropic/claude-sonnet-4.6)")
        step_parser.add_argument("--phase-model", action="append", default=[],
                                 help="Per-phase model override: --phase-model critique=hermes:openai/gpt-5")
        step_parser.add_argument("--fresh", action="store_true")
        step_parser.add_argument("--persist", action="store_true")
        step_parser.add_argument("--ephemeral", action="store_true")
        if name == "execute":
            step_parser.add_argument("--confirm-destructive", action="store_true")
            step_parser.add_argument("--user-approved", action="store_true")
            step_parser.add_argument("--batch", type=int, default=None, help="Execute a specific global batch number (1-indexed)")
        if name == "review":
            step_parser.add_argument("--confirm-self-review", action="store_true")

    config_parser = subparsers.add_parser("config", help="View or edit megaplan configuration")
    config_sub = config_parser.add_subparsers(dest="config_action", required=True)
    config_sub.add_parser("show")
    set_parser = config_sub.add_parser("set")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    config_sub.add_parser("reset")

    step_parser = subparsers.add_parser("step", help="Edit plan step sections without hand-editing markdown")
    step_subparsers = step_parser.add_subparsers(dest="step_action", required=True)

    step_add_parser = step_subparsers.add_parser("add", help="Insert a new step after an existing step")
    step_add_parser.add_argument("--plan")
    step_add_parser.add_argument("--after")
    step_add_parser.add_argument("description")

    step_remove_parser = step_subparsers.add_parser("remove", help="Remove a step and renumber the plan")
    step_remove_parser.add_argument("--plan")
    step_remove_parser.add_argument("step_id")

    step_move_parser = step_subparsers.add_parser("move", help="Move a step after another step and renumber")
    step_move_parser.add_argument("--plan")
    step_move_parser.add_argument("step_id")
    step_move_parser.add_argument("--after", required=True)

    override_parser = subparsers.add_parser("override")
    override_parser.add_argument("override_action", choices=["abort", "force-proceed", "add-note", "replan"])
    override_parser.add_argument("--plan")
    override_parser.add_argument("--reason", default="")
    override_parser.add_argument("--note")

    debt_parser = subparsers.add_parser("debt", help="Inspect or manage persistent tech debt entries")
    debt_subparsers = debt_parser.add_subparsers(dest="debt_action", required=True)

    debt_list_parser = debt_subparsers.add_parser("list", help="List debt entries")
    debt_list_parser.add_argument("--all", action="store_true", help="Include resolved entries")

    debt_add_parser = debt_subparsers.add_parser("add", help="Add or increment a debt entry")
    debt_add_parser.add_argument("--subsystem", required=True)
    debt_add_parser.add_argument("--concern", required=True)
    debt_add_parser.add_argument("--flag-ids", default="")
    debt_add_parser.add_argument("--plan")

    debt_resolve_parser = debt_subparsers.add_parser("resolve", help="Resolve a debt entry")
    debt_resolve_parser.add_argument("debt_id")
    debt_resolve_parser.add_argument("--plan")

    loop_init_parser = subparsers.add_parser("loop-init", help="Initialize a MegaLoop workflow")
    loop_init_parser.add_argument("--project-dir", required=True)
    loop_init_parser.add_argument("--command", required=True)
    loop_init_parser.add_argument("--goal", dest="goal_option")
    loop_init_parser.add_argument("--name")
    loop_init_parser.add_argument("--iterations", type=int, default=3)
    loop_init_parser.add_argument("--time-budget", type=int, default=300)
    loop_init_parser.add_argument("--observe-interval", type=int)
    loop_init_parser.add_argument("--observe-break-patterns")
    loop_init_parser.add_argument("--agent", choices=["claude", "codex", "hermes"])
    loop_init_parser.add_argument("--hermes", nargs="?", const="", default=None,
                                  help="Use Hermes agent for loop phases. Optional: specify default model")
    loop_init_parser.add_argument("--phase-model", action="append", default=[],
                                  help="Per-phase model override: --phase-model loop_execute=hermes:openai/gpt-5")
    loop_init_parser.add_argument("--fresh", action="store_true")
    loop_init_parser.add_argument("--persist", action="store_true")
    loop_init_parser.add_argument("--ephemeral", action="store_true")
    loop_init_parser.add_argument("goal", nargs="?")

    loop_run_parser = subparsers.add_parser("loop-run", help="Run an existing MegaLoop workflow")
    loop_run_parser.add_argument("name")
    loop_run_parser.add_argument("--project-dir")
    loop_run_parser.add_argument("--iterations", type=int)
    loop_run_parser.add_argument("--time-budget", type=int)
    loop_run_parser.add_argument("--agent", choices=["claude", "codex", "hermes"])
    loop_run_parser.add_argument("--hermes", nargs="?", const="", default=None,
                                 help="Use Hermes agent for loop phases. Optional: specify default model")
    loop_run_parser.add_argument("--phase-model", action="append", default=[],
                                 help="Per-phase model override: --phase-model loop_execute=hermes:openai/gpt-5")
    loop_run_parser.add_argument("--fresh", action="store_true")
    loop_run_parser.add_argument("--persist", action="store_true")
    loop_run_parser.add_argument("--ephemeral", action="store_true")

    loop_status_parser = subparsers.add_parser("loop-status", help="Show MegaLoop state")
    loop_status_parser.add_argument("name")
    loop_status_parser.add_argument("--project-dir")

    loop_pause_parser = subparsers.add_parser("loop-pause", help="Pause a MegaLoop workflow")
    loop_pause_parser.add_argument("name")
    loop_pause_parser.add_argument("--project-dir")
    loop_pause_parser.add_argument("--reason", default="")

    return parser


COMMAND_HANDLERS: dict[str, Callable[..., StepResponse]] = {
    "init": handle_init,
    "plan": handle_plan,
    "prep": handle_prep,
    "critique": handle_critique,
    "revise": handle_revise,
    "gate": handle_gate,
    "finalize": handle_finalize,
    "execute": handle_execute,
    "review": handle_review,
    "status": handle_status,
    "audit": handle_audit,
    "progress": handle_progress,
    "list": handle_list,
    "loop-init": handle_loop_init,
    "loop-run": handle_loop_run,
    "loop-status": handle_loop_status,
    "loop-pause": handle_loop_pause,
    "debt": handle_debt,
    "step": handle_step,
    "override": handle_override,
}


def cli_entry() -> None:
    sys.exit(main())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    try:
        if args.command == "setup":
            return render_response(handle_setup(args))
        if args.command == "config":
            return render_response(handle_config(args))
    except CliError as error:
        return error_response(error)

    root = Path.cwd()
    ensure_runtime_layout(root)
    try:
        handler = COMMAND_HANDLERS.get(args.command)
        if handler is None:
            raise CliError("invalid_command", f"Unknown command {args.command!r}")
        if args.command == "override" and remaining:
            if not args.note:
                args.note = " ".join(remaining)
            remaining = []
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        if args.command == "override" and args.override_action == "add-note" and not args.note:
            raise CliError("invalid_args", "override add-note requires a note")
        return render_response(handler(root, args))
    except CliError as error:
        return error_response(error)


if __name__ == "__main__":
    sys.exit(main())
