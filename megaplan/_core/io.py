"""Atomic I/O, JSON helpers, path resolution, and config management."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from megaplan.schemas import SCHEMAS, strict_schema
from megaplan.types import KNOWN_AGENTS


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------

def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str, max_length: int = 30) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) <= max_length:
        return slug or "plan"
    truncated = slug[:max_length]
    last_hyphen = truncated.rfind("-")
    if last_hyphen > 10:
        truncated = truncated[:last_hyphen]
    return truncated or "plan"


def json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_text(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def compute_task_batches(
    tasks: list[dict[str, Any]],
    completed_ids: set[str] | None = None,
) -> list[list[str]]:
    completed = set(completed_ids or set())
    if not tasks:
        return []

    task_ids = [task["id"] for task in tasks]
    task_id_set = set(task_ids)
    remaining: dict[str, set[str]] = {}
    order_index = {task_id: index for index, task_id in enumerate(task_ids)}

    for task in tasks:
        task_id = task["id"]
        deps = task.get("depends_on", [])
        if not isinstance(deps, list):
            deps = []
        normalized_deps: set[str] = set()
        for dep in deps:
            if dep in task_id_set:
                normalized_deps.add(dep)
                continue
            if dep in completed:
                continue
            raise ValueError(f"Unknown dependency ID '{dep}' for task '{task_id}'")
        remaining[task_id] = normalized_deps

    batches: list[list[str]] = []
    satisfied = set(completed)
    unscheduled = set(task_ids)

    while unscheduled:
        ready = [
            task_id
            for task_id in unscheduled
            if remaining[task_id].issubset(satisfied)
        ]
        ready.sort(key=order_index.__getitem__)
        if not ready:
            cycle_ids = sorted(unscheduled, key=order_index.__getitem__)
            raise ValueError("Cyclic dependency graph detected among tasks: " + ", ".join(cycle_ids))
        batches.append(ready)
        satisfied.update(ready)
        unscheduled.difference_update(ready)

    return batches


def compute_global_batches(finalize_data: dict[str, Any]) -> list[list[str]]:
    tasks = finalize_data.get("tasks", [])
    return compute_task_batches(tasks)


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json_dump(data))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_finalize_snapshot(plan_dir: Path) -> dict[str, Any]:
    return read_json(plan_dir / "finalize_snapshot.json")


def render_final_md(finalize_data: dict[str, Any], *, phase: str = "finalize") -> str:
    show_execution_gaps = phase in ("execute", "review")
    show_review_gaps = phase == "review"
    tasks = finalize_data.get("tasks", [])
    sense_checks = finalize_data.get("sense_checks", [])

    lines = ["# Execution Checklist", ""]
    gap_counts: dict[str, int] = {}
    for task in tasks:
        status = task.get("status")
        checkbox = "[x]" if status == "done" else "[ ]"
        status_suffix = " (skipped)" if status == "skipped" else ""
        lines.append(f"- {checkbox} **{task['id']}:** {task['description']}{status_suffix}")
        depends_on = task.get("depends_on", [])
        if depends_on:
            lines.append(f"  Depends on: {', '.join(depends_on)}")
        executor_notes = task.get("executor_notes", "")
        if executor_notes.strip():
            lines.append(f"  Executor notes: {executor_notes}")
        elif show_execution_gaps and status != "pending":
            lines.append("  Executor notes: [MISSING]")
            gap_counts["Executor notes missing"] = gap_counts.get("Executor notes missing", 0) + 1
        files_changed = task.get("files_changed", [])
        if files_changed:
            lines.append("  Files changed:")
            for path in files_changed:
                lines.append(f"    - {path}")
        if show_execution_gaps and status == "pending":
            gap_counts["Tasks without executor updates"] = gap_counts.get("Tasks without executor updates", 0) + 1
        reviewer_verdict = task.get("reviewer_verdict", "")
        if reviewer_verdict.strip():
            lines.append(f"  Reviewer verdict: {reviewer_verdict}")
            evidence_files = task.get("evidence_files", [])
            if evidence_files:
                lines.append("  Evidence files:")
                for path in evidence_files:
                    lines.append(f"    - {path}")
        elif show_review_gaps:
            lines.append("  Reviewer verdict: [PENDING]")
            gap_counts["Reviewer verdicts pending"] = gap_counts.get("Reviewer verdicts pending", 0) + 1
        lines.append("")

    lines.extend(["## Watch Items", ""])
    watch_items = finalize_data.get("watch_items", [])
    if watch_items:
        for item in watch_items:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.extend(["## Sense Checks", ""])
    if sense_checks:
        for sense_check in sense_checks:
            lines.append(f"- **{sense_check['id']}** ({sense_check['task_id']}): {sense_check['question']}")
            executor_note = sense_check.get("executor_note", "")
            if executor_note.strip():
                lines.append(f"  Executor note: {executor_note}")
            elif show_execution_gaps:
                lines.append("  Executor note: [MISSING]")
                gap_counts["Sense-check acknowledgments missing"] = gap_counts.get("Sense-check acknowledgments missing", 0) + 1
            verdict = sense_check.get("verdict", "")
            if verdict.strip():
                lines.append(f"  Verdict: {verdict}")
            elif show_review_gaps:
                lines.append("  Verdict: [PENDING]")
                gap_counts["Sense-check verdicts pending"] = gap_counts.get("Sense-check verdicts pending", 0) + 1
            lines.append("")
    else:
        lines.extend(["- None.", ""])

    lines.extend(["## Meta", ""])
    meta_commentary = (finalize_data.get("meta_commentary") or "").strip()
    lines.append(meta_commentary or "None.")
    lines.append("")

    if gap_counts:
        lines.extend(["## Coverage Gaps", ""])
        for label, count in gap_counts.items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def config_dir(home: Path | None = None) -> Path:
    if home is None:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "megaplan"
        home = Path.home()
    return home / ".config" / "megaplan"


def load_config(home: Path | None = None) -> dict[str, Any]:
    path = config_dir(home) / "config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        import sys
        print(f"megaplan: warning: ignoring malformed config at {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_config(config: dict[str, Any], home: Path | None = None) -> Path:
    path = config_dir(home) / "config.json"
    atomic_write_json(path, config)
    return path


def get_effective(section: str, key: str) -> Any:
    from megaplan.types import DEFAULTS

    default_key = f"{section}.{key}"
    if default_key not in DEFAULTS:
        raise KeyError(default_key)
    config = load_config()
    section_config = config.get(section)
    if isinstance(section_config, dict) and key in section_config:
        return section_config[key]
    return DEFAULTS[default_key]


def detect_available_agents() -> list[str]:
    # Access shutil via the _core package so monkeypatches on megaplan._core.shutil work.
    import megaplan._core as _core_pkg
    _shutil_ref = _core_pkg.shutil
    available = [a for a in KNOWN_AGENTS if a != "hermes" and _shutil_ref.which(a)]
    # Hermes is a Python library, not a CLI binary
    try:
        import run_agent  # noqa: F401
        available.append("hermes")
    except ImportError:
        pass
    return available


# ---------------------------------------------------------------------------
# Runtime layout / path helpers
# ---------------------------------------------------------------------------

def ensure_runtime_layout(root: Path) -> None:
    megaplan_rt = root / ".megaplan"
    (megaplan_rt / "plans").mkdir(parents=True, exist_ok=True)
    schemas_dir = megaplan_rt / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in SCHEMAS.items():
        atomic_write_json(schemas_dir / filename, strict_schema(schema))


def megaplan_root(root: Path) -> Path:
    return root / ".megaplan"


def plans_root(root: Path) -> Path:
    return megaplan_root(root) / "plans"


def schemas_root(root: Path) -> Path:
    return megaplan_root(root) / "schemas"


def artifact_path(plan_dir: Path, filename: str) -> Path:
    return plan_dir / filename


def batch_artifact_path(plan_dir: Path, batch_number: int) -> Path:
    return plan_dir / f"execution_batch_{batch_number}.json"


def list_batch_artifacts(plan_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"execution_batch_(\d+)\.json", path.name)
        if match is None:
            raise ValueError(f"Unexpected batch artifact filename: {path.name}")
        return (int(match.group(1)), path.name)

    return sorted(
        (
            path
            for path in plan_dir.glob("execution_batch_*.json")
            if path.is_file() and re.fullmatch(r"execution_batch_(\d+)\.json", path.name)
        ),
        key=sort_key,
    )


def current_iteration_artifact(plan_dir: Path, prefix: str, iteration: int) -> Path:
    return plan_dir / f"{prefix}_v{iteration}.json"


def current_iteration_raw_artifact(plan_dir: Path, prefix: str, iteration: int) -> Path:
    return plan_dir / f"{prefix}_v{iteration}_raw.txt"


# ---------------------------------------------------------------------------
# Git diff summary (used by prompts)
# ---------------------------------------------------------------------------

import subprocess


def collect_git_diff_summary(project_dir: Path) -> str:
    if not (project_dir / ".git").exists():
        return "Project directory is not a git repository."
    try:
        process = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(project_dir),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "git not found on PATH."
    except subprocess.TimeoutExpired:
        return "git status timed out."
    if process.returncode != 0:
        return f"Unable to read git status: {process.stderr.strip() or process.stdout.strip()}"
    return process.stdout.strip() or "No git changes detected."


def find_command(name: str) -> str | None:
    import megaplan._core as _core_pkg
    return _core_pkg.shutil.which(name)
