"""Gate-signal scoring and loop diagnostics."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from megaplan.types import (
    FLAG_BLOCKING_STATUSES,
    FlagRecord,
    GateArtifact,
    GateCheckResult,
    GatePayload,
    GateSignals,
    PlanState,
)
from megaplan._core import (
    configured_robustness,
    current_iteration_artifact,
    latest_plan_meta_path,
    latest_plan_path,
    load_flag_registry,
    normalize_text,
    read_json,
    scope_creep_flags,
    unresolved_significant_flags,
)


PLAN_STRUCTURE_REQUIRED_STEP_ISSUE = "Plan must include at least one step section (`## Step N:` or `### Step N:` under a phase)."
_PLAN_HEADING_RE = re.compile(r"^##\s+.+$")
_PLAN_PHASE_HEADING_RE = re.compile(r"^###\s+.+$")
_PLAN_STEP_RE = re.compile(r"^##\s+Step\s+(\d+):\s+.+$")
_PLAN_PHASE_STEP_RE = re.compile(r"^###\s+Step\s+(\d+):\s+.+$")
_GENERIC_ACKS = {
    "ack",
    "checked",
    "confirmed",
    "done",
    "good",
    "looks good",
    "n/a",
    "na",
    "ok",
    "verified",
    "yes",
}


@dataclass(frozen=True)
class PlanSection:
    heading: str
    body: str
    id: str | None
    start_line: int
    end_line: int


def _normalize_repo_path(path: str) -> str:
    return Path(path.strip()).as_posix()


def _parse_git_status_paths(stdout: str) -> set[str]:
    paths: set[str] = set()
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        path_text = raw_line[3:].strip() if len(raw_line) >= 4 else raw_line.strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        cleaned = path_text.strip().strip('"')
        if cleaned:
            paths.add(_normalize_repo_path(cleaned))
    return paths


def _is_perfunctory_ack(note: str) -> bool:
    normalized = normalize_text(note)
    return len(note.strip()) < 10 or normalized in _GENERIC_ACKS


def validate_execution_evidence(finalize_data: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    findings: list[str] = []
    files_claimed = sorted(
        {
            _normalize_repo_path(path)
            for task in finalize_data.get("tasks", [])
            for path in task.get("files_changed", [])
            if isinstance(path, str) and path.strip()
        }
    )

    if not (project_dir / ".git").exists():
        return {
            "findings": findings,
            "files_in_diff": [],
            "files_claimed": files_claimed,
            "skipped": True,
            "reason": "Project directory is not a git repository.",
        }

    try:
        process = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(project_dir),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        return {
            "findings": findings,
            "files_in_diff": [],
            "files_claimed": files_claimed,
            "skipped": True,
            "reason": "git not found on PATH.",
        }
    except subprocess.TimeoutExpired:
        return {
            "findings": findings,
            "files_in_diff": [],
            "files_claimed": files_claimed,
            "skipped": True,
            "reason": "git status timed out.",
        }

    if process.returncode != 0:
        return {
            "findings": findings,
            "files_in_diff": [],
            "files_claimed": files_claimed,
            "skipped": True,
            "reason": f"git status failed: {process.stderr.strip() or process.stdout.strip()}",
        }

    files_in_diff = sorted(_parse_git_status_paths(process.stdout))
    claimed_set = set(files_claimed)
    diff_set = set(files_in_diff)

    phantom_claims = sorted(claimed_set - diff_set)
    if phantom_claims:
        findings.append(
            "Executor claimed changed files not present in git status: "
            + ", ".join(phantom_claims)
        )

    unclaimed_changes = sorted(diff_set - claimed_set)
    if unclaimed_changes:
        findings.append(
            "Git status shows changed files not claimed by any task: "
            + ", ".join(unclaimed_changes)
        )

    for sense_check in finalize_data.get("sense_checks", []):
        sense_check_id = sense_check.get("id", "?")
        note = sense_check.get("executor_note", "")
        if not isinstance(note, str) or not note.strip():
            findings.append(f"Sense check {sense_check_id} is missing an executor acknowledgment.")
            continue
        if _is_perfunctory_ack(note):
            findings.append(
                f"Sense check {sense_check_id} acknowledgment is perfunctory: {note.strip()!r}."
            )

    return {
        "findings": findings,
        "files_in_diff": files_in_diff,
        "files_claimed": files_claimed,
        "skipped": False,
        "reason": "",
    }


def flag_weight(flag: FlagRecord) -> float:
    """Weight a flag for gate context. Higher = more blocking."""
    category = flag.get("category", "other")
    concern = flag.get("concern", "").lower()

    if category == "security":
        return 3.0

    implementation_detail_signals = [
        "column",
        "schema",
        "field",
        "as written",
        "pseudocode",
        "seed sql",
        "placeholder",
    ]
    if any(signal in concern for signal in implementation_detail_signals):
        return 0.5

    weights = {
        "correctness": 2.0,
        "completeness": 1.5,
        "performance": 1.0,
        "maintainability": 0.75,
        "other": 1.0,
    }
    return weights.get(category, 1.0)


def compute_plan_delta_percent(previous_text: str | None, current_text: str) -> float | None:
    if previous_text is None:
        return None
    ratio = SequenceMatcher(None, previous_text, current_text).ratio()
    return round((1.0 - ratio) * 100.0, 2)


def compute_recurring_critiques(plan_dir: Path, iteration: int) -> list[str]:
    if iteration < 2:
        return []
    previous = read_json(current_iteration_artifact(plan_dir, "critique", iteration - 1))
    current = read_json(current_iteration_artifact(plan_dir, "critique", iteration))
    previous_concerns = {normalize_text(flag["concern"]) for flag in previous.get("flags", [])}
    current_concerns = {normalize_text(flag["concern"]) for flag in current.get("flags", [])}
    return sorted(previous_concerns.intersection(current_concerns))


def _strip_fenced_blocks(text: str) -> str:
    kept_lines: list[str] = []
    inside_fence = False
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            kept_lines.append(line)
    if inside_fence:
        # Unclosed fence — return original text rather than silently dropping content
        return text
    return "".join(kept_lines)


def _match_section_boundary(line: str) -> tuple[bool, str | None]:
    """Check if a line is a section boundary. Returns (is_boundary, section_id)."""
    step_match = _PLAN_STEP_RE.match(line) or _PLAN_PHASE_STEP_RE.match(line)
    if step_match:
        return True, f"S{step_match.group(1)}"
    if _PLAN_HEADING_RE.match(line) or _PLAN_PHASE_HEADING_RE.match(line):
        return True, None
    return False, None


def parse_plan_sections(plan_text: str) -> list[PlanSection]:
    lines = plan_text.splitlines(keepends=True)
    if not lines:
        return [PlanSection(heading="", body="", id=None, start_line=1, end_line=0)]

    boundaries: list[tuple[int, int, str, str | None]] = []
    inside_fence = False
    fence_open_line = -1
    for index, line in enumerate(lines):
        if line.startswith("```"):
            if not inside_fence:
                fence_open_line = index
            inside_fence = not inside_fence
            continue
        if inside_fence:
            continue
        is_boundary, section_id = _match_section_boundary(line)
        if is_boundary:
            boundaries.append((index, index + 1, line.rstrip("\n"), section_id))

    if inside_fence:
        # Unclosed fence — re-scan ignoring fence state so we don't silently lose sections
        boundaries = []
        for index, line in enumerate(lines):
            is_boundary, section_id = _match_section_boundary(line)
            if is_boundary:
                boundaries.append((index, index + 1, line.rstrip("\n"), section_id))

    if not boundaries:
        return [PlanSection(heading="", body=plan_text, id=None, start_line=1, end_line=len(lines))]

    sections: list[PlanSection] = []
    first_index, first_line, _, _ = boundaries[0]
    if first_index > 0:
        sections.append(
            PlanSection(
                heading="",
                body="".join(lines[:first_index]),
                id=None,
                start_line=1,
                end_line=first_line - 1,
            )
        )

    for boundary_index, (start_index, start_line, heading, section_id) in enumerate(boundaries):
        next_start_index = boundaries[boundary_index + 1][0] if boundary_index + 1 < len(boundaries) else len(lines)
        sections.append(
            PlanSection(
                heading=heading,
                body="".join(lines[start_index:next_start_index]),
                id=section_id,
                start_line=start_line,
                end_line=next_start_index,
            )
        )
    return sections


def reassemble_plan(sections: list[PlanSection]) -> str:
    return "".join(section.body for section in sections)


def renumber_steps(sections: list[PlanSection]) -> list[PlanSection]:
    renumbered: list[PlanSection] = []
    step_number = 1
    for section in sections:
        if section.id is None:
            renumbered.append(section)
            continue
        # Detect heading level (## or ###) and preserve it
        step_prefix_match = re.match(r"^(#{2,3})\s+Step\s+\d+:", section.heading)
        if not step_prefix_match:
            renumbered.append(section)
            continue
        hashes = step_prefix_match.group(1)
        new_heading = re.sub(rf"^{hashes}\s+Step\s+\d+:", f"{hashes} Step {step_number}:", section.heading, count=1)
        new_body = re.sub(rf"^{hashes}\s+Step\s+\d+:", f"{hashes} Step {step_number}:", section.body, count=1, flags=re.MULTILINE)
        renumbered.append(
            PlanSection(
                heading=new_heading,
                body=new_body,
                id=f"S{step_number}",
                start_line=section.start_line,
                end_line=section.end_line,
            )
        )
        step_number += 1
    return renumbered


def validate_plan_structure(plan_text: str) -> list[str]:
    issues: list[str] = []
    stripped = _strip_fenced_blocks(plan_text)

    if len(re.findall(r"(?mi)^#\s+.+$", stripped)) != 1:
        issues.append("Plan should have exactly one H1 title.")
    if not re.search(r"(?mi)^##\s+Overview\s*$", stripped):
        issues.append("Plan should include a `## Overview` section.")

    # Accept both flat (## Step N:) and hierarchical (### Step N: under ## Phase)
    step_matches = list(re.finditer(r"(?im)^#{2,3}\s+Step\s+\d+:\s+.+$", stripped))
    if not step_matches:
        issues.append(PLAN_STRUCTURE_REQUIRED_STEP_ISSUE)
        return issues

    if not (
        re.search(r"(?mi)^##\s+Execution Order\s*$", stripped)
        or re.search(r"(?mi)^##\s+Validation Order\s*$", stripped)
    ):
        issues.append("Plan should include `## Execution Order` or `## Validation Order`.")

    missing_substeps = False
    missing_file_refs = False
    for index, match in enumerate(step_matches):
        start = match.end()
        next_heading = re.search(r"(?im)^#{2,3}\s+.+$", stripped[start:])
        end = start + next_heading.start() if next_heading else len(stripped)
        section = stripped[match.start():end]
        if not re.search(r"(?m)^\d+\.\s+", stripped[start:end]):
            missing_substeps = True
        if not re.search(r"`[^`]+`", section):
            missing_file_refs = True

    if missing_substeps:
        issues.append("Each step section should include at least one numbered substep.")
    if missing_file_refs:
        issues.append("Each step section should reference at least one file in backticks.")
    return issues


def _previous_iteration_plan_path(plan_dir: Path, state: PlanState) -> Path | None:
    current_version = state["iteration"]
    previous_version = current_version - 1
    if previous_version < 1:
        return None
    matching = [
        record
        for record in state["plan_versions"]
        if record.get("version") == previous_version
    ]
    if not matching:
        return None
    return plan_dir / matching[-1]["file"]


def build_gate_signals(plan_dir: Path, state: PlanState) -> GateSignals:
    iteration = state["iteration"]
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    robustness = configured_robustness(state)
    open_scope_creep = scope_creep_flags(flag_registry, statuses=FLAG_BLOCKING_STATUSES)
    significant_count = len(
        [
            flag
            for flag in flag_registry["flags"]
            if flag.get("severity") == "significant" and flag["status"] != "verified"
        ]
    )
    weighted_score = round(sum(flag_weight(flag) for flag in unresolved), 2)
    weighted_history = list(state["meta"].get("weighted_scores", []))
    latest_plan_text = latest_plan_path(plan_dir, state).read_text(encoding="utf-8")
    previous_plan_path = _previous_iteration_plan_path(plan_dir, state)
    previous_text = None
    if previous_plan_path is not None and previous_plan_path.exists():
        previous_text = previous_plan_path.read_text(encoding="utf-8")
    plan_delta = compute_plan_delta_percent(previous_text, latest_plan_text)
    recurring = compute_recurring_critiques(plan_dir, iteration)
    resolved_flags = [
        {
            "id": flag["id"],
            "concern": flag["concern"],
            "resolution": flag.get("evidence", ""),
        }
        for flag in flag_registry["flags"]
        if flag["status"] == "verified"
    ]

    delta_history = state["meta"].get("plan_deltas", [])
    if weighted_history:
        trajectory = " -> ".join(str(score) for score in weighted_history) + f" -> {weighted_score}"
    else:
        trajectory = str(weighted_score)
    delta_summary = ", ".join(
        "n/a" if delta is None else f"{delta:.1f}%"
        for delta in delta_history
    ) or "n/a"
    loop_summary = (
        f"Iteration {iteration}. Weighted score trajectory: {trajectory}. "
        f"Plan deltas: {delta_summary}. "
        f"Recurring critiques: {len(recurring)}. "
        f"Resolved flags: {len(resolved_flags)}. "
        f"Open significant flags: {len(unresolved)}."
    )

    result: GateSignals = {
        "robustness": robustness,
        "signals": {
            "iteration": iteration,
            "idea": state.get("idea", ""),
            "significant_flags": significant_count,
            "unresolved_flags": [
                {
                    "id": flag["id"],
                    "concern": flag["concern"],
                    "category": flag["category"],
                    "severity": flag.get("severity", "unknown"),
                    "status": flag["status"],
                }
                for flag in unresolved
            ],
            "resolved_flags": resolved_flags,
            "weighted_score": weighted_score,
            "weighted_history": weighted_history,
            "plan_delta_from_previous": plan_delta,
            "recurring_critiques": recurring,
            "scope_creep_flags": [flag["id"] for flag in open_scope_creep],
            "loop_summary": loop_summary,
        },
        "warnings": [],
    }
    if open_scope_creep:
        result["warnings"].append(
            "Scope creep detected: the plan appears to be expanding beyond the original idea or recorded user notes."
        )
    if iteration >= 5:
        result["warnings"].append(f"Iteration {iteration}: high iteration count.")
    if iteration >= 12:
        result["warnings"].append(
            f"Iteration {iteration}: hard iteration limit reached. Escalation is likely warranted."
        )
    return result


def run_gate_checks(
    plan_dir: Path,
    state: PlanState,
    *,
    command_lookup: Callable[[str], str | None] | None = None,
) -> GateCheckResult:
    project_dir = Path(state["config"]["project_dir"])
    meta = read_json(latest_plan_meta_path(plan_dir, state))
    flag_registry = load_flag_registry(plan_dir)
    unresolved = unresolved_significant_flags(flag_registry)
    lookup = command_lookup or (lambda name: None)
    checks = {
        "project_dir_exists": project_dir.exists(),
        "project_dir_writable": os.access(project_dir, os.W_OK),
        "success_criteria_present": bool(meta.get("success_criteria")),
        "claude_available": bool(lookup("claude")),
        "codex_available": bool(lookup("codex")),
    }
    return {
        "passed": all(checks.values()),
        "criteria_check": {
            "count": len(meta.get("success_criteria", [])),
            "items": meta.get("success_criteria", []),
        },
        "preflight_results": checks,
        "unresolved_flags": unresolved,
    }


def build_gate_artifact(
    signals: dict[str, Any],
    gate_payload: GatePayload,
    *,
    override_forced: bool,
    orchestrator_guidance: str = "",
) -> GateArtifact:
    preflight = signals["preflight_results"]
    recommendation = gate_payload["recommendation"]
    warnings = list(signals.get("warnings", [])) + list(gate_payload.get("warnings", []))
    return {
        "passed": recommendation == "PROCEED" and all(preflight.values()),
        "criteria_check": signals["criteria_check"],
        "preflight_results": preflight,
        "unresolved_flags": signals["unresolved_flags"],
        "recommendation": recommendation,
        "rationale": gate_payload["rationale"],
        "signals_assessment": gate_payload["signals_assessment"],
        "warnings": warnings,
        "settled_decisions": list(gate_payload.get("settled_decisions", [])),
        "override_forced": override_forced,
        "orchestrator_guidance": orchestrator_guidance,
        "robustness": signals.get("robustness"),
        "signals": signals["signals"],
    }


def build_orchestrator_guidance(
    gate_payload: GatePayload,
    signals: dict[str, Any],
    preflight_passed: bool,
    preflight_results: dict[str, bool],
    robustness: str,
    plan_name: str,
) -> str:
    """Return plain-language next-step guidance for the orchestrator."""
    recommendation = gate_payload["recommendation"]
    iteration = int(signals.get("iteration", 0))
    weighted_score = float(signals.get("weighted_score", 0.0))
    weighted_history = list(signals.get("weighted_history", []))
    recurring_critiques = list(signals.get("recurring_critiques", []))
    unresolved_flags = list(signals.get("unresolved_flags", []))
    scope_creep = list(signals.get("scope_creep_flags", []))
    previous_score = float(weighted_history[-1]) if weighted_history else None
    plateaued = previous_score is not None and weighted_score >= previous_score
    worsening = previous_score is not None and weighted_score > previous_score
    improving = previous_score is not None and weighted_score < previous_score

    if iteration == 1:
        guidance = f"First iteration; follow gate recommendation: {recommendation}."
    elif recommendation == "PROCEED" and preflight_passed:
        guidance = "Plan passed gate and preflight. Proceed to finalize."
    elif recommendation == "PROCEED":
        failing_checks = ", ".join(
            name for name, passed in preflight_results.items() if not passed
        )
        guidance = f"Gate says PROCEED but preflight blocked. Fix: {failing_checks}."
    elif recommendation == "ESCALATE" and robustness == "light" and weighted_score <= 4.0:
        guidance = (
            "Auto-force-proceed eligible. Run: "
            f'`megaplan override force-proceed --plan {plan_name} --reason "light robustness, score {weighted_score}"`'
        )
    elif recommendation == "ESCALATE":
        guidance = "Gate escalated. Ask the user: force-proceed, add-note, or abort."
    elif recommendation == "ITERATE" and plateaued and recurring_critiques:
        guidance = (
            "Score plateaued with recurring critiques the loop can't fix. Consider "
            f"force-proceeding: `megaplan override force-proceed --plan {plan_name}`"
        )
    elif recommendation == "ITERATE" and improving:
        guidance = f"Score improving ({previous_score} -> {weighted_score}). Continue to revise."
    elif recommendation == "ITERATE" and worsening:
        guidance = (
            f"Score worsening ({previous_score} -> {weighted_score}). "
            "Investigate; the loop may be diverging."
        )
    else:
        guidance = "Gate recommends another iteration. Revise the plan."

    hints: list[str] = []
    if unresolved_flags:
        hints.append("Verify unresolved flags against the plan and project code before accepting.")
    if recurring_critiques:
        critiques = ", ".join(recurring_critiques)
        hints.append(
            f"Recurring critiques ({critiques}); the loop likely can't fix these, so judge if they are real blockers."
        )
    if scope_creep:
        hints.append("Scope creep detected; compare the current plan against the original idea.")

    return " ".join([guidance, *hints]).strip()
