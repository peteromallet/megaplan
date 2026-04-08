from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from megaplan._core import load_flag_registry, save_flag_registry
from megaplan.types import FlagRecord, FlagRegistry


def next_flag_number(flags: list[FlagRecord]) -> int:
    highest = 0
    for flag in flags:
        match = re.fullmatch(r"FLAG-(\d+)", flag["id"])
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def make_flag_id(number: int) -> str:
    return f"FLAG-{number:03d}"


def resolve_severity(hint: str) -> str:
    if hint == "likely-significant":
        return "significant"
    if hint == "likely-minor":
        return "minor"
    if hint == "uncertain":
        return "significant"
    logging.getLogger("megaplan").warning(f"Unexpected severity_hint: {hint!r}, defaulting to significant")
    return "significant"


def normalize_flag_record(raw_flag: dict[str, Any], fallback_id: str) -> FlagRecord:
    category = raw_flag.get("category", "other")
    if category not in {"correctness", "security", "completeness", "performance", "maintainability", "other"}:
        category = "other"
    severity_hint = raw_flag.get("severity_hint") or "uncertain"
    if severity_hint not in {"likely-significant", "likely-minor", "uncertain"}:
        severity_hint = "uncertain"
    raw_id = raw_flag.get("id")
    return {
        "id": fallback_id if raw_id in {None, "", "FLAG-000"} else raw_id,
        "concern": raw_flag.get("concern", "").strip(),
        "category": category,
        "severity_hint": severity_hint,
        "evidence": raw_flag.get("evidence", "").strip(),
    }


def _review_flag_id(check_id: str, index: int) -> str:
    stem = re.sub(r"[^A-Z0-9]+", "_", check_id.upper()).strip("_") or "CHECK"
    return f"REVIEW-{stem}-{index:03d}"


def _synthesize_flags_from_checks(
    checks: list[dict[str, Any]],
    *,
    category_map: dict[str, str],
    get_check_def: Callable[[str], Any],
    id_prefix: str,
) -> list[dict[str, Any]]:
    synthetic_flags: list[dict[str, Any]] = []
    for check in checks:
        check_id = check.get("id", "")
        if not isinstance(check_id, str) or not check_id:
            continue
        flagged_findings = [
            finding
            for finding in check.get("findings", [])
            if isinstance(finding, dict) and finding.get("flagged")
        ]
        for index, finding in enumerate(flagged_findings, start=1):
            check_def = get_check_def(check_id)
            if isinstance(check_def, dict):
                severity = check_def.get("default_severity", "uncertain")
            else:
                severity = getattr(check_def, "default_severity", "uncertain")
            if id_prefix == "REVIEW":
                flag_id = _review_flag_id(check_id, index)
            else:
                flag_id = check_id if len(flagged_findings) == 1 else f"{check_id}-{index}"
            synthetic_flags.append(
                {
                    "id": flag_id,
                    "concern": f"{check.get('question', '')}: {finding.get('detail', '')}",
                    "category": category_map.get(check_id, "correctness"),
                    "severity_hint": severity,
                    "evidence": finding.get("detail", ""),
                }
            )
    return synthetic_flags


def _apply_flag_updates(
    payload: dict[str, Any],
    *,
    plan_dir: Path,
    iteration: int,
    artifact_prefix: str,
) -> FlagRegistry:
    registry = load_flag_registry(plan_dir)
    flags = registry.setdefault("flags", [])
    by_id: dict[str, FlagRecord] = {flag["id"]: flag for flag in flags}
    next_number = next_flag_number(flags)

    for verified_id in payload.get("verified_flag_ids", []):
        if verified_id in by_id:
            by_id[verified_id]["status"] = "verified"
            by_id[verified_id]["verified"] = True
            by_id[verified_id]["verified_in"] = f"{artifact_prefix}_v{iteration}.json"

    for disputed_id in payload.get("disputed_flag_ids", []):
        if disputed_id in by_id:
            by_id[disputed_id]["status"] = "disputed"

    for raw_flag in payload.get("flags", []):
        proposed_id = raw_flag.get("id")
        if not proposed_id or proposed_id in {"", "FLAG-000"}:
            proposed_id = make_flag_id(next_number)
            next_number += 1
        normalized = normalize_flag_record(raw_flag, proposed_id)
        if normalized["id"] in by_id:
            existing = by_id[normalized["id"]]
            existing.update(normalized)
            existing["status"] = "open"
            existing["severity"] = resolve_severity(normalized.get("severity_hint", "uncertain"))
            existing["raised_in"] = f"{artifact_prefix}_v{iteration}.json"
            continue
        severity = resolve_severity(normalized.get("severity_hint", "uncertain"))
        created: FlagRecord = {
            **normalized,
            "raised_in": f"{artifact_prefix}_v{iteration}.json",
            "status": "open",
            "severity": severity,
            "verified": False,
        }
        flags.append(created)
        by_id[created["id"]] = created

    save_flag_registry(plan_dir, registry)
    return registry


def update_flags_after_critique(plan_dir: Path, critique: dict[str, Any], *, iteration: int) -> FlagRegistry:
    from megaplan.checks import build_check_category_map, get_check_by_id

    critique.setdefault("flags", []).extend(
        _synthesize_flags_from_checks(
            critique.get("checks", []),
            category_map=build_check_category_map(),
            get_check_def=get_check_by_id,
            id_prefix="CRITIQUE",
        )
    )
    return _apply_flag_updates(critique, plan_dir=plan_dir, iteration=iteration, artifact_prefix="critique")


def update_flags_after_review(plan_dir: Path, review_payload: dict[str, Any], *, iteration: int) -> FlagRegistry:
    from megaplan.review_checks import build_check_category_map, get_check_by_id

    payload_for_registry = dict(review_payload)
    payload_for_registry["flags"] = [*list(review_payload.get("flags", [])), *(
        _synthesize_flags_from_checks(
            review_payload.get("checks", []),
            category_map=build_check_category_map(),
            get_check_def=get_check_by_id,
            id_prefix="REVIEW",
        )
    )]
    return _apply_flag_updates(payload_for_registry, plan_dir=plan_dir, iteration=iteration, artifact_prefix="review")


def update_flags_after_revise(
    plan_dir: Path,
    flags_addressed: list[str],
    *,
    plan_file: str,
    summary: str,
) -> FlagRegistry:
    registry = load_flag_registry(plan_dir)
    for flag in registry["flags"]:
        if flag["id"] in flags_addressed:
            flag["status"] = "addressed"
            flag["addressed_in"] = plan_file
            flag["evidence"] = summary
    save_flag_registry(plan_dir, registry)
    return registry


def update_flags_after_gate(
    plan_dir: Path,
    resolutions: list[dict[str, Any]],
) -> FlagRegistry:
    """Persist flag status changes from validated gate resolutions."""
    registry = load_flag_registry(plan_dir)
    by_id: dict[str, FlagRecord] = {flag["id"]: flag for flag in registry["flags"]}
    for res in resolutions:
        flag_id = res.get("flag_id", "")
        action = res.get("action", "")
        if flag_id not in by_id:
            continue
        if action == "dispute":
            by_id[flag_id]["status"] = "gate_disputed"
        elif action == "accept_tradeoff":
            by_id[flag_id]["status"] = "accepted_tradeoff"
    save_flag_registry(plan_dir, registry)
    return registry
