"""Critique check registry and helpers."""

from __future__ import annotations

from typing import Any, Final, TypedDict


VALID_SEVERITY_HINTS: Final[set[str]] = {"likely-significant", "likely-minor", "uncertain"}


class CritiqueCheckSpec(TypedDict):
    id: str
    question: str
    guidance: str
    category: str
    default_severity: str


CRITIQUE_CHECKS: Final[tuple[CritiqueCheckSpec, ...]] = (
    {
        "id": "issue_hints",
        "question": "Did the work fully address the issue hints, user notes, and approved plan requirements?",
        "guidance": (
            "Cross-check the result against explicit user notes, critique corrections, and watch items. "
            "Flag anything the implementation ignored, contradicted, or only partially covered."
        ),
        "category": "completeness",
        "default_severity": "likely-significant",
    },
    {
        "id": "correctness",
        "question": "Are the proposed changes technically correct?",
        "guidance": (
            "Look for logic errors, invalid assumptions, broken invariants, schema mismatches, "
            "or behavior that would fail at runtime. When the fix adds a conditional branch, "
            "check whether it handles all relevant cases — not just the one reported in the issue."
        ),
        "category": "correctness",
        "default_severity": "likely-significant",
    },
    {
        "id": "scope",
        "question": "Is the scope and scale of the change appropriate?",
        "guidance": (
            "Flag missing required work, out-of-scope edits, or changes that expand behavior without "
            "clear justification. Also ask: is a minimal patch the right scale, or does this issue "
            "need new code, new methods, or restructuring?"
        ),
        "category": "completeness",
        "default_severity": "likely-significant",
    },
    {
        "id": "all_locations",
        "question": "Does the change touch all locations AND supporting infrastructure?",
        "guidance": (
            "Search for all instances of the symbol/pattern being changed. Also ask: does this "
            "feature require setup, registration, or integration code beyond the core logic? "
            "Missing glue code causes test failures even when the core fix is correct."
        ),
        "category": "completeness",
        "default_severity": "likely-significant",
    },
    {
        "id": "callers",
        "question": "Is the change in the right place, and would it break any callers?",
        "guidance": (
            "First ask: should this change be HERE, or in a caller, callee, or new method? "
            "Then check function signatures, return types, and any code that calls or depends on "
            "the changed function."
        ),
        "category": "correctness",
        "default_severity": "likely-significant",
    },
    {
        "id": "conventions",
        "question": "Does the approach match how the codebase solves similar problems?",
        "guidance": (
            "Check not just naming/style but how similar PROBLEMS are solved in this codebase. "
            "If the codebase adds new methods for similar cases, the plan should too. "
            "Do not spend findings on trivial stylistic preferences."
        ),
        "category": "maintainability",
        "default_severity": "likely-minor",
    },
    {
        "id": "verification",
        "question": "Is there convincing verification for the change?",
        "guidance": (
            "Flag missing tests or weak validation. If verification tests exist, trace the test's "
            "execution path through your patch — does every branch it exercises produce the expected "
            "result? A patch can look correct but fail because it misses one code path the test covers. "
            "If you manually verify an edge case because existing tests don't cover it, also test the cases next to it."
        ),
        "category": "completeness",
        "default_severity": "uncertain",
    },
    {
        "id": "criteria_quality",
        "question": "Are the success criteria well-prioritized and verifiable?",
        "guidance": (
            "Check that each `must` criterion has a clear yes/no answer verifiable from code, tests, or "
            "git diff. Subjective goals, numeric guidelines, and aspirational targets should be `should`, "
            "not `must`. Criteria requiring manual testing or human judgment should be `info`. "
            "Flag any `must` criterion that is ambiguous, subjective, or unverifiable in the review pipeline."
        ),
        "category": "completeness",
        "default_severity": "likely-significant",
    },
)

_CHECK_BY_ID: Final[dict[str, CritiqueCheckSpec]] = {check["id"]: check for check in CRITIQUE_CHECKS}


def get_check_ids() -> list[str]:
    return [check["id"] for check in CRITIQUE_CHECKS]


def get_check_by_id(check_id: str) -> CritiqueCheckSpec | None:
    return _CHECK_BY_ID.get(check_id)


def build_check_category_map() -> dict[str, str]:
    return {check["id"]: check["category"] for check in CRITIQUE_CHECKS}


def build_empty_template() -> list[dict[str, Any]]:
    return [
        {
            "id": check["id"],
            "question": check["question"],
            "findings": [],
        }
        for check in CRITIQUE_CHECKS
    ]


def _valid_findings(findings: Any) -> bool:
    if not isinstance(findings, list) or not findings:
        return False
    for finding in findings:
        if not isinstance(finding, dict):
            return False
        detail = finding.get("detail")
        flagged = finding.get("flagged")
        if not isinstance(detail, str) or not detail.strip():
            return False
        if not isinstance(flagged, bool):
            return False
    return True


def validate_critique_checks(payload: Any) -> list[str]:
    raw_checks = payload.get("checks") if isinstance(payload, dict) else payload
    if not isinstance(raw_checks, list):
        return get_check_ids()

    expected_ids = get_check_ids()
    expected_set = set(expected_ids)
    valid_ids: set[str] = set()
    invalid_expected_ids: set[str] = set()
    invalid_unknown_ids: set[str] = set()
    seen_counts: dict[str, int] = {}

    for raw_check in raw_checks:
        if not isinstance(raw_check, dict):
            continue
        check_id = raw_check.get("id")
        if not isinstance(check_id, str) or not check_id:
            continue

        seen_counts[check_id] = seen_counts.get(check_id, 0) + 1
        if check_id not in expected_set:
            invalid_unknown_ids.add(check_id)
            continue
        if seen_counts[check_id] > 1:
            invalid_expected_ids.add(check_id)
            continue

        question = raw_check.get("question")
        findings = raw_check.get("findings")
        if not isinstance(question, str) or not question.strip():
            invalid_expected_ids.add(check_id)
            continue
        if not _valid_findings(findings):
            invalid_expected_ids.add(check_id)
            continue
        valid_ids.add(check_id)

    return [
        check_id
        for check_id in expected_ids
        if check_id not in valid_ids or check_id in invalid_expected_ids
    ] + sorted(invalid_unknown_ids)


__all__ = [
    "CRITIQUE_CHECKS",
    "VALID_SEVERITY_HINTS",
    "build_check_category_map",
    "build_empty_template",
    "get_check_by_id",
    "get_check_ids",
    "validate_critique_checks",
]
