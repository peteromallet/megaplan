"""Review check registry and helpers.

This largely mirrors :mod:`megaplan.checks`; the shared helper body is an
extraction candidate once critique and review settle on the same abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True)
class ReviewCheckSpec:
    id: str
    question: str
    guidance: str
    category: str
    default_severity: str
    tier: str


REVIEW_CHECKS: Final[tuple[ReviewCheckSpec, ...]] = (
    ReviewCheckSpec(
        id="coverage",
        question="Does the diff cover every concrete failing example, symptom, or 'X should Y' statement in the original issue?",
        guidance=(
            "List each concrete failing example, symptom, or behavioral requirement in the issue. "
            "For each one, cite the diff line that addresses it and flag anything still uncovered. "
            "Also watch for: "
            "(a) Opt-in vs default: if the issue wants new default behavior but the diff adds a flag, flag it. "
            "(b) Accepted-change violations: if the issue accepts a behavior change, flag code that preserves the old behavior anyway. "
            "(c) Secondary sentences: treat extra expectations in side phrases as real requirements. "
            "Do NOT flag a verification test referenced by the plan that doesn't exist in the repo as a coverage gap. "
            "Test files are a task-level no-modify structural constraint already accepted by the gate; if it still matters, downgrade to status: significant. "
            "Focus blocking coverage findings on source changes that miss concrete issue symptoms."
        ),
        category="completeness",
        default_severity="likely-significant",
        tier="core",
    ),
    ReviewCheckSpec(
        id="placement",
        question="Is the fix placed where the bad state originates, or only where the symptom surfaces?",
        guidance=(
            "Do a backward trace, not just a forward one. Start from the changed code and ask: where is the bad value first introduced? "
            "If your fix runs AFTER the bad state already exists, you are papering over a downstream symptom — flag it, even if the symptom from the issue goes away. "
            "Then identify at least two distinct code paths by which a user could trigger this bug (e.g., direct function call, alternate entry point, reflection/registry lookup, import-time evaluation). "
            "Does your fix block ALL of them, or only the one the issue demonstrates? If only one, flag it. "
            "If the issue involves parsing, serialization, or transformation, verify the fix is at the source of the bad representation, not a downstream consumer that compensates for it. "
            "Ask: would a maintainer reviewing this PR expect the fix to live here, or somewhere more fundamental?"
        ),
        category="correctness",
        default_severity="likely-significant",
        tier="core",
    ),
    ReviewCheckSpec(
        id="adjacent_calls",
        question="Do adjacent callers, sibling usage sites, or downstream consumers still exhibit the bug or mishandle the new values?",
        guidance=(
            "First: look at other code that calls the changed function, uses the same surrounding pattern, or is registered alongside the changed class. "
            "For each sibling site, check whether the reported bug would still affect it and flag any adjacent cases left unfixed. "
            "Specifically: if the patch adds a new class/handler alongside existing siblings, enumerate every method those siblings override and ask whether ours should override them too. "
            "Second: for any value the patch returns or constructs in a new/error path, trace where it flows next. "
            "Check whether downstream consumers actually handle that value correctly, and whether any new sentinel or return-value semantics introduce inconsistencies. "
            "Flag anything a maintainer would mark as incomplete."
        ),
        category="completeness",
        default_severity="likely-significant",
        tier="core",
    ),
    ReviewCheckSpec(
        id="simplicity",
        question="Is this the smallest change that would make the issue's reported example work, and does it match how sibling code solves similar problems?",
        guidance=(
            "First: describe the smallest possible change that would resolve the reported symptom, then compare that baseline against the actual diff. "
            "Every extra line in the patch should be justified against a specific requirement from the issue — if it cannot, flag it as unjustified scope. "
            "Second: find 2-3 structurally similar classes or functions in the same module. "
            "Enumerate the methods, overrides, or conventions they share, then compare the new code against that pattern. "
            "If the patch diverges from the neighborhood's conventions, the divergence must be explicitly justified; otherwise flag it. "
            "Over-engineered patches (large refactors when a 1-line fix would do, new config options the issue did not request, backward-compat shims against changes the issue explicitly accepted) should be flagged."
        ),
        category="maintainability",
        default_severity="likely-significant",
        tier="core",
    ),
)

_CHECK_BY_ID: Final[dict[str, ReviewCheckSpec]] = {check.id: check for check in REVIEW_CHECKS}
_CORE_REVIEW_CHECKS: Final[tuple[ReviewCheckSpec, ...]] = tuple(
    check for check in REVIEW_CHECKS if check.tier == "core"
)
# All current review checks are core. The tier field is retained for parity
# with `megaplan/checks.py` and future extensibility — if we add narrower
# specialized checks later, demote them to "extended" and they will be
# excluded from standard/light robustness here.


def get_check_ids() -> list[str]:
    return [check.id for check in REVIEW_CHECKS]


def get_check_by_id(check_id: str) -> ReviewCheckSpec | None:
    return _CHECK_BY_ID.get(check_id)


def build_check_category_map() -> dict[str, str]:
    return {check.id: check.category for check in REVIEW_CHECKS}


def checks_for_robustness(robustness: str) -> tuple[ReviewCheckSpec, ...]:
    if robustness == "heavy":
        return REVIEW_CHECKS
    if robustness in {"light", "tiny"}:
        return ()
    return _CORE_REVIEW_CHECKS


def build_empty_template(checks: tuple[ReviewCheckSpec, ...] | None = None) -> list[dict[str, Any]]:
    active_checks = REVIEW_CHECKS if checks is None else checks
    return [
        {
            "id": check.id,
            "question": check.question,
            "findings": [],
        }
        for check in active_checks
    ]


_MIN_FINDING_DETAIL_LENGTH = 40


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
        if len(detail.strip()) < _MIN_FINDING_DETAIL_LENGTH:
            return False
        if not isinstance(flagged, bool):
            return False
    return True


def validate_review_checks(
    payload: Any,
    *,
    expected_ids: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    raw_checks = payload.get("checks") if isinstance(payload, dict) else payload
    expected = get_check_ids() if expected_ids is None else list(expected_ids)
    if not isinstance(raw_checks, list):
        return expected

    expected_set = set(expected)
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
        for check_id in expected
        if check_id not in valid_ids or check_id in invalid_expected_ids
    ] + sorted(invalid_unknown_ids)


__all__ = [
    "REVIEW_CHECKS",
    "ReviewCheckSpec",
    "build_check_category_map",
    "build_empty_template",
    "checks_for_robustness",
    "get_check_by_id",
    "get_check_ids",
    "validate_review_checks",
]
