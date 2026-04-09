from __future__ import annotations

from megaplan.review_checks import REVIEW_CHECKS, checks_for_robustness, get_check_by_id, validate_review_checks


def _finding(detail: str, *, flagged: bool) -> dict[str, object]:
    return {"detail": detail, "flagged": flagged}


def test_review_checks_registry_has_expected_ids_and_no_criteria_verdict() -> None:
    ids = [check.id for check in REVIEW_CHECKS]

    # Optimized from 8 to 4 orthogonal checks. Dropped entries (dead_guard,
    # parser_source, return_value_downstream, parity) were narrower
    # specializations of the core concerns that produced redundant or
    # inapplicable output on a 1-task validation. Their intent has been
    # folded into the guidance of the 4 remaining checks.
    assert len(ids) == 4
    assert ids == [
        "coverage",
        "placement",
        "adjacent_calls",
        "simplicity",
    ]
    assert "criteria_verdict" not in ids
    assert "parity" not in ids
    assert "dead_guard" not in ids
    assert "parser_source" not in ids
    assert "return_value_downstream" not in ids


def test_checks_for_robustness_matches_expected_tiers() -> None:
    assert [check.id for check in checks_for_robustness("heavy")] == [check.id for check in REVIEW_CHECKS]
    # All 4 remaining checks are core-tier, so standard mode would return the
    # same set — but standard mode is gated out of parallel_review entirely in
    # handlers.py, so this value is unused in practice. Retained for parity
    # with megaplan/checks.py's shape.
    assert [check.id for check in checks_for_robustness("standard")] == [
        "coverage",
        "placement",
        "adjacent_calls",
        "simplicity",
    ]
    assert checks_for_robustness("light") == ()
    assert checks_for_robustness("tiny") == ()


def test_validate_review_checks_rejects_duplicate_ids() -> None:
    payload = {
        "checks": [
            {
                "id": "coverage",
                "question": "Does the diff cover the issue?",
                "findings": [
                    _finding(
                        "Checked every issue example against the patch and found the primary path was covered in detail.",
                        flagged=False,
                    )
                ],
            },
            {
                "id": "coverage",
                "question": "Does the diff cover the issue?",
                "findings": [
                    _finding(
                        "Checked the duplicate coverage slot and confirmed this payload should fail validation immediately.",
                        flagged=False,
                    )
                ],
            },
        ]
    }

    assert validate_review_checks(payload, expected_ids=("coverage",)) == ["coverage"]


def test_coverage_guidance_mentions_verification_test_structural_constraint() -> None:
    coverage = get_check_by_id("coverage")

    assert coverage is not None
    guidance = coverage.guidance.lower()
    assert "verification test" in guidance
    assert "structural" in guidance
