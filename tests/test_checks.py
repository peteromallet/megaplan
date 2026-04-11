from __future__ import annotations

from megaplan.checks import (
    build_empty_template,
    checks_for_robustness,
    get_check_by_id,
    validate_critique_checks,
)


def _payload_for(checks: tuple[dict[str, str], ...]) -> dict[str, object]:
    return {
        "checks": [
            {
                "id": check["id"],
                "question": check["question"],
                "findings": [
                    {
                        "detail": "Checked this critique dimension against the repository and found no concrete issue to flag.",
                        "flagged": False,
                    }
                ],
            }
            for check in checks
        ]
    }


def test_checks_for_robustness_returns_expected_check_sets() -> None:
    robust_checks = checks_for_robustness("robust")
    superrobust_checks = checks_for_robustness("superrobust")
    standard_checks = checks_for_robustness("standard")
    light_checks = checks_for_robustness("light")
    tiny_checks = checks_for_robustness("tiny")

    assert len(robust_checks) == 8
    assert len(superrobust_checks) == 8
    assert [check["id"] for check in standard_checks] == [
        "issue_hints",
        "correctness",
        "scope",
        "all_locations",
        "callers",
    ]
    assert get_check_by_id("verification")["tier"] == "extended"
    assert get_check_by_id("conventions")["tier"] == "extended"
    assert get_check_by_id("criteria_quality")["tier"] == "extended"
    assert light_checks == ()
    assert tiny_checks == ()


def test_build_empty_template_uses_filtered_checks() -> None:
    template = build_empty_template(checks_for_robustness("standard"))

    assert [entry["id"] for entry in template] == [
        "issue_hints",
        "correctness",
        "scope",
        "all_locations",
        "callers",
    ]
    assert all(entry["findings"] == [] for entry in template)


def test_validate_critique_checks_accepts_filtered_standard_ids() -> None:
    standard_checks = checks_for_robustness("standard")
    payload = _payload_for(standard_checks)

    assert validate_critique_checks(
        payload,
        expected_ids=[check["id"] for check in standard_checks],
    ) == []


def test_validate_critique_checks_accepts_light_mode_empty_checks() -> None:
    assert validate_critique_checks({"checks": []}, expected_ids=[]) == []


def test_validate_critique_checks_rejects_light_mode_stray_checks() -> None:
    stray_payload = _payload_for((checks_for_robustness("standard")[0],))

    assert validate_critique_checks(stray_payload, expected_ids=[]) == ["issue_hints"]
