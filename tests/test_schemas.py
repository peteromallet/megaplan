"""Direct tests for megaplan.schemas."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft7Validator

from megaplan.schemas import SCHEMAS, strict_schema


def _review_disk_schema() -> dict[str, object]:
    return json.loads((Path(__file__).resolve().parents[1] / ".megaplan" / "schemas" / "review.json").read_text(encoding="utf-8"))


def _minimal_review_payload() -> dict[str, object]:
    return {
        "review_verdict": "approved",
        "checks": [],
        "pre_check_flags": [],
        "verified_flag_ids": [],
        "disputed_flag_ids": [],
        "criteria": [],
        "issues": [],
        "rework_items": [],
        "summary": "Approved.",
        "task_verdicts": [],
        "sense_check_verdicts": [],
    }


def test_schema_registry_matches_5_step_workflow() -> None:
    required = {"plan.json", "prep.json", "revise.json", "gate.json", "critique.json", "finalize.json", "execution.json", "review.json"}
    assert required.issubset(set(SCHEMAS))


# ---------------------------------------------------------------------------
# strict_schema tests
# ---------------------------------------------------------------------------


def test_strict_schema_adds_additional_properties_false() -> None:
    result = strict_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    assert result["additionalProperties"] is False


def test_strict_schema_preserves_existing_additional_properties() -> None:
    result = strict_schema({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True})
    assert result["additionalProperties"] is True


def test_strict_schema_sets_required_from_properties() -> None:
    result = strict_schema({"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "number"}}})
    assert set(result["required"]) == {"x", "y"}


def test_strict_schema_overwrites_partial_required_arrays_recursively() -> None:
    schema = {
        "type": "object",
        "required": ["stale_root"],
        "properties": {
            "inner": {
                "type": "object",
                "required": ["stale_inner"],
                "properties": {"child": {"type": "string"}},
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["stale_item"],
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    }

    result = strict_schema(schema)

    assert result["required"] == ["inner", "items"]
    assert result["properties"]["inner"]["required"] == ["child"]
    assert result["properties"]["items"]["items"]["required"] == ["name"]


def test_strict_schema_nested_objects_get_additional_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
    }
    result = strict_schema(schema)
    assert result["properties"]["inner"]["additionalProperties"] is False
    assert result["properties"]["inner"]["required"] == ["a"]


def test_strict_schema_array_items_are_strict() -> None:
    schema = {
        "type": "object",
        "properties": {
            "list": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}}},
            }
        },
    }
    result = strict_schema(schema)
    assert result["properties"]["list"]["items"]["additionalProperties"] is False


def test_strict_schema_deeply_nested() -> None:
    schema = {
        "type": "object",
        "properties": {
            "l1": {
                "type": "object",
                "properties": {
                    "l2": {
                        "type": "object",
                        "properties": {"l3": {"type": "string"}},
                    }
                },
            }
        },
    }
    result = strict_schema(schema)
    assert result["properties"]["l1"]["properties"]["l2"]["additionalProperties"] is False


def test_strict_schema_non_object_untouched() -> None:
    assert strict_schema({"type": "string"}) == {"type": "string"}
    assert strict_schema(42) == 42
    assert strict_schema("hello") == "hello"
    assert strict_schema([1, 2]) == [1, 2]


# ---------------------------------------------------------------------------
# Schema completeness tests
# ---------------------------------------------------------------------------


def test_schema_registry_has_all_expected_steps() -> None:
    required_schemas = {"plan.json", "prep.json", "revise.json", "gate.json", "critique.json", "finalize.json", "execution.json", "review.json"}
    assert required_schemas.issubset(set(SCHEMAS.keys()))


def test_schema_registry_entries_include_required_field() -> None:
    for name, schema in SCHEMAS.items():
        assert "required" in schema, f"Schema '{name}' missing 'required' field"
        assert isinstance(schema["required"], list)


def test_schema_registry_entries_are_objects() -> None:
    for name, schema in SCHEMAS.items():
        assert schema.get("type") == "object", f"Schema '{name}' is not type 'object'"
        assert "properties" in schema, f"Schema '{name}' missing 'properties'"


def test_critique_schema_flags_have_expected_structure() -> None:
    critique = SCHEMAS["critique.json"]
    flags_schema = critique["properties"]["flags"]
    assert flags_schema["type"] == "array"
    item_schema = flags_schema["items"]
    assert "id" in item_schema["properties"]
    assert "concern" in item_schema["properties"]
    assert "category" in item_schema["properties"]
    assert "severity_hint" in item_schema["properties"]
    assert "evidence" in item_schema["properties"]


def test_finalize_schema_tracks_structured_execution_fields() -> None:
    finalize = SCHEMAS["finalize.json"]
    assert "tasks" in finalize["properties"]
    assert "sense_checks" in finalize["properties"]
    assert "validation" in finalize["properties"]
    assert "validation" in finalize["required"]
    assert "final_plan" not in finalize["properties"]
    assert "task_count" not in finalize["properties"]
    task_schema = finalize["properties"]["tasks"]["items"]
    assert set(task_schema["properties"]) == {
        "id",
        "description",
        "depends_on",
        "status",
        "executor_notes",
        "files_changed",
        "commands_run",
        "evidence_files",
        "reviewer_verdict",
    }
    assert task_schema["properties"]["status"]["enum"] == ["pending", "done", "skipped"]
    assert "executor_note" in finalize["properties"]["sense_checks"]["items"]["properties"]
    # Validation sub-schema
    validation_schema = finalize["properties"]["validation"]
    assert "plan_steps_covered" in validation_schema["properties"]
    assert "orphan_tasks" in validation_schema["properties"]
    assert "completeness_notes" in validation_schema["properties"]
    assert "coverage_complete" in validation_schema["properties"]
    step_item = validation_schema["properties"]["plan_steps_covered"]["items"]
    assert "plan_step_summary" in step_item["properties"]
    assert "finalize_task_ids" in step_item["properties"]
    assert step_item["properties"]["finalize_task_ids"]["type"] == "array"


def test_execution_schema_requires_task_updates() -> None:
    execution = SCHEMAS["execution.json"]
    assert "task_updates" in execution["properties"]
    assert "task_updates" in execution["required"]
    assert "sense_check_acknowledgments" in execution["properties"]
    assert "sense_check_acknowledgments" in execution["required"]
    item_schema = execution["properties"]["task_updates"]["items"]
    assert item_schema["properties"]["status"]["enum"] == ["done", "skipped"]
    assert "files_changed" in item_schema["properties"]
    assert "commands_run" in item_schema["properties"]


def test_review_schema_requires_task_and_sense_check_verdicts() -> None:
    review = SCHEMAS["review.json"]
    assert "review_verdict" in review["properties"]
    assert "task_verdicts" in review["properties"]
    assert "sense_check_verdicts" in review["properties"]
    assert "rework_items" in review["properties"]
    assert "review_verdict" in review["required"]
    assert "task_verdicts" in review["required"]
    assert "sense_check_verdicts" in review["required"]
    assert "rework_items" in review["required"]
    assert "evidence_files" in review["properties"]["task_verdicts"]["items"]["properties"]
    # Rework items sub-schema
    rework_item = review["properties"]["rework_items"]["items"]
    assert "task_id" in rework_item["properties"]
    assert "issue" in rework_item["properties"]
    assert "expected" in rework_item["properties"]
    assert "actual" in rework_item["properties"]
    assert "evidence_file" in rework_item["properties"]
    assert set(rework_item["required"]) == {"task_id", "issue", "expected", "actual", "evidence_file"}


def test_review_schema_accepts_heavy_mode_extensions_in_both_copies() -> None:
    payload = {
        "review_verdict": "needs_rework",
        "checks": [
            {
                "id": "coverage",
                "question": "Does the diff cover the issue?",
                "guidance": "Inspect the changed module for missing review follow-up.",
                "findings": [
                    {
                        "detail": "Coverage review found one concrete issue example that the diff still does not handle.",
                        "flagged": True,
                        "status": "blocking",
                        "evidence_file": "pkg/module.py",
                    }
                ],
                "prior_findings": [],
            }
        ],
        "pre_check_flags": [
            {
                "id": "PRECHECK-SOURCE_TOUCH",
                "check": "source_touch",
                "detail": "The diff touches a package source file.",
                "severity": "minor",
                "evidence_file": "pkg/module.py",
            }
        ],
        "verified_flag_ids": ["REVIEW-COVERAGE-001"],
        "disputed_flag_ids": ["REVIEW-PARITY-001"],
        "criteria": [{"name": "criterion", "priority": "must", "pass": "fail", "evidence": "Missing coverage."}],
        "issues": ["Coverage review found a blocking issue."],
        "rework_items": [
            {
                "task_id": "REVIEW",
                "issue": "Coverage gap remains.",
                "expected": "All issue examples are covered.",
                "actual": "One issue example remains uncovered.",
                "evidence_file": "pkg/module.py",
                "source": "review_coverage",
            }
        ],
        "summary": "Heavy review found a blocking issue.",
        "task_verdicts": [
            {
                "task_id": "T1",
                "reviewer_verdict": "Needs follow-up.",
                "evidence_files": ["pkg/module.py"],
            }
        ],
        "sense_check_verdicts": [{"sense_check_id": "SC1", "verdict": "Needs follow-up."}],
    }
    disk_schema = _review_disk_schema()

    assert list(Draft7Validator(SCHEMAS["review.json"]).iter_errors(payload)) == []
    assert list(Draft7Validator(disk_schema).iter_errors(payload)) == []


def test_review_schema_accepts_optional_rework_item_flag_id() -> None:
    payload = _minimal_review_payload()
    payload["review_verdict"] = "needs_rework"
    payload["issues"] = ["Critique flag remains unresolved."]
    payload["rework_items"] = [
        {
            "task_id": "REVIEW",
            "issue": "Critique flag remains unresolved.",
            "expected": "The final diff addresses the flagged concern directly.",
            "actual": "The diff leaves the flagged behavior unchanged.",
            "evidence_file": "megaplan/prompts/review.py",
            "flag_id": "FLAG-001",
            "source": "review_flag_reverify",
        }
    ]
    disk_schema = _review_disk_schema()

    assert list(Draft7Validator(SCHEMAS["review.json"]).iter_errors(payload)) == []
    assert list(Draft7Validator(disk_schema).iter_errors(payload)) == []


def test_review_schema_still_accepts_rework_items_without_flag_id() -> None:
    payload = _minimal_review_payload()
    payload["review_verdict"] = "needs_rework"
    payload["issues"] = ["Executor still needs to finish the review follow-up."]
    payload["rework_items"] = [
        {
            "task_id": "REVIEW",
            "issue": "Executor still needs to finish the review follow-up.",
            "expected": "All required review follow-up work is complete.",
            "actual": "One required review follow-up item is still missing.",
            "evidence_file": "megaplan/handlers.py",
        }
    ]
    disk_schema = _review_disk_schema()

    assert list(Draft7Validator(SCHEMAS["review.json"]).iter_errors(payload)) == []
    assert list(Draft7Validator(disk_schema).iter_errors(payload)) == []


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------


def test_gate_schema_is_strict_and_requires_all_fields() -> None:
    schema = strict_schema(SCHEMAS["gate.json"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "recommendation",
        "rationale",
        "signals_assessment",
        "warnings",
        "settled_decisions",
        "flag_resolutions",
        "accepted_tradeoffs",
    ]
    assert schema["properties"]["recommendation"]["enum"] == ["PROCEED", "ITERATE", "ESCALATE"]


def test_plan_schema_has_core_fields_only() -> None:
    schema = strict_schema(SCHEMAS["plan.json"])
    assert set(schema["required"]) == {"plan", "questions", "success_criteria", "assumptions"}
    assert "self_flags" not in schema["properties"]
    assert "gate_recommendation" not in schema["properties"]


def test_prep_schema_exists_and_has_expected_structure() -> None:
    schema = strict_schema(SCHEMAS["prep.json"])
    assert set(schema["required"]) == {
        "skip",
        "task_summary",
        "key_evidence",
        "relevant_code",
        "test_expectations",
        "constraints",
        "suggested_approach",
    }
    evidence_schema = schema["properties"]["key_evidence"]["items"]
    relevant_code_schema = schema["properties"]["relevant_code"]["items"]
    test_expectation_schema = schema["properties"]["test_expectations"]["items"]
    assert set(evidence_schema["required"]) == {"point", "source", "relevance"}
    assert evidence_schema["properties"]["relevance"]["enum"] == ["high", "medium", "low"]
    assert set(relevant_code_schema["required"]) == {"file_path", "why", "functions"}
    assert relevant_code_schema["properties"]["functions"]["items"]["type"] == "string"
    assert set(test_expectation_schema["required"]) == {"test_id", "what_it_checks", "status"}
    assert test_expectation_schema["properties"]["status"]["enum"] == ["fail_to_pass", "pass_to_pass"]


def test_gate_schema_includes_settled_decisions_structure() -> None:
    schema = strict_schema(SCHEMAS["gate.json"])
    item_schema = schema["properties"]["settled_decisions"]["items"]
    assert set(item_schema["required"]) == {"id", "decision", "rationale"}
    assert "rationale" in item_schema["properties"]


def test_schema_registry_covers_the_six_strict_mode_required_fixes() -> None:
    revise = SCHEMAS["revise.json"]
    gate = SCHEMAS["gate.json"]
    review = SCHEMAS["review.json"]
    review_check = review["properties"]["checks"]["items"]
    review_finding = review_check["properties"]["findings"]["items"]
    pre_check_flag = review["properties"]["pre_check_flags"]["items"]

    assert set(revise["required"]) == {
        "plan",
        "changes_summary",
        "flags_addressed",
        "assumptions",
        "success_criteria",
        "questions",
    }
    assert set(gate["required"]) == {
        "recommendation",
        "rationale",
        "signals_assessment",
        "warnings",
        "settled_decisions",
        "flag_resolutions",
        "accepted_tradeoffs",
    }
    assert set(review["required"]) == {
        "review_verdict",
        "checks",
        "pre_check_flags",
        "verified_flag_ids",
        "disputed_flag_ids",
        "criteria",
        "issues",
        "rework_items",
        "summary",
        "task_verdicts",
        "sense_check_verdicts",
    }
    assert set(review_check["required"]) == {"id", "question", "guidance", "findings", "prior_findings"}
    assert set(review_finding["required"]) == {"detail", "flagged", "status", "evidence_file"}
    assert set(pre_check_flag["required"]) == {"id", "check", "detail", "severity", "evidence_file"}


def test_strict_schema_new_tracking_objects_are_strict() -> None:
    schema = strict_schema(SCHEMAS["finalize.json"])
    task_schema = schema["properties"]["tasks"]["items"]
    sense_check_schema = schema["properties"]["sense_checks"]["items"]
    assert task_schema["additionalProperties"] is False
    assert set(task_schema["required"]) == {
        "id",
        "description",
        "depends_on",
        "status",
        "executor_notes",
        "files_changed",
        "commands_run",
        "evidence_files",
        "reviewer_verdict",
    }
    assert sense_check_schema["additionalProperties"] is False
    assert set(sense_check_schema["required"]) == {"id", "task_id", "question", "executor_note", "verdict"}
