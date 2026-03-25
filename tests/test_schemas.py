"""Direct tests for megaplan.schemas."""

from __future__ import annotations

from megaplan.schemas import SCHEMAS, strict_schema


def test_schema_registry_matches_5_step_workflow() -> None:
    assert set(SCHEMAS) == {
        "plan.json",
        "revise.json",
        "gate.json",
        "critique.json",
        "finalize.json",
        "execution.json",
        "review.json",
    }


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
    expected_schemas = {"plan.json", "revise.json", "gate.json", "critique.json", "finalize.json", "execution.json", "review.json"}
    assert set(SCHEMAS.keys()) == expected_schemas


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
    assert "review_verdict" in review["required"]
    assert "task_verdicts" in review["required"]
    assert "sense_check_verdicts" in review["required"]
    assert "evidence_files" in review["properties"]["task_verdicts"]["items"]["properties"]


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
    ]
    assert schema["properties"]["recommendation"]["enum"] == ["PROCEED", "ITERATE", "ESCALATE"]


def test_plan_schema_keeps_light_mode_gate_fields_optional() -> None:
    schema = strict_schema(SCHEMAS["plan.json"])
    assert set(schema["required"]) == {"plan", "questions", "success_criteria", "assumptions"}
    assert "self_flags" in schema["properties"]
    assert "gate_recommendation" in schema["properties"]
    assert "gate_rationale" in schema["properties"]
    assert "settled_decisions" in schema["properties"]


def test_gate_schema_includes_settled_decisions_structure() -> None:
    schema = strict_schema(SCHEMAS["gate.json"])
    item_schema = schema["properties"]["settled_decisions"]["items"]
    assert set(item_schema["required"]) == {"id", "decision"}
    assert "rationale" in item_schema["properties"]


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
