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
    ]
    assert schema["properties"]["recommendation"]["enum"] == ["PROCEED", "ITERATE", "ESCALATE"]
