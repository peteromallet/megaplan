"""Direct tests for megaplan.schemas module."""
from __future__ import annotations

from megaplan.schemas import strict_schema, SCHEMAS


class TestStrictSchema:
    def test_sets_additional_properties_false(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        result = strict_schema(schema)
        assert result["additionalProperties"] is False

    def test_does_not_override_existing_additional_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        # setdefault keeps existing value
        result = strict_schema(schema)
        assert result["additionalProperties"] is True

    def test_sets_required_to_all_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        }
        result = strict_schema(schema)
        assert sorted(result["required"]) == ["x", "y"]

    def test_overwrites_partial_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }
        result = strict_schema(schema)
        assert sorted(result["required"]) == ["a", "b"]

    def test_nested_objects_are_strict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"},
                    },
                },
            },
        }
        result = strict_schema(schema)
        # Top-level
        assert result["additionalProperties"] is False
        # Nested
        nested = result["properties"]["outer"]
        assert nested["additionalProperties"] is False
        assert nested["required"] == ["inner"]

    def test_array_items_objects_are_strict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            },
        }
        result = strict_schema(schema)
        item_schema = result["properties"]["items_list"]["items"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["required"] == ["name"]

    def test_non_object_types_unchanged(self) -> None:
        schema = {"type": "string"}
        result = strict_schema(schema)
        assert result == {"type": "string"}
        assert "additionalProperties" not in result

    def test_scalar_passthrough(self) -> None:
        assert strict_schema(42) == 42
        assert strict_schema("hello") == "hello"
        assert strict_schema(None) is None

    def test_list_passthrough(self) -> None:
        result = strict_schema([{"type": "object", "properties": {"a": {"type": "string"}}}])
        assert isinstance(result, list)
        assert result[0]["additionalProperties"] is False


class TestSCHEMAS:
    def test_schemas_contains_expected_keys(self) -> None:
        expected = {"clarify.json", "plan.json", "integrate.json", "critique.json", "execution.json", "review.json", "test-both.json"}
        assert expected == set(SCHEMAS.keys())

    def test_all_schemas_are_objects(self) -> None:
        for name, schema in SCHEMAS.items():
            assert schema["type"] == "object", f"{name} should be type object"
            assert "properties" in schema, f"{name} should have properties"

    def test_strict_schema_on_real_schemas(self) -> None:
        """strict_schema applied to every real schema produces valid output."""
        for name, schema in SCHEMAS.items():
            result = strict_schema(schema)
            assert result["additionalProperties"] is False, f"{name} top-level"
            assert "required" in result, f"{name} should have required"
