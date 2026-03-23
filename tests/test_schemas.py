"""Direct tests for megaplan.schemas."""

from __future__ import annotations

from megaplan.schemas import SCHEMAS, strict_schema


def test_schema_registry_matches_5_step_workflow() -> None:
    assert set(SCHEMAS) == {
        "plan.json",
        "revise.json",
        "gate.json",
        "critique.json",
        "execution.json",
        "review.json",
    }


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
