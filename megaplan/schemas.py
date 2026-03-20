"""JSON schema definitions for megaplan step outputs."""

from __future__ import annotations

from typing import Any


SCHEMAS: dict[str, dict[str, Any]] = {
    "clarify.json": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["question", "context"],
                },
            },
            "refined_idea": {"type": "string"},
            "intent_summary": {"type": "string"},
        },
        "required": ["questions", "refined_idea", "intent_summary"],
    },
    "plan.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plan", "questions", "success_criteria", "assumptions"],
    },
    "integrate.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "changes_summary": {"type": "string"},
            "flags_addressed": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plan", "changes_summary", "flags_addressed"],
    },
    "critique.json": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "concern": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "correctness",
                                "security",
                                "completeness",
                                "performance",
                                "maintainability",
                                "other",
                            ],
                        },
                        "severity_hint": {
                            "type": "string",
                            "enum": ["likely-significant", "likely-minor", "uncertain"],
                        },
                        "evidence": {"type": "string"},
                    },
                    "required": ["id", "concern", "category", "severity_hint", "evidence"],
                },
            },
            "verified_flag_ids": {"type": "array", "items": {"type": "string"}},
            "disputed_flag_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["flags"],
    },
    "execution.json": {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "commands_run": {"type": "array", "items": {"type": "string"}},
            "deviations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["output", "files_changed", "commands_run", "deviations"],
    },
    "review.json": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "pass": {"type": "boolean"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["name", "pass", "evidence"],
                },
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["criteria", "issues"],
    },
}


def strict_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        updated = {key: strict_schema(value) for key, value in schema.items()}
        if updated.get("type") == "object":
            updated.setdefault("additionalProperties", False)
            if "properties" in updated:
                updated["required"] = list(updated["properties"].keys())
        return updated
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    return schema
