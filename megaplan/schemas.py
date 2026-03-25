"""JSON schema definitions for megaplan step outputs."""

from __future__ import annotations

from typing import Any


SCHEMAS: dict[str, dict[str, Any]] = {
    "plan.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "self_flags": {
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
            "gate_recommendation": {
                "type": "string",
                "enum": ["PROCEED", "ITERATE", "ESCALATE"],
            },
            "gate_rationale": {"type": "string"},
            "settled_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "decision": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["id", "decision"],
                },
            },
        },
        "required": ["plan", "questions", "success_criteria", "assumptions"],
    },
    "revise.json": {
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
    "gate.json": {
        "type": "object",
        "properties": {
            "recommendation": {
                "type": "string",
                "enum": ["PROCEED", "ITERATE", "ESCALATE"],
            },
            "rationale": {"type": "string"},
            "signals_assessment": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "settled_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "decision": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["id", "decision"],
                },
            },
        },
        "required": ["recommendation", "rationale", "signals_assessment", "warnings", "settled_decisions"],
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
        "required": ["flags", "verified_flag_ids", "disputed_flag_ids"],
    },
    "finalize.json": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "status": {"type": "string", "enum": ["pending", "done", "skipped"]},
                        "executor_notes": {"type": "string"},
                        "files_changed": {"type": "array", "items": {"type": "string"}},
                        "commands_run": {"type": "array", "items": {"type": "string"}},
                        "evidence_files": {"type": "array", "items": {"type": "string"}},
                        "reviewer_verdict": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "description",
                        "depends_on",
                        "status",
                        "executor_notes",
                        "files_changed",
                        "commands_run",
                        "evidence_files",
                        "reviewer_verdict",
                    ],
                },
            },
            "watch_items": {"type": "array", "items": {"type": "string"}},
            "sense_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "task_id": {"type": "string"},
                        "question": {"type": "string"},
                        "executor_note": {"type": "string"},
                        "verdict": {"type": "string"},
                    },
                    "required": ["id", "task_id", "question", "executor_note", "verdict"],
                },
            },
            "meta_commentary": {"type": "string"},
        },
        "required": ["tasks", "watch_items", "sense_checks", "meta_commentary"],
    },
    "execution.json": {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "commands_run": {"type": "array", "items": {"type": "string"}},
            "deviations": {"type": "array", "items": {"type": "string"}},
            "task_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "status": {"type": "string", "enum": ["done", "skipped"]},
                        "executor_notes": {"type": "string"},
                        "files_changed": {"type": "array", "items": {"type": "string"}},
                        "commands_run": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["task_id", "status", "executor_notes", "files_changed", "commands_run"],
                },
            },
            "sense_check_acknowledgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sense_check_id": {"type": "string"},
                        "executor_note": {"type": "string"},
                    },
                    "required": ["sense_check_id", "executor_note"],
                },
            },
        },
        "required": ["output", "files_changed", "commands_run", "deviations", "task_updates", "sense_check_acknowledgments"],
    },
    "review.json": {
        "type": "object",
        "properties": {
            "review_verdict": {"type": "string", "enum": ["approved", "needs_rework"]},
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
            "task_verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "reviewer_verdict": {"type": "string"},
                        "evidence_files": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["task_id", "reviewer_verdict", "evidence_files"],
                },
            },
            "sense_check_verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sense_check_id": {"type": "string"},
                        "verdict": {"type": "string"},
                    },
                    "required": ["sense_check_id", "verdict"],
                },
            },
        },
        "required": ["review_verdict", "criteria", "issues", "summary", "task_verdicts", "sense_check_verdicts"],
    },
}


def strict_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        updated = {key: strict_schema(value) for key, value in schema.items()}
        if updated.get("type") == "object":
            updated.setdefault("additionalProperties", False)
            if "properties" in updated and "required" not in updated:
                updated["required"] = list(updated["properties"].keys())
        return updated
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    return schema
