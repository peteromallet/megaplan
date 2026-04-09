"""JSON schema definitions for megaplan step outputs."""

from __future__ import annotations

from typing import Any


SCHEMAS: dict[str, dict[str, Any]] = {
    "plan.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "priority": {"type": "string", "enum": ["must", "should", "info"]},
                    },
                    "required": ["criterion", "priority"],
                },
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plan", "questions", "success_criteria", "assumptions"],
    },
    "prep.json": {
        "type": "object",
        "properties": {
            "skip": {"type": "boolean"},
            "task_summary": {"type": "string"},
            "key_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "source": {"type": "string"},
                        "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["point", "source", "relevance"],
                },
            },
            "relevant_code": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "why": {"type": "string"},
                        "functions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["file_path", "why", "functions"],
                },
            },
            "test_expectations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_id": {"type": "string"},
                        "what_it_checks": {"type": "string"},
                        "status": {"type": "string", "enum": ["fail_to_pass", "pass_to_pass"]},
                    },
                    "required": ["test_id", "what_it_checks", "status"],
                },
            },
            "constraints": {"type": "array", "items": {"type": "string"}},
            "suggested_approach": {"type": "string"},
        },
        "required": [
            "skip",
            "task_summary",
            "key_evidence",
            "relevant_code",
            "test_expectations",
            "constraints",
            "suggested_approach",
        ],
    },
    "revise.json": {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "changes_summary": {"type": "string"},
            "flags_addressed": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "priority": {"type": "string", "enum": ["must", "should", "info"]},
                    },
                    "required": ["criterion", "priority"],
                },
            },
            "questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "plan",
            "changes_summary",
            "flags_addressed",
            "assumptions",
            "success_criteria",
            "questions",
        ],
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
                    "required": ["id", "decision", "rationale"],
                },
            },
            "flag_resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "flag_id": {"type": "string"},
                        "action": {"type": "string", "enum": ["dispute", "accept_tradeoff"]},
                        "evidence": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["flag_id", "action", "evidence", "rationale"],
                },
            },
            "accepted_tradeoffs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "flag_id": {"type": "string"},
                        "concern": {"type": "string"},
                        "subsystem": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["flag_id", "concern", "subsystem", "rationale"],
                },
            },
        },
        "required": [
            "recommendation",
            "rationale",
            "signals_assessment",
            "warnings",
            "settled_decisions",
            "flag_resolutions",
            "accepted_tradeoffs",
        ],
    },
    "critique.json": {
        "type": "object",
        "properties": {
            "checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "detail": {"type": "string"},
                                    "flagged": {"type": "boolean"},
                                },
                                "required": ["detail", "flagged"],
                            },
                        },
                    },
                    "required": ["id", "question", "findings"],
                },
            },
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
        "required": ["checks", "flags", "verified_flag_ids", "disputed_flag_ids"],
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
            "validation": {
                "type": "object",
                "properties": {
                    "plan_steps_covered": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "plan_step_summary": {"type": "string"},
                                "finalize_task_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["plan_step_summary", "finalize_task_ids"],
                        },
                    },
                    "orphan_tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "completeness_notes": {"type": "string"},
                    "coverage_complete": {"type": "boolean"},
                },
                "required": [
                    "plan_steps_covered",
                    "orphan_tasks",
                    "completeness_notes",
                    "coverage_complete",
                ],
            },
        },
        "required": ["tasks", "watch_items", "sense_checks", "meta_commentary", "validation"],
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
    "loop_plan.json": {
        "type": "object",
        "properties": {
            "spec_updates": {
                "type": "object",
                "additionalProperties": True,
            },
            "next_action": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["spec_updates", "next_action", "reasoning"],
    },
    "loop_execute.json": {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "fix_description": {"type": "string"},
            "files_to_change": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string"},
            "outcome": {"type": "string"},
            "should_pause": {"type": "boolean"},
        },
        "required": ["diagnosis", "fix_description", "files_to_change", "confidence", "outcome", "should_pause"],
    },
    "review.json": {
        "type": "object",
        "properties": {
            "review_verdict": {"type": "string", "enum": ["approved", "needs_rework"]},
            "checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "guidance": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "detail": {"type": "string"},
                                    "flagged": {"type": "boolean"},
                                    "status": {"type": "string"},
                                    "evidence_file": {"type": "string"},
                                },
                                "required": ["detail", "flagged", "status", "evidence_file"],
                            },
                        },
                        "prior_findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "detail": {"type": "string"},
                                    "flagged": {"type": "boolean"},
                                    "status": {"type": "string"},
                                },
                                "required": ["detail", "flagged", "status"],
                            },
                        },
                    },
                    "required": ["id", "question", "guidance", "findings", "prior_findings"],
                },
            },
            "pre_check_flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "check": {"type": "string"},
                        "detail": {"type": "string"},
                        "severity": {"type": "string"},
                        "evidence_file": {"type": "string"},
                    },
                    "required": ["id", "check", "detail", "severity", "evidence_file"],
                },
            },
            "verified_flag_ids": {"type": "array", "items": {"type": "string"}},
            "disputed_flag_ids": {"type": "array", "items": {"type": "string"}},
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "priority": {"type": "string", "enum": ["must", "should", "info"]},
                        "pass": {"type": "string", "enum": ["pass", "fail", "waived"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["name", "priority", "pass", "evidence"],
                },
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "rework_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "issue": {"type": "string"},
                        "expected": {"type": "string"},
                        "actual": {"type": "string"},
                        "evidence_file": {"type": "string"},
                        "flag_id": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["task_id", "issue", "expected", "actual", "evidence_file"],
                },
            },
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
        "required": [
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
        ],
    },
}


def _preserve_explicit_required(path: tuple[str, ...]) -> bool:
    # `review.rework_items[].flag_id` and `source` intentionally stay optional.
    return path[-3:] == ("properties", "rework_items", "items")


def strict_schema(schema: Any, _path: tuple[str, ...] = ()) -> Any:
    if isinstance(schema, dict):
        updated = {key: strict_schema(value, _path + (key,)) for key, value in schema.items()}
        if updated.get("type") == "object":
            updated.setdefault("additionalProperties", False)
            if "properties" in updated and not _preserve_explicit_required(_path):
                updated["required"] = list(updated["properties"].keys())
        return updated
    if isinstance(schema, list):
        return [strict_schema(item, _path) for item in schema]
    return schema
