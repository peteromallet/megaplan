from __future__ import annotations

from typing import Any, Callable


# Common field name aliases that models use instead of the canonical names.
# Models often use finalize.json's field names (e.g. "id") instead of the
# execute schema's names (e.g. "task_id").
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "task_id": ("id", "taskId", "task"),
    "sense_check_id": ("id", "senseCheckId", "check_id"),
    "executor_notes": ("notes", "executor_note", "note"),
    "executor_note": ("notes", "executor_notes", "note"),
    "concern": ("summary", "description", "issue", "finding"),
    "evidence": ("detail", "details", "explanation", "reasoning"),
}


# Normalize enum values to canonical forms.
_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "status": {"completed": "done", "complete": "done", "skip": "skipped"},
}


def _normalize_field_aliases(entry: dict[str, Any], required_fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy aliased field values to canonical names if the canonical name is missing,
    and normalize enum value synonyms."""
    for field in required_fields:
        if field in entry:
            continue
        aliases = _FIELD_ALIASES.get(field, ())
        for alias in aliases:
            if alias in entry:
                entry[field] = entry[alias]
                break
    # Default missing array fields to [] and missing string fields to ""
    # rather than rejecting. Models often omit empty arrays/strings.
    for field in required_fields:
        if field not in entry:
            if field in ("files_changed", "commands_run"):
                entry[field] = []
            elif field in ("executor_notes", "executor_note"):
                entry[field] = "(not provided)"
    # Normalize enum value aliases
    for field, value_map in _VALUE_ALIASES.items():
        if field in entry and isinstance(entry[field], str):
            canonical = value_map.get(entry[field])
            if canonical is not None:
                entry[field] = canonical
    return entry


def _validate_merge_inputs(
    entries: Any,
    *,
    required_fields: tuple[str, ...],
    enum_fields: dict[str, set[str]] | None = None,
    nonempty_fields: set[str] | None = None,
    array_fields: tuple[str, ...] = (),
    deviations: list[str] | None = None,
    label: str,
) -> list[dict[str, Any]]:
    enum_fields = enum_fields or {}
    nonempty_fields = nonempty_fields or set()
    array_field_set = set(array_fields)
    valid_entries: list[dict[str, Any]] = []
    if not isinstance(entries, list):
        return valid_entries
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            if deviations is not None:
                deviations.append(f"Skipped malformed {label}[{index}]: expected object.")
            continue
        # Normalize field aliases before checking required fields
        _normalize_field_aliases(entry, required_fields)
        if any(field not in entry for field in required_fields):
            if deviations is not None:
                deviations.append(f"Skipped malformed {label}[{index}]: missing required keys.")
            continue
        normalized: dict[str, Any] = {}
        malformed = False
        for field in required_fields:
            value = entry[field]
            if field in array_field_set:
                if not isinstance(value, list):
                    malformed = True
                    break
                normalized[field] = list(value)
                continue
            if not isinstance(value, str):
                malformed = True
                break
            allowed = enum_fields.get(field)
            if allowed is not None and value not in allowed:
                malformed = True
                break
            normalized[field] = value
        if malformed:
            if deviations is not None:
                deviations.append(f"Skipped malformed {label}[{index}]: invalid field types or enum values.")
            continue
        empty_field = next((field for field in nonempty_fields if normalized.get(field, "").strip() == ""), None)
        if empty_field is not None:
            if deviations is not None:
                deviations.append(f"Skipped {label}[{index}]: '{empty_field}' must not be empty.")
            continue
        valid_entries.append(normalized)
    return valid_entries


def _merge_validated_entries(
    entries: list[dict[str, Any]],
    *,
    targets_by_id: dict[str, dict[str, Any]],
    id_field: str,
    merge_fields: tuple[str, ...],
    issues: list[str],
    label: str,
) -> int:
    """Merge validated entries into targets, deduplicating by ID. Returns unique merge count."""
    seen: set[str] = set()
    for entry in entries:
        entry_id = entry[id_field]
        target = targets_by_id.get(entry_id)
        if target is None:
            issues.append(f"Skipped {label} for unknown {id_field} '{entry_id}'.")
            continue
        if entry_id in seen:
            issues.append(f"Duplicate {label} for '{entry_id}' — last entry wins.")
        for field in merge_fields:
            target[field] = entry[field]
        seen.add(entry_id)
    return len(seen)


def _validate_and_merge_batch(
    entries: Any,
    *,
    required_fields: tuple[str, ...],
    targets_by_id: dict[str, dict[str, Any]],
    id_field: str,
    merge_fields: tuple[str, ...],
    issues: list[str],
    validation_label: str,
    merge_label: str,
    incomplete_message: Callable[[int, int], str] | None = None,
    enum_fields: dict[str, set[str]] | None = None,
    nonempty_fields: set[str] | None = None,
    array_fields: tuple[str, ...] = (),
) -> tuple[int, int]:
    valid_entries = _validate_merge_inputs(
        entries,
        required_fields=required_fields,
        enum_fields=enum_fields,
        nonempty_fields=nonempty_fields,
        array_fields=array_fields,
        deviations=issues,
        label=validation_label,
    )
    total = len(targets_by_id)
    merged_count = _merge_validated_entries(
        valid_entries,
        targets_by_id=targets_by_id,
        id_field=id_field,
        merge_fields=merge_fields,
        issues=issues,
        label=merge_label,
    )
    if incomplete_message is not None and merged_count < total:
        issues.append(incomplete_message(merged_count, total))
    return merged_count, total
