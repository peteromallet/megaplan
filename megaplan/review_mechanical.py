"""Cheap advisory pre-checks for heavy review."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megaplan._core import collect_git_diff_patch, intent_and_notes_block
from megaplan.types import PlanState


@dataclass
class _DiffMetadata:
    patch: str
    files: list[str]
    hunks: int
    changed_lines: int
    added_lines: dict[str, set[int]]


def _pre_check_flag(
    check: str,
    detail: str,
    *,
    severity: str = "minor",
    evidence_file: str | None = None,
    suffix: str | None = None,
) -> dict[str, str]:
    stem = check.upper().replace("-", "_")
    flag = {
        "id": f"PRECHECK-{stem}{f'-{suffix}' if suffix else ''}",
        "check": check,
        "detail": detail,
        "severity": severity,
    }
    if evidence_file:
        flag["evidence_file"] = evidence_file
    return flag


# Paths that should never be counted as part of a source patch. These are
# workspace/infrastructure files that `git add -N . && git diff HEAD` will
# happily include but that the mechanical pre-checks should ignore.
_DIFF_NOISE_PREFIXES: tuple[str, ...] = (
    ".megaplan/",       # megaplan's own plan/critique/execution metadata
    "megaplan/.megaplan/",
    ".git/",            # git internals (shouldn't appear, but defensive)
    "__pycache__/",     # compiled python
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    "node_modules/",
    ".venv/",
    "venv/",
    ".hermes/",         # hermes agent state
)
_DIFF_NOISE_SUBSTRINGS: tuple[str, ...] = (
    "/__pycache__/",
    "/.megaplan/",
    "/node_modules/",
)


def _is_diff_noise(path: str) -> bool:
    """Return True if the diff entry is workspace/infrastructure noise, not source."""
    normalized = path.replace("\\", "/")
    if any(normalized.startswith(prefix) for prefix in _DIFF_NOISE_PREFIXES):
        return True
    if any(substr in normalized for substr in _DIFF_NOISE_SUBSTRINGS):
        return True
    if normalized.endswith((".pyc", ".pyo")):
        return True
    return False


def _parse_diff_metadata(patch: str) -> _DiffMetadata:
    if patch in {
        "No git changes detected.",
        "Project directory is not a git repository.",
        "git not found on PATH.",
        "git diff timed out.",
    }:
        return _DiffMetadata(
            patch=patch,
            files=[],
            hunks=0,
            changed_lines=0,
            added_lines={},
        )

    files: list[str] = []
    file_set: set[str] = set()
    added_lines: dict[str, set[int]] = {}
    changed_lines = 0
    hunk_count = 0
    current_file: str | None = None
    current_new_line = 0
    skip_current_file = False

    for raw_line in patch.splitlines():
        if raw_line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+) b/(.+)", raw_line)
            current_file = match.group(2) if match else None
            skip_current_file = bool(current_file and _is_diff_noise(current_file))
            if current_file and not skip_current_file and current_file not in file_set:
                file_set.add(current_file)
                files.append(current_file)
                added_lines.setdefault(current_file, set())
            continue

        if skip_current_file:
            continue

        if raw_line.startswith("@@ "):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
            if match:
                current_new_line = int(match.group(1))
                hunk_count += 1
            continue

        if current_file is None:
            continue

        if raw_line.startswith("+++ ") or raw_line.startswith("--- "):
            continue
        if raw_line.startswith("+"):
            changed_lines += 1
            added_lines.setdefault(current_file, set()).add(current_new_line)
            current_new_line += 1
            continue
        if raw_line.startswith("-"):
            changed_lines += 1
            continue
        if raw_line.startswith(" "):
            current_new_line += 1

    return _DiffMetadata(
        patch=patch,
        files=files,
        hunks=hunk_count,
        changed_lines=changed_lines,
        added_lines=added_lines,
    )


def _is_test_path(path: str) -> bool:
    pure = Path(path)
    name = pure.name.lower()
    parts = {part.lower() for part in pure.parts}
    return (
        "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def _source_touch_flags(metadata: _DiffMetadata) -> list[dict[str, str]]:
    if not metadata.files:
        return [
            _pre_check_flag(
                "source_touch",
                "No changed files were detected in the git diff, so heavy review cannot confirm that package source changed.",
                severity="significant",
            )
        ]
    non_test_files = [path for path in metadata.files if not _is_test_path(path)]
    if non_test_files:
        return []
    return [
        _pre_check_flag(
            "source_touch",
            "The diff only touches test-like files. Heavy review expects at least one non-test package file to change.",
            severity="significant",
            evidence_file=metadata.files[0],
        )
    ]


def _rough_issue_expectation(state: PlanState) -> int:
    intent_block = intent_and_notes_block(state)
    fenced_snippets = intent_block.count("```") // 2
    return max(1, fenced_snippets) * 10


def _diff_size_sanity_flags(metadata: _DiffMetadata, state: PlanState) -> list[dict[str, str]]:
    expected_lines = _rough_issue_expectation(state)
    actual_lines = metadata.changed_lines
    if expected_lines <= 0:
        return []
    ratio = actual_lines / expected_lines if expected_lines else 0.0
    if actual_lines == 0 or ratio > 3.0 or ratio < 0.3:
        direction = "larger" if ratio > 3.0 else "smaller"
        if actual_lines == 0:
            detail = (
                f"Diff size sanity check found no changed lines, versus a rough expectation of about {expected_lines} lines "
                f"from the issue context (files={len(metadata.files)}, hunks={metadata.hunks})."
            )
        else:
            detail = (
                f"Diff size looks {direction} than expected: changed_lines={actual_lines}, expected≈{expected_lines}, "
                f"ratio={ratio:.2f}, files={len(metadata.files)}, hunks={metadata.hunks}."
            )
        severity = "significant" if ratio > 3.0 or actual_lines == 0 else "minor"
        evidence_file = metadata.files[0] if metadata.files else None
        return [_pre_check_flag("diff_size_sanity", detail, severity=severity, evidence_file=evidence_file)]
    return []


def _iter_python_files(project_dir: Path) -> list[Path]:
    return [
        path
        for path in project_dir.rglob("*.py")
        if ".git" not in path.parts and "__pycache__" not in path.parts
    ]


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _falsey_literal(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return node.value in {None, False, 0, 0.0, ""}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return len(node.elts if hasattr(node, "elts") else node.keys) == 0
    return False


def _literal_truthiness(node: ast.AST | None) -> bool | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return bool(node.elts if hasattr(node, "elts") else node.keys)
    return None


def _function_param_defaults(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.AST | None]:
    positional = list(func.args.posonlyargs) + list(func.args.args)
    defaults: dict[str, ast.AST | None] = {arg.arg: None for arg in positional}
    if func.args.defaults:
        for arg, default in zip(positional[-len(func.args.defaults):], func.args.defaults):
            defaults[arg.arg] = default
    for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
        defaults[arg.arg] = default
    return defaults


def _argument_for_param(call: ast.Call, func: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str) -> ast.AST | None:
    positional = [arg.arg for arg in list(func.args.posonlyargs) + list(func.args.args)]
    if param_name in positional:
        index = positional.index(param_name)
        if index < len(call.args):
            return call.args[index]
    for keyword in call.keywords:
        if keyword.arg == param_name:
            return keyword.value
    return None


def _candidate_call_truthiness(project_dir: Path, function_name: str, param_name: str, func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[bool | None]:
    truthiness: list[bool | None] = []
    for path in _iter_python_files(project_dir):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == function_name:
                argument = _argument_for_param(node, func, param_name)
                truthiness.append(_literal_truthiness(argument))
            elif isinstance(callee, ast.Attribute) and callee.attr == function_name:
                argument = _argument_for_param(node, func, param_name)
                truthiness.append(_literal_truthiness(argument))
    return truthiness


def _dead_guard_static_flags(project_dir: Path, metadata: _DiffMetadata) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    python_files = [path for path in metadata.files if path.endswith(".py") and metadata.added_lines.get(path)]
    for rel_path in python_files:
        path = project_dir / rel_path
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as exc:
            flags.append(
                _pre_check_flag(
                    "dead_guard_static",
                    f"Skipped dead-guard static analysis because {rel_path} could not be parsed as Python: {exc.msg}.",
                    severity="minor",
                    evidence_file=rel_path,
                    suffix="PARSE",
                )
            )
            continue
        except OSError:
            continue

        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if node.lineno not in metadata.added_lines.get(rel_path, set()):
                continue
            if not isinstance(node.test, ast.Name):
                continue

            func = _enclosing_function(node, parents)
            if func is None:
                continue

            guard_name = node.test.id
            defaults = _function_param_defaults(func)
            if guard_name not in defaults:
                continue
            if _falsey_literal(defaults.get(guard_name)):
                continue

            truthiness = _candidate_call_truthiness(project_dir, func.name, guard_name, func)
            if not truthiness:
                continue
            if any(value is None or value is False for value in truthiness):
                continue

            detail = (
                f"New guard `if {guard_name}:` in {rel_path}:{node.lineno} appears unreachable by the observed call sites for "
                f"`{func.name}()`: every statically readable caller passes a truthy literal for `{guard_name}`."
            )
            flags.append(
                _pre_check_flag(
                    "dead_guard_static",
                    detail,
                    severity="significant",
                    evidence_file=rel_path,
                    suffix=str(node.lineno),
                )
            )
    return flags


def run_pre_checks(plan_dir: Path, state: PlanState, project_dir: Path) -> list[dict[str, str]]:
    del plan_dir
    metadata = _parse_diff_metadata(collect_git_diff_patch(project_dir))
    flags: list[dict[str, str]] = []
    flags.extend(_source_touch_flags(metadata))
    flags.extend(_diff_size_sanity_flags(metadata, state))
    flags.extend(_dead_guard_static_flags(project_dir, metadata))
    return flags


__all__ = ["run_pre_checks"]
