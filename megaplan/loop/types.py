"""Types for the MegaLoop iterative agent workflow."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from megaplan.types import PlanConfig, SessionInfo


class Observation(TypedDict):
    elapsed_seconds: int
    tail_output: str
    metric: float | None
    action: str


class IterationResult(TypedDict, total=False):
    iteration: int
    phase: str
    outcome: str
    diagnosis: str
    fix_description: str
    files_to_change: list[str]
    confidence: str
    should_pause: bool
    returncode: int
    command_output: str
    metric: float
    commit_sha: str
    reverted: bool
    reasoning: str
    observations: list[Observation]


class LoopSpec(TypedDict, total=False):
    goal: str
    command: str
    success_criteria: list[str]
    allowed_changes: list[str]
    optimization_strategy: str
    bug_finding_approach: str
    philosophy: str
    metric_pattern: str
    known_issues: list[str]
    tried_and_failed: list[str]
    best_result_summary: str
    current_best: IterationResult
    observe_interval: int
    observe_break_patterns: list[str]
    observe_metric_stall: int


class LoopState(TypedDict):
    name: str
    spec: LoopSpec
    phase: str
    status: str
    iteration: int
    config: PlanConfig
    sessions: dict[str, SessionInfo]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]
    max_iterations: NotRequired[int]
    results: NotRequired[list[IterationResult]]
    current_best: NotRequired[IterationResult]
    pause_requested: NotRequired[bool]
    pause_reason: NotRequired[str]
