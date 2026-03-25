from __future__ import annotations

import pytest

from megaplan._core import (
    batch_artifact_path,
    compute_global_batches,
    compute_task_batches,
    list_batch_artifacts,
)


def _task(task_id: str, depends_on: list[str] | None = None) -> dict[str, object]:
    return {"id": task_id, "depends_on": depends_on or []}


def test_compute_task_batches_linear_chain() -> None:
    tasks = [_task("T1"), _task("T2", ["T1"]), _task("T3", ["T2"])]
    assert compute_task_batches(tasks) == [["T1"], ["T2"], ["T3"]]


def test_compute_task_batches_independent_tasks_share_batch() -> None:
    tasks = [_task("T1"), _task("T2"), _task("T3")]
    assert compute_task_batches(tasks) == [["T1", "T2", "T3"]]


def test_compute_task_batches_diamond_graph() -> None:
    tasks = [
        _task("T1"),
        _task("T2", ["T1"]),
        _task("T3", ["T1"]),
        _task("T4", ["T2", "T3"]),
    ]
    assert compute_task_batches(tasks) == [["T1"], ["T2", "T3"], ["T4"]]


def test_compute_task_batches_cycle_raises() -> None:
    tasks = [_task("T1", ["T2"]), _task("T2", ["T1"])]
    with pytest.raises(ValueError, match="Cyclic dependency graph"):
        compute_task_batches(tasks)


def test_compute_task_batches_unknown_dependency_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dependency ID 'T9'"):
        compute_task_batches([_task("T1", ["T9"])])


def test_compute_task_batches_empty_input_returns_empty_list() -> None:
    assert compute_task_batches([]) == []


def test_compute_task_batches_completed_ids_satisfy_pending_dependencies() -> None:
    tasks = [_task("T2", ["T1"])]
    assert compute_task_batches(tasks, completed_ids={"T1"}) == [["T2"]]


def test_compute_task_batches_completed_ids_allow_parallel_pending_tasks() -> None:
    tasks = [_task("T2", ["T1"]), _task("T3", ["T1"])]
    assert compute_task_batches(tasks, completed_ids={"T1"}) == [["T2", "T3"]]


def test_batch_artifact_path_returns_expected_path(tmp_path) -> None:
    assert batch_artifact_path(tmp_path, 3) == tmp_path / "execution_batch_3.json"


def test_list_batch_artifacts_returns_sorted_existing_paths(tmp_path) -> None:
    batch_three = tmp_path / "execution_batch_3.json"
    batch_one = tmp_path / "execution_batch_1.json"
    batch_two = tmp_path / "execution_batch_2.json"
    for path in (batch_three, batch_one, batch_two):
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "execution_batch_notes.json").write_text("{}", encoding="utf-8")

    assert list_batch_artifacts(tmp_path) == [batch_one, batch_two, batch_three]


def test_compute_global_batches_ignores_completed_status_for_stable_partition() -> None:
    finalize_data = {
        "tasks": [
            {"id": "T1", "status": "done", "depends_on": []},
            {"id": "T2", "status": "pending", "depends_on": ["T1"]},
            {"id": "T3", "status": "skipped", "depends_on": ["T1"]},
            {"id": "T4", "status": "pending", "depends_on": ["T2", "T3"]},
        ]
    }

    assert compute_global_batches(finalize_data) == [["T1"], ["T2", "T3"], ["T4"]]
