from __future__ import annotations

from types import SimpleNamespace

from evals.benchmarks.swe_bench import read_prompt


def _prepared(problem_statement: str, fail_to_pass: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        instance={"problem_statement": problem_statement},
        metadata={"fail_to_pass": fail_to_pass},
    )


def test_read_prompt_includes_problem_statement_and_verification_tests() -> None:
    prompt = read_prompt(
        _prepared(
            "Fix widget parsing for empty input.",
            [
                "tests/test_widget.py::test_empty_input",
                "widgets.tests.ParserTests.test_handles_whitespace_only",
            ],
        )
    )

    assert "Fix widget parsing for empty input." in prompt
    assert "VERIFICATION TESTS" in prompt
    assert "tests/test_widget.py::test_empty_input" in prompt
    assert "widgets.tests.ParserTests.test_handles_whitespace_only" in prompt
    assert "do NOT create new test files" in prompt
    assert "MUST include a final task" in prompt


def test_read_prompt_handles_empty_fail_to_pass_list() -> None:
    prompt = read_prompt(_prepared("Handle missing verification metadata.", []))

    assert "Handle missing verification metadata." in prompt
    assert "VERIFICATION TESTS" not in prompt
    assert "No specific FAIL_TO_PASS verification tests were provided." in prompt
