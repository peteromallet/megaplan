from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import megaplan
import megaplan.cli as cli_module
import megaplan._core.io as io_module
from megaplan._core import get_effective
from megaplan.types import DEFAULTS


@pytest.fixture
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_path = tmp_path / ".config" / "megaplan"

    def fake_config_dir(home: Path | None = None) -> Path:
        del home
        return config_path

    monkeypatch.setattr(io_module, "config_dir", fake_config_dir)
    monkeypatch.setattr(cli_module, "config_dir", fake_config_dir)
    return config_path


def test_get_effective_returns_default(isolated_config_dir: Path) -> None:
    assert not (isolated_config_dir / "config.json").exists()
    assert get_effective("execution", "worker_timeout_seconds") == DEFAULTS["execution.worker_timeout_seconds"]


def test_get_effective_returns_override(isolated_config_dir: Path) -> None:
    isolated_config_dir.mkdir(parents=True, exist_ok=True)
    (isolated_config_dir / "config.json").write_text(
        json.dumps({"execution": {"worker_timeout_seconds": 1234}}),
        encoding="utf-8",
    )

    assert get_effective("execution", "worker_timeout_seconds") == 1234


def test_config_set_numeric(isolated_config_dir: Path) -> None:
    response = megaplan.handle_config(
        Namespace(
            config_action="set",
            key="execution.worker_timeout_seconds",
            value="3600",
        )
    )

    assert response["success"] is True
    assert response["value"] == 3600
    saved = json.loads((isolated_config_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["execution"]["worker_timeout_seconds"] == 3600
    assert isinstance(saved["execution"]["worker_timeout_seconds"], int)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("YeS", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("No", False),
        ("off", False),
    ],
)
def test_config_set_execution_auto_approve_bool_tokens(
    isolated_config_dir: Path,
    value: str,
    expected: bool,
) -> None:
    response = megaplan.handle_config(
        Namespace(
            config_action="set",
            key="execution.auto_approve",
            value=value,
        )
    )

    assert response["success"] is True
    assert response["value"] is expected
    saved = json.loads((isolated_config_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["execution"]["auto_approve"] is expected


def test_config_set_execution_auto_approve_invalid_token(isolated_config_dir: Path) -> None:
    with pytest.raises(
        megaplan.CliError,
        match=r"execution\.auto_approve must be one of: true, false, 1, 0, yes, no, on, off",
    ):
        megaplan.handle_config(
            Namespace(
                config_action="set",
                key="execution.auto_approve",
                value="maybe",
            )
        )


@pytest.mark.parametrize("value", ["tiny", "light", "standard", "robust", "superrobust"])
def test_config_set_execution_robustness_enum(
    isolated_config_dir: Path,
    value: str,
) -> None:
    response = megaplan.handle_config(
        Namespace(
            config_action="set",
            key="execution.robustness",
            value=value,
        )
    )

    assert response["success"] is True
    assert response["value"] == value
    saved = json.loads((isolated_config_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["execution"]["robustness"] == value


def test_config_set_execution_robustness_invalid_value(isolated_config_dir: Path) -> None:
    with pytest.raises(
        megaplan.CliError,
        match=r"execution\.robustness must be one of: tiny, light, standard, robust, superrobust",
    ):
        megaplan.handle_config(
            Namespace(
                config_action="set",
                key="execution.robustness",
                value="turbo",
            )
        )


def test_config_set_invalid_key(isolated_config_dir: Path) -> None:
    with pytest.raises(megaplan.CliError, match=r"Unknown config key 'foo\.bar'"):
        megaplan.handle_config(
            Namespace(
                config_action="set",
                key="foo.bar",
                value="1",
            )
        )


def test_config_set_invalid_type(isolated_config_dir: Path) -> None:
    with pytest.raises(
        megaplan.CliError,
        match=r"execution\.worker_timeout_seconds must be an integer",
    ):
        megaplan.handle_config(
            Namespace(
                config_action="set",
                key="execution.worker_timeout_seconds",
                value="notanumber",
            )
        )


def test_config_set_orchestration_mode(isolated_config_dir: Path) -> None:
    response = megaplan.handle_config(
        Namespace(
            config_action="set",
            key="orchestration.mode",
            value="inline",
        )
    )

    assert response["success"] is True
    assert response["value"] == "inline"
    saved = json.loads((isolated_config_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["orchestration"]["mode"] == "inline"


def test_config_set_orchestration_mode_invalid(isolated_config_dir: Path) -> None:
    with pytest.raises(
        megaplan.CliError,
        match=r"orchestration\.mode must be 'inline' or 'subagent'",
    ):
        megaplan.handle_config(
            Namespace(
                config_action="set",
                key="orchestration.mode",
                value="bogus",
            )
        )


def test_build_parser_init_flags_are_tristate() -> None:
    from megaplan.cli import build_parser

    parser = build_parser()

    parsed = parser.parse_args(["init", "--project-dir", "/tmp", "idea"])
    explicit = parser.parse_args(
        ["init", "--project-dir", "/tmp", "--auto-approve", "--robustness", "robust", "idea"]
    )

    assert parsed.auto_approve is None
    assert parsed.robustness is None
    assert explicit.auto_approve is True
    assert explicit.robustness == "robust"


def test_handle_init_uses_config_defaults_when_flags_omitted(isolated_config_dir: Path, tmp_path: Path) -> None:
    isolated_config_dir.mkdir(parents=True, exist_ok=True)
    (isolated_config_dir / "config.json").write_text(
        json.dumps({"execution": {"auto_approve": True, "robustness": "robust"}}),
        encoding="utf-8",
    )
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()

    response = megaplan.handle_init(
        root,
        Namespace(
            project_dir=str(project_dir),
            name="config-backed-init",
            auto_approve=None,
            robustness=None,
            hermes=None,
            phase_model=[],
            idea="idea",
        ),
    )
    state = json.loads((root / ".megaplan" / "plans" / response["plan"] / "state.json").read_text(encoding="utf-8"))

    assert response["auto_approve"] is True
    assert response["robustness"] == "robust"
    assert state["config"]["auto_approve"] is True
    assert state["config"]["robustness"] == "robust"


def test_handle_init_explicit_robustness_beats_config_default(
    isolated_config_dir: Path,
    tmp_path: Path,
) -> None:
    isolated_config_dir.mkdir(parents=True, exist_ok=True)
    (isolated_config_dir / "config.json").write_text(
        json.dumps({"execution": {"auto_approve": True, "robustness": "robust"}}),
        encoding="utf-8",
    )
    root = tmp_path / "root"
    project_dir = tmp_path / "project"
    root.mkdir()
    project_dir.mkdir()

    response = megaplan.handle_init(
        root,
        Namespace(
            project_dir=str(project_dir),
            name="explicit-robustness-init",
            auto_approve=None,
            robustness="light",
            hermes=None,
            phase_model=[],
            idea="idea",
        ),
    )
    state = json.loads((root / ".megaplan" / "plans" / response["plan"] / "state.json").read_text(encoding="utf-8"))

    assert response["auto_approve"] is True
    assert response["robustness"] == "light"
    assert state["config"]["auto_approve"] is True
    assert state["config"]["robustness"] == "light"
