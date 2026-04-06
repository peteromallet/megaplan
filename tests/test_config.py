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
