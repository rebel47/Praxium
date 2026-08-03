from __future__ import annotations

import json

from praxium.cli.main import main


def test_doctor_reports_required_environment(capsys: object) -> None:
    assert main(["doctor", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["ok"] is True
    assert output["checks"]["python"]["ok"] is True


def test_graph_command_renders_import_target(capsys: object) -> None:
    assert main(["graph", "examples.server:graph", "--output", "mermaid"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "flowchart TD" in output
    assert 'echo["Echo"]' in output


def test_plugins_command_handles_empty_discovery(capsys: object) -> None:
    assert main(["plugins", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert isinstance(output, list)
