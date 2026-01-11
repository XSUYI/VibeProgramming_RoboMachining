from __future__ import annotations

from vibeik.cli import run


def test_unknown_robot_returns_warning(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = run("Use the UnknownBot robot with a Drill_8mm tool at [0, 0, 0]")
    assert response.ok is False
    assert any("Unknown robot" in warning for warning in response.warnings)


def test_unknown_tool_returns_warning(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = run("Use the KR120R2500 robot with a Mystery tool at [0, 0, 0]")
    assert response.ok is False
    assert any("Unknown tool" in warning for warning in response.warnings)
