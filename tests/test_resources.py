from __future__ import annotations

from pathlib import Path

import numpy as np

from vibeik.resources import build_resource_index, load_robot, load_tool


BASE_DIR = Path(__file__).resolve().parents[1]
RESOURCES_DIR = BASE_DIR / "RobotResources"


def test_resource_scanning_finds_known_assets():
    index = build_resource_index(RESOURCES_DIR)
    assert index.match_robot("KR120R2500") is not None
    assert index.match_tool("Drill_8mm") is not None


def test_resource_loader_shapes():
    index = build_resource_index(RESOURCES_DIR)
    _, robot_path = index.match_robot("KR120R2500")
    _, tool_path = index.match_tool("Drill_8mm")
    robot = load_robot(robot_path)
    tool = load_tool(tool_path)
    assert robot.dh.shape == (6, 4)
    assert tool.tcp.shape == (4, 4)
    assert np.isfinite(robot.dh).all()
    assert np.isfinite(tool.tcp).all()


def _write_tool_file(path: Path) -> None:
    path.write_text("T_TCP = eye(4);")


def test_tool_match_uses_llm_resolver_when_enabled(monkeypatch, tmp_path):
    tools_dir = tmp_path / "Tools"
    tools_dir.mkdir()
    tool_path = tools_dir / "Drill_8mm.m"
    _write_tool_file(tool_path)
    index = build_resource_index(tmp_path)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("vibeik.resources._llm_pick_candidate", lambda query, candidates: "Drill_8mm")

    match = index.match_tool("8mm drilling tool")

    assert match == ("Drill_8mm", tool_path)


def test_tool_match_skips_llm_when_api_key_missing(monkeypatch, tmp_path):
    tools_dir = tmp_path / "Tools"
    tools_dir.mkdir()
    tool_path = tools_dir / "Drill_8mm.m"
    _write_tool_file(tool_path)
    index = build_resource_index(tmp_path)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _raise_if_called(query, candidates):
        raise AssertionError("LLM resolver should not be called without OPENAI_API_KEY")

    monkeypatch.setattr("vibeik.resources._llm_pick_candidate", _raise_if_called)

    match = index.match_tool("8mm drilling tool")

    assert match is None
