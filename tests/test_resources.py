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
