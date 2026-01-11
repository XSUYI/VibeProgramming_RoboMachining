from __future__ import annotations

import os

import numpy as np

from vibeik.kinematics import default_rotation, rotation_from_orientation
from vibeik.nl_parse import parse_instruction


EXAMPLE_TEXT = (
    "I am using the KUKA KR120 R2500 robot. "
    "I want to move the tooltip of an 8mm drilling tool to [1.5m, 0.1m, 1.0m]"
)


def test_default_orientation_matrix():
    rotation = rotation_from_orientation(None)
    assert np.allclose(rotation, default_rotation())


def test_fallback_parser_extracts_fields(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    parsed = parse_instruction(EXAMPLE_TEXT)
    assert parsed.robot_model == "KUKA KR120 R2500"
    assert parsed.tool_name == "8mm drilling"
    assert parsed.target.x == 1.5
    assert parsed.target.y == 0.1
    assert parsed.target.z == 1.0
