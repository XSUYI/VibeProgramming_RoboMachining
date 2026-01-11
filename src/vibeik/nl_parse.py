from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI

from .types import Orientation, ParsedInstruction, TargetPosition


SYSTEM_PROMPT = """You are a helpful parser for robot inverse kinematics instructions.
Return ONLY JSON that matches the schema exactly."""


def parse_instruction(text: str) -> ParsedInstruction:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return _parse_with_llm(text, api_key)
    return _parse_with_fallback(text)


def _parse_with_llm(text: str, api_key: str) -> ParsedInstruction:
    client = OpenAI(api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            "robot_model": {"type": "string"},
            "tool_name": {"type": "string"},
            "target": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                },
                "required": ["x", "y", "z"],
                "additionalProperties": False,
            },
            "orientation": {
                "type": ["object", "null"],
                "properties": {
                    "roll": {"type": ["number", "null"]},
                    "pitch": {"type": ["number", "null"]},
                    "yaw": {"type": ["number", "null"]},
                    "quaternion": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "keyword": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            "task": {"type": ["string", "null"]},
        },
        "required": ["robot_model", "tool_name", "target"],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "ik_intent", "schema": schema}},
    )
    payload = json.loads(response.output_text)
    orientation = None
    if payload.get("orientation"):
        orientation = Orientation(**payload["orientation"])
    return ParsedInstruction(
        robot_model=payload["robot_model"],
        tool_name=payload["tool_name"],
        target=TargetPosition(**payload["target"]),
        orientation=orientation,
        task=payload.get("task"),
    )


def _parse_with_fallback(text: str) -> ParsedInstruction:
    robot_match = re.search(r"(?:using|use)\s+the\s+(.+?)\s+robot", text, re.IGNORECASE)
    if not robot_match:
        robot_match = re.search(r"(?:using|use)\s+(.+?)\s+robot", text, re.IGNORECASE)
    tool_match = None
    tool_candidates = list(re.finditer(r"(?:an|a)\s+([A-Za-z0-9_ .-]+?)\s+tool", text, re.IGNORECASE))
    if tool_candidates:
        tool_match = tool_candidates[-1]
    position_match = re.search(r"\[([^\]]+)\]", text)

    if not (robot_match and tool_match and position_match):
        raise ValueError("Unable to parse instruction without OPENAI_API_KEY")

    robot_model = robot_match.group(1).strip()
    tool_name = tool_match.group(1).strip()
    coords = [coord.strip() for coord in position_match.group(1).split(",")]
    values = []
    for coord in coords:
        coord = coord.replace("m", "").strip()
        values.append(float(coord))
    if len(values) != 3:
        raise ValueError("Expected three coordinates for target position")

    return ParsedInstruction(
        robot_model=robot_model,
        tool_name=tool_name,
        target=TargetPosition(x=values[0], y=values[1], z=values[2]),
        orientation=None,
        task=None,
    )
