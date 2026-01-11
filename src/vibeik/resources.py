from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Dict, Optional

import numpy as np

from .matlab_m_parser import extract_matrix


@dataclass(frozen=True)
class ResourceIndex:
    robots: Dict[str, Path]
    tools: Dict[str, Path]
    display_robot_names: Dict[str, str]
    display_tool_names: Dict[str, str]

    def resolve_robot(self, name: str) -> Optional[Path]:
        return self.robots.get(_normalize_name(name))

    def resolve_tool(self, name: str) -> Optional[Path]:
        return self.tools.get(_normalize_name(name))

    def match_robot(self, name: str) -> Optional[tuple[str, Path]]:
        match = _match_resource(name, self.robots, self.display_robot_names)
        if match:
            return match
        normalized = _normalize_name(name)
        for key, display in self.display_robot_names.items():
            if normalized and normalized in key:
                return display, self.robots[key]
        return None

    def match_tool(self, name: str) -> Optional[tuple[str, Path]]:
        match = _match_resource(name, self.tools, self.display_tool_names)
        if match:
            return match
        if not os.getenv("OPENAI_API_KEY"):
            return None
        if not self.display_tool_names:
            return None
        chosen = _llm_pick_candidate(name, list(self.display_tool_names.values()))
        if not chosen:
            return None
        reverse_map = {display: key for key, display in self.display_tool_names.items()}
        key = reverse_map.get(chosen)
        if not key:
            return None
        path = self.tools.get(key)
        if not path:
            return None
        return chosen, path


@dataclass(frozen=True)
class RobotResource:
    name: str
    dh: np.ndarray


@dataclass(frozen=True)
class ToolResource:
    name: str
    tcp: np.ndarray


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _scan_directory(path: Path) -> Dict[str, Path]:
    resources = {}
    for file in path.glob("*.m"):
        key = _normalize_name(file.stem)
        resources[key] = file
    return resources


def _match_resource(
    name: str, resources: Dict[str, Path], display_names: Dict[str, str]
) -> Optional[tuple[str, Path]]:
    lowered = name.lower()
    for key, display in display_names.items():
        if display.lower() == lowered:
            return display, resources[key]
    normalized = _normalize_name(name)
    if normalized in resources:
        return display_names[normalized], resources[normalized]
    synonyms = {"drilling": "drill"}
    alt_name = name
    for src, dst in synonyms.items():
        alt_name = re.sub(src, dst, alt_name, flags=re.IGNORECASE)
    alt_normalized = _normalize_name(alt_name)
    if alt_normalized in resources:
        return display_names[alt_normalized], resources[alt_normalized]
    return None


def _llm_pick_candidate(query: str, candidates: list[str]) -> Optional[str]:
    if not candidates:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        return None
    client = OpenAI(api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            "tool_id": {
                "anyOf": [
                    {"type": "string", "enum": candidates},
                    {"type": "null"},
                ]
            }
        },
        "required": ["tool_id"],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Select the best matching tool_id from the provided list. "
                    "Return null if none match."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\nCandidates: {candidates}",
            },
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "tool_match", "schema": schema}},
    )
    try:
        payload = json.loads(response.output_text)
    except json.JSONDecodeError:
        return None
    tool_id = payload.get("tool_id")
    if tool_id in candidates:
        return tool_id
    return None


def build_resource_index(base_dir: Path) -> ResourceIndex:
    robots_dir = base_dir / "RobotModels"
    tools_dir = base_dir / "Tools"
    robots = _scan_directory(robots_dir) if robots_dir.exists() else {}
    tools = _scan_directory(tools_dir) if tools_dir.exists() else {}
    display_robot_names = {k: v.stem for k, v in robots.items()}
    display_tool_names = {k: v.stem for k, v in tools.items()}
    return ResourceIndex(robots=robots, tools=tools,
                         display_robot_names=display_robot_names,
                         display_tool_names=display_tool_names)


def load_robot(path: Path) -> RobotResource:
    text = path.read_text()
    try:
        parsed = extract_matrix(text, ["DH", "dh", "DH_table"], expected_shape=(6, 4))
    except ValueError as exc:
        raise ValueError(f"Robot file {path.name} is missing a DH matrix") from exc
    return RobotResource(name=path.stem, dh=parsed.value)


def load_tool(path: Path) -> ToolResource:
    text = path.read_text()
    try:
        parsed = extract_matrix(text, ["T_TCP", "TCP", "tcp", "tool"], expected_shape=(4, 4))
    except ValueError as exc:
        raise ValueError(f"Tool file {path.name} is missing a TCP transform") from exc
    return ToolResource(name=path.stem, tcp=parsed.value)
