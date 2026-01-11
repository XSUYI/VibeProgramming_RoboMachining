from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ik import solve_ik
from .kinematics import make_transform, rotation_from_orientation
from .nl_parse import parse_instruction
from .resources import build_resource_index, load_robot, load_tool
from .types import SolveResponse


BASE_DIR = Path(__file__).resolve().parents[2]
RESOURCES_DIR = BASE_DIR / "RobotResources"


def run(text: str) -> SolveResponse:
    index = build_resource_index(RESOURCES_DIR)
    try:
        parsed = parse_instruction(text)
    except ValueError as exc:
        return SolveResponse(ok=False, warnings=[str(exc)])

    robot_match = index.match_robot(parsed.robot_model)
    if not robot_match:
        return SolveResponse(ok=False, warnings=[f"Unknown robot model: {parsed.robot_model}"])
    robot_name, robot_path = robot_match

    tool_match = index.match_tool(parsed.tool_name)
    if not tool_match:
        return SolveResponse(ok=False, warnings=[f"Unknown tool: {parsed.tool_name}"])
    tool_name, tool_path = tool_match

    try:
        robot = load_robot(robot_path)
        tool = load_tool(tool_path)
    except ValueError as exc:
        return SolveResponse(ok=False, warnings=[str(exc)])

    rotation = rotation_from_orientation(parsed.orientation)
    translation = np.array([parsed.target.x, parsed.target.y, parsed.target.z])
    target = make_transform(rotation, translation)

    flange_target = target @ np.linalg.inv(tool.tcp)
    ik_result = solve_ik(robot.dh, flange_target)
    if not ik_result.ok:
        return SolveResponse(
            ok=False,
            warnings=[ik_result.warning or "IK failed"],
            meta={
                "robot_model": robot_name,
                "tool_name": tool_name,
                "residual_error": ik_result.residual_error,
                "solver": "ikine_LM",
            },
        )

    joint_angles = ik_result.joint_angles.tolist() if ik_result.joint_angles is not None else []
    joint_angles_deg = [float(angle * 180.0 / np.pi) for angle in joint_angles]
    return SolveResponse(
        ok=True,
        joint_angles_rad=joint_angles,
        joint_angles_deg=joint_angles_deg,
        warnings=[],
        meta={
            "robot_model": robot_name,
            "tool_name": tool_name,
            "residual_error": ik_result.residual_error,
            "solver": "ikine_LM",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibe IK Assistant CLI")
    parser.add_argument("text", help="Natural language instruction")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    response = run(args.text)
    if args.json:
        print(response.json())
        return

    if not response.ok:
        print("Warning: " + "; ".join(response.warnings))
        return

    print("Joint angles (rad):", response.joint_angles_rad)
    print("Joint angles (deg):", response.joint_angles_deg)


if __name__ == "__main__":
    main()
