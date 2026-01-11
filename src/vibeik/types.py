from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class TargetPosition(BaseModel):
    x: float
    y: float
    z: float


class Orientation(BaseModel):
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    quaternion: Optional[List[float]] = None
    keyword: Optional[str] = None


class ParsedInstruction(BaseModel):
    robot_model: str
    tool_name: str
    target: TargetPosition
    orientation: Optional[Orientation] = None
    task: Optional[str] = None


class SolveRequest(BaseModel):
    text: str = Field(..., description="Natural language instruction")


class SolveMeta(BaseModel):
    robot_model: Optional[str] = None
    tool_name: Optional[str] = None
    residual_error: Optional[float] = None
    solver: Optional[str] = None
    notes: Optional[List[str]] = None
    raw: Optional[Any] = None


class SolveResponse(BaseModel):
    ok: bool
    joint_angles_rad: Optional[List[float]] = None
    joint_angles_deg: Optional[List[float]] = None
    warnings: List[str] = Field(default_factory=list)
    meta: SolveMeta = Field(default_factory=SolveMeta)
