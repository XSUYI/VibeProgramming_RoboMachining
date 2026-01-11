from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class IKResult:
    ok: bool
    joint_angles: Optional[np.ndarray]
    residual_error: Optional[float]
    warning: Optional[str]


def _build_robot_from_dh(dh: np.ndarray):
    import roboticstoolbox as rtb

    links: List[rtb.RevoluteDH] = []
    for alpha, a, theta, d in dh:
        links.append(rtb.RevoluteDH(a=a, alpha=alpha, d=d, offset=theta))
    return rtb.DHRobot(links, name="Robot")


def solve_ik(dh: np.ndarray, target: np.ndarray) -> IKResult:
    major_version = int(np.__version__.split(".")[0])
    if major_version >= 2:
        return IKResult(
            ok=False,
            joint_angles=None,
            residual_error=None,
            warning="roboticstoolbox-python is incompatible with NumPy 2.x in this environment",
        )
    from spatialmath import SE3

    robot = _build_robot_from_dh(dh)
    target_se3 = SE3(target)
    q0 = np.zeros(robot.n)
    solution = robot.ikine_LM(target_se3, q0=q0)
    success = getattr(solution, "success", False)
    q = getattr(solution, "q", None)
    if not success or q is None:
        return IKResult(ok=False, joint_angles=None, residual_error=None, warning="IK solver failed")

    fk = robot.fkine(q)
    position_error = np.linalg.norm(fk.t - target_se3.t)
    residual = float(position_error)
    if residual > 1e-3:
        return IKResult(ok=False, joint_angles=None, residual_error=residual, warning="Target unreachable or residual too large")

    jacobian = robot.jacob0(q)
    condition = np.linalg.cond(jacobian)
    if condition > 1e6:
        return IKResult(ok=False, joint_angles=None, residual_error=residual, warning="Solution near singularity")

    return IKResult(ok=True, joint_angles=q, residual_error=residual, warning=None)
