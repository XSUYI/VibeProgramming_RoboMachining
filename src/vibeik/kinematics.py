from __future__ import annotations

import numpy as np

from .types import Orientation


def default_rotation() -> np.ndarray:
    """Return the default tool rotation when orientation is omitted.

    Tool axes in base frame:
    - z_tool = [0, 0, -1]
    - x_tool = [1, 0, 0]
    - y_tool = [0, -1, 0]
    """
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation and translation."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rotation_from_orientation(orientation: Orientation | None) -> np.ndarray:
    """Convert an optional orientation into a rotation matrix."""
    if orientation is None:
        return default_rotation()
    if orientation.quaternion:
        q = orientation.quaternion
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )
    if orientation.roll is not None and orientation.pitch is not None and orientation.yaw is not None:
        cr = np.cos(orientation.roll)
        sr = np.sin(orientation.roll)
        cp = np.cos(orientation.pitch)
        sp = np.sin(orientation.pitch)
        cy = np.cos(orientation.yaw)
        sy = np.sin(orientation.yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
    return default_rotation()
