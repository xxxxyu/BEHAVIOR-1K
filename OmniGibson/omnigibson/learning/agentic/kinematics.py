"""Bounded read-only kinematics helpers for agentic control."""

from __future__ import annotations

import numpy as np


def bounded_position_ik_step(
    *,
    joint_positions: np.ndarray,
    jacobian: np.ndarray,
    eef_position: np.ndarray,
    target_position: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    max_target_delta_m: float,
    max_joint_delta_rad: float,
) -> dict[str, np.ndarray | float | bool]:
    """Compute one bounded damped-least-squares position IK update.

    This is a local target generator, not a trajectory planner. It keeps the
    current end-effector orientation by supplying zero rotational error and
    never mutates simulator or controller state.
    """

    q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    j = np.asarray(jacobian, dtype=np.float64)
    eef = np.asarray(eef_position, dtype=np.float64).reshape(-1)
    target = np.asarray(target_position, dtype=np.float64).reshape(-1)
    lower = np.asarray(joint_lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(joint_upper, dtype=np.float64).reshape(-1)
    if q.size != 7 or eef.size != 3 or target.size != 3 or lower.size != 7 or upper.size != 7:
        raise ValueError("R1Pro bounded IK requires 7 arm joints and 3-D EEF positions.")
    if j.shape != (6, 7):
        raise ValueError(f"R1Pro EEF Jacobian must have shape (6, 7), got {j.shape}.")
    if not all(np.isfinite(value).all() for value in (q, j, eef, target, lower, upper)):
        raise ValueError("Bounded IK inputs must be finite.")
    if max_target_delta_m <= 0 or max_joint_delta_rad <= 0:
        raise ValueError("Bounded IK limits must be positive.")
    if np.any(lower > upper):
        raise ValueError("Joint lower limits exceed upper limits.")

    requested_delta = target - eef
    requested_distance = float(np.linalg.norm(requested_delta))
    target_clipped = requested_distance > max_target_delta_m
    used_delta = requested_delta.copy()
    if target_clipped:
        used_delta *= max_target_delta_m / requested_distance

    pose_error = np.concatenate((used_delta, np.zeros(3, dtype=np.float64)))
    damping = 1e-4
    normal = j @ j.T + damping * np.eye(6, dtype=np.float64)
    joint_delta = j.T @ np.linalg.solve(normal, pose_error)
    max_abs_delta = float(np.max(np.abs(joint_delta)))
    joint_delta_clipped = max_abs_delta > max_joint_delta_rad
    if joint_delta_clipped:
        joint_delta *= max_joint_delta_rad / max_abs_delta
    target_joint_positions = np.clip(q + joint_delta, lower, upper)

    return {
        "joint_target": target_joint_positions.astype(np.float32),
        "joint_delta": (target_joint_positions - q).astype(np.float32),
        "used_target_position": (eef + used_delta).astype(np.float32),
        "requested_distance_m": requested_distance,
        "used_distance_m": float(np.linalg.norm(used_delta)),
        "target_delta_clipped": target_clipped,
        "joint_delta_clipped": joint_delta_clipped,
    }
