"""Bounded read-only kinematics helpers for agentic control."""

from __future__ import annotations

import numpy as np


def _canonicalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a quaternion and choose one deterministic representative."""

    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("EEF quaternion must have nonzero magnitude.")
    normalized = quaternion / norm
    if normalized[3] < 0 or (normalized[3] == 0 and next((value for value in normalized[:3] if value != 0), 0.0) < 0):
        normalized = -normalized
    return normalized


def _axis_angle_to_quaternion_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle <= np.finfo(np.float64).eps:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    half_angle = 0.5 * angle
    return np.concatenate((axis_angle * (np.sin(half_angle) / angle), [np.cos(half_angle)]))


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_xyz, left_w = left[:3], left[3]
    right_xyz, right_w = right[:3], right[3]
    return np.concatenate(
        (
            left_w * right_xyz + right_w * left_xyz + np.cross(left_xyz, right_xyz),
            [left_w * right_w - np.dot(left_xyz, right_xyz)],
        )
    )


def _quaternion_to_axis_angle_xyzw(quaternion: np.ndarray) -> np.ndarray:
    shortest = _canonicalize_quaternion_xyzw(quaternion)
    vector_norm = float(np.linalg.norm(shortest[:3]))
    if vector_norm <= np.finfo(np.float64).eps:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, shortest[3])
    return shortest[:3] * (angle / vector_norm)


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


def bounded_pose_delta_ik_step(
    *,
    joint_positions: np.ndarray,
    jacobian: np.ndarray,
    eef_position: np.ndarray,
    eef_quaternion_xyzw: np.ndarray,
    target_position: np.ndarray,
    orientation_delta_eef_axis_angle_rad: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    max_target_delta_m: float,
    max_orientation_delta_rad: float,
    max_joint_delta_rad: float,
) -> dict[str, np.ndarray | float | bool]:
    """Compute one bounded local pose-delta IK target without changing state.

    Quaternions use ``xyzw`` ordering and represent EEF orientation in the
    robot-base frame. The orientation request is body-fixed: its target is
    ``R_current @ R_delta_eef``. The angular Jacobian is robot-base-frame, so
    the solved angular error is the shortest arc from
    ``R_target @ R_current.T``. This is not a trajectory or collision planner.
    """

    q = np.asarray(joint_positions, dtype=np.float64)
    j = np.asarray(jacobian, dtype=np.float64)
    eef = np.asarray(eef_position, dtype=np.float64)
    eef_quaternion = np.asarray(eef_quaternion_xyzw, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    requested_orientation = np.asarray(orientation_delta_eef_axis_angle_rad, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    expected_shapes = {
        "joint_positions": (q.shape, (7,)),
        "eef_position": (eef.shape, (3,)),
        "eef_quaternion_xyzw": (eef_quaternion.shape, (4,)),
        "target_position": (target.shape, (3,)),
        "orientation_delta_eef_axis_angle_rad": (requested_orientation.shape, (3,)),
        "joint_lower": (lower.shape, (7,)),
        "joint_upper": (upper.shape, (7,)),
    }
    malformed = [name for name, (actual, expected) in expected_shapes.items() if actual != expected]
    if malformed:
        raise ValueError(f"R1Pro bounded pose IK received malformed shapes for: {', '.join(malformed)}.")
    if j.shape != (6, 7):
        raise ValueError(f"R1Pro EEF Jacobian must have shape (6, 7), got {j.shape}.")
    arrays = (q, j, eef, eef_quaternion, target, requested_orientation, lower, upper)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Bounded pose IK inputs must be finite.")
    limits = (max_target_delta_m, max_orientation_delta_rad, max_joint_delta_rad)
    if not all(np.isfinite(value) and value > 0 for value in limits):
        raise ValueError("Bounded pose IK limits must be finite and positive.")
    if np.any(lower > upper):
        raise ValueError("Joint lower limits exceed upper limits.")
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("Current joint positions must be inside physical joint limits.")

    current_quaternion = _canonicalize_quaternion_xyzw(eef_quaternion)

    requested_translation = target - eef
    requested_distance = float(np.linalg.norm(requested_translation))
    translation_clipped = requested_distance > max_target_delta_m
    used_translation = requested_translation.copy()
    if translation_clipped:
        used_translation *= max_target_delta_m / requested_distance

    requested_orientation_magnitude = float(np.linalg.norm(requested_orientation))
    orientation_clipped = requested_orientation_magnitude > max_orientation_delta_rad
    used_orientation_eef = requested_orientation.copy()
    if orientation_clipped:
        used_orientation_eef *= max_orientation_delta_rad / requested_orientation_magnitude

    delta_quaternion_eef = _axis_angle_to_quaternion_xyzw(used_orientation_eef)
    target_quaternion_raw = _quaternion_multiply_xyzw(current_quaternion, delta_quaternion_eef)
    target_quaternion = _canonicalize_quaternion_xyzw(target_quaternion_raw)
    current_conjugate = np.concatenate((-current_quaternion[:3], current_quaternion[3:]))
    robot_error_quaternion = _quaternion_multiply_xyzw(target_quaternion_raw, current_conjugate)
    used_orientation_robot = _quaternion_to_axis_angle_xyzw(robot_error_quaternion)

    pose_error = np.concatenate((used_translation, used_orientation_robot))
    damping = 1e-4
    normal = j @ j.T + damping * np.eye(6, dtype=np.float64)
    solved_joint_delta = j.T @ np.linalg.solve(normal, pose_error)
    max_abs_delta = float(np.max(np.abs(solved_joint_delta)))
    joint_delta_clipped = max_abs_delta > max_joint_delta_rad
    bounded_joint_delta = solved_joint_delta.copy()
    if joint_delta_clipped:
        bounded_joint_delta *= max_joint_delta_rad / max_abs_delta
    unclipped_joint_target = q + bounded_joint_delta
    target_joint_positions = np.clip(unclipped_joint_target, lower, upper)
    joint_limit_clipped = not np.array_equal(unclipped_joint_target, target_joint_positions)

    return {
        "joint_target": target_joint_positions.astype(np.float32),
        "joint_delta": (target_joint_positions - q).astype(np.float32),
        "current_eef_position_robot_m": eef.astype(np.float32),
        "current_eef_quaternion_robot_xyzw": current_quaternion.astype(np.float32),
        "target_eef_quaternion_robot_xyzw": target_quaternion.astype(np.float32),
        "requested_translation_delta_robot_m": requested_translation.astype(np.float32),
        "used_translation_delta_robot_m": used_translation.astype(np.float32),
        "requested_orientation_delta_eef_axis_angle_rad": requested_orientation.astype(np.float32),
        "used_orientation_delta_eef_axis_angle_rad": used_orientation_eef.astype(np.float32),
        "used_orientation_delta_robot_axis_angle_rad": used_orientation_robot.astype(np.float32),
        "requested_translation_magnitude_m": requested_distance,
        "used_translation_magnitude_m": float(np.linalg.norm(used_translation)),
        "requested_orientation_magnitude_rad": requested_orientation_magnitude,
        "used_orientation_magnitude_rad": float(np.linalg.norm(used_orientation_eef)),
        "translation_delta_clipped": translation_clipped,
        "orientation_delta_clipped": orientation_clipped,
        "joint_delta_clipped": joint_delta_clipped,
        "joint_limit_clipped": joint_limit_clipped,
    }
