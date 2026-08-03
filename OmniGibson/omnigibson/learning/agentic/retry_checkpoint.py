"""Task-local integrity checks for reusable agentic retry checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


RADIO_PRIMARY_MAX_POSITION_DRIFT_M = 0.03
RADIO_PRIMARY_MAX_ORIENTATION_DRIFT_DEG = 12.0
RADIO_PRIMARY_MAX_LINEAR_SPEED_M_S = 0.03
RADIO_PRIMARY_MAX_ANGULAR_SPEED_RAD_S = 0.15
RADIO_PRECLOSE_MAX_POSITION_DRIFT_M = 0.03
RADIO_PRECLOSE_MAX_ORIENTATION_DRIFT_DEG = 12.0
RADIO_PRECLOSE_MAX_LINEAR_SPEED_M_S = 0.01
RADIO_PRECLOSE_MAX_ANGULAR_SPEED_RAD_S = 0.05
RADIO_PRECLOSE_MAX_TABLE_DRIFT_M = 0.005
RADIO_PRECLOSE_MAX_TABLE_DRIFT_DEG = 2.0
RADIO_PRECLOSE_MAX_BASE_DRIFT_M = 0.005
RADIO_PRECLOSE_MAX_BASE_DRIFT_DEG = 2.0
RADIO_PRECLOSE_MIN_OPEN_GRIPPER_COMMAND = 0.90


def assess_radio_demo_primary_checkpoint(
    initial: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Assess hidden Radio state without exposing it through the agent tool surface."""

    initial_position = _vector(initial, "position", 3)
    candidate_position = _vector(candidate, "position", 3)
    initial_orientation = _unit_quaternion(initial, "orientation")
    candidate_orientation = _unit_quaternion(candidate, "orientation")
    linear_velocity = _vector(candidate, "linear_velocity", 3)
    angular_velocity = _vector(candidate, "angular_velocity", 3)

    position_drift_m = float(np.linalg.norm(candidate_position - initial_position))
    orientation_drift_deg = _orientation_drift_deg(initial_orientation, candidate_orientation)
    linear_speed_m_s = float(np.linalg.norm(linear_velocity))
    angular_speed_rad_s = float(np.linalg.norm(angular_velocity))
    on_table = bool(candidate.get("on_table", False))
    grasped = bool(candidate.get("grasping_left", False) or candidate.get("grasping_right", False))

    checks = {
        "on_table": on_table,
        "ungrasped": not grasped,
        "position_near_initial": position_drift_m <= RADIO_PRIMARY_MAX_POSITION_DRIFT_M,
        "orientation_near_initial": orientation_drift_deg <= RADIO_PRIMARY_MAX_ORIENTATION_DRIFT_DEG,
        "linear_velocity_stationary": linear_speed_m_s <= RADIO_PRIMARY_MAX_LINEAR_SPEED_M_S,
        "angular_velocity_stationary": angular_speed_rad_s <= RADIO_PRIMARY_MAX_ANGULAR_SPEED_RAD_S,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "measurements": {
            "position_drift_m": position_drift_m,
            "orientation_drift_deg": orientation_drift_deg,
            "linear_speed_m_s": linear_speed_m_s,
            "angular_speed_rad_s": angular_speed_rad_s,
        },
        "limits": {
            "max_position_drift_m": RADIO_PRIMARY_MAX_POSITION_DRIFT_M,
            "max_orientation_drift_deg": RADIO_PRIMARY_MAX_ORIENTATION_DRIFT_DEG,
            "max_linear_speed_m_s": RADIO_PRIMARY_MAX_LINEAR_SPEED_M_S,
            "max_angular_speed_rad_s": RADIO_PRIMARY_MAX_ANGULAR_SPEED_RAD_S,
        },
    }


def assess_radio_pickup_preclose_checkpoint(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Validate a source-pinned, stationary right-hand Radio pre-close state."""

    radio_position_drift_m = float(
        np.linalg.norm(_vector(candidate, "position", 3) - _vector(reference, "position", 3))
    )
    radio_orientation_drift_deg = _orientation_drift_deg(
        _unit_quaternion(reference, "orientation"),
        _unit_quaternion(candidate, "orientation"),
    )
    table_position_drift_m = float(
        np.linalg.norm(_vector(candidate, "table_position", 3) - _vector(reference, "table_position", 3))
    )
    table_orientation_drift_deg = _orientation_drift_deg(
        _unit_quaternion(reference, "table_orientation"),
        _unit_quaternion(candidate, "table_orientation"),
    )
    base_position_drift_m = float(
        np.linalg.norm(_vector(candidate, "base_position", 3) - _vector(reference, "base_position", 3))
    )
    base_orientation_drift_deg = _orientation_drift_deg(
        _unit_quaternion(reference, "base_orientation"),
        _unit_quaternion(candidate, "base_orientation"),
    )
    radio_linear_speed_m_s = float(np.linalg.norm(_vector(candidate, "linear_velocity", 3)))
    radio_angular_speed_rad_s = float(np.linalg.norm(_vector(candidate, "angular_velocity", 3)))
    table_linear_speed_m_s = float(np.linalg.norm(_vector(candidate, "table_linear_velocity", 3)))
    table_angular_speed_rad_s = float(np.linalg.norm(_vector(candidate, "table_angular_velocity", 3)))
    base_linear_speed_m_s = float(np.linalg.norm(_vector(candidate, "base_linear_velocity", 3)))
    base_angular_speed_rad_s = float(np.linalg.norm(_vector(candidate, "base_angular_velocity", 3)))

    contact_paths = _string_set(candidate, "contact_body_paths")
    table_link_paths = _string_set(candidate, "table_link_paths")
    assisted = candidate.get("assisted_grasp_attachments")
    if not isinstance(assisted, Mapping):
        raise ValueError("Radio checkpoint state requires assisted_grasp_attachments mapping.")
    no_assisted_attachment = all(value is None for value in assisted.values())
    grasped = bool(candidate.get("grasping_left", False) or candidate.get("grasping_right", False))
    right_gripper_command = float(candidate.get("right_gripper_command", float("nan")))
    if not math.isfinite(right_gripper_command):
        raise ValueError("Radio checkpoint state requires finite right_gripper_command.")

    checks = {
        "robot_near_radio": bool(candidate.get("robot_near_radio", False)),
        "radio_not_picked_up": not bool(candidate.get("radio_picked_up", True)),
        "radio_off": not bool(candidate.get("radio_on", True)),
        "on_table": bool(candidate.get("on_table", False)),
        "ungrasped": not grasped,
        "no_assisted_attachment": no_assisted_attachment,
        "table_only_contact": bool(contact_paths) and contact_paths <= table_link_paths,
        "radio_position_near_reference": radio_position_drift_m <= RADIO_PRECLOSE_MAX_POSITION_DRIFT_M,
        "radio_orientation_near_reference": (radio_orientation_drift_deg <= RADIO_PRECLOSE_MAX_ORIENTATION_DRIFT_DEG),
        "radio_linear_velocity_stationary": radio_linear_speed_m_s <= RADIO_PRECLOSE_MAX_LINEAR_SPEED_M_S,
        "radio_angular_velocity_stationary": radio_angular_speed_rad_s <= RADIO_PRECLOSE_MAX_ANGULAR_SPEED_RAD_S,
        "table_position_stable": table_position_drift_m <= RADIO_PRECLOSE_MAX_TABLE_DRIFT_M,
        "table_orientation_stable": table_orientation_drift_deg <= RADIO_PRECLOSE_MAX_TABLE_DRIFT_DEG,
        "table_linear_velocity_stationary": table_linear_speed_m_s <= RADIO_PRECLOSE_MAX_LINEAR_SPEED_M_S,
        "table_angular_velocity_stationary": table_angular_speed_rad_s <= RADIO_PRECLOSE_MAX_ANGULAR_SPEED_RAD_S,
        "base_position_stable": base_position_drift_m <= RADIO_PRECLOSE_MAX_BASE_DRIFT_M,
        "base_orientation_stable": base_orientation_drift_deg <= RADIO_PRECLOSE_MAX_BASE_DRIFT_DEG,
        "base_linear_velocity_stationary": base_linear_speed_m_s <= RADIO_PRECLOSE_MAX_LINEAR_SPEED_M_S,
        "base_angular_velocity_stationary": base_angular_speed_rad_s <= RADIO_PRECLOSE_MAX_ANGULAR_SPEED_RAD_S,
        "right_gripper_open": right_gripper_command >= RADIO_PRECLOSE_MIN_OPEN_GRIPPER_COMMAND,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "measurements": {
            "radio_position_drift_m": radio_position_drift_m,
            "radio_orientation_drift_deg": radio_orientation_drift_deg,
            "radio_linear_speed_m_s": radio_linear_speed_m_s,
            "radio_angular_speed_rad_s": radio_angular_speed_rad_s,
            "table_position_drift_m": table_position_drift_m,
            "table_orientation_drift_deg": table_orientation_drift_deg,
            "table_linear_speed_m_s": table_linear_speed_m_s,
            "table_angular_speed_rad_s": table_angular_speed_rad_s,
            "base_position_drift_m": base_position_drift_m,
            "base_orientation_drift_deg": base_orientation_drift_deg,
            "base_linear_speed_m_s": base_linear_speed_m_s,
            "base_angular_speed_rad_s": base_angular_speed_rad_s,
            "right_gripper_command": right_gripper_command,
            "contact_body_paths": sorted(contact_paths),
        },
        "limits": {
            "max_radio_position_drift_m": RADIO_PRECLOSE_MAX_POSITION_DRIFT_M,
            "max_radio_orientation_drift_deg": RADIO_PRECLOSE_MAX_ORIENTATION_DRIFT_DEG,
            "max_linear_speed_m_s": RADIO_PRECLOSE_MAX_LINEAR_SPEED_M_S,
            "max_angular_speed_rad_s": RADIO_PRECLOSE_MAX_ANGULAR_SPEED_RAD_S,
            "max_table_position_drift_m": RADIO_PRECLOSE_MAX_TABLE_DRIFT_M,
            "max_table_orientation_drift_deg": RADIO_PRECLOSE_MAX_TABLE_DRIFT_DEG,
            "max_base_position_drift_m": RADIO_PRECLOSE_MAX_BASE_DRIFT_M,
            "max_base_orientation_drift_deg": RADIO_PRECLOSE_MAX_BASE_DRIFT_DEG,
            "min_open_gripper_command": RADIO_PRECLOSE_MIN_OPEN_GRIPPER_COMMAND,
        },
    }


def _vector(state: Mapping[str, object], key: str, width: int) -> np.ndarray:
    value = np.asarray(state.get(key), dtype=np.float64)
    if value.shape != (width,) or not np.isfinite(value).all():
        raise ValueError(f"Radio checkpoint state requires finite {key} with shape ({width},).")
    return value


def _unit_quaternion(state: Mapping[str, object], key: str) -> np.ndarray:
    value = _vector(state, key, 4)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError(f"Radio checkpoint state requires a nonzero {key} quaternion.")
    return value / norm


def _orientation_drift_deg(first: np.ndarray, second: np.ndarray) -> float:
    quaternion_dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(quaternion_dot))


def _string_set(state: Mapping[str, object], key: str) -> set[str]:
    value = state.get(key)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Radio checkpoint state requires {key} as a string sequence.")
    return set(value)
