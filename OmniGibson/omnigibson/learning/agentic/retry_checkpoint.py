"""Task-local integrity checks for reusable agentic retry checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


RADIO_PRIMARY_MAX_POSITION_DRIFT_M = 0.03
RADIO_PRIMARY_MAX_ORIENTATION_DRIFT_DEG = 12.0
RADIO_PRIMARY_MAX_LINEAR_SPEED_M_S = 0.03
RADIO_PRIMARY_MAX_ANGULAR_SPEED_RAD_S = 0.15


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
    quaternion_dot = float(np.clip(abs(np.dot(initial_orientation, candidate_orientation)), 0.0, 1.0))
    orientation_drift_deg = math.degrees(2.0 * math.acos(quaternion_dot))
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
