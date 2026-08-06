"""Read-only semantic geometry for the Radio navigation diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from typing import Any

import numpy as np

from omnigibson.object_states import ContactBodies
from omnigibson.object_states import OnTop


RADIO_TASK_BINDING = "radio_receiver.n.01_1"
SUPPORT_TABLE_TASK_BINDING = "table.n.02_1"
WORLD_FRAME = "simulator_world"
ROBOT_BASE_FRAME = "robot_base_footprint"
RADIO_FRAME = "radio_base"
SUPPORT_TABLE_FRAME = "support_table_base"
RADIO_UPRIGHT_TOLERANCE_RAD = 0.12
SEARCH_MARGIN_M = 0.50


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def canonical_quaternion_xyzw(value: Any) -> np.ndarray:
    quaternion = _numpy(value).reshape(-1)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must contain four finite xyzw values.")
    magnitude = float(np.linalg.norm(quaternion))
    if magnitude <= np.finfo(np.float64).eps:
        raise ValueError("Quaternion must have nonzero magnitude.")
    quaternion = quaternion / magnitude
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def quaternion_angle_rad(left: Any, right: Any) -> float:
    left_q = canonical_quaternion_xyzw(left)
    right_q = canonical_quaternion_xyzw(right)
    dot = float(np.clip(abs(np.dot(left_q, right_q)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _rotation_matrix_xyzw(value: Any) -> np.ndarray:
    x, y, z, w = canonical_quaternion_xyzw(value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_record(obj: object, *, child_frame: str) -> dict[str, object]:
    position, quaternion = obj.get_position_orientation()
    position_array = _numpy(position).reshape(-1)
    if position_array.shape != (3,) or not np.isfinite(position_array).all():
        raise ValueError(f"{child_frame} position must contain three finite values.")
    return {
        "parent_frame": WORLD_FRAME,
        "child_frame": child_frame,
        "translation_m": position_array.tolist(),
        "quaternion_xyzw": canonical_quaternion_xyzw(quaternion).tolist(),
    }


def _oriented_rectangle_xy(center: Any, quaternion_xyzw: Any, extent_xy: Any) -> list[list[float]]:
    center_array = _numpy(center).reshape(-1)
    extent_array = _numpy(extent_xy).reshape(-1)
    if center_array.size < 2 or extent_array.shape != (2,):
        raise ValueError("Footprint center and extent must contain two XY values.")
    if not np.isfinite(center_array[:2]).all() or not np.isfinite(extent_array).all() or np.any(extent_array <= 0):
        raise ValueError("Footprint center and extent must be finite and positive.")
    half_x, half_y = 0.5 * extent_array
    local = np.asarray(
        [[-half_x, -half_y, 0.0], [half_x, -half_y, 0.0], [half_x, half_y, 0.0], [-half_x, half_y, 0.0]],
        dtype=np.float64,
    )
    world = local @ _rotation_matrix_xyzw(quaternion_xyzw).T
    world[:, :2] += center_array[:2]
    return world[:, :2].tolist()


def _collision_box(obj: object, *, frame: str) -> dict[str, object]:
    center, quaternion, extent, _ = obj.get_base_aligned_bbox(visual=False, xy_aligned=True)
    center_array = _numpy(center).reshape(-1)
    extent_array = _numpy(extent).reshape(-1)
    if center_array.shape != (3,) or extent_array.shape != (3,):
        raise ValueError(f"{frame} collision box must be three-dimensional.")
    quaternion_array = canonical_quaternion_xyzw(quaternion)
    return {
        "source_api": "get_base_aligned_bbox(visual=False,xy_aligned=True)",
        "parent_frame": WORLD_FRAME,
        "box_frame": frame,
        "center_m": center_array.tolist(),
        "quaternion_xyzw": quaternion_array.tolist(),
        "extent_m": extent_array.tolist(),
        "footprint_polygon_world_xy_m": _oriented_rectangle_xy(center_array, quaternion_array, extent_array[:2]),
    }


def _robot_collision_footprint(robot: object) -> dict[str, object]:
    position, quaternion = robot.get_position_orientation()
    position_array = _numpy(position).reshape(-1)
    extent = _numpy(robot.reset_joint_pos_aabb_extent).reshape(-1)
    if position_array.shape != (3,) or extent.size < 2 or not np.isfinite(extent[:2]).all():
        raise ValueError("Robot base footprint API returned malformed geometry.")
    quaternion_array = canonical_quaternion_xyzw(quaternion)
    return {
        "source_api": "robot.reset_joint_pos_aabb_extent[:2]",
        "parent_frame": WORLD_FRAME,
        "footprint_frame": ROBOT_BASE_FRAME,
        "center_m": position_array.tolist(),
        "quaternion_xyzw": quaternion_array.tolist(),
        "extent_xy_m": extent[:2].tolist(),
        "polygon_world_xy_m": _oriented_rectangle_xy(position_array, quaternion_array, extent[:2]),
    }


def _task_bound_object(env: object, binding: str) -> tuple[object, dict[str, object]]:
    scope = getattr(getattr(env, "task", None), "object_scope", None)
    if not isinstance(scope, Mapping) or binding not in scope:
        raise KeyError(f"Required BDDL task object binding is absent: {binding}")
    resolved = scope[binding]
    if getattr(resolved, "exists", True) is not True:
        raise RuntimeError(f"Required BDDL task object does not exist: {binding}")
    obj = getattr(resolved, "unwrapped", resolved)
    return obj, {
        "resolution": "exact_task_object_scope_binding",
        "task_binding": binding,
        "name": str(getattr(obj, "name", "")),
        "category": str(getattr(obj, "category", "")),
        "prim_path": str(getattr(obj, "prim_path", "")),
    }


def _link_paths(obj: object) -> set[str]:
    links = getattr(obj, "links", {})
    values = links.values() if isinstance(links, Mapping) else links
    return {str(getattr(link, "prim_path", link)) for link in values}


def _contact_paths(obj: object) -> list[str]:
    bodies = obj.states[ContactBodies].get_value()
    return sorted(str(getattr(body, "prim_path", body)) for body in bodies)


def _contact_report(robot: object, radio: object, table: object) -> dict[str, object]:
    subjects = {"robot": robot, "radio": radio, "support_table": table}
    owned_paths = {name: _link_paths(obj) for name, obj in subjects.items()}
    paths_by_subject = {name: _contact_paths(obj) for name, obj in subjects.items()}
    pairs: set[tuple[str, str, str]] = set()
    for subject, paths in paths_by_subject.items():
        for path in paths:
            other = next((name for name, owned in owned_paths.items() if name != subject and path in owned), None)
            label = other if other is not None else f"other:{path}"
            left, right = sorted((subject, label))
            pairs.add((left, right, path))
    return {
        "source_api": "ContactBodies.get_value()",
        "paths_by_subject": paths_by_subject,
        "pairs": [{"body_a": left, "body_b": right, "contact_body_path": path} for left, right, path in sorted(pairs)],
    }


def capture_navigation_geometry(
    env: object,
    robot: object,
    *,
    backend_capabilities: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Capture semantic state using task bindings and read-only simulator APIs."""

    radio, radio_identity = _task_bound_object(env, RADIO_TASK_BINDING)
    table, table_identity = _task_bound_object(env, SUPPORT_TABLE_TASK_BINDING)
    radio_pose = _pose_record(radio, child_frame=RADIO_FRAME)
    table_pose = _pose_record(table, child_frame=SUPPORT_TABLE_FRAME)
    base_pose = _pose_record(robot, child_frame=ROBOT_BASE_FRAME)
    radio_rotation = _rotation_matrix_xyzw(radio_pose["quaternion_xyzw"])
    upright_tilt_rad = math.acos(float(np.clip(radio_rotation[2, 2], -1.0, 1.0)))
    table_box = _collision_box(table, frame=SUPPORT_TABLE_FRAME)
    radio_box = _collision_box(radio, frame=RADIO_FRAME)
    robot_footprint = _robot_collision_footprint(robot)
    support = bool(radio.states[OnTop].get_value(table))
    contacts = _contact_report(robot, radio, table)

    points = np.asarray(
        [
            base_pose["translation_m"][:2],
            radio_pose["translation_m"][:2],
            *table_box["footprint_polygon_world_xy_m"],
        ],
        dtype=np.float64,
    )
    lower = np.min(points, axis=0) - SEARCH_MARGIN_M
    upper = np.max(points, axis=0) + SEARCH_MARGIN_M
    capabilities: dict[str, Mapping[str, object]] = {
        "semantic_pose_and_footprint": {"status": "available"},
        "contact_and_support_state": {"status": "available"},
        "hypothetical_base_pose_whole_arm_ik": {
            "status": "unavailable",
            "reason": "The current runtime exposes only local current-configuration linearized IK.",
        },
        "arm_trajectory_collision": {
            "status": "unavailable",
            "reason": "No read-only whole-corridor collision API is exposed by the current runtime.",
        },
        "arm_table_clearance": {
            "status": "unavailable",
            "reason": "No read-only articulated-link distance query is exposed by the current runtime.",
        },
    }
    if backend_capabilities is not None:
        capabilities.update(backend_capabilities)
    missing_capability_note = (
        "Whole-arm hypothetical-pose IK, collision, and arm-table clearance are unavailable."
        if any(
            capabilities[name].get("status") != "available"
            for name in (
                "hypothetical_base_pose_whole_arm_ik",
                "arm_trajectory_collision",
                "arm_table_clearance",
            )
        )
        else "CuRobo whole-arm IK, articulated collision, and arm-table clearance are available in diagnostic mode."
    )
    return {
        "frames": {
            "world": WORLD_FRAME,
            "robot_base": ROBOT_BASE_FRAME,
            "radio": RADIO_FRAME,
            "support_table": SUPPORT_TABLE_FRAME,
            "quaternion_order": "xyzw",
            "pose_convention": "T_parent_child",
        },
        "robot": {"semantic_role": "robot", "pose": base_pose, "footprint": robot_footprint},
        "radio": {
            "semantic_role": "task_radio",
            "identity": radio_identity,
            "pose": radio_pose,
            "collision_box": radio_box,
            "upright": upright_tilt_rad <= RADIO_UPRIGHT_TOLERANCE_RAD,
            "upright_tilt_rad": upright_tilt_rad,
            "upright_tolerance_rad": RADIO_UPRIGHT_TOLERANCE_RAD,
            "supported_by_task_table": support,
        },
        "support_table": {
            "semantic_role": "task_support_table",
            "identity": table_identity,
            "pose": table_pose,
            "collision_box": table_box,
        },
        "contacts": contacts,
        "navigation_model": {
            "scope": "bounded_support_table_footprint_model",
            "search_bounds_world_xy_m": [lower.tolist(), upper.tolist()],
            "obstacle_polygons_world_xy_m": [
                table_box["footprint_polygon_world_xy_m"],
                radio_box["footprint_polygon_world_xy_m"],
            ],
            "limitations": [
                "Only the current task Radio and support-table collision footprints are semantic dynamic obstacles.",
                missing_capability_note,
            ],
        },
        "capabilities": capabilities,
    }


def navigation_geometry_deltas(baseline: Mapping[str, object], current: Mapping[str, object]) -> dict[str, object]:
    def pose_delta(key: str) -> dict[str, object]:
        baseline_pose = baseline[key]["pose"]
        current_pose = current[key]["pose"]
        baseline_position = np.asarray(baseline_pose["translation_m"], dtype=np.float64)
        current_position = np.asarray(current_pose["translation_m"], dtype=np.float64)
        delta = current_position - baseline_position
        return {
            "translation_world_m": delta.tolist(),
            "translation_xy_norm_m": float(np.linalg.norm(delta[:2])),
            "translation_norm_m": float(np.linalg.norm(delta)),
            "orientation_angle_rad": quaternion_angle_rad(
                baseline_pose["quaternion_xyzw"], current_pose["quaternion_xyzw"]
            ),
        }

    baseline_pairs = baseline["contacts"]["pairs"]
    current_pairs = current["contacts"]["pairs"]
    return {
        "robot": pose_delta("robot"),
        "radio": pose_delta("radio"),
        "support_table": pose_delta("support_table"),
        "radio_support_changed": (
            baseline["radio"]["supported_by_task_table"] != current["radio"]["supported_by_task_table"]
        ),
        "contact_pairs_changed": baseline_pairs != current_pairs,
        "baseline_contact_pairs": copy.deepcopy(baseline_pairs),
        "current_contact_pairs": copy.deepcopy(current_pairs),
    }
