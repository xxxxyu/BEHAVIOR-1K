from __future__ import annotations

import copy

import numpy as np
import pytest

from omnigibson.learning.agentic.runtime_server import AgenticEnvironmentWebsocketServer
from omnigibson.learning.agentic.runtime_server import AgenticEvaluatorRuntime


class _FakeController:
    dof_idx = np.arange(7)
    control_type = "position"
    _control_limits = {"position": (np.full(7, -2.0), np.full(7, 2.0))}


class _FakeRobot:
    def __init__(self) -> None:
        self.controllers = {"arm_right": _FakeController()}
        jacobian = np.zeros((6, 7), dtype=np.float32)
        jacobian[:, :6] = np.eye(6, dtype=np.float32)
        self.control_dict = {
            "joint_position": np.zeros(7, dtype=np.float32),
            "eef_right_jacobian_relative": jacobian,
            "eef_right_pos_relative": np.asarray([0.1, -0.2, 0.5], dtype=np.float32),
            "eef_right_quat_relative": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        self.actions: list[np.ndarray] = []

    def get_control_dict(self):
        return self.control_dict


def test_plan_eef_pose_delta_runtime_is_read_only_and_public_only():
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.robot = _FakeRobot()
    runtime.evaluator = type("Evaluator", (), {"env": type("Environment", (), {"_current_step": 41})()})()
    before_control = copy.deepcopy(runtime.robot.control_dict)
    before_step = runtime.evaluator.env._current_step

    result = runtime.plan_eef_pose_delta(
        arm="right",
        target_position=[0.1, -0.2, 0.5],
        orientation_delta_eef_axis_angle_rad=[0.0, 0.0, 0.08],
    )

    assert runtime.evaluator.env._current_step == before_step
    assert runtime.robot.actions == []
    for key, value in before_control.items():
        np.testing.assert_array_equal(runtime.robot.control_dict[key], value)
    assert result["collision_checked"] is False
    assert result["frame_policy"]["orientation_policy"] == "bounded_eef_frame_pose_delta"
    assert set(result) == {
        "arm",
        "controller_segment",
        "joint_target",
        "joint_delta",
        "current_eef_position_robot_m",
        "current_eef_quaternion_robot_xyzw",
        "requested_translation_delta_robot_m",
        "used_translation_delta_robot_m",
        "requested_orientation_delta_eef_axis_angle_rad",
        "used_orientation_delta_eef_axis_angle_rad",
        "used_orientation_delta_robot_axis_angle_rad",
        "requested_translation_magnitude_m",
        "used_translation_magnitude_m",
        "requested_orientation_magnitude_rad",
        "used_orientation_magnitude_rad",
        "translation_delta_clipped",
        "orientation_delta_clipped",
        "joint_delta_clipped",
        "joint_limit_clipped",
        "frame_policy",
        "collision_checked",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_target_delta_m": 0.051}, r"\(0, 0.05\]"),
        ({"max_orientation_delta_rad": 0.351}, r"\(0, 0.35\]"),
        ({"max_joint_delta_rad": 0.351}, r"\(0, 0.35\]"),
    ],
)
def test_plan_eef_pose_delta_runtime_enforces_hard_caps(overrides, message):
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.robot = _FakeRobot()
    arguments = {
        "arm": "right",
        "target_position": [0.1, -0.2, 0.5],
        "orientation_delta_eef_axis_angle_rad": [0.0, 0.0, 0.08],
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        runtime.plan_eef_pose_delta(**arguments)

    assert runtime.robot.actions == []


def test_plan_eef_pose_delta_websocket_dispatch_calls_planner_without_action():
    class FakeRuntime:
        def __init__(self) -> None:
            self.requests = []

        def plan_eef_pose_delta(self, **request):
            self.requests.append(request)
            return {"joint_target": [0.0] * 7, "collision_checked": False}

    runtime = FakeRuntime()
    server = AgenticEnvironmentWebsocketServer.__new__(AgenticEnvironmentWebsocketServer)
    server.runtime = runtime

    result = server._dispatch(
        {
            "operation": "plan_eef_pose_delta",
            "arm": "right",
            "target_position": [0.0, 0.0, 0.0],
            "orientation_delta_eef_axis_angle_rad": [0.08, 0.0, 0.0],
        }
    )

    assert result == {"joint_target": [0.0] * 7, "collision_checked": False}
    assert runtime.requests == [
        {
            "arm": "right",
            "target_position": [0.0, 0.0, 0.0],
            "orientation_delta_eef_axis_angle_rad": [0.08, 0.0, 0.0],
            "max_target_delta_m": 0.03,
            "max_orientation_delta_rad": 0.17,
            "max_joint_delta_rad": 0.12,
        }
    ]
