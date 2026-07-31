from __future__ import annotations

import pytest

from controller_manifest import ControllerManifestError
from controller_manifest import build_controller_manifest
from controller_manifest import validate_runtime_action
from controller_manifest import validate_r1pro_manifest


class FakeController:
    def __init__(self, *, dim, dof_idx, motor_type, input_limits, output_limits=None, physical_limits=None):
        self.command_dim = dim
        self.dof_idx = dof_idx
        self._motor_type = motor_type
        self.control_type = {"position": 0, "velocity": 1}[motor_type]
        self._use_delta_commands = False
        self.command_input_limits = input_limits
        self.command_output_limits = output_limits
        if physical_limits is not None:
            self._control_limits = {self.control_type: physical_limits}


class R1Pro:
    name = "robot_r1"
    action_dim = 23
    control_freq = 30.0
    _action_normalize = False
    controller_order = ("base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right")

    def __init__(self):
        position_lower = [-3.0] * 28
        position_upper = [3.0] * 28
        position_limits = (position_lower, position_upper)
        self.controllers = {
            "base": FakeController(
                dim=3,
                dof_idx=[0, 1, 5],
                motor_type="velocity",
                input_limits=([-1.0] * 3, [1.0] * 3),
                output_limits=([-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]),
            ),
            "trunk": FakeController(
                dim=4,
                dof_idx=[6, 7, 8, 9],
                motor_type="position",
                input_limits=None,
                physical_limits=position_limits,
            ),
            "arm_left": FakeController(
                dim=7,
                dof_idx=list(range(10, 17)),
                motor_type="position",
                input_limits=None,
                physical_limits=position_limits,
            ),
            "gripper_left": FakeController(
                dim=1,
                dof_idx=[17, 18],
                motor_type="position",
                input_limits=([-1.0], [1.0]),
                output_limits=([-0.05], [0.05]),
            ),
            "arm_right": FakeController(
                dim=7,
                dof_idx=list(range(19, 26)),
                motor_type="position",
                input_limits=None,
                physical_limits=position_limits,
            ),
            "gripper_right": FakeController(
                dim=1,
                dof_idx=[26, 27],
                motor_type="position",
                input_limits=([-1.0], [1.0]),
                output_limits=([-0.05], [0.05]),
            ),
        }
        self.controller_action_idx = {}
        start = 0
        for name in self.controller_order:
            stop = start + self.controllers[name].command_dim
            self.controller_action_idx[name] = list(range(start, stop))
            start = stop


def test_build_and_validate_r1pro_manifest():
    manifest = build_controller_manifest(R1Pro(), physics_frequency_hz=120.0)

    validate_r1pro_manifest(manifest)
    assert manifest.action_dim == 23
    assert [segment.name for segment in manifest.segments] == list(R1Pro.controller_order)
    assert manifest.segments[0].semantic == "normalized_velocity"
    assert manifest.segments[1].safety_limit_source == "physical_position_limits"
    assert manifest.segments[2].safety_input_limits == ((-3.0,) * 7, (3.0,) * 7)
    assert manifest.segments[3].semantic == "normalized_gripper"
    assert manifest.fingerprint == manifest.to_dict()["fingerprint"]


def test_rejects_noncontiguous_runtime_action_indices():
    robot = R1Pro()
    robot.controller_action_idx["arm_left"] = list(range(8, 15))

    with pytest.raises(ControllerManifestError, match="not the expected contiguous slice"):
        build_controller_manifest(robot, physics_frequency_hz=120.0)


def test_rejects_unbounded_nonposition_controller():
    robot = R1Pro()
    robot.controllers["base"].command_input_limits = None

    with pytest.raises(ControllerManifestError, match="no bounded input contract"):
        build_controller_manifest(robot, physics_frequency_hz=120.0)


def test_r1pro_validation_rejects_global_action_normalization():
    robot = R1Pro()
    robot._action_normalize = True
    manifest = build_controller_manifest(robot, physics_frequency_hz=120.0)

    with pytest.raises(ControllerManifestError, match="action_normalize=False"):
        validate_r1pro_manifest(manifest)


def test_environment_side_action_validation_uses_manifest_limits():
    manifest = build_controller_manifest(R1Pro(), physics_frequency_hz=120.0)
    action = [0.0] * 23

    assert validate_runtime_action(manifest, action).shape == (23,)
    action[0] = 1.01
    with pytest.raises(ControllerManifestError, match=r"base\[0\]"):
        validate_runtime_action(manifest, action)
