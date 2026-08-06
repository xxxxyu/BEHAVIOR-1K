from __future__ import annotations

import pytest
import torch as th

from omnigibson.learning.agentic.semantic_geometry_backend import INTERPOLATION_MAX_JOINT_DELTA_RAD
from omnigibson.learning.agentic.semantic_geometry_backend import RadioSemanticGeometryBackend
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_clearance


class _PoseLink:
    def __init__(self, position, orientation=(0.0, 0.0, 0.0, 1.0)):
        self.position = th.tensor(position, dtype=th.float32)
        self.orientation = th.tensor(orientation, dtype=th.float32)

    def get_position_orientation(self):
        return self.position.clone(), self.orientation.clone()


class _Robot:
    def __init__(self):
        self.base_idx = th.arange(6)
        self.root_link = _PoseLink([10.0, 20.0, 0.0])
        self.base_pose = _PoseLink([11.0, 22.0, 0.0])
        self.eef_links = {"left": _PoseLink([12.0, 22.0, 1.0])}
        self.q = th.zeros(8, dtype=th.float32)

    def get_joint_positions(self):
        return self.q.clone()

    def get_position_orientation(self):
        return self.base_pose.get_position_orientation()


def _pose(position):
    return {
        "parent_frame": "simulator_world",
        "child_frame": "robot_base_footprint",
        "translation_m": position,
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_candidate_joint_state_encodes_world_pose_relative_to_immutable_root():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.robot = _Robot()

    result = backend._candidate_joint_state(_pose([13.0, 25.0, 0.0]))

    th.testing.assert_close(result[:6], th.tensor([3.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
    th.testing.assert_close(backend.robot.get_joint_positions(), th.zeros(8))


def test_left_hold_pose_is_carried_with_hypothetical_base_instead_of_world_fixed():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.robot = _Robot()

    position, orientation = backend._left_hold_pose(_pose([14.0, 26.0, 0.0]))

    th.testing.assert_close(position, th.tensor([15.0, 26.0, 1.0]))
    th.testing.assert_close(orientation, th.tensor([0.0, 0.0, 0.0, 1.0]))


def test_joint_interpolation_respects_declared_maximum_increment():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    trajectory = backend._interpolated(th.zeros(3), th.tensor([0.061, -0.02, 0.0]))

    assert trajectory.shape == (4, 3)
    assert float(th.max(th.abs(th.diff(trajectory, dim=0)))) <= INTERPOLATION_MAX_JOINT_DELTA_RAD


def test_clearance_summary_uses_the_minimum_sample_for_saturation_semantics():
    spheres = th.tensor([[[[0.0, 0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 0.02]]]])
    result = _summarize_clearance(
        th.tensor([[[-1.0, -0.04]]]),
        spheres,
        trajectory_samples=1,
    )

    assert result["minimum_clearance_m"] == pytest.approx(0.02)
    assert result["clearance_is_lower_bound"] is False
    assert result["minimum_sample_esdf_saturated"] is False


def test_clearance_summary_marks_a_far_saturated_minimum_as_lower_bound():
    spheres = th.tensor([[[[0.0, 0.0, 0.0, 0.05]]]])
    result = _summarize_clearance(
        th.tensor([[[-1.0]]]),
        spheres,
        trajectory_samples=1,
    )

    assert result["minimum_clearance_m"] == pytest.approx(0.95)
    assert result["clearance_is_lower_bound"] is True
    assert result["minimum_sample_esdf_saturated"] is True
