from __future__ import annotations

import pytest
import torch as th

from omnigibson.learning.agentic.semantic_geometry_backend import INTERPOLATION_MAX_JOINT_DELTA_RAD
from omnigibson.learning.agentic.semantic_geometry_backend import RadioSemanticGeometryBackend
from omnigibson.learning.agentic.semantic_geometry_backend import _resolve_curobo_device
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_clearance
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_ik_solution
from omnigibson.macros import gm


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


def test_curobo_device_uses_explicit_omnigibson_gpu():
    previous = gm.GPU_ID
    try:
        with gm.unlocked():
            gm.GPU_ID = "3"
        assert _resolve_curobo_device(None) == "cuda:3"
    finally:
        with gm.unlocked():
            gm.GPU_ID = previous


def test_curobo_device_rejects_physics_cpu_device():
    with pytest.raises(ValueError, match="indexed CUDA device"):
        _resolve_curobo_device("cpu")


def test_curobo_device_requires_explicit_gpu_selection():
    previous = gm.GPU_ID
    try:
        with gm.unlocked():
            gm.GPU_ID = None
        with pytest.raises(RuntimeError, match="OMNIGIBSON_GPU_ID"):
            _resolve_curobo_device(None)
    finally:
        with gm.unlocked():
            gm.GPU_ID = previous


def test_candidate_joint_state_encodes_world_pose_relative_to_immutable_root():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.robot = _Robot()

    result = backend._candidate_joint_state(_pose([13.0, 25.0, 0.0]))

    th.testing.assert_close(result[:6], th.tensor([3.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
    th.testing.assert_close(backend.robot.get_joint_positions(), th.zeros(8))


def test_candidate_joint_state_is_copied_to_curobo_device_before_ik():
    class _TensorArgs:
        def __init__(self):
            self.values = []

        def to_device(self, value):
            self.values.append(value.clone())
            return value + 10.0

    class _MotionGenerator:
        def __init__(self):
            self.tensor_args = _TensorArgs()

    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.robot = _Robot()
    backend.motion_generator = _MotionGenerator()

    result = backend._candidate_joint_state_for_curobo(_pose([13.0, 25.0, 0.0]))

    expected = th.tensor([3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    th.testing.assert_close(backend.motion_generator.tensor_args.values[0], expected)
    th.testing.assert_close(result, expected + 10.0)
    th.testing.assert_close(backend.robot.get_joint_positions(), th.zeros(8))


def test_corridor_ik_seed_generator_reset_is_scoped_to_arm_solver():
    class _IkSolver:
        def __init__(self):
            self.reset_calls = 0

        def reset_seed(self):
            self.reset_calls += 1

    arm_solver = _IkSolver()
    default_solver = _IkSolver()
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.motion_generator = type(
        "_MotionGenerator",
        (),
        {
            "mg": {
                "arm": type("_ArmMotionGenerator", (), {"ik_solver": arm_solver})(),
                "default": type("_DefaultMotionGenerator", (), {"ik_solver": default_solver})(),
            }
        },
    )()

    backend._reset_ik_seed_generator()

    assert arm_solver.reset_calls == 1
    assert default_solver.reset_calls == 0


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


def test_ik_goal_uses_existing_full_joint_state_without_reaugmenting_locked_joints():
    class _MotionGenerator:
        def __init__(self):
            self.calls = []

        def path_to_joint_trajectory(self, joint_state, *, get_full_js, emb_sel):
            self.calls.append((joint_state, get_full_js, emb_sel))
            return th.arange(8, dtype=th.float32)

    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.motion_generator = _MotionGenerator()
    joint_state = object()

    result = backend._ik_goal_joint_positions(joint_state, expected_shape=th.Size([8]))

    th.testing.assert_close(result, th.arange(8, dtype=th.float32))
    assert backend.motion_generator.calls == [(joint_state, False, "arm")]


def test_ik_goal_rejects_incomplete_joint_state():
    class _MotionGenerator:
        def path_to_joint_trajectory(self, *args, **kwargs):
            return th.zeros(7)

    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.motion_generator = _MotionGenerator()

    with pytest.raises(RuntimeError, match="expected \\(8,\\)"):
        backend._ik_goal_joint_positions(object(), expected_shape=th.Size([8]))


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


def test_ik_summary_retains_auditable_joint_values_instead_of_an_opaque_digest():
    result = _summarize_ik_solution(th.tensor([0.1, -0.2, 0.3]))

    assert result["status"] == "available"
    assert result["feasible"] is True
    assert result["solution_joint_positions_rad"] == pytest.approx([0.1, -0.2, 0.3])


def test_ik_summary_rejects_nonfinite_or_nonvector_solutions():
    with pytest.raises(ValueError, match="finite joint vector"):
        _summarize_ik_solution(th.tensor([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite joint vector"):
        _summarize_ik_solution(th.tensor([0.0, float("nan")]))
