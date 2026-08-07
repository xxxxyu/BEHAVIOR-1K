from __future__ import annotations

import pytest
import torch as th

from omnigibson.learning.agentic.semantic_geometry_backend import INTERPOLATION_MAX_JOINT_DELTA_RAD
from omnigibson.learning.agentic.semantic_geometry_backend import DIAGNOSTIC_SOLVER_INPUT_QUANTUM
from omnigibson.learning.agentic.semantic_geometry_backend import RadioSemanticGeometryBackend
from omnigibson.learning.agentic.semantic_geometry_backend import _quantize_pose
from omnigibson.learning.agentic.semantic_geometry_backend import _quantize_tensor
from omnigibson.learning.agentic.semantic_geometry_backend import _resolve_curobo_device
from omnigibson.learning.agentic.semantic_geometry_backend import _first_positive_sample
from omnigibson.learning.agentic.semantic_geometry_backend import _max_positive_value
from omnigibson.learning.agentic.semantic_geometry_backend import _pose_residual
from omnigibson.learning.agentic.semantic_geometry_backend import _stage_collision_policy
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_clearance
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_ik_solution
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_per_sphere_collision
from omnigibson.learning.agentic.semantic_geometry_backend import _summarize_retained_lift_motion
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


def test_collision_channel_summary_preserves_first_sample_and_positive_maximum():
    values = th.tensor([-0.2, 0.0, 0.03, 0.01], dtype=th.float32)

    assert _first_positive_sample(values) == 2
    assert _max_positive_value(values) == pytest.approx(0.03)
    assert _first_positive_sample(th.tensor([-1.0, 0.0])) is None
    assert _max_positive_value(th.tensor([-1.0, 0.0])) is None


def test_collision_channel_summary_accepts_serialized_scores():
    values = [[-0.2], [0.0], [0.03], [0.01]]

    assert _first_positive_sample(values) == 2
    assert _max_positive_value(values) == pytest.approx(0.03)


def test_per_sphere_collision_summary_preserves_sample_and_link_attribution():
    sample_scores, summary = _summarize_per_sphere_collision(
        th.tensor([[[0.0, 0.0]], [[0.2, 0.5]], [[0.0, 0.1]]]),
        sphere_to_link={0: "arm", 1: "finger"},
    )

    th.testing.assert_close(sample_scores, th.tensor([0.0, 0.7, 0.1]))
    assert summary == {
        "detected": True,
        "first_collision_sample": 1,
        "maximum_constraint_score": pytest.approx(0.7),
        "worst_sphere_index": 1,
        "worst_sphere_link": "finger",
        "worst_sphere_maximum_score": pytest.approx(0.5),
    }


def test_stage_collision_policy_allows_only_late_exact_target_contact():
    channels = {
        "target_object_collision": {"detected": True, "first_collision_sample": 27},
        "non_target_world_collision": {"detected": False},
        "self_collision": {"detected": False},
    }

    preclose = _stage_collision_policy("preclose", channels, contact_phase_start_sample=19)
    assert preclose["collision_safe_for_stage"] is True
    assert preclose["target_contact_allowed"] is True

    early = {
        **channels,
        "target_object_collision": {"detected": True, "first_collision_sample": 18},
    }
    rejected = _stage_collision_policy("preclose", early, contact_phase_start_sample=19)
    assert rejected["collision_safe_for_stage"] is False
    assert rejected["blocking_reasons"] == ["target_contact_before_preclose_contact_phase"]

    pregrasp = _stage_collision_policy("pregrasp", channels, contact_phase_start_sample=None)
    assert pregrasp["collision_safe_for_stage"] is False


def test_stage_collision_policy_never_suppresses_non_target_or_self_collision():
    channels = {
        "target_object_collision": {"detected": True, "first_collision_sample": 27},
        "non_target_world_collision": {"detected": True},
        "self_collision": {"detected": True},
    }

    result = _stage_collision_policy("lift_1", channels, contact_phase_start_sample=None)

    assert result["target_contact_allowed"] is True
    assert result["collision_safe_for_stage"] is False
    assert result["blocking_reasons"] == ["self_collision", "non_target_world_collision"]


def test_retained_lift_motion_uses_relative_positive_z_and_cumulative_xy_contract():
    gate = {
        "stage_names": ["lift_1", "lift_2"],
        "lift_count": 2,
        "axis": "robot_base_footprint:+z",
        "increment_max_m": 0.025,
        "actions_per_lift_max": 18,
        "cumulative_eef_xy_drift_max_m": 0.01,
        "radio_local_feature_following_required": True,
    }

    accepted = _summarize_retained_lift_motion(
        "lift_1",
        th.tensor([0.56876, -0.17226, 0.63901]),
        th.tensor([0.56632, -0.17154, 0.66346]),
        th.tensor([0.56876, -0.17226, 0.63901]),
        interpolation_steps=18,
        gate=gate,
    )

    assert accepted["accepted"] is True
    assert accepted["z_rise_m"] == pytest.approx(0.02445, abs=1e-6)
    assert accepted["displacement_m"] < 0.025
    assert accepted["cumulative_eef_xy_drift_from_preclose_m"] < 0.01

    rejected = _summarize_retained_lift_motion(
        "lift_2",
        th.tensor([0.56632, -0.17154, 0.66346]),
        th.tensor([0.58000, -0.17154, 0.66200]),
        th.tensor([0.56876, -0.17226, 0.63901]),
        interpolation_steps=19,
        gate=gate,
    )
    assert rejected["accepted"] is False
    assert rejected["blocking_reasons"] == [
        "lift_does_not_move_in_positive_base_z",
        "cumulative_eef_xy_drift_exceeds_max",
        "lift_interpolation_steps_exceed_max",
    ]


def test_pose_residual_aligns_target_dtype_with_fk_output():
    result = _pose_residual(
        th.tensor([1.0, 2.0, 3.0], dtype=th.float64),
        th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float64),
        th.tensor([1.0, 2.0, 3.0], dtype=th.float32),
        th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )

    assert result == {"translation_m": 0.0, "orientation_rad": 0.0}


@pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA to reproduce the CuRobo device boundary")
def test_pose_residual_aligns_cpu_targets_with_cuda_fk_output():
    result = _pose_residual(
        th.tensor([1.0, 2.0, 3.0], device="cuda:0"),
        th.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0"),
        th.tensor([1.0, 2.0, 3.0]),
        th.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    assert result == {"translation_m": 0.0, "orientation_rad": 0.0}


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


def test_candidate_current_base_pose_reproduces_current_virtual_base_joints():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.robot = _Robot()
    backend.robot.q[:6] = th.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    result = backend._candidate_joint_state(_pose([11.0, 22.0, 0.0]))

    th.testing.assert_close(result, backend.robot.get_joint_positions())


def test_ordered_joint_state_updates_embodiment_locks_before_fk_ordering():
    class _FullState:
        def __init__(self, label):
            self.label = label
            self.order_calls = []

        def get_ordered_joint_state(self, names):
            self.order_calls.append(list(names))
            return (self.label, tuple(names))

    class _MotionGenerator:
        def __init__(self):
            self.mg = {
                "arm": type(
                    "_ArmMotionGenerator",
                    (),
                    {"kinematics": type("_Kinematics", (), {"joint_names": ["right_arm_joint1"]})()},
                )()
            }
            self.lock_updates = []

        def update_locked_joints(self, joint_state, emb_sel):
            self.lock_updates.append((joint_state, emb_sel))

    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.motion_generator = _MotionGenerator()
    trajectory_state = _FullState("trajectory")
    locked_state = _FullState("fixture")
    states = iter((trajectory_state, locked_state))
    backend._full_joint_state_for_curobo = lambda _: next(states)

    result = backend._ordered_joint_state_for_curobo(
        th.zeros((2, 3)),
        emb_sel="arm",
        locked_joint_positions=th.zeros(3),
    )

    assert backend.motion_generator.lock_updates == [(locked_state, "arm")]
    assert trajectory_state.order_calls == [["right_arm_joint1"]]
    assert result == ("trajectory", ("right_arm_joint1",))


def test_world_target_is_expressed_in_curobo_base_frame_before_fk_comparison():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    backend.robot = _Robot()
    backend.robot.links = {"base_footprint_x": backend.robot.root_link}
    backend.motion_generator = type(
        "_MotionGenerator",
        (),
        {"base_link": {"arm": "base_footprint_x"}},
    )()
    target = {
        "parent_frame": "simulator_world",
        "child_frame": "right_eef_pregrasp",
        "translation_m": [10.54, 19.79, 0.64],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    position, orientation = backend._target_pose_for_curobo(target, "pregrasp")

    th.testing.assert_close(position, th.tensor([0.54, -0.21, 0.64]), atol=1e-4, rtol=0.0)
    th.testing.assert_close(orientation, th.tensor([0.0, 0.0, 0.0, 1.0]))


def test_solver_inputs_quantize_restore_noise_without_losing_quaternion_normalization():
    assert DIAGNOSTIC_SOLVER_INPUT_QUANTUM == 1e-4
    left = th.tensor([1.234561, -0.500001])
    right = th.tensor([1.234559, -0.499999])

    th.testing.assert_close(_quantize_tensor(left), _quantize_tensor(right))
    position, quaternion = _quantize_pose(
        th.tensor([0.100001, -0.200001, 0.300001]),
        th.tensor([0.100001, 0.200001, 0.300001, 0.900001]),
    )

    th.testing.assert_close(position, th.tensor([0.1, -0.2, 0.3]))
    th.testing.assert_close(th.linalg.vector_norm(quaternion), th.tensor(1.0))


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


def test_g013_provenance_trajectory_changes_only_right_arm_and_gripper_indices():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    start = th.arange(14, dtype=th.float32)
    arm_indices = th.tensor([1, 3, 5, 7, 9, 11, 13])
    gripper_indices = th.tensor([0, 2])
    start[gripper_indices] = th.tensor([0.05, 0.05])
    branch = {
        "open_gripper_joint_positions_m": [0.05, 0.05],
        "closed_gripper_joint_positions_m": [0.021, 0.016],
        "gripper_close_interpolation_steps": 4,
        "stages": {
            "preclose": {
                "right_arm_waypoints_rad": [[0.1] * 7, [0.2] * 7],
                "interpolation_steps": [2, 3],
            }
        },
    }

    goal, trajectory, provenance = backend._provenance_trajectory(
        start,
        branch=branch,
        stage_name="preclose",
        arm_indices=arm_indices,
        gripper_indices=gripper_indices,
    )

    unlocked = set(arm_indices.tolist() + gripper_indices.tolist())
    locked = th.tensor([index for index in range(len(start)) if index not in unlocked])
    assert trajectory.shape == (6, 14)
    th.testing.assert_close(trajectory[:, locked], start[locked].expand(6, -1))
    th.testing.assert_close(goal[arm_indices], th.full((7,), 0.2))
    th.testing.assert_close(goal[gripper_indices], th.tensor([0.05, 0.05]))
    assert provenance["interpolation_steps"] == [2, 3]


def test_g013_lift_trajectory_closes_before_moving_the_arm():
    backend = RadioSemanticGeometryBackend.__new__(RadioSemanticGeometryBackend)
    start = th.zeros(10, dtype=th.float32)
    arm_indices = th.arange(7)
    gripper_indices = th.tensor([8, 9])
    start[gripper_indices] = 0.05
    branch = {
        "open_gripper_joint_positions_m": [0.05, 0.05],
        "closed_gripper_joint_positions_m": [0.021, 0.016],
        "gripper_close_interpolation_steps": 4,
        "stages": {
            "lift_1": {
                "right_arm_waypoints_rad": [[0.2] * 7],
                "interpolation_steps": [3],
            }
        },
    }

    goal, trajectory, provenance = backend._provenance_trajectory(
        start,
        branch=branch,
        stage_name="lift_1",
        arm_indices=arm_indices,
        gripper_indices=gripper_indices,
    )

    assert trajectory.shape == (8, 10)
    th.testing.assert_close(trajectory[:5, arm_indices], th.zeros((5, 7)))
    th.testing.assert_close(trajectory[4:, gripper_indices], goal[gripper_indices].expand(4, -1))
    assert provenance["gripper_close_interpolation_steps"] == 4


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
