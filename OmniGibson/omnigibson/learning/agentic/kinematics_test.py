import numpy as np
import pytest

from omnigibson.learning.agentic.kinematics import bounded_pose_delta_ik_step
from omnigibson.learning.agentic.kinematics import bounded_position_ik_step


def test_bounded_position_ik_step_clips_cartesian_and_joint_updates():
    jacobian = np.zeros((6, 7), dtype=np.float32)
    jacobian[:3, :3] = np.eye(3, dtype=np.float32)

    result = bounded_position_ik_step(
        joint_positions=np.zeros(7),
        jacobian=jacobian,
        eef_position=np.zeros(3),
        target_position=np.asarray([1.0, 0.0, 0.0]),
        joint_lower=np.full(7, -1.0),
        joint_upper=np.full(7, 1.0),
        max_target_delta_m=0.1,
        max_joint_delta_rad=0.05,
    )

    assert result["target_delta_clipped"] is True
    assert result["joint_delta_clipped"] is True
    np.testing.assert_allclose(result["used_target_position"], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(result["joint_target"][:3], [0.05, 0.0, 0.0], atol=1e-5)


def test_bounded_position_ik_step_keeps_solution_inside_joint_limits():
    jacobian = np.zeros((6, 7), dtype=np.float32)
    jacobian[:3, :3] = np.eye(3, dtype=np.float32)

    result = bounded_position_ik_step(
        joint_positions=np.asarray([0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        jacobian=jacobian,
        eef_position=np.zeros(3),
        target_position=np.asarray([0.05, 0.0, 0.0]),
        joint_lower=np.full(7, -0.1),
        joint_upper=np.full(7, 0.1),
        max_target_delta_m=0.1,
        max_joint_delta_rad=0.1,
    )

    assert result["joint_target"][0] == np.float32(0.1)


def _pose_step(**overrides):
    jacobian = np.zeros((6, 7), dtype=np.float32)
    jacobian[:, :6] = np.eye(6, dtype=np.float32)
    values = {
        "joint_positions": np.zeros(7, dtype=np.float32),
        "jacobian": jacobian,
        "eef_position": np.zeros(3, dtype=np.float32),
        "eef_quaternion_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "target_position": np.zeros(3, dtype=np.float32),
        "orientation_delta_eef_axis_angle_rad": np.zeros(3, dtype=np.float32),
        "joint_lower": np.full(7, -2.0, dtype=np.float32),
        "joint_upper": np.full(7, 2.0, dtype=np.float32),
        "max_target_delta_m": 0.03,
        "max_orientation_delta_rad": 0.17,
        "max_joint_delta_rad": 0.12,
    }
    values.update(overrides)
    return bounded_pose_delta_ik_step(**values)


def test_bounded_pose_delta_ik_step_zero_delta_is_identity():
    result = _pose_step(eef_quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, -2.0]))

    np.testing.assert_array_equal(result["joint_target"], np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(result["used_translation_delta_robot_m"], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(result["used_orientation_delta_robot_axis_angle_rad"], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(
        result["current_eef_quaternion_robot_xyzw"], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert not any(
        result[key]
        for key in (
            "translation_delta_clipped",
            "orientation_delta_clipped",
            "joint_delta_clipped",
            "joint_limit_clipped",
        )
    )


def test_bounded_pose_delta_ik_step_converts_eef_delta_to_robot_frame():
    half_sqrt = np.sqrt(0.5)
    result = _pose_step(
        eef_quaternion_xyzw=np.asarray([0.0, 0.0, half_sqrt, half_sqrt]),
        orientation_delta_eef_axis_angle_rad=np.asarray([0.1, 0.0, 0.0]),
    )

    np.testing.assert_allclose(result["used_orientation_delta_robot_axis_angle_rad"], [0.0, 0.1, 0.0], atol=1e-7)
    assert result["joint_target"][4] == pytest.approx(0.1 / 1.0001, abs=1e-6)


def test_bounded_pose_delta_ik_step_uses_shortest_arc_and_is_quaternion_sign_invariant():
    quaternion = np.asarray([0.2, -0.3, 0.1, 0.9], dtype=np.float64)
    orientation_delta = np.asarray([0.04, -0.03, 0.02])

    positive = _pose_step(
        eef_quaternion_xyzw=quaternion,
        orientation_delta_eef_axis_angle_rad=orientation_delta,
    )
    negative = _pose_step(
        eef_quaternion_xyzw=-quaternion,
        orientation_delta_eef_axis_angle_rad=orientation_delta,
    )

    for key in (
        "current_eef_quaternion_robot_xyzw",
        "target_eef_quaternion_robot_xyzw",
        "used_orientation_delta_robot_axis_angle_rad",
        "joint_target",
    ):
        np.testing.assert_allclose(positive[key], negative[key], atol=1e-7)
    assert positive["target_eef_quaternion_robot_xyzw"][3] >= 0
    assert positive["used_orientation_magnitude_rad"] == pytest.approx(np.linalg.norm(orientation_delta))


def test_bounded_pose_delta_ik_step_clips_translation_orientation_and_joints_independently():
    result = _pose_step(
        target_position=np.asarray([0.2, 0.0, 0.0]),
        orientation_delta_eef_axis_angle_rad=np.asarray([0.0, 0.4, 0.0]),
        max_target_delta_m=0.03,
        max_orientation_delta_rad=0.17,
        max_joint_delta_rad=0.02,
    )

    np.testing.assert_allclose(result["used_translation_delta_robot_m"], [0.03, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(result["used_orientation_delta_eef_axis_angle_rad"], [0.0, 0.17, 0.0], atol=1e-7)
    assert result["translation_delta_clipped"]
    assert result["orientation_delta_clipped"]
    assert result["joint_delta_clipped"]
    assert np.max(np.abs(result["joint_delta"])) == pytest.approx(0.02, abs=1e-7)


def test_bounded_pose_delta_ik_step_clips_physical_joint_limits_after_delta_bound():
    upper = np.full(7, 2.0, dtype=np.float32)
    upper[0] = 0.01
    result = _pose_step(
        target_position=np.asarray([0.02, 0.0, 0.0]),
        joint_upper=upper,
        max_joint_delta_rad=0.12,
    )

    assert result["joint_target"][0] == pytest.approx(0.01, abs=1e-7)
    assert not result["joint_delta_clipped"]
    assert result["joint_limit_clipped"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"joint_positions": np.zeros((1, 7))}, "malformed shapes"),
        ({"jacobian": np.zeros((7, 6))}, r"shape \(6, 7\)"),
        ({"eef_quaternion_xyzw": np.zeros(4)}, "nonzero magnitude"),
        ({"joint_positions": np.full(7, 3.0)}, "inside physical joint limits"),
        ({"orientation_delta_eef_axis_angle_rad": np.asarray([0.0, np.nan, 0.0])}, "finite"),
        ({"max_orientation_delta_rad": np.inf}, "finite and positive"),
    ],
)
def test_bounded_pose_delta_ik_step_rejects_malformed_and_nonfinite_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        _pose_step(**overrides)
