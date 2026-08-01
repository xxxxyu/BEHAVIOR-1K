import numpy as np

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
