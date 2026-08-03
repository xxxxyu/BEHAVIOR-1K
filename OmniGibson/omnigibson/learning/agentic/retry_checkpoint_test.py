import math

import pytest

from omnigibson.learning.agentic.retry_checkpoint import assess_radio_demo_primary_checkpoint


def _state(**overrides):
    state = {
        "position": [3.53, 5.52, 0.53],
        "orientation": [0.0, 0.0, -0.47, 0.88],
        "linear_velocity": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "on_table": True,
        "grasping_left": False,
        "grasping_right": False,
    }
    state.update(overrides)
    return state


def test_accepts_stationary_upright_radio_near_initial_pose():
    result = assess_radio_demo_primary_checkpoint(
        _state(),
        _state(position=[3.54, 5.52, 0.53], orientation=[0.0, 0.0, -0.46, 0.89]),
    )

    assert result["accepted"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    [
        ({"position": [3.23, 5.95, 0.51]}, "position_near_initial"),
        ({"orientation": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]}, "orientation_near_initial"),
        ({"on_table": False}, "on_table"),
        ({"grasping_right": True}, "ungrasped"),
        ({"linear_velocity": [0.04, 0.0, 0.0]}, "linear_velocity_stationary"),
        ({"angular_velocity": [0.0, 0.0, 0.20]}, "angular_velocity_stationary"),
    ],
)
def test_rejects_contaminated_or_moving_radio(overrides, failed_check):
    result = assess_radio_demo_primary_checkpoint(_state(), _state(**overrides))

    assert result["accepted"] is False
    assert result["checks"][failed_check] is False
