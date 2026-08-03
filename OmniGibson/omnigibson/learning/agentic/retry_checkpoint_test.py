import math

import pytest

from omnigibson.learning.agentic.retry_checkpoint import assess_radio_demo_primary_checkpoint
from omnigibson.learning.agentic.retry_checkpoint import assess_radio_pickup_preclose_checkpoint


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


def _preclose_state(**overrides):
    state = {
        **_state(),
        "table_position": [3.2, 5.6, 0.25],
        "table_orientation": [0.0, 0.0, 0.0, 1.0],
        "table_linear_velocity": [0.0, 0.0, 0.0],
        "table_angular_velocity": [0.0, 0.0, 0.0],
        "base_position": [2.5, 5.2, 0.0],
        "base_orientation": [0.0, 0.0, 0.0, 1.0],
        "base_linear_velocity": [0.0, 0.0, 0.0],
        "base_angular_velocity": [0.0, 0.0, 0.0],
        "contact_body_paths": ["/World/scene_0/coffee_table/base_link"],
        "table_link_paths": ["/World/scene_0/coffee_table/base_link"],
        "assisted_grasp_attachments": {"left": None, "right": None},
        "robot_near_radio": True,
        "radio_picked_up": False,
        "radio_on": False,
        "right_gripper_command": 1.0,
    }
    state.update(overrides)
    return state


def test_accepts_source_pinned_radio_pickup_preclose_state():
    result = assess_radio_pickup_preclose_checkpoint(_preclose_state(), _preclose_state())

    assert result["accepted"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    [
        ({"robot_near_radio": False}, "robot_near_radio"),
        ({"radio_picked_up": True}, "radio_not_picked_up"),
        ({"radio_on": True}, "radio_off"),
        ({"grasping_right": True}, "ungrasped"),
        ({"assisted_grasp_attachments": {"left": None, "right": "radio"}}, "no_assisted_attachment"),
        ({"contact_body_paths": ["/World/scene_0/robot/right_finger"]}, "table_only_contact"),
        ({"linear_velocity": [0.02, 0.0, 0.0]}, "radio_linear_velocity_stationary"),
        ({"table_position": [3.21, 5.6, 0.25]}, "table_position_stable"),
        ({"base_position": [2.51, 5.2, 0.0]}, "base_position_stable"),
        ({"right_gripper_command": 0.5}, "right_gripper_open"),
    ],
)
def test_rejects_invalid_radio_pickup_preclose_state(overrides, failed_check):
    result = assess_radio_pickup_preclose_checkpoint(_preclose_state(), _preclose_state(**overrides))

    assert result["accepted"] is False
    assert result["checks"][failed_check] is False
