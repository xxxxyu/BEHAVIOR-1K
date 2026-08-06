from __future__ import annotations

import copy

import numpy as np
import pytest

from omnigibson.learning.agentic.runtime_server import AgenticEvaluatorRuntime
from omnigibson.learning.agentic.semantic_geometry import RADIO_TASK_BINDING
from omnigibson.learning.agentic.semantic_geometry import SUPPORT_TABLE_TASK_BINDING
from omnigibson.learning.agentic.semantic_geometry import capture_navigation_geometry
from omnigibson.learning.agentic.semantic_geometry import navigation_geometry_deltas
from omnigibson.object_states import ContactBodies
from omnigibson.object_states import OnTop


class _State:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def get_value(self, *_args):
        self.calls += 1
        return self.value


class _Link:
    def __init__(self, prim_path):
        self.prim_path = prim_path


class _Object:
    def __init__(
        self,
        *,
        name,
        category,
        prim_path,
        position,
        quaternion,
        extent,
        contacts=(),
        supported=False,
    ):
        self.name = name
        self.category = category
        self.prim_path = prim_path
        self.position = np.asarray(position, dtype=np.float64)
        self.quaternion = np.asarray(quaternion, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)
        self.links = {"base": _Link(f"{prim_path}/base")}
        self.states = {
            ContactBodies: _State(set(contacts)),
            OnTop: _State(supported),
        }

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def get_base_aligned_bbox(self, *, visual, xy_aligned):
        assert visual is False
        assert xy_aligned is True
        return self.position.copy(), self.quaternion.copy(), self.extent.copy(), np.zeros(3)


class _Binding:
    exists = True

    def __init__(self, value):
        self.unwrapped = value


def _fixture():
    table_link = _Link("/World/table/base")
    robot_link = _Link("/World/robot/base")
    radio = _Object(
        name="portable_radio_exact",
        category="radio_receiver",
        prim_path="/World/radio",
        position=[2.0, 1.0, 0.55],
        quaternion=[0.0, 0.0, 0.0, -1.0],
        extent=[0.2, 0.1, 0.3],
        contacts=[table_link],
        supported=True,
    )
    table = _Object(
        name="coffee_table_exact",
        category="coffee_table",
        prim_path="/World/table",
        position=[2.0, 1.0, 0.35],
        quaternion=[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        extent=[1.2, 0.6, 0.7],
        contacts=[robot_link],
    )
    robot = _Object(
        name="robot",
        category="agent",
        prim_path="/World/robot",
        position=[0.0, 0.0, 0.0],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        extent=[0.7, 0.5, 1.5],
        contacts=[table_link],
    )
    robot.reset_joint_pos_aabb_extent = np.asarray([0.7, 0.5, 1.5])
    robot.links = {"base": robot_link}
    table.links = {"base": table_link}
    decoy = _Object(
        name="portable_radio_decoy",
        category="radio_receiver",
        prim_path="/World/decoy",
        position=[9.0, 9.0, 9.0],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        extent=[1.0, 1.0, 1.0],
    )
    task = type(
        "Task",
        (),
        {"object_scope": {RADIO_TASK_BINDING: _Binding(radio), SUPPORT_TABLE_TASK_BINDING: _Binding(table)}},
    )()
    env = type("Environment", (), {"task": task, "_current_step": 256})()
    env.decoy_with_radio_name = decoy
    return env, robot, radio, table


def test_semantic_identity_uses_exact_task_bindings_and_collision_footprints():
    env, robot, _radio, _table = _fixture()

    result = capture_navigation_geometry(env, robot)

    assert result["radio"]["identity"] == {
        "resolution": "exact_task_object_scope_binding",
        "task_binding": RADIO_TASK_BINDING,
        "name": "portable_radio_exact",
        "category": "radio_receiver",
        "prim_path": "/World/radio",
    }
    assert result["support_table"]["identity"]["task_binding"] == SUPPORT_TABLE_TASK_BINDING
    assert result["support_table"]["collision_box"]["source_api"].startswith("get_base_aligned_bbox")
    assert result["radio"]["supported_by_task_table"] is True
    polygon = np.asarray(result["support_table"]["collision_box"]["footprint_polygon_world_xy_m"])
    np.testing.assert_allclose(np.ptp(polygon, axis=0), [0.6, 1.2], atol=1e-12)
    assert result["robot"]["footprint"]["extent_xy_m"] == [0.7, 0.5]
    assert {tuple((pair["body_a"], pair["body_b"])) for pair in result["contacts"]["pairs"]} >= {
        ("radio", "support_table"),
        ("robot", "support_table"),
    }


def test_geometry_deltas_are_quaternion_sign_invariant_and_frame_labelled():
    env, robot, radio, _table = _fixture()
    baseline = capture_navigation_geometry(env, robot)
    radio.quaternion *= -1
    current = capture_navigation_geometry(env, robot)

    deltas = navigation_geometry_deltas(baseline, current)

    assert baseline["frames"]["pose_convention"] == "T_parent_child"
    assert baseline["radio"]["pose"]["parent_frame"] == "simulator_world"
    assert baseline["radio"]["pose"]["child_frame"] == "radio_base"
    assert deltas["radio"]["orientation_angle_rad"] == pytest.approx(0.0, abs=1e-12)


def test_runtime_query_is_repeatable_and_has_no_pose_contact_step_or_baseline_side_effects():
    env, robot, radio, table = _fixture()
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.task_name = "turning_on_radio"
    runtime.enable_semantic_geometry_diagnostic = True
    runtime.robot = robot
    runtime.evaluator = type("Evaluator", (), {"env": env, "obs": {"public": np.asarray([1.0])}})()
    runtime._observation_id = lambda _observation: "obs-256-test"
    runtime._navigation_geometry_baseline = capture_navigation_geometry(env, robot)
    runtime._navigation_geometry_baseline_snapshot_id = "sha256:fixture"
    runtime.last_reward = 0.0
    runtime.last_info = {}
    runtime.terminated = False
    runtime.truncated = False
    runtime.finished = False
    runtime._metrics_finished = False
    runtime._last_metrics = None
    before = {
        "step": env._current_step,
        "radio_pose": (radio.position.copy(), radio.quaternion.copy()),
        "table_pose": (table.position.copy(), table.quaternion.copy()),
        "robot_pose": (robot.position.copy(), robot.quaternion.copy()),
        "baseline": copy.deepcopy(runtime._navigation_geometry_baseline),
    }

    first = runtime.inspect_navigation_geometry()
    second = runtime.inspect_navigation_geometry()

    assert first == second
    assert first["observation_id"] == "obs-256-test"
    assert first["env_step"] == 256
    assert first["read_only"] is True
    assert env._current_step == before["step"]
    assert runtime._navigation_geometry_baseline == before["baseline"]
    for obj, key in ((radio, "radio_pose"), (table, "table_pose"), (robot, "robot_pose")):
        np.testing.assert_array_equal(obj.position, before[key][0])
        np.testing.assert_array_equal(obj.quaternion, before[key][1])


def test_runtime_query_requires_explicit_restore_baseline():
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.task_name = "turning_on_radio"
    runtime.enable_semantic_geometry_diagnostic = True
    runtime._navigation_geometry_baseline = None
    runtime._navigation_geometry_baseline_snapshot_id = None

    with pytest.raises(RuntimeError, match="explicit immutable fixture restore"):
        runtime.inspect_navigation_geometry()


def test_runtime_query_is_permission_gated_outside_generation_015():
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.task_name = "turning_on_radio"
    runtime.enable_semantic_geometry_diagnostic = False

    with pytest.raises(PermissionError, match="disabled outside"):
        runtime.inspect_navigation_geometry()
