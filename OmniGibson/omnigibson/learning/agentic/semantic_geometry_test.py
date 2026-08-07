from __future__ import annotations

import copy

import numpy as np
import pytest
import torch as th

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
        self.joint_positions = th.zeros(8, dtype=th.float32)
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

    def get_joint_positions(self):
        return self.joint_positions.clone()


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


def test_semantic_geometry_capability_override_is_reported_without_changing_default_contract():
    env, robot, _radio, _table = _fixture()
    available = {
        name: {"status": "available", "source_api": f"test:{name}"}
        for name in (
            "hypothetical_base_pose_whole_arm_ik",
            "arm_trajectory_collision",
            "arm_table_clearance",
        )
    }

    default = capture_navigation_geometry(env, robot)
    overridden = capture_navigation_geometry(env, robot, backend_capabilities=available)

    assert default["capabilities"]["hypothetical_base_pose_whole_arm_ik"]["status"] == "unavailable"
    assert all(overridden["capabilities"][name] == value for name, value in available.items())
    assert "are available in diagnostic mode" in overridden["navigation_model"]["limitations"][-1]


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


class _FakeSemanticBackend:
    def __init__(self, robot, *, mutate_joint_state=False):
        self.robot = robot
        self.mutate_joint_state = mutate_joint_state
        self.requests = []

    def capabilities(self):
        return {
            name: {"status": "available"}
            for name in (
                "hypothetical_base_pose_whole_arm_ik",
                "arm_trajectory_collision",
                "arm_table_clearance",
            )
        }

    def evaluate_corridor(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.mutate_joint_state:
            self.robot.joint_positions[0] += 1.0
        return {"schema_version": 1, "status": "available", "stages": {}}


def _runtime_with_semantic_backend(*, mutate_joint_state=False):
    env, robot, _radio, _table = _fixture()
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.task_name = "turning_on_radio"
    runtime.enable_semantic_geometry_diagnostic = True
    runtime.robot = robot
    runtime.evaluator = type("Evaluator", (), {"env": env, "obs": {"public": np.asarray([1.0])}})()
    runtime._observation_id = lambda _observation: "obs-256-test"
    runtime._semantic_geometry_backend = _FakeSemanticBackend(
        robot,
        mutate_joint_state=mutate_joint_state,
    )
    runtime._semantic_geometry_backend_error = None
    return runtime, env, robot


def test_semantic_corridor_runtime_rejects_stale_observation_before_backend_call():
    runtime, _env, _robot = _runtime_with_semantic_backend()

    with pytest.raises(RuntimeError, match="not bound to the current observation"):
        runtime.evaluate_semantic_pickup_corridor(
            observation_id="stale-observation",
            env_step=255,
            candidate_base_pose={"candidate": True},
            target_poses={"targets": True},
        )

    assert runtime._semantic_geometry_backend.requests == []


def test_semantic_corridor_runtime_is_observation_bound_and_preserves_simulator_state():
    runtime, env, robot = _runtime_with_semantic_backend()
    before_q = robot.get_joint_positions()
    candidate = {"candidate": True}
    targets = {"targets": True}
    provenance = {"branches": [{"branch_id": "g013-nominal-01"}]}
    retained_lift_gate = {"stage_names": ["lift_1", "lift_2"]}

    result = runtime.evaluate_semantic_pickup_corridor(
        observation_id="obs-256-test",
        env_step=256,
        candidate_base_pose=candidate,
        target_poses=targets,
        joint_provenance=provenance,
        retained_lift_gate=retained_lift_gate,
    )

    assert result["observation_id"] == "obs-256-test"
    assert result["env_step"] == 256
    assert result["read_only"] is True
    assert result["privileged"] is True
    assert runtime._semantic_geometry_backend.requests == [
        {
            "candidate_base_pose": candidate,
            "target_poses": targets,
            "joint_provenance": provenance,
            "retained_lift_gate": retained_lift_gate,
        }
    ]
    assert env._current_step == 256
    assert th.equal(robot.get_joint_positions(), before_q)


def test_semantic_corridor_runtime_detects_backend_joint_state_mutation():
    runtime, _env, _robot = _runtime_with_semantic_backend(mutate_joint_state=True)

    with pytest.raises(RuntimeError, match="changed simulator joint positions"):
        runtime.evaluate_semantic_pickup_corridor(
            observation_id="obs-256-test",
            env_step=256,
            candidate_base_pose={"candidate": True},
            target_poses={"targets": True},
        )
