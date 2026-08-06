from __future__ import annotations

import asyncio
import copy
from http.client import HTTPConnection
import socket
import threading
import time

import numpy as np
import pytest
import websockets
from websockets.sync.client import connect

from omnigibson.learning.agentic.runtime_server import AgenticEnvironmentWebsocketServer
from omnigibson.learning.agentic.runtime_server import AgenticEvaluatorRuntime
from omnigibson.learning.agentic.runtime_server import DuplicateCompletionError
from omnigibson.learning.agentic.runtime_server import JAW_CORRIDOR_CALIBRATION
from omnigibson.learning.agentic.runtime_server import NetworkStartupError
from omnigibson.learning.agentic.runtime_server import NetworkThreadError
from omnigibson.learning.agentic.runtime_server import RequestBridge
from omnigibson.learning.agentic.runtime_server import RequestBridgeShutdownError
from omnigibson.learning.agentic.runtime_server import RequestQueueFullError
from omnigibson.learning.utils.network_utils import Packer
from omnigibson.learning.utils.network_utils import unpackb


def test_jaw_corridor_calibration_declares_only_public_synchronized_sources():
    assert JAW_CORRIDOR_CALIBRATION == {
        "schema_version": 1,
        "camera": "right_wrist",
        "depth_unit": "meter",
        "camera_model": "pinhole_linear_depth",
        "gripper_frame": "right_eef",
        "depth_uncertainty_floor_m": 0.001,
        "source_fields": [
            "right_wrist_depth_linear",
            "right_wrist_camera_pose_robot",
            "right_wrist_camera_intrinsics",
            "right_eef_pose_robot",
        ],
    }
    serialized = repr(JAW_CORRIDOR_CALIBRATION).lower()
    assert not any(
        forbidden in serialized
        for forbidden in ("object_pose", "semantic", "contact", "grasp", "evaluator", "demonstration", "trace")
    )


class _FakeController:
    dof_idx = np.arange(7)
    control_type = "position"
    _control_limits = {"position": (np.full(7, -2.0), np.full(7, 2.0))}


class _FakeRobot:
    def __init__(self) -> None:
        self.controllers = {"arm_right": _FakeController()}
        jacobian = np.zeros((6, 7), dtype=np.float32)
        jacobian[:, :6] = np.eye(6, dtype=np.float32)
        self.control_dict = {
            "joint_position": np.zeros(7, dtype=np.float32),
            "eef_right_jacobian_relative": jacobian,
            "eef_right_pos_relative": np.asarray([0.1, -0.2, 0.5], dtype=np.float32),
            "eef_right_quat_relative": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        self.actions: list[np.ndarray] = []

    def get_control_dict(self):
        return self.control_dict


def test_plan_eef_pose_delta_runtime_is_read_only_and_public_only():
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.robot = _FakeRobot()
    runtime.evaluator = type("Evaluator", (), {"env": type("Environment", (), {"_current_step": 41})()})()
    before_control = copy.deepcopy(runtime.robot.control_dict)
    before_step = runtime.evaluator.env._current_step

    result = runtime.plan_eef_pose_delta(
        arm="right",
        target_position=[0.1, -0.2, 0.5],
        orientation_delta_eef_axis_angle_rad=[0.0, 0.0, 0.08],
    )

    assert runtime.evaluator.env._current_step == before_step
    assert runtime.robot.actions == []
    for key, value in before_control.items():
        np.testing.assert_array_equal(runtime.robot.control_dict[key], value)
    assert result["collision_checked"] is False
    assert result["frame_policy"]["orientation_policy"] == "bounded_eef_frame_pose_delta"
    assert set(result) == {
        "arm",
        "controller_segment",
        "joint_target",
        "joint_delta",
        "current_eef_position_robot_m",
        "current_eef_quaternion_robot_xyzw",
        "requested_translation_delta_robot_m",
        "used_translation_delta_robot_m",
        "requested_orientation_delta_eef_axis_angle_rad",
        "used_orientation_delta_eef_axis_angle_rad",
        "used_orientation_delta_robot_axis_angle_rad",
        "requested_translation_magnitude_m",
        "used_translation_magnitude_m",
        "requested_orientation_magnitude_rad",
        "used_orientation_magnitude_rad",
        "translation_delta_clipped",
        "orientation_delta_clipped",
        "joint_delta_clipped",
        "joint_limit_clipped",
        "frame_policy",
        "collision_checked",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_target_delta_m": 0.051}, r"\(0, 0.05\]"),
        ({"max_orientation_delta_rad": 0.351}, r"\(0, 0.35\]"),
        ({"max_joint_delta_rad": 0.351}, r"\(0, 0.35\]"),
    ],
)
def test_plan_eef_pose_delta_runtime_enforces_hard_caps(overrides, message):
    runtime = AgenticEvaluatorRuntime.__new__(AgenticEvaluatorRuntime)
    runtime.robot = _FakeRobot()
    arguments = {
        "arm": "right",
        "target_position": [0.1, -0.2, 0.5],
        "orientation_delta_eef_axis_angle_rad": [0.0, 0.0, 0.08],
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        runtime.plan_eef_pose_delta(**arguments)

    assert runtime.robot.actions == []


def test_plan_eef_pose_delta_websocket_dispatch_calls_planner_without_action():
    class FakeRuntime:
        def __init__(self) -> None:
            self.requests = []

        def plan_eef_pose_delta(self, **request):
            self.requests.append(request)
            return {"joint_target": [0.0] * 7, "collision_checked": False}

    runtime = FakeRuntime()
    server = AgenticEnvironmentWebsocketServer.__new__(AgenticEnvironmentWebsocketServer)
    server.runtime = runtime

    result = server._dispatch(
        {
            "operation": "plan_eef_pose_delta",
            "arm": "right",
            "target_position": [0.0, 0.0, 0.0],
            "orientation_delta_eef_axis_angle_rad": [0.08, 0.0, 0.0],
        }
    )

    assert result == {"joint_target": [0.0] * 7, "collision_checked": False}
    assert runtime.requests == [
        {
            "arm": "right",
            "target_position": [0.0, 0.0, 0.0],
            "orientation_delta_eef_axis_angle_rad": [0.08, 0.0, 0.0],
            "max_target_delta_m": 0.03,
            "max_orientation_delta_rad": 0.17,
            "max_joint_delta_rad": 0.12,
        }
    ]


def test_semantic_navigation_geometry_websocket_dispatch_is_explicit_and_read_only():
    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def inspect_navigation_geometry(self):
            self.calls += 1
            return {"observation_id": "obs-256", "env_step": 256, "read_only": True}

    runtime = FakeRuntime()
    server = AgenticEnvironmentWebsocketServer.__new__(AgenticEnvironmentWebsocketServer)
    server.runtime = runtime

    result = server._dispatch({"operation": "inspect_navigation_geometry"})

    assert result == {"observation_id": "obs-256", "env_step": 256, "read_only": True}
    assert runtime.calls == 1


class _FakeBridgeRuntime:
    def __init__(self) -> None:
        self.metadata_threads: list[int] = []
        self.dispatch_threads: list[int] = []
        self.steps: list[object] = []

    @property
    def metadata(self):
        self.metadata_threads.append(threading.get_ident())
        return {
            "service": "test_agentic_environment",
            "protocol_version": 2,
            "completed_steps": len(self.steps),
        }

    def observe(self):
        self.dispatch_threads.append(threading.get_ident())
        return {"sequence": len(self.dispatch_threads)}

    def step(self, action):
        self.dispatch_threads.append(threading.get_ident())
        self.steps.append(action)
        return {"action": action}


def _serve_with_client(server, client):
    ready = threading.Event()
    client_error = []

    def run_client():
        try:
            assert ready.wait(5.0)
            client(server)
        except BaseException as exc:
            client_error.append(exc)
            server.shutdown()

    client_thread = threading.Thread(target=run_client, name="runtime-server-test-client")
    client_thread.start()
    server.serve_forever(on_ready=ready.set)
    client_thread.join(5.0)
    assert not client_thread.is_alive()
    if client_error:
        raise client_error[0]


def test_websocket_network_is_off_main_thread_and_protocol_behavior_is_preserved():
    runtime = _FakeBridgeRuntime()
    server = AgenticEnvironmentWebsocketServer(runtime, host="127.0.0.1", port=0)
    packer = Packer()
    main_thread_ident = threading.get_ident()
    results = {}

    def client(active_server):
        port = active_server.bound_port
        assert port is not None
        connection = HTTPConnection("127.0.0.1", port, timeout=2.0)
        connection.request("GET", "/healthz")
        health = connection.getresponse()
        results["health"] = (health.status, health.read())
        connection.close()

        uri = f"ws://127.0.0.1:{port}"
        with connect(uri, compression=None, max_size=None) as controller:
            results["metadata"] = unpackb(controller.recv())

            with connect(uri, compression=None, max_size=None) as duplicate:
                with pytest.raises(websockets.ConnectionClosed) as closed:
                    duplicate.recv()
                results["duplicate_close_code"] = closed.value.rcvd.code

            controller.send(packer.pack({"operation": "step", "action": 1}))
            results["step_1"] = unpackb(controller.recv())
            controller.send(packer.pack({"operation": "unsupported"}))
            results["typed_error"] = unpackb(controller.recv())
            controller.send(packer.pack({"operation": "step", "action": 2}))
            results["step_2"] = unpackb(controller.recv())

        with connect(uri, compression=None, max_size=None) as replacement:
            results["replacement_metadata"] = unpackb(replacement.recv())
            replacement.send(packer.pack({"operation": "observe"}))
            results["observe"] = unpackb(replacement.recv())

        active_server.shutdown()

    _serve_with_client(server, client)

    assert results["health"] == (200, b"OK\n")
    assert results["metadata"] == {
        "service": "test_agentic_environment",
        "protocol_version": 2,
        "completed_steps": 0,
    }
    assert results["replacement_metadata"] == {
        "service": "test_agentic_environment",
        "protocol_version": 2,
        "completed_steps": 2,
    }
    assert results["duplicate_close_code"] == 1013
    assert results["step_1"] == {"ok": True, "result": {"action": 1}}
    assert results["typed_error"] == {
        "ok": False,
        "error": {
            "type": "ValueError",
            "message": "Unsupported agentic environment operation: 'unsupported'.",
        },
    }
    assert results["step_2"] == {"ok": True, "result": {"action": 2}}
    assert results["observe"] == {"ok": True, "result": {"sequence": 3}}
    assert runtime.steps == [1, 2]
    assert runtime.metadata_threads == [main_thread_ident] * 5
    assert runtime.dispatch_threads == [main_thread_ident, main_thread_ident, main_thread_ident]
    assert server.network_thread_ident != main_thread_ident

    port = server.bound_port
    assert port is not None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.2)


def test_request_bridge_queue_full_duplicate_completion_and_shutdown_fail_closed():
    packer = Packer()
    bridge = RequestBridge(capacity=1)
    first = bridge.submit(packer.pack({"operation": "step", "action": 1}))

    with pytest.raises(RequestQueueFullError, match="queue is full"):
        bridge.submit(packer.pack({"operation": "step", "action": 2}))

    assert bridge.closed
    assert isinstance(first.response.exception(), RequestQueueFullError)
    assert bridge.get(0.0) is None

    duplicate_bridge = RequestBridge(capacity=1)
    envelope = duplicate_bridge.submit(packer.pack({"operation": "observe"}))
    assert duplicate_bridge.get(0.0) is envelope
    duplicate_bridge.complete(envelope, b"first")
    with pytest.raises(DuplicateCompletionError, match="more than once"):
        duplicate_bridge.complete(envelope, b"second")
    assert duplicate_bridge.closed
    assert isinstance(duplicate_bridge.close_reason, DuplicateCompletionError)

    shutdown_bridge = RequestBridge(capacity=1)
    pending = shutdown_bridge.submit(packer.pack({"operation": "step", "action": 3}))
    shutdown_bridge.close(RequestBridgeShutdownError("test shutdown"))
    assert isinstance(pending.response.exception(), RequestBridgeShutdownError)
    assert shutdown_bridge.get(0.0) is None


def test_request_timeout_fails_closed_without_replaying_mutation():
    class SlowRuntime(_FakeBridgeRuntime):
        def step(self, action):
            self.dispatch_threads.append(threading.get_ident())
            self.steps.append(action)
            time.sleep(0.1)
            return {"action": action}

    runtime = SlowRuntime()
    server = AgenticEnvironmentWebsocketServer(
        runtime,
        host="127.0.0.1",
        port=0,
        request_timeout=0.02,
    )
    ready = threading.Event()
    response = []
    client_error = []

    def client():
        try:
            assert ready.wait(5.0)
            with connect(f"ws://127.0.0.1:{server.bound_port}", compression=None, max_size=None) as websocket:
                unpackb(websocket.recv())
                websocket.send(Packer().pack({"operation": "step", "action": "mutate-once"}))
                response.append(unpackb(websocket.recv()))
        except BaseException as exc:
            client_error.append(exc)

    client_thread = threading.Thread(target=client, name="runtime-timeout-test-client")
    client_thread.start()
    with pytest.raises(NetworkThreadError, match="failed closed"):
        server.serve_forever(on_ready=ready.set)
    client_thread.join(5.0)

    assert not client_thread.is_alive()
    assert client_error == []
    assert response == [
        {
            "ok": False,
            "error": {
                "type": "RequestTimeoutError",
                "message": "Runtime request did not complete within 0.02 seconds.",
            },
        }
    ]
    assert runtime.steps == ["mutate-once"]


def test_network_thread_startup_and_post_start_failure_are_fail_closed(monkeypatch):
    startup_server = AgenticEnvironmentWebsocketServer(_FakeBridgeRuntime(), host="127.0.0.1", port=0)

    async def fail_startup():
        raise OSError("injected bind failure")

    monkeypatch.setattr(startup_server._network, "_run", fail_startup)
    with pytest.raises(NetworkStartupError, match="failed during startup"):
        startup_server.serve_forever()
    assert startup_server._bridge.closed
    assert not startup_server._network.is_alive

    death_server = AgenticEnvironmentWebsocketServer(_FakeBridgeRuntime(), host="127.0.0.1", port=0)

    async def die_after_startup():
        death_server._network._loop = asyncio.get_running_loop()
        death_server._network._shutdown_event = asyncio.Event()
        death_server._network.bound_port = 43210
        death_server._network._ready.set()
        await asyncio.sleep(0.02)
        raise RuntimeError("injected network thread death")

    monkeypatch.setattr(death_server._network, "_run", die_after_startup)
    with pytest.raises(NetworkThreadError, match="failed closed"):
        death_server.serve_forever()
    assert death_server._bridge.closed
    assert not death_server._network.is_alive


def test_keyboard_interrupt_closes_network_thread_and_listener():
    server = AgenticEnvironmentWebsocketServer(_FakeBridgeRuntime(), host="127.0.0.1", port=0)

    def interrupt_after_bind():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        server.serve_forever(on_ready=interrupt_after_bind)

    assert not server._network.is_alive
    port = server.bound_port
    assert port is not None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.2)
