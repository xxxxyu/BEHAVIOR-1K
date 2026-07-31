from __future__ import annotations

import numpy as np
import pytest
import sys
import types

import controller_manifest
from controller_manifest import build_controller_manifest
from controller_manifest_test import R1Pro
import snapshot
from snapshot import CompositeSnapshotManager


sys.modules.setdefault("omnigibson", types.ModuleType("omnigibson"))
sys.modules.setdefault("omnigibson.learning", types.ModuleType("omnigibson.learning"))
sys.modules.setdefault("omnigibson.learning.agentic", types.ModuleType("omnigibson.learning.agentic"))
sys.modules["omnigibson.learning.agentic.controller_manifest"] = controller_manifest
sys.modules["omnigibson.learning.agentic.snapshot"] = snapshot

from environment_session import AgenticEnvironmentSession  # noqa: E402
from environment_session import SessionState  # noqa: E402


class FakeSimulator:
    def __init__(self):
        self.state = {"object": [1.0]}

    def dump_state(self, *, serialized):
        return self.state

    def load_state(self, state, *, serialized):
        self.state = state

    def propagate(self):
        return {"backend": "fake"}


class FakeEnv:
    def __init__(self):
        self._current_step = 0
        self._current_episode = 0
        self.actions = []

    def step(self, action, *, n_render_iterations):
        self.actions.append(action.copy())
        self._current_step += 1
        return {"rgb": "frame"}, 0.25, False, False, {"render_iterations": n_render_iterations}

    def get_obs(self):
        return {"rgb": "frame"}, {"source": "fake"}


def make_session():
    env = FakeEnv()
    simulator = FakeSimulator()
    manifest = build_controller_manifest(R1Pro(), physics_frequency_hz=120.0)
    manager = CompositeSnapshotManager(
        simulator=simulator,
        env=env,
        runtime_fingerprint="behavior@test",
        controller_fingerprint=manifest.fingerprint,
        propagate_once=simulator.propagate,
    )
    return AgenticEnvironmentSession(env=env, controller_manifest=manifest, snapshot_manager=manager), simulator


def test_session_validates_and_executes_one_owned_step():
    session, _ = make_session()

    result = session.step(np.zeros(23), n_render_iterations=2)

    assert result.env_step == 1
    assert result.reward == 0.25
    assert result.info == {"render_iterations": 2}
    assert len(session.env.actions) == 1


def test_session_rejects_invalid_action_before_environment_step():
    session, _ = make_session()
    action = np.zeros(23)
    action[0] = 2.0

    with pytest.raises(ValueError, match="outside safety limits"):
        session.step(action)

    assert not session.env.actions


def test_snapshot_and_restore_require_explicit_pause():
    session, simulator = make_session()
    with pytest.raises(RuntimeError, match="must be paused"):
        session.snapshot(task_name="task", instance_id="1", episode_id="e")

    session.pause()
    snapshot = session.snapshot(task_name="task", instance_id="1", episode_id="e")
    simulator.state["object"][0] = 9.0
    report = session.restore(snapshot)

    assert simulator.state == {"object": [1.0]}
    assert report.propagation_sim_steps == 1
    session.resume()
    assert session.state == SessionState.ACTIVE


def test_closed_session_rejects_control_operations():
    session, _ = make_session()
    session.close()

    with pytest.raises(RuntimeError, match="while closed"):
        session.step(np.zeros(23))
    with pytest.raises(RuntimeError, match="closed"):
        session.observe()
