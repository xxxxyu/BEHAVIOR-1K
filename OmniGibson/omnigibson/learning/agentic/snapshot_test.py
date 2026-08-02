from __future__ import annotations

import dataclasses
from pathlib import Path
import random

import pytest

from snapshot import AttributeSnapshotComponent
from snapshot import CompositeSnapshotManager
from snapshot import load_persisted_snapshot
from snapshot import persist_snapshot
from snapshot import SnapshotCompatibilityError
from snapshot import SnapshotRestoreError


class FakeSimulator:
    def __init__(self):
        self.state = {"position": [1.0, 2.0], "grasped": True}
        self.propagations = 0

    def dump_state(self, *, serialized):
        assert serialized
        return self.state

    def load_state(self, state, *, serialized):
        assert serialized
        self.state = state

    def propagate(self):
        self.propagations += 1
        return {"backend": "fake", "propagation_index": self.propagations}


@dataclasses.dataclass
class FakeEnv:
    _current_step: int = 12
    _current_episode: int = 3


@dataclasses.dataclass
class EpisodeState:
    latest_progress: dict[str, bool]
    action_queue: list[list[float]]


def make_manager(simulator, env, episode_state, **kwargs):
    component = AttributeSnapshotComponent(
        name="episode_state",
        owner=episode_state,
        attributes=("latest_progress", "action_queue"),
        after_propagation_callback=lambda _: {"progress_recomputed": True},
    )
    return CompositeSnapshotManager(
        simulator=simulator,
        env=env,
        runtime_fingerprint=kwargs.get("runtime_fingerprint", "behavior@abc"),
        controller_fingerprint="controller@123",
        components=(component,),
        propagate_once=simulator.propagate,
    )


def test_composite_snapshot_round_trip_restores_all_registered_state():
    simulator = FakeSimulator()
    env = FakeEnv()
    episode_state = EpisodeState({"near": True}, [[0.1, 0.2]])
    manager = make_manager(simulator, env, episode_state)
    random.seed(7)
    snapshot = manager.capture(task_name="turning_on_radio", instance_id="42", episode_id="episode-0")
    expected_random = random.random()

    simulator.state["position"][0] = 99.0
    env._current_step = 99
    env._current_episode = 10
    episode_state.latest_progress["near"] = False
    episode_state.action_queue.append([9.0])
    random.seed(99)

    report = manager.restore(snapshot)

    assert simulator.state == {"position": [1.0, 2.0], "grasped": True}
    assert env == FakeEnv(_current_step=12, _current_episode=3)
    assert episode_state == EpisodeState({"near": True}, [[0.1, 0.2]])
    assert random.random() == expected_random
    assert report.snapshot_id == snapshot.snapshot_id
    assert report.propagation_sim_steps == 1
    assert report.propagation_details[0]["environment_step_increment"] == 0
    assert report.component_reports == {"episode_state": {"progress_recomputed": True}}


def test_snapshot_is_immutable_relative_to_live_mutation():
    simulator = FakeSimulator()
    env = FakeEnv()
    episode_state = EpisodeState({"near": True}, [[0.1]])
    manager = make_manager(simulator, env, episode_state)
    snapshot = manager.capture(task_name="task", instance_id="1", episode_id="e")

    simulator.state["position"].append(3.0)
    episode_state.action_queue[0][0] = 4.0

    assert snapshot.simulator_state["position"] == [1.0, 2.0]
    assert snapshot.component_states["episode_state"]["action_queue"] == [[0.1]]


def test_restore_rejects_runtime_mismatch_before_loading_state():
    simulator = FakeSimulator()
    env = FakeEnv()
    episode_state = EpisodeState({}, [])
    snapshot = make_manager(simulator, env, episode_state).capture(task_name="task", instance_id="1", episode_id="e")
    simulator.state = {"sentinel": True}
    incompatible = make_manager(simulator, env, episode_state, runtime_fingerprint="behavior@different")

    with pytest.raises(SnapshotCompatibilityError, match="runtime fingerprint"):
        incompatible.restore(snapshot)

    assert simulator.state == {"sentinel": True}


def test_snapshot_id_changes_with_parent_lineage():
    simulator = FakeSimulator()
    env = FakeEnv()
    episode_state = EpisodeState({}, [])
    manager = make_manager(simulator, env, episode_state)
    first = manager.capture(task_name="task", instance_id="1", episode_id="e")
    second = manager.capture(task_name="task", instance_id="1", episode_id="e", parent_snapshot_id=first.snapshot_id)

    assert first.snapshot_id != second.snapshot_id


def test_failed_restore_rolls_back_simulator_components_and_lifecycle():
    simulator = FakeSimulator()
    env = FakeEnv()
    episode_state = EpisodeState({"near": False}, [[0.1]])

    def reject_false_progress(state):
        if state["latest_progress"]["near"] is False:
            raise SnapshotRestoreError("injected post-propagation failure")
        return {"progress_recomputed": True}

    component = AttributeSnapshotComponent(
        name="episode_state",
        owner=episode_state,
        attributes=("latest_progress", "action_queue"),
        after_propagation_callback=reject_false_progress,
    )
    manager = CompositeSnapshotManager(
        simulator=simulator,
        env=env,
        runtime_fingerprint="behavior@abc",
        controller_fingerprint="controller@123",
        components=(component,),
        propagate_once=simulator.propagate,
    )
    rejected = manager.capture(task_name="task", instance_id="1", episode_id="e")
    simulator.state = {"position": [7.0, 8.0], "grasped": False}
    env._current_step = 42
    env._current_episode = 9
    episode_state.latest_progress = {"near": True}
    episode_state.action_queue = [[9.0]]

    with pytest.raises(SnapshotRestoreError, match="rolled back.*injected post-propagation failure"):
        manager.restore(rejected)

    assert simulator.state == {"position": [7.0, 8.0], "grasped": False}
    assert env == FakeEnv(_current_step=42, _current_episode=9)
    assert episode_state == EpisodeState({"near": True}, [[9.0]])


def test_persisted_snapshot_round_trip_and_hash_validation(tmp_path):
    simulator = FakeSimulator()
    env = FakeEnv()
    episode_state = EpisodeState({"near": True}, [[0.1]])
    snapshot = make_manager(simulator, env, episode_state).capture(task_name="task", instance_id="1", episode_id="e")

    paths = persist_snapshot(snapshot, tmp_path / "snapshots")
    loaded = load_persisted_snapshot(Path(paths["manifest_path"]))

    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.simulator_state == snapshot.simulator_state

    payload_path = Path(paths["payload_path"])
    payload_path.write_bytes(payload_path.read_bytes() + b"tampered")
    with pytest.raises(SnapshotCompatibilityError, match="payload hash"):
        load_persisted_snapshot(Path(paths["manifest_path"]))
