from __future__ import annotations

import dataclasses
import sys
import types

import pytest

import snapshot


sys.modules.setdefault("omnigibson", types.ModuleType("omnigibson"))
sys.modules.setdefault("omnigibson.learning", types.ModuleType("omnigibson.learning"))
sys.modules.setdefault("omnigibson.learning.agentic", types.ModuleType("omnigibson.learning.agentic"))
sys.modules["omnigibson.learning.agentic.snapshot"] = snapshot

from evaluator_state import EvaluatorSnapshotComponent  # noqa: E402

SnapshotRestoreError = snapshot.SnapshotRestoreError


@dataclasses.dataclass
class Named:
    name: str


class FakeScene:
    def __init__(self):
        self.objects = [Named("robot"), Named("radio")]
        self.active_systems = {"water": object()}


class FakeCondition:
    def __init__(self):
        self._done = False
        self._goal_status = {"satisfied": [], "unsatisfied": ["radio_on"]}


class FakeReward:
    def __init__(self):
        self._reward = 0.0
        self._info = {}
        self._potential = 1.0


class FakeTask:
    def __init__(self):
        self._reward = 0.0
        self._done = False
        self._success = False
        self._info = {}
        self._termination_conditions = {"predicate": FakeCondition()}
        self._reward_functions = {"potential": FakeReward()}


class FakeEnv:
    def __init__(self):
        self.scene = FakeScene()
        self.task = FakeTask()


class FakeMetric:
    def __init__(self):
        self.timesteps = 4
        self.values = [1.0]


class FakeEvaluator:
    def __init__(self):
        self.env = FakeEnv()
        self.metrics = [FakeMetric()]
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0.0
        self.robot_action = [0.0]
        self.latest_task_progress = {"radio_on": False}
        self._last_logged_task_progress = {"radio_on": False}
        self.latest_inference_metadata = None
        self._latest_logged_inference_metadata = None
        self._inference_log_offset = 0
        self._inference_log_fallback_logged = False
        self._current_rollout_metadata = {"instance": 1}
        self._current_task_progress_summary = {"max": 0.0}
        self.obs = {"rgb": "old"}


def test_evaluator_component_restores_state_and_verifies_progress():
    evaluator = FakeEvaluator()
    component = EvaluatorSnapshotComponent(
        evaluator,
        progress_reader=lambda: {"radio_on": False},
        observation_reader=lambda: {"rgb": "refreshed"},
    )
    state = component.capture()

    evaluator.latest_task_progress["radio_on"] = True
    evaluator.metrics[0].timesteps = 99
    evaluator.env.task._success = True
    evaluator.env.task._termination_conditions["predicate"]._done = True
    evaluator.env.task._reward_functions["potential"]._potential = -1.0
    component.restore(state)
    report = component.after_propagation(state)

    assert evaluator.latest_task_progress == {"radio_on": False}
    assert evaluator.metrics[0].timesteps == 4
    assert evaluator.env.task._success is False
    assert evaluator.env.task._termination_conditions["predicate"]._done is False
    assert evaluator.env.task._reward_functions["potential"]._potential == 1.0
    assert evaluator.obs == {"rgb": "refreshed"}
    assert report == {
        "topology_verified": True,
        "progress_verified": True,
        "observation_refreshed": True,
    }


def test_evaluator_component_rejects_progress_drift():
    evaluator = FakeEvaluator()
    component = EvaluatorSnapshotComponent(evaluator, progress_reader=lambda: {"radio_on": True})
    state = component.capture()

    with pytest.raises(SnapshotRestoreError, match="Task progress drifted"):
        component.after_propagation(state)


def test_evaluator_component_rejects_topology_drift():
    evaluator = FakeEvaluator()
    component = EvaluatorSnapshotComponent(evaluator)
    state = component.capture()
    evaluator.env.scene.objects.append(Named("new_object"))

    with pytest.raises(SnapshotRestoreError, match="topology drifted"):
        component.after_propagation(state)
