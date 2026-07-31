"""Snapshot adapter for Python state held outside the OmniGibson simulator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses

from omnigibson.learning.agentic.snapshot import SnapshotError
from omnigibson.learning.agentic.snapshot import SnapshotRestoreError
from omnigibson.learning.agentic.snapshot import _clone_value


_EVALUATOR_FIELDS = (
    "n_trials",
    "n_success_trials",
    "total_time",
    "robot_action",
    "latest_task_progress",
    "_last_logged_task_progress",
    "latest_inference_metadata",
    "_latest_logged_inference_metadata",
    "_inference_log_offset",
    "_inference_log_fallback_logged",
    "_current_rollout_metadata",
    "_current_task_progress_summary",
    "obs",
)
_TASK_FIELDS = (
    "_reward",
    "_done",
    "_success",
    "_info",
    "currently_viewed_index",
    "currently_viewed_instruction",
)
_TERMINATION_FIELDS = ("_done", "_goal_status", "_n_collisions")
_REWARD_FIELDS = (
    "_reward",
    "_info",
    "_potential",
    "prev_grasping",
    "prev_eef_pos",
    "prev_eef_rot",
    "prev_eef_quat",
)


def _capture_present_attributes(owner: object, names: tuple[str, ...]) -> dict[str, object]:
    return {name: _clone_value(getattr(owner, name)) for name in names if hasattr(owner, name)}


def _restore_attributes(owner: object, values: Mapping[str, object]) -> None:
    for name, value in values.items():
        setattr(owner, name, _clone_value(value))


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _scene_topology(env: object) -> dict[str, tuple[str, ...]]:
    scene = env.scene
    return {
        "objects": tuple(sorted(str(obj.name) for obj in scene.objects)),
        "systems": tuple(sorted(str(name) for name in scene.active_systems)),
    }


@dataclasses.dataclass
class EvaluatorSnapshotComponent:
    """Captures the current challenge Evaluator state not owned by ``og.sim``."""

    evaluator: object
    progress_reader: Callable[[], Mapping[str, object] | None] | None = None
    observation_reader: Callable[[], object] | None = None
    name: str = "behavior_evaluator"

    def capture(self) -> object:
        missing = [name for name in _EVALUATOR_FIELDS if not hasattr(self.evaluator, name)]
        if missing:
            raise SnapshotError(f"Evaluator snapshot is missing required fields: {missing}.")
        env = self.evaluator.env
        task = env.task
        metrics = tuple(self.evaluator.metrics)
        return {
            "evaluator": _capture_present_attributes(self.evaluator, _EVALUATOR_FIELDS),
            "task": {
                "class": _qualified_class_name(task),
                "state": _capture_present_attributes(task, _TASK_FIELDS),
            },
            "termination_conditions": {
                name: {
                    "class": _qualified_class_name(condition),
                    "state": _capture_present_attributes(condition, _TERMINATION_FIELDS),
                }
                for name, condition in task._termination_conditions.items()
            },
            "reward_functions": {
                name: {
                    "class": _qualified_class_name(reward),
                    "state": _capture_present_attributes(reward, _REWARD_FIELDS),
                }
                for name, reward in task._reward_functions.items()
            },
            "metrics": [
                {"class": _qualified_class_name(metric), "state": _clone_value(metric.__dict__)} for metric in metrics
            ],
            "topology": _scene_topology(env),
        }

    def restore(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise SnapshotRestoreError("Evaluator snapshot component state is not a mapping.")
        env = self.evaluator.env
        task = env.task
        self._validate_owner(task, state.get("task"), "task")
        _restore_attributes(self.evaluator, self._mapping(state, "evaluator"))
        _restore_attributes(task, self._mapping(self._mapping(state, "task"), "state"))

        termination_states = self._mapping(state, "termination_conditions")
        if set(termination_states) != set(task._termination_conditions):
            raise SnapshotRestoreError("Termination-condition set changed since snapshot capture.")
        for name, condition in task._termination_conditions.items():
            entry = termination_states[name]
            self._validate_owner(condition, entry, f"termination condition {name}")
            _restore_attributes(condition, self._mapping(entry, "state"))

        reward_states = self._mapping(state, "reward_functions")
        if set(reward_states) != set(task._reward_functions):
            raise SnapshotRestoreError("Reward-function set changed since snapshot capture.")
        for name, reward in task._reward_functions.items():
            entry = reward_states[name]
            self._validate_owner(reward, entry, f"reward function {name}")
            _restore_attributes(reward, self._mapping(entry, "state"))

        metric_states = state.get("metrics")
        if not isinstance(metric_states, list) or len(metric_states) != len(self.evaluator.metrics):
            raise SnapshotRestoreError("Metric set changed since snapshot capture.")
        for index, (metric, entry) in enumerate(zip(self.evaluator.metrics, metric_states, strict=True)):
            self._validate_owner(metric, entry, f"metric {index}")
            metric.__dict__.clear()
            metric.__dict__.update(_clone_value(self._mapping(entry, "state")))

    def after_propagation(self, state: object) -> Mapping[str, object]:
        if not isinstance(state, Mapping):
            raise SnapshotRestoreError("Evaluator snapshot component state is not a mapping.")
        expected_topology = self._mapping(state, "topology")
        observed_topology = _scene_topology(self.evaluator.env)
        if observed_topology != expected_topology:
            raise SnapshotRestoreError(
                f"Scene topology drifted during restore: expected {expected_topology}, got {observed_topology}."
            )

        progress_checked = False
        if self.progress_reader is not None:
            progress_checked = True
            expected_progress = self._mapping(state, "evaluator").get("latest_task_progress")
            observed_progress = self.progress_reader()
            if _clone_value(observed_progress) != _clone_value(expected_progress):
                raise SnapshotRestoreError(
                    f"Task progress drifted during restore: expected {expected_progress}, got {observed_progress}."
                )
        observation_refreshed = False
        if self.observation_reader is not None:
            self.evaluator.obs = self.observation_reader()
            observation_refreshed = True
        return {
            "topology_verified": True,
            "progress_verified": progress_checked,
            "observation_refreshed": observation_refreshed,
        }

    @staticmethod
    def _mapping(value: object, key: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise SnapshotRestoreError(f"Evaluator snapshot field {key} is not a mapping.")
        child = value.get(key)
        if not isinstance(child, Mapping):
            raise SnapshotRestoreError(f"Evaluator snapshot field {key} is missing or invalid.")
        return child

    @staticmethod
    def _validate_owner(owner: object, entry: object, label: str) -> None:
        if not isinstance(entry, Mapping) or entry.get("class") != _qualified_class_name(owner):
            raise SnapshotRestoreError(f"Evaluator snapshot {label} class does not match the active runtime.")
