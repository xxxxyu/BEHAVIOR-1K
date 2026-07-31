"""Single-owner control plane for one agentic BEHAVIOR environment."""

from __future__ import annotations

import dataclasses
import enum

import numpy as np

from omnigibson.learning.agentic.controller_manifest import ControllerManifest
from omnigibson.learning.agentic.controller_manifest import validate_runtime_action
from omnigibson.learning.agentic.snapshot import CompositeSnapshot
from omnigibson.learning.agentic.snapshot import CompositeSnapshotManager
from omnigibson.learning.agentic.snapshot import RestoreReport


class SessionState(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclasses.dataclass(frozen=True)
class EnvironmentStepResult:
    observation: object
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    env_step: int
    executed_action: np.ndarray


class AgenticEnvironmentSession:
    """Serializes stepping and branching operations for one loaded environment."""

    def __init__(
        self,
        *,
        env: object,
        controller_manifest: ControllerManifest,
        snapshot_manager: CompositeSnapshotManager,
    ) -> None:
        if snapshot_manager.env is not env:
            raise ValueError("Environment session and snapshot manager must own the same environment object.")
        if snapshot_manager.controller_fingerprint != controller_manifest.fingerprint:
            raise ValueError("Snapshot manager controller fingerprint does not match the runtime manifest.")
        self.env = env
        self.controller_manifest = controller_manifest
        self.snapshot_manager = snapshot_manager
        self._state = SessionState.ACTIVE

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def env_step(self) -> int:
        return int(getattr(self.env, "_current_step"))

    def step(self, action: object, *, n_render_iterations: int = 1) -> EnvironmentStepResult:
        if self._state != SessionState.ACTIVE:
            raise RuntimeError(f"Cannot step environment session while {self._state.value}.")
        if n_render_iterations < 1:
            raise ValueError("n_render_iterations must be positive.")
        validated = validate_runtime_action(self.controller_manifest, action)
        observation, reward, terminated, truncated, info = self.env.step(
            validated, n_render_iterations=n_render_iterations
        )
        return EnvironmentStepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            env_step=self.env_step,
            executed_action=validated,
        )

    def observe(self) -> tuple[object, object]:
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Cannot observe a closed environment session.")
        return self.env.get_obs()

    def pause(self) -> None:
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Cannot pause a closed environment session.")
        self._state = SessionState.PAUSED

    def resume(self) -> None:
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Cannot resume a closed environment session.")
        self._state = SessionState.ACTIVE

    def snapshot(
        self,
        *,
        task_name: str,
        instance_id: str,
        episode_id: str,
        parent_snapshot_id: str | None = None,
    ) -> CompositeSnapshot:
        self._require_paused("snapshot")
        return self.snapshot_manager.capture(
            task_name=task_name,
            instance_id=instance_id,
            episode_id=episode_id,
            parent_snapshot_id=parent_snapshot_id,
        )

    def restore(self, snapshot: CompositeSnapshot) -> RestoreReport:
        self._require_paused("restore")
        return self.snapshot_manager.restore(snapshot)

    def close(self) -> None:
        self._state = SessionState.CLOSED

    def _require_paused(self, operation: str) -> None:
        if self._state != SessionState.PAUSED:
            raise RuntimeError(f"Environment session must be paused before {operation}; state={self._state.value}.")
