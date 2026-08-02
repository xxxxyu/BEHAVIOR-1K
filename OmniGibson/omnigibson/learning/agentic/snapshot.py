"""Composite in-process snapshots for deterministic agentic branching experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any, Protocol


SNAPSHOT_SCHEMA_VERSION = 1
PERSISTED_SNAPSHOT_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


class SnapshotError(RuntimeError):
    """Base snapshot failure."""


class SnapshotCompatibilityError(SnapshotError):
    """Snapshot does not match the active runtime."""


class SnapshotRestoreError(SnapshotError):
    """Snapshot restore failed after compatibility validation."""


class SnapshotComponent(Protocol):
    name: str

    def capture(self) -> object: ...

    def restore(self, state: object) -> None: ...

    def after_propagation(self, state: object) -> Mapping[str, object] | None: ...


@dataclasses.dataclass(frozen=True)
class SnapshotMetadata:
    task_name: str
    instance_id: str
    episode_id: str
    env_step: int
    runtime_fingerprint: str
    controller_fingerprint: str
    parent_snapshot_id: str | None = None
    created_wall_time_ns: int = dataclasses.field(default_factory=time.time_ns)
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported snapshot schema version: {self.schema_version}.")
        if not all((self.task_name, self.instance_id, self.episode_id)):
            raise ValueError("Snapshot metadata requires task, instance, and episode identifiers.")
        if self.env_step < 0:
            raise ValueError("Snapshot env_step cannot be negative.")
        if not self.runtime_fingerprint or not self.controller_fingerprint:
            raise ValueError("Snapshot metadata requires runtime and controller fingerprints.")

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "instance_id": self.instance_id,
            "episode_id": self.episode_id,
            "env_step": self.env_step,
            "runtime_fingerprint": self.runtime_fingerprint,
            "controller_fingerprint": self.controller_fingerprint,
            "parent_snapshot_id": self.parent_snapshot_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "created_wall_time_ns": self.created_wall_time_ns}


@dataclasses.dataclass(frozen=True)
class RngSnapshot:
    python_state: object
    numpy_state: object | None
    torch_cpu_state: object | None
    torch_cuda_states: tuple[object, ...]


@dataclasses.dataclass(frozen=True)
class CompositeSnapshot:
    snapshot_id: str
    metadata: SnapshotMetadata
    simulator_state: object
    env_episode: int
    component_states: Mapping[str, object]
    rng_state: RngSnapshot


@dataclasses.dataclass(frozen=True)
class RestoreReport:
    snapshot_id: str
    restored_env_step: int
    propagation_sim_steps: int
    propagation_details: tuple[Mapping[str, object], ...]
    component_reports: Mapping[str, Mapping[str, object]]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "restored_env_step": self.restored_env_step,
            "propagation_sim_steps": self.propagation_sim_steps,
            "propagation_details": [dict(item) for item in self.propagation_details],
            "component_reports": {name: dict(value) for name, value in self.component_reports.items()},
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclasses.dataclass
class AttributeSnapshotComponent:
    """Captures an explicit attribute allowlist from one Python-side owner."""

    name: str
    owner: object
    attributes: tuple[str, ...]
    after_propagation_callback: Callable[[object], Mapping[str, object] | None] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.attributes:
            raise ValueError("Attribute snapshot component requires a name and at least one attribute.")

    def capture(self) -> object:
        missing = [attribute for attribute in self.attributes if not hasattr(self.owner, attribute)]
        if missing:
            raise SnapshotError(f"Component {self.name} is missing attributes: {missing}.")
        return {attribute: _clone_value(getattr(self.owner, attribute)) for attribute in self.attributes}

    def restore(self, state: object) -> None:
        if not isinstance(state, Mapping) or set(state) != set(self.attributes):
            raise SnapshotRestoreError(f"Component {self.name} state does not match its attribute allowlist.")
        for attribute in self.attributes:
            setattr(self.owner, attribute, _clone_value(state[attribute]))

    def after_propagation(self, state: object) -> Mapping[str, object] | None:
        if self.after_propagation_callback is None:
            return None
        return self.after_propagation_callback(state)


def _clone_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if hasattr(value, "copy") and type(value).__module__.startswith("numpy"):
        return value.copy()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_value(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return copy.deepcopy(value)


def _hash_update(hasher: Any, value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str, bytes)):
        hasher.update(pickle.dumps(value, protocol=5))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SnapshotError("Cannot hash a non-finite scalar snapshot value.")
        hasher.update(pickle.dumps(value, protocol=5))
        return
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        hasher.update(str(value.dtype).encode())
        hasher.update(repr(tuple(value.shape)).encode())
        hasher.update(value.tobytes())
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_update(hasher, key)
            _hash_update(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        hasher.update(f"sequence:{type(value).__name__}[".encode())
        for item in value:
            _hash_update(hasher, item)
        hasher.update(b"]")
        return
    hasher.update(pickle.dumps(value, protocol=5))


def _snapshot_id(
    metadata: SnapshotMetadata,
    simulator_state: object,
    env_episode: int,
    component_states: Mapping[str, object],
    rng_state: RngSnapshot,
) -> str:
    hasher = hashlib.sha256()
    _hash_update(hasher, metadata.identity_dict())
    _hash_update(hasher, simulator_state)
    _hash_update(hasher, env_episode)
    _hash_update(hasher, component_states)
    _hash_update(hasher, dataclasses.astuple(rng_state))
    return f"sha256:{hasher.hexdigest()}"


def _capture_rng() -> RngSnapshot:
    numpy_state = None
    try:
        import numpy as np

        numpy_state = _clone_value(np.random.get_state())
    except ImportError:
        pass

    torch_cpu_state = None
    torch_cuda_states: tuple[object, ...] = ()
    try:
        import torch

        torch_cpu_state = _clone_value(torch.random.get_rng_state())
        if torch.cuda.is_initialized():
            torch_cuda_states = tuple(_clone_value(state) for state in torch.cuda.get_rng_state_all())
    except ImportError:
        pass

    return RngSnapshot(
        python_state=_clone_value(random.getstate()),
        numpy_state=numpy_state,
        torch_cpu_state=torch_cpu_state,
        torch_cuda_states=torch_cuda_states,
    )


def _restore_rng(state: RngSnapshot) -> None:
    random.setstate(_clone_value(state.python_state))
    if state.numpy_state is not None:
        import numpy as np

        np.random.set_state(_clone_value(state.numpy_state))
    if state.torch_cpu_state is not None:
        import torch

        torch.random.set_rng_state(_clone_value(state.torch_cpu_state))
        if state.torch_cuda_states:
            if not torch.cuda.is_initialized():
                raise SnapshotCompatibilityError("Snapshot contains CUDA RNG state but CUDA is not initialized.")
            if len(state.torch_cuda_states) != torch.cuda.device_count():
                raise SnapshotCompatibilityError("Snapshot CUDA RNG device count differs from the active runtime.")
            torch.cuda.set_rng_state_all([_clone_value(item) for item in state.torch_cuda_states])


class CompositeSnapshotManager:
    """Captures and restores simulator plus explicitly registered episode state."""

    def __init__(
        self,
        *,
        simulator: object,
        env: object,
        runtime_fingerprint: str,
        controller_fingerprint: str,
        components: Sequence[SnapshotComponent] = (),
        propagate_once: Callable[[], Mapping[str, object] | None],
        restore_diagnostic: Callable[[str, CompositeSnapshot], None] | None = None,
        serialized_simulator_state: bool = True,
    ) -> None:
        if not runtime_fingerprint or not controller_fingerprint:
            raise ValueError("Snapshot manager requires runtime and controller fingerprints.")
        names = [component.name for component in components]
        if len(names) != len(set(names)):
            raise ValueError("Snapshot component names must be unique.")
        self.simulator = simulator
        self.env = env
        self.runtime_fingerprint = runtime_fingerprint
        self.controller_fingerprint = controller_fingerprint
        self.components = tuple(components)
        self.propagate_once = propagate_once
        self.restore_diagnostic = restore_diagnostic
        self.serialized_simulator_state = serialized_simulator_state
        self._restore_failure_once: str | None = None

    def inject_restore_failure_once(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Injected restore failure requires a reason.")
        if self._restore_failure_once is not None:
            raise RuntimeError("A restore failure is already pending.")
        self._restore_failure_once = reason

    def capture(
        self,
        *,
        task_name: str,
        instance_id: str,
        episode_id: str,
        parent_snapshot_id: str | None = None,
    ) -> CompositeSnapshot:
        env_step = int(getattr(self.env, "_current_step"))
        env_episode = int(getattr(self.env, "_current_episode"))
        metadata = SnapshotMetadata(
            task_name=task_name,
            instance_id=instance_id,
            episode_id=episode_id,
            env_step=env_step,
            runtime_fingerprint=self.runtime_fingerprint,
            controller_fingerprint=self.controller_fingerprint,
            parent_snapshot_id=parent_snapshot_id,
        )
        simulator_state = _clone_value(self.simulator.dump_state(serialized=self.serialized_simulator_state))
        component_states = {component.name: _clone_value(component.capture()) for component in self.components}
        rng_state = _capture_rng()
        snapshot_id = _snapshot_id(metadata, simulator_state, env_episode, component_states, rng_state)
        return CompositeSnapshot(
            snapshot_id=snapshot_id,
            metadata=metadata,
            simulator_state=simulator_state,
            env_episode=env_episode,
            component_states=component_states,
            rng_state=rng_state,
        )

    def restore(self, snapshot: CompositeSnapshot) -> RestoreReport:
        self._validate_compatibility(snapshot)
        rollback = self.capture(
            task_name=snapshot.metadata.task_name,
            instance_id=snapshot.metadata.instance_id,
            episode_id=snapshot.metadata.episode_id,
            parent_snapshot_id=snapshot.snapshot_id,
        )
        self._emit_restore_diagnostic("before_target_restore", snapshot)
        started = time.monotonic()
        try:
            report = self._restore_once(snapshot, started=started)
            self._emit_restore_diagnostic("target_restore_complete", snapshot)
            return report
        except Exception as exc:
            self._emit_restore_diagnostic("target_restore_failed_before_rollback", snapshot)
            self._restore_failure_once = None
            try:
                self._restore_once(rollback, started=time.monotonic())
                self._emit_restore_diagnostic("rollback_complete", rollback)
            except Exception as rollback_exc:
                raise SnapshotRestoreError(
                    f"Snapshot {snapshot.snapshot_id} restore failed ({exc}); rollback to "
                    f"{rollback.snapshot_id} also failed ({rollback_exc})."
                ) from rollback_exc
            raise SnapshotRestoreError(
                f"Snapshot {snapshot.snapshot_id} restore failed and was rolled back: {exc}"
            ) from exc

    def _emit_restore_diagnostic(self, phase: str, snapshot: CompositeSnapshot) -> None:
        if self.restore_diagnostic is None:
            return
        try:
            self.restore_diagnostic(phase, snapshot)
        except Exception:
            # Diagnostics must never alter restore semantics.
            logger.exception("Snapshot restore diagnostic callback failed during %s.", phase)
            return

    def _restore_once(self, snapshot: CompositeSnapshot, *, started: float) -> RestoreReport:
        self.simulator.load_state(_clone_value(snapshot.simulator_state), serialized=self.serialized_simulator_state)
        setattr(self.env, "_current_step", snapshot.metadata.env_step)
        setattr(self.env, "_current_episode", snapshot.env_episode)
        for component in self.components:
            component.restore(_clone_value(snapshot.component_states[component.name]))
        _restore_rng(snapshot.rng_state)

        propagation = self.propagate_once()
        propagation_details = (
            {
                "kind": "omnigibson_post_load_state_propagation",
                "sim_steps": 1,
                "environment_step_increment": 0,
                **dict(propagation or {}),
            },
        )
        component_reports: dict[str, Mapping[str, object]] = {}
        for component in self.components:
            report = component.after_propagation(_clone_value(snapshot.component_states[component.name]))
            if report is not None:
                component_reports[component.name] = dict(report)
        if self._restore_failure_once is not None:
            reason = self._restore_failure_once
            self._restore_failure_once = None
            raise SnapshotRestoreError(f"Injected post-propagation restore failure: {reason}")
        return RestoreReport(
            snapshot_id=snapshot.snapshot_id,
            restored_env_step=int(getattr(self.env, "_current_step")),
            propagation_sim_steps=1,
            propagation_details=propagation_details,
            component_reports=component_reports,
            elapsed_seconds=time.monotonic() - started,
        )

    def _validate_compatibility(self, snapshot: CompositeSnapshot) -> None:
        if snapshot.metadata.runtime_fingerprint != self.runtime_fingerprint:
            raise SnapshotCompatibilityError("Snapshot runtime fingerprint does not match the active runtime.")
        if snapshot.metadata.controller_fingerprint != self.controller_fingerprint:
            raise SnapshotCompatibilityError("Snapshot controller fingerprint does not match the active controller.")
        expected_components = {component.name for component in self.components}
        if set(snapshot.component_states) != expected_components:
            raise SnapshotCompatibilityError("Snapshot component set does not match the active snapshot manager.")


def persist_snapshot(snapshot: CompositeSnapshot, directory: Path) -> dict[str, str]:
    """Persist one runtime-private snapshot with an independently verified payload hash."""

    expected_id = _snapshot_id(
        snapshot.metadata,
        snapshot.simulator_state,
        snapshot.env_episode,
        snapshot.component_states,
        snapshot.rng_state,
    )
    if snapshot.snapshot_id != expected_id:
        raise SnapshotError("Snapshot identity does not match its payload.")
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    stem = snapshot.snapshot_id.removeprefix("sha256:")
    payload_path = directory / f"{stem}.pkl"
    manifest_path = directory / f"{stem}.json"
    if payload_path.exists() and manifest_path.exists():
        existing = load_persisted_snapshot(manifest_path)
        if existing.snapshot_id != snapshot.snapshot_id:
            raise FileExistsError(f"Conflicting persisted snapshot: {snapshot.snapshot_id}.")
        return {"manifest_path": str(manifest_path), "payload_path": str(payload_path)}
    if payload_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Incomplete persisted snapshot: {snapshot.snapshot_id}.")
    payload = pickle.dumps(snapshot, protocol=5)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": PERSISTED_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "payload_file": payload_path.name,
        "payload_sha256": payload_sha256,
        "metadata": snapshot.metadata.to_dict(),
    }
    temporary_payload = payload_path.with_name(f".{payload_path.name}.{os.getpid()}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary_payload.write_bytes(payload)
    os.chmod(temporary_payload, 0o600)
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary_manifest, 0o600)
    temporary_payload.replace(payload_path)
    temporary_manifest.replace(manifest_path)
    return {"manifest_path": str(manifest_path), "payload_path": str(payload_path)}


def load_persisted_snapshot(manifest_path: Path) -> CompositeSnapshot:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PERSISTED_SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotCompatibilityError("Unsupported persisted snapshot schema.")
    payload_path = manifest_path.parent / str(manifest.get("payload_file"))
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("payload_sha256"):
        raise SnapshotCompatibilityError("Persisted snapshot payload hash does not match its manifest.")
    snapshot = pickle.loads(payload)
    if not isinstance(snapshot, CompositeSnapshot):
        raise SnapshotCompatibilityError("Persisted payload is not a CompositeSnapshot.")
    expected_id = _snapshot_id(
        snapshot.metadata,
        snapshot.simulator_state,
        snapshot.env_episode,
        snapshot.component_states,
        snapshot.rng_state,
    )
    if snapshot.snapshot_id != manifest.get("snapshot_id") or snapshot.snapshot_id != expected_id:
        raise SnapshotCompatibilityError("Persisted snapshot identity validation failed.")
    return snapshot
