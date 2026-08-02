"""Runtime-derived robot action contract for agentic control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import math
from typing import Any

import numpy as np


CONTROLLER_MANIFEST_SCHEMA_VERSION = 1
RUNTIME_REFERENCE_LIMIT_TOLERANCE = 1e-5
R1PRO_CONTROLLER_WIDTHS = {
    "base": 3,
    "trunk": 4,
    "arm_left": 7,
    "gripper_left": 1,
    "arm_right": 7,
    "gripper_right": 1,
}


class ControllerManifestError(ValueError):
    """Raised when the loaded controller contract is unsafe or inconsistent."""


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _float_tuple(value: Any, *, expected: int, label: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in _to_list(value))
    if len(values) == 1 and expected > 1:
        values *= expected
    if len(values) != expected:
        raise ControllerManifestError(f"{label} has width {len(values)}, expected {expected}.")
    if not all(math.isfinite(item) for item in values):
        raise ControllerManifestError(f"{label} contains a non-finite value.")
    return values


def _int_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _to_list(value))
    except (TypeError, ValueError) as exc:
        raise ControllerManifestError(f"{label} is not an integer index sequence.") from exc


def _limit_pair(value: Any, *, expected: int, label: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if value is None or not isinstance(value, Sequence) or len(value) != 2:
        raise ControllerManifestError(f"{label} must contain lower and upper limits.")
    lower = _float_tuple(value[0], expected=expected, label=f"{label}.lower")
    upper = _float_tuple(value[1], expected=expected, label=f"{label}.upper")
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ControllerManifestError(f"{label} has a lower limit above its upper limit.")
    return lower, upper


def _frequency_from_dt(value: Any, *, label: str) -> float:
    dt = float(value)
    if not math.isfinite(dt) or dt <= 0:
        raise ControllerManifestError(f"{label} must be a positive finite timestep.")
    return 1.0 / dt


def _controller_motor_type(controller: object) -> str:
    motor_type = getattr(controller, "_motor_type", None)
    if motor_type is not None:
        return str(motor_type).lower()
    control_type = getattr(controller, "control_type", None)
    return {0: "position", 1: "velocity", 2: "effort"}.get(control_type, f"unknown:{control_type}")


def _semantic_for(name: str, controller: object, *, input_limits_present: bool) -> str:
    motor_type = _controller_motor_type(controller)
    delta = bool(getattr(controller, "_use_delta_commands", False))
    if name.startswith("gripper_"):
        return "normalized_gripper" if input_limits_present else f"{motor_type}_gripper"
    if motor_type == "position":
        return "delta_joint_position" if delta else "absolute_joint_position"
    if motor_type == "velocity":
        return "normalized_velocity" if input_limits_present else "joint_velocity"
    return f"{motor_type}_command"


def _physical_command_limits(
    controller: object, *, command_dim: int, name: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    control_type = getattr(controller, "control_type", None)
    control_limits = getattr(controller, "_control_limits", None)
    if not isinstance(control_limits, Mapping) or control_type not in control_limits:
        raise ControllerManifestError(
            f"Controller {name} has no command input limits and exposes no physical control limits."
        )
    lower_all, upper_all = control_limits[control_type]
    dof_idx = _int_tuple(getattr(controller, "dof_idx", ()), label=f"{name}.dof_idx")
    if len(dof_idx) != command_dim:
        raise ControllerManifestError(
            f"Controller {name} needs {command_dim} physical limits but controls {len(dof_idx)} DOFs."
        )
    lower_values = _to_list(lower_all)
    upper_values = _to_list(upper_all)
    try:
        lower = tuple(float(lower_values[index]) for index in dof_idx)
        upper = tuple(float(upper_values[index]) for index in dof_idx)
    except (IndexError, TypeError, ValueError) as exc:
        raise ControllerManifestError(f"Controller {name} physical limits cannot be indexed by dof_idx.") from exc
    return _limit_pair((lower, upper), expected=command_dim, label=f"{name}.physical_limits")


@dataclasses.dataclass(frozen=True)
class ControllerSegment:
    name: str
    start: int
    stop: int
    controller_class: str
    command_dim: int
    dof_indices: tuple[int, ...]
    motor_type: str
    semantic: str
    uses_delta_commands: bool
    command_input_limits: tuple[tuple[float, ...], tuple[float, ...]] | None
    command_output_limits: tuple[tuple[float, ...], tuple[float, ...]] | None
    safety_input_limits: tuple[tuple[float, ...], tuple[float, ...]]
    safety_limit_source: str

    def __post_init__(self) -> None:
        if not self.name or self.start < 0 or self.stop <= self.start:
            raise ControllerManifestError(f"Invalid controller segment: {self.name!r} [{self.start}, {self.stop}).")
        if self.command_dim != self.stop - self.start:
            raise ControllerManifestError(f"Controller {self.name} command_dim does not match its action slice.")
        _limit_pair(self.safety_input_limits, expected=self.command_dim, label=f"{self.name}.safety_input_limits")

    def to_dict(self) -> dict[str, object]:
        def limits(value: tuple[tuple[float, ...], tuple[float, ...]] | None) -> list[list[float]] | None:
            return None if value is None else [list(value[0]), list(value[1])]

        return {
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "controller_class": self.controller_class,
            "command_dim": self.command_dim,
            "dof_indices": list(self.dof_indices),
            "motor_type": self.motor_type,
            "semantic": self.semantic,
            "uses_delta_commands": self.uses_delta_commands,
            "command_input_limits": limits(self.command_input_limits),
            "command_output_limits": limits(self.command_output_limits),
            "safety_input_limits": limits(self.safety_input_limits),
            "safety_limit_source": self.safety_limit_source,
        }


@dataclasses.dataclass(frozen=True)
class ControllerManifest:
    robot_name: str
    robot_class: str
    action_dim: int
    action_frequency_hz: float
    physics_frequency_hz: float
    action_normalize: bool
    segments: tuple[ControllerSegment, ...]
    schema_version: int = CONTROLLER_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTROLLER_MANIFEST_SCHEMA_VERSION:
            raise ControllerManifestError(f"Unsupported controller manifest schema: {self.schema_version}.")
        if not self.robot_name or not self.robot_class:
            raise ControllerManifestError("Controller manifest requires robot_name and robot_class.")
        if self.action_dim <= 0 or self.action_frequency_hz <= 0 or self.physics_frequency_hz <= 0:
            raise ControllerManifestError("Action dimensions and frequencies must be positive.")
        expected_start = 0
        names: set[str] = set()
        for segment in self.segments:
            if segment.start != expected_start:
                raise ControllerManifestError("Controller segments must be contiguous from action index zero.")
            if segment.name in names:
                raise ControllerManifestError(f"Duplicate controller segment {segment.name}.")
            names.add(segment.name)
            expected_start = segment.stop
        if expected_start != self.action_dim:
            raise ControllerManifestError(
                f"Controller segments end at {expected_start}, but robot action_dim is {self.action_dim}."
            )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "robot_name": self.robot_name,
            "robot_class": self.robot_class,
            "action_dim": self.action_dim,
            "action_frequency_hz": self.action_frequency_hz,
            "physics_frequency_hz": self.physics_frequency_hz,
            "action_normalize": self.action_normalize,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


def normalize_runtime_reference_action(
    action: object,
    manifest: ControllerManifest,
    *,
    tolerance: float = RUNTIME_REFERENCE_LIMIT_TOLERANCE,
) -> np.ndarray:
    """Clamp runtime-generated no-op references only within numerical tolerance."""

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ControllerManifestError("Runtime reference tolerance must be finite and non-negative.")
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape != (manifest.action_dim,) or not np.isfinite(array).all():
        raise ControllerManifestError(
            f"Runtime reference action must be finite with shape ({manifest.action_dim},), got {array.shape}."
        )
    lower = np.concatenate(
        [np.asarray(segment.safety_input_limits[0], dtype=np.float32) for segment in manifest.segments]
    )
    upper = np.concatenate(
        [np.asarray(segment.safety_input_limits[1], dtype=np.float32) for segment in manifest.segments]
    )
    below = lower - array
    above = array - upper
    violation = np.maximum(below, above)
    if np.any(violation > tolerance):
        channel = int(np.argmax(violation))
        raise ControllerManifestError(
            "Runtime-generated reference action exceeds controller limits beyond numerical tolerance: "
            f"channel={channel} value={float(array[channel])} "
            f"limits=[{float(lower[channel])}, {float(upper[channel])}] tolerance={tolerance}."
        )
    return np.clip(array, lower, upper).astype(np.float32, copy=False)


def build_controller_manifest(
    robot: object,
    *,
    simulator: object | None = None,
    action_frequency_hz: float | None = None,
    physics_frequency_hz: float | None = None,
) -> ControllerManifest:
    """Builds a fail-closed action manifest from the loaded runtime controllers."""

    try:
        controller_order = tuple(robot.controller_order)
        action_indices = robot.controller_action_idx
        controllers = robot.controllers
        action_dim = int(robot.action_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControllerManifestError("Robot does not expose the required runtime controller interface.") from exc

    if action_frequency_hz is None:
        action_frequency_hz = getattr(robot, "control_freq", getattr(robot, "_control_freq", None))
    if action_frequency_hz is None and simulator is not None:
        action_frequency_hz = _frequency_from_dt(simulator.get_sim_step_dt(), label="sim_step_dt")
    if physics_frequency_hz is None and simulator is not None:
        physics_frequency_hz = _frequency_from_dt(simulator.get_physics_dt(), label="physics_dt")
    if action_frequency_hz is None or physics_frequency_hz is None:
        raise ControllerManifestError("Action and physics frequencies must be explicit or runtime-discoverable.")

    segments: list[ControllerSegment] = []
    expected_start = 0
    for name in controller_order:
        if name not in controllers or name not in action_indices:
            raise ControllerManifestError(f"Controller {name} is absent from controllers or controller_action_idx.")
        controller = controllers[name]
        command_dim = int(controller.command_dim)
        indices = _int_tuple(action_indices[name], label=f"{name}.action_indices")
        expected_indices = tuple(range(expected_start, expected_start + command_dim))
        if indices != expected_indices:
            raise ControllerManifestError(
                f"Controller {name} action indices {indices} are not the expected contiguous slice {expected_indices}."
            )

        raw_input_limits = getattr(controller, "command_input_limits", None)
        raw_output_limits = getattr(controller, "command_output_limits", None)
        input_limits = (
            None
            if raw_input_limits is None
            else _limit_pair(raw_input_limits, expected=command_dim, label=f"{name}.command_input_limits")
        )
        output_limits = (
            None
            if raw_output_limits is None
            else _limit_pair(raw_output_limits, expected=command_dim, label=f"{name}.command_output_limits")
        )
        if input_limits is not None:
            safety_limits = input_limits
            safety_source = "command_input_limits"
        elif _controller_motor_type(controller) == "position" and not getattr(controller, "_use_delta_commands", False):
            safety_limits = _physical_command_limits(controller, command_dim=command_dim, name=name)
            safety_source = "physical_position_limits"
        else:
            raise ControllerManifestError(
                f"Controller {name} has no bounded input contract suitable for direct agent actions."
            )

        segments.append(
            ControllerSegment(
                name=name,
                start=expected_start,
                stop=expected_start + command_dim,
                controller_class=type(controller).__name__,
                command_dim=command_dim,
                dof_indices=_int_tuple(getattr(controller, "dof_idx", ()), label=f"{name}.dof_idx"),
                motor_type=_controller_motor_type(controller),
                semantic=_semantic_for(name, controller, input_limits_present=input_limits is not None),
                uses_delta_commands=bool(getattr(controller, "_use_delta_commands", False)),
                command_input_limits=input_limits,
                command_output_limits=output_limits,
                safety_input_limits=safety_limits,
                safety_limit_source=safety_source,
            )
        )
        expected_start += command_dim

    manifest = ControllerManifest(
        robot_name=str(getattr(robot, "name", type(robot).__name__)),
        robot_class=type(robot).__name__,
        action_dim=action_dim,
        action_frequency_hz=float(action_frequency_hz),
        physics_frequency_hz=float(physics_frequency_hz),
        action_normalize=bool(getattr(robot, "_action_normalize", False)),
        segments=tuple(segments),
    )
    return manifest


def validate_r1pro_manifest(manifest: ControllerManifest) -> None:
    """Rejects an R1Pro runtime that differs from the validated 23-D challenge contract."""

    if manifest.robot_class != "R1Pro":
        raise ControllerManifestError(f"Expected R1Pro, got {manifest.robot_class}.")
    if manifest.action_normalize:
        raise ControllerManifestError("R1Pro challenge actions require action_normalize=False.")
    actual = {segment.name: segment.command_dim for segment in manifest.segments}
    if actual != R1PRO_CONTROLLER_WIDTHS:
        raise ControllerManifestError(f"Unexpected R1Pro controller layout: {actual}.")
    if tuple(actual) != tuple(R1PRO_CONTROLLER_WIDTHS):
        raise ControllerManifestError(f"Unexpected R1Pro controller order: {tuple(actual)}.")
    if manifest.action_dim != 23:
        raise ControllerManifestError(f"Expected 23-D R1Pro action, got {manifest.action_dim}.")
    for segment in manifest.segments:
        if segment.name in {"trunk", "arm_left", "arm_right"}:
            if segment.semantic != "absolute_joint_position" or segment.uses_delta_commands:
                raise ControllerManifestError(f"{segment.name} must use absolute position commands.")
        elif segment.name == "base" and segment.semantic != "normalized_velocity":
            raise ControllerManifestError("R1Pro base must use bounded normalized velocity commands.")
        elif segment.name.startswith("gripper_") and segment.semantic != "normalized_gripper":
            raise ControllerManifestError(f"{segment.name} must use bounded normalized gripper commands.")


def validate_runtime_action(manifest: ControllerManifest, action: object) -> np.ndarray:
    """Validates one environment action against the loaded controller contract."""

    try:
        array = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ControllerManifestError("Runtime action is not a numeric array.") from exc
    if array.shape != (manifest.action_dim,):
        raise ControllerManifestError(f"Runtime action shape must be ({manifest.action_dim},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ControllerManifestError("Runtime action contains NaN or Inf.")
    for segment in manifest.segments:
        lower, upper = segment.safety_input_limits
        values = array[segment.start : segment.stop]
        violation = np.flatnonzero((values < np.asarray(lower)) | (values > np.asarray(upper)))
        if violation.size:
            local_index = int(violation[0])
            channel = segment.start + local_index
            raise ControllerManifestError(
                f"Runtime action channel {channel} ({segment.name}[{local_index}]) is outside safety limits."
            )
    return array.copy()
