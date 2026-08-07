"""Local websocket control plane for one real agentic BEHAVIOR episode."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
import hashlib
import http
from inspect import getsourcefile
import json
import logging
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.learning.agentic.controller_manifest import build_controller_manifest
from omnigibson.learning.agentic.controller_manifest import normalize_runtime_reference_action
from omnigibson.learning.agentic.controller_manifest import validate_r1pro_manifest
from omnigibson.learning.agentic.environment_session import AgenticEnvironmentSession
from omnigibson.learning.agentic.evaluator_state import EvaluatorSnapshotComponent
from omnigibson.learning.agentic.kinematics import bounded_pose_delta_ik_step
from omnigibson.learning.agentic.kinematics import bounded_position_ik_step
from omnigibson.learning.agentic.retry_checkpoint import assess_radio_demo_primary_checkpoint
from omnigibson.learning.agentic.retry_checkpoint import assess_radio_pickup_preclose_checkpoint
from omnigibson.learning.agentic.semantic_geometry import capture_navigation_geometry
from omnigibson.learning.agentic.semantic_geometry import navigation_geometry_deltas
from omnigibson.learning.agentic.semantic_geometry_backend import RadioSemanticGeometryBackend
from omnigibson.learning.agentic.snapshot import CompositeSnapshot
from omnigibson.learning.agentic.snapshot import CompositeSnapshotManager
from omnigibson.learning.agentic.snapshot import AttributeSnapshotComponent
from omnigibson.learning.agentic.snapshot import load_persisted_snapshot
from omnigibson.learning.agentic.snapshot import persist_snapshot
from omnigibson.learning.eval_custom import Evaluator
from omnigibson.learning.eval_custom import _normalize_task_progress
from omnigibson.object_states import ContactBodies
from omnigibson.object_states import OnTop
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import CAMERA_INTRINSICS
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.network_utils import Packer
from omnigibson.learning.utils.network_utils import unpackb
from omnigibson.learning.utils.task_progress_utils import CHALLENGE_TASKS_PROGRESS_APPROXIMATION
from omnigibson.learning.utils.task_progress_utils import _resolve_object
from omnigibson.macros import gm
import torch as th
import websockets
import websockets.asyncio.server as websocket_server


logger = logging.getLogger(__name__)
PROTOCOL_VERSION = 2
CAMERA_POSE_KEY = "robot_r1::cam_rel_poses"
JAW_CORRIDOR_CALIBRATION = {
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
RADIO_PICKUP_PRECLOSE_ROLE = "radio_pickup_preclose_right"
CHECKPOINT_ROLES = frozenset({"general", "demo_primary", "direct_press_fallback", RADIO_PICKUP_PRECLOSE_ROLE})
REQUEST_QUEUE_CAPACITY = 1
REQUEST_TIMEOUT_SECONDS = 300.0
NETWORK_STARTUP_TIMEOUT_SECONDS = 10.0
NETWORK_SHUTDOWN_TIMEOUT_SECONDS = 10.0
DISPATCH_POLL_SECONDS = 0.05


def _no_op_action(robot: object) -> np.ndarray:
    control_dict = robot.get_control_dict()
    actions = [
        robot.controllers[name].compute_no_op_action(control_dict=control_dict) for name in robot.controller_order
    ]
    return th.cat(actions).detach().cpu().numpy().astype(np.float32, copy=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class AgenticEvaluatorRuntime:
    """Keeps the official evaluator state synchronized with explicit tool steps."""

    def __init__(
        self,
        config: object,
        *,
        instance_id: int,
        episode_id: str,
        information_arm: str = "vision_rgb",
        snapshot_dir: Path | None = None,
        snapshot_import_dirs: tuple[Path, ...] = (),
        radio_pickup_fixture_reference_snapshot_id: str | None = None,
        enable_semantic_geometry_diagnostic: bool = False,
    ) -> None:
        if information_arm not in {"vision_rgb", "vision_rgb_depth"}:
            raise ValueError(f"Unsupported agentic information arm: {information_arm!r}.")
        self.config = config
        self.information_arm = information_arm
        self.task_name = str(config.task.name)
        self.instance_id = int(instance_id)
        self.episode_id = episode_id
        self.snapshot_dir = snapshot_dir
        self.restore_diagnostic_path = (
            snapshot_dir.parent / "restore_diagnostics.jsonl" if snapshot_dir is not None else None
        )
        self.snapshot_import_dirs = snapshot_import_dirs
        self.enable_semantic_geometry_diagnostic = enable_semantic_geometry_diagnostic
        if enable_semantic_geometry_diagnostic and self.task_name != "turning_on_radio":
            raise ValueError("Semantic geometry diagnostics are available only for turning_on_radio.")
        self.evaluator = Evaluator(config)
        self.evaluator.reset()
        self.evaluator.load_task_instance(self.instance_id)
        self.evaluator.obs = self.evaluator._preprocess_obs(self.evaluator.env.get_obs()[0])
        self.evaluator.begin_rollout_metadata(self.instance_id, 0)
        for metric in self.evaluator.metrics:
            metric.start_callback(self.evaluator.env)

        self.snapshots: dict[str, CompositeSnapshot] = {}
        self.last_reward = 0.0
        self.last_info: dict[str, object] = {}
        self.terminated = False
        self.truncated = False
        self.finished = False
        self._metrics_finished = False
        self._last_metrics: dict[str, object] | None = None
        self.attempt_index = 0
        self._navigation_geometry_baseline: dict[str, object] | None = None
        self._navigation_geometry_baseline_snapshot_id: str | None = None

        self.robot = self.evaluator.robot
        self._semantic_geometry_backend: RadioSemanticGeometryBackend | None = None
        self._semantic_geometry_backend_error: str | None = None
        if self.enable_semantic_geometry_diagnostic:
            try:
                self._semantic_geometry_backend = RadioSemanticGeometryBackend(
                    self.evaluator.env,
                    self.robot,
                )
            except Exception as exc:
                # Keep the diagnostic runtime usable and report the missing capability
                # explicitly.  Route acceptance remains fail-closed until this succeeds.
                self._semantic_geometry_backend_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Could not initialize the semantic-geometry backend: %s", self._semantic_geometry_backend_error
                )
        self.controller_manifest = build_controller_manifest(self.robot, simulator=og.sim)
        validate_r1pro_manifest(self.controller_manifest)
        progress_fn = CHALLENGE_TASKS_PROGRESS_APPROXIMATION[self.task_name]

        def progress_reader() -> dict[str, object]:
            return _normalize_task_progress(progress_fn(self.evaluator.env)) or {}

        evaluator_component = EvaluatorSnapshotComponent(
            evaluator=self.evaluator,
            progress_reader=progress_reader,
            observation_reader=lambda: self.evaluator._preprocess_obs(self.evaluator.env.get_obs()[0]),
        )
        runtime_component = AttributeSnapshotComponent(
            name="agentic_runtime",
            owner=self,
            attributes=(
                "last_reward",
                "last_info",
                "terminated",
                "truncated",
                "finished",
                "_metrics_finished",
                "_last_metrics",
            ),
        )
        manager = CompositeSnapshotManager(
            simulator=og.sim,
            env=self.evaluator.env,
            runtime_fingerprint=f"omnigibson:{og.__version__}",
            controller_fingerprint=self.controller_manifest.fingerprint,
            components=(evaluator_component, runtime_component),
            propagate_once=self._propagate_once,
            restore_diagnostic=self._log_restore_diagnostic,
        )
        self.session = AgenticEnvironmentSession(
            env=self.evaluator.env,
            controller_manifest=self.controller_manifest,
            snapshot_manager=manager,
        )
        self.session.pause()
        try:
            self.initial_snapshot = self.session.snapshot(
                task_name=self.task_name,
                instance_id=str(self.instance_id),
                episode_id=self.episode_id,
            )
            self.snapshots[self.initial_snapshot.snapshot_id] = self.initial_snapshot
            # Attempt zero and every retry begin from the same documented
            # post-restore propagation semantics.
            self.initial_restore_report = self.session.restore(self.initial_snapshot)
        finally:
            self.session.resume()
        if self.snapshot_dir is not None:
            persist_snapshot(self.initial_snapshot, self.snapshot_dir)
        self.snapshot_roles = {self.initial_snapshot.snapshot_id: "initial"}
        self.demo_primary_snapshot_id: str | None = None
        self.radio_pickup_preclose_snapshot_id: str | None = None
        self._initial_radio_checkpoint_state = (
            self._radio_checkpoint_state() if self.task_name == "turning_on_radio" else None
        )
        self.radio_pickup_fixture_reference_snapshot_id = radio_pickup_fixture_reference_snapshot_id
        self.radio_pickup_fixture_reference_restore_report = None
        self._radio_pickup_fixture_reference_state: dict[str, object] | None = None
        self.imported_snapshot_count = self._load_imported_snapshots()
        if self.radio_pickup_fixture_reference_snapshot_id is not None:
            self._establish_radio_pickup_fixture_reference(self.radio_pickup_fixture_reference_snapshot_id)
        self.evaluator.obs = self.evaluator._preprocess_obs(self.evaluator.env.get_obs()[0])

    @property
    def metadata(self) -> dict[str, object]:
        metadata = {
            "service": "behavior_agentic_environment",
            "protocol_version": PROTOCOL_VERSION,
            "task": self.task_name,
            "instance_id": self.instance_id,
            "episode_id": self.episode_id,
            "attempt_index": self.attempt_index,
            "initial_snapshot_id": self.initial_snapshot.snapshot_id,
            "initial_restore_report": self.initial_restore_report.to_dict(),
            "controller_manifest": self.controller_manifest.to_dict(),
            "camera_keys": {camera: f"{name}::rgb" for camera, name in ROBOT_CAMERA_NAMES["R1Pro"].items()},
            "proprio_key": "robot_r1::proprio",
            "camera_pose_key": CAMERA_POSE_KEY,
            "camera_pose_order": list(ROBOT_CAMERA_NAMES["R1Pro"]),
            "camera_intrinsics": {
                camera: intrinsics.tolist() for camera, intrinsics in CAMERA_INTRINSICS["R1Pro"].items()
            },
            "camera_native_sizes": {
                "head": [720, 720],
                "left_wrist": [480, 480],
                "right_wrist": [480, 480],
            },
            "proprio_fields": {
                name: [int(index.start), int(index.stop)] for name, index in PROPRIOCEPTION_INDICES["R1Pro"].items()
            },
            "information_arm": self.information_arm,
            "available_modalities": ["rgb", "proprio"],
            "durable_snapshots": self.snapshot_dir is not None,
            "imported_snapshot_count": self.imported_snapshot_count,
            "checkpoint_roles": sorted(CHECKPOINT_ROLES),
            "demo_primary_integrity_tasks": ["turning_on_radio"],
            "radio_pickup_fixture_reference_snapshot_id": self.radio_pickup_fixture_reference_snapshot_id,
            "radio_pickup_fixture_reference_restore_report": (
                self.radio_pickup_fixture_reference_restore_report.to_dict()
                if self.radio_pickup_fixture_reference_restore_report is not None
                else None
            ),
        }
        if self.enable_semantic_geometry_diagnostic:
            metadata["semantic_geometry_diagnostic"] = {
                "profile": "radio_generation_015_semantic_geometry",
                "privileged": True,
                "read_only_query": "inspect_navigation_geometry",
            }
        if self.information_arm == "vision_rgb_depth":
            metadata["depth_keys"] = {
                camera: f"{name}::depth_linear" for camera, name in ROBOT_CAMERA_NAMES["R1Pro"].items()
            }
            metadata["jaw_corridor_calibration"] = JAW_CORRIDOR_CALIBRATION
            metadata["available_modalities"].append("depth_linear")
        return metadata

    def plan_eef_position(
        self,
        *,
        arm: str,
        target_position: object,
        max_target_delta_m: float = 0.12,
        max_joint_delta_rad: float = 0.20,
    ) -> dict[str, object]:
        """Return one bounded local IK target without executing or changing state."""

        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'.")
        if not 0 < max_target_delta_m <= 0.20:
            raise ValueError("max_target_delta_m must be within (0, 0.20].")
        if not 0 < max_joint_delta_rad <= 0.35:
            raise ValueError("max_joint_delta_rad must be within (0, 0.35].")
        controller = self.robot.controllers[f"arm_{arm}"]
        control_dict = self.robot.get_control_dict()
        dof_indices = np.asarray(controller.dof_idx, dtype=np.int64)
        lower_all, upper_all = controller._control_limits[controller.control_type]
        result = bounded_position_ik_step(
            joint_positions=torch_to_numpy(control_dict["joint_position"])[dof_indices],
            jacobian=torch_to_numpy(control_dict[f"eef_{arm}_jacobian_relative"][:, dof_indices]),
            eef_position=torch_to_numpy(control_dict[f"eef_{arm}_pos_relative"]),
            target_position=np.asarray(target_position, dtype=np.float32),
            joint_lower=torch_to_numpy(lower_all)[dof_indices],
            joint_upper=torch_to_numpy(upper_all)[dof_indices],
            max_target_delta_m=max_target_delta_m,
            max_joint_delta_rad=max_joint_delta_rad,
        )
        return {
            "observation_id": self._observation_id(torch_to_numpy(self.evaluator.obs)),
            "env_step": int(self.evaluator.env._current_step),
            "arm": arm,
            "controller_segment": f"arm_{arm}",
            "current_eef_position_robot_m": _jsonable(control_dict[f"eef_{arm}_pos_relative"]),
            "requested_target_position_robot_m": _jsonable(np.asarray(target_position, dtype=np.float32)),
            "orientation_policy": "keep_current_orientation_local_linearization",
            "collision_checked": False,
            **_jsonable(result),
        }

    def plan_eef_pose_delta(
        self,
        *,
        arm: str,
        target_position: object,
        orientation_delta_eef_axis_angle_rad: object,
        max_target_delta_m: float = 0.03,
        max_orientation_delta_rad: float = 0.17,
        max_joint_delta_rad: float = 0.12,
    ) -> dict[str, object]:
        """Return one bounded local pose-delta IK target without executing."""

        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'.")
        if not np.isfinite(max_target_delta_m) or not 0 < max_target_delta_m <= 0.05:
            raise ValueError("max_target_delta_m must be within (0, 0.05].")
        if not np.isfinite(max_orientation_delta_rad) or not 0 < max_orientation_delta_rad <= 0.35:
            raise ValueError("max_orientation_delta_rad must be within (0, 0.35].")
        if not np.isfinite(max_joint_delta_rad) or not 0 < max_joint_delta_rad <= 0.35:
            raise ValueError("max_joint_delta_rad must be within (0, 0.35].")
        controller = self.robot.controllers[f"arm_{arm}"]
        control_dict = self.robot.get_control_dict()
        dof_indices = np.asarray(controller.dof_idx, dtype=np.int64)
        lower_all, upper_all = controller._control_limits[controller.control_type]
        result = bounded_pose_delta_ik_step(
            joint_positions=torch_to_numpy(control_dict["joint_position"])[dof_indices],
            jacobian=torch_to_numpy(control_dict[f"eef_{arm}_jacobian_relative"][:, dof_indices]),
            eef_position=torch_to_numpy(control_dict[f"eef_{arm}_pos_relative"]),
            eef_quaternion_xyzw=torch_to_numpy(control_dict[f"eef_{arm}_quat_relative"]),
            target_position=np.asarray(target_position, dtype=np.float32),
            orientation_delta_eef_axis_angle_rad=np.asarray(orientation_delta_eef_axis_angle_rad, dtype=np.float32),
            joint_lower=torch_to_numpy(lower_all)[dof_indices],
            joint_upper=torch_to_numpy(upper_all)[dof_indices],
            max_target_delta_m=max_target_delta_m,
            max_orientation_delta_rad=max_orientation_delta_rad,
            max_joint_delta_rad=max_joint_delta_rad,
        )
        result.pop("target_eef_quaternion_robot_xyzw")
        return {
            "arm": arm,
            "controller_segment": f"arm_{arm}",
            **_jsonable(result),
            "frame_policy": {
                "orientation_policy": "bounded_eef_frame_pose_delta",
                "quaternion_order": "xyzw",
                "composition": "q_target_robot = q_current_robot * q_delta_eef",
                "jacobian_angular_error_frame": "robot_base",
                "quaternion_sign_policy": "normalized_shortest_arc_nonnegative_w",
            },
            "collision_checked": False,
        }

    def observe(self) -> dict[str, object]:
        observation = torch_to_numpy(self.evaluator.obs)
        reference_action = normalize_runtime_reference_action(_no_op_action(self.robot), self.controller_manifest)
        return {
            "observation_id": self._observation_id(observation),
            "env_step": int(self.evaluator.env._current_step),
            "policy_observation": observation,
            "reference_action": reference_action,
            "reward": float(self.last_reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "executed_action": np.asarray(self.evaluator.robot_action, dtype=np.float32).reshape(-1)
            if np.asarray(self.evaluator.robot_action).size == self.controller_manifest.action_dim
            else reference_action,
            "attempt_index": self.attempt_index,
        }

    def inspect_navigation_geometry(self) -> dict[str, object]:
        """Return observation-bound privileged geometry without changing runtime state."""

        if not self.enable_semantic_geometry_diagnostic:
            raise PermissionError("Semantic navigation geometry is disabled outside the Generation-015 diagnostic.")
        if self.task_name != "turning_on_radio":
            raise RuntimeError("Semantic navigation geometry is available only for turning_on_radio.")
        if self._navigation_geometry_baseline is None or self._navigation_geometry_baseline_snapshot_id is None:
            raise RuntimeError("Semantic navigation geometry requires an explicit immutable fixture restore first.")
        observation_id = self._observation_id(torch_to_numpy(self.evaluator.obs))
        env_step = int(self.evaluator.env._current_step)
        runtime_state = _jsonable(
            {
                "last_reward": self.last_reward,
                "last_info": self.last_info,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "finished": self.finished,
                "metrics_finished": self._metrics_finished,
                "last_metrics": self._last_metrics,
            }
        )
        backend = getattr(self, "_semantic_geometry_backend", None)
        backend_error = getattr(self, "_semantic_geometry_backend_error", None)
        backend_capabilities = backend.capabilities() if backend is not None else None
        if backend_capabilities is None and backend_error is not None:
            backend_capabilities = {
                name: {"status": "unavailable", "reason": backend_error}
                for name in (
                    "hypothetical_base_pose_whole_arm_ik",
                    "arm_trajectory_collision",
                    "arm_table_clearance",
                )
            }
        current = capture_navigation_geometry(
            self.evaluator.env,
            self.robot,
            backend_capabilities=backend_capabilities,
        )
        if int(self.evaluator.env._current_step) != env_step:
            raise RuntimeError("Read-only semantic geometry query unexpectedly advanced the environment.")
        if runtime_state != _jsonable(
            {
                "last_reward": self.last_reward,
                "last_info": self.last_info,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "finished": self.finished,
                "metrics_finished": self._metrics_finished,
                "last_metrics": self._last_metrics,
            }
        ):
            raise RuntimeError("Read-only semantic geometry query unexpectedly changed evaluator state.")
        return {
            "schema_version": 1,
            "diagnostic_profile": "radio_generation_015_semantic_geometry",
            "privileged": True,
            "read_only": True,
            "observation_id": observation_id,
            "env_step": env_step,
            "baseline_snapshot_id": self._navigation_geometry_baseline_snapshot_id,
            "geometry": current,
            "immutable_baseline_deltas": navigation_geometry_deltas(
                self._navigation_geometry_baseline,
                current,
            ),
        }

    def evaluate_semantic_pickup_corridor(
        self,
        *,
        observation_id: str,
        env_step: int,
        candidate_base_pose: Mapping[str, object],
        target_poses: Mapping[str, Mapping[str, object]],
        joint_provenance: Mapping[str, object] | None = None,
        retained_lift_gate: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Run privileged CuRobo corridor queries without changing simulator state."""

        if not self.enable_semantic_geometry_diagnostic:
            raise PermissionError("Semantic navigation geometry is disabled outside the Generation-015 diagnostic.")
        if self._semantic_geometry_backend is None:
            raise RuntimeError(
                "Semantic geometry backend is unavailable: "
                f"{self._semantic_geometry_backend_error or 'not initialized'}."
            )
        current_observation_id = self._observation_id(torch_to_numpy(self.evaluator.obs))
        current_env_step = int(self.evaluator.env._current_step)
        if observation_id != current_observation_id or int(env_step) != current_env_step:
            raise RuntimeError("Semantic corridor query is not bound to the current observation and environment step.")
        before_q = self.robot.get_joint_positions().detach().clone()
        before_geometry = capture_navigation_geometry(
            self.evaluator.env,
            self.robot,
            backend_capabilities=self._semantic_geometry_backend.capabilities(),
        )
        result = self._semantic_geometry_backend.evaluate_corridor(
            candidate_base_pose=candidate_base_pose,
            target_poses=target_poses,
            joint_provenance=joint_provenance,
            retained_lift_gate=retained_lift_gate,
        )
        after_q = self.robot.get_joint_positions().detach().clone()
        after_geometry = capture_navigation_geometry(
            self.evaluator.env,
            self.robot,
            backend_capabilities=self._semantic_geometry_backend.capabilities(),
        )
        if current_env_step != int(self.evaluator.env._current_step):
            raise RuntimeError("Semantic corridor query unexpectedly advanced the environment.")
        if not th.equal(before_q, after_q):
            raise RuntimeError("Semantic corridor query unexpectedly changed simulator joint positions.")
        deltas = navigation_geometry_deltas(before_geometry, after_geometry)
        if (
            any(
                deltas[name][metric] != 0.0
                for name in ("robot", "radio", "support_table")
                for metric in ("translation_norm_m", "orientation_angle_rad")
            )
            or deltas["radio_support_changed"]
            or deltas["contact_pairs_changed"]
        ):
            raise RuntimeError("Semantic corridor query unexpectedly changed simulator geometry or contacts.")
        result.update(
            {
                "privileged": True,
                "read_only": True,
                "observation_id": current_observation_id,
                "env_step": current_env_step,
                "candidate_base_pose": _jsonable(candidate_base_pose),
                "target_pose_digest": hashlib.sha256(
                    json.dumps(_jsonable(target_poses), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        )
        return result

    def step(self, action: object) -> dict[str, object]:
        if self.finished:
            raise RuntimeError("Cannot step a finalized agentic environment.")
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a terminal agentic environment.")
        result = self.session.step(action)
        self.evaluator.robot_action = th.from_numpy(result.executed_action.copy())
        self.evaluator._update_task_progress(result.info.get("task_progress"), source=f"step {result.env_step}")
        self.evaluator.obs = self.evaluator._preprocess_obs(result.observation)
        for metric in self.evaluator.metrics:
            metric.step_callback(self.evaluator.env)
        self.last_reward = result.reward
        self.last_info = result.info
        self.terminated = result.terminated
        self.truncated = result.truncated
        if self.terminated or self.truncated:
            self.evaluator.n_trials += 1
            if bool(result.info.get("done", {}).get("success", False)):
                self.evaluator.n_success_trials += 1
        payload = self.observe()
        payload["executed_action"] = result.executed_action
        return payload

    def snapshot(
        self,
        *,
        parent_snapshot_id: str | None = None,
        checkpoint_role: str = "general",
    ) -> dict[str, object]:
        if checkpoint_role not in CHECKPOINT_ROLES:
            raise ValueError(f"Unsupported checkpoint role: {checkpoint_role!r}.")
        if checkpoint_role == "demo_primary":
            if self.demo_primary_snapshot_id is not None:
                raise RuntimeError("The immutable demo_primary checkpoint is already established for this runtime.")
            self._validate_demo_primary_checkpoint()
        if checkpoint_role == RADIO_PICKUP_PRECLOSE_ROLE:
            if self.radio_pickup_preclose_snapshot_id is not None:
                raise RuntimeError(f"The immutable {RADIO_PICKUP_PRECLOSE_ROLE} checkpoint is already established.")
            self._validate_radio_pickup_preclose_checkpoint()
        self.session.pause()
        try:
            snapshot = self.session.snapshot(
                task_name=self.task_name,
                instance_id=str(self.instance_id),
                episode_id=self.episode_id,
                parent_snapshot_id=parent_snapshot_id,
            )
            if self.snapshot_dir is not None:
                persist_snapshot(snapshot, self.snapshot_dir)
            self.snapshots[snapshot.snapshot_id] = snapshot
            self.snapshot_roles[snapshot.snapshot_id] = checkpoint_role
            if checkpoint_role == "demo_primary":
                self.demo_primary_snapshot_id = snapshot.snapshot_id
            if checkpoint_role == RADIO_PICKUP_PRECLOSE_ROLE:
                self.radio_pickup_preclose_snapshot_id = snapshot.snapshot_id
        finally:
            self.session.resume()
        return {
            "snapshot_id": snapshot.snapshot_id,
            "metadata": snapshot.metadata.to_dict(),
            "checkpoint_role": checkpoint_role,
        }

    def restore(self, snapshot_id: str) -> dict[str, object]:
        try:
            snapshot = self.snapshots[snapshot_id]
        except KeyError as exc:
            raise ValueError(f"Unknown in-process environment snapshot: {snapshot_id}.") from exc
        self.session.pause()
        try:
            report = self.session.restore(snapshot)
        finally:
            self.session.resume()
        self._capture_navigation_geometry_baseline(snapshot_id)
        return {"snapshot_id": snapshot_id, "restore_report": report.to_dict(), **self.observe()}

    def finish(self) -> dict[str, object]:
        if self._last_metrics is not None:
            return self._last_metrics
        if not self._metrics_finished:
            for metric in self.evaluator.metrics:
                metric.end_callback(self.evaluator.env)
            self._metrics_finished = True
        metrics: dict[str, object] = {}
        for metric in self.evaluator.metrics:
            metrics.update(metric.gather_results())
        metrics["rollout"] = self.evaluator.finalize_rollout_metadata()
        metrics["terminated"] = self.terminated
        metrics["truncated"] = self.truncated
        metrics["env_step"] = int(self.evaluator.env._current_step)
        metrics["success"] = bool(self.last_info.get("done", {}).get("success", False))
        self.finished = True
        self._last_metrics = _jsonable(metrics)
        return self._last_metrics

    def reset_attempt(self, snapshot_id: str | None = None) -> dict[str, object]:
        if not self.finished:
            raise RuntimeError("Current attempt must be scored before reset.")
        if snapshot_id is None:
            snapshot = self.initial_snapshot
        else:
            try:
                snapshot = self.snapshots[snapshot_id]
            except KeyError as exc:
                raise ValueError(f"Unknown in-process environment snapshot: {snapshot_id}.") from exc
        self.session.pause()
        try:
            report = self.session.restore(snapshot)
        finally:
            self.session.resume()
        self._capture_navigation_geometry_baseline(snapshot.snapshot_id)
        self.attempt_index += 1
        self.evaluator.obs = self.evaluator._preprocess_obs(self.evaluator.env.get_obs()[0])
        return {
            "reset": True,
            "attempt_index": self.attempt_index,
            "initial_snapshot_id": self.initial_snapshot.snapshot_id,
            "restored_snapshot_id": snapshot.snapshot_id,
            "restore_report": report.to_dict(),
            **self.observe(),
        }

    def close(self) -> None:
        self.session.close()
        self.evaluator.env.close()

    def _capture_navigation_geometry_baseline(self, snapshot_id: str) -> None:
        if not self.enable_semantic_geometry_diagnostic or self.task_name != "turning_on_radio":
            return
        self._navigation_geometry_baseline = capture_navigation_geometry(self.evaluator.env, self.robot)
        self._navigation_geometry_baseline_snapshot_id = snapshot_id

    def _load_imported_snapshots(self) -> int:
        imported = 0
        for directory in self.snapshot_import_dirs:
            for manifest_path in sorted(directory.glob("*.json")):
                snapshot = load_persisted_snapshot(manifest_path)
                self.session.snapshot_manager._validate_compatibility(snapshot)
                if snapshot.metadata.task_name != self.task_name:
                    raise ValueError(
                        f"Imported snapshot task {snapshot.metadata.task_name!r} does not match {self.task_name!r}."
                    )
                if snapshot.metadata.instance_id != str(self.instance_id):
                    raise ValueError(
                        f"Imported snapshot instance {snapshot.metadata.instance_id!r} does not match {self.instance_id}."
                    )
                if snapshot.snapshot_id not in self.snapshots:
                    self.snapshots[snapshot.snapshot_id] = snapshot
                    self.snapshot_roles[snapshot.snapshot_id] = "imported"
                    imported += 1
        return imported

    def _radio_checkpoint_state(self) -> dict[str, object]:
        env = self.evaluator.env
        radio = _resolve_object(env, "radio_receiver.n.01_1").unwrapped
        table = _resolve_object(env, "table.n.02_1").unwrapped
        position, orientation = radio.get_position_orientation()
        return {
            "position": torch_to_numpy(position),
            "orientation": torch_to_numpy(orientation),
            "linear_velocity": torch_to_numpy(radio.get_linear_velocity()),
            "angular_velocity": torch_to_numpy(radio.get_angular_velocity()),
            "on_table": bool(radio.states[OnTop].get_value(table)),
            "grasping_left": int(self.robot.is_grasping(arm="left", candidate_obj=radio)) == 1,
            "grasping_right": int(self.robot.is_grasping(arm="right", candidate_obj=radio)) == 1,
        }

    def _radio_pickup_preclose_checkpoint_state(self) -> dict[str, object]:
        env = self.evaluator.env
        radio = _resolve_object(env, "radio_receiver.n.01_1").unwrapped
        table = _resolve_object(env, "table.n.02_1").unwrapped
        table_position, table_orientation = table.get_position_orientation()
        base_position, base_orientation = self.robot.get_position_orientation()
        contact_bodies = radio.states[ContactBodies].get_value()
        table_links = getattr(table, "links", {})
        link_values = table_links.values() if isinstance(table_links, Mapping) else table_links
        progress = _normalize_task_progress(CHALLENGE_TASKS_PROGRESS_APPROXIMATION[self.task_name](env)) or {}
        reference_action = normalize_runtime_reference_action(_no_op_action(self.robot), self.controller_manifest)
        right_gripper = next(
            segment for segment in self.controller_manifest.segments if segment.name == "gripper_right"
        )
        return {
            **self._radio_checkpoint_state(),
            "table_position": torch_to_numpy(table_position),
            "table_orientation": torch_to_numpy(table_orientation),
            "table_linear_velocity": torch_to_numpy(table.get_linear_velocity()),
            "table_angular_velocity": torch_to_numpy(table.get_angular_velocity()),
            "base_position": torch_to_numpy(base_position),
            "base_orientation": torch_to_numpy(base_orientation),
            "base_linear_velocity": torch_to_numpy(self.robot.get_linear_velocity()),
            "base_angular_velocity": torch_to_numpy(self.robot.get_angular_velocity()),
            "contact_body_paths": sorted(str(getattr(body, "prim_path", body)) for body in contact_bodies),
            "table_link_paths": sorted(str(getattr(link, "prim_path", link)) for link in link_values),
            "assisted_grasp_attachments": {
                arm: getattr(obj, "name", None) for arm, obj in self.robot._ag_obj_in_hand.items()
            },
            "robot_near_radio": bool(progress.get("robot_near_radio", False)),
            "radio_picked_up": bool(progress.get("radio_picked_up", False)),
            "radio_on": bool(progress.get("radio_on", False)),
            "right_gripper_command": float(reference_action[right_gripper.start]),
        }

    def _establish_radio_pickup_fixture_reference(self, snapshot_id: str) -> None:
        if self.task_name != "turning_on_radio":
            raise ValueError("Radio pickup fixture references are valid only for turning_on_radio.")
        try:
            snapshot = self.snapshots[snapshot_id]
        except KeyError as exc:
            raise ValueError(f"Unknown Radio pickup fixture reference snapshot: {snapshot_id}.") from exc
        if self.snapshot_roles.get(snapshot_id) != "imported":
            raise ValueError("Radio pickup fixture reference must be an explicitly imported snapshot.")
        self.session.pause()
        try:
            report = self.session.restore(snapshot)
        finally:
            self.session.resume()
        self.evaluator.obs = self.evaluator._preprocess_obs(self.evaluator.env.get_obs()[0])
        self.radio_pickup_fixture_reference_restore_report = report
        self._radio_pickup_fixture_reference_state = self._radio_pickup_preclose_checkpoint_state()

    def _validate_radio_pickup_preclose_checkpoint(self) -> None:
        if self.task_name != "turning_on_radio" or self._radio_pickup_fixture_reference_state is None:
            raise RuntimeError(f"{RADIO_PICKUP_PRECLOSE_ROLE} requires a source-pinned imported reference snapshot.")
        candidate = self._radio_pickup_preclose_checkpoint_state()
        assessment = assess_radio_pickup_preclose_checkpoint(
            self._radio_pickup_fixture_reference_state,
            candidate,
        )
        details = {
            "created_wall_time_ns": time.time_ns(),
            "phase": "radio_pickup_preclose_checkpoint_validation",
            "env_step": int(self.evaluator.env._current_step),
            "reference_snapshot_id": self.radio_pickup_fixture_reference_snapshot_id,
            "assessment": assessment,
            "reference": _jsonable(self._radio_pickup_fixture_reference_state),
            "candidate": _jsonable(candidate),
        }
        self._append_private_diagnostic("checkpoint_integrity.jsonl", details)
        if not bool(assessment["accepted"]):
            raise RuntimeError(
                f"{RADIO_PICKUP_PRECLOSE_ROLE} checkpoint integrity failed; rebuild from the pinned seed and "
                "replay prefix before saving."
            )

    def _validate_demo_primary_checkpoint(self) -> None:
        if self.task_name != "turning_on_radio" or self._initial_radio_checkpoint_state is None:
            return
        candidate = self._radio_checkpoint_state()
        assessment = assess_radio_demo_primary_checkpoint(self._initial_radio_checkpoint_state, candidate)
        details = {
            "created_wall_time_ns": time.time_ns(),
            "phase": "demo_primary_checkpoint_validation",
            "env_step": int(self.evaluator.env._current_step),
            "assessment": assessment,
            "initial": _jsonable(self._initial_radio_checkpoint_state),
            "candidate": _jsonable(candidate),
        }
        self._append_private_diagnostic("checkpoint_integrity.jsonl", details)
        if not bool(assessment["accepted"]):
            raise RuntimeError(
                "demo_primary checkpoint integrity failed; restore the initial state and approach without contacting "
                "the Radio before saving the primary checkpoint."
            )

    def _append_private_diagnostic(self, filename: str, details: Mapping[str, object]) -> None:
        if self.snapshot_dir is None:
            return
        path = self.snapshot_dir.parent / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        with path.open("a", encoding="utf-8") as diagnostics:
            diagnostics.write(json.dumps(_jsonable(details), sort_keys=True) + "\n")

    def _log_restore_diagnostic(self, phase: str, snapshot: CompositeSnapshot) -> None:
        if self.task_name != "turning_on_radio":
            return
        env = self.evaluator.env
        radio = _resolve_object(env, "radio_receiver.n.01_1").unwrapped
        table = _resolve_object(env, "table.n.02_1").unwrapped
        position, orientation = radio.get_position_orientation()
        contact_bodies = radio.states[ContactBodies].get_value()
        details = {
            "created_wall_time_ns": time.time_ns(),
            "phase": phase,
            "snapshot_id": snapshot.snapshot_id,
            "env_step": int(env._current_step),
            "radio_position": _jsonable(position),
            "radio_orientation": _jsonable(orientation),
            "radio_linear_velocity": _jsonable(radio.get_linear_velocity()),
            "radio_angular_velocity": _jsonable(radio.get_angular_velocity()),
            "radio_on_table": bool(radio.states[OnTop].get_value(table)),
            "grasping_left": int(self.robot.is_grasping(arm="left", candidate_obj=radio)),
            "grasping_right": int(self.robot.is_grasping(arm="right", candidate_obj=radio)),
            "contact_body_paths": sorted(str(getattr(body, "prim_path", body)) for body in contact_bodies),
            "assisted_grasp_attachments": {
                arm: getattr(obj, "name", None) for arm, obj in self.robot._ag_obj_in_hand.items()
            },
        }
        if self.restore_diagnostic_path is not None:
            self.restore_diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.restore_diagnostic_path.exists():
                descriptor = os.open(
                    self.restore_diagnostic_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                os.close(descriptor)
            with self.restore_diagnostic_path.open("a", encoding="utf-8") as diagnostics:
                diagnostics.write(json.dumps(details, sort_keys=True) + "\n")
        logger.info("agentic_radio_restore_diagnostic %s", json.dumps(details, sort_keys=True))

    def _propagate_once(self) -> dict[str, float]:
        og.sim.step()
        # A loaded simulator state reaches the camera render products one render
        # after the physics propagation. Without this refresh, reset/restore can
        # return current proprioception with RGB cached from the prior attempt.
        og.sim.render()
        return {
            "sim_step_dt": float(og.sim.get_sim_step_dt()),
            "post_restore_render_steps": 1.0,
        }

    def _observation_id(self, observation: Mapping[str, object]) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(int(self.evaluator.env._current_step)).encode())
        sensor_keys = list(self.metadata["camera_keys"].values())
        sensor_keys.extend(self.metadata.get("depth_keys", {}).values())
        sensor_keys.append(self.metadata["proprio_key"])
        sensor_keys.append(self.metadata["camera_pose_key"])
        for key in sensor_keys:
            value = np.asarray(observation[str(key)])
            hasher.update(str(value.dtype).encode())
            hasher.update(repr(value.shape).encode())
            hasher.update(value.tobytes())
        return f"obs-{int(self.evaluator.env._current_step)}-{hasher.hexdigest()[:16]}"


class RequestBridgeError(RuntimeError):
    """Base class for failures that close the request bridge."""


class RequestQueueFullError(RequestBridgeError):
    pass


class RequestTimeoutError(RequestBridgeError):
    pass


class DuplicateCompletionError(RequestBridgeError):
    pass


class RequestBridgeShutdownError(RequestBridgeError):
    pass


class NetworkThreadError(RequestBridgeError):
    pass


class NetworkStartupError(NetworkThreadError):
    pass


@dataclass(frozen=True, eq=False)
class RequestEnvelope:
    """Immutable request bytes paired with one thread-safe response channel."""

    payload: bytes
    response: Future[bytes] = field(default_factory=Future, repr=False, compare=False)
    _completion_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def complete(self, response: bytes) -> None:
        with self._completion_lock:
            if self.response.done():
                raise DuplicateCompletionError("A runtime request response was completed more than once.")
            self.response.set_result(response)

    def fail(self, error: RequestBridgeError) -> None:
        with self._completion_lock:
            if self.response.done():
                raise DuplicateCompletionError("A runtime request response was completed more than once.")
            self.response.set_exception(error)


class RequestBridge:
    """Single-consumer bounded handoff between network and runtime threads."""

    def __init__(self, capacity: int = REQUEST_QUEUE_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("Request queue capacity must be positive.")
        self._queue: queue.Queue[RequestEnvelope] = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._pending: set[RequestEnvelope] = set()
        self._closed = threading.Event()
        self._close_reason: RequestBridgeError | None = None

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def close_reason(self) -> RequestBridgeError | None:
        with self._lock:
            return self._close_reason

    def submit(self, payload: bytes) -> RequestEnvelope:
        envelope = RequestEnvelope(payload=payload)
        queue_error = None
        with self._lock:
            if self._closed.is_set():
                raise self._close_reason or RequestBridgeShutdownError("The runtime request bridge is closed.")
            try:
                self._queue.put_nowait(envelope)
                self._pending.add(envelope)
            except queue.Full:
                queue_error = RequestQueueFullError("The runtime request queue is full.")
        if queue_error is not None:
            self.close(queue_error)
            raise queue_error
        return envelope

    def get(self, timeout: float) -> RequestEnvelope | None:
        if self._closed.is_set():
            return None
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def complete(self, envelope: RequestEnvelope, response: bytes) -> None:
        with self._lock:
            try:
                envelope.complete(response)
            except DuplicateCompletionError as exc:
                if not self._closed.is_set():
                    self._close_reason = exc
                    self._closed.set()
                    for pending_envelope in self._pending:
                        if pending_envelope is not envelope:
                            pending_envelope.fail(exc)
                    self._pending.clear()
                raise
            self._pending.discard(envelope)

    def close(self, reason: RequestBridgeError) -> None:
        with self._lock:
            if self._closed.is_set():
                return
            self._close_reason = reason
            self._closed.set()
            for envelope in self._pending:
                envelope.fail(reason)
            self._pending.clear()


def _error_payload(exc: BaseException) -> dict[str, object]:
    return {
        "ok": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class _WebsocketNetworkWorker:
    """Owns websocket and asyncio state without retaining the simulator runtime."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        metadata_frame: bytes,
        bridge: RequestBridge,
        request_timeout: float,
    ) -> None:
        self.host = host
        self.port = port
        self.bridge = bridge
        self.request_timeout = request_timeout
        self.bound_port: int | None = None
        self.thread_ident: int | None = None
        self._thread = threading.Thread(target=self._thread_main, name="agentic-websocket-network", daemon=False)
        self._ready = threading.Event()
        self._expected_shutdown = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._metadata_lock = threading.Lock()
        self._metadata_frame = metadata_frame
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._client_lock: asyncio.Lock | None = None

    @property
    def failure(self) -> BaseException | None:
        with self._failure_lock:
            return self._failure

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def update_metadata(self, metadata_frame: bytes) -> None:
        with self._metadata_lock:
            self._metadata_frame = metadata_frame

    def _current_metadata(self) -> bytes:
        with self._metadata_lock:
            return self._metadata_frame

    def start(self, timeout: float = NETWORK_STARTUP_TIMEOUT_SECONDS) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            error = NetworkStartupError("Timed out waiting for the websocket network thread to bind.")
            self._record_failure(error)
            self.stop()
            self.join()
            raise error
        if self.failure is not None or not self._thread.is_alive() or self.bound_port is None:
            raise NetworkStartupError("The websocket network thread failed during startup.") from self.failure

    def stop(self) -> None:
        self._expected_shutdown.set()
        loop = self._loop
        shutdown_event = self._shutdown_event
        if loop is not None and shutdown_event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(shutdown_event.set)

    def join(self, timeout: float = NETWORK_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        if self._thread.ident is None or threading.get_ident() == self._thread.ident:
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise NetworkThreadError("The websocket network thread did not stop cleanly.")

    def _record_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = exc
        reason = exc if isinstance(exc, RequestBridgeError) else NetworkThreadError(str(exc))
        self.bridge.close(reason)
        loop = self._loop
        shutdown_event = self._shutdown_event
        if loop is not None and shutdown_event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(shutdown_event.set)

    def _thread_main(self) -> None:
        self.thread_ident = threading.get_ident()
        try:
            asyncio.run(self._run())
            if not self._expected_shutdown.is_set() and self.failure is None:
                self._record_failure(NetworkThreadError("The websocket network thread stopped unexpectedly."))
        except BaseException as exc:
            self._record_failure(exc)
        finally:
            self._ready.set()

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()
        self._client_lock = asyncio.Lock()
        if self._expected_shutdown.is_set():
            self._shutdown_event.set()
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            sockets = server.sockets
            if not sockets:
                raise NetworkStartupError("The websocket server did not create a listening socket.")
            self.bound_port = int(sockets[0].getsockname()[1])
            logger.info("Agentic BEHAVIOR environment listening on %s:%s", self.host, self.bound_port)
            self._ready.set()
            await self._shutdown_event.wait()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        if self._client_lock is None:
            raise NetworkThreadError("The websocket client lock was not initialized.")
        if self._client_lock.locked():
            await websocket.close(code=1013, reason="Agentic environment already has an active controller.")
            return
        async with self._client_lock:
            packer = Packer()
            await websocket.send(self._current_metadata())
            while True:
                try:
                    message = await websocket.recv()
                except websockets.ConnectionClosed:
                    break
                try:
                    payload = bytes(message)
                    request = unpackb(payload, strict_map_key=False)
                    if not isinstance(request, dict):
                        raise ValueError("Agentic environment request must be an object.")
                    envelope = self.bridge.submit(payload)
                except RequestBridgeError as exc:
                    await self._send_fatal_error(websocket, packer, exc)
                    break
                except Exception as exc:
                    logger.error("Agentic environment request failed:\n%s", traceback.format_exc())
                    try:
                        await websocket.send(packer.pack(_error_payload(exc)))
                    except websockets.ConnectionClosed:
                        break
                    continue

                try:
                    response_waiter = asyncio.wrap_future(envelope.response)
                    response_waiter.add_done_callback(_consume_future_exception)
                    response = await asyncio.wait_for(
                        asyncio.shield(response_waiter),
                        timeout=self.request_timeout,
                    )
                    await websocket.send(response)
                except asyncio.TimeoutError:
                    error = RequestTimeoutError(
                        f"Runtime request did not complete within {self.request_timeout:g} seconds."
                    )
                    self.bridge.close(error)
                    await self._send_fatal_error(websocket, packer, error)
                    break
                except RequestBridgeError as exc:
                    await self._send_fatal_error(websocket, packer, exc)
                    break
                except websockets.ConnectionClosed:
                    break

    async def _send_fatal_error(
        self,
        websocket: websocket_server.ServerConnection,
        packer: Any,
        exc: RequestBridgeError,
    ) -> None:
        if not isinstance(exc, RequestBridgeShutdownError):
            self._record_failure(exc)
        try:
            await websocket.send(packer.pack(_error_payload(exc)))
            await websocket.close(code=1011, reason="Agentic runtime bridge closed.")
        except websockets.ConnectionClosed:
            pass


class AgenticEnvironmentWebsocketServer:
    def __init__(
        self,
        runtime: AgenticEvaluatorRuntime,
        *,
        host: str,
        port: int,
        enable_restore_failure_injection: bool = False,
        request_queue_capacity: int = REQUEST_QUEUE_CAPACITY,
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The initial agentic environment service must bind only to loopback.")
        if request_timeout <= 0:
            raise ValueError("Request timeout must be positive.")
        self.runtime = runtime
        self.host = host
        self.port = port
        self.enable_restore_failure_injection = enable_restore_failure_injection
        self._bridge = RequestBridge(request_queue_capacity)
        self._response_packer = Packer()
        self._network = _WebsocketNetworkWorker(
            host=host,
            port=port,
            metadata_frame=self._response_packer.pack(runtime.metadata),
            bridge=self._bridge,
            request_timeout=request_timeout,
        )
        self._shutdown_lock = threading.Lock()
        self._shutdown = False

    @property
    def bound_port(self) -> int | None:
        return self._network.bound_port

    @property
    def network_thread_ident(self) -> int | None:
        return self._network.thread_ident

    def serve_forever(self, on_ready: Callable[[], None] | None = None) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("The agentic runtime dispatch loop must run on the process main thread.")
        with self._shutdown_lock:
            if self._shutdown:
                raise RequestBridgeShutdownError("The runtime request bridge is already shut down.")
        try:
            self._network.start()
            if on_ready is not None:
                on_ready()
            self._dispatch_forever()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            first_shutdown = not self._shutdown
            self._shutdown = True
        if first_shutdown:
            self._bridge.close(RequestBridgeShutdownError("The runtime request bridge is shutting down."))
        self._network.stop()
        self._network.join()

    def _dispatch_forever(self) -> None:
        while not self._bridge.closed:
            failure = self._network.failure
            if failure is not None:
                raise NetworkThreadError("The websocket network thread failed.") from failure
            if not self._network.is_alive:
                raise NetworkThreadError("The websocket network thread stopped unexpectedly.")
            envelope = self._bridge.get(DISPATCH_POLL_SECONDS)
            if envelope is None:
                continue
            self._dispatch_envelope(envelope)
        reason = self._bridge.close_reason
        if reason is not None and not isinstance(reason, RequestBridgeShutdownError):
            raise NetworkThreadError("The websocket runtime bridge failed closed.") from reason

    def _dispatch_envelope(self, envelope: RequestEnvelope) -> None:
        try:
            request = unpackb(envelope.payload, strict_map_key=False)
            if not isinstance(request, dict):
                raise ValueError("Agentic environment request must be an object.")
            result = self._dispatch(request)
            response = self._response_packer.pack({"ok": True, "result": result})
        except Exception as exc:
            logger.error("Agentic environment request failed:\n%s", traceback.format_exc())
            response = self._response_packer.pack(_error_payload(exc))
        try:
            self._network.update_metadata(self._response_packer.pack(self.runtime.metadata))
        except Exception as exc:
            error = RequestBridgeError(f"Failed to refresh runtime metadata: {exc}")
            self._bridge.close(error)
            raise NetworkThreadError("The websocket runtime bridge failed closed on metadata refresh.") from exc
        try:
            self._bridge.complete(envelope, response)
        except DuplicateCompletionError as exc:
            self._bridge.close(exc)
            raise NetworkThreadError("The websocket runtime bridge failed closed on duplicate completion.") from exc

    def _dispatch(self, request: Mapping[str, object]) -> dict[str, object]:
        operation = request.get("operation")
        if operation == "observe":
            return self.runtime.observe()
        if operation == "inspect_navigation_geometry":
            return self.runtime.inspect_navigation_geometry()
        if operation == "evaluate_semantic_pickup_corridor":
            candidate_base_pose = request.get("candidate_base_pose")
            target_poses = request.get("target_poses")
            if not isinstance(candidate_base_pose, Mapping) or not isinstance(target_poses, Mapping):
                raise ValueError("Semantic corridor query requires candidate_base_pose and target_poses objects.")
            return self.runtime.evaluate_semantic_pickup_corridor(
                observation_id=str(request.get("observation_id") or ""),
                env_step=int(request.get("env_step", -1)),
                candidate_base_pose=candidate_base_pose,
                target_poses=target_poses,
                joint_provenance=request.get("joint_provenance")
                if isinstance(request.get("joint_provenance"), Mapping)
                else None,
                retained_lift_gate=request.get("retained_lift_gate")
                if isinstance(request.get("retained_lift_gate"), Mapping)
                else None,
            )
        if operation == "step":
            return self.runtime.step(request.get("action"))
        if operation == "snapshot":
            parent = request.get("parent_snapshot_id")
            return self.runtime.snapshot(
                parent_snapshot_id=str(parent) if parent else None,
                checkpoint_role=str(request.get("checkpoint_role") or "general"),
            )
        if operation == "restore":
            return self.runtime.restore(str(request.get("snapshot_id")))
        if operation == "finish":
            return self.runtime.finish()
        if operation == "reset_attempt":
            snapshot_id = request.get("snapshot_id")
            return self.runtime.reset_attempt(str(snapshot_id) if snapshot_id else None)
        if operation == "plan_eef_position":
            return self.runtime.plan_eef_position(
                arm=str(request.get("arm")),
                target_position=request.get("target_position"),
                max_target_delta_m=float(request.get("max_target_delta_m", 0.12)),
                max_joint_delta_rad=float(request.get("max_joint_delta_rad", 0.20)),
            )
        if operation == "plan_eef_pose_delta":
            return self.runtime.plan_eef_pose_delta(
                arm=str(request.get("arm")),
                target_position=request.get("target_position"),
                orientation_delta_eef_axis_angle_rad=request.get("orientation_delta_eef_axis_angle_rad"),
                max_target_delta_m=float(request.get("max_target_delta_m", 0.03)),
                max_orientation_delta_rad=float(request.get("max_orientation_delta_rad", 0.17)),
                max_joint_delta_rad=float(request.get("max_joint_delta_rad", 0.12)),
            )
        if operation == "inject_next_restore_failure":
            if not self.enable_restore_failure_injection:
                raise PermissionError("Restore failure injection is disabled for this server.")
            self.runtime.session.snapshot_manager.inject_restore_failure_once(
                str(request.get("reason") or "real websocket transaction smoke")
            )
            return {"injected": True}
        raise ValueError(f"Unsupported agentic environment operation: {operation!r}.")


def _health_check(connection: websocket_server.ServerConnection, request: websocket_server.Request) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _compose_config(args: argparse.Namespace) -> object:
    register_omegaconf_resolvers()
    config_dir = f"{Path(getsourcefile(Evaluator)).parents[0]}/configs"
    overrides = [
        "policy=local",
        f"task.name={args.task}",
        f"log_path={args.log_path}",
        "check_task_progress=true",
        "write_video=false",
        "save_rollout=false",
        "perturb_pose=false",
        (
            "env_wrapper._target_=omnigibson.learning.agentic.rgb_depth_wrapper.AgenticRGBDepthWrapper"
            if args.information_arm == "vision_rgb_depth"
            else "env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper"
        ),
    ]
    if args.max_steps is not None:
        overrides.append(f"max_steps={args.max_steps}")
    overrides.extend(args.hydra_override)
    with hydra.initialize_config_dir(config_dir, version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=overrides)
    OmegaConf.resolve(config)
    return config


def _raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--episode-id", default="episode-0")
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--information-arm",
        choices=("vision_rgb", "vision_rgb_depth"),
        default="vision_rgb",
    )
    parser.add_argument("--hydra-override", action="append", default=[])
    parser.add_argument("--startup-report", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--import-snapshot-dir", type=Path, action="append", default=[])
    parser.add_argument("--radio-pickup-fixture-reference-snapshot-id")
    parser.add_argument("--enable-semantic-geometry-diagnostic", action="store_true")
    parser.add_argument("--enable-restore-failure-injection", action="store_true")
    args = parser.parse_args()
    args.log_path.mkdir(parents=True, exist_ok=False)
    startup_report = args.startup_report or args.log_path.parent / "agentic_environment_startup.json"

    config = _compose_config(args)
    gm.HEADLESS = bool(config.headless)
    runtime = None
    server = None
    exit_code = 0
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        runtime = AgenticEvaluatorRuntime(
            config,
            instance_id=args.instance_id,
            episode_id=args.episode_id,
            information_arm=args.information_arm,
            snapshot_dir=args.snapshot_dir or args.log_path.parent / "snapshots",
            snapshot_import_dirs=tuple(args.import_snapshot_dir),
            radio_pickup_fixture_reference_snapshot_id=args.radio_pickup_fixture_reference_snapshot_id,
            enable_semantic_geometry_diagnostic=args.enable_semantic_geometry_diagnostic,
        )
        server = AgenticEnvironmentWebsocketServer(
            runtime,
            host=args.host,
            port=args.port,
            enable_restore_failure_injection=args.enable_restore_failure_injection,
        )

        def write_ready_report() -> None:
            _write_json_atomic(
                startup_report,
                {
                    "status": "ready",
                    "task": args.task,
                    "instance_id": args.instance_id,
                    "host": args.host,
                    "port": args.port,
                    "information_arm": args.information_arm,
                    "available_modalities": runtime.metadata["available_modalities"],
                    "controller_manifest": runtime.controller_manifest.to_dict(),
                    "durable_snapshots": runtime.metadata["durable_snapshots"],
                    "imported_snapshot_count": runtime.metadata["imported_snapshot_count"],
                    "radio_pickup_fixture_reference_snapshot_id": runtime.metadata[
                        "radio_pickup_fixture_reference_snapshot_id"
                    ],
                },
            )

        server.serve_forever(on_ready=write_ready_report)
    except BaseException as exc:
        exit_code = 1
        report = {
            "status": "failed",
            "task": args.task,
            "instance_id": args.instance_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(startup_report, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    finally:
        if server is not None:
            server.shutdown()
        if runtime is not None:
            runtime.close()
        og.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
