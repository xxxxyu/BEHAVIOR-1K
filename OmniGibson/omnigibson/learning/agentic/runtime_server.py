"""Local websocket control plane for one real agentic BEHAVIOR episode."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import hashlib
import http
from inspect import getsourcefile
import json
import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.learning.agentic.controller_manifest import build_controller_manifest
from omnigibson.learning.agentic.controller_manifest import normalize_runtime_reference_action
from omnigibson.learning.agentic.controller_manifest import validate_r1pro_manifest
from omnigibson.learning.agentic.environment_session import AgenticEnvironmentSession
from omnigibson.learning.agentic.evaluator_state import EvaluatorSnapshotComponent
from omnigibson.learning.agentic.kinematics import bounded_position_ik_step
from omnigibson.learning.agentic.retry_checkpoint import assess_radio_demo_primary_checkpoint
from omnigibson.learning.agentic.retry_checkpoint import assess_radio_pickup_preclose_checkpoint
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
RADIO_PICKUP_PRECLOSE_ROLE = "radio_pickup_preclose_right"
CHECKPOINT_ROLES = frozenset({"general", "demo_primary", "direct_press_fallback", RADIO_PICKUP_PRECLOSE_ROLE})


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

        self.robot = self.evaluator.robot
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
        if self.information_arm == "vision_rgb_depth":
            metadata["depth_keys"] = {
                camera: f"{name}::depth_linear" for camera, name in ROBOT_CAMERA_NAMES["R1Pro"].items()
            }
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


class AgenticEnvironmentWebsocketServer:
    def __init__(
        self,
        runtime: AgenticEvaluatorRuntime,
        *,
        host: str,
        port: int,
        enable_restore_failure_injection: bool = False,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The initial agentic environment service must bind only to loopback.")
        self.runtime = runtime
        self.host = host
        self.port = port
        self.enable_restore_failure_injection = enable_restore_failure_injection
        self._client_lock = asyncio.Lock()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("Agentic BEHAVIOR environment listening on %s:%s", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        if self._client_lock.locked():
            await websocket.close(code=1013, reason="Agentic environment already has an active controller.")
            return
        async with self._client_lock:
            packer = Packer()
            await websocket.send(packer.pack(self.runtime.metadata))
            while True:
                try:
                    request = unpackb(await websocket.recv(), strict_map_key=False)
                    if not isinstance(request, dict):
                        raise ValueError("Agentic environment request must be an object.")
                    result = self._dispatch(request)
                    await websocket.send(packer.pack({"ok": True, "result": result}))
                except websockets.ConnectionClosed:
                    break
                except Exception as exc:
                    logger.error("Agentic environment request failed:\n%s", traceback.format_exc())
                    await websocket.send(
                        packer.pack(
                            {
                                "ok": False,
                                "error": {"type": type(exc).__name__, "message": str(exc)},
                            }
                        )
                    )

    def _dispatch(self, request: dict[str, object]) -> dict[str, object]:
        operation = request.get("operation")
        if operation == "observe":
            return self.runtime.observe()
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
    parser.add_argument("--enable-restore-failure-injection", action="store_true")
    args = parser.parse_args()
    args.log_path.mkdir(parents=True, exist_ok=False)
    startup_report = args.startup_report or args.log_path.parent / "agentic_environment_startup.json"

    config = _compose_config(args)
    gm.HEADLESS = bool(config.headless)
    runtime = None
    exit_code = 0
    try:
        runtime = AgenticEvaluatorRuntime(
            config,
            instance_id=args.instance_id,
            episode_id=args.episode_id,
            information_arm=args.information_arm,
            snapshot_dir=args.snapshot_dir or args.log_path.parent / "snapshots",
            snapshot_import_dirs=tuple(args.import_snapshot_dir),
            radio_pickup_fixture_reference_snapshot_id=args.radio_pickup_fixture_reference_snapshot_id,
        )
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
        AgenticEnvironmentWebsocketServer(
            runtime,
            host=args.host,
            port=args.port,
            enable_restore_failure_injection=args.enable_restore_failure_injection,
        ).serve_forever()
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
        if runtime is not None:
            runtime.close()
        og.shutdown()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
