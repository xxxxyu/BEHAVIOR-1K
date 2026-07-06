import csv
from inspect import getsourcefile
import json
import logging
import os
from pathlib import Path
import shutil
from signal import SIGINT
from signal import signal
import sys
import time
import traceback
from typing import Any

from av.container import Container
from av.stream import Stream
import cv2
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.robots.sim_robot.og_teleop_utils import augment_rooms
from gello.robots.sim_robot.og_teleop_utils import generate_robot_config
from gello.robots.sim_robot.og_teleop_utils import get_task_relevant_room_types
from gello.robots.sim_robot.og_teleop_utils import load_available_tasks
import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.pose_perturbator import PosePerturbator
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.debug_video_utils import DEBUG_VIDEO_HEAD_SIZE
from omnigibson.learning.utils.debug_video_utils import DEBUG_VIDEO_RESOLUTION
from omnigibson.learning.utils.debug_video_utils import DEBUG_VIDEO_TEXT_HEIGHT
from omnigibson.learning.utils.debug_video_utils import DEBUG_VIDEO_WIDTH
from omnigibson.learning.utils.debug_video_utils import build_debug_video_frame
from omnigibson.learning.utils.debug_video_utils import resize_debug_video_views
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.learning.utils.eval_utils import WRIST_RESOLUTION
from omnigibson.learning.utils.eval_utils import flatten_obs_dict
from omnigibson.learning.utils.eval_utils import generate_basic_environment_config
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.learning.utils.obs_utils import write_video
from omnigibson.learning.utils.task_progress_utils import CHALLENGE_TASKS_PROGRESS_APPROXIMATION
from omnigibson.learning.wrappers import TaskProgressWrapper
from omnigibson.macros import create_module_macros
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric
from omnigibson.metrics import MetricBase
from omnigibson.metrics import TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
import omnigibson.utils.transform_utils as T
import torch as th

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_EVAL_INSTANCES = 10
m.NUM_TRAIN_INSTANCES = 200


# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# rollout video
ROLLOUT_CAMERA_NAMES = [
    "head",
    "left_wrist",
    "right_wrist",
]
# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


def _normalize_task_progress(task_progress: dict | None) -> dict | None:
    if task_progress is None:
        return None

    normalized = {}
    for key, value in task_progress.items():
        if isinstance(value, th.Tensor):
            value = value.detach().cpu()
            value = value.item() if value.numel() == 1 else value.numpy().tolist()
        elif isinstance(value, np.ndarray):
            value = value.item() if value.shape == () else value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        normalized[str(key)] = value
    return normalized


def _task_progress_value_is_satisfied(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return bool(value)


def _close_video_writer(video_writer: tuple[Container, Stream] | None) -> None:
    if video_writer is None:
        return

    container, stream = video_writer
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _obs_rgb_frame(obs: dict, camera_name: str, size: int) -> np.ndarray:
    frame = obs[ROBOT_CAMERA_NAMES["R1Pro"][camera_name] + "::rgb"].numpy()[..., :3]
    return cv2.resize(frame, (size, size))


class Evaluator:
    """
    Evaluator class for running and evaluating policies for behavior task.
    This class manages the setup, execution, and evaluation of policy rollouts in OmniGibson environment,
    tracking metrics such as the number of trials, successes, and total time. It supports loading environments,
    robots, policies, and metrics, and provides methods for stepping through the environment, resetting state,
    and handling video outputs and loggings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        self.latest_task_progress = None
        self._last_logged_task_progress = None
        self.latest_inference_metadata = None
        self._latest_logged_inference_metadata = None
        self._inference_log_offset = 0
        self._inference_log_fallback_logged = False
        self._debug_video_shape_logged = False
        self._current_rollout_metadata = None
        self._current_task_progress_summary = None

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None
        self._rollout_video_writers = None

        if self.cfg.perturb_pose:
            self._pose_perturbator = PosePerturbator(logger)
            np.random.seed(self.cfg.perturb_pose_seed)

        logger.info(f"{self.cfg=}")

    def _inference_log_path(self) -> Path:
        return Path(self.cfg.log_path) / "inference_prompts.jsonl"

    def _inference_log_size(self) -> int:
        log_path = self._inference_log_path()
        try:
            return log_path.stat().st_size
        except OSError:
            return 0

    def _task_progress_fraction(self, task_progress: dict | None) -> dict[str, Any] | None:
        task_progress = _normalize_task_progress(task_progress)
        if not task_progress:
            return None

        if self._current_task_progress_summary is None:
            return None

        initial = self._current_task_progress_summary.get("initial_state") or {}
        trackable_keys = [
            key for key, initial_value in initial.items() if not _task_progress_value_is_satisfied(initial_value)
        ]
        if not trackable_keys:
            trackable_keys = list(task_progress.keys())

        satisfied_keys = [
            key
            for key in trackable_keys
            if key in task_progress and _task_progress_value_is_satisfied(task_progress[key])
        ]
        denominator = len(trackable_keys)
        fraction = len(satisfied_keys) / denominator if denominator else 0.0
        return {
            "fraction": fraction,
            "percent": fraction * 100.0,
            "satisfied": len(satisfied_keys),
            "total": denominator,
            "satisfied_keys": satisfied_keys,
            "trackable_keys": trackable_keys,
        }

    def _update_task_progress_summary(self, task_progress: dict | None) -> None:
        if self._current_task_progress_summary is None:
            return
        task_progress = _normalize_task_progress(task_progress)
        if not task_progress:
            return

        current = self._task_progress_fraction(task_progress)
        if current is None:
            return

        self._current_task_progress_summary["final_state"] = dict(task_progress)
        self._current_task_progress_summary["final_fraction"] = current["fraction"]
        self._current_task_progress_summary["final_percent"] = current["percent"]
        if current["fraction"] >= self._current_task_progress_summary["max_fraction"]:
            self._current_task_progress_summary.update(
                {
                    "max_fraction": current["fraction"],
                    "max_percent": current["percent"],
                    "max_satisfied": current["satisfied"],
                    "total_trackable": current["total"],
                    "max_satisfied_keys": current["satisfied_keys"],
                    "trackable_keys": current["trackable_keys"],
                    "max_state": dict(task_progress),
                }
            )

    def begin_rollout_metadata(self, instance_id: int, episode_id: int, rollout_paths: dict | None = None) -> None:
        initial_task_progress = _normalize_task_progress(self.latest_task_progress) or {}
        self._current_task_progress_summary = {
            "initial_state": dict(initial_task_progress),
            "final_state": dict(initial_task_progress),
            "max_state": dict(initial_task_progress),
            "final_fraction": 0.0,
            "final_percent": 0.0,
            "max_fraction": 0.0,
            "max_percent": 0.0,
            "max_satisfied": 0,
            "total_trackable": 0,
            "max_satisfied_keys": [],
            "trackable_keys": [],
        }
        self._update_task_progress_summary(initial_task_progress)
        self._current_rollout_metadata = {
            "task_name": self.cfg.task.name,
            "instance_id": int(instance_id),
            "episode_id": int(episode_id),
            "inference_log_path": str(self._inference_log_path()),
            "inference_log_start_byte": self._inference_log_size(),
            "rollout_paths": dict(rollout_paths or {}),
        }

    def _summarize_inference_log_slice(self, start_byte: int, end_byte: int) -> dict[str, Any]:
        log_path = self._inference_log_path()
        summary = {
            "path": str(log_path),
            "start_byte": start_byte,
            "end_byte": end_byte,
            "records": 0,
            "inference_index_min": None,
            "inference_index_max": None,
            "mapped_subtasks": {},
            "first_prompts": [],
        }
        if end_byte <= start_byte or not log_path.exists():
            return summary

        try:
            with log_path.open("rb") as f:
                f.seek(start_byte)
                payload = f.read(end_byte - start_byte).decode("utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"Failed to summarize inference prompt log {log_path}: {exc}")
            return summary

        indices = []
        mapped_subtasks = {}
        first_prompts = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "infer":
                continue
            summary["records"] += 1
            inference_index = record.get("inference_index")
            if isinstance(inference_index, int):
                indices.append(inference_index)
            mapped_subtask = record.get("mapped_subtask")
            if isinstance(mapped_subtask, str):
                mapped_subtasks[mapped_subtask] = mapped_subtasks.get(mapped_subtask, 0) + 1
            prompt = record.get("prompt")
            if isinstance(prompt, str) and len(first_prompts) < 5:
                first_prompts.append(prompt)
        if indices:
            summary["inference_index_min"] = min(indices)
            summary["inference_index_max"] = max(indices)
        summary["mapped_subtasks"] = mapped_subtasks
        summary["first_prompts"] = first_prompts
        return summary

    def finalize_rollout_metadata(self) -> dict[str, Any]:
        metadata = dict(self._current_rollout_metadata or {})
        start_byte = int(metadata.get("inference_log_start_byte", 0) or 0)
        end_byte = self._inference_log_size()
        metadata["inference_log_end_byte"] = end_byte
        metadata["inference_log"] = self._summarize_inference_log_slice(start_byte, end_byte)
        metadata["task_progress"] = dict(self._current_task_progress_summary or {})
        return metadata

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False
        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"
        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        # optionally wrap env with TaskProgressWrapper
        if self.cfg.check_task_progress:
            env = TaskProgressWrapper(env)
        return env

    def load_robot(self) -> BaseRobot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        """
        Loads and returns the policy instance.
        """
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> list[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> tuple[bool, bool, dict]:
        """
        Performs a single step of the task by executing the policy, interacting with the environment,
        processing observations, updating metrics, and tracking trial success.

        Returns:
            Tuple[bool, bool]:
                - terminated (bool): Whether the episode has terminated (i.e., reached a terminal state).
                - truncated (bool): Whether the episode was truncated (i.e., stopped due to a time limit or other constraint).

        Workflow:
            1. Computes the next action using the policy based on the current observation.
            2. Steps the environment with the computed action and retrieves the next observation,
               termination and truncation flags, and additional info.
            3. If the episode has ended (terminated or truncated), increments the trial counter and
               updates the count of successful trials if the task was completed successfully.
            4. Preprocesses the new observation.
            5. Invokes step callbacks for all registered metrics to update their state.
            6. Returns the termination and truncation status.
        """
        self.robot_action = self.policy.forward(obs=self.obs)
        self.latest_inference_metadata = self._read_policy_inference_metadata()

        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        self._update_task_progress(info.get("task_progress"), source=f"step {self.env._current_step}")
        # process obs
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated, info

    @property
    def video_writer(self) -> tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: tuple[Container, Stream]) -> None:
        _close_video_writer(self._video_writer)
        self._video_writer = video_writer

    @property
    def rollout_video_writers(self) -> dict[str, tuple[Container, Stream]]:
        """
        Returns the video writer for the current rollout.
        """
        return self._rollout_video_writers

    @rollout_video_writers.setter
    def rollout_video_writers(self, rollout_video_writers: dict[str, tuple[Container, Stream]]) -> None:
        if self._rollout_video_writers is not None:
            for video_writer in self._rollout_video_writers.values():
                _close_video_writer(video_writer)
        self._rollout_video_writers = rollout_video_writers

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only] Whether to load the hidden test instance.
        """
        self.env.task.activity_instance_id = int(instance_id)
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
        with open(tro_file_path) as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)

                if self.cfg.perturb_pose:
                    perturbed_pos, perturbed_quat = self._pose_perturbator.perturb_robot_root_pose(
                        robot_pos, robot_quat
                    )
                    presampled_robot_poses[self.robot.model_name][0]["position"] = perturbed_pos
                    presampled_robot_poses[self.robot.model_name][0]["orientation"] = perturbed_quat

                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()
        if self.cfg.check_task_progress:
            self._read_current_task_progress(source=f"task instance {instance_id}")
            self._attach_task_progress_to_obs(self.obs)

    def _read_current_task_progress(self, source: str) -> None:
        assert (
            self.env.task.activity_name in CHALLENGE_TASKS_PROGRESS_APPROXIMATION
        ), f"Task {self.env.task.activity_name} not supported in TaskProgressWrapper!"
        progress_fn = CHALLENGE_TASKS_PROGRESS_APPROXIMATION[self.env.task.activity_name]
        self._update_task_progress(progress_fn(self.env), source=source)

    def _update_task_progress(self, task_progress: dict | None, source: str) -> None:
        if not self.cfg.check_task_progress:
            return

        task_progress = _normalize_task_progress(task_progress)
        if task_progress is None:
            return

        if self._last_logged_task_progress is None:
            logger.info(f"Task progress initialized ({source}): {task_progress}")
        else:
            changed = {
                key: (self._last_logged_task_progress.get(key), value)
                for key, value in task_progress.items()
                if self._last_logged_task_progress.get(key) != value
            }
            if changed:
                logger.info(f"Task progress changed ({source}): {changed}")

        self.latest_task_progress = task_progress
        self._last_logged_task_progress = dict(task_progress)
        self._update_task_progress_summary(task_progress)

    def _attach_task_progress_to_obs(self, obs: dict) -> dict:
        if self.cfg.check_task_progress and self.latest_task_progress is not None:
            obs["task_progress"] = dict(self.latest_task_progress)
        return obs

    def _read_policy_inference_metadata(self) -> dict | None:
        metadata = getattr(self.policy, "last_inference_metadata", None)
        if isinstance(metadata, dict):
            return metadata

        # Keep this fallback explicit: WebsocketPolicy wraps WebsocketClientPolicy,
        # which stores the raw response metadata separately.
        inner_policy = getattr(self.policy, "policy", None)
        metadata = getattr(inner_policy, "last_response_metadata", None)
        return metadata if isinstance(metadata, dict) else None

    def _read_latest_inference_log_metadata(self) -> dict | None:
        log_path = Path(self.cfg.log_path) / "inference_prompts.jsonl"
        if not log_path.exists():
            return self._latest_logged_inference_metadata

        try:
            if log_path.stat().st_size < self._inference_log_offset:
                self._inference_log_offset = 0
                self._latest_logged_inference_metadata = None

            with log_path.open("r", encoding="utf-8") as f:
                f.seek(self._inference_log_offset)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("event") != "infer":
                        continue
                    self._latest_logged_inference_metadata = {
                        "inference_index": record.get("inference_index"),
                        "mapped_subtask": record.get("mapped_subtask"),
                        "prompt": record.get("prompt"),
                        "task_progress": record.get("task_progress"),
                    }
                self._inference_log_offset = f.tell()
        except OSError as exc:
            logger.warning(f"Failed to read inference prompt log {log_path}: {exc}")

        return self._latest_logged_inference_metadata

    def _current_inference_metadata(self) -> dict:
        policy_metadata = self.latest_inference_metadata if isinstance(self.latest_inference_metadata, dict) else {}
        if policy_metadata.get("mapped_subtask") and policy_metadata.get("prompt"):
            return policy_metadata

        logged_metadata = self._read_latest_inference_log_metadata()
        if not isinstance(logged_metadata, dict):
            return policy_metadata

        if not self._inference_log_fallback_logged:
            logger.info("Using inference_prompts.jsonl as debug-video metadata fallback.")
            self._inference_log_fallback_logged = True

        metadata = dict(logged_metadata)
        metadata.update({key: value for key, value in policy_metadata.items() if value is not None})
        return metadata

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        # append task id to obs
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return self._attach_task_progress_to_obs(obs)

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        # concatenate obs
        left_wrist_raw = self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy()[..., :3]
        right_wrist_raw = self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy()[..., :3]
        head_raw = self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy()[..., :3]
        left_wrist_rgb, right_wrist_rgb, head_rgb = resize_debug_video_views(
            left_wrist_rgb=left_wrist_raw,
            right_wrist_rgb=right_wrist_raw,
            head_rgb=head_raw,
        )
        metadata = self._current_inference_metadata()
        task_progress = metadata.get("task_progress") or self.latest_task_progress
        frame = build_debug_video_frame(
            left_wrist_rgb=left_wrist_rgb,
            right_wrist_rgb=right_wrist_rgb,
            head_rgb=head_rgb,
            header_lines=[
                f"task: {self.cfg.task.name} | env_step: {self.env._current_step} | "
                f"inference: {metadata.get('inference_index') if metadata.get('inference_index') is not None else 'n/a'}",
            ],
            context_label="task_pregress (from env)",
            context_value=task_progress,
            mapped_subtask=metadata.get("mapped_subtask"),
            prompt=metadata.get("prompt"),
        )
        if not self._debug_video_shape_logged:
            logger.info(
                "Debug video layout: "
                f"raw_rgb_shapes={{'left_wrist': {left_wrist_raw.shape}, "
                f"'right_wrist': {right_wrist_raw.shape}, 'head': {head_raw.shape}}}, "
                f"resized_rgb_shapes={{'left_wrist': {left_wrist_rgb.shape}, "
                f"'right_wrist': {right_wrist_rgb.shape}, 'head': {head_rgb.shape}}}, "
                f"text_panel={(DEBUG_VIDEO_TEXT_HEIGHT, DEBUG_VIDEO_WIDTH, 3)}, "
                f"camera_panel={(DEBUG_VIDEO_HEAD_SIZE, DEBUG_VIDEO_WIDTH, 3)}, "
                f"output_frame={frame.shape}, writer_resolution={DEBUG_VIDEO_RESOLUTION}"
            )
            self._debug_video_shape_logged = True
        write_video(
            np.expand_dims(frame, 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _write_rollout(self) -> None:
        """
        Write the current robot observations to rollout video.
        """
        # logger.info(f"{self.obs['robot_r1::proprio'].shape=}, dtype={self.obs['robot_r1::proprio'].dtype}")
        # logger.info(f"{self.robot_action.shape=}, dtype={self.robot_action.dtype}")

        self.rollout_state_action["state"].append(self.obs["robot_r1::proprio"].cpu().numpy())
        self.rollout_state_action["action"].append(self.robot_action.cpu().numpy())

        for camera_name in ROLLOUT_CAMERA_NAMES:
            write_video(
                self.obs[ROBOT_CAMERA_NAMES["R1Pro"][camera_name] + "::rgb"].numpy()[None, ...],
                video_writer=self.rollout_video_writers[camera_name],
                batch_size=1,
                mode="rgb",
            )

    def success_callback(self, success: bool) -> None:
        """
        Callback function to be called when the task is completed successfully.
        """
        # self.video_writer = None
        # self.rollout_video_writers = None

        if success:
            self.n_success_trials += 1
            if hasattr(self, "cur_video_name"):
                success_video_name = self.cur_video_name.replace(".mp4", "_success.mp4")
                try:
                    shutil.move(self.cur_video_name, success_video_name)
                except:
                    logger.warning(f"Failed to move video {self.cur_video_name} to {success_video_name}")
                self.cur_video_name = success_video_name
            if hasattr(self, "rollout_paths"):
                np.savez_compressed(self.rollout_paths["state_action"], self.rollout_state_action)
                logger.info(f"Saved rollout data to {self.rollout_paths['state_action']}")
        elif hasattr(self, "rollout_paths"):
            for camera_name in ROLLOUT_CAMERA_NAMES:
                try:
                    os.remove(self.rollout_paths[camera_name])
                except:
                    logger.warning(f"Failed to remove rollout video {self.rollout_paths.get(camera_name)}")

    def reset(self) -> None:
        """
        Reset the environment, policy, and compute metrics.
        """
        raw_obs = self.env.reset()[0]
        if self.cfg.check_task_progress:
            self.latest_task_progress = None
            self._last_logged_task_progress = None
            self._read_current_task_progress(source="reset")
        self.obs = self._preprocess_obs(raw_obs)
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # print stats
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.rollout_video_writers = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    # set headless mode
    gm.HEADLESS = config.headless
    # set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    if config.save_rollout:
        rollout_path = Path(config.log_path).expanduser() / "rollouts"
        rollout_path.mkdir(parents=True, exist_ok=True)
    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")

    # get run instances
    if config.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, set eval_on_train_instances to False for test instances."
        )
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(m.NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]

    elif config.test_hidden:
        instances_to_run = (
            config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"

    else:
        # parallel evaluator
        if config.use_parallel_evaluator:
            instances_to_run = set(range(config.parallel_evaluator_start_idx, config.parallel_evaluator_end_idx))
            logger.info(
                f"Using parallel evaluator with start index {config.parallel_evaluator_start_idx} and end index {config.parallel_evaluator_end_idx}"
            )
        else:
            instances_to_run = (
                config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
            )

        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
        # load csv file
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path) as f:
            lines = list(csv.reader(f))[1:]
        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name
        ), f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]

    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        # NOTE: we change the order of the loops to make diversity
        for epi in range(m.NUM_EVAL_EPISODES):
            evaluator.reset()
            for idx in instances_to_run:
                evaluator.reset()
                evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
                logger.info(f"Starting task instance {idx} / episode {epi} for evaluation...")

                done = False
                if config.write_video:
                    evaluator.cur_video_name = f"{video_path!s}/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=evaluator.cur_video_name,
                        resolution=DEBUG_VIDEO_RESOLUTION,
                    )

                rollout_paths = {}
                if config.save_rollout:
                    rollout_video_writers = {}
                    for camera_name in ROLLOUT_CAMERA_NAMES:
                        rollout_id_path = Path(rollout_path) / f"{int(idx):04d}_{int(epi):04d}"
                        rollout_id_path.mkdir(parents=True, exist_ok=True)
                        rollout_paths[camera_name] = str(rollout_id_path / f"{camera_name}.mp4")
                        rollout_video_writers[camera_name] = create_video_writer(
                            fpath=rollout_paths[camera_name],
                            resolution=HEAD_RESOLUTION if camera_name == "head" else WRIST_RESOLUTION,
                        )

                    evaluator.rollout_video_writers = rollout_video_writers
                    rollout_paths["state_action"] = str(rollout_id_path / "state_action.npz")
                    evaluator.rollout_paths = rollout_paths
                    evaluator.rollout_state_action = {"state": [], "action": []}
                    logger.info(f"Created rollout video writers under {rollout_id_path}")

                evaluator.begin_rollout_metadata(idx, epi, rollout_paths=rollout_paths)

                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                while not done:
                    time_start = time.time()
                    terminated, truncated, info = evaluator.step()
                    time_step = time.time() - time_start

                    if time_step > 15 * 60:
                        logger.error(f"Step timeout: {time_step} seconds, terminating evaluation")
                        exit(1)

                    if terminated or truncated:
                        done = True
                    if config.save_rollout:
                        evaluator._write_rollout()
                    if config.write_video and evaluator.env._current_step % 20 == 0:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")

                    # callback for end of episode
                    if terminated or truncated:
                        if config.write_video:
                            evaluator.video_writer = None
                        if config.save_rollout:
                            evaluator.rollout_video_writers = None
                        evaluator.success_callback(info["done"]["success"])

                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                metrics["rollout"] = evaluator.finalize_rollout_metadata()
                metrics["task_progress"] = metrics["rollout"]["task_progress"]
                metrics["inference_log"] = metrics["rollout"]["inference_log"]
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                # reset video writer
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {evaluator.cur_video_name}")
                # reset rollout video writers
                if config.save_rollout:
                    saved_rollout_paths = evaluator.rollout_paths
                    evaluator.rollout_video_writers = None
                    evaluator.rollout_paths = None
                    evaluator.rollout_state_action = None
                    logger.info(f"Saved rollout output to {saved_rollout_paths}")
                else:
                    logger.warning("No observations were recorded.")
