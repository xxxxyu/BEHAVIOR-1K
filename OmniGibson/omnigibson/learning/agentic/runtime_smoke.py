"""Real-runtime smoke for the agentic R1Pro manifest and snapshot substrate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import traceback

from gello.robots.sim_robot.og_teleop_utils import generate_robot_config
from gello.robots.sim_robot.og_teleop_utils import load_available_tasks
import numpy as np
import omnigibson as og
from omnigibson.learning.agentic.controller_manifest import build_controller_manifest
from omnigibson.learning.agentic.controller_manifest import validate_r1pro_manifest
from omnigibson.learning.agentic.environment_session import AgenticEnvironmentSession
from omnigibson.learning.agentic.evaluator_state import EvaluatorSnapshotComponent
from omnigibson.learning.agentic.snapshot import CompositeSnapshotManager
from omnigibson.learning.utils.eval_utils import generate_basic_environment_config
from omnigibson.learning.utils.task_progress_utils import CHALLENGE_TASKS_PROGRESS_APPROXIMATION
from omnigibson.macros import gm
import torch as th


def _no_op_action(robot: object) -> np.ndarray:
    control_dict = robot.get_control_dict()
    actions = [
        robot.controllers[name].compute_no_op_action(control_dict=control_dict) for name in robot.controller_order
    ]
    return th.cat(actions).detach().cpu().numpy().astype(np.float32, copy=False)


def _propagate_once() -> dict[str, float]:
    og.sim.step()
    return {"sim_step_dt": float(og.sim.get_sim_step_dt())}


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _robot_kinematics(robot: object) -> dict[str, np.ndarray]:
    root_position, root_orientation = robot.get_position_orientation()
    return {
        "root_position": _numpy(root_position),
        "root_orientation": _numpy(root_orientation),
        "root_linear_velocity": _numpy(robot.get_linear_velocity()),
        "root_angular_velocity": _numpy(robot.get_angular_velocity()),
        "joint_positions": _numpy(robot.get_joint_positions()),
        "joint_velocities": _numpy(robot.get_joint_velocities()),
    }


def _max_abs_drift(observed: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, float]:
    if set(observed) != set(expected):
        raise RuntimeError("Robot kinematic state fields changed during snapshot smoke.")
    return {name: float(np.max(np.abs(observed[name] - expected[name]))) for name in expected}


def _drift_violations(
    drift_groups: dict[str, dict[str, float]], *, position_tolerance: float, velocity_tolerance: float
) -> dict[str, dict[str, float]]:
    violations = {}
    for group, drifts in drift_groups.items():
        for field, drift in drifts.items():
            tolerance = velocity_tolerance if field.endswith(("velocity", "velocities")) else position_tolerance
            if drift > tolerance:
                violations[f"{group}.{field}"] = {"drift": drift, "tolerance": tolerance}
    return violations


def _normalize_progress(progress: dict[str, object]) -> dict[str, object]:
    normalized = {}
    for key, value in progress.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item() if value.shape == () else value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        normalized[str(key)] = value
    return normalized


def _runtime_evaluator(env: object, progress_reader: Any) -> SimpleNamespace:
    return SimpleNamespace(
        env=env,
        metrics=[SimpleNamespace(name="runtime-smoke", step_count=0)],
        n_trials=0,
        n_success_trials=0,
        total_time=0.0,
        robot_action=np.zeros(23, dtype=np.float32),
        latest_task_progress=progress_reader(),
        _last_logged_task_progress=None,
        latest_inference_metadata=None,
        _latest_logged_inference_metadata=None,
        _inference_log_offset=0,
        _inference_log_fallback_logged=False,
        _current_rollout_metadata={"episode_id": "runtime-smoke"},
        _current_task_progress_summary=None,
        obs=env.get_obs()[0],
    )


def _write_report(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(f"{payload}\n", encoding="utf-8")
    temporary_path.replace(path)


def run_smoke(
    task_name: str, *, position_tolerance: float, velocity_tolerance: float, branch_steps: int
) -> dict[str, object]:
    if position_tolerance <= 0 or velocity_tolerance <= 0:
        raise ValueError("Snapshot drift tolerances must be positive.")
    if branch_steps < 1:
        raise ValueError("Snapshot smoke branch_steps must be positive.")
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    available_tasks = load_available_tasks()
    if task_name not in available_tasks:
        raise ValueError(f"Unknown BEHAVIOR task: {task_name}.")
    task_cfg = available_tasks[task_name][0]
    config = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    config["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
    config["robots"][0]["obs_modalities"] = ["proprio"]
    config["task"]["include_obs"] = False
    config["task"]["termination_config"]["max_steps"] = 10

    env = None
    try:
        env = og.Environment(configs=config)
        robot = env.scene.object_registry("name", "robot_r1")
        manifest = build_controller_manifest(robot, simulator=og.sim)
        validate_r1pro_manifest(manifest)
        progress_fn = CHALLENGE_TASKS_PROGRESS_APPROXIMATION[task_name]
        progress_reader = lambda: _normalize_progress(progress_fn(env))
        evaluator = _runtime_evaluator(env, progress_reader)
        evaluator_component = EvaluatorSnapshotComponent(
            evaluator=evaluator,
            progress_reader=progress_reader,
            observation_reader=lambda: env.get_obs()[0],
        )
        manager = CompositeSnapshotManager(
            simulator=og.sim,
            env=env,
            runtime_fingerprint=f"omnigibson:{og.__version__}",
            controller_fingerprint=manifest.fingerprint,
            components=(evaluator_component,),
            propagate_once=_propagate_once,
        )
        session = AgenticEnvironmentSession(env=env, controller_manifest=manifest, snapshot_manager=manager)
        session.pause()
        snapshot = session.snapshot(task_name=task_name, instance_id="seed", episode_id="runtime-smoke")

        # Restore performs one required simulator propagation step. Compare it with
        # the matched uninterrupted trajectory after the same propagation, rather
        # than comparing post-propagation state with the raw snapshot state.
        og.sim.step()
        reference_state = _robot_kinematics(robot)
        session.resume()
        for _ in range(branch_steps):
            step_result = session.step(_no_op_action(robot))
        uninterrupted_branch_state = _robot_kinematics(robot)
        session.pause()
        evaluator.n_trials = 7
        evaluator.latest_task_progress = {"deliberately_mutated": True}
        evaluator.metrics[0].step_count = 9

        restore_report = session.restore(snapshot)
        restored_state = _robot_kinematics(robot)
        reference_drift = _max_abs_drift(restored_state, reference_state)
        if evaluator.n_trials != 0 or evaluator.metrics[0].step_count != 0:
            raise RuntimeError("Evaluator or metric state did not restore to its captured value.")

        session.resume()
        for _ in range(branch_steps):
            step_result = session.step(_no_op_action(robot))
        restored_branch_state = _robot_kinematics(robot)
        branch_drift = _max_abs_drift(restored_branch_state, uninterrupted_branch_state)
        session.pause()

        repeated_restore_report = session.restore(snapshot)
        repeated_state = _robot_kinematics(robot)
        repeated_restore_drift = _max_abs_drift(repeated_state, restored_state)
        drift_groups = {
            "matched_propagation": reference_drift,
            "matched_branch": branch_drift,
            "repeated_restore": repeated_restore_drift,
        }
        violations = _drift_violations(
            drift_groups,
            position_tolerance=position_tolerance,
            velocity_tolerance=velocity_tolerance,
        )
        return {
            "task": task_name,
            "controller_manifest": manifest.to_dict(),
            "executed_env_step": step_result.env_step,
            "restored_env_step": restore_report.restored_env_step,
            "repeated_restored_env_step": repeated_restore_report.restored_env_step,
            "propagation_sim_steps": restore_report.propagation_sim_steps,
            "branch_steps": branch_steps,
            "evaluator_restore": restore_report.component_reports[evaluator_component.name],
            "max_abs_drift": drift_groups,
            "position_tolerance": position_tolerance,
            "velocity_tolerance": velocity_tolerance,
            "violations": violations,
            "within_tolerance": not violations,
            "scope": "controller_manifest_simulator_and_evaluator_snapshot_substrate",
        }
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--position-tolerance", "--tolerance", dest="position_tolerance", type=float, default=1e-3)
    parser.add_argument("--velocity-tolerance", type=float, default=5e-3)
    parser.add_argument("--branch-steps", type=int, default=3)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    exit_code = 0
    try:
        result = run_smoke(
            args.task,
            position_tolerance=args.position_tolerance,
            velocity_tolerance=args.velocity_tolerance,
            branch_steps=args.branch_steps,
        )
        report = {"status": "passed" if result["within_tolerance"] else "failed", **result}
        exit_code = 0 if result["within_tolerance"] else 1
    except Exception as exc:
        exit_code = 1
        report = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    try:
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.report_path is not None:
            _write_report(args.report_path, payload)
        print(payload, flush=True)
    finally:
        og.shutdown()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
