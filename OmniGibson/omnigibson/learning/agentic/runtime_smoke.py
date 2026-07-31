"""Real-runtime smoke for the agentic R1Pro manifest and snapshot substrate."""

from __future__ import annotations

import argparse
import json

from gello.robots.sim_robot.og_teleop_utils import generate_robot_config
from gello.robots.sim_robot.og_teleop_utils import load_available_tasks
import numpy as np
import omnigibson as og
from omnigibson.learning.agentic.controller_manifest import build_controller_manifest
from omnigibson.learning.agentic.controller_manifest import validate_r1pro_manifest
from omnigibson.learning.agentic.environment_session import AgenticEnvironmentSession
from omnigibson.learning.agentic.snapshot import CompositeSnapshotManager
from omnigibson.learning.utils.eval_utils import generate_basic_environment_config
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


def run_smoke(task_name: str, *, tolerance: float) -> dict[str, object]:
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

    env = og.Environment(configs=config)
    robot = env.scene.object_registry("name", "robot_r1")
    manifest = build_controller_manifest(robot, simulator=og.sim)
    validate_r1pro_manifest(manifest)
    manager = CompositeSnapshotManager(
        simulator=og.sim,
        env=env,
        runtime_fingerprint=f"omnigibson:{og.__version__}",
        controller_fingerprint=manifest.fingerprint,
        propagate_once=_propagate_once,
    )
    session = AgenticEnvironmentSession(env=env, controller_manifest=manifest, snapshot_manager=manager)
    session.pause()
    before_root = robot.get_position_orientation()[0].detach().cpu().numpy()
    before_joints = robot.get_joint_positions().detach().cpu().numpy()
    snapshot = session.snapshot(task_name=task_name, instance_id="seed", episode_id="runtime-smoke")

    session.resume()
    step_result = session.step(_no_op_action(robot))
    session.pause()
    restore_report = session.restore(snapshot)
    restored_root = robot.get_position_orientation()[0].detach().cpu().numpy()
    restored_joints = robot.get_joint_positions().detach().cpu().numpy()
    root_max_abs_drift = float(np.max(np.abs(restored_root - before_root)))
    joint_max_abs_drift = float(np.max(np.abs(restored_joints - before_joints)))
    if root_max_abs_drift > tolerance or joint_max_abs_drift > tolerance:
        raise RuntimeError(
            f"Snapshot round-trip drift exceeds {tolerance}: root={root_max_abs_drift}, joints={joint_max_abs_drift}."
        )
    return {
        "task": task_name,
        "controller_manifest": manifest.to_dict(),
        "executed_env_step": step_result.env_step,
        "restored_env_step": restore_report.restored_env_step,
        "propagation_sim_steps": restore_report.propagation_sim_steps,
        "root_max_abs_drift": root_max_abs_drift,
        "joint_max_abs_drift": joint_max_abs_drift,
        "tolerance": tolerance,
        "scope": "controller_manifest_and_simulator_snapshot_substrate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.task, tolerance=args.tolerance), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
