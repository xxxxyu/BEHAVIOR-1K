"""Privileged, read-only CuRobo geometry queries for the Radio diagnostic.

This module is intentionally imported and initialized only by the semantic-geometry
diagnostic runtime.  It must never become part of the ordinary candidate tool set.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
from omnigibson.action_primitives.curobo import create_world_mesh_collision
from omnigibson.learning.agentic.semantic_geometry import RADIO_TASK_BINDING
from omnigibson.macros import gm


CORRIDOR_STAGE_NAMES = ("pregrasp", "preclose", "lift_1", "lift_2")
RIGHT_ARM_CLEARANCE_LINKS = (
    "right_arm_link1",
    "right_arm_link2",
    "right_arm_link3",
    "right_arm_link4",
    "right_arm_link5",
    "right_arm_link6",
    "right_arm_link7",
    "right_gripper_link",
    "right_gripper_finger_link1",
    "right_gripper_finger_link2",
    "right_realsense_link",
)
TABLE_CLEARANCE_MAX_DISTANCE_M = 1.0
INTERPOLATION_MAX_JOINT_DELTA_RAD = 0.03
DIAGNOSTIC_SOLVER_INPUT_QUANTUM = 1e-4
PROVENANCE_TARGET_TRANSLATION_TOLERANCE_M = 0.002
PROVENANCE_TARGET_ORIENTATION_TOLERANCE_RAD = 0.01


def _resolve_curobo_device(device: str | None) -> str:
    """Resolve CuRobo's CUDA device independently of the physics tensor backend."""

    if device is None:
        if gm.GPU_ID is None:
            raise RuntimeError("CuRobo diagnostics require an explicit OMNIGIBSON_GPU_ID.")
        device = f"cuda:{int(gm.GPU_ID)}"
    resolved = th.device(device)
    if resolved.type != "cuda" or resolved.index is None:
        raise ValueError(f"CuRobo diagnostics require an indexed CUDA device, got {device!r}.")
    return str(resolved)


def _as_pose(value: Mapping[str, object], *, parent: str, child: str) -> tuple[th.Tensor, th.Tensor]:
    if value.get("parent_frame") != parent or value.get("child_frame") != child:
        raise ValueError(f"Expected pose {parent}->{child}.")
    position = th.as_tensor(value.get("translation_m"), dtype=th.float32)
    quaternion = th.as_tensor(value.get("quaternion_xyzw"), dtype=th.float32)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError(f"Malformed pose {parent}->{child}.")
    if not th.isfinite(position).all() or not th.isfinite(quaternion).all():
        raise ValueError(f"Pose {parent}->{child} contains non-finite values.")
    norm = th.linalg.vector_norm(quaternion)
    if float(norm) <= 1e-8:
        raise ValueError(f"Pose {parent}->{child} has a zero quaternion.")
    return position, quaternion / norm


def _pose_digest(position: th.Tensor, quaternion: th.Tensor) -> str:
    payload = th.cat([position, quaternion]).detach().cpu().numpy().astype("<f4", copy=False).tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _pose_residual(
    actual_position: th.Tensor,
    actual_orientation: th.Tensor,
    target_position: th.Tensor,
    target_orientation: th.Tensor,
) -> dict[str, float]:
    target_position = target_position.to(device=actual_position.device, dtype=actual_position.dtype)
    target_orientation = target_orientation.to(device=actual_orientation.device, dtype=actual_orientation.dtype)
    actual_orientation = actual_orientation / th.linalg.vector_norm(actual_orientation)
    target_orientation = target_orientation / th.linalg.vector_norm(target_orientation)
    dot = th.clamp(th.abs(th.dot(actual_orientation, target_orientation)), 0.0, 1.0)
    return {
        "translation_m": float(th.linalg.vector_norm(actual_position - target_position).detach().cpu()),
        "orientation_rad": float((2.0 * th.acos(dot)).detach().cpu()),
    }


def _quantize_tensor(value: th.Tensor) -> th.Tensor:
    return th.round(value / DIAGNOSTIC_SOLVER_INPUT_QUANTUM) * DIAGNOSTIC_SOLVER_INPUT_QUANTUM


def _quantize_pose(position: th.Tensor, quaternion: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    position = _quantize_tensor(position)
    quaternion = _quantize_tensor(quaternion)
    norm = th.linalg.vector_norm(quaternion)
    if float(norm) <= 1e-8:
        raise ValueError("Quantized pose has a zero quaternion.")
    return position, quaternion / norm


def _summarize_clearance(
    distance: th.Tensor,
    spheres: th.Tensor,
    *,
    trajectory_samples: int,
) -> dict[str, object]:
    """Convert center ESDF into right-arm surface clearance without hiding saturation."""

    clearance = -distance - spheres[..., 3]
    flat_clearance = clearance.reshape(-1)
    flat_distance = distance.reshape(-1)
    minimum_index = int(th.argmin(flat_clearance).item())
    minimum = float(flat_clearance[minimum_index].detach().cpu())
    minimum_center_distance = float(flat_distance[minimum_index].detach().cpu())
    lower_bound = minimum_center_distance <= -TABLE_CLEARANCE_MAX_DISTANCE_M + 1e-5
    saturated = abs(minimum_center_distance) >= TABLE_CLEARANCE_MAX_DISTANCE_M - 1e-5
    return {
        "status": "available",
        "minimum_clearance_m": minimum,
        "clearance_is_lower_bound": lower_bound,
        "minimum_sample_esdf_saturated": saturated,
        "sample_count": trajectory_samples,
        "sphere_count": int(spheres.shape[2]),
    }


def _summarize_ik_solution(goal_q: th.Tensor) -> dict[str, object]:
    if goal_q.ndim != 1 or not th.isfinite(goal_q).all():
        raise ValueError("IK solution must be a finite joint vector.")
    return {
        "status": "available",
        "feasible": True,
        "solution_joint_positions_rad": goal_q.detach().cpu().tolist(),
    }


def _first_positive_sample(values: object) -> int | None:
    positive = th.as_tensor(values).detach().reshape(-1) > 0.0
    indices = th.nonzero(positive, as_tuple=False).reshape(-1)
    return int(indices[0].item()) if indices.numel() else None


def _max_positive_value(values: object) -> float | None:
    values = th.as_tensor(values).detach().reshape(-1)
    positive = values[values > 0.0]
    return float(th.max(positive).cpu()) if positive.numel() else None


def _summarize_per_sphere_collision(
    values: th.Tensor,
    *,
    sphere_to_link: Mapping[int, str],
) -> tuple[th.Tensor, dict[str, object]]:
    """Summarize per-sphere world cost without losing the colliding robot link."""

    if values.ndim < 2:
        raise ValueError("Per-sphere collision scores must include sample and sphere dimensions.")
    sphere_count = int(values.shape[-1])
    sample_scores = th.sum(values, dim=-1).reshape(-1)
    flat_values = values.reshape(-1, sphere_count)
    worst_flat_index = int(th.argmax(flat_values).item())
    worst_sphere_index = worst_flat_index % sphere_count
    worst_sphere_values = flat_values[:, worst_sphere_index]
    return sample_scores, {
        "detected": bool(th.any(sample_scores > 0.0).item()),
        "first_collision_sample": _first_positive_sample(sample_scores),
        "maximum_constraint_score": _max_positive_value(sample_scores),
        "worst_sphere_index": worst_sphere_index,
        "worst_sphere_link": sphere_to_link.get(worst_sphere_index),
        "worst_sphere_maximum_score": _max_positive_value(worst_sphere_values),
    }


def _stage_collision_policy(
    stage_name: str,
    collision_channels: Mapping[str, object],
    *,
    contact_phase_start_sample: int | None,
) -> dict[str, object]:
    """Allow only exact target-Radio contact in the grasp-contact corridor."""

    target = collision_channels["target_object_collision"]
    non_target = collision_channels["non_target_world_collision"]
    self_collision = collision_channels["self_collision"]
    target_detected = target.get("detected") is True
    target_first = target.get("first_collision_sample")
    reasons = []
    if self_collision.get("detected") is True:
        reasons.append("self_collision")
    if non_target.get("detected") is True:
        reasons.append("non_target_world_collision")

    if stage_name == "pregrasp":
        target_contact_allowed = not target_detected
        if target_detected:
            reasons.append("target_contact_before_grasp_approach")
        policy = "target_contact_forbidden"
    elif stage_name == "preclose":
        if contact_phase_start_sample is None:
            raise ValueError("Preclose provenance must declare its grasp-contact phase start sample.")
        target_contact_allowed = not target_detected or (
            isinstance(target_first, int) and target_first >= contact_phase_start_sample
        )
        if not target_contact_allowed:
            reasons.append("target_contact_before_preclose_contact_phase")
        policy = "target_contact_allowed_only_in_final_approach_segment"
    elif stage_name in ("lift_1", "lift_2"):
        target_contact_allowed = True
        policy = "retained_target_contact_allowed"
    else:
        raise ValueError(f"Unknown corridor stage {stage_name!r}.")

    return {
        "policy": policy,
        "exact_task_binding": RADIO_TASK_BINDING,
        "contact_phase_start_sample": contact_phase_start_sample,
        "target_contact_detected": target_detected,
        "target_contact_allowed": target_contact_allowed,
        "blocking_reasons": reasons,
        "collision_safe_for_stage": not reasons,
    }


class RadioSemanticGeometryBackend:
    """CuRobo-backed geometry oracle with no simulator mutation."""

    def __init__(self, env: object, robot: object, *, device: str | None = None) -> None:
        if len(og.sim.scenes) != 1:
            raise RuntimeError("Radio geometry backend requires exactly one simulator scene.")
        self.env = env
        self.robot = robot
        self.device = _resolve_curobo_device(device)
        self.motion_generator = CuRoboMotionGenerator(
            robot,
            robot_cfg_path={
                CuRoboEmbodimentSelection.ARM: robot.curobo_path[CuRoboEmbodimentSelection.ARM],
                CuRoboEmbodimentSelection.DEFAULT: robot.curobo_path[CuRoboEmbodimentSelection.DEFAULT],
            },
            device=self.device,
            batch_size=1,
            use_cuda_graph=False,
            debug=False,
            collision_activation_distance=0.005,
        )
        self._table_checker = None
        self._table_query_buffer = None
        self._table_mesh_digest = None

    def capabilities(self) -> dict[str, dict[str, object]]:
        return {
            "hypothetical_base_pose_whole_arm_ik": {
                "status": "available",
                "source_api": "CuRoboMotionGenerator.compute_trajectories(ARM,initial_joint_pos,ik_only=True)",
                "model": "R1Pro arm embodiment with base and grippers locked to hypothetical joint state",
                "seed_strategy": "reset CuRobo ARM IKSolver Halton generator before each corridor evaluation",
                "input_quantum": DIAGNOSTIC_SOLVER_INPUT_QUANTUM,
            },
            "arm_trajectory_collision": {
                "status": "available",
                "source_api": "CuRoboMotionGenerator.check_collisions(DEFAULT)",
                "model": "CuRobo R1Pro collision spheres against current BEHAVIOR collision meshes",
            },
            "arm_table_clearance": {
                "status": "available",
                "source_api": "CuRobo WorldMeshCollision.get_sphere_distance(compute_esdf=True)",
                "model": "right articulated-arm spheres against task support-table collision meshes",
                "distance_convention": "table signed distance minus sphere radius; positive means clearance",
            },
        }

    def _candidate_joint_state(self, candidate_base_pose: Mapping[str, object]) -> th.Tensor:
        position, orientation = _as_pose(
            candidate_base_pose,
            parent="simulator_world",
            child="robot_base_footprint",
        )
        root_position, root_orientation = self.robot.root_link.get_position_orientation()
        root_position = root_position.detach().clone()
        root_orientation = root_orientation.detach().clone()
        position = position.to(device=root_position.device)
        orientation = orientation.to(device=root_orientation.device)
        root_inv_position, root_inv_orientation = T.invert_pose_transform(root_position, root_orientation)
        relative_position, relative_orientation = T.pose_transform(
            root_inv_position,
            root_inv_orientation,
            position,
            orientation,
        )
        intrinsic_eulers = T.mat2euler_intrinsic(T.quat2mat(relative_orientation))
        joint_state = self.robot.get_joint_positions().detach().clone()
        joint_state[self.robot.base_idx] = th.cat([relative_position, intrinsic_eulers])
        return joint_state

    def _candidate_joint_state_for_curobo(self, candidate_base_pose: Mapping[str, object]) -> th.Tensor:
        candidate = self.motion_generator.tensor_args.to_device(self._candidate_joint_state(candidate_base_pose))
        return _quantize_tensor(candidate)

    def _full_joint_state_for_curobo(self, joint_positions: th.Tensor):
        positions = joint_positions.unsqueeze(0) if joint_positions.ndim == 1 else joint_positions
        if positions.ndim != 2:
            raise ValueError("CuRobo joint positions must be a vector or a batch of vectors.")
        return lazy.curobo.types.state.JointState(
            position=self.motion_generator.tensor_args.to_device(positions),
            joint_names=self.motion_generator.robot_joint_names,
        )

    def _ordered_joint_state_for_curobo(
        self,
        joint_positions: th.Tensor,
        *,
        emb_sel: CuRoboEmbodimentSelection,
        locked_joint_positions: th.Tensor | None = None,
    ):
        """Update embodiment locks before ordering active joints for FK.

        CuRobo omits locked joints from ``kinematics.joint_names`` and stores
        their values in mutable fixed transforms. Direct FK therefore must
        update those transforms just like ``compute_trajectories`` does.
        """

        full_state = self._full_joint_state_for_curobo(joint_positions)
        locked_state = (
            full_state if locked_joint_positions is None else self._full_joint_state_for_curobo(locked_joint_positions)
        )
        self.motion_generator.update_locked_joints(locked_state, emb_sel)
        return full_state.get_ordered_joint_state(self.motion_generator.mg[emb_sel].kinematics.joint_names)

    def _reset_ik_seed_generator(self) -> None:
        """Make a read-only corridor query independent of prior diagnostic queries."""

        self.motion_generator.mg[CuRoboEmbodimentSelection.ARM].ik_solver.reset_seed()

    def _table_world(self):
        robot_transform = T.pose_inv(T.pose2mat(self.robot.root_link.get_position_orientation()))
        scope = getattr(getattr(self.env, "task", None), "object_scope", None)
        if not isinstance(scope, Mapping) or "table.n.02_1" not in scope:
            raise KeyError("Required BDDL support-table binding is absent: table.n.02_1")
        resolved = scope["table.n.02_1"]
        table = getattr(resolved, "unwrapped", resolved)
        meshes = []
        digest_parts = []
        for link in table.links.values():
            for collision_mesh in link.collision_meshes.values():
                obj_pose = T.pose2mat(collision_mesh.get_position_orientation())
                pose = robot_transform @ obj_pose
                pos, orn = T.mat2pose(pose)
                orn = orn[[3, 0, 1, 2]]
                vertices = collision_mesh.points.numpy()
                faces = collision_mesh.faces.numpy()
                scale = collision_mesh.get_world_scale().numpy()
                meshes.append(
                    lazy.curobo.geom.types.Mesh(
                        name=collision_mesh.prim_path,
                        pose=th.cat([pos, orn]).tolist(),
                        vertices=vertices,
                        faces=faces,
                        scale=scale,
                    )
                )
                digest_parts.extend(
                    [
                        str(collision_mesh.prim_path).encode(),
                        vertices.tobytes(),
                        faces.tobytes(),
                        scale.tobytes(),
                        pos.detach().cpu().numpy().tobytes(),
                        orn.detach().cpu().numpy().tobytes(),
                    ]
                )
        if not meshes:
            raise RuntimeError("Support table has no collision meshes.")
        digest = hashlib.sha256(b"".join(digest_parts)).hexdigest()
        return lazy.curobo.geom.types.WorldConfig(
            cuboid=None,
            sphere=None,
            mesh=meshes,
            cylinder=None,
            capsule=None,
        ).get_collision_check_world(), digest

    def _ensure_table_checker(self) -> None:
        world, digest = self._table_world()
        if self._table_checker is None:
            self._table_checker = create_world_mesh_collision(
                self.motion_generator.tensor_args,
                obb_cache_size=1,
                mesh_cache_size=max(32, len(world.mesh) + 1),
                max_distance=TABLE_CLEARANCE_MAX_DISTANCE_M,
            )
        if self._table_mesh_digest != digest:
            self._table_checker.load_collision_model(world)
            self._table_mesh_digest = digest

    def _target_pose(self, value: Mapping[str, object], stage: str) -> tuple[th.Tensor, th.Tensor]:
        return _quantize_pose(*_as_pose(value, parent="simulator_world", child=f"right_eef_{stage}"))

    def _target_pose_for_curobo(
        self,
        value: Mapping[str, object],
        stage: str,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Express a world-frame target in CuRobo's kinematic base frame."""

        target_position, target_orientation = self._target_pose(value, stage)
        base_link_name = self.motion_generator.base_link[CuRoboEmbodimentSelection.ARM]
        base_position, base_orientation = self.robot.links[base_link_name].get_position_orientation()
        target_position = target_position.to(device=base_position.device, dtype=base_position.dtype)
        target_orientation = target_orientation.to(device=base_orientation.device, dtype=base_orientation.dtype)
        base_inverse = T.invert_pose_transform(base_position, base_orientation)
        return _quantize_pose(*T.pose_transform(*base_inverse, target_position, target_orientation))

    def _left_hold_pose(self, candidate_base_pose: Mapping[str, object]) -> tuple[th.Tensor, th.Tensor]:
        """Carry the current base-relative left EEF pose to the hypothetical base."""

        candidate_position, candidate_orientation = _as_pose(
            candidate_base_pose,
            parent="simulator_world",
            child="robot_base_footprint",
        )
        current_base_position, current_base_orientation = self.robot.get_position_orientation()
        current_left_position, current_left_orientation = self.robot.eef_links["left"].get_position_orientation()
        candidate_position = candidate_position.to(device=current_base_position.device)
        candidate_orientation = candidate_orientation.to(device=current_base_orientation.device)
        base_inverse = T.invert_pose_transform(current_base_position, current_base_orientation)
        base_to_left = T.pose_transform(*base_inverse, current_left_position, current_left_orientation)
        return _quantize_pose(*T.pose_transform(candidate_position, candidate_orientation, *base_to_left))

    def _interpolated(self, start: th.Tensor, goal: th.Tensor) -> th.Tensor:
        intervals = max(1, math.ceil(float(th.max(th.abs(goal - start))) / INTERPOLATION_MAX_JOINT_DELTA_RAD))
        fractions = th.linspace(0.0, 1.0, intervals + 1, device=start.device)
        return start.unsqueeze(0) + fractions.unsqueeze(1) * (goal - start).unsqueeze(0)

    def _ik_goal_joint_positions(self, joint_state: object, *, expected_shape: th.Size) -> th.Tensor:
        positions = self.motion_generator.path_to_joint_trajectory(
            joint_state,
            get_full_js=False,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        )
        if positions.ndim == 2 and positions.shape[0] == 1:
            positions = positions[0]
        if positions.ndim != 1 or positions.shape != expected_shape:
            raise RuntimeError(
                f"IK-only ARM result has shape {tuple(positions.shape)}, expected {tuple(expected_shape)}."
            )
        return positions

    def _right_arm_sphere_indices(self) -> th.Tensor:
        config = self.motion_generator.mg[CuRoboEmbodimentSelection.ARM].kinematics.kinematics_config
        indices = []
        for name in RIGHT_ARM_CLEARANCE_LINKS:
            if name not in config.link_name_to_idx_map:
                continue
            link_indices = config.get_sphere_index_from_link_name(name)
            if link_indices.numel():
                indices.append(link_indices)
        if not indices:
            raise RuntimeError("CuRobo ARM model has no right-arm collision spheres.")
        return th.cat(indices).unique(sorted=True)

    def _sphere_link_names(self, *, emb_sel: CuRoboEmbodimentSelection) -> dict[int, str]:
        config = self.motion_generator.mg[emb_sel].kinematics.kinematics_config
        sphere_to_link: dict[int, str] = {}
        for link_name in config.link_name_to_idx_map:
            for sphere_index in config.get_sphere_index_from_link_name(link_name).detach().cpu().tolist():
                sphere_to_link[int(sphere_index)] = link_name
        return sphere_to_link

    def _radio_collision_mesh_identity(self, world_checker: object) -> dict[str, object]:
        scope = getattr(getattr(self.env, "task", None), "object_scope", None)
        if not isinstance(scope, Mapping) or RADIO_TASK_BINDING not in scope:
            raise KeyError(f"Required BDDL Radio binding is absent: {RADIO_TASK_BINDING}")
        resolved = scope[RADIO_TASK_BINDING]
        if getattr(resolved, "exists", True) is not True:
            raise RuntimeError(f"Required BDDL Radio object does not exist: {RADIO_TASK_BINDING}")
        radio = getattr(resolved, "unwrapped", resolved)
        mesh_names = sorted(
            str(collision_mesh.prim_path)
            for link in radio.links.values()
            for collision_mesh in link.collision_meshes.values()
        )
        if not mesh_names:
            raise RuntimeError("Task-bound Radio has no collision meshes.")
        available = set(world_checker.get_obstacle_names())
        missing = [name for name in mesh_names if name not in available]
        if missing:
            raise RuntimeError(f"Task-bound Radio collision meshes are absent from CuRobo world: {missing}")
        return {
            "resolution": "exact_task_object_scope_binding",
            "task_binding": RADIO_TASK_BINDING,
            "name": str(getattr(radio, "name", "")),
            "category": str(getattr(radio, "category", "")),
            "prim_path": str(getattr(radio, "prim_path", "")),
            "collision_mesh_names": mesh_names,
        }

    def _per_sphere_world_collision(
        self,
        robot_spheres: th.Tensor,
        *,
        world_cost: object,
    ) -> th.Tensor:
        from curobo.geom.sdf.world import CollisionQueryBuffer

        query_buffer = CollisionQueryBuffer()
        query_buffer.update_buffer_shape(
            robot_spheres.shape,
            self.motion_generator.tensor_args,
            world_cost.world_coll_checker.collision_types,
        )
        with th.no_grad():
            values = world_cost.world_coll_checker.get_sphere_distance(
                robot_spheres,
                query_buffer,
                world_cost.weight,
                world_cost.activation_distance,
                return_loss=world_cost.return_loss,
                sum_collisions=True,
            )
        return values.unsqueeze(-1) if values.ndim == 2 else values

    def _collision_channels(
        self,
        trajectory: th.Tensor,
        *,
        initial_joint_pos: th.Tensor,
    ) -> dict[str, object]:
        """Evaluate world and self collision constraints independently.

        ``check_collisions`` intentionally returns their union.  The diagnostic needs
        the two channels separately so a conservative self-collision model cannot be
        mistaken for a table or scene collision.
        """

        emb_sel = CuRoboEmbodimentSelection.DEFAULT
        joint_state = self._ordered_joint_state_for_curobo(
            trajectory,
            emb_sel=emb_sel,
            locked_joint_positions=initial_joint_pos,
        )
        robot_spheres = self.motion_generator.mg[emb_sel].compute_kinematics(joint_state).robot_spheres
        robot_spheres = robot_spheres.unsqueeze(1)
        rollout = self.motion_generator.mg[emb_sel].rollout_fn
        with th.no_grad():
            world_constraint = rollout.primitive_collision_constraint.forward(robot_spheres).squeeze(1)
            self_constraint = rollout.robot_self_collision_constraint.forward(robot_spheres).squeeze(1)

        world_cost = rollout.primitive_collision_constraint
        world_checker = world_cost.world_coll_checker
        per_sphere_world = self._per_sphere_world_collision(robot_spheres, world_cost=world_cost)
        sphere_to_link = self._sphere_link_names(emb_sel=emb_sel)
        reconstructed_world, world_summary = _summarize_per_sphere_collision(
            per_sphere_world,
            sphere_to_link=sphere_to_link,
        )
        radio_identity = self._radio_collision_mesh_identity(world_checker)
        disabled_meshes = []
        try:
            for mesh_name in radio_identity["collision_mesh_names"]:
                world_checker.enable_obstacle(mesh_name, enable=False)
                disabled_meshes.append(mesh_name)
            per_sphere_non_target = self._per_sphere_world_collision(robot_spheres, world_cost=world_cost)
        finally:
            for mesh_name in disabled_meshes:
                world_checker.enable_obstacle(mesh_name, enable=True)
        non_target_world, non_target_summary = _summarize_per_sphere_collision(
            per_sphere_non_target,
            sphere_to_link=sphere_to_link,
        )
        per_sphere_target = th.clamp(per_sphere_world - per_sphere_non_target, min=0.0)
        target_world, target_summary = _summarize_per_sphere_collision(
            per_sphere_target,
            sphere_to_link=sphere_to_link,
        )
        raw_world = world_constraint.detach().reshape(-1)
        if raw_world.shape != reconstructed_world.shape:
            raise RuntimeError(
                "CuRobo world collision summary shape differs from the per-sphere diagnostic decomposition."
            )
        decomposition_residual = float(th.max(th.abs(raw_world - reconstructed_world)).cpu())
        world_summary["decomposition_max_abs_residual"] = decomposition_residual
        non_target_summary["constraint_score"] = non_target_world.detach().cpu().tolist()
        target_summary.update(
            {
                "constraint_score": target_world.detach().cpu().tolist(),
                "identity": radio_identity,
                "attribution_method": "disable_exact_task_bound_collision_meshes_and_requery",
            }
        )
        if world_summary["detected"] and target_summary["detected"] and not non_target_summary["detected"]:
            attribution = "target_object_only"
        elif world_summary["detected"] and target_summary["detected"] and non_target_summary["detected"]:
            attribution = "target_and_non_target"
        elif world_summary["detected"]:
            attribution = "non_target_only"
        else:
            attribution = "none"
        world_summary["attribution"] = attribution
        return {
            "world_constraint_score": world_constraint.detach().cpu().tolist(),
            "self_constraint_score": self_constraint.detach().cpu().tolist(),
            "world_collision": world_summary,
            "non_target_world_collision": non_target_summary,
            "target_object_collision": target_summary,
            "self_collision": {
                "detected": bool(th.any(self_constraint > 0.0).item()),
                "first_collision_sample": _first_positive_sample(self_constraint),
                "maximum_constraint_score": _max_positive_value(self_constraint),
            },
            "sphere_count": int(robot_spheres.shape[2]),
        }

    def _table_clearance(self, trajectory: th.Tensor) -> dict[str, object]:
        self._ensure_table_checker()
        arm_mg = self.motion_generator.mg[CuRoboEmbodimentSelection.ARM]
        cu_joint_state = self._ordered_joint_state_for_curobo(
            trajectory,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        )
        spheres = arm_mg.compute_kinematics(cu_joint_state).robot_spheres
        indices = self._right_arm_sphere_indices()
        spheres = spheres[:, indices, :].unsqueeze(0)
        from curobo.geom.sdf.world import CollisionQueryBuffer

        if self._table_query_buffer is None:
            self._table_query_buffer = CollisionQueryBuffer()
        self._table_query_buffer.update_buffer_shape(
            spheres.shape,
            self.motion_generator.tensor_args,
            self._table_checker.collision_types,
        )
        distance = self._table_checker.get_sphere_distance(
            spheres,
            self._table_query_buffer,
            self.motion_generator.tensor_args.to_device([1.0]),
            self.motion_generator.tensor_args.to_device([0.0]),
            sum_collisions=False,
            compute_esdf=True,
        )
        # CuRobo ESDF is positive inside the table and negative outside. Its
        # value is for the sphere center, so the sphere radius is subtracted.
        result = _summarize_clearance(
            distance,
            spheres,
            trajectory_samples=int(trajectory.shape[0]),
        )
        result.update(
            {
                "table_collision_mesh_digest": f"sha256:{self._table_mesh_digest}",
            }
        )
        return result

    def _provenance_indices(self) -> tuple[th.Tensor, th.Tensor]:
        try:
            arm_indices = self.robot.arm_control_idx["right"]
            gripper_indices = self.robot.gripper_control_idx["right"]
        except (KeyError, AttributeError) as exc:
            raise RuntimeError("R1Pro robot does not expose right-arm/gripper control indices.") from exc
        if len(arm_indices) != 7 or len(gripper_indices) != 2:
            raise RuntimeError("G013 provenance requires seven right-arm and two right-finger joints.")
        return arm_indices.to(device=self.robot.get_joint_positions().device), gripper_indices.to(
            device=self.robot.get_joint_positions().device
        )

    def _fixed_step_interpolation(self, start: th.Tensor, goal: th.Tensor, steps: int) -> th.Tensor:
        if steps <= 0:
            raise ValueError("Provenance interpolation steps must be positive.")
        fractions = th.linspace(0.0, 1.0, steps + 1, device=start.device)
        return start.unsqueeze(0) + fractions.unsqueeze(1) * (goal - start).unsqueeze(0)

    def _provenance_trajectory(
        self,
        start_q: th.Tensor,
        *,
        branch: Mapping[str, object],
        stage_name: str,
        arm_indices: th.Tensor,
        gripper_indices: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, dict[str, object]]:
        stage = branch["stages"][stage_name]
        waypoints = stage["right_arm_waypoints_rad"]
        steps = stage["interpolation_steps"]
        closed = stage_name in ("lift_1", "lift_2")
        gripper_target = (
            branch["closed_gripper_joint_positions_m"] if closed else branch["open_gripper_joint_positions_m"]
        )
        current = start_q.detach().clone()
        if stage_name == "pregrasp":
            open_target = th.as_tensor(
                branch["open_gripper_joint_positions_m"], device=current.device, dtype=current.dtype
            )
            if float(th.max(th.abs(current[gripper_indices] - open_target))) > 0.002:
                raise RuntimeError("G013 provenance requires an initially open right gripper.")
        trajectory_parts = []
        close_steps = None
        if stage_name == "lift_1":
            close_steps = int(branch["gripper_close_interpolation_steps"])
            close_goal = current.detach().clone()
            close_goal[gripper_indices] = th.as_tensor(
                branch["closed_gripper_joint_positions_m"], device=current.device, dtype=current.dtype
            )
            trajectory_parts.append(self._fixed_step_interpolation(current, close_goal, close_steps))
            current = close_goal
        for waypoint, step_count in zip(waypoints, steps, strict=True):
            goal = current.detach().clone()
            goal[arm_indices] = th.as_tensor(waypoint, device=current.device, dtype=current.dtype)
            goal[gripper_indices] = th.as_tensor(gripper_target, device=current.device, dtype=current.dtype)
            segment = self._fixed_step_interpolation(current, goal, int(step_count))
            trajectory_parts.append(segment if not trajectory_parts else segment[1:])
            current = goal
        trajectory = th.cat(trajectory_parts, dim=0)
        contact_phase_start_sample = None
        if stage_name == "preclose":
            contact_phase_start_sample = sum(int(step) for step in steps[:-1]) + 1
        return (
            current,
            trajectory,
            {
                "source": "g013_frozen_right_arm_joint_path",
                "stage": stage_name,
                "waypoint_count": len(waypoints),
                "interpolation_steps": [int(step) for step in steps],
                "gripper_close_interpolation_steps": close_steps,
                "contact_phase_start_sample": contact_phase_start_sample,
                "locked_segments": ["base", "trunk", "arm_left", "gripper_left"],
            },
        )

    def _right_eef_pose_for_joint_positions(self, joint_positions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        arm_mg = self.motion_generator.mg[CuRoboEmbodimentSelection.ARM]
        joint_state = self._ordered_joint_state_for_curobo(
            joint_positions,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        )
        link_pose = arm_mg.compute_kinematics(joint_state).link_poses[self.robot.eef_link_names["right"]]
        return (
            link_pose.position.reshape(-1, 3)[-1],
            link_pose.quaternion.reshape(-1, 4)[-1][[1, 2, 3, 0]],
        )

    def _right_eef_target_comparison(
        self,
        joint_positions: th.Tensor,
        *,
        target_pose: Mapping[str, object],
        stage_name: str,
    ) -> dict[str, object]:
        target_position, target_orientation = self._target_pose_for_curobo(target_pose, stage_name)
        actual_position, actual_orientation = self._right_eef_pose_for_joint_positions(joint_positions)
        base_link_name = self.motion_generator.base_link[CuRoboEmbodimentSelection.ARM]
        return {
            "kinematic_base_link": base_link_name,
            "actual_pose_in_kinematic_base": {
                "translation_m": actual_position.detach().cpu().tolist(),
                "quaternion_xyzw": actual_orientation.detach().cpu().tolist(),
            },
            "target_pose_in_kinematic_base": {
                "translation_m": target_position.detach().cpu().tolist(),
                "quaternion_xyzw": target_orientation.detach().cpu().tolist(),
            },
            "residual": _pose_residual(actual_position, actual_orientation, target_position, target_orientation),
        }

    def _candidate_joint_state_diagnostic(self, joint_positions: th.Tensor) -> dict[str, object]:
        names = self.motion_generator.robot_joint_names
        if len(names) != int(joint_positions.numel()):
            raise RuntimeError("Candidate joint vector does not match the simulator joint-name ordering.")
        arm_mg = self.motion_generator.mg[CuRoboEmbodimentSelection.ARM]
        full_state = self._full_joint_state_for_curobo(joint_positions)
        self.motion_generator.update_locked_joints(full_state, CuRoboEmbodimentSelection.ARM)
        locked_state = arm_mg.kinematics.kinematics_config.lock_jointstate

        actual_position, actual_orientation = self._right_eef_pose_for_joint_positions(joint_positions)
        base_link_name = self.motion_generator.base_link[CuRoboEmbodimentSelection.ARM]
        base_position, base_orientation = self.robot.links[base_link_name].get_position_orientation()
        simulator_position, simulator_orientation = self.robot.eef_links["right"].get_position_orientation()
        simulator_position, simulator_orientation = T.pose_transform(
            *T.invert_pose_transform(base_position, base_orientation),
            simulator_position,
            simulator_orientation,
        )
        return {
            "joint_positions_by_name": dict(zip(names, joint_positions.detach().cpu().tolist(), strict=True)),
            "curobo_active_joint_names": list(arm_mg.kinematics.joint_names),
            "curobo_locked_joint_positions_by_name": dict(
                zip(
                    locked_state.joint_names,
                    locked_state.position.detach().cpu().tolist(),
                    strict=True,
                )
            ),
            "kinematic_base_link": base_link_name,
            "kinematic_base_pose_world": {
                "translation_m": base_position.detach().cpu().tolist(),
                "quaternion_xyzw": base_orientation.detach().cpu().tolist(),
            },
            "right_eef_pose_in_kinematic_base": {
                "curobo_fk": {
                    "translation_m": actual_position.detach().cpu().tolist(),
                    "quaternion_xyzw": actual_orientation.detach().cpu().tolist(),
                },
                "simulator": {
                    "translation_m": simulator_position.detach().cpu().tolist(),
                    "quaternion_xyzw": simulator_orientation.detach().cpu().tolist(),
                },
                "residual": _pose_residual(
                    actual_position,
                    actual_orientation,
                    simulator_position,
                    simulator_orientation,
                ),
            },
        }

    def _provenance_branch(
        self,
        *,
        candidate_base_pose: Mapping[str, object],
        target_poses: Mapping[str, Mapping[str, object]],
        branch: Mapping[str, object],
    ) -> dict[str, object]:
        arm_indices, gripper_indices = self._provenance_indices()
        start_q = self._candidate_joint_state_for_curobo(candidate_base_pose)
        candidate_joint_state_diagnostic = self._candidate_joint_state_diagnostic(start_q)
        self.motion_generator.update_obstacles()
        stages: dict[str, object] = {}
        for stage_name in CORRIDOR_STAGE_NAMES:
            try:
                goal_q, trajectory, provenance = self._provenance_trajectory(
                    start_q,
                    branch=branch,
                    stage_name=stage_name,
                    arm_indices=arm_indices,
                    gripper_indices=gripper_indices,
                )
            except (RuntimeError, ValueError) as exc:
                stages[stage_name] = {
                    "ik": {"status": "available", "feasible": False, "reason": str(exc)},
                    "collision": {"status": "not_evaluated", "collision_free": False, "reason": "provenance_failed"},
                    "clearance": {"status": "not_evaluated", "reason": "provenance_failed"},
                    "provenance": {"source": "g013_frozen_right_arm_joint_path", "branch_id": branch["branch_id"]},
                }
                break
            target_comparison = self._right_eef_target_comparison(
                goal_q,
                target_pose=target_poses[stage_name],
                stage_name=stage_name,
            )
            target_residual = target_comparison["residual"]
            target_matches = (
                target_residual["translation_m"] <= PROVENANCE_TARGET_TRANSLATION_TOLERANCE_M
                and target_residual["orientation_rad"] <= PROVENANCE_TARGET_ORIENTATION_TOLERANCE_RAD
            )
            if not target_matches:
                stages[stage_name] = {
                    "ik": {
                        "status": "available",
                        "feasible": False,
                        "reason": "provenance_endpoint_misses_semantic_target",
                        "target_residual": target_residual,
                        "kinematic_comparison": target_comparison,
                    },
                    "collision": {"status": "not_evaluated", "collision_free": False, "reason": "ik_failed"},
                    "clearance": {"status": "not_evaluated", "reason": "ik_failed"},
                    "provenance": {**provenance, "branch_id": branch["branch_id"]},
                }
                break
            collision_channels = self._collision_channels(trajectory, initial_joint_pos=start_q)
            world_collision = collision_channels["world_collision"]
            self_collision = collision_channels["self_collision"]
            collision = th.as_tensor(collision_channels["world_constraint_score"]) > 0.0
            collision = collision | (th.as_tensor(collision_channels["self_constraint_score"]) > 0.0)
            collision_free = not bool(th.any(collision).item())
            collision_policy = _stage_collision_policy(
                stage_name,
                collision_channels,
                contact_phase_start_sample=provenance["contact_phase_start_sample"],
            )
            collision_safe = collision_policy["collision_safe_for_stage"] is True
            clearance = self._table_clearance(trajectory)
            stages[stage_name] = {
                "ik": {
                    "status": "available",
                    "feasible": True,
                    "solver": "g013_frozen_right_arm_joint_path",
                    "solution_joint_positions_rad": goal_q.detach().cpu().tolist(),
                    "target_residual": target_residual,
                    "kinematic_comparison": target_comparison,
                },
                "collision": {
                    "status": "available",
                    "collision_free": collision_free,
                    "collision_safe_for_stage": collision_safe,
                    "target_contact_policy": collision_policy,
                    "sample_count": int(trajectory.shape[0]),
                    "first_collision_sample": _first_positive_sample(collision_channels["world_constraint_score"]),
                    "model": "g013_segmented_joint_path_checked_at_each_sample",
                    "world_collision": world_collision,
                    "self_collision": self_collision,
                    "collision_channels": collision_channels,
                },
                "clearance": clearance,
                "provenance": {**provenance, "branch_id": branch["branch_id"]},
            }
            if not collision_safe:
                break
            start_q = goal_q
        for stage_name in CORRIDOR_STAGE_NAMES:
            stages.setdefault(
                stage_name,
                {
                    "ik": {"status": "not_evaluated", "feasible": False, "reason": "prior_stage_failed"},
                    "collision": {"status": "not_evaluated", "collision_free": False, "reason": "prior_stage_failed"},
                    "clearance": {"status": "not_evaluated", "reason": "prior_stage_failed"},
                    "provenance": {"source": "g013_frozen_right_arm_joint_path", "branch_id": branch["branch_id"]},
                },
            )
        return {
            "branch_id": branch["branch_id"],
            "source_run": branch["source_run"],
            "candidate_joint_state_diagnostic": candidate_joint_state_diagnostic,
            "joint_provenance": {
                "source": "g013_frozen_right_arm_joint_path",
                "source_generation": "013",
                "source_fixture_environment_step": 550,
                "branch_id": branch["branch_id"],
                "source_run": branch["source_run"],
                "arm": "right",
                "locked_segments": ["base", "trunk", "arm_left", "gripper_left"],
            },
            "stages": stages,
        }

    def evaluate_corridor(
        self,
        *,
        candidate_base_pose: Mapping[str, object],
        target_poses: Mapping[str, Mapping[str, object]],
        joint_provenance: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if tuple(target_poses) != CORRIDOR_STAGE_NAMES:
            raise ValueError(f"Target poses must contain {CORRIDOR_STAGE_NAMES} in order.")
        if joint_provenance is not None:
            branches = joint_provenance.get("branches")
            if not isinstance(branches, list) or not branches:
                raise ValueError("G013 joint provenance must contain a non-empty branches list.")
            evaluated = [
                self._provenance_branch(
                    candidate_base_pose=candidate_base_pose,
                    target_poses=target_poses,
                    branch=branch,
                )
                for branch in branches
            ]
            return {
                "schema_version": 1,
                "status": "available",
                "candidate_base_pose_digest": _pose_digest(
                    *_as_pose(
                        candidate_base_pose,
                        parent="simulator_world",
                        child="robot_base_footprint",
                    )
                ),
                "branch_mode": "validated_g013_joint_provenance",
                "branches": evaluated,
                "stages": evaluated[0]["stages"],
                "backend": self.capabilities(),
            }
        self._reset_ik_seed_generator()
        candidate_q = self._candidate_joint_state_for_curobo(candidate_base_pose)
        self.motion_generator.update_obstacles()
        current_left_position, current_left_orientation = self._left_hold_pose(candidate_base_pose)
        start_q = candidate_q
        stages: dict[str, object] = {}
        for stage in CORRIDOR_STAGE_NAMES:
            target_position, target_orientation = self._target_pose(target_poses[stage], stage)
            target_pos = {
                self.motion_generator.ee_link[CuRoboEmbodimentSelection.ARM]: th.stack([current_left_position]),
                self.robot.eef_link_names["right"]: th.stack([target_position]),
            }
            target_quat = {
                self.motion_generator.ee_link[CuRoboEmbodimentSelection.ARM]: th.stack([current_left_orientation]),
                self.robot.eef_link_names["right"]: th.stack([target_orientation]),
            }
            successes, joint_states = self.motion_generator.compute_trajectories(
                target_pos=target_pos,
                target_quat=target_quat,
                initial_joint_pos=start_q,
                is_local=False,
                max_attempts=2,
                timeout=5.0,
                ik_fail_return=5,
                enable_finetune_trajopt=False,
                finetune_attempts=0,
                success_ratio=1.0,
                skip_obstacle_update=True,
                ik_only=True,
                ik_world_collision_check=False,
                emb_sel=CuRoboEmbodimentSelection.ARM,
            )
            if not bool(successes[0].item()):
                stages[stage] = {
                    "ik": {"status": "available", "feasible": False, "reason": "no_whole_arm_ik_solution"},
                    "collision": {"status": "not_evaluated", "collision_free": False, "reason": "ik_failed"},
                    "clearance": {"status": "not_evaluated", "reason": "ik_failed"},
                }
                break
            goal_q = self._ik_goal_joint_positions(joint_states[0], expected_shape=start_q.shape)
            trajectory = self._interpolated(start_q, goal_q)
            collision_channels = self._collision_channels(trajectory, initial_joint_pos=start_q)
            world_collision = collision_channels["world_collision"]
            self_collision = collision_channels["self_collision"]
            collision = th.as_tensor(collision_channels["world_constraint_score"]) > 0.0
            collision = collision | (th.as_tensor(collision_channels["self_constraint_score"]) > 0.0)
            collision_free = not bool(th.any(collision).item())
            collision_result = {
                "status": "available",
                "collision_free": collision_free,
                "sample_count": int(trajectory.shape[0]),
                "first_collision_sample": next(
                    (index for index, value in enumerate(collision.detach().cpu().tolist()) if value),
                    None,
                ),
                "model": "linear_joint_interpolation_checked_at_each_sample",
                "world_collision": world_collision,
                "self_collision": self_collision,
                "collision_channels": collision_channels,
            }
            clearance = self._table_clearance(trajectory)
            stages[stage] = {
                "ik": _summarize_ik_solution(goal_q),
                "collision": collision_result,
                "clearance": clearance,
            }
            if not collision_free:
                break
            start_q = goal_q
        for stage in CORRIDOR_STAGE_NAMES:
            stages.setdefault(
                stage,
                {
                    "ik": {"status": "not_evaluated", "feasible": False, "reason": "prior_stage_failed"},
                    "collision": {"status": "not_evaluated", "collision_free": False, "reason": "prior_stage_failed"},
                    "clearance": {"status": "not_evaluated", "reason": "prior_stage_failed"},
                },
            )
        return {
            "schema_version": 1,
            "status": "available",
            "candidate_base_pose_digest": _pose_digest(
                *_as_pose(
                    candidate_base_pose,
                    parent="simulator_world",
                    child="robot_base_footprint",
                )
            ),
            "stages": stages,
            "backend": self.capabilities(),
        }
