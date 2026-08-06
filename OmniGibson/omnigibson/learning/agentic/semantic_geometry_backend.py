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


class RadioSemanticGeometryBackend:
    """CuRobo-backed geometry oracle with no simulator mutation."""

    def __init__(self, env: object, robot: object, *, device: str | None = None) -> None:
        if len(og.sim.scenes) != 1:
            raise RuntimeError("Radio geometry backend requires exactly one simulator scene.")
        self.env = env
        self.robot = robot
        self.device = str(device or og.sim.device)
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
        return _as_pose(value, parent="simulator_world", child=f"right_eef_{stage}")

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
        return T.pose_transform(candidate_position, candidate_orientation, *base_to_left)

    def _interpolated(self, start: th.Tensor, goal: th.Tensor) -> th.Tensor:
        intervals = max(1, math.ceil(float(th.max(th.abs(goal - start))) / INTERPOLATION_MAX_JOINT_DELTA_RAD))
        fractions = th.linspace(0.0, 1.0, intervals + 1, device=start.device)
        return start.unsqueeze(0) + fractions.unsqueeze(1) * (goal - start).unsqueeze(0)

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

    def _table_clearance(self, trajectory: th.Tensor) -> dict[str, object]:
        self._ensure_table_checker()
        arm_mg = self.motion_generator.mg[CuRoboEmbodimentSelection.ARM]
        cu_joint_state = lazy.curobo.types.state.JointState(
            position=self.motion_generator.tensor_args.to_device(trajectory),
            joint_names=self.motion_generator.robot_joint_names,
        ).get_ordered_joint_state(arm_mg.kinematics.joint_names)
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

    def evaluate_corridor(
        self,
        *,
        candidate_base_pose: Mapping[str, object],
        target_poses: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        if tuple(target_poses) != CORRIDOR_STAGE_NAMES:
            raise ValueError(f"Target poses must contain {CORRIDOR_STAGE_NAMES} in order.")
        candidate_q = self._candidate_joint_state(candidate_base_pose)
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
            goal_q = self.motion_generator.path_to_joint_trajectory(
                joint_states[0], get_full_js=True, emb_sel=CuRoboEmbodimentSelection.ARM
            )[-1]
            trajectory = self._interpolated(start_q, goal_q)
            collision = self.motion_generator.check_collisions(
                trajectory,
                initial_joint_pos=start_q,
                self_collision_check=True,
                skip_obstacle_update=True,
            )
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
            }
            clearance = self._table_clearance(trajectory)
            stages[stage] = {
                "ik": {
                    "status": "available",
                    "feasible": True,
                    "solution_digest": "sha256:"
                    + hashlib.sha256(goal_q.detach().cpu().numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
                },
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
