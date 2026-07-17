import os

import torch as th
from omnigibson.object_states import (
    ToggledOn,
    OnTop,
    Inside,
    Open,
    NextTo,
    Under,
    AttachedTo,
    Touching,
    OnFire,
    Covered,
    Contains,
    Filled,
    Cooked,
    Frozen,
    IsGrasping,
)

ROBOT_OBJECT_DISTANCE_THRESHOLD = 0.5  # meters
PROGRESS_OPEN_FRACTION_THRESHOLD = 0.5
MOVING_BOXES_DOOR_DISTANCE_THRESHOLD = 1.2
MOVING_BOXES_DOOR_OPEN_FRACTION_THRESHOLD = 0.85
MOVING_BOXES_CONTAINER_EEF_THRESHOLD = 0.4
MOVING_BOXES_GARAGE_PLACE_X = -2.44
MOVING_BOXES_GARAGE_PLACE_Y = 3.40
MOVING_BOXES_GARAGE_PLACE_THRESHOLD = 1.25
HALLOWEEN_CALDRON_LIFT_CLEARANCE_THRESHOLD = 0.08
HALLOWEEN_PICKUP_EEF_THRESHOLD = 0.5
HALLOWEEN_SUPPORT_EEF_THRESHOLD = 0.5
SHOES_PICKUP_EEF_THRESHOLD = 0.2
SHOES_HALLSTAND_THRESHOLD = 1.0
BUGS_PICKUP_EEF_THRESHOLD = 0.4
BUGS_SPRAY_EEF_THRESHOLD = 0.5
POPCORN_OPEN_EEF_THRESHOLD = 1.0
POPCORN_OPEN_FRACTION_THRESHOLD = 0.85


def _near_profile():
    return os.environ.get("BEHAVIOR_TASK_PROGRESS_NEAR_PROFILE", "current").strip().lower()


def _activity_instance_id(env):
    task = getattr(env, "task", None)
    instance_id = getattr(task, "activity_instance_id", None)
    if instance_id is None:
        return None
    return int(instance_id)


def _moving_boxes_door_threshold():
    return float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_BOXES_DOOR_THRESHOLD",
            str(MOVING_BOXES_DOOR_DISTANCE_THRESHOLD),
        )
    )


def _moving_boxes_door_open_threshold():
    return float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_BOXES_DOOR_OPEN_THRESHOLD",
            str(MOVING_BOXES_DOOR_OPEN_FRACTION_THRESHOLD),
        )
    )


def _moving_boxes_garage_place_point():
    x = os.environ.get(
        "BEHAVIOR_TASK_PROGRESS_BOXES_GARAGE_PLACE_X",
        str(MOVING_BOXES_GARAGE_PLACE_X),
    )
    y = os.environ.get(
        "BEHAVIOR_TASK_PROGRESS_BOXES_GARAGE_PLACE_Y",
        str(MOVING_BOXES_GARAGE_PLACE_Y),
    )
    return th.tensor([float(x), float(y)])


def _moving_boxes_garage_place_threshold():
    return float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_BOXES_GARAGE_PLACE_THRESHOLD",
            str(MOVING_BOXES_GARAGE_PLACE_THRESHOLD),
        )
    )


def _halloween_caldron_lift_clearance_threshold():
    return float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_HALLOWEEN_CALDRON_LIFT_CLEARANCE",
            str(HALLOWEEN_CALDRON_LIFT_CLEARANCE_THRESHOLD),
        )
    )


def _halloween_pickup_near_spec(obj_key):
    threshold = os.environ.get(
        "BEHAVIOR_TASK_PROGRESS_HALLOWEEN_PICKUP_EEF_THRESHOLD",
        str(HALLOWEEN_PICKUP_EEF_THRESHOLD),
    )
    if threshold.strip().lower() in {"current", "any"}:
        return ("near", "agent.n.01_1", obj_key)
    return ("near_eef_threshold", "agent.n.01_1", obj_key, float(threshold))


def _halloween_support_near_spec(obj_key):
    eef_threshold = os.environ.get("BEHAVIOR_TASK_PROGRESS_HALLOWEEN_SUPPORT_EEF_THRESHOLD")
    if eef_threshold is not None:
        if eef_threshold.strip().lower() in {"current", "any"}:
            return ("near", "agent.n.01_1", obj_key)
        return ("near_eef_threshold", "agent.n.01_1", obj_key, float(eef_threshold))
    base_threshold = os.environ.get("BEHAVIOR_TASK_PROGRESS_HALLOWEEN_SUPPORT_BASE_THRESHOLD")
    if base_threshold is not None:
        return ("near_base_threshold", "agent.n.01_1", obj_key, float(base_threshold))
    return ("near_eef_threshold", "agent.n.01_1", obj_key, HALLOWEEN_SUPPORT_EEF_THRESHOLD)


def _shoes_hallstand_near_threshold():
    return float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_SHOES_HALLSTAND_THRESHOLD",
            str(SHOES_HALLSTAND_THRESHOLD),
        )
    )


def _shoes_pickup_near_spec(obj_key):
    threshold = float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_SHOES_PICKUP_EEF_THRESHOLD",
            str(SHOES_PICKUP_EEF_THRESHOLD),
        )
    )
    return ("near_eef_threshold", "agent.n.01_1", obj_key, threshold)


def _moving_boxes_container_near_spec(obj_key):
    threshold = float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_BOXES_PICKUP_EEF_THRESHOLD",
            str(MOVING_BOXES_CONTAINER_EEF_THRESHOLD),
        )
    )
    return ("near_eef_threshold", "agent.n.01_1", obj_key, threshold)


def _optional_task_eef_near_spec(env_name, obj_key, default_threshold=None):
    threshold = os.environ.get(env_name)
    if threshold is None:
        if default_threshold is None:
            return ("near", "agent.n.01_1", obj_key)
        threshold = default_threshold
    if str(threshold).strip().lower() in {"current", "any"}:
        return ("near", "agent.n.01_1", obj_key)
    return ("near_eef_threshold", "agent.n.01_1", obj_key, float(threshold))


def _wood_plywood_near_spec(obj_key):
    return _optional_task_eef_near_spec(
        "BEHAVIOR_TASK_PROGRESS_WOOD_PICKUP_EEF_THRESHOLD",
        obj_key,
    )


def _wood_door_near_spec():
    threshold = os.environ.get("BEHAVIOR_TASK_PROGRESS_WOOD_DOOR_BASE_THRESHOLD")
    if threshold is None:
        return ("near", "agent.n.01_1", "door_vudhlc_1")
    return ("near_base_threshold", "agent.n.01_1", "door_vudhlc_1", float(threshold))


def _wood_door_open_threshold():
    return _task_open_fraction_threshold("BEHAVIOR_TASK_PROGRESS_WOOD_DOOR_OPEN_FRACTION")


def _bugs_plant_near_spec(obj_key):
    base_threshold = os.environ.get("BEHAVIOR_TASK_PROGRESS_BUGS_SPRAY_BASE_THRESHOLD")
    if base_threshold is not None:
        return ("near_base_threshold", "agent.n.01_1", obj_key, float(base_threshold))
    return _optional_task_eef_near_spec(
        "BEHAVIOR_TASK_PROGRESS_BUGS_SPRAY_EEF_THRESHOLD",
        obj_key,
        BUGS_SPRAY_EEF_THRESHOLD,
    )


def _popcorn_open_fraction_threshold():
    return float(
        os.environ.get(
            "BEHAVIOR_TASK_PROGRESS_POPCORN_OPEN_FRACTION_THRESHOLD",
            str(POPCORN_OPEN_FRACTION_THRESHOLD),
        )
    )


def _task_open_fraction_threshold(env_name):
    return float(os.environ.get(env_name, str(PROGRESS_OPEN_FRACTION_THRESHOLD)))


def _debug_task_progress_open():
    return bool(os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_OPEN"))


def _debug_composite_progress(name, spec, component_results):
    requested = {
        item.strip()
        for item in os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_COMPOSITE", "").split(",")
        if item.strip()
    }
    if name not in requested:
        return

    current = tuple(bool(result) for result in component_results)
    previous = getattr(_debug_composite_progress, "_previous", {})
    if previous.get(name) == current:
        return
    previous[name] = current
    setattr(_debug_composite_progress, "_previous", previous)
    print(
        "[task_progress_debug_composite] "
        f"name={name} result={all(current)} components={list(zip(spec[1], current))}",
        flush=True,
    )


def _debug_moving_boxes_object_poses(env):
    if not bool(os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_BOX_POSES")):
        return

    counter = getattr(_debug_moving_boxes_object_poses, "_counter", 0)
    setattr(_debug_moving_boxes_object_poses, "_counter", counter + 1)
    interval = max(1, int(os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_BOX_POSES_INTERVAL", "10")))

    container_specs = [
        ("container_1", "storage_container.n.01_1"),
        ("container_2", "storage_container.n.01_2"),
    ]
    on_garage = {
        label: _check_state_spec(env, ("state", key, OnTop, "floor.n.01_2", True))
        for label, key in container_specs
    }
    if counter % interval != 0 and not any(on_garage.values()):
        return

    scene = getattr(env, "scene", None)
    seg_map = getattr(scene, "seg_map", None) if scene is not None else None

    def pose_record(key):
        entity = _resolve_object(env, key)
        if not entity.exists:
            return {"exists": False}
        pos = entity.unwrapped.get_position_orientation()[0]
        xy = pos[:2]
        room = None
        if seg_map is not None:
            try:
                room = seg_map.get_room_instance_by_point(xy)
            except Exception as exc:
                room = f"error:{type(exc).__name__}"
        return {
            "exists": True,
            "pos": [float(v) for v in pos.tolist()],
            "room": room,
        }

    robot_entity = _resolve_object(env, "agent.n.01_1")
    robot_xy = None
    if robot_entity.exists:
        robot_xy = _robot_base_link(robot_entity.unwrapped).get_position_orientation()[0][:2]

    records = {
        "robot_xy": None if robot_xy is None else [float(v) for v in robot_xy.tolist()],
        "garage_floor": pose_record("floor.n.01_2"),
    }
    for label, key in container_specs:
        records[label] = pose_record(key)
        records[label]["on_garage_floor"] = on_garage[label]
        records[label]["grasped"] = _check_grasping_spec(env, ("grasping", "agent.n.01_1", key, True))

    print(
        "[task_progress_debug_boxes] "
        f"counter={counter} records={records}",
        flush=True,
    )


def _near_mode_and_thresholds(check_type, spec):
    if check_type == "near_eef_threshold":
        return "eef", float(spec[3]), None
    if check_type == "near_base_threshold":
        return "base", float(spec[3]), None
    if check_type in {"near_threshold", "near_category_threshold"}:
        return "any", float(spec[3]), None

    profile = _near_profile()
    if profile in {"", "current", "any_0.5", "any050"}:
        return "any", ROBOT_OBJECT_DISTANCE_THRESHOLD, None
    if profile in {"base_0.35", "base035"}:
        return "base", 0.35, None
    if profile in {"base_0.45", "base045"}:
        return "base", 0.45, None
    if profile in {"base_0.55", "base055"}:
        return "base", 0.55, None
    if profile in {"eef_0.10", "eef010"}:
        return "eef", 0.10, None
    if profile in {"eef_0.20", "eef020"}:
        return "eef", 0.20, None
    if profile in {"eef_0.30", "eef030"}:
        return "eef", 0.30, None
    if profile in {"eef_0.25", "eef025"}:
        return "eef", 0.25, None
    if profile in {"eef_0.35", "eef035"}:
        return "eef", 0.35, None
    if profile in {"eef_0.40", "eef040"}:
        return "eef", 0.40, None
    if profile in {"eef_0.50", "eef050"}:
        return "eef", 0.50, None
    if profile in {"base045_or_eef030", "base_0.45_or_eef_0.30"}:
        return "base_or_eef", 0.45, 0.30
    if profile in {"base045_or_eef035", "base_0.45_or_eef_0.35"}:
        return "base_or_eef", 0.45, 0.35
    if profile == "demo_calibrated":
        base_threshold = float(os.environ.get("BEHAVIOR_TASK_PROGRESS_NEAR_DEMO_BASE_THRESHOLD", "0.45"))
        eef_threshold = float(os.environ.get("BEHAVIOR_TASK_PROGRESS_NEAR_DEMO_EEF_THRESHOLD", "0.35"))
        return "base_or_eef", base_threshold, eef_threshold
    raise ValueError(
        "Unknown BEHAVIOR_TASK_PROGRESS_NEAR_PROFILE="
        f"{profile!r}. Expected current, base_0.35, base_0.45, base_0.55, "
        "eef_0.10, eef_0.20, eef_0.25, eef_0.30, eef_0.35, eef_0.40, eef_0.50, "
        "base045_or_eef030, base045_or_eef035, or demo_calibrated."
    )


def _robot_base_link(robot):
    return robot.base_footprint_link


def _object_room_names(obj_entity):
    rooms = getattr(obj_entity.unwrapped, "in_rooms", None)
    if rooms is None:
        return set()
    if isinstance(rooms, str):
        return {rooms}
    return set(rooms)


class _ProgressObject:
    """Thin adapter for scene objects that are not present in task.object_scope."""

    def __init__(self, obj):
        self._obj = obj

    @property
    def exists(self):
        return self._obj is not None

    @property
    def unwrapped(self):
        return getattr(self._obj, "unwrapped", self._obj)


def _category_candidates_from_key(key):
    # Convert BDDL-style instance keys such as toilet.n.01_1 to OG category
    # candidates such as toilet.
    stem, _, suffix = key.rpartition("_")
    if not suffix.isdigit():
        stem = key
    if ".n." in stem:
        stem = stem.split(".n.", 1)[0]
    return [stem] if stem else []


def _instance_index_from_key(key):
    _, _, suffix = key.rpartition("_")
    return int(suffix) - 1 if suffix.isdigit() else 0


def _resolve_object(env, key):
    objs = env.task.object_scope
    if key in objs:
        return objs[key]

    scene = getattr(env, "scene", None)
    if scene is not None:
        obj = scene.object_registry("name", key, None)
        if obj is not None:
            return _ProgressObject(obj)

        for category in _category_candidates_from_key(key):
            matches = list(scene.object_registry("category", category, default_val=[]))
            if matches:
                matches = sorted(matches, key=lambda obj: getattr(obj, "name", ""))
                idx = min(_instance_index_from_key(key), len(matches) - 1)
                return _ProgressObject(matches[idx])

    raise KeyError(f"Could not resolve object key for task progress: {key}")


def _resolve_category_objects(env, category_key):
    objs = env.task.object_scope
    results = []
    for inst, category in env.task.object_instance_to_category.items():
        if category == category_key and inst in objs:
            results.append(objs[inst])
    if results:
        return results

    scene = getattr(env, "scene", None)
    if scene is not None:
        for category in _category_candidates_from_key(category_key):
            for obj in scene.object_registry("category", category, default_val=[]):
                results.append(_ProgressObject(obj))

    return results


def _object_open_fraction(obj):
    open_state = obj.unwrapped.states[Open]
    if open_state.relevant_joints_info is None:
        return 1.0 if open_state.get_value() else 0.0
    both_sides, relevant_joints, joint_directions = open_state.relevant_joints_info
    if not relevant_joints:
        return 0.0

    sides = [1, -1] if both_sides else [1]
    max_fraction = 0.0
    for side in sides:
        for joint, joint_direction in zip(relevant_joints, joint_directions):
            direction = joint_direction * side
            closed_end = joint.lower_limit if direction == 1 else joint.upper_limit
            open_end = joint.upper_limit if direction == 1 else joint.lower_limit
            denom = open_end - closed_end
            if abs(denom) < 1e-6:
                continue
            fraction = (joint.get_state()[0] - closed_end) / denom
            fraction = th.as_tensor(fraction).clamp(0.0, 1.0).item()
            max_fraction = max(max_fraction, float(fraction))
    return max_fraction


def _check_state_spec(env, spec):
    objs = env.task.object_scope
    if len(spec) == 4 and isinstance(spec[3], bool):
        obj = _resolve_object(env, spec[1])
        return obj.exists and obj.unwrapped.states[spec[2]].get_value() == spec[3]
    if len(spec) == 5:
        if spec[3] not in objs:
            # Special case: e.g. category "tree.n.01" is not in object scope.
            all_instances = _resolve_category_objects(env, spec[3])
            assert len(all_instances) > 0, f"Could not find any instances for category {spec[3]}"
            obj1 = _resolve_object(env, spec[1])
            matches = [
                obj1.exists
                and obj2.exists
                and obj1.unwrapped.states[spec[2]].get_value(obj2.unwrapped) == spec[4]
                for obj2 in all_instances
            ]
            return any(matches) if spec[4] else all(matches)
        obj1, obj2 = _resolve_object(env, spec[1]), _resolve_object(env, spec[3])
        return (
            obj1.exists
            and obj2.exists
            and obj1.unwrapped.states[spec[2]].get_value(obj2.unwrapped) == spec[4]
        )
    raise ValueError(f"Invalid state spec: {spec}")


def _check_grasping_spec(env, spec):
    robot, obj = _resolve_object(env, spec[1]), _resolve_object(env, spec[2])
    return (
        robot.exists
        and obj.exists
        and robot.unwrapped.states[IsGrasping].get_value(obj.unwrapped) == spec[3]
    )


def _check_arm_grasping_spec(env, spec):
    robot, obj = _resolve_object(env, spec[1]), _resolve_object(env, spec[2])
    if not (robot.exists and obj.exists):
        return False
    result = robot.unwrapped.is_grasping(arm=spec[3], candidate_obj=obj.unwrapped)
    return (int(result) == 1) == spec[4]


def _check_aabb_bottom_above_spec(env, spec):
    obj = _resolve_object(env, spec[1])
    reference = _resolve_object(env, spec[2])
    if not (obj.exists and reference.exists):
        return False
    obj_min, _ = obj.unwrapped.aabb
    _, reference_max = reference.unwrapped.aabb
    clearance = float((obj_min[2] - reference_max[2]).item())
    result = clearance >= float(spec[3])
    if bool(os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_AABB")):
        print(
            "[task_progress_debug_aabb] "
            f"obj={spec[1]} reference={spec[2]} clearance={clearance} threshold={float(spec[3])} result={result}",
            flush=True,
        )
    return result


def _check_composite_spec(env, spec):
    check_type = spec[0]
    if check_type == "state":
        return _check_state_spec(env, spec)
    if check_type == "grasping":
        return _check_grasping_spec(env, spec)
    if check_type == "grasping_arm":
        return _check_arm_grasping_spec(env, spec)
    if check_type == "aabb_bottom_above":
        return _check_aabb_bottom_above_spec(env, spec)
    if check_type == "all_state":
        return all(_check_composite_spec(env, state_spec) for state_spec in spec[1])
    if check_type == "any_state":
        return any(_check_composite_spec(env, state_spec) for state_spec in spec[1])
    raise ValueError(f"Invalid composite progress spec: {spec}")


def check_progress(env, check_specs):
    """Generic progress checker using declarative specs."""
    results = {}

    for name, spec in check_specs.items():
        check_type = spec[0]

        if check_type in {"near", "near_threshold", "near_eef_threshold", "near_base_threshold", "near_category_threshold"}:
            # ("near", robot_key, obj_key) - robot should always be first.
            # ("near_threshold", robot_key, obj_key, threshold) allows task-local stricter navigation gates.
            # ("near_eef_threshold", robot_key, obj_key, threshold) gates on end-effector distance only.
            # ("near_base_threshold", robot_key, obj_key, threshold) ignores arms/eef for navigation gates.
            # ("near_category_threshold", robot_key, category_key, threshold) accepts any category instance.
            robot_entity = _resolve_object(env, spec[1])
            obj_entities = (
                _resolve_category_objects(env, spec[2])
                if check_type == "near_category_threshold"
                else [_resolve_object(env, spec[2])]
            )
            obj_entities = [obj_entity for obj_entity in obj_entities if obj_entity.exists]
            if not (robot_entity.exists and obj_entities):
                results[name] = False
                continue
            robot = robot_entity.unwrapped
            near_mode, threshold, eef_threshold = _near_mode_and_thresholds(check_type, spec)
            robot_base_link = _robot_base_link(robot)

            # Get all robot links to check: navigation base link and all eef_links
            robot_links_to_check = (
                [robot_base_link]
                if near_mode == "base"
                else list(robot.eef_links.values())
                if near_mode == "eef"
                else [robot_base_link] + list(robot.eef_links.values())
            )
            debug_near = bool(os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_NEAR"))
            diagnostic_robot_links = [robot_base_link] + list(robot.eef_links.values()) if debug_near else robot_links_to_check

            # Get all object links
            candidate_obj_links = []
            for obj_entity in obj_entities:
                obj = obj_entity.unwrapped
                candidate_obj_links.extend((getattr(obj, "name", ""), obj_link_name, obj_link) for obj_link_name, obj_link in obj.links.items())

            # Check minimum distance between any robot link and any object link
            is_near = False
            base_min_dist = None
            eef_min_dist = None
            all_min_dist = None
            nearest_obj_name = None
            nearest_obj_link_name = None
            nearest_robot_link_name = None
            base_nearest_obj_link_name = None
            eef_nearest_obj_link_name = None
            for robot_link in robot_links_to_check:
                robot_pos = robot_link.get_position_orientation()[0]
                robot_link_name = "base" if robot_link is robot_base_link else getattr(robot_link, "name", "eef")
                for obj_name, obj_link_name, obj_link in candidate_obj_links:
                    obj_pos = obj_link.get_position_orientation()[0]
                    # Only consider x and y coordinates (horizontal distance)
                    dist = th.linalg.norm(robot_pos[:2] - obj_pos[:2])
                    dist_float = float(dist.item())
                    if all_min_dist is None or dist_float < all_min_dist:
                        nearest_obj_name = obj_name
                        nearest_obj_link_name = obj_link_name
                        nearest_robot_link_name = robot_link_name
                    if robot_link is robot_base_link:
                        if base_min_dist is None or dist_float < base_min_dist:
                            base_min_dist = dist_float
                            base_nearest_obj_link_name = obj_link_name
                    else:
                        if eef_min_dist is None or dist_float < eef_min_dist:
                            eef_min_dist = dist_float
                            eef_nearest_obj_link_name = obj_link_name
                    all_min_dist = dist_float if all_min_dist is None else min(all_min_dist, dist_float)
                    if near_mode == "base_or_eef":
                        link_threshold = threshold if robot_link is robot_base_link else eef_threshold
                    else:
                        link_threshold = threshold
                    if link_threshold is not None and dist < link_threshold:
                        is_near = True
                        if not debug_near:
                            break
                if is_near and not debug_near:
                    break

            if debug_near and near_mode == "base":
                for robot_link in diagnostic_robot_links:
                    if robot_link is robot_base_link:
                        continue
                    robot_pos = robot_link.get_position_orientation()[0]
                    for obj_name, obj_link_name, obj_link in candidate_obj_links:
                        obj_pos = obj_link.get_position_orientation()[0]
                        dist_float = float(th.linalg.norm(robot_pos[:2] - obj_pos[:2]).item())
                        if all_min_dist is None or dist_float < all_min_dist:
                            all_min_dist = dist_float
                            nearest_obj_name = obj_name
                            nearest_obj_link_name = obj_link_name
                            nearest_robot_link_name = getattr(robot_link, "name", "eef")
                        if eef_min_dist is None or dist_float < eef_min_dist:
                            eef_min_dist = dist_float
                            eef_nearest_obj_link_name = obj_link_name

            results[name] = is_near
            if debug_near:
                print(
                    "[task_progress_debug_near] "
                    f"name={name} check_type={check_type} profile={_near_profile()} mode={near_mode} "
                    f"threshold={threshold} eef_threshold={eef_threshold} result={is_near} "
                    f"base_min={base_min_dist} eef_min={eef_min_dist} all_min={all_min_dist} "
                    f"nearest_robot_link={nearest_robot_link_name} nearest_obj={nearest_obj_name} "
                    f"nearest_obj_link={nearest_obj_link_name} base_nearest_obj_link={base_nearest_obj_link_name} "
                    f"eef_nearest_obj_link={eef_nearest_obj_link_name}",
                    flush=True,
                )

        elif check_type == "state":
            # Check if it's a relational or non-relational state based on argument pattern
            results[name] = _check_state_spec(env, spec)

        elif check_type in {"robot_in_room_of", "robot_in_room_zone_of"}:
            # ("robot_in_room_of", robot_key, obj_key) checks whether the robot base
            # is in one of the room instances assigned to the target scene object.
            # ("robot_in_room_zone_of", robot_key, obj_key, point_fn, threshold_fn)
            # additionally requires proximity to a task-local XY placement zone when
            # a point is configured.
            robot_entity = _resolve_object(env, spec[1])
            obj_entity = _resolve_object(env, spec[2])
            scene = getattr(env, "scene", None)
            seg_map = getattr(scene, "seg_map", None) if scene is not None else None
            target_rooms = _object_room_names(obj_entity)
            if not (robot_entity.exists and obj_entity.exists and seg_map is not None and target_rooms):
                results[name] = False
                continue

            robot = robot_entity.unwrapped
            robot_xy = _robot_base_link(robot).get_position_orientation()[0][:2]
            room_instance = seg_map.get_room_instance_by_point(robot_xy)
            in_target_room = room_instance in target_rooms
            target_xy = None
            zone_dist = None
            threshold = None
            if check_type == "robot_in_room_zone_of":
                target_xy = spec[3]()
                threshold = spec[4]()
                if target_xy is not None:
                    target_xy = target_xy.to(robot_xy.device)
                    zone_dist = float(th.linalg.norm(robot_xy - target_xy).item())
            results[name] = in_target_room and (target_xy is None or zone_dist <= threshold)
            if bool(os.environ.get("BEHAVIOR_TASK_PROGRESS_DEBUG_NEAR")):
                print(
                    "[task_progress_debug_room] "
                    f"name={name} check_type={check_type} result={results[name]} "
                    f"robot_room_instance={room_instance} target_rooms={sorted(target_rooms)} "
                    f"robot_xy={[float(v) for v in robot_xy.tolist()]} "
                    f"target_xy={None if target_xy is None else [float(v) for v in target_xy.tolist()]} "
                    f"zone_dist={zone_dist} threshold={threshold}",
                    flush=True,
                )

        elif check_type == "any_state":
            # ("any_state", [state_spec, ...]) - useful for BDDL goals with OR branches.
            results[name] = any(_check_composite_spec(env, state_spec) for state_spec in spec[1])

        elif check_type == "all_state":
            # ("all_state", [state_spec, ...]) - useful for grouped progress stages.
            component_results = [_check_composite_spec(env, state_spec) for state_spec in spec[1]]
            results[name] = all(component_results)
            _debug_composite_progress(name, spec, component_results)

        elif check_type == "grasping":
            # ("grasping", robot_key, obj_key, expected_bool) - robot is grasping object
            results[name] = _check_grasping_spec(env, spec)

        elif check_type == "grasping_arm":
            # ("grasping_arm", robot_key, obj_key, arm, expected_bool)
            results[name] = _check_arm_grasping_spec(env, spec)

        elif check_type == "aabb_bottom_above":
            # ("aabb_bottom_above", obj_key, reference_key, threshold) checks object clearance above a support.
            results[name] = _check_aabb_bottom_above_spec(env, spec)

        elif check_type == "open_fraction":
            # ("open_fraction", obj_key, min_fraction, expected_bool) - stricter than symbolic Open
            obj = _resolve_object(env, spec[1])
            fraction = _object_open_fraction(obj) if obj.exists else None
            results[name] = (
                obj.exists
                and (fraction >= spec[2]) == spec[3]
            )
            if _debug_task_progress_open():
                print(
                    "[task_progress_debug_open] "
                    f"name={name} obj={spec[1]} threshold={spec[2]} expected={spec[3]} "
                    f"fraction={fraction} result={results[name]}",
                    flush=True,
                )

        elif check_type == "exists":
            # ("exists", obj_key, expected_bool) - check if object exists
            if len(spec) == 3 and isinstance(spec[2], bool):
                try:
                    obj = _resolve_object(env, spec[1])
                    exists = obj.exists
                except KeyError:
                    exists = False
                results[name] = exists == spec[2]
            else:
                raise ValueError(f"Invalid exists spec: {spec}")

    return results


def _moving_boxes_to_storage_progress(env):
    _debug_moving_boxes_object_poses(env)
    return check_progress(
        env,
        {
            "robot_near_door": (
                "near_base_threshold",
                "agent.n.01_1",
                "door_bexenl_0",
                _moving_boxes_door_threshold(),
            ),
            "door_opened": ("open_fraction", "door_bexenl_0", _moving_boxes_door_open_threshold(), True),
            "robot_near_container_1": _moving_boxes_container_near_spec("storage_container.n.01_1"),
            "robot_near_container_2": _moving_boxes_container_near_spec("storage_container.n.01_2"),
            "robot_near_garage_floor": (
                "robot_in_room_zone_of",
                "agent.n.01_1",
                "floor.n.01_2",
                _moving_boxes_garage_place_point,
                _moving_boxes_garage_place_threshold,
            ),
            "container_1_picked_up": ("state", "storage_container.n.01_1", OnTop, "floor.n.01_1", False),
            "container_2_picked_up": ("state", "storage_container.n.01_2", OnTop, "floor.n.01_1", False),
            "container_1_in_garage": (
                "all_state",
                [
                    ("state", "storage_container.n.01_1", OnTop, "floor.n.01_2", True),
                    ("grasping", "agent.n.01_1", "storage_container.n.01_1", False),
                ],
            ),
            "container_2_in_garage": (
                "all_state",
                [
                    ("state", "storage_container.n.01_2", OnTop, "floor.n.01_2", True),
                    ("grasping", "agent.n.01_1", "storage_container.n.01_2", False),
                ],
            ),
            "containers_stacked": (
                "any_state",
                [
                    (
                        "all_state",
                        [
                            ("state", "storage_container.n.01_1", OnTop, "storage_container.n.01_2", True),
                            ("grasping", "agent.n.01_1", "storage_container.n.01_1", False),
                        ],
                    ),
                    (
                        "all_state",
                        [
                            ("state", "storage_container.n.01_2", OnTop, "storage_container.n.01_1", True),
                            ("grasping", "agent.n.01_1", "storage_container.n.01_2", False),
                        ],
                    ),
                ],
            ),
            "container_1_stacked": (
                "all_state",
                [
                    ("state", "storage_container.n.01_1", OnTop, "storage_container.n.01_2", True),
                    ("grasping", "agent.n.01_1", "storage_container.n.01_1", False),
                ],
            ),
            "container_2_stacked": (
                "all_state",
                [
                    ("state", "storage_container.n.01_2", OnTop, "storage_container.n.01_1", True),
                    ("grasping", "agent.n.01_1", "storage_container.n.01_2", False),
                ],
            ),
        },
    )


def _picking_up_trash_near_threshold(env):
    profile = _near_profile()
    if profile in {"trash_eef020_mixed", "trash_eef_0.20_mixed", "eef_0.20_mixed"}:
        instance_overrides = {
            106: 0.30,
            171: 0.10,
        }
        return instance_overrides.get(_activity_instance_id(env), 0.20)
    return 0.20


def _picking_up_trash_progress(env):
    near_threshold = _picking_up_trash_near_threshold(env)
    return check_progress(
        env,
        {
            "robot_near_trash_can": ("near_eef_threshold", "agent.n.01_1", "ashcan.n.01_1", near_threshold),
            "trash_can_picked_up": ("grasping", "agent.n.01_1", "ashcan.n.01_1", True),
            "robot_near_can_of_soda_1": (
                "near_eef_threshold",
                "agent.n.01_1",
                "can__of__soda.n.01_1",
                near_threshold,
            ),
            "robot_near_can_of_soda_2": (
                "near_eef_threshold",
                "agent.n.01_1",
                "can__of__soda.n.01_2",
                near_threshold,
            ),
            "robot_near_can_of_soda_3": (
                "near_eef_threshold",
                "agent.n.01_1",
                "can__of__soda.n.01_3",
                near_threshold,
            ),
            "can_of_soda_1_picked_up": ("state", "can__of__soda.n.01_1", OnTop, "floor.n.01_1", False),
            "can_of_soda_2_picked_up": ("state", "can__of__soda.n.01_2", OnTop, "floor.n.01_1", False),
            "can_of_soda_3_picked_up": ("state", "can__of__soda.n.01_3", OnTop, "floor.n.01_1", False),
            "can_of_soda_1_not_on_floor_2": ("state", "can__of__soda.n.01_1", OnTop, "floor.n.01_2", False),
            "can_of_soda_2_not_on_floor_2": ("state", "can__of__soda.n.01_2", OnTop, "floor.n.01_2", False),
            "can_of_soda_3_not_on_floor_2": ("state", "can__of__soda.n.01_3", OnTop, "floor.n.01_2", False),
            "can_of_soda_1_in_trash": ("state", "can__of__soda.n.01_1", Inside, "ashcan.n.01_1", True),
            "can_of_soda_2_in_trash": ("state", "can__of__soda.n.01_2", Inside, "ashcan.n.01_1", True),
            "can_of_soda_3_in_trash": ("state", "can__of__soda.n.01_3", Inside, "ashcan.n.01_1", True),
            "trash_can_on_floor": ("state", "ashcan.n.01_1", OnTop, "floor.n.01", True),
        },
    )


# Task specifications
CHALLENGE_TASKS_PROGRESS_APPROXIMATION = {
    "turning_on_radio": lambda env: check_progress(
        env,
        {
            "robot_near_radio": ("near", "agent.n.01_1", "radio_receiver.n.01_1"),
            "radio_picked_up": ("state", "radio_receiver.n.01_1", OnTop, "table.n.02_1", False),
            "radio_on": ("state", "radio_receiver.n.01_1", ToggledOn, True),
        },
    ),
    "picking_up_trash": _picking_up_trash_progress,
    "putting_away_Halloween_decorations": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": _halloween_support_near_spec("cabinet.n.01_1"),
            "cabinet_open": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "robot_near_candle_1": _halloween_pickup_near_spec("candle.n.01_1"),
            "robot_near_candle_2": _halloween_pickup_near_spec("candle.n.01_2"),
            "robot_near_candle_3": _halloween_pickup_near_spec("candle.n.01_3"),
            "candle_1_picked_up": ("state", "candle.n.01_1", OnTop, "floor.n.01_1", False),
            "candle_2_picked_up": ("state", "candle.n.01_2", OnTop, "floor.n.01_1", False),
            "candle_3_picked_up": ("state", "candle.n.01_3", OnTop, "floor.n.01_1", False),
            "candle_1_in_cabinet": ("state", "candle.n.01_1", Inside, "cabinet.n.01_1", True),
            "candle_2_in_cabinet": ("state", "candle.n.01_2", Inside, "cabinet.n.01_1", True),
            "candle_3_in_cabinet": ("state", "candle.n.01_3", Inside, "cabinet.n.01_1", True),
            "robot_near_pumpkin_1": _halloween_pickup_near_spec("pumpkin.n.02_1"),
            "robot_near_pumpkin_2": _halloween_pickup_near_spec("pumpkin.n.02_2"),
            "pumpkin_1_picked_up": ("state", "pumpkin.n.02_1", OnTop, "floor.n.01_1", False),
            "pumpkin_2_picked_up": ("state", "pumpkin.n.02_2", OnTop, "floor.n.01_1", False),
            "pumpkin_1_in_cabinet": ("state", "pumpkin.n.02_1", Inside, "cabinet.n.01_1", True),
            "pumpkin_2_in_cabinet": ("state", "pumpkin.n.02_2", Inside, "cabinet.n.01_1", True),
            "robot_near_caldron": _halloween_pickup_near_spec("caldron.n.01_1"),
            "caldron_picked_up": (
                "all_state",
                [
                    ("grasping", "agent.n.01_1", "caldron.n.01_1", True),
                    (
                        "aabb_bottom_above",
                        "caldron.n.01_1",
                        "floor.n.01_1",
                        _halloween_caldron_lift_clearance_threshold(),
                    ),
                ],
            ),
            "robot_near_table": _halloween_support_near_spec("table.n.02_1"),
            "caldron_next_to_table": (
                "all_state",
                [
                    ("state", "caldron.n.01_1", NextTo, "table.n.02_1", True),
                    ("grasping", "agent.n.01_1", "caldron.n.01_1", False),
                ],
            ),
        },
    ),
    "cleaning_up_plates_and_food": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "breakfast_table.n.01_1"),
            "robot_near_sink": ("near", "agent.n.01_1", "sink.n.01_1"),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "plate_1_picked_up": ("state", "plate.n.04_1", OnTop, "breakfast_table.n.01_1", False),
            "plate_2_picked_up": ("state", "plate.n.04_2", OnTop, "breakfast_table.n.01_1", False),
            "pizza_1_in_fridge": ("state", "pizza.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "pizza_2_in_fridge": ("state", "pizza.n.01_2", Inside, "electric_refrigerator.n.01_1", True),
            "plate_1_in_fridge": ("state", "plate.n.04_1", Inside, "electric_refrigerator.n.01_1", True),
            "plate_2_in_fridge": ("state", "plate.n.04_2", Inside, "electric_refrigerator.n.01_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "bowl_1_picked_up": ("state", "bowl.n.01_1", OnTop, "breakfast_table.n.01_1", False),
            "bowl_2_picked_up": ("state", "bowl.n.01_2", OnTop, "breakfast_table.n.01_1", False),
            "bowl_1_in_sink": ("state", "bowl.n.01_1", Inside, "sink.n.01_1", True),
            "bowl_2_in_sink": ("state", "bowl.n.01_2", Inside, "sink.n.01_1", True),
        },
    ),
    "can_meat": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_chopping_board": ("near", "agent.n.01_1", "chopping_board.n.01_1"),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "jar_1_picked_up": ("state", "hinged_jar.n.01_1", Inside, "cabinet.n.01_1", False),
            "jar_2_picked_up": ("state", "hinged_jar.n.01_2", Inside, "cabinet.n.01_1", False),
            "jar_1_opened": ("open_fraction", "hinged_jar.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "jar_2_opened": ("open_fraction", "hinged_jar.n.01_2", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "bratwurst_1_picked_up": ("state", "bratwurst.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_2_picked_up": ("state", "bratwurst.n.01_2", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_3_picked_up": ("state", "bratwurst.n.01_3", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_4_picked_up": ("state", "bratwurst.n.01_4", OnTop, "chopping_board.n.01_1", False),
            "bratwurst_1_in_jar": ("state", "bratwurst.n.01_1", Inside, "hinged_jar.n.01_1", True),
            "bratwurst_2_in_jar": ("state", "bratwurst.n.01_2", Inside, "hinged_jar.n.01_1", True),
            "bratwurst_3_in_jar": ("state", "bratwurst.n.01_3", Inside, "hinged_jar.n.01_2", True),
            "bratwurst_4_in_jar": ("state", "bratwurst.n.01_4", Inside, "hinged_jar.n.01_2", True),
            "jar_1_closed": ("state", "hinged_jar.n.01_1", Open, False),
            "jar_2_closed": ("state", "hinged_jar.n.01_2", Open, False),
            "jar_1_back_in_cabinet": ("state", "hinged_jar.n.01_1", Inside, "cabinet.n.01_1", True),
            "jar_2_back_in_cabinet": ("state", "hinged_jar.n.01_2", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, False),
        },
    ),
    "setting_mousetraps": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_sink": ("near", "agent.n.01_1", "sink.n.01_1"),
            "robot_near_toilet": ("near", "agent.n.01_1", "toilet.n.01_1"),
            "mousetrap_1_picked_up": ("state", "mousetrap.n.01_1", OnTop, "cabinet.n.01_1", False),
            "mousetrap_2_picked_up": ("state", "mousetrap.n.01_2", OnTop, "cabinet.n.01_1", False),
            "mousetrap_3_picked_up": ("state", "mousetrap.n.01_3", OnTop, "cabinet.n.01_1", False),
            "mousetrap_4_picked_up": ("state", "mousetrap.n.01_4", OnTop, "cabinet.n.01_1", False),
            "mousetrap_1_on_floor": ("state", "mousetrap.n.01_1", OnTop, "floor.n.01_1", True),
            "mousetrap_2_on_floor": ("state", "mousetrap.n.01_2", OnTop, "floor.n.01_1", True),
            "mousetrap_3_on_floor": ("state", "mousetrap.n.01_3", OnTop, "floor.n.01_1", True),
            "mousetrap_4_on_floor": ("state", "mousetrap.n.01_4", OnTop, "floor.n.01_1", True),
            "mousetrap_1_near_sink": ("state", "mousetrap.n.01_1", Under, "sink.n.01_1", True),
            "mousetrap_2_near_sink": ("state", "mousetrap.n.01_2", Under, "sink.n.01_1", True),
            "mousetrap_3_under_toilet": ("state", "mousetrap.n.01_3", Under, "toilet.n.01_1", True),
            "mousetrap_4_under_toilet": ("state", "mousetrap.n.01_4", Under, "toilet.n.01_1", True),
        },
    ),
    "hiding_Easter_eggs": lambda env: check_progress(
        env,
        {
            "robot_near_basket": ("near", "agent.n.01_1", "wicker_basket.n.01_1"),
            "robot_near_tree": ("near_category_threshold", "agent.n.01_1", "tree.n.01", 1.2),
            "basket_picked_up": ("state", "wicker_basket.n.01_1", OnTop, "lawn.n.01_1", False),
            "basket_next_to_tree": (
                "all_state",
                [
                    ("state", "wicker_basket.n.01_1", NextTo, "tree.n.01", True),
                    ("state", "wicker_basket.n.01_1", OnTop, "lawn.n.01_1", True),
                    ("grasping", "agent.n.01_1", "wicker_basket.n.01_1", False),
                ],
            ),
            "egg_1_out_of_basket": ("grasping", "agent.n.01_1", "easter_egg.n.01_1", True),
            "egg_2_out_of_basket": ("grasping", "agent.n.01_1", "easter_egg.n.01_2", True),
            "egg_3_out_of_basket": ("grasping", "agent.n.01_1", "easter_egg.n.01_3", True),
            "egg_1_on_lawn": (
                "all_state",
                [
                    ("state", "easter_egg.n.01_1", Inside, "wicker_basket.n.01_1", False),
                    ("state", "easter_egg.n.01_1", OnTop, "lawn.n.01_1", True),
                    ("grasping", "agent.n.01_1", "easter_egg.n.01_1", False),
                ],
            ),
            "egg_2_on_lawn": (
                "all_state",
                [
                    ("state", "easter_egg.n.01_2", Inside, "wicker_basket.n.01_1", False),
                    ("state", "easter_egg.n.01_2", OnTop, "lawn.n.01_1", True),
                    ("grasping", "agent.n.01_1", "easter_egg.n.01_2", False),
                ],
            ),
            "egg_3_on_lawn": (
                "all_state",
                [
                    ("state", "easter_egg.n.01_3", Inside, "wicker_basket.n.01_1", False),
                    ("state", "easter_egg.n.01_3", OnTop, "lawn.n.01_1", True),
                    ("grasping", "agent.n.01_1", "easter_egg.n.01_3", False),
                ],
            ),
            "egg_1_next_to_tree": (
                "all_state",
                [
                    ("state", "easter_egg.n.01_1", Inside, "wicker_basket.n.01_1", False),
                    ("state", "easter_egg.n.01_1", OnTop, "lawn.n.01_1", True),
                    ("state", "easter_egg.n.01_1", NextTo, "tree.n.01", True),
                    ("grasping", "agent.n.01_1", "easter_egg.n.01_1", False),
                ],
            ),
            "egg_2_next_to_tree": (
                "all_state",
                [
                    ("state", "easter_egg.n.01_2", Inside, "wicker_basket.n.01_1", False),
                    ("state", "easter_egg.n.01_2", OnTop, "lawn.n.01_1", True),
                    ("state", "easter_egg.n.01_2", NextTo, "tree.n.01", True),
                    ("grasping", "agent.n.01_1", "easter_egg.n.01_2", False),
                ],
            ),
            "egg_3_next_to_tree": (
                "all_state",
                [
                    ("state", "easter_egg.n.01_3", Inside, "wicker_basket.n.01_1", False),
                    ("state", "easter_egg.n.01_3", OnTop, "lawn.n.01_1", True),
                    ("state", "easter_egg.n.01_3", NextTo, "tree.n.01", True),
                    ("grasping", "agent.n.01_1", "easter_egg.n.01_3", False),
                ],
            ),
        },
    ),
    "picking_up_toys": lambda env: check_progress(
        env,
        {
            "robot_near_bed": ("near", "agent.n.01_1", "bed.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_toy_box": ("near", "agent.n.01_1", "toy_box.n.01_1"),
            "robot_near_board_game_1": ("near", "agent.n.01_1", "board_game.n.01_1"),
            "robot_near_board_game_2": ("near", "agent.n.01_1", "board_game.n.01_2"),
            "robot_near_board_game_3": ("near", "agent.n.01_1", "board_game.n.01_3"),
            "robot_near_jigsaw_1": ("near", "agent.n.01_1", "jigsaw_puzzle.n.01_1"),
            "robot_near_jigsaw_2": ("near", "agent.n.01_1", "jigsaw_puzzle.n.01_2"),
            "robot_near_tennis_ball": ("near", "agent.n.01_1", "tennis_ball.n.01_1"),
            "board_game_1_picked_up": (
                "all_state",
                [
                    ("state", "board_game.n.01_1", OnTop, "bed.n.01_1", False),
                    ("grasping", "agent.n.01_1", "board_game.n.01_1", True),
                ],
            ),
            "board_game_2_picked_up": (
                "all_state",
                [
                    ("state", "board_game.n.01_2", OnTop, "bed.n.01_1", False),
                    ("grasping", "agent.n.01_1", "board_game.n.01_2", True),
                ],
            ),
            "board_game_3_picked_up": (
                "all_state",
                [
                    ("state", "board_game.n.01_3", OnTop, "table.n.02_1", False),
                    ("grasping", "agent.n.01_1", "board_game.n.01_3", True),
                ],
            ),
            "jigsaw_1_picked_up": (
                "all_state",
                [
                    ("state", "jigsaw_puzzle.n.01_1", OnTop, "table.n.02_1", False),
                    ("grasping", "agent.n.01_1", "jigsaw_puzzle.n.01_1", True),
                ],
            ),
            "jigsaw_2_picked_up": (
                "all_state",
                [
                    ("state", "jigsaw_puzzle.n.01_2", OnTop, "table.n.02_1", False),
                    ("grasping", "agent.n.01_1", "jigsaw_puzzle.n.01_2", True),
                ],
            ),
            "tennis_ball_picked_up": (
                "all_state",
                [
                    ("state", "tennis_ball.n.01_1", OnTop, "table.n.02_1", False),
                    ("grasping", "agent.n.01_1", "tennis_ball.n.01_1", True),
                ],
            ),
            "board_game_1_in_box": ("state", "board_game.n.01_1", Inside, "toy_box.n.01_1", True),
            "board_game_2_in_box": ("state", "board_game.n.01_2", Inside, "toy_box.n.01_1", True),
            "board_game_3_in_box": ("state", "board_game.n.01_3", Inside, "toy_box.n.01_1", True),
            "jigsaw_1_in_box": ("state", "jigsaw_puzzle.n.01_1", Inside, "toy_box.n.01_1", True),
            "jigsaw_2_in_box": ("state", "jigsaw_puzzle.n.01_2", Inside, "toy_box.n.01_1", True),
            "tennis_ball_in_box": ("state", "tennis_ball.n.01_1", Inside, "toy_box.n.01_1", True),
        },
    ),
    "rearranging_kitchen_furniture": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "toaster_picked_up": ("state", "toaster.n.02_1", OnTop, "countertop.n.01_1", False),
            "food_processor_picked_up": ("state", "food_processor.n.01_1", OnTop, "countertop.n.01_1", False),
            "french_press_picked_up": ("state", "french_press.n.01_1", OnTop, "countertop.n.01_1", False),
            "toaster_in_cabinet": ("state", "toaster.n.02_1", Inside, "cabinet.n.01_1", True),
            "food_processor_in_cabinet": ("state", "food_processor.n.01_1", Inside, "cabinet.n.01_1", True),
            "french_press_in_cabinet": ("state", "french_press.n.01_1", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, False),
        },
    ),
    "putting_up_Christmas_decorations_inside": lambda env: check_progress(
        env,
        {
            "robot_near_basket": ("near", "agent.n.01_1", "wicker_basket.n.01_1"),
            "robot_near_tree": ("near", "agent.n.01_1", "christmas_tree.n.05_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_sofa": ("near", "agent.n.01_1", "sofa.n.01_1"),
            "basket_picked_up": ("state", "wicker_basket.n.01_1", OnTop, "floor.n.01_1", False),
            "basket_on_table": ("state", "wicker_basket.n.01_1", OnTop, "table.n.02_1", True),
            "wreath_out_of_basket": ("state", "wreath.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "candy_cane_1_out_of_basket": ("state", "candy_cane.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "candy_cane_2_out_of_basket": ("state", "candy_cane.n.01_2", Inside, "wicker_basket.n.01_1", False),
            "candy_cane_3_out_of_basket": ("state", "candy_cane.n.01_3", Inside, "wicker_basket.n.01_1", False),
            "candle_1_out_of_basket": ("state", "pillar_candle.n.01_1", Inside, "wicker_basket.n.01_1", False),
            "candle_2_out_of_basket": ("state", "pillar_candle.n.01_2", Inside, "wicker_basket.n.01_1", False),
            "gift_1_near_tree": ("state", "gift_box.n.01_1", NextTo, "christmas_tree.n.05_1", True),
            "gift_2_near_tree": ("state", "gift_box.n.01_2", NextTo, "christmas_tree.n.05_1", True),
            "gift_3_near_tree": ("state", "gift_box.n.01_3", NextTo, "christmas_tree.n.05_1", True),
            "candle_1_on_table": ("state", "pillar_candle.n.01_1", OnTop, "table.n.02_1", True),
            "candle_2_on_table": ("state", "pillar_candle.n.01_2", OnTop, "table.n.02_1", True),
            "candy_cane_on_table": ("state", "candy_cane.n.01_1", OnTop, "table.n.02_1", True),
            "wreath_on_sofa": ("state", "wreath.n.01_1", OnTop, "sofa.n.01_1", True),
            "candy_cane_2_on_sofa": ("state", "candy_cane.n.01_2", OnTop, "sofa.n.01_1", True),
            "candy_cane_3_on_sofa": ("state", "candy_cane.n.01_3", OnTop, "sofa.n.01_1", True),
        },
    ),
    "set_up_a_coffee_station_in_your_kitchen": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_coffee_cup": ("near", "agent.n.01_1", "coffee_cup.n.01_1"),
            "robot_near_saucer": ("near", "agent.n.01_1", "saucer.n.02_1"),
            "robot_near_electric_kettle": ("near", "agent.n.01_1", "electric_kettle.n.01_1"),
            "robot_near_paper_coffee_filter": ("near", "agent.n.01_1", "paper_coffee_filter.n.01_1"),
            "robot_near_bottle_of_coffee": ("near", "agent.n.01_1", "bottle__of__coffee.n.01_1"),
            "robot_near_shelf": ("near", "agent.n.01_1", "shelf.n.01_1"),
            "robot_near_coffee_maker": ("near", "agent.n.01_1", "coffee_maker.n.01_1"),
            "filter_picked_up": ("state", "paper_coffee_filter.n.01_1", OnTop, "countertop.n.01_1", False),
            "filter_in_coffee_maker": ("state", "paper_coffee_filter.n.01_1", OnTop, "coffee_maker.n.01_1", True),
            "coffee_bottle_picked_up": ("state", "bottle__of__coffee.n.01_1", OnTop, "shelf.n.01_1", False),
            "coffee_bottle_near_maker": ("state", "bottle__of__coffee.n.01_1", NextTo, "coffee_maker.n.01_1", True),
            "kettle_picked_up": ("state", "electric_kettle.n.01_1", OnTop, "countertop.n.01_1", False),
            "kettle_repositioned": ("state", "electric_kettle.n.01_1", NextTo, "coffee_maker.n.01_1", True),
            "saucer_near_maker": ("state", "saucer.n.02_1", NextTo, "coffee_maker.n.01_1", True),
            "cup_picked_up": ("state", "coffee_cup.n.01_1", OnTop, "countertop.n.01_1", False),
            "cup_on_saucer": ("state", "coffee_cup.n.01_1", OnTop, "saucer.n.02_1", True),
        },
    ),
    "putting_dishes_away_after_cleaning": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_countertop_1": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_countertop_2": ("near", "agent.n.01_1", "countertop.n.01_2"),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "plate_1_picked_up": ("state", "plate.n.04_1", OnTop, "countertop.n.01_1", False),
            "plate_2_picked_up": ("state", "plate.n.04_2", OnTop, "countertop.n.01_1", False),
            "plate_3_picked_up": ("state", "plate.n.04_3", OnTop, "countertop.n.01_1", False),
            "plate_4_picked_up": ("state", "plate.n.04_4", OnTop, "countertop.n.01_1", False),
            "plate_5_picked_up": ("state", "plate.n.04_5", OnTop, "countertop.n.01_2", False),
            "plate_6_picked_up": ("state", "plate.n.04_6", OnTop, "countertop.n.01_2", False),
            "plate_7_picked_up": ("state", "plate.n.04_7", OnTop, "countertop.n.01_2", False),
            "plate_8_picked_up": ("state", "plate.n.04_8", OnTop, "countertop.n.01_2", False),
            "plate_1_in_cabinet": ("state", "plate.n.04_1", Inside, "cabinet.n.01_1", True),
            "plate_2_in_cabinet": ("state", "plate.n.04_2", Inside, "cabinet.n.01_1", True),
            "plate_3_in_cabinet": ("state", "plate.n.04_3", Inside, "cabinet.n.01_1", True),
            "plate_4_in_cabinet": ("state", "plate.n.04_4", Inside, "cabinet.n.01_1", True),
            "plate_5_in_cabinet": ("state", "plate.n.04_5", Inside, "cabinet.n.01_1", True),
            "plate_6_in_cabinet": ("state", "plate.n.04_6", Inside, "cabinet.n.01_1", True),
            "plate_7_in_cabinet": ("state", "plate.n.04_7", Inside, "cabinet.n.01_1", True),
            "plate_8_in_cabinet": ("state", "plate.n.04_8", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, False),
        },
    ),
    "preparing_lunch_box": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_chopping_board": ("near", "agent.n.01_1", "chopping_board.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "lunch_box_picked_up": ("state", "packing_box.n.02_1", OnTop, "countertop.n.01_1", False),
            "sandwich_picked_up": ("state", "club_sandwich.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "apple_1_picked_up": ("state", "half__apple.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "apple_2_picked_up": ("state", "half__apple.n.01_2", OnTop, "chopping_board.n.01_1", False),
            "cookie_picked_up": ("state", "chocolate_chip_cookie.n.01_1", OnTop, "chopping_board.n.01_1", False),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tea_picked_up": ("state", "bottle__of__tea.n.01_1", Inside, "electric_refrigerator.n.01_1", False),
            "sandwich_in_box": ("state", "club_sandwich.n.01_1", Inside, "packing_box.n.02_1", True),
            "apple_1_in_box": ("state", "half__apple.n.01_1", Inside, "packing_box.n.02_1", True),
            "apple_2_in_box": ("state", "half__apple.n.01_2", Inside, "packing_box.n.02_1", True),
            "cookie_in_box": ("state", "chocolate_chip_cookie.n.01_1", Inside, "packing_box.n.02_1", True),
            "tea_in_box": ("state", "bottle__of__tea.n.01_1", Inside, "packing_box.n.02_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
        },
    ),
    "loading_the_car": lambda env: check_progress(
        env,
        {
            "robot_near_car": ("near", "agent.n.01_1", "car.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_container": ("near", "agent.n.01_1", "container.n.01_1"),
            "car_opened": ("open_fraction", "car.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "container_picked_up_floor_1": ("state", "container.n.01_1", OnTop, "floor.n.01_1", False),
            "container_picked_up_floor_2": ("state", "container.n.01_1", OnTop, "floor.n.01_2", False),
            "camera_picked_up": ("state", "digital_camera.n.01_1", OnTop, "table.n.02_1", False),
            "racket_picked_up": ("state", "tennis_racket.n.01_1", OnTop, "table.n.02_1", False),
            "container_in_car": ("state", "container.n.01_1", Inside, "car.n.01_1", True),
            "camera_in_container": ("state", "digital_camera.n.01_1", Inside, "container.n.01_1", True),
            "racket_in_car": ("state", "tennis_racket.n.01_1", Inside, "car.n.01_1", True),
            "car_closed": ("state", "car.n.01_1", Open, False),
        },
    ),
    "carrying_in_groceries": lambda env: check_progress(
        env,
        {
            "robot_near_car": ("near", "agent.n.01_1", "car.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "bag_picked_up": ("state", "sack.n.01_1", Inside, "car.n.01_1", False),
            "car_closed": ("state", "car.n.01_1", Open, False),
            "tomato_out_of_bag": ("state", "beefsteak_tomato.n.01_1", Inside, "sack.n.01_1", False),
            "milk_out_of_bag": ("state", "carton__of__milk.n.01_1", Inside, "sack.n.01_1", False),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tomato_in_fridge": ("state", "beefsteak_tomato.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "milk_in_fridge": ("state", "carton__of__milk.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
        },
    ),
    "bringing_in_wood": lambda env: check_progress(
        env,
        {
            "robot_near_plywood_1": _wood_plywood_near_spec("plywood.n.01_1"),
            "robot_near_plywood_2": _wood_plywood_near_spec("plywood.n.01_2"),
            "robot_near_plywood_3": _wood_plywood_near_spec("plywood.n.01_3"),
            "robot_near_wood_door": _wood_door_near_spec(),
            "wood_door_opened": ("open_fraction", "door_vudhlc_1", _wood_door_open_threshold(), True),
            "robot_in_corridor": ("robot_in_room_of", "agent.n.01_1", "floor.n.01_2"),
            **{
                f"plywood_{index}_picked_up": (
                    "all_state",
                    [
                        ("state", f"plywood.n.01_{index}", OnTop, "floor.n.01_1", False),
                        ("grasping", "agent.n.01_1", f"plywood.n.01_{index}", True),
                    ],
                )
                for index in (1, 2, 3)
            },
            **{
                f"plywood_{index}_indoors": (
                    "all_state",
                    [
                        ("state", f"plywood.n.01_{index}", OnTop, "floor.n.01_2", True),
                        ("grasping", "agent.n.01_1", f"plywood.n.01_{index}", False),
                    ],
                )
                for index in (1, 2, 3)
            },
        },
    ),
    "moving_boxes_to_storage": _moving_boxes_to_storage_progress,
    "bringing_water": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_WATER_FRIDGE_EEF_THRESHOLD",
                "electric_refrigerator.n.01_1",
            ),
            "robot_near_coffee_table": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_WATER_TABLE_EEF_THRESHOLD",
                "coffee_table.n.01_1",
            ),
            "fridge_opened": (
                "open_fraction",
                "electric_refrigerator.n.01_1",
                _task_open_fraction_threshold("BEHAVIOR_TASK_PROGRESS_WATER_FRIDGE_OPEN_FRACTION"),
                True,
            ),
            "bottle_1_picked_up": (
                "all_state",
                [
                    ("state", "bottle.n.01_1", Inside, "electric_refrigerator.n.01_1", False),
                    (
                        "any_state",
                        [
                            ("grasping_arm", "agent.n.01_1", "bottle.n.01_1", "left", True),
                            ("grasping_arm", "agent.n.01_1", "bottle.n.01_1", "right", True),
                        ],
                    ),
                ],
            ),
            "bottle_2_picked_up": (
                "all_state",
                [
                    ("state", "bottle.n.01_2", Inside, "electric_refrigerator.n.01_1", False),
                    (
                        "any_state",
                        [
                            ("grasping_arm", "agent.n.01_1", "bottle.n.01_2", "left", True),
                            ("grasping_arm", "agent.n.01_1", "bottle.n.01_2", "right", True),
                        ],
                    ),
                ],
            ),
            "bottle_1_grasped_left": ("grasping_arm", "agent.n.01_1", "bottle.n.01_1", "left", True),
            "bottle_1_grasped_right": ("grasping_arm", "agent.n.01_1", "bottle.n.01_1", "right", True),
            "bottle_2_grasped_left": ("grasping_arm", "agent.n.01_1", "bottle.n.01_2", "left", True),
            "bottle_2_grasped_right": ("grasping_arm", "agent.n.01_1", "bottle.n.01_2", "right", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "bottle_1_on_table": (
                "all_state",
                [
                    ("state", "bottle.n.01_1", OnTop, "coffee_table.n.01_1", True),
                    ("grasping", "agent.n.01_1", "bottle.n.01_1", False),
                ],
            ),
            "bottle_2_on_table": (
                "all_state",
                [
                    ("state", "bottle.n.01_2", OnTop, "coffee_table.n.01_1", True),
                    ("grasping", "agent.n.01_1", "bottle.n.01_2", False),
                ],
            ),
        },
    ),
    "tidying_bedroom": lambda env: check_progress(
        env,
        {
            "robot_near_book": ("near", "agent.n.01_1", "book.n.02_1"),
            "robot_near_sandal_1": ("near", "agent.n.01_1", "sandal.n.01_1"),
            "robot_near_sandal_2": ("near", "agent.n.01_1", "sandal.n.01_2"),
            "robot_near_bed": ("near", "agent.n.01_1", "bed.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "book_picked_up": ("state", "book.n.02_1", OnTop, "bed.n.01_1", False),
            "book_on_table": ("state", "book.n.02_1", OnTop, "table.n.02_1", True),
            "sandal_1_picked_up": ("state", "sandal.n.01_1", OnTop, "floor.n.01_1", False),
            "sandal_2_picked_up": ("state", "sandal.n.01_2", OnTop, "floor.n.01_1", False),
            "sandal_1_near_bed": ("state", "sandal.n.01_1", NextTo, "bed.n.01_1", True),
            "sandal_2_near_sandal_1": ("state", "sandal.n.01_2", NextTo, "sandal.n.01_1", True),
        },
    ),
    "outfit_a_basic_toolbox": lambda env: check_progress(
        env,
        {
            "robot_near_toolbox": ("near", "agent.n.01_1", "toolbox.n.01_1"),
            "toolbox_opened": ("open_fraction", "toolbox.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "drill_picked_up": ("state", "drill.n.01_1", OnTop, "tabletop.n.01_1", False),
            "pliers_picked_up": ("state", "pliers.n.01_1", OnTop, "tabletop.n.01_1", False),
            "flashlight_picked_up": ("state", "flashlight.n.01_1", OnTop, "tabletop.n.01_1", False),
            "wrench_picked_up": ("state", "allen_wrench.n.01_1", OnTop, "tabletop.n.01_1", False),
            "screwdriver_picked_up": ("state", "screwdriver.n.01_1", OnTop, "tabletop.n.01_1", False),
            "drill_in_toolbox": ("state", "drill.n.01_1", Inside, "toolbox.n.01_1", True),
            "pliers_in_toolbox": ("state", "pliers.n.01_1", Inside, "toolbox.n.01_1", True),
            "flashlight_in_toolbox": ("state", "flashlight.n.01_1", Inside, "toolbox.n.01_1", True),
            "wrench_in_toolbox": ("state", "allen_wrench.n.01_1", Inside, "toolbox.n.01_1", True),
            "screwdriver_in_toolbox": ("state", "screwdriver.n.01_1", Inside, "toolbox.n.01_1", True),
            "toolbox_closed": ("state", "toolbox.n.01_1", Open, False),
        },
    ),
    "sorting_vegetables": lambda env: check_progress(
        env,
        {
            "robot_near_basket_1": ("near", "agent.n.01_1", "wicker_basket.n.01_1"),
            "robot_near_basket_2": ("near", "agent.n.01_1", "wicker_basket.n.01_2"),
            "robot_near_bowls": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "basket_1_picked_up": ("state", "wicker_basket.n.01_1", OnTop, "floor.n.01_1", False),
            "basket_2_picked_up": ("state", "wicker_basket.n.01_2", OnTop, "floor.n.01_1", False),
            "bok_choy_1_sorted": ("state", "bok_choy.n.02_1", Inside, "mixing_bowl.n.01_2", True),
            "bok_choy_2_sorted": ("state", "bok_choy.n.02_2", Inside, "mixing_bowl.n.01_2", True),
            "bok_choy_3_sorted": ("state", "bok_choy.n.02_3", Inside, "mixing_bowl.n.01_2", True),
            "onion_1_sorted": ("state", "vidalia_onion.n.01_1", Inside, "mixing_bowl.n.01_2", True),
            "onion_2_sorted": ("state", "vidalia_onion.n.01_2", Inside, "mixing_bowl.n.01_2", True),
            "onion_3_sorted": ("state", "vidalia_onion.n.01_3", Inside, "mixing_bowl.n.01_2", True),
            "leek_1_sorted": ("state", "leek.n.02_1", Inside, "mixing_bowl.n.01_1", True),
            "leek_2_sorted": ("state", "leek.n.02_2", Inside, "mixing_bowl.n.01_1", True),
            "broccoli_1_sorted": ("state", "broccoli.n.02_1", Inside, "mixing_bowl.n.01_1", True),
            "broccoli_2_sorted": ("state", "broccoli.n.02_2", Inside, "mixing_bowl.n.01_1", True),
            "corn_1_sorted": ("state", "sweet_corn.n.02_1", Inside, "mixing_bowl.n.01_3", True),
            "corn_2_sorted": ("state", "sweet_corn.n.02_2", Inside, "mixing_bowl.n.01_3", True),
            "corn_3_sorted": ("state", "sweet_corn.n.02_3", Inside, "mixing_bowl.n.01_3", True),
        },
    ),
    "collecting_childrens_toys": lambda env: check_progress(
        env,
        {
            "robot_near_teddy_1": ("near", "agent.n.01_1", "teddy.n.01_1"),
            "robot_near_teddy_2": ("near", "agent.n.01_1", "teddy.n.01_2"),
            "robot_near_die_1": ("near", "agent.n.01_1", "die.n.01_1"),
            "robot_near_die_2": ("near", "agent.n.01_1", "die.n.01_2"),
            "robot_near_board_game_1": ("near", "agent.n.01_1", "board_game.n.01_1"),
            "robot_near_board_game_2": ("near", "agent.n.01_1", "board_game.n.01_2"),
            "robot_near_train": ("near", "agent.n.01_1", "train_set.n.01_1"),
            "robot_near_bookcase": ("near", "agent.n.01_1", "bookcase.n.01_1"),
            "teddy_1_picked_up": ("state", "teddy.n.01_1", OnTop, "floor.n.01_1", False),
            "teddy_2_picked_up": ("state", "teddy.n.01_2", OnTop, "floor.n.01_1", False),
            "die_1_picked_up": ("state", "die.n.01_1", OnTop, "bed.n.01_1", False),
            "die_2_picked_up": ("state", "die.n.01_2", OnTop, "bed.n.01_1", False),
            "board_game_1_picked_up": ("state", "board_game.n.01_1", OnTop, "desk.n.01_1", False),
            "board_game_2_picked_up": ("state", "board_game.n.01_2", OnTop, "bed.n.01_1", False),
            "train_picked_up": ("state", "train_set.n.01_1", OnTop, "desk.n.01_1", False),
            "teddy_1_in_bookcase": ("state", "teddy.n.01_1", Inside, "bookcase.n.01_1", True),
            "teddy_2_in_bookcase": ("state", "teddy.n.01_2", Inside, "bookcase.n.01_1", True),
            "die_1_in_bookcase": ("state", "die.n.01_1", Inside, "bookcase.n.01_1", True),
            "die_2_in_bookcase": ("state", "die.n.01_2", Inside, "bookcase.n.01_1", True),
            "board_game_1_in_bookcase": ("state", "board_game.n.01_1", Inside, "bookcase.n.01_1", True),
            "board_game_2_in_bookcase": ("state", "board_game.n.01_2", Inside, "bookcase.n.01_1", True),
        },
    ),
    "putting_shoes_on_rack": lambda env: check_progress(
        env,
        {
            "robot_near_gym_shoe_1": _shoes_pickup_near_spec("gym_shoe.n.01_1"),
            "robot_near_gym_shoe_2": _shoes_pickup_near_spec("gym_shoe.n.01_2"),
            "robot_near_sandal_1": _shoes_pickup_near_spec("sandal.n.01_1"),
            "robot_near_sandal_2": _shoes_pickup_near_spec("sandal.n.01_2"),
            "robot_near_hallstand": (
                "near_threshold",
                "agent.n.01_1",
                "hallstand.n.01_1",
                _shoes_hallstand_near_threshold(),
            ),
            "gym_shoe_1_picked_up": ("grasping", "agent.n.01_1", "gym_shoe.n.01_1", True),
            "gym_shoe_2_picked_up": ("grasping", "agent.n.01_1", "gym_shoe.n.01_2", True),
            "sandal_1_picked_up": ("grasping", "agent.n.01_1", "sandal.n.01_1", True),
            "sandal_2_picked_up": ("grasping", "agent.n.01_1", "sandal.n.01_2", True),
            "gym_shoe_1_on_floor": (
                "all_state",
                [
                    ("state", "gym_shoe.n.01_1", OnTop, "floor.n.01_1", True),
                    ("grasping", "agent.n.01_1", "gym_shoe.n.01_1", False),
                ],
            ),
            "gym_shoe_2_on_floor": (
                "all_state",
                [
                    ("state", "gym_shoe.n.01_2", OnTop, "floor.n.01_1", True),
                    ("grasping", "agent.n.01_1", "gym_shoe.n.01_2", False),
                ],
            ),
            "sandal_1_on_floor": (
                "all_state",
                [
                    ("state", "sandal.n.01_1", OnTop, "floor.n.01_1", True),
                    ("grasping", "agent.n.01_1", "sandal.n.01_1", False),
                ],
            ),
            "sandal_2_on_floor": (
                "all_state",
                [
                    ("state", "sandal.n.01_2", OnTop, "floor.n.01_1", True),
                    ("grasping", "agent.n.01_1", "sandal.n.01_2", False),
                ],
            ),
            "gym_shoe_1_on_rack": (
                "all_state",
                [
                    ("state", "gym_shoe.n.01_1", Touching, "hallstand.n.01_1", True),
                    ("state", "gym_shoe.n.01_1", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "gym_shoe.n.01_1", False),
                ],
            ),
            "gym_shoe_2_on_rack": (
                "all_state",
                [
                    ("state", "gym_shoe.n.01_2", Touching, "hallstand.n.01_1", True),
                    ("state", "gym_shoe.n.01_2", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "gym_shoe.n.01_2", False),
                ],
            ),
            "sandal_1_on_rack": (
                "all_state",
                [
                    ("state", "sandal.n.01_1", Touching, "hallstand.n.01_1", True),
                    ("state", "sandal.n.01_1", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "sandal.n.01_1", False),
                ],
            ),
            "sandal_2_on_rack": (
                "all_state",
                [
                    ("state", "sandal.n.01_2", Touching, "hallstand.n.01_1", True),
                    ("state", "sandal.n.01_2", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "sandal.n.01_2", False),
                ],
            ),
            "gym_shoes_on_rack": (
                "all_state",
                [
                    ("state", "gym_shoe.n.01_1", Touching, "hallstand.n.01_1", True),
                    ("state", "gym_shoe.n.01_1", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "gym_shoe.n.01_1", False),
                    ("state", "gym_shoe.n.01_2", Touching, "hallstand.n.01_1", True),
                    ("state", "gym_shoe.n.01_2", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "gym_shoe.n.01_2", False),
                ],
            ),
            "sandals_on_rack": (
                "all_state",
                [
                    ("state", "sandal.n.01_1", Touching, "hallstand.n.01_1", True),
                    ("state", "sandal.n.01_1", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "sandal.n.01_1", False),
                    ("state", "sandal.n.01_2", Touching, "hallstand.n.01_1", True),
                    ("state", "sandal.n.01_2", Touching, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "sandal.n.01_2", False),
                ],
            ),
            "gym_shoes_paired": ("state", "gym_shoe.n.01_1", NextTo, "gym_shoe.n.01_2", True),
            "sandals_paired": ("state", "sandal.n.01_1", NextTo, "sandal.n.01_2", True),
        },
    ),
    "boxing_books_up_for_storage": lambda env: check_progress(
        env,
        {
            "robot_near_box": ("near", "agent.n.01_1", "box.n.01_1"),
            "book_1_retrieved": ("state", "book.n.02_1", Inside, "bookcase.n.01", False),
            "book_2_retrieved": ("state", "book.n.02_2", Inside, "bookcase.n.01", False),
            "book_3_retrieved": ("state", "book.n.02_3", Inside, "bookcase.n.01", False),
            "book_4_retrieved": ("state", "book.n.02_4", Inside, "bookcase.n.01", False),
            "book_5_retrieved": ("state", "book.n.02_5", Inside, "bookcase.n.01", False),
            "book_6_retrieved": ("state", "book.n.02_6", Inside, "bookcase.n.01", False),
            "book_1_in_box": ("state", "book.n.02_1", Inside, "box.n.01_1", True),
            "book_2_in_box": ("state", "book.n.02_2", Inside, "box.n.01_1", True),
            "book_3_in_box": ("state", "book.n.02_3", Inside, "box.n.01_1", True),
            "book_4_in_box": ("state", "book.n.02_4", Inside, "box.n.01_1", True),
            "book_5_in_box": ("state", "book.n.02_5", Inside, "box.n.01_1", True),
            "book_6_in_box": ("state", "book.n.02_6", Inside, "box.n.01_1", True),
        },
    ),
    "storing_food": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "chips_1_picked_up": ("state", "bag__of__chips.n.01_1", OnTop, "countertop.n.01_1", False),
            "chips_2_picked_up": ("state", "bag__of__chips.n.01_2", OnTop, "countertop.n.01_1", False),
            "oil_1_picked_up": ("state", "bottle__of__olive_oil.n.01_1", OnTop, "countertop.n.01_1", False),
            "oil_2_picked_up": ("state", "bottle__of__olive_oil.n.01_2", OnTop, "countertop.n.01_1", False),
            "sugar_1_picked_up": ("state", "jar__of__sugar.n.01_1", OnTop, "countertop.n.01_1", False),
            "sugar_2_picked_up": ("state", "jar__of__sugar.n.01_2", OnTop, "countertop.n.01_1", False),
            "oatmeal_1_picked_up": ("state", "box__of__oatmeal.n.01_1", OnTop, "countertop.n.01_1", False),
            "oatmeal_2_picked_up": ("state", "box__of__oatmeal.n.01_2", OnTop, "countertop.n.01_1", False),
            "chips_1_stored": ("state", "bag__of__chips.n.01_1", Inside, "cabinet.n.01", True),
            "chips_2_stored": ("state", "bag__of__chips.n.01_2", Inside, "cabinet.n.01", True),
            "oil_1_stored": ("state", "bottle__of__olive_oil.n.01_1", Inside, "cabinet.n.01", True),
            "oil_2_stored": ("state", "bottle__of__olive_oil.n.01_2", Inside, "cabinet.n.01", True),
            "sugar_1_stored": ("state", "jar__of__sugar.n.01_1", Inside, "cabinet.n.01", True),
            "sugar_2_stored": ("state", "jar__of__sugar.n.01_2", Inside, "cabinet.n.01", True),
            "oatmeal_1_stored": ("state", "box__of__oatmeal.n.01_1", Inside, "cabinet.n.01", True),
            "oatmeal_2_stored": ("state", "box__of__oatmeal.n.01_2", Inside, "cabinet.n.01", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, False),
        },
    ),
    "clearing_food_from_table_into_fridge": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_table": ("near", "agent.n.01_1", "breakfast_table.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "tupperware_1_picked_up": ("state", "tupperware.n.01_1", OnTop, "countertop.n.01_1", False),
            "tupperware_2_picked_up": ("state", "tupperware.n.01_2", OnTop, "countertop.n.01_1", False),
            "chicken_picked_up": ("state", "half__chicken.n.01_1", OnTop, "plate.n.04_1", False),
            "pie_picked_up": ("state", "half__apple_pie.n.01_1", OnTop, "plate.n.04_2", False),
            "chicken_in_tupperware": ("state", "half__chicken.n.01_1", Inside, "tupperware.n.01_1", True),
            "pie_in_tupperware": ("state", "half__apple_pie.n.01_1", Inside, "tupperware.n.01_2", True),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tupperware_1_in_fridge": ("state", "tupperware.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "tupperware_2_in_fridge": ("state", "tupperware.n.01_2", Inside, "electric_refrigerator.n.01_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
        },
    ),
    "assembling_gift_baskets": lambda env: check_progress(
        env,
        {
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "basket_1_picked_up": ("state", "wicker_basket.n.01_1", OnTop, "floor.n.01_1", False),
            "basket_2_picked_up": ("state", "wicker_basket.n.01_2", OnTop, "floor.n.01_1", False),
            "basket_3_picked_up": ("state", "wicker_basket.n.01_3", OnTop, "floor.n.01_1", False),
            "basket_4_picked_up": ("state", "wicker_basket.n.01_4", OnTop, "floor.n.01_1", False),
            "candle_1_in_basket": ("state", "candle.n.01_1", Inside, "wicker_basket.n.01", True),
            "candle_2_in_basket": ("state", "candle.n.01_2", Inside, "wicker_basket.n.01", True),
            "candle_3_in_basket": ("state", "candle.n.01_3", Inside, "wicker_basket.n.01", True),
            "candle_4_in_basket": ("state", "candle.n.01_4", Inside, "wicker_basket.n.01", True),
            "cookie_1_in_basket": ("state", "butter_cookie.n.01_1", Inside, "wicker_basket.n.01", True),
            "cookie_2_in_basket": ("state", "butter_cookie.n.01_2", Inside, "wicker_basket.n.01", True),
            "cookie_3_in_basket": ("state", "butter_cookie.n.01_3", Inside, "wicker_basket.n.01", True),
            "cookie_4_in_basket": ("state", "butter_cookie.n.01_4", Inside, "wicker_basket.n.01", True),
            "cheese_1_in_basket": ("state", "swiss_cheese.n.01_1", Inside, "wicker_basket.n.01", True),
            "cheese_2_in_basket": ("state", "swiss_cheese.n.01_2", Inside, "wicker_basket.n.01", True),
            "cheese_3_in_basket": ("state", "swiss_cheese.n.01_3", Inside, "wicker_basket.n.01", True),
            "cheese_4_in_basket": ("state", "swiss_cheese.n.01_4", Inside, "wicker_basket.n.01", True),
            "bow_1_in_basket": ("state", "bow.n.08_1", OnTop, "wicker_basket.n.01", True),
            "bow_2_in_basket": ("state", "bow.n.08_2", OnTop, "wicker_basket.n.01", True),
            "bow_3_in_basket": ("state", "bow.n.08_3", OnTop, "wicker_basket.n.01", True),
            "bow_4_in_basket": ("state", "bow.n.08_4", OnTop, "wicker_basket.n.01", True),
        },
    ),
    "sorting_household_items": lambda env: check_progress(
        env,
        {
            "robot_near_basket_1": ("near", "agent.n.01_1", "basket.n.01_1"),
            "robot_near_basket_2": ("near", "agent.n.01_1", "basket.n.01_2"),
            "robot_near_sink": ("near", "agent.n.01_1", "sink.n.01_1"),
            "robot_near_shelf": ("near", "agent.n.01_1", "shelf.n.01_1"),
            "detergent_1_picked_up": ("state", "bottle__of__detergent.n.01_1", Inside, "basket.n.01_1", False),
            "detergent_2_picked_up": ("state", "bottle__of__detergent.n.01_2", Inside, "basket.n.01_1", False),
            "sanitary_napkins_picked_up": ("state", "box__of__sanitary_napkin.n.01_1", Inside, "basket.n.01_1", False),
            "soap_dispenser_picked_up": ("state", "soap_dispenser.n.01_1", Inside, "basket.n.01_2", False),
            "toothpaste_picked_up": ("state", "tube__of__toothpaste.n.01_1", Inside, "basket.n.01_2", False),
            "toothbrush_picked_up": ("state", "toothbrush.n.01_1", Inside, "basket.n.01_2", False),
            "detergent_1_under_sink": ("state", "bottle__of__detergent.n.01_1", Under, "sink.n.01_1", True),
            "detergent_2_under_sink": ("state", "bottle__of__detergent.n.01_2", Under, "sink.n.01_1", True),
            "detergents_together": (
                "state",
                "bottle__of__detergent.n.01_1",
                NextTo,
                "bottle__of__detergent.n.01_2",
                True,
            ),
            "sanitary_napkins_on_shelf": ("state", "box__of__sanitary_napkin.n.01_1", OnTop, "shelf.n.01_1", True),
            "soap_dispenser_on_sink": ("state", "soap_dispenser.n.01_1", OnTop, "sink.n.01_1", True),
            "toothbrush_in_cup": ("state", "toothbrush.n.01_1", Inside, "cup.n.01_1", True),
            "toothpaste_in_cup": ("state", "tube__of__toothpaste.n.01_1", Inside, "cup.n.01_1", True),
        },
    ),
    "getting_organized_for_work": lambda env: check_progress(
        env,
        {
            "robot_near_chair": ("near", "agent.n.01_1", "swivel_chair.n.01_1"),
            "robot_near_desk": ("near", "agent.n.01_1", "desk.n.01_1"),
            "chair_near_desk": ("state", "swivel_chair.n.01_1", NextTo, "desk.n.01_1", True),
            "keyboard_picked_up": ("state", "keyboard.n.01_1", OnTop, "notebook.n.01_1", False),
            "keyboard_on_desk": ("state", "keyboard.n.01_1", OnTop, "desk.n.01_1", True),
            "keyboard_near_monitor": ("state", "keyboard.n.01_1", NextTo, "monitor.n.04_1", True),
            "mouse_on_desk": ("state", "mouse.n.04_1", OnTop, "desk.n.01_1", True),
            "mouse_near_keyboard": ("state", "mouse.n.04_1", NextTo, "keyboard.n.01_1", True),
            "pen_picked_up": ("state", "pen.n.01_1", OnTop, "folder.n.02_1", False),
            "folder_picked_up": ("state", "folder.n.02_1", OnTop, "swivel_chair.n.01_1", False),
            "folder_on_desk": ("state", "folder.n.02_1", OnTop, "desk.n.01_1", True),
            "folder_near_mouse": ("state", "folder.n.02_1", NextTo, "mouse.n.04_1", True),
            "notebook_on_folder": ("state", "notebook.n.01_1", OnTop, "folder.n.02_1", True),
            "pen_on_notebook": ("state", "pen.n.01_1", OnTop, "notebook.n.01_1", True),
        },
    ),
    "clean_up_your_desk": lambda env: check_progress(
        env,
        {
            "robot_near_chair": ("near", "agent.n.01_1", "chair.n.01_1"),
            "robot_near_desk": ("near", "agent.n.01_1", "desk.n.01_1"),
            "robot_near_bookcase": ("near", "agent.n.01_1", "bookcase.n.01_1"),
            "robot_near_bed": ("near", "agent.n.01_1", "bed.n.01_1"),
            "robot_near_pencil_box": ("near", "agent.n.01_1", "pencil_box.n.01_1"),
            "folder_1_picked_up": ("state", "folder.n.02_1", OnTop, "desk.n.01_1", False),
            "folder_2_picked_up": ("state", "folder.n.02_2", OnTop, "chair.n.01_1", False),
            "folder_1_in_bookcase": ("state", "folder.n.02_1", Inside, "bookcase.n.01_1", True),
            "folder_2_in_bookcase": ("state", "folder.n.02_2", Inside, "bookcase.n.01_1", True),
            "pen_1_picked_up": ("state", "pen.n.01_1", OnTop, "desk.n.01_1", False),
            "pen_2_picked_up": ("state", "pen.n.01_2", OnTop, "desk.n.01_1", False),
            "pencil_picked_up": ("state", "pencil.n.01_1", OnTop, "desk.n.01_1", False),
            "pen_1_in_box": ("state", "pen.n.01_1", Inside, "pencil_box.n.01_1", True),
            "pen_2_in_box": ("state", "pen.n.01_2", Inside, "pencil_box.n.01_1", True),
            "pencil_in_box": ("state", "pencil.n.01_1", Inside, "pencil_box.n.01_1", True),
            "book_1_picked_up": ("state", "paperback_book.n.01_1", OnTop, "desk.n.01_1", False),
            "book_2_picked_up": ("state", "paperback_book.n.01_2", OnTop, "desk.n.01_1", False),
            "book_1_in_bookcase": ("state", "paperback_book.n.01_1", Inside, "bookcase.n.01_1", True),
            "book_2_in_bookcase": ("state", "paperback_book.n.01_2", Inside, "bookcase.n.01_1", True),
            "stapler_picked_up": ("state", "stapler.n.01_1", Inside, "bookcase.n.01_1", False),
            "stapler_on_desk": ("state", "stapler.n.01_1", OnTop, "desk.n.01_1", True),
            "laptop_picked_up": ("state", "laptop.n.01_1", OnTop, "bed.n.01_1", False),
            "laptop_on_desk": ("state", "laptop.n.01_1", OnTop, "desk.n.01_1", True),
            "laptop_closed": ("state", "laptop.n.01_1", Open, False),
        },
    ),
    "setting_the_fire": lambda env: check_progress(
        env,
        {
            "robot_near_table": ("near", "agent.n.01_1", "table.n.02_1"),
            "robot_near_fireplace": ("near", "agent.n.01_1", "wood_fireplace.n.01_1"),
            "robot_near_firewood_1": ("near", "agent.n.01_1", "firewood.n.01_1"),
            "robot_near_firewood_2": ("near", "agent.n.01_1", "firewood.n.01_2"),
            "newspaper_picked_up": ("state", "newspaper.n.03_1", OnTop, "table.n.02_1", False),
            "lighter_picked_up": ("state", "cigar_lighter.n.01_1", OnTop, "table.n.02_1", False),
            "firewood_1_picked_up": ("state", "firewood.n.01_1", OnTop, "floor.n.01_1", False),
            "firewood_2_picked_up": ("state", "firewood.n.01_2", OnTop, "floor.n.01_1", False),
            "newspaper_in_fireplace": ("state", "newspaper.n.03_1", Inside, "wood_fireplace.n.01_1", True),
            "firewood_1_in_fireplace": ("state", "firewood.n.01_1", Inside, "wood_fireplace.n.01_1", True),
            "firewood_2_in_fireplace": ("state", "firewood.n.01_2", Inside, "wood_fireplace.n.01_1", True),
            "lighter_on": ("state", "cigar_lighter.n.01_1", ToggledOn, True),
            "newspaper_on_fire": ("state", "newspaper.n.03_1", OnFire, True),
            "firewood_1_on_fire": ("state", "firewood.n.01_1", OnFire, True),
            "firewood_2_on_fire": ("state", "firewood.n.01_2", OnFire, True),
            "firewood_1_on_newspaper": ("state", "firewood.n.01_1", OnTop, "newspaper.n.03_1", True),
            "firewood_2_on_newspaper": ("state", "firewood.n.01_2", OnTop, "newspaper.n.03_1", True),
            "lighter_off": ("state", "cigar_lighter.n.01_1", ToggledOn, False),
        },
    ),
    "clean_boxing_gloves": lambda env: check_progress(
        env,
        {
            "robot_near_washer": ("near", "agent.n.01_1", "washer.n.03_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "washer_opened": ("open_fraction", "washer.n.03_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "glove_1_picked_up": ("state", "boxing_glove.n.01_1", OnTop, "countertop.n.01_1", False),
            "glove_2_picked_up": ("state", "boxing_glove.n.01_2", OnTop, "countertop.n.01_1", False),
            "glove_1_in_washer": ("state", "boxing_glove.n.01_1", Inside, "washer.n.03_1", True),
            "glove_2_in_washer": ("state", "boxing_glove.n.01_2", Inside, "washer.n.03_1", True),
            "washer_closed": ("state", "washer.n.03_1", Open, False),
            "washer_turned_on": ("state", "washer.n.03_1", ToggledOn, True),
            "glove_1_clean": ("state", "boxing_glove.n.01_1", Covered, "dust.n.01_1", False),
            "glove_2_clean": ("state", "boxing_glove.n.01_2", Covered, "dust.n.01_1", False),
        },
    ),
    "wash_a_baseball_cap": lambda env: check_progress(
        env,
        {
            "robot_near_washer": ("near", "agent.n.01_1", "washer.n.03_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "washer_opened": ("open_fraction", "washer.n.03_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "cap_1_picked_up": ("state", "baseball_cap.n.01_1", OnTop, "countertop.n.01_1", False),
            "cap_2_picked_up": ("state", "baseball_cap.n.01_2", OnTop, "countertop.n.01_1", False),
            "cap_1_in_washer": ("state", "baseball_cap.n.01_1", Inside, "washer.n.03_1", True),
            "cap_2_in_washer": ("state", "baseball_cap.n.01_2", Inside, "washer.n.03_1", True),
            "washer_closed": ("state", "washer.n.03_1", Open, False),
            "washer_turned_on": ("state", "washer.n.03_1", ToggledOn, True),
            "cap_1_clean": ("state", "baseball_cap.n.01_1", Covered, "dirt.n.02_1", False),
            "cap_2_clean": ("state", "baseball_cap.n.01_2", Covered, "dirt.n.02_1", False),
        },
    ),
    "wash_dog_toys": lambda env: check_progress(
        env,
        {
            "robot_near_washer": ("near", "agent.n.01_1", "washer.n.03_1"),
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "washer_opened": ("open_fraction", "washer.n.03_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "teddy_1_picked_up": ("state", "teddy.n.01_1", Inside, "cabinet.n.01_1", False),
            "teddy_2_picked_up": ("state", "teddy.n.01_2", Inside, "cabinet.n.01_1", False),
            "tennis_ball_picked_up": ("state", "tennis_ball.n.01_1", Inside, "cabinet.n.01_1", False),
            "softball_picked_up": ("state", "softball.n.01_1", Inside, "cabinet.n.01_1", False),
            "teddy_1_in_washer": ("state", "teddy.n.01_1", Inside, "washer.n.03_1", True),
            "teddy_2_in_washer": ("state", "teddy.n.01_2", Inside, "washer.n.03_1", True),
            "tennis_ball_in_washer": ("state", "tennis_ball.n.01_1", Inside, "washer.n.03_1", True),
            "softball_in_washer": ("state", "softball.n.01_1", Inside, "washer.n.03_1", True),
            "washer_closed": ("state", "washer.n.03_1", Open, False),
            "washer_turned_on": ("state", "washer.n.03_1", ToggledOn, True),
            "teddy_1_clean_dirt": ("state", "teddy.n.01_1", Covered, "dirt.n.02_1", False),
            "teddy_2_clean_dust": ("state", "teddy.n.01_2", Covered, "dust.n.01_1", False),
            "tennis_ball_clean": ("state", "tennis_ball.n.01_1", Covered, "debris.n.01_1", False),
            "softball_clean": ("state", "softball.n.01_1", Covered, "dirt.n.02_1", False),
        },
    ),
    "hanging_pictures": lambda env: check_progress(
        env,
        {
            "robot_near_poster": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_PICTURES_PICKUP_EEF_THRESHOLD",
                "poster.n.01_1",
            ),
            "robot_near_wall_nail": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_PICTURES_HANG_EEF_THRESHOLD",
                "wall_nail.n.01_1",
            ),
            "poster_picked_up": ("state", "poster.n.01_1", OnTop, "countertop.n.01_1", False),
            "poster_attached_to_nail": ("state", "poster.n.01_1", AttachedTo, "wall_nail.n.01_1", True),
        },
    ),
    "attach_a_camera_to_a_tripod": lambda env: check_progress(
        env,
        {
            "robot_near_camera": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_CAMERA_PICKUP_EEF_THRESHOLD",
                "digital_camera.n.01_1",
            ),
            "robot_near_tripod": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_CAMERA_TRIPOD_EEF_THRESHOLD",
                "camera_tripod.n.01_1",
            ),
            "camera_picked_up": (
                "all_state",
                (
                    ("state", "digital_camera.n.01_1", OnTop, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "digital_camera.n.01_1", True),
                ),
            ),
            "tripod_held": ("grasping", "agent.n.01_1", "camera_tripod.n.01_1", True),
            "camera_attached_to_tripod": ("state", "digital_camera.n.01_1", AttachedTo, "camera_tripod.n.01_1", True),
            "tripod_released": (
                "all_state",
                (
                    ("state", "digital_camera.n.01_1", AttachedTo, "camera_tripod.n.01_1", True),
                    ("grasping", "agent.n.01_1", "camera_tripod.n.01_1", False),
                ),
            ),
        },
    ),
    "clean_a_patio": lambda env: check_progress(
        env,
        {
            "robot_near_broom": ("near", "agent.n.01_1", "broom.n.01_1"),
            "broom_picked_up": ("state", "broom.n.01_1", OnTop, "floor.n.01_1", False),
            "floor_completely_clean": ("state", "floor.n.01_1", Covered, "mud.n.03_1", False),
        },
    ),
    "clean_a_trumpet": lambda env: check_progress(
        env,
        {
            "robot_near_scrub_brush": ("near", "agent.n.01_1", "scrub_brush.n.01_1"),
            "robot_near_cornet": ("near", "agent.n.01_1", "cornet.n.01_1"),
            "scrub_brush_picked_up": (
                "all_state",
                [
                    ("state", "scrub_brush.n.01_1", OnTop, "desk.n.01_1", False),
                    ("grasping", "agent.n.01_1", "scrub_brush.n.01_1", True),
                ],
            ),
            "scrub_brush_grasped_left": (
                "grasping_arm",
                "agent.n.01_1",
                "scrub_brush.n.01_1",
                "left",
                True,
            ),
            "scrub_brush_grasped_right": (
                "grasping_arm",
                "agent.n.01_1",
                "scrub_brush.n.01_1",
                "right",
                True,
            ),
            "cornet_completely_clean": ("state", "cornet.n.01_1", Covered, "dust.n.01_1", False),
            "scrub_brush_released_on_desk": (
                "all_state",
                [
                    ("state", "scrub_brush.n.01_1", OnTop, "desk.n.01_1", True),
                    ("grasping", "agent.n.01_1", "scrub_brush.n.01_1", False),
                ],
            ),
        },
    ),
    "spraying_for_bugs": lambda env: check_progress(
        env,
        {
            "robot_near_atomizer": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_BUGS_PICKUP_EEF_THRESHOLD",
                "insectifuge__atomizer.n.01_1",
                BUGS_PICKUP_EEF_THRESHOLD,
            ),
            "robot_near_plant_1": _bugs_plant_near_spec("pot_plant.n.01_1"),
            "robot_near_plant_2": _bugs_plant_near_spec("pot_plant.n.01_2"),
            "atomizer_picked_up": (
                "all_state",
                [
                    ("state", "insectifuge__atomizer.n.01_1", OnTop, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "insectifuge__atomizer.n.01_1", True),
                ],
            ),
            "atomizer_turned_on": ("state", "insectifuge__atomizer.n.01_1", ToggledOn, True),
            "plant_1_covered": ("state", "pot_plant.n.01_1", Covered, "insectifuge.n.01_1", True),
            "plant_2_covered": ("state", "pot_plant.n.01_2", Covered, "insectifuge.n.01_1", True),
            "atomizer_turned_off": ("state", "insectifuge__atomizer.n.01_1", ToggledOn, False),
        },
    ),
    "spraying_fruit_trees": lambda env: check_progress(
        env,
        {
            "robot_near_atomizer": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_FRUIT_TREES_PICKUP_EEF_THRESHOLD",
                "pesticide__atomizer.n.01_1",
            ),
            "robot_near_tree_1": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_FRUIT_TREES_SPRAY_EEF_THRESHOLD",
                "tree.n.01_1",
            ),
            "robot_near_tree_2": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_FRUIT_TREES_SPRAY_EEF_THRESHOLD",
                "tree.n.01_2",
            ),
            "atomizer_picked_up": (
                "all_state",
                [
                    ("state", "pesticide__atomizer.n.01_1", OnTop, "floor.n.01_1", False),
                    ("grasping", "agent.n.01_1", "pesticide__atomizer.n.01_1", True),
                ],
            ),
            "atomizer_turned_on": ("state", "pesticide__atomizer.n.01_1", ToggledOn, True),
            "tree_1_covered": ("state", "tree.n.01_1", Covered, "pesticide.n.01_1", True),
            "tree_2_covered": ("state", "tree.n.01_2", Covered, "pesticide.n.01_1", True),
            "atomizer_turned_off": ("state", "pesticide__atomizer.n.01_1", ToggledOn, False),
        },
    ),
    "make_microwave_popcorn": lambda env: check_progress(
        env,
        {
            "robot_near_microwave_open": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_POPCORN_OPEN_EEF_THRESHOLD",
                "microwave.n.02_1",
                POPCORN_OPEN_EEF_THRESHOLD,
            ),
            "robot_near_popcorn_bag": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_POPCORN_PICKUP_EEF_THRESHOLD",
                "popcorn__bag.n.01_1",
            ),
            "robot_near_microwave_place": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_POPCORN_PLACE_EEF_THRESHOLD",
                "microwave.n.02_1",
            ),
            "microwave_opened": (
                "open_fraction",
                "microwave.n.02_1",
                _popcorn_open_fraction_threshold(),
                True,
            ),
            "popcorn_bag_picked_up": (
                "all_state",
                [
                    ("state", "popcorn__bag.n.01_1", OnTop, "countertop.n.01_1", False),
                    ("grasping", "agent.n.01_1", "popcorn__bag.n.01_1", True),
                ],
            ),
            "popcorn_bag_in_microwave": (
                "all_state",
                [
                    ("state", "popcorn__bag.n.01_1", Inside, "microwave.n.02_1", True),
                    ("grasping", "agent.n.01_1", "popcorn__bag.n.01_1", False),
                ],
            ),
            "microwave_closed": ("state", "microwave.n.02_1", Open, False),
            "microwave_turned_on": ("state", "microwave.n.02_1", ToggledOn, True),
            "popcorn_cooked": ("exists", "cooked__popcorn.n.01_1", True),
        },
    ),
    "cook_cabbage": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_stove": ("near", "agent.n.01_1", "stove.n.01_1"),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "cabbage_picked_up": ("state", "head_cabbage.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
            "chili_picked_up": ("state", "chili.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "cabbage_on_board": ("state", "head_cabbage.n.02_1", OnTop, "chopping_board.n.01_1", True),
            "chili_on_board": ("state", "chili.n.02_1", OnTop, "chopping_board.n.01_1", True),
            "pan_on_stove": ("state", "frying_pan.n.01_1", OnTop, "stove.n.01_1", True),
            "cabbage_diced": ("exists", "head_cabbage.n.02_1", False),
            "chili_diced": ("exists", "chili.n.02_1", False),
            "cabbage_in_pan": ("state", "frying_pan.n.01_1", Contains, "cooked__diced__head_cabbage.n.01_1", True),
            "chili_in_pan": ("state", "frying_pan.n.01_1", Contains, "cooked__diced__chili.n.01_1", True),
            "stove_on": ("state", "stove.n.01_1", ToggledOn, True),
        },
    ),
    "chop_an_onion": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_sink": ("near", "agent.n.01_1", "sink.n.01_1"),
            "board_picked_up": ("state", "chopping_board.n.01_1", OnTop, "countertop.n.01_1", False),
            "board_placed_near_bowl": ("state", "chopping_board.n.01_1", NextTo, "bowl.n.01_1", True),
            "parer_picked_up": ("state", "parer.n.02_1", OnTop, "countertop.n.01_1", False),
            "onion_picked_up": ("state", "vidalia_onion.n.01_1", Inside, "sink.n.01_1", False),
            "onion_on_board": ("state", "vidalia_onion.n.01_1", OnTop, "chopping_board.n.01_1", True),
            "onion_diced": ("exists", "diced__vidalia_onion.n.01_1", True),
            "diced_onion_in_bowl": ("state", "bowl.n.01_1", Contains, "diced__vidalia_onion.n.01_1", True),
            "parer_in_sink": ("state", "parer.n.02_1", Inside, "sink.n.01_1", True),
            "board_in_sink": ("state", "chopping_board.n.01_1", Inside, "sink.n.01_1", True),
        },
    ),
    "slicing_vegetables": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "pepper_1_retrieved": ("state", "bell_pepper.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
            "pepper_2_retrieved": ("state", "bell_pepper.n.02_2", Inside, "electric_refrigerator.n.01_1", False),
            "beet_1_retrieved": ("state", "beet.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
            "beet_2_retrieved": ("state", "beet.n.02_2", Inside, "electric_refrigerator.n.01_1", False),
            "zucchini_retrieved": ("state", "zucchini.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "veggies_on_board_1": ("state", "bell_pepper.n.02_1", OnTop, "chopping_board.n.01", True),
            "veggies_on_board_2": ("state", "bell_pepper.n.02_2", OnTop, "chopping_board.n.01", True),
            "beet_1_on_board": ("state", "beet.n.02_1", OnTop, "chopping_board.n.01", True),
            "beet_2_on_board": ("state", "beet.n.02_2", OnTop, "chopping_board.n.01", True),
            "zucchini_on_board": ("state", "zucchini.n.02_1", OnTop, "chopping_board.n.01", True),
            "parer_picked_up": ("state", "parer.n.02_1", OnTop, "countertop.n.01_1", False),
            "peppers_diced": ("exists", "diced__bell_pepper.n.01_1", True),
            "beets_diced": ("exists", "diced__beet.n.01_1", True),
            "zucchini_diced": ("exists", "diced__zucchini.n.01_1", True),
        },
    ),
    "chopping_wood": lambda env: check_progress(
        env,
        {
            "robot_near_driveway": ("near", "agent.n.01_1", "driveway.n.01_1"),
            "robot_near_chopping_block": ("near", "agent.n.01_1", "chopping_block.n.01_1"),
            "axe_picked_up": ("state", "ax.n.01_1", OnTop, "driveway.n.01_1", False),
            "log_1_picked_up": ("state", "log.n.01_1", OnTop, "driveway.n.01_1", False),
            "log_2_picked_up": ("state", "log.n.01_2", OnTop, "driveway.n.01_1", False),
            "log_3_picked_up": ("state", "log.n.01_3", OnTop, "driveway.n.01_1", False),
            "log_4_picked_up": ("state", "log.n.01_4", OnTop, "driveway.n.01_1", False),
            "log_1_on_block": ("state", "log.n.01_1", OnTop, "chopping_block.n.01_1", True),
            "log_1_chopped": ("exists", "half__log.n.01_1", True),
            "log_1_chopped_2": ("exists", "half__log.n.01_2", True),
            "log_2_on_block": ("state", "log.n.01_2", OnTop, "chopping_block.n.01_1", True),
            "log_2_chopped": ("exists", "half__log.n.01_3", True),
            "log_2_chopped_2": ("exists", "half__log.n.01_4", True),
            "log_3_on_block": ("state", "log.n.01_3", OnTop, "chopping_block.n.01_1", True),
            "log_3_chopped": ("exists", "half__log.n.01_5", True),
            "log_3_chopped_2": ("exists", "half__log.n.01_6", True),
            "log_4_on_block": ("state", "log.n.01_4", OnTop, "chopping_block.n.01_1", True),
            "log_4_chopped": ("exists", "half__log.n.01_7", True),
            "log_4_chopped_2": ("exists", "half__log.n.01_8", True),
        },
    ),
    "cook_hot_dogs": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_HOTDOG_FRIDGE_EEF_THRESHOLD",
                "electric_refrigerator.n.01_1",
            ),
            "robot_near_microwave": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_HOTDOG_MICROWAVE_EEF_THRESHOLD",
                "microwave.n.02_1",
            ),
            "robot_near_hotdog_1": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_HOTDOG_PICKUP_EEF_THRESHOLD",
                "hotdog.n.02_1",
            ),
            "robot_near_hotdog_2": _optional_task_eef_near_spec(
                "BEHAVIOR_TASK_PROGRESS_HOTDOG_PICKUP_EEF_THRESHOLD",
                "hotdog.n.02_2",
            ),
            "fridge_opened": (
                "open_fraction",
                "electric_refrigerator.n.01_1",
                _task_open_fraction_threshold("BEHAVIOR_TASK_PROGRESS_HOTDOG_FRIDGE_OPEN_FRACTION"),
                True,
            ),
            "hotdog_1_retrieved": (
                "all_state",
                (
                    ("state", "hotdog.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
                    ("grasping", "agent.n.01_1", "hotdog.n.02_1", True),
                ),
            ),
            "hotdog_2_retrieved": (
                "all_state",
                (
                    ("state", "hotdog.n.02_2", Inside, "electric_refrigerator.n.01_1", False),
                    ("grasping", "agent.n.01_1", "hotdog.n.02_2", True),
                ),
            ),
            "hotdog_1_grasped": ("grasping", "agent.n.01_1", "hotdog.n.02_1", True),
            "hotdog_2_grasped": ("grasping", "agent.n.01_1", "hotdog.n.02_2", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "microwave_opened": (
                "open_fraction",
                "microwave.n.02_1",
                _task_open_fraction_threshold("BEHAVIOR_TASK_PROGRESS_HOTDOG_MICROWAVE_OPEN_FRACTION"),
                True,
            ),
            "hotdog_1_on_countertop": (
                "all_state",
                (
                    ("state", "hotdog.n.02_1", OnTop, "countertop.n.01_1", True),
                    ("grasping", "agent.n.01_1", "hotdog.n.02_1", False),
                ),
            ),
            "hotdog_2_on_countertop": (
                "all_state",
                (
                    ("state", "hotdog.n.02_2", OnTop, "countertop.n.01_1", True),
                    ("grasping", "agent.n.01_1", "hotdog.n.02_2", False),
                ),
            ),
            "hotdog_1_in_microwave": (
                "all_state",
                (
                    ("state", "hotdog.n.02_1", Inside, "microwave.n.02_1", True),
                    ("grasping", "agent.n.01_1", "hotdog.n.02_1", False),
                ),
            ),
            "hotdog_2_in_microwave": (
                "all_state",
                (
                    ("state", "hotdog.n.02_2", Inside, "microwave.n.02_1", True),
                    ("grasping", "agent.n.01_1", "hotdog.n.02_2", False),
                ),
            ),
            "microwave_closed": ("state", "microwave.n.02_1", Open, False),
            "microwave_on": ("state", "microwave.n.02_1", ToggledOn, True),
            "hotdog_1_cooked": ("state", "hotdog.n.02_1", Cooked, True),
            "hotdog_2_cooked": ("state", "hotdog.n.02_2", Cooked, True),
        },
    ),
    "cook_bacon": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_stove": ("near", "agent.n.01_1", "stove.n.01_1"),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tray_retrieved": ("state", "tray.n.01_1", Inside, "electric_refrigerator.n.01_1", False),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "bacon_in_pan": ("state", "bacon.n.01_1", Inside, "frying_pan.n.01_1", True),
            "pan_on_stove": ("state", "frying_pan.n.01_1", OnTop, "stove.n.01_1", True),
            "stove_on": ("state", "stove.n.01_1", ToggledOn, True),
            "bacon_1_cooked": ("state", "bacon.n.01_1", Cooked, True),
            "bacon_2_cooked": ("state", "bacon.n.01_2", Cooked, True),
            "bacon_3_cooked": ("state", "bacon.n.01_3", Cooked, True),
            "bacon_4_cooked": ("state", "bacon.n.01_4", Cooked, True),
            "bacon_5_cooked": ("state", "bacon.n.01_5", Cooked, True),
            "bacon_6_cooked": ("state", "bacon.n.01_6", Cooked, True),
        },
    ),
    "freeze_pies": lambda env: check_progress(
        env,
        {
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tupperware_1_retrieved": ("state", "tupperware.n.01_1", Inside, "cabinet.n.01_1", False),
            "tupperware_2_retrieved": ("state", "tupperware.n.01_2", Inside, "cabinet.n.01_1", False),
            "pie_1_picked_up": ("state", "apple_pie.n.01_1", OnTop, "plate.n.04_1", False),
            "pie_2_picked_up": ("state", "apple_pie.n.01_2", OnTop, "plate.n.04_2", False),
            "pie_1_in_tupperware": ("state", "apple_pie.n.01_1", Inside, "tupperware.n.01_1", True),
            "pie_2_in_tupperware": ("state", "apple_pie.n.01_2", Inside, "tupperware.n.01_2", True),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tupperware_1_in_fridge": ("state", "tupperware.n.01_1", Inside, "electric_refrigerator.n.01_1", True),
            "tupperware_2_in_fridge": ("state", "tupperware.n.01_2", Inside, "electric_refrigerator.n.01_1", True),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "pie_1_frozen": ("state", "apple_pie.n.01_1", Frozen, True),
            "pie_2_frozen": ("state", "apple_pie.n.01_2", Frozen, True),
        },
    ),
    "canning_food": lambda env: check_progress(
        env,
        {
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_cabinet": ("near", "agent.n.01_1", "cabinet.n.01_1"),
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "steak_retrieved": ("state", "steak.n.01_1", Inside, "electric_refrigerator.n.01_1", False),
            "pineapple_retrieved": ("state", "pineapple.n.02_1", Inside, "electric_refrigerator.n.01_1", False),
            "fridge_closed": ("state", "electric_refrigerator.n.01_1", Open, False),
            "cabinet_opened": ("open_fraction", "cabinet.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "bowls_retrieved": ("state", "bowl.n.01_1", Inside, "cabinet.n.01_1", False),
            "bowl_2_retrieved": ("state", "bowl.n.01_2", Inside, "cabinet.n.01_1", False),
            "steak_on_board": ("state", "steak.n.01_1", OnTop, "chopping_board.n.01_1", True),
            "pineapple_on_board": ("state", "pineapple.n.02_1", OnTop, "chopping_board.n.01_1", True),
            "steak_diced": ("exists", "diced__steak.n.01_1", True),
            "pineapple_diced": ("exists", "diced__pineapple.n.01_1", True),
            "steak_in_bowl": ("state", "bowl.n.01_1", Filled, "diced__steak.n.01_1", True),
            "pineapple_in_bowl": ("state", "bowl.n.01_2", Filled, "diced__pineapple.n.01_1", True),
            "bowls_in_cabinet": ("state", "bowl.n.01_1", Inside, "cabinet.n.01_1", True),
            "bowl_2_in_cabinet": ("state", "bowl.n.01_2", Inside, "cabinet.n.01_1", True),
            "cabinet_closed": ("state", "cabinet.n.01_1", Open, False),
        },
    ),
    "make_pizza": lambda env: check_progress(
        env,
        {
            "robot_near_countertop": ("near", "agent.n.01_1", "countertop.n.01_1"),
            "robot_near_fridge": ("near", "agent.n.01_1", "electric_refrigerator.n.01_1"),
            "robot_near_oven": ("near", "agent.n.01_1", "oven.n.01_1"),
            "onion_chopped": ("exists", "vidalia_onion.n.01_1", False),
            "fridge_opened": ("open_fraction", "electric_refrigerator.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "tupperware_1_retrieved": ("state", "tupperware.n.01_1", Inside, "electric_refrigerator.n.01_1", False),
            "tupperware_2_retrieved": ("state", "tupperware.n.01_2", Inside, "electric_refrigerator.n.01_1", False),
            "tupperware_3_retrieved": ("state", "tupperware.n.01_3", Inside, "electric_refrigerator.n.01_1", False),
            "pepperoni_1_in_bowl": ("state", "pepperoni.n.01_1", Inside, "bowl.n.01_1", True),
            "pepperoni_2_in_bowl": ("state", "pepperoni.n.01_2", Inside, "bowl.n.01_1", True),
            "pepperoni_3_in_bowl": ("state", "pepperoni.n.01_3", Inside, "bowl.n.01_1", True),
            "pepperoni_4_in_bowl": ("state", "pepperoni.n.01_4", Inside, "bowl.n.01_1", True),
            "mushroom_1_chopped": ("exists", "mushroom.n.05_1", False),
            "mushroom_2_chopped": ("exists", "mushroom.n.05_2", False),
            "pepperoni_1_on_dough": ("state", "pepperoni.n.01_1", OnTop, "pizza_dough.n.01_1", True),
            "pepperoni_2_on_dough": ("state", "pepperoni.n.01_2", OnTop, "pizza_dough.n.01_1", True),
            "pepperoni_3_on_dough": ("state", "pepperoni.n.01_3", OnTop, "pizza_dough.n.01_1", True),
            "pepperoni_4_on_dough": ("state", "pepperoni.n.01_4", OnTop, "pizza_dough.n.01_1", True),
            "cheese_on_pizza": ("state", "pizza_dough.n.01_1", Covered, "grated_cheese.n.01_1", True),
            "oven_opened": ("open_fraction", "oven.n.01_1", PROGRESS_OPEN_FRACTION_THRESHOLD, True),
            "sheet_in_oven": ("state", "cookie_sheet.n.01_1", Inside, "oven.n.01_1", True),
            "oven_closed": ("state", "oven.n.01_1", Open, False),
            "oven_on": ("state", "oven.n.01_1", ToggledOn, True),
            "pizza_created": ("exists", "pizza.n.01_1", True),
            "pizza_on_sheet": ("state", "pizza.n.01_1", OnTop, "cookie_sheet.n.01_1", True),
        },
    ),
}
