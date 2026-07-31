import torch as th

from omnigibson.metrics.agent_metric import AgentMetric


class _Robot:
    arm_names = ["left", "right"]

    def __init__(self):
        self.base = th.tensor([1.0, 2.0, 0.0])
        self.eef = {
            "left": th.tensor([0.0, 0.0, 0.0]),
            "right": th.tensor([1.0, 0.0, 0.0]),
        }

    def get_position_orientation(self):
        return self.base.clone(), th.tensor([0.0, 0.0, 0.0, 1.0])

    def get_eef_position(self, arm):
        return self.eef[arm].clone()


class _Environment:
    def __init__(self):
        self.robots = [_Robot()]


def test_start_callback_supports_zero_action_finalization():
    env = _Environment()
    metric = AgentMetric(
        human_stats={"distance_traveled": 1.0, "left_eef_displacement": 1.0, "right_eef_displacement": 1.0}
    )

    metric.start_callback(env)
    result = metric.gather_results()

    assert result["agent_distance"] == {"base": 0, "left": 0, "right": 0}
    assert result["normalized_agent_distance"] == {"base": float("inf"), "left": float("inf"), "right": float("inf")}


def test_first_step_distance_is_measured_from_start_state():
    env = _Environment()
    metric = AgentMetric(
        human_stats={"distance_traveled": 2.0, "left_eef_displacement": 2.0, "right_eef_displacement": 2.0}
    )
    metric.start_callback(env)
    env.robots[0].base[0] += 1.0
    env.robots[0].eef["left"][1] += 0.5
    env.robots[0].eef["right"][2] += 0.25

    metric.step_callback(env)
    result = metric.gather_results()

    assert result["agent_distance"] == {"base": 1.0, "left": 0.5, "right": 0.25}
    assert result["normalized_agent_distance"] == {"base": 2.0, "left": 4.0, "right": 8.0}
