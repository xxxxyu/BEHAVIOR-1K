from omnigibson.metrics.task_metric import TaskMetric


def test_gather_results_supports_zero_action_finalization():
    metric = TaskMetric(human_stats={"length": 120})
    metric.timesteps = 0
    metric.render_timestep = 1.0 / 30.0
    metric.final_q_score = 0.0

    result = metric.gather_results()

    assert result["q_score"] == {"final": 0.0}
    assert result["time"] == {
        "simulator_steps": 0,
        "simulator_time": 0.0,
        "normalized_time": float("inf"),
    }
