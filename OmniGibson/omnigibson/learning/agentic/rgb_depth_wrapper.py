"""BEHAVIOR wrapper exposing only RGB, metric depth, and proprioception to agentic runs."""

from omnigibson.envs import Environment
from omnigibson.envs import EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.eval_utils import WRIST_RESOLUTION
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger("AgenticRGBDepthWrapper")


class AgenticRGBDepthWrapper(EnvironmentWrapper):
    """Match training camera geometry while adding only linear metric depth."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            sensor = robot.sensors[sensor_name]
            if camera_id == "head":
                sensor.horizontal_aperture = 40.0
                sensor.image_height = HEAD_RESOLUTION[0]
                sensor.image_width = HEAD_RESOLUTION[1]
            else:
                sensor.image_height = WRIST_RESOLUTION[0]
                sensor.image_width = WRIST_RESOLUTION[1]
            sensor.add_modality("depth_linear")
        env.load_observation_space()
        logger.info("Reloaded observation space with RGB and linear depth only.")
