# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Literal
import torch

import numpy as np
import tyro

from p2_env import P2Env
from p2_teleop.p2.p2_math_utils import AngleType

import sys
sys.path.append("/opt/bg/ws/src/bg_p2/groot/Isaac-GR00T")
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""


#####################################################################################


def main(args: ArgsConfig):
    # Create a policy
    # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
    # the model path, transform name, embodiment tag, and denoising steps for the robot
    # inference system. This policy object is then utilized in the server mode to start
    # the Robot Inference Server for making predictions based on the specified model and
    # configuration.

    # we will use an existing data config to create the modality config and transform
    # if a new data config is specified, this expect user to
    # construct your own modality config and transform
    # see gr00t/utils/data.py for more details
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()


    #Define the robot environment
    # p2_env = P2Env(freq=30,  # Frequency of the environment
    #                is_delta_actions=False,  # Whether the actions are delta actions
    #                angle_type=AngleType.AXANGLE,  # Type of angle representation
    #                use_blocking_move=True,  # Whether to use blocking move commands
    #                sleep_after_move=0.2,  # Time to sleep after a move command
    #                left_reset_config=None,  # Configuration for resetting the left arm
    #                right_reset_config=None,  # Configuration for resetting the right arm
    #                yawing_gripper_reset_position=0.0,  # Yawing gripper reset position
    #                modality_config=modality_config
    #            )

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    import time

    print("Available modality config available:")
    modality_configs = policy.get_modality_config()
    print(modality_configs.keys())

    # Making prediction...
    # - obs: video.hfg_left: (1, 1200, 1920, 3)
    # - obs: video.hfg_right: (1, 1200, 1920, 3)
    # - obs: state.ee_pos: (1, 3)
    # - obs: state.ee_rot: (1, 3)
    # - obs: state.wrist: (1, 3)

    # - action: action.ee_pos: (16, 3)
    # - action: action.ee_rot: (16, 3)
    # - action: action.valve: (16, 1)
    obs = {
        "video.hfg_left": np.random.randint(0, 256, (1, 1200, 1920, 3), dtype=np.uint8),
        "video.hfg_right": np.random.randint(0, 256, (1, 1200, 1920, 3), dtype=np.uint8),
        "state.ee_pos": np.random.rand(1, 3),
        "state.ee_rot": np.random.rand(1, 3),
        "state.wrist": np.random.rand(1, 3),
    }

    # obs, _ = p2_env.reset()  # Reset the environment to start fresh
    while True:
        time_start = time.time()
        action = policy.get_action(obs)
        obs, _, is_done, is_truncated, info = p2_env.step(action)
        print(f"Total time taken to get action from server: {time.time() - time_start} seconds")


    for key, value in action.items():
        print(f"Action: {key}: {value.shape}")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
