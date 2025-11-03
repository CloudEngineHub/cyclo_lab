# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="RobotisLab-Real-Pick-Place-Bottle-FFW-SG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FFWSG2BottlePickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="RobotisLab-Real-Mimic-Pick-Place-Bottle-FFW-SG2-v0",
#     entry_point="robotis_lab.real_world_tasks.manager_based.FFW-SG2.pick_place.pick_place_mimic_env:FFW-SG2PickPlaceMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.pick_place_mimic_env_cfg:FFW-SG2PickPlaceMimicEnvCfg",
#     },
#     disable_env_checker=True,
# )
