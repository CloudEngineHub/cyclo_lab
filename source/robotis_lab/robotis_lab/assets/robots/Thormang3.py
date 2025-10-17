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

from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR

THORMANG_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/Thormang/thormang3.usd",
        activate_contact_sensors=True,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*_leg_hip_y": 0.0,
            ".*_leg_hip_r": 0.0,
            "l_leg_hip_p": -0.28,
            "r_leg_hip_p": 0.28,
            "l_leg_kn_p": 0.79,
            "r_leg_kn_p": -0.79,
            "l_leg_an_p": 0.52,
            "r_leg_an_p": -0.52,
            ".*_leg_an_r": 0.0,
            ".*_arm_sh_p1": 0.0,
            "l_arm_sh_r": 1.56,
            "r_arm_sh_r": -1.56,
            ".*_arm_sh_p2": 0.0,
            "l_arm_el_y": -1.25,
            "r_arm_el_y": 1.25,
            ".*_arm_wr.*": 0.0,
            ".*_arm_grip.*": 0.0,
            "torso_y": 0.0,
            "head_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_.*",
                ".*_kn_.*",
                "torso_y",
            ],
            effort_limit_sim=2000,
            stiffness={
                ".*_hip_.*": 800.0,
                ".*_kn_.*": 600.0,
                "torso_y": 500.0,
            },
            damping={
                ".*_hip_.*": 10.0,
                ".*_kn_.*": 8.0,
                "torso_y": 8.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_kn_.*": 0.01,
                "torso_y": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=100,
            joint_names_expr=[".*_leg_an_.*"],
            stiffness=50.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_arm_.*",
            ],
            effort_limit_sim=600,
            stiffness=80.0,
            damping=20.0,
            armature={
                ".*_arm_.*": 0.01,
            },
        ),
    },
)
