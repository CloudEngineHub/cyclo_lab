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

import numpy as np
from pxr import Gf, Usd, UsdPhysics

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import clone

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR


# Source: ROBOTIS-GIT/ai_worker
# ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf
# URDF revision: 50f368f42ed836302035008f3c17960a873a6ccf.
# Entries: (mass, center_of_mass, (ixx, ixy, ixz, iyy, iyz, izz)).
# Units: kg, m, and kg*m^2.
# Fixed child links are included in the mass properties of their USD rigid body.
# Wrist-camera inertias are marked as unreliable in the reference URDF.
_SG2_MASS_PROPERTIES = {
    # base_link: URDF lines 7-29; lift_link: URDF lines 75-96; fixed joint: lines 97-101.
    "world": (
        53.889438999999996,
        (-0.04274878723454516, -5.329800579293468e-06, 0.2971384994933794),
        (
            11.3018586003064,
            -1.8250358259134208e-05,
            1.3775171805477766,
            13.758312097086169,
            0.0001888273032177941,
            3.0443403029354714,
        ),
    ),
    # arm_base_link: URDF lines 651-672.
    "arm_base_link": (
        6.1939559,
        (-0.014922878, 0.0027315224999999994, -0.043425264),
        (0.10828866, 2.9296925e-06, -0.0015727508, 0.054323697, 0.0016989706, 0.079600843),
    ),
    # arm_l_link1: URDF lines 793-814.
    "arm_l_link1": (
        2.0013322,
        (0.014210373, 0.11553718, 0.0001120644),
        (0.0025075577, -0.00021963558, 1.1512519e-05, 0.0028652997, -2.3110079e-06, 0.0033653716),
    ),
    # arm_l_link2: URDF lines 823-844.
    "arm_l_link2": (
        2.128261,
        (0.0091098413, -0.00010530165, -0.14386407),
        (0.0099761666, -1.4206416e-06, -0.002370609, 0.010888353, 4.8321415e-06, 0.0031685827),
    ),
    # arm_l_link3: URDF lines 853-874.
    "arm_l_link3": (
        1.6847551,
        (0.029875373, 0.013913949, -0.1210139),
        (0.0028780772, -0.00026152652, 0.00066794083, 0.0027070961, 0.00032962522, 0.0025663506),
    ),
    # arm_l_link4: URDF lines 883-904.
    "arm_l_link4": (
        1.5082144,
        (-0.038538653, 0.0098072286, -0.13131228),
        (0.0062523038, -0.00021627982, -0.00051136601, 0.0057050864, -0.0015977591, 0.0020980278),
    ),
    # arm_l_link5: URDF lines 913-934.
    "arm_l_link5": (
        1.3917831,
        (-3.1226161e-05, 0.016819227, -0.098738156),
        (0.0018856721, 1.3269158e-06, 2.4184848e-07, 0.0012852729, 0.00012877617, 0.0016893353),
    ),
    # arm_l_link6: URDF lines 943-964.
    "arm_l_link6": (
        0.65770741,
        (0.0022438917, 0.018764105, -0.064805576),
        (0.0014161387, 2.8390103e-05, 3.5822295e-05, 0.0011423344, -0.00053568368, 0.0010175176),
    ),
    # arm_l_link7: URDF lines 973-994; gripper base: lines 1279-1300; camera: lines 1007-1028.
    # Fixed joints: URDF lines 996-1006 and 1273-1277.
    "arm_l_link7": (
        0.4748563,
        (0.02385782034514181, 6.681941238281537e-15, -0.08512533040088517),
        (
            0.0009395985081883375,
            1.623981642346965e-14,
            0.0001177717081959763,
            0.005170822451455743,
            -1.6265342674638071e-15,
            0.004799972303267405,
        ),
    ),
    # arm_r_link1: URDF lines 1037-1058.
    "arm_r_link1": (
        2.0013322,
        (0.014210373, -0.11553718, -0.0001120644),
        (0.0025075577, 0.00021963558, -1.1512519e-05, 0.0028652997, -2.3110079e-06, 0.0033653716),
    ),
    # arm_r_link2: URDF lines 1067-1088.
    "arm_r_link2": (
        2.128261,
        (0.0091098413, -0.00010530165, -0.14386407),
        (0.0099761666, -1.4206416e-06, -0.002370609, 0.010888353, 4.8321415e-06, 0.0031685827),
    ),
    # arm_r_link3: URDF lines 1097-1118.
    "arm_r_link3": (
        1.6847551,
        (0.029845765000000003, -0.013922959, -0.12104351),
        (0.0028779712, 0.00026321599, 0.00066779985, 0.0027073784, -0.00032759299, 0.0025635286),
    ),
    # arm_r_link4: URDF lines 1127-1148.
    "arm_r_link4": (
        1.4942511,
        (-0.038677468, -0.0093080099, -0.13229713),
        (0.0060323579, 0.00020289973, -0.00048540519, 0.0055364235, 0.0015071246, 0.0020329997),
    ),
    # arm_r_link5: URDF lines 1157-1178.
    "arm_r_link5": (
        1.3917831,
        (0.0008387686, -0.016798499, -0.098738156),
        (0.0018824928, 3.480391e-05, 6.8067431e-06, 0.0012884522, -0.00012859637, 0.0016893353),
    ),
    # arm_r_link6: URDF lines 1187-1208.
    "arm_r_link6": (
        0.65770741,
        (0.0032674912, -0.01861259, -0.064792616),
        (0.0014179762, -1.325841e-05, 6.5344079e-06, 0.0011396474, 0.00053667073, 0.0010174762),
    ),
    # arm_r_link7: URDF lines 1217-1238; gripper base: lines 1454-1475; camera: lines 1251-1272.
    # Fixed joints: URDF lines 1240-1250 and 1448-1452.
    "arm_r_link7": (
        0.4748563,
        (0.023844386632236873, 0.0004905307037744535, -0.08512533040088517),
        (
            0.0009408544408752721,
            -1.9443434156458756e-05,
            0.00011794489033584805,
            0.005169756554323796,
            -6.3236845422431215e-06,
            0.004800162338822394,
        ),
    ),
    # head_link1: URDF lines 681-702.
    "head_link1": (
        0.123518,
        (0.025339656, 0.0, 0.047745146),
        (6.8709657e-06, 8.5922602e-10, -3.0026645e-07, 1.7715102e-05, -4.4479737e-10, 1.6311135e-05),
    ),
}


def _principal_inertia(inertia: tuple[float, ...]) -> tuple[Gf.Vec3f, Gf.Quatf]:
    """Convert a URDF inertia tensor to USD principal moments and axes."""
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    moments, axes = np.linalg.eigh(tensor)
    # Keep the eigenvectors right-handed so they describe a rotation.
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1

    # Gf uses row vectors; NumPy returns principal axes as columns.
    rotation = Gf.Matrix3d(*axes.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    principal_axes = Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary()))
    return Gf.Vec3f(*moments.tolist()), principal_axes


@clone
def spawn_sg2_with_mass_properties(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the original SG2 USD and apply URDF-based mass properties."""
    robot = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    applied_bodies = set()
    for body in Usd.PrimRange(robot):
        body_name = body.GetName()
        if body_name not in _SG2_MASS_PROPERTIES or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        mass, center_of_mass, inertia = _SG2_MASS_PROPERTIES[body_name]
        diagonal_inertia, principal_axes = _principal_inertia(inertia)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
        mass_api.CreateDiagonalInertiaAttr(diagonal_inertia)
        mass_api.CreatePrincipalAxesAttr(principal_axes)
        applied_bodies.add(body_name)

    missing_bodies = _SG2_MASS_PROPERTIES.keys() - applied_bodies
    if missing_bodies:
        raise RuntimeError(f"SG2 mass properties could not be applied to: {sorted(missing_bodies)}")
    return robot


FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        func=spawn_sg2_with_mass_properties,
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # # Swerve base joints
            # "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
            # "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
            # "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,

            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

            # Head joints
            "head_joint1": 0.0,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={
        # Actuators for swerve base
        # "base": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "left_wheel_drive", "left_wheel_steer",
        #         "right_wheel_drive", "right_wheel_steer",
        #         "rear_wheel_drive", "rear_wheel_steer",
        #     ],
        #     velocity_limit_sim=30.0,
        #     effort_limit_sim=100000.0,
        #     stiffness=10000.0,
        #     damping=100.0,
        # ),

        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42" : ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Actuators for grippers
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=0.0,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)
