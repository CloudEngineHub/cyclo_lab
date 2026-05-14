# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bring up the FFW SH5 USD model and control it from DDS JointTrajectory topics.

* spawns the FFW_SH5 USD as an Isaac Lab articulation,
* subscribes to retargeted JointTrajectory topics from Cyclo Control,
* applies the latest received joint positions as simulation joint targets,
* publishes the simulated robot state on ``/joint_states` , ``/tf``.

Example:

.. code-block:: bash

    python scripts/sim2real/bringup/sh5_dds_bringup.py --enable_cameras

"""

import argparse
import os
import threading
import time
from collections.abc import Iterable
from copy import deepcopy

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="FFW SH5 DDS bringup for Isaac Sim.")
parser.add_argument("--right_arm_topic", type=str, default="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory")
parser.add_argument("--right_hand_topic", type=str, default="/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory")
parser.add_argument("--left_arm_topic", type=str, default="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory")
parser.add_argument("--left_hand_topic", type=str, default="/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory")
parser.add_argument("--head_topic", type=str, default="/leader/joystick_controller_left/joint_trajectory")
parser.add_argument("--lift_topic", type=str, default="/leader/joystick_controller_right/joint_trajectory")
parser.add_argument("--disable_left", action="store_true", help="Do not subscribe to left arm/hand topics.")
parser.add_argument("--disable_head_lift", action="store_true", help="Do not subscribe to head/lift topics.")
parser.add_argument("--joint_states_topic", type=str, default="/joint_states")
parser.add_argument("--tf_topic", type=str, default="/tf")
parser.add_argument("--base_frame", type=str, default="base_link")
parser.add_argument("--publish_hz", type=float, default=60.0)
parser.add_argument("--step_hz", type=float, default=120.0)
parser.add_argument("--usd_path", type=str, default=None, help="Override the default FFW_SH5.usd path.")
parser.add_argument("--robot_x", type=float, default=0.0, help="Initial robot x position.")
parser.add_argument("--robot_y", type=float, default=0.0, help="Initial robot y position.")
parser.add_argument("--robot_z", type=float, default=-0.18, help="Initial robot z position. Use a negative value if the USD floats above the ground.")
parser.add_argument("--lift_init_pos", type=float, default=0.0, help="Initial lift_joint position.")
parser.add_argument("--domain_id", type=int, default=None, help="DDS domain id. Defaults to ROS_DOMAIN_ID or 0.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity on the SH5 rigid bodies.")
parser.add_argument(
    "--articulation_root_prim_path",
    type=str,
    default="/base_link/base_link",
    help="Articulation root path inside the loaded SH5 USD.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robotis_dds_python.idl.builtin_interfaces.msg import Time_
from robotis_dds_python.idl.geometry_msgs.msg import Quaternion_, Transform_, TransformStamped_, Vector3_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager

from robotis_lab.assets.robots import FFW_SH5_CFG


def _default_sh5_usd_path() -> str:
    return FFW_SH5_CFG.spawn.usd_path


@configclass
class SH5BringupSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot: ArticulationCfg = None


class SH5DdsBridge:
    def __init__(
        self,
        robot,
        topic_manager: TopicManager,
        topic_names: dict[str, str],
        joint_states_topic: str,
        tf_topic: str,
        base_frame: str,
    ):
        self.robot = robot
        self.base_frame = base_frame
        self.running = True
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.unknown_joints: set[str] = set()
        self._warned_missing_base_frame = False
        self.readers = []
        self.threads = []
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name=joint_states_topic,
            topic_type=JointState_,
        )
        self.tf_writer = topic_manager.topic_writer(
            topic_name=tf_topic,
            topic_type=TFMessage_,
        )

        for label, topic_name in topic_names.items():
            if not topic_name:
                continue
            reader = topic_manager.topic_reader(topic_name=topic_name, topic_type=JointTrajectory_)
            thread = threading.Thread(
                target=self._trajectory_loop,
                args=(label, reader),
                daemon=True,
            )
            self.readers.append(reader)
            self.threads.append(thread)
            thread.start()
            print(f"[DDS] Subscribing {label}: {topic_name}")

    def _trajectory_loop(self, label: str, reader):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_trajectory(label, msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] {label} subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _store_trajectory(self, label: str, msg):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        joint_names = list(msg.joint_names)
        positions = list(point.positions)

        if len(joint_names) != len(positions):
            print(
                f"[DDS] Ignoring {label} message: joint_names={len(joint_names)} "
                f"positions={len(positions)}"
            )
            return

        with self.lock:
            self.pending_positions.update(dict(zip(joint_names, positions)))

    def apply_latest_targets(self):
        with self.lock:
            if not self.pending_positions:
                return
            commands = dict(self.pending_positions)

        joint_names = self.robot.data.joint_names
        target = self.robot.data.joint_pos_target.clone()

        for name, position in commands.items():
            if name not in joint_names:
                if name not in self.unknown_joints:
                    self.unknown_joints.add(name)
                    print(f"[DDS] Joint '{name}' is not in the SH5 USD articulation; ignoring it.")
                continue
            joint_id = joint_names.index(name)
            target[:, joint_id] = float(position)

        self.robot.set_joint_position_target(target)

    def publish_joint_states(self):
        now = time.time()
        stamp = Time_(sec=int(now), nanosec=int((now - int(now)) * 1_000_000_000))
        header = Header_(stamp=stamp, frame_id="base_link")

        joint_names = list(self.robot.data.joint_names)
        positions = self.robot.data.joint_pos.squeeze(0).detach().cpu().tolist()
        velocities = self.robot.data.joint_vel.squeeze(0).detach().cpu().tolist()
        efforts = [0.0] * len(joint_names)

        msg = JointState_(
            header=header,
            name=joint_names,
            position=positions,
            velocity=velocities,
            effort=efforts,
        )
        try:
            self.joint_state_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] joint_states write error: {exc}")

    def publish_tf(self):
        body_names = list(self.robot.data.body_names)
        if self.base_frame not in body_names:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"[DDS] Cannot publish TF: base frame '{self.base_frame}' is not in SH5 body names. "
                    f"Available bodies: {body_names}"
                )
            return

        now = time.time()
        stamp = Time_(sec=int(now), nanosec=int((now - int(now)) * 1_000_000_000))
        base_id = body_names.index(self.base_frame)
        body_pose_w = self.robot.data.body_link_state_w[0, :, :7]
        base_pose_w = body_pose_w[base_id]
        base_pos_w = base_pose_w[:3].unsqueeze(0)
        base_quat_w = base_pose_w[3:7].unsqueeze(0)

        transforms = []
        for body_id, child_frame in enumerate(body_names):
            if child_frame == self.base_frame:
                continue

            child_pose_w = body_pose_w[body_id]
            child_pos_b, child_quat_b = math_utils.subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                child_pose_w[:3].unsqueeze(0),
                child_pose_w[3:7].unsqueeze(0),
            )
            pos = child_pos_b.squeeze(0).detach().cpu().tolist()
            quat_wxyz = child_quat_b.squeeze(0).detach().cpu().tolist()

            transforms.append(
                TransformStamped_(
                    header=Header_(stamp=stamp, frame_id=self.base_frame),
                    child_frame_id=child_frame,
                    transform=Transform_(
                        translation=Vector3_(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        rotation=Quaternion_(
                            x=float(quat_wxyz[1]),
                            y=float(quat_wxyz[2]),
                            z=float(quat_wxyz[3]),
                            w=float(quat_wxyz[0]),
                        ),
                    ),
                )
            )

        try:
            self.tf_writer.write(TFMessage_(transforms=transforms))
        except Exception as exc:
            print(f"[DDS] tf write error: {exc}")

    def shutdown(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        for reader in self.readers:
            try:
                reader.Close()
            except Exception:
                pass
        try:
            self.joint_state_writer.Close()
        except Exception:
            pass
        try:
            self.tf_writer.Close()
        except Exception:
            pass


def _enabled_topics() -> dict[str, str]:
    topics = {
        "right_arm": args_cli.right_arm_topic,
        "right_hand": args_cli.right_hand_topic,
    }
    if not args_cli.disable_left:
        topics["left_arm"] = args_cli.left_arm_topic
        topics["left_hand"] = args_cli.left_hand_topic
    if not args_cli.disable_head_lift:
        topics["head"] = args_cli.head_topic
        topics["lift"] = args_cli.lift_topic
    return topics


def _print_joint_groups(joint_names: Iterable[str]):
    names = list(joint_names)
    print("[INFO] SH5 articulation joints:")
    for name in names:
        print(f"  - {name}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SH5DdsBridge):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / args_cli.step_hz if args_cli.step_hz > 0 else 0.0
    publish_period = 1.0 / args_cli.publish_hz if args_cli.publish_hz > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()

    while simulation_app.is_running():
        bridge.apply_latest_targets()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        now = time.time()
        if publish_period == 0.0 or now - last_publish >= publish_period:
            bridge.publish_joint_states()
            bridge.publish_tf()
            last_publish = now

        if step_period > 0.0:
            next_step = last_step + step_period
            sleep_time = next_step - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            last_step = next_step if sleep_time > 0.0 else time.time()


def main():
    usd_path = args_cli.usd_path or _default_sh5_usd_path()
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD not found: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / args_cli.step_hz)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.8, -2.2, 1.8], [0.0, 0.0, 0.8])

    scene_cfg = SH5BringupSceneCfg(num_envs=1, env_spacing=2.0)
    robot_pos = (args_cli.robot_x, args_cli.robot_y, args_cli.robot_z)
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.usd_path = usd_path
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    robot_cfg.articulation_root_prim_path = args_cli.articulation_root_prim_path
    robot_cfg.init_state.pos = robot_pos
    robot_cfg.init_state.joint_pos["lift_joint"] = args_cli.lift_init_pos
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.write_data_to_sim()
    _print_joint_groups(robot.data.joint_names)

    domain_id = args_cli.domain_id if args_cli.domain_id is not None else int(os.getenv("ROS_DOMAIN_ID", 0))
    topic_manager = TopicManager(domain_id=domain_id)
    bridge = SH5DdsBridge(
        robot=robot,
        topic_manager=topic_manager,
        topic_names=_enabled_topics(),
        joint_states_topic=args_cli.joint_states_topic,
        tf_topic=args_cli.tf_topic,
        base_frame=args_cli.base_frame,
    )

    print(f"[INFO] FFW SH5 DDS bringup ready. ROS_DOMAIN_ID={domain_id}")
    print(f"[DDS] Publishing joint states: {args_cli.joint_states_topic}")
    print(f"[DDS] Publishing TF: {args_cli.tf_topic} ({args_cli.base_frame} -> robot links)")

    try:
        run_simulator(sim, scene, bridge)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
