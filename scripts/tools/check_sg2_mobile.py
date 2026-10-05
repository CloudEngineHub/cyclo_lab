# Copyright 2026 ROBOTIS CO., LTD.
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

"""Check SG2 wheel contact, continuous rotation, forward motion, and sideways motion.

Run in the cyclo_lab Isaac Lab environment:
    python scripts/tools/check_sg2_mobile.py --headless

This checks the mobile asset configuration, rather than the fixed-base pick-and-place task.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaacsim.core.utils.prims import create_prim

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from cyclo_lab.assets.robots.FFW_SG2 import (
    FFW_SG2_CFG,
    FFW_SG2_MOBILE_CFG,
    SG2_SWERVE_STEERING_JOINTS,
    SG2_SWERVE_WHEEL_JOINTS,
    SG2_SWERVE_WHEEL_RADIUS,
)

# Test commands: 10 rad/s for 3 s exceeds the original +/-3-turn drive limit.
WHEEL_SPEED = 10.0
DRIVE_SECONDS = 3.0


def _step(sim: sim_utils.SimulationContext, robot: Articulation, seconds: float) -> torch.Tensor:
    """Advance the simulation and measure joint travel, including wrapped angles."""
    dt = sim.get_physics_dt()
    joint_travel = torch.zeros_like(robot.data.joint_vel)
    for _ in range(round(seconds / dt)):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed before the SG2 check finished.")
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        assert torch.isfinite(robot.data.root_state_w).all(), "Non-finite SG2 root state."
        assert torch.isfinite(robot.data.joint_vel).all(), "Non-finite SG2 joint velocity."
        # Test tolerances: stay near the floor and within 10 degrees of upright.
        assert (robot.data.root_pos_w[:, 2].abs() < SG2_SWERVE_WHEEL_RADIUS).all(), "SG2 lost floor support."
        quat = robot.data.root_quat_w
        vertical_axis_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        assert (vertical_axis_z > math.cos(math.radians(10.0))).all(), "SG2 tilted more than 10 degrees."
        joint_travel += robot.data.joint_vel * dt
    return joint_travel


def main() -> None:
    """Run the same bounded driving checks on two cloned SG2 robots."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args_cli.device))
    sim.set_camera_view((4.0, 6.0, 3.0), (0.0, 2.0, 0.8))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)

    # Keep the two robots far enough apart to avoid collisions during the check.
    origins = ((0.0, 0.0, 0.0), (0.0, 4.0, 0.0))
    for index, origin in enumerate(origins):
        create_prim(f"/World/env_{index}", "Xform", translation=origin)
    robot = Articulation(FFW_SG2_MOBILE_CFG.replace(prim_path="/World/env_.*/Robot"))
    sim.reset()

    assert robot.num_instances == 2, "SG2 cloning did not create two articulations."
    assert not robot.is_fixed_base, "SG2 is still anchored to the world."
    assert FFW_SG2_CFG.spawn.rigid_props.disable_gravity, "Mobile configuration changed the fixed-base configuration."
    assert "base_drive" not in FFW_SG2_CFG.actuators, "Mobile actuators leaked into the fixed-base configuration."

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += torch.tensor(origins, device=robot.device)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    position_targets = robot.data.default_joint_pos.clone()
    velocity_targets = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(position_targets, velocity_targets)
    robot.reset()
    robot.set_joint_position_target(position_targets)
    robot.set_joint_velocity_target(velocity_targets)
    _step(sim, robot, 2.0)

    drive_ids, _ = robot.find_joints(SG2_SWERVE_WHEEL_JOINTS)
    steer_ids, _ = robot.find_joints(SG2_SWERVE_STEERING_JOINTS)
    start = robot.data.root_pos_w.clone()
    velocity_targets[:, drive_ids] = WHEEL_SPEED
    robot.set_joint_velocity_target(velocity_targets)
    travel = _step(sim, robot, DRIVE_SECONDS)
    forward = robot.data.root_pos_w - start
    assert (travel[:, drive_ids] > 6.0 * math.pi).all(), "A drive wheel stopped before completing three turns."
    assert (forward[:, 0] > 0.1).all(), f"Forward movement below 0.1 m: {forward.tolist()}"
    print(f"[PASS] Forward displacement (m): {forward.tolist()}")

    velocity_targets[:, drive_ids] = 0.0
    robot.set_joint_velocity_target(velocity_targets)
    _step(sim, robot, 1.0)
    position_targets[:, steer_ids] = math.pi / 2.0
    robot.set_joint_position_target(position_targets)
    _step(sim, robot, 1.0)
    start = robot.data.root_pos_w.clone()
    velocity_targets[:, drive_ids] = WHEEL_SPEED
    robot.set_joint_velocity_target(velocity_targets)
    _step(sim, robot, DRIVE_SECONDS)
    sideways = robot.data.root_pos_w - start
    assert (sideways[:, 1] > 0.1).all(), f"Sideways movement below 0.1 m: {sideways.tolist()}"
    print(f"[PASS] Sideways displacement (m): {sideways.tolist()}")

    velocity_targets[:, drive_ids] = 0.0
    robot.set_joint_velocity_target(velocity_targets)
    _step(sim, robot, 1.0)
    print("[PASS] SG2 mobile checks completed for both cloned robots.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
