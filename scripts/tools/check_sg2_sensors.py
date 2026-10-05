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

"""Check SG2 cameras, ideal wrist depth, and both RTX LiDARs on two robot instances.

Run in the cyclo_lab Isaac Lab environment:
    python scripts/tools/check_sg2_sensors.py --headless

Camera rendering is enabled automatically. Images and LiDAR returns are saved
under outputs/sg2_sensors. Test fixtures are outside the robot USD.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", default="outputs/sg2_sensors")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import isaaclab.sim as sim_utils
import numpy as np
from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG
from cyclo_lab.manager_based.manipulation.pick_place.config.ffw_sg2.joint_pos_env_cfg import (
    FFWSG2PickPlaceEnvCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from pxr import Gf, UsdGeom

task_cfg = FFWSG2PickPlaceEnvCfg()
CAMERAS = ("cam_head", "cam_wrist_left", "cam_wrist_right")
LIDARS = ("lidar_l", "lidar_r")


@configclass
class SensorSceneCfg(InteractiveSceneCfg):
    robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cam_head = task_cfg.scene.cam_head.copy()
    cam_wrist_left = task_cfg.scene.cam_wrist_left.copy()
    cam_wrist_right = task_cfg.scene.cam_wrist_right.copy()
    lidar_l = task_cfg.scene.lidar_l.copy()
    lidar_r = task_cfg.scene.lidar_r.copy()


def step(sim, scene, count):
    for _ in range(count):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed before sensor checks finished.")
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def main():
    # Disable Fabric only for this check so USD pose queries follow physics updates.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, use_fabric=False))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)
    scene = InteractiveScene(SensorSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=False))
    sim.reset()
    robot = scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.reset()
    step(sim, scene, 20)

    # Test fixtures: green cubes 0.25 m in front of each camera, and walls whose
    # front faces are 1.5 m, then 2.0 m, along each LiDAR's local +X direction.
    cache = UsdGeom.XformCache()
    cube = sim_utils.CuboidCfg(
        size=(0.08, 0.08, 0.08), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.8, 0.05))
    )
    wall = sim_utils.CuboidCfg(size=(0.1, 0.8, 0.8))
    walls = []
    for env_id in range(2):
        for name in CAMERAS + LIDARS:
            sensor = scene[name]
            assert sensor.num_instances == 2, f"Missing sensor clone: {name}"
            path = sensor.cfg.prim_path.replace("env_.*", f"env_{env_id}")
            prim = sim.stage.GetPrimAtPath(path)
            transform = cache.GetLocalToWorldTransform(prim)
            quat = transform.ExtractRotationQuat()
            orientation = (quat.GetReal(), *quat.GetImaginary())
            fixture = f"/World/Target_{name}_{env_id}"
            if name in CAMERAS:
                position = transform.Transform(Gf.Vec3d(0.0, 0.0, -0.25))
                cube.func(fixture, cube, translation=tuple(position), orientation=orientation)
            else:
                position = transform.Transform(Gf.Vec3d(1.55, 0.0, 0.0))
                wall.func(fixture, wall, translation=tuple(position), orientation=orientation)
                walls.append((fixture, transform))

    output = Path(args_cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for phase, expected_distance in enumerate((1.5, 2.0)):
        if phase:
            for path, transform in walls:
                position = transform.Transform(Gf.Vec3d(expected_distance + 0.05, 0.0, 0.0))
                sim.stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set(position)
        seen = set()
        for _ in range(300):
            step(sim, scene, 1)
            for name in LIDARS:
                data = scene[name].data
                for env_id, points in enumerate(data.point_clouds):
                    if len(points) == 0:
                        continue
                    assert np.isfinite(points).all(), f"Non-finite {name} returns."
                    assert np.all((data.ranges[env_id] >= 0.04) & (data.ranges[env_id] <= 20.05))
                    # Test tolerance accommodates URDF noise and range quantization.
                    hit = (np.abs(points[:, 0] - expected_distance) < 0.05) & (np.abs(points[:, 1]) < 0.3)
                    if hit.any():
                        seen.add((name, env_id))
                        np.save(output / f"{name}_{env_id}_phase{phase}.npy", points)
        assert len(seen) == 4, f"Missing LiDAR wall measurements in phase {phase}: {seen}"

    for name in CAMERAS:
        sensor = scene[name]
        rgb = sensor.data.output["rgb"].cpu().numpy()[..., :3]
        assert rgb.shape == (2, sensor.cfg.height, sensor.cfg.width, 3), name
        for env_id, image in enumerate(rgb):
            pixels = image.astype(np.float32)
            green = (pixels[..., 1] > pixels[..., 0] + 30) & (pixels[..., 1] > pixels[..., 2] + 30)
            assert green.any(), f"Camera cannot see its green target: {name}, environment {env_id}."
            assert cv2.imwrite(str(output / f"{name}_{env_id}.png"), image[..., ::-1])
            if name != "cam_head":
                depth = sensor.data.output["distance_to_image_plane"][env_id, ..., 0].cpu().numpy()
                # Target front is 0.25 - 0.08/2 = 0.21 m from the camera plane.
                assert np.abs(np.median(depth[green]) - 0.21) < 0.03, f"Unexpected wrist depth: {name}"
    print(f"PASS: cameras, wrist depth, and moving-wall LiDAR measurements on both clones. Results: {output}")


try:
    main()
finally:
    simulation_app.close()
