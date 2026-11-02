"""Evaluate a supplied Go2 checkpoint in simulation; never commands hardware."""

import argparse
import hashlib
import json
import math
import time
import subprocess
from pathlib import Path

import sys
from asset_manager import manifest_entries, verify_assets

ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-Velocity-Flat-UnitreeGo2"
BACKENDS = {"physx": "isaacsim_physx", "newton": "newton_mjwarp"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--go2-backend", choices=BACKENDS, default="physx")
parser.add_argument("--height", type=float, default=0.1)
parser.add_argument("--flat-plane", action="store_true", help="Use original plane terrain for flat baseline diagnosis")
parser.add_argument("--nominal-physics", action="store_true", help="Disable mass and COM randomization for matched evaluation")
parser.add_argument("--stair-count", type=int, default=1)
parser.add_argument("--tread", type=float, default=0.3)
parser.add_argument("--course-file", type=Path)
parser.add_argument("--suite-case", help="Case ID within a course suite JSON")
parser.add_argument("--stage", type=int, help="1-based independent stage start; default traverses whole course")
parser.add_argument("--camera", choices=("follow", "overview", "fixed"), default="follow")
parser.add_argument("--heading-hold", action="store_true")
parser.add_argument("--approach-offset", type=float, default=0.0)
parser.add_argument("--rough", action="store_true", help="公式不整地版の235次元入力を使用")
parser.add_argument("--video-dir", type=Path)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--start-after", type=int, default=0, help="このstep数は静止指令とし、その後前進")
parser.add_argument("--stop-after", type=int, help="このstep数の前進後に速度指令をゼロへ変更")
parser.add_argument("--realtime", action="store_true", help="各stepを実時間に合わせる")
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument(
    "--checkpoint", type=Path, required=True, help="評価する学習済みcheckpoint"
)
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
args = parser.parse_args()
if args.suite_case and (not args.course_file or args.stage is not None):
    parser.error("--suite-case requires --course-file and cannot be combined with --stage")
if not 0 <= args.height <= 0.4:
    parser.error("height must be 0..0.4 m")
if not 1 <= args.stair_count <= 6 or not 0.2 <= args.tread <= 0.6:
    parser.error("stair-count must be 1..6 and tread 0.2..0.6 m")
if not math.isfinite(args.approach_offset) or abs(args.approach_offset) > .5:
    parser.error("approach-offset must be within +/-0.5m")
TASK = "Isaac-Velocity-Rough-UnitreeGo2" if args.rough else TASK
PHYSICS = BACKENDS[args.go2_backend]
CHECKPOINT = args.checkpoint.resolve()
if args.steps <= 0 or not math.isfinite(args.speed) or not 0.0 <= args.speed <= 1.0:
    parser.error("--steps must be positive and --speed must be in [0, 1]")
if not 0 <= args.start_after < args.steps:
    parser.error("--start-after must be between 0 and steps - 1")
if args.stop_after is not None and not args.start_after < args.stop_after < args.steps:
    parser.error("--stop-after must be greater than --start-after and less than steps")
if not CHECKPOINT.is_file():
    parser.error(f"checkpointがありません: {CHECKPOINT}")
checkpoint_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
checkpoint_source = "provided_checkpoint"
if args.output.exists() or args.output.is_symlink():
    parser.error("既存の評価結果は上書きできません")
try:
    assets = verify_assets()
    _, asset_hashes = manifest_entries()
except (OSError, ValueError, RuntimeError) as error:
    parser.error(f"Go2固定資産を検証できません: {error}. scripts/asset_manager.py prepareを実行してください")

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

cfg, agent_cfg = resolve_task_config(
    TASK, "rsl_rl_cfg_entry_point", play_mode=True, overrides=[f"physics={PHYSICS}"]
)
cfg.scene.robot.spawn.usd_path = str(assets["Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"])
if not args.rough:
    if cfg.scene.height_scanner is not None or cfg.observations.policy.height_scan is not None:
        raise RuntimeError("Current baseline must not contain terrain observation")

# Controlled single step: spawn at origin, leading edge x=1.0 m.
import numpy as np
import trimesh
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg

# Each block has a vertical riser and flat tread; final block is a landing.
blocks = [(1.0 + i * args.tread,
           (args.tread if i < args.stair_count - 1 else (1.2 if args.stair_count > 1 else 5.0)),
           args.height * (i + 1)) for i in range(args.stair_count)]
course_width = 1.8 if args.stair_count > 1 else 8.0
stage_specs = []
spawn_x = spawn_z = 0.0
course_config = {}
if args.course_file:
    from course_layout import build_course
    course_config = json.loads(args.course_file.read_text())
    if "cases" in course_config:
        matches = [case for case in course_config["cases"] if case["id"] == args.suite_case]
        if len(matches) != 1:
            parser.error("Choose an existing --suite-case")
        course_config = matches[0]
    blocks, stage_specs, course_width = build_course(course_config)
    if args.stage is not None:
        if not 1 <= args.stage <= len(stage_specs):
            parser.error("stage outside course")
        entry = stage_specs[args.stage - 1]
        spawn_x = entry["entry_x_m"] - 1.0
        spawn_z = entry["base_height_m"]
elif args.stage is not None:
    parser.error("--stage requires --course-file")
floor_length = max(12., 2 * (blocks[-1][0] + blocks[-1][1] + 2.),
                   2 * (args.speed * args.steps * 0.02 + 2.))
center_x = floor_length / 2

def single_step(difficulty, terrain_cfg):
    meshes = []
    spans = [(-floor_length / 2, floor_length, 0.)]
    if course_config.get("kind") == "gap":
        edge = course_config["first_edge_m"]
        far = edge + course_config["gap_m"]
        spans = [(-floor_length/2, edge+floor_length/2, 0.),
                 (far, floor_length/2-far, 0.),
                 (-floor_length/2, floor_length, -course_config["depth_m"])]
    for start, length, top in spans:
        floor = trimesh.creation.box(extents=(length, 8, 0.1))
        floor.apply_translation((center_x + start + length/2, 4, top - 0.05))
        meshes.append(floor)
    if args.height > 0 or args.course_file:
        for edge, length, top in blocks:
            if top <= 1e-8:
                continue
            platform = trimesh.creation.box(extents=(length, course_width, top))
            platform.apply_translation((center_x + edge + length / 2, 4, top / 2))
            meshes.append(platform)
    return meshes, np.array([center_x, 4., 0.])

if args.flat_plane:
    if args.height != 0 or args.course_file:
        parser.error("--flat-plane requires --height 0 and no course")
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
else:
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(size=(floor_length, 8.), num_rows=1, num_cols=1, use_cache=False, sub_terrains={"step": SubTerrainBaseCfg(function=single_step)})
cfg.scene.terrain.max_init_terrain_level = 0
from isaaclab import sim as sim_utils
if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.color_scheme = "none"
cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.52, 0.55))
cfg.curriculum.terrain_levels = None
cfg.events.reset_base.params["pose_range"] = {k: (0., 0.) for k in ("x", "y", "yaw")}
cfg.events.reset_base.params["pose_range"].update(x=(spawn_x + args.approach_offset, spawn_x + args.approach_offset), z=(spawn_z, spawn_z))
cfg.events.reset_base.params["velocity_range"] = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
cfg.events.reset_robot_joints.params["position_range"] = (1., 1.)
cfg.events.push_robot = None
if args.nominal_physics:
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
if args.video_dir and (args.height > 0 or args.course_file):
    # Visual-only top surface and edge markers; collision terrain is unchanged.
    from isaaclab.assets import AssetBaseCfg
    from isaaclab import sim as sim_utils
    for i, (edge, length, top) in enumerate(blocks):
        if top <= 1e-8:
            continue
        color = (0.13, 0.33, 0.52) if i % 2 == 0 else (0.28, 0.57, 0.68)
        setattr(cfg.scene, f"platform_visual_{i}", AssetBaseCfg(
            prim_path=f"/World/PlatformVisual{i}",
            spawn=sim_utils.CuboidCfg(size=(length + 0.002, course_width + 0.002, top + 0.002), collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(edge + length / 2, 0., top / 2)),
        ))
        setattr(cfg.scene, f"edge_visual_{i}", AssetBaseCfg(
            prim_path=f"/World/StepEdgeVisual{i}",
            spawn=sim_utils.CuboidCfg(size=(0.018, course_width, 0.002), collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., 0.35, 0.05))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(edge, 0., top + 0.003)),
        ))
cfg.scene.num_envs = 1
cfg.seed = 42
cfg.observations.policy.enable_corruption = False
command = cfg.commands.base_velocity
command.heading_command = args.heading_hold
command.rel_heading_envs = 1.0
command.ranges.heading = (0.0, 0.0)
command.rel_standing_envs = 0.0
command.resampling_time_range = (1.0e6, 1.0e6)
command.ranges.lin_vel_x = (args.speed, args.speed)
command.ranges.lin_vel_y = (0.0, 0.0)
if not args.heading_hold:
    command.ranges.ang_vel_z = (0.0, 0.0)
cfg.episode_length_s = max(float(cfg.episode_length_s), args.steps * 0.02 + 10.0)
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
from isaaclab.utils.io import dump_yaml
args.output.parent.mkdir(parents=True, exist_ok=True)
dump_yaml(str(args.output.with_suffix(".env.yaml")), cfg)
if args.video_dir:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    if args.video_dir.exists():
        parser.error("video directory already exists; choose a new run")
    cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(2.2, -2.5, 1.1), lookat=(1.4, 0., 0.25), window_width=1280, window_height=720)]
    args.video_dir.mkdir(parents=True)

with launch_simulation(cfg, args):
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    encoder = None
    video_frames = 0
    try:
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(str(CHECKPOINT))
        if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != checkpoint_sha:
            raise RuntimeError("評価中にcheckpointが変更されました")
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        robot = env.unwrapped.scene["robot"]
        origin = env.unwrapped.scene.env_origins[0].detach().cpu().clone()
        start = robot.data.root_pos_w.torch[0].detach().cpu().clone() - origin
        initial_quat = robot.data.root_quat_w.torch[0].detach().cpu().clone()
        foot_ids, _ = robot.find_bodies(".*_foot")
        if len(foot_ids) != 4:
            raise RuntimeError("Expected four feet")
        contact_sensor = env.unwrapped.scene["contact_forces"]
        contact_ids, contact_names = contact_sensor.find_bodies(".*_foot")
        foot_names = [robot.body_names[i] for i in foot_ids]
        contact_ids = [contact_ids[contact_names.index(name)] for name in foot_names]
        from course_metrics import LandingGoal
        goal = LandingGoal(blocks[-1], course_width, env.unwrapped.step_dt) if args.suite_case else None
        stage_goals = []
        if goal:
            cursor = 0
            for stage in stage_specs:
                cursor += stage["count"]
                stage_goals.append((stage["id"], LandingGoal(blocks[cursor - 1], course_width, env.unwrapped.step_dt)))
        samples = []
        error = None
        min_height = float(start[2])
        for index in range(args.steps):
            step_start = time.monotonic()
            if args.video_dir:
                p = robot.data.root_pos_w.torch[0].detach().cpu().tolist()
                if args.camera == "fixed":
                    env.unwrapped.sim.set_camera_view((0.6, -2.5, 1.0), (0.2, 0., 0.4))
                elif args.camera == "follow" or not args.course_file:
                    env.unwrapped.sim.set_camera_view((p[0] + 0.5, -3.4, p[2] + 1.0), (p[0] + 0.8, 0., p[2] - 0.10))
                else:
                    mid = (blocks[-1][0] + blocks[-1][1]) / 2
                    env.unwrapped.sim.set_camera_view((mid + 3., -16., 8.), (mid, 0., 0.6))
            with torch.inference_mode():
                current_command = env.unwrapped.command_manager.get_command("base_velocity")[0]
                target_speed = (
                    0.0
                    if index < args.start_after or (args.stop_after is not None and index >= args.stop_after)
                    else args.speed
                )
                if args.start_after or args.stop_after is not None:
                    current_command[:2] = torch.tensor([target_speed, 0.0], device=current_command.device)
                    if not args.heading_hold:
                        current_command[2] = 0.0
                    obs["policy"][0, 9:12] = current_command
                expected_command = torch.tensor([target_speed, 0.0, float(current_command[2]) if args.heading_hold else 0.0], device=current_command.device)
                if (
                    obs["policy"].shape != (1, 235 if args.rough else 48)
                    or not torch.allclose(current_command, expected_command, atol=1e-5, rtol=0.0)
                    or not torch.allclose(obs["policy"][0, 9:12], current_command, atol=1e-5, rtol=0.0)
                ):
                    error = f"command or observation contract mismatch at step {index}: {current_command.tolist()}"
                    break
                if not torch.isfinite(obs["policy"]).all():
                    error = f"nonfinite policy observation at step {index}"
                    break
                observation_before = obs["policy"][0].detach().cpu().clone()
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    error = f"nonfinite actions at step {index}"
                    break
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            if bool(dones[0]):
                error = f"environment terminated/reset at step {index + 1}"
                break
            if args.video_dir:
                visualizer = next(v for v in env.unwrapped.sim.visualizers if v.cfg.visualizer_type == "newton_gl")
                frame = visualizer.render_rgb_array()
                if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise RuntimeError("Invalid RGB frame")
                if encoder is None:
                    h, w, _ = frame.shape
                    encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(1.0 / env.unwrapped.step_dt), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(args.video_dir / "raw.mp4")], stdin=subprocess.PIPE)
                encoder.stdin.write(frame.tobytes())
                video_frames += 1
            position = robot.data.root_pos_w.torch[0].detach().cpu().clone() - origin
            quat = robot.data.root_quat_w.torch[0].detach().cpu().clone()
            gravity = robot.data.projected_gravity_b.torch[0].detach().cpu().clone()
            linear_velocity = robot.data.root_lin_vel_b.torch[0].detach().cpu().clone()
            if not (
                torch.isfinite(position).all()
                and torch.isfinite(quat).all()
                and torch.isfinite(gravity).all()
                and torch.isfinite(linear_velocity).all()
            ):
                error = f"nonfinite state at step {index}"
                break
            if args.realtime:
                time.sleep(max(0.0, env.unwrapped.step_dt - (time.monotonic() - step_start)))
            min_height = min(min_height, float(position[2]))
            samples.append(
                {
                    "step": index + 1,
                    "feet_positions_m": (robot.data.body_pos_w.torch[0, foot_ids].detach().cpu() - origin).tolist(),
                    "feet_contact_forces_n": contact_sensor.data.net_forces_w.torch[0, contact_ids].detach().cpu().tolist(),
                    "total_body_contact_norm_n": float(torch.linalg.vector_norm(contact_sensor.data.net_forces_w.torch[0], dim=-1).sum()),
                    "joint_efforts_nm": robot.actuators.applied_effort.torch[0].detach().cpu().tolist(),
                    "position_m": position.tolist(),
                    "linear_velocity_body_m_s": linear_velocity.tolist(),
                    "quaternion_xyzw": quat.tolist(),
                    "projected_gravity": gravity.tolist(),
                    "command_before_step": current_command.detach().cpu().tolist(),
                    "policy_observation_before_step": observation_before.tolist(),
                    "action_before_step": actions[0].detach().cpu().tolist(),
                }
            )
            if goal:
                if course_config.get("kind") == "gap" and any(
                    foot[2] < -0.05 and force[2] > 1.
                    for foot, force in zip(samples[-1]["feet_positions_m"], samples[-1]["feet_contact_forces_n"])
                ):
                    error = "gap below-bank contact"
                    break
                for _, stage_goal in stage_goals:
                    stage_goal.update(samples[-1]["feet_positions_m"], samples[-1]["feet_contact_forces_n"], float(gravity[2]))
                goal.update(samples[-1]["feet_positions_m"], samples[-1]["feet_contact_forces_n"], float(gravity[2]))
                if goal.off_course:
                    error = "course lateral boundary crossed by a foot"
                    break
                if goal.success:
                    break
        end = samples[-1]["position_m"] if samples else start.tolist()
        x, y, z, w = (float(v) for v in initial_quat)
        heading = (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z))
        forward = (end[0] - float(start[0])) * heading[0] + (end[1] - float(start[1])) * heading[1]
        upright = bool(samples) and all(sample["projected_gravity"][2] < -0.7 for sample in samples)
        passed = (
            error is None and len(samples) == args.steps and min_height >= 0.2 and forward >= 0.15 and upright
        )
        result = {
            "heading_hold": args.heading_hold,
            "approach_offset_m": args.approach_offset,
            "terrain_type": cfg.scene.terrain.terrain_type,
            "nominal_physics": args.nominal_physics,
            "body_masses_kg": robot.data.body_mass.torch[0].detach().cpu().tolist(),
            "body_com_positions_b_m": robot.data.body_com_pos_b.torch[0].detach().cpu().tolist(),
            "suite_case": args.suite_case,
            "course_definition": course_config,
            "step_height_m": None if args.course_file else args.height,
            "stage_specs": stage_specs,
            "independent_stage_start": args.stage,
            "course_file_sha256": hashlib.sha256(args.course_file.read_bytes()).hexdigest() if args.course_file else None,
            "course": {"stair_count": args.stair_count, "tread_m": args.tread, "width_m": course_width, "blocks_edge_length_height_m": blocks},
            "observation_contract": "official_rough_235_with_height_scan" if args.rough else "proprioceptive_48_no_terrain",
            "course_success": goal.success if goal else None,
            "stages_completed": [name for name, stage_goal in stage_goals if stage_goal.success],
            "course_outcome": ("success" if goal.success else "off_course" if goal.off_course else "terminated" if error else "timeout") if goal else None,
            "goal_rule": "no foot outside course width; all feet in final landing and upright for 1s; each foot contact >1N in that window and >=80% frames have support" if goal else None,
            "foot_names": foot_names,
            "video_dir": str(args.video_dir) if args.video_dir else None,
            "video_frames": video_frames,
            "camera": args.camera,
            "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "step_edge_x_m": 1.0,
            "probe_note": "passed is inherited locomotion smoke criterion, NOT step success",
            "task": TASK,
            "physics": PHYSICS,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_source": checkpoint_source,
            "robot_usd_sha256": asset_hashes["Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"],
            "robot_mesh_sha256": asset_hashes[
                "Isaac/IsaacLab/Robots/Unitree/Go2/Props/instanceable_meshes.usd"
            ],
            "seed": 42,
            "command_m_s": args.speed,
            "start_after_steps": args.start_after,
            "stop_after_steps": args.stop_after,
            "steps_requested": args.steps,
            "step_dt_s": env.unwrapped.step_dt,
            "start_position_m": start.tolist(),
            "initial_quaternion_xyzw": initial_quat.tolist(),
            "end_position_m": end,
            "end_position_is_last_pre_reset_observation": True,
            "forward_distance_m": forward,
            "minimum_height_m": min_height,
            "upright": upright,
            "error": error,
            "passed": passed,
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
        print(
            f"GO2_LAB30_EVAL passed={passed} forward_m={forward:.4f} min_height_m={min_height:.4f} output={args.output}",
            flush=True,
        )
        if (goal and not goal.success) or (not goal and not passed):
            raise RuntimeError("Go2 checkpoint evaluation did not satisfy the movement and posture checks")
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise RuntimeError("Video encoding failed")
        env.close()
