"""Official rough task, random initialization; only local USD resolution is substituted."""
import argparse
from pathlib import Path
import hashlib
import json
from asset_manager import verify_assets
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=1500)
parser.add_argument("--num-envs", type=int, default=4096)
add_launcher_args(parser)
a = parser.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_rl.entrypoints.common import scoped_torch_backend_flags
from isaaclab.utils.io import dump_yaml
with scoped_torch_backend_flags(
    cuda_matmul_allow_tf32=True,
    cudnn_allow_tf32=True,
    cudnn_deterministic=False,
    cudnn_benchmark=True,
):
    TASK = "Isaac-Velocity-Rough-UnitreeGo2"
    cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", play_mode=False, overrides=["physics=newton_mjwarp"])
    cfg.scene.robot.spawn.usd_path = str(verify_assets()["Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"])
    cfg.scene.num_envs = a.num_envs
    cfg.seed = agent.seed = 42
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    agent.device = "cuda:0"
    dump_yaml(str(a.output / "env.yaml"), cfg)
    dump_yaml(str(a.output / "agent.yaml"), agent)
    with launch_simulation(cfg, a):
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent.clip_actions)
        try:
            runner = create_rsl_rl_runner(env, agent, log_dir=str(a.output / "logs"))
            runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
            runner.save(str(a.output / "final.pt"))
            (a.output / "complete.json").write_text(json.dumps({"initialization": "random", "seed": 42, "iterations": a.iterations, "sha256": hashlib.sha256((a.output / "final.pt").read_bytes()).hexdigest()}, indent=2))
        finally:
            env.close()
