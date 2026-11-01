"""Terrain-observed rough Go2 fine-tuning; obstacle exposure and reward controls."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch,trimesh
from asset_manager import verify_assets
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser()
p.add_argument('--kind',choices=['step','gap'],required=True)
p.add_argument('--variant',choices=['standard','agile','task','jump','jump_balance','sequence'],required=True)
p.add_argument('--stage',choices=['first','jump','hard'],default='first')
p.add_argument('--iterations',type=int,default=400)
p.add_argument('--num-envs',type=int,default=1024)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
add_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import resolve_task_config
from isaaclab.managers import RewardTermCfg,TerminationTermCfg
from isaaclab.terrains import TerrainGeneratorCfg,SubTerrainBaseCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,create_rsl_rl_runner,handle_deprecated_rsl_rl_cfg,check_rsl_rl_version
TASK='Isaac-Velocity-Rough-UnitreeGo2'
cfg,agent=resolve_task_config(TASK,'rsl_rl_cfg_entry_point',play_mode=False,overrides=['physics=newton_mjwarp'])
assets=verify_assets();cfg.scene.robot.spawn.usd_path=str(assets['Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'])
terrain_records=[]
def obstacle(difficulty,terrain_cfg):
    value=.18+.12*difficulty if a.kind=='step' else ((.55+.35*difficulty) if a.stage=='hard' else ((.40+.30*difficulty) if a.stage=='jump' else (.20+.25*difficulty)))
    terrain_records.append({'difficulty':float(difficulty),'size_m':float(value)})
    meshes=[]
    def box(start,length,top,width=8.):
        mesh=trimesh.creation.box(extents=(length,width,.1 if top<=0 else top))
        mesh.apply_translation((start+length/2,4.,top-.05 if top<=0 else top/2));meshes.append(mesh)
    if a.kind=='step':
        box(0,8,0);box(3.5,4.5,value,1.8)
    else:
        box(0,3.5,0);box(3.5+value,4.5-value,0);box(0,8,-.4)
    return meshes,np.array([2.,4.,0.])
cfg.scene.terrain.terrain_type='generator'
cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(size=(8.,8.),num_rows=4,num_cols=4,use_cache=False,curriculum=False,difficulty_range=(0.,1.),sub_terrains={'obstacle':SubTerrainBaseCfg(function=obstacle)})
cfg.scene.terrain.max_init_terrain_level=None
cfg.curriculum.terrain_levels=None
cfg.scene.num_envs=a.num_envs;cfg.seed=42;cfg.episode_length_s=8.
cfg.events.push_robot=None;cfg.events.add_base_mass=None;cfg.events.base_com=None
cfg.events.reset_base.params['pose_range']={'x':(-.3,.3),'y':(-.1,.1),'yaw':(-.1,.1)}
cfg.events.reset_base.params['velocity_range']={k:(0.,0.) for k in ['x','y','z','roll','pitch','yaw']}
cfg.events.reset_robot_joints.params['position_range']=(1.,1.)
cfg.events.reset_robot_joints.params['velocity_range']=(0.,0.)
cmd=cfg.commands.base_velocity;cmd.heading_command=True;cmd.rel_heading_envs=1.;cmd.rel_standing_envs=0.;cmd.debug_vis=False
cmd.resampling_time_range=(1e6,1e6);cmd.ranges.heading=(0.,0.);cmd.ranges.lin_vel_y=(0.,0.)
speed=.5 if a.kind=='step' else 1.
cmd.ranges.lin_vel_x=(speed,speed)
# Keep official yaw command range and heading gain; both train and test hold heading zero.
def feet_local(env):
    robot=env.scene['robot'];ids,_=robot.find_bodies('.*_foot')
    return robot.data.body_pos_w.torch[:,ids]-env.scene.env_origins[:,None,:]
def out_of_corridor(env):
    feet=feet_local(env)
    return (feet[:,:,1].abs()>=.9).any(-1)|(feet[:,:,2]<-.12).any(-1)
def course_complete(env):
    r=env.scene['robot'];local=r.data.root_pos_w.torch-env.scene.env_origins
    return (local[:,0]>3.3)&(r.data.projected_gravity_b.torch[:,2]<-.7)
cfg.terminations.corridor=TerminationTermCfg(func=out_of_corridor)
cfg.terminations.completed=TerminationTermCfg(func=course_complete,time_out=True)
# Reward changes only; action scale, actuator limits, observations and all other terms stay fixed.
if a.variant in ['agile','task','jump','jump_balance','sequence']:
    cfg.rewards.lin_vel_z_l2.weight=-.2
    cfg.rewards.ang_vel_xy_l2.weight=-.01
    cfg.rewards.base_height_l2.weight=-10.
    cfg.rewards.action_rate_l2.weight=-.001
    cfg.rewards.dof_torques_l2.weight=-.0001

def crossing_progress(env):
    r=env.scene['robot'];pos=r.data.root_pos_w.torch-env.scene.env_origins
    upright=(r.data.projected_gravity_b.torch[:,2]<-.5).float()
    return r.data.root_lin_vel_w.torch[:,0].clamp(-1.5,1.5)*upright*(pos[:,1].abs()<.7).float()
def far_support(env):
    r=env.scene['robot'];feet=feet_local(env);s=env.scene['contact_forces'];ids,_=s.find_bodies('.*_foot')
    after=(feet[:,:,0]>(2.5 if a.stage=='hard' else (2.3 if a.stage=='jump' else 2.05))).all(-1)
    supported=(s.data.net_forces_w.torch[:,ids,2]>1.).any(-1)
    return (after&supported&(r.data.projected_gravity_b.torch[:,2]<-.7)).float()
if a.variant in ['task','jump','jump_balance','sequence']:
    cfg.rewards.crossing_progress=RewardTermCfg(func=crossing_progress,weight=2.)
    cfg.rewards.far_support=RewardTermCfg(func=far_support,weight=4.)
def takeoff(env):
    r=env.scene['robot'];pos=r.data.root_pos_w.torch-env.scene.env_origins
    zone=(pos[:,0]>1.05)&(pos[:,0]<1.65)&(pos[:,1].abs()<.7)
    upright=r.data.projected_gravity_b.torch[:,2]<-.5
    return zone.float()*upright.float()*r.data.root_lin_vel_w.torch[:,2].clamp(0.,1.5)/1.5

def airborne_progress(env):
    r=env.scene['robot'];pos=r.data.root_pos_w.torch-env.scene.env_origins
    free=torch.linalg.vector_norm(env.scene['contact_forces'].data.net_forces_w.torch,dim=-1).sum(-1)<1.
    zone=(pos[:,0]>1.35)&(pos[:,0]<(2.5 if a.stage=='hard' else 2.3))&(pos[:,1].abs()<.7)
    return zone.float()*free.float()*r.data.root_lin_vel_w.torch[:,0].clamp(0.,2.)
if a.variant in ['jump','jump_balance','sequence']:
    cfg.rewards.takeoff=RewardTermCfg(func=takeoff,weight=4.)
    cfg.rewards.airborne_progress=RewardTermCfg(func=airborne_progress,weight=1.)
def sideways(env):
    r=env.scene['robot'];pos=r.data.root_pos_w.torch-env.scene.env_origins
    return pos[:,1].square()+.5*r.data.root_lin_vel_w.torch[:,1].square()
def roll_error(env):
    return env.scene['robot'].data.projected_gravity_b.torch[:,1].square()
def flight_clearance(env):
    r=env.scene['robot'];pos=r.data.root_pos_w.torch-env.scene.env_origins
    zone=(pos[:,0]>1.35)&(pos[:,0]<(2.5 if a.stage=='hard' else 2.3))&(r.data.projected_gravity_b.torch[:,2]<-.5)
    raised=(feet_local(env)[:,:,2].amin(-1)-.03).clamp(0.,.12)/.12
    return zone.float()*raised*r.data.root_lin_vel_w.torch[:,0].clamp(0.,1.5)/1.5
if a.variant in ['jump_balance','sequence']:
    cfg.rewards.takeoff.weight=2.
    cfg.rewards.sideways=RewardTermCfg(func=sideways,weight=-2.)
    cfg.rewards.roll_error=RewardTermCfg(func=roll_error,weight=-1.)
    cfg.rewards.flight_clearance=RewardTermCfg(func=flight_clearance,weight=2.)
def contact_sequence(env):
    from jump_sequence import JumpSequence
    if not hasattr(env,'_jump_sequence_reward'):
        env._jump_sequence_reward=JumpSequence(env.num_envs,env.device)
        env._jump_far_edges=1.5+torch.tensor([r['size_m'] for r in terrain_records],device=env.device)
    r=env.scene['robot'];sensor=env.scene['contact_forces'];ids,_=sensor.find_bodies('.*_foot')
    total=torch.linalg.vector_norm(sensor.data.net_forces_w.torch,dim=-1).sum(-1)
    return env._jump_sequence_reward.update(feet_local(env),sensor.data.net_forces_w.torch[:,ids],total,
        (r.data.root_pos_w.torch-env.scene.env_origins)[:,0],r.data.projected_gravity_b.torch[:,2],
        r.data.root_lin_vel_w.torch[:,0],env.episode_length_buf<=1,far_x=env._jump_far_edges[env.scene['terrain'].terrain_levels*4+env.scene['terrain'].terrain_types])
if a.variant=='sequence':
    cfg.rewards.contact_sequence=RewardTermCfg(func=contact_sequence,weight=1.)
agent=handle_deprecated_rsl_rl_cfg(agent,check_rsl_rl_version());agent.device='cuda:0';agent.algorithm.learning_rate=3e-4;agent.save_interval=100
assert cfg.scene.height_scanner is not None and cfg.observations.policy.height_scan is not None
dump_yaml(str(a.output/'env.yaml'),cfg);dump_yaml(str(a.output/'agent.yaml'),agent)
meta={'stage':a.stage,'kind':a.kind,'variant':a.variant,'iterations':a.iterations,'num_envs':a.num_envs,'seed':42,'initial_checkpoint':str(a.checkpoint.resolve()),'initial_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'observation':'235 = 48 + 187 terrain scan, actor and critic','heading_hold':True,'speed_m_s':speed,'train_dimension_range_m':[.18,.30] if a.kind=='step' else ([.55,.90] if a.stage=='hard' else ([.40,.70] if a.stage=='jump' else [.20,.45])),'training_success_is_not_evaluation_success':True,'optimizer':'fresh; actor and critic loaded'}
(a.output/'meta.json').write_text(json.dumps(meta,indent=2))
with launch_simulation(cfg,a):
    env=RslRlVecEnvWrapper(gym.make(TASK,cfg=cfg),clip_actions=agent.clip_actions)
    try:
        assert env.get_observations()['policy'].shape==(a.num_envs,235)
        (a.output/'terrain_samples.json').write_text(json.dumps(terrain_records,indent=2))
        runner=create_rsl_rl_runner(env,agent,log_dir=str(a.output/'logs'))
        runner.load(str(a.checkpoint),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})
        start=time.monotonic();runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
        runner.save(str(a.output/'final.pt'))
        if a.variant=='sequence':
            tracker=env.unwrapped._jump_sequence_reward
            (a.output/'sequence_events.json').write_text(json.dumps({'takeoffs':int(tracker.takeoff_count.item()),'stable_landings':int(tracker.landing_count.item()),'note':'training reward events, not evaluation successes'}))
        (a.output/'complete.json').write_text(json.dumps({'elapsed_s':time.monotonic()-start,'transitions':a.iterations*24*a.num_envs,'sha256':hashlib.sha256((a.output/'final.pt').read_bytes()).hexdigest()}))
    finally:env.close()
