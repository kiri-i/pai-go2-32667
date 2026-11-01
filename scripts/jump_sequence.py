"""Training-only contact sequence reward. No policy observation or action changes."""
import torch

class JumpSequence:
    def __init__(self,n,device):
        self.takeoff_count=torch.zeros((),device=device,dtype=torch.long)
        self.landing_count=torch.zeros((),device=device,dtype=torch.long)
        self.phase=torch.zeros(n,dtype=torch.long,device=device)
        self.air_steps=torch.zeros_like(self.phase)
        self.stable_steps=torch.zeros_like(self.phase)
        self.takeoff_paid=torch.zeros(n,dtype=torch.bool,device=device)
        self.landing_paid=torch.zeros_like(self.takeoff_paid)

    def update(self,feet,forces,total_contact,root_x,gravity_z,vx,reset,far_x=2.4):
        # phases: 0 waiting, 1 near-bank support, 2 flight, 3 far-bank contact,
        # 4 invalid (straddled, fell, or landed back on near bank), 5 completed.
        for t in [self.phase,self.air_steps,self.stable_steps,self.takeoff_paid,self.landing_paid]:t[reset]=0
        boundary=torch.as_tensor(far_x,device=feet.device)
        if boundary.ndim:boundary=boundary[:,None]
        contact=forces[:,:,2]>1.
        any_contact=contact.any(-1)
        near=any_contact & ((~contact)|((feet[:,:,0]<1.5)&(feet[:,:,2].abs()<.08))).all(-1)
        far=any_contact & ((~contact)|((feet[:,:,0]>boundary)&(feet[:,:,2].abs()<.08))).all(-1)
        free=total_contact<1.
        upright=gravity_z<-.7
        old=self.phase.clone()
        arm=(old==0)&near&upright&(root_x>.7)&(root_x<1.6)&(vx>.3)
        self.phase[arm]=1
        lift=(old==1)&free&upright
        flying=((old==2)|lift)&free&upright
        self.phase[lift]=2
        self.air_steps[flying]+=1
        takeoff=flying&(self.air_steps>=2)&(~self.takeoff_paid)
        self.takeoff_paid[takeoff]=True
        self.takeoff_count+=takeoff.sum()
        land=(old==2)&far&upright&(self.air_steps>=2)
        self.phase[land]=3
        invalid=((old==1)&any_contact&(~near))|((old==2)&(~free)&(~land))|((old>=1)&(old<=3)&(~upright))
        self.phase[invalid]=4
        stable=(self.phase==3)&far&upright&((feet[:,:,0]>boundary)&(feet[:,:,2].abs()<.08)).all(-1)&(feet[:,:,2]>-.02).all(-1)
        self.stable_steps=torch.where(stable,self.stable_steps+1,torch.zeros_like(self.stable_steps))
        completed=stable&(self.stable_steps>=10)&(~self.landing_paid)
        self.phase[completed]=5;self.landing_paid[completed]=True
        self.landing_count+=completed.sum()
        # Manager multiplies by dt=.02: takeoff event=.2, stable landing event=2.
        reward=10.*takeoff.float()+2.*flying.float()*vx.clamp(0.,2.)+100.*completed.float()
        return reward
