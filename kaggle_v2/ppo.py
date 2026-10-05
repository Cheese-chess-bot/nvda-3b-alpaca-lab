"""Clipped PPO with GAE on CPU, NVIDIA CUDA, AMD ROCm or Intel XPU; optional DDP."""
import os, random, math
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from core import *
from hardware import resolve_device, clear_cache


class ActorCritic(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(dimension,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.actor=nn.Linear(128,len(ACTIONS)); self.critic=nn.Linear(128,1)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight,math.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,.01); nn.init.orthogonal_(self.critic.weight,1.)
    def forward(self,x):
        h=self.body(x); return self.actor(h),self.critic(h).squeeze(-1)


def save_torch(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def choose_model(model,device):
    def choose(obs,_):
        with torch.no_grad():
            logits,_=model(torch.as_tensor(obs,device=device).unsqueeze(0))
            return int(logits.argmax(-1).item())
    return choose


def evaluate(model,part,device,cost=POLICY['cost_bps']):
    model.eval()
    result=run_backtest(part['x'],part['returns'],choose_model(model,device),cost)
    model.train()
    return result


class SingleDevice(nn.Module):
    def __init__(self,module):
        super().__init__(); self.module=module
    def forward(self,x): return self.module(x)


def train():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    device=resolve_device(rank=rank); distributed=world>1
    torch.set_num_threads(max(1,int(os.environ.get('NVDA_CPU_THREADS','2'))))
    backend={'cuda':'nccl','xpu':'xccl','cpu':'gloo'}[device.type]
    if distributed: dist.init_process_group(backend,timeout=timedelta(minutes=10))
    def sync(tensor,op=None):
        if distributed: dist.all_reduce(tensor,op=op or dist.ReduceOp.SUM)
    def barrier():
        if distributed: dist.barrier()
    if rank==0: print('PPO device:',device,'workers:',world,'backend:',backend if distributed else 'single',flush=True)
    work=Path(os.environ['NVDA_WORK']); cfg=read(work/'config.json'); data=read(work/'prepared.json')
    if data['config_hash']!=digest(cfg) or data['policy_hash']!=digest(POLICY): raise RuntimeError('Configuration mismatch')
    partition=data['partitions']; trainpart=partition['train']; dimension=len(data['names'])+4
    checkpoint_dir=work/'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
    candidates=[]
    try:
        for candidate in range(cfg['candidates']):
            seed=cfg['seed']+candidate*10000
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model=ActorCritic(dimension).to(device)
            ddp=DDP(model,device_ids=([rank] if device.type!='cpu' else None),broadcast_buffers=False) if distributed else SingleDevice(model)
            lr=cfg['learning_rates'][candidate]
            optimizer=torch.optim.Adam(ddp.parameters(),lr=lr,eps=1e-5)
            ckpt=checkpoint_dir/f'candidate_{candidate}.pt'
            history=[]; best=-float('inf'); best_state=None; best_metrics=None; stale=0; start=0; done=False
            if ckpt.exists():
                state=torch.load(ckpt,map_location='cpu',weights_only=True)
                if state['fingerprint']!=data['fingerprint']: raise RuntimeError('Checkpoint/data mismatch')
                ddp.module.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                history=state['history']; best=state['best']; best_state=state['best_state']; best_metrics=state['best_metrics']
                start=state['update']; stale=state['stale']; done=state['done']
            if rank==0: print('PPO candidate',candidate,'learning rate',lr,'resume update',start,flush=True)
            for update in range(start,cfg['updates']):
                if done: break
                # Every update starts fresh seeded episodes. A saved update is exactly replayable
                # without serializing Python, NumPy, CUDA RNG or environment internals.
                update_seed=seed+update*17+rank*1000000
                rng=np.random.default_rng(update_seed); torch.manual_seed(update_seed)
                env=MarketEnv(trainpart['x'],trainpart['returns'])
                episode=min(128,len(trainpart['x'])); max_start=len(trainpart['x'])-episode
                obs=env.reset(int(rng.integers(max_start+1)),episode)
                observations=[]; actions=[]; logps=[]; values=[]; rewards=[]; dones=[]
                ddp.module.eval()
                for step in range(cfg['rollout_steps']):
                    with torch.no_grad():
                        logits,value=ddp.module(torch.as_tensor(obs,device=device).unsqueeze(0))
                        distribution=Categorical(logits=logits); action=distribution.sample()
                        logp=distribution.log_prob(action)
                    observations.append(obs.copy()); actions.append(int(action.item()))
                    logps.append(float(logp.item())); values.append(float(value.item()))
                    obs,reward,terminal,_=env.step(actions[-1]); rewards.append(reward); dones.append(terminal)
                    if terminal: obs=env.reset(int(rng.integers(max_start+1)),episode)
                with torch.no_grad():
                    _,nv=ddp.module(torch.as_tensor(obs,device=device).unsqueeze(0))
                advantages,targets=gae(rewards,values,dones,float(nv.item()))
                bobs=torch.as_tensor(np.stack(observations),device=device)
                bact=torch.as_tensor(actions,device=device); blogp=torch.as_tensor(logps,device=device)
                badv=torch.as_tensor(advantages,device=device); btarget=torch.as_tensor(targets,device=device)
                # Normalize over every participating worker.
                sums=torch.stack((badv.sum(),(badv*badv).sum(),torch.tensor(float(len(badv)),device=device)))
                sync(sums)
                mean=sums[0]/sums[2]; variance=(sums[1]/sums[2]-mean*mean).clamp(min=1e-8)
                badv=(badv-mean)/variance.sqrt()
                ddp.train(); stop_kl=False; loss_value=0.; kl_value=0.
                for epoch in range(cfg['ppo_epochs']):
                    order=rng.permutation(len(bact))
                    for offset in range(0,len(order),cfg['minibatch']):
                        idx=torch.as_tensor(order[offset:offset+cfg['minibatch']],device=device)
                        logits,value=ddp(bobs[idx]); distribution=Categorical(logits=logits)
                        logp=distribution.log_prob(bact[idx]); logratio=logp-blogp[idx]; ratio=logratio.exp()
                        surrogate=torch.minimum(ratio*badv[idx],ratio.clamp(.8,1.2)*badv[idx])
                        loss=-surrogate.mean()+.5*(value-btarget[idx]).square().mean()-.01*distribution.entropy().mean()
                        valid=torch.tensor(int(torch.isfinite(loss).item()),device=device)
                        sync(valid,op=dist.ReduceOp.MIN)
                        if not valid.item(): raise RuntimeError('Non-finite PPO loss; last atomic checkpoint retained')
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(ddp.parameters(),.5,error_if_nonfinite=True)
                        optimizer.step()
                        kl=((ratio-1)-logratio).mean().detach(); sync(kl); kl/=world
                        loss_value=float(loss.item()); kl_value=float(kl.item())
                        if kl_value>.03: stop_kl=True; break
                    if stop_kl: break
                # All ranks must wait while rank zero validates the same policy.
                barrier()
                if rank==0:
                    val=evaluate(ddp.module,partition['validation'],device)
                    value=score(val); improved=value>best+1e-8
                    if improved:
                        best=value; best_state={k:v.detach().cpu().clone() for k,v in ddp.module.state_dict().items()}
                        best_metrics=val; stale=0
                    else: stale+=1
                    history.append(dict(update=update+1,validation=val,score=value,loss=loss_value,kl=kl_value))
                    done=update+1>=cfg['updates'] or stale>=cfg['patience']
                    save_torch(ckpt,dict(fingerprint=data['fingerprint'],model=ddp.module.state_dict(),
                        optimizer=optimizer.state_dict(),update=update+1,best=best,best_state=best_state,
                        best_metrics=best_metrics,history=history,stale=stale,done=done))
                    dump(work/f'candidate_{candidate}_history.json',history)
                    print(f'Candidate {candidate} update {update+1}: validation net={val["total_return"]:.4%}, DD={val["max_drawdown"]:.3%}',flush=True)
                flag=torch.tensor(int(done),device=device)
                if distributed: dist.broadcast(flag,src=0)
                done=bool(flag.item())
                barrier()
            if rank==0:
                state=torch.load(ckpt,map_location='cpu',weights_only=True)
                bundle=work/f'candidate_{candidate}.pt'
                save_torch(bundle,dict(schema=VERSION,weights=state['best_state'],dimension=dimension,
                    names=data['names'],scaler=data['scaler'],policy=POLICY,fingerprint=data['fingerprint'],
                    config=cfg,model_revision=cfg['model_revision'],news_prompt='nvda-news-json-v2',
                    validation=state['best_metrics'],candidate=candidate))
                candidates.append(dict(candidate=candidate,bundle=bundle.name,validation=state['best_metrics'],
                                       score=score(state['best_metrics']),sha256=filehash(bundle)))
            barrier(); del ddp,model,optimizer; clear_cache(device)
        if rank==0:
            winner=max(candidates,key=lambda c:c['score'])
            dump(work/'selection.json',dict(fingerprint=data['fingerprint'],candidates=candidates,winner=winner,
                 selection_rule='validation total_return minus 2*max_drawdown; never select on test'))
            print('Validation selected candidate',winner['candidate'],flush=True)
    finally:
        if distributed: dist.destroy_process_group()


if __name__=='__main__': train()
