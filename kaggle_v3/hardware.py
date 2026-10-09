"""Device selection shared by the launcher, news inference and PPO workers.
A usable PyTorch backend/driver is required for GPU use; CPU always remains an option.
"""
import argparse, json, os, platform


def choose_plan(inventory, requested='auto', max_devices=2):
    aliases={'nvidia':'cuda','amd':'rocm','intel':'xpu'}
    requested=aliases.get(requested,requested)
    if requested not in ('auto','cpu','cuda','rocm','xpu') or max_devices<1:
        raise ValueError('Use auto/cpu/cuda/rocm/xpu and a positive GPU limit')
    detected=inventory['devices']
    kind=next((k for k in ('cuda','rocm','xpu') if detected.get(k)), 'cpu') if requested=='auto' else requested
    if kind!='cpu' and not detected.get(kind):
        raise RuntimeError(f'{kind} is unavailable in this PyTorch build/driver. Choose auto or install the matching official PyTorch build.')
    names=detected.get(kind) or [inventory.get('cpu','CPU')]
    device_type='cuda' if kind=='rocm' else kind
    backend={'cuda':'nccl','rocm':'nccl','xpu':'xccl','cpu':'gloo'}[kind]
    available=inventory.get('distributed',{}).get(backend,False)
    workers=min(len(names),max_devices) if kind!='cpu' and available else 1
    return dict(kind=kind,device_type=device_type,workers=workers,
                distributed_backend=backend if workers>1 else None,
                names=names[:workers],torch_version=inventory['torch_version'],
                note=('No usable accelerator detected; training uses CPU.' if kind=='cpu' else
                      'Collective backend unavailable; using one GPU.' if len(names)>1 and not available else ''))


def inventory(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    t=torch_module; devices={}; issues=[]
    for api_name in ('cuda','xpu'):
        api=getattr(t,api_name,None)
        try:
            if api is not None and api.is_available():
                key='rocm' if api_name=='cuda' and getattr(t.version,'hip',None) else api_name
                devices[key]=[api.get_device_name(i) for i in range(api.device_count())]
        except Exception as exc:
            issues.append(api_name+': '+type(exc).__name__)
    distributed={}
    d=getattr(t,'distributed',None)
    for backend in ('nccl','gloo','xccl'):
        check=getattr(d,'is_'+backend+'_available',None)
        if check is None and d is not None:
            check=getattr(getattr(d,'distributed_c10d',None),'is_'+backend+'_available',None)
        distributed[backend]=bool(check and check())
    return dict(devices=devices,distributed=distributed,torch_version=t.__version__,
                cpu=platform.processor() or platform.machine(),issues=issues)


def resolve_device(requested=None,rank=0):
    import torch
    requested=requested or os.environ.get('NVDA_DEVICE','auto')
    plan=choose_plan(inventory(torch),requested,max_devices=max(1,rank+1))
    kind=plan['device_type']
    if kind=='cpu': return torch.device('cpu')
    api=getattr(torch,kind)
    if rank>=api.device_count(): raise RuntimeError('Worker rank exceeds available GPU count')
    api.set_device(rank)
    return torch.device(kind,rank)


def clear_cache(device):
    import torch
    if device.type in ('cuda','xpu'):
        getattr(torch,device.type).empty_cache()


def smoke_device(device):
    """Verify actual allocations, kernels and backward before a long run."""
    import torch
    layer=torch.nn.Linear(8,5).to(device)
    x=torch.ones((4,8),device=device); logits=layer(x).tanh()
    distribution=torch.distributions.Categorical(logits=logits)
    action=distribution.sample()
    loss=-distribution.log_prob(action).mean(); loss.backward()
    if not bool(torch.isfinite(loss).item()) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in layer.parameters()):
        raise RuntimeError('Device failed numerical smoke check')
    optimizer=torch.optim.Adam(layer.parameters(),lr=3e-4); optimizer.step()
    del layer,x,logits,loss,optimizer,distribution,action; clear_cache(device)


def probe(requested='auto',max_devices=2):
    info=inventory(); plan=choose_plan(info,requested,max_devices)
    try:
        for rank in range(plan['workers']): smoke_device(resolve_device(plan['kind'],rank))
    except Exception as exc:
        if requested!='auto' or plan['kind']=='cpu': raise
        plan=choose_plan(info,'cpu',1)
        plan['note']='GPU kernel probe failed ('+type(exc).__name__+'); automatic CPU fallback selected.'
        smoke_device(resolve_device('cpu'))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='auto'); parser.add_argument('--max-devices',type=int,default=2)
    args=parser.parse_args(); print(json.dumps(probe(args.device,args.max_devices)))
