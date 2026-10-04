"""One selected candidate, one reserved holdout, independent promotion gate."""
import os, shutil
from pathlib import Path
import numpy as np
import torch
from core import *
from ppo import ActorCritic, evaluate


def load_bundle(path,device='cpu'):
    b=torch.load(path,map_location='cpu',weights_only=True)
    if b['schema']!=VERSION or b['policy']!=POLICY: raise RuntimeError('Incompatible policy bundle')
    model=ActorCritic(b['dimension']).to(device); model.load_state_dict(b['weights']); model.eval()
    return b,model


def main():
    work=Path(os.environ['NVDA_WORK']); state=Path(os.environ.get('NVDA_STATE',str(work))); cfg=read(work/'config.json'); data=read(work/'prepared.json')
    selection=read(work/'selection.json')
    if selection['fingerprint']!=data['fingerprint']: raise RuntimeError('Selection/data mismatch')
    winner=selection['winner']; bundle=work/winner['bundle']; reportfile=work/'holdout_report.json'
    if filehash(bundle)!=winner['sha256']: raise RuntimeError('Selected artifact changed')
    if reportfile.exists():
        report=read(reportfile)
        if report['fingerprint']!=data['fingerprint'] or report['candidate_sha256']!=winner['sha256']:
            raise RuntimeError('An evaluated run cannot be modified')
        print('Existing holdout result:',json.dumps(report,indent=2)); return
    test=data['partitions']['test']; registry=state/'champion.json'; incumbent_entry=None
    if registry.exists(): incumbent_entry=verify_champion(registry)
    # Reserve before loading outcomes into evaluation. A crash cannot unlock reuse.
    claim_holdout(state/'holdout_ledger.json',test['dates'][0],test['label_ends'][-1],data['fingerprint'])
    b,model=load_bundle(bundle)
    candidate=evaluate(model,test,'cpu'); stress=evaluate(model,test,'cpu',POLICY['stress_bps'])
    ix=data['names'].index('NVDA_r20'); iz=data['names'].index('NVDA_z20'); raw=np.asarray(test['raw'])
    baselines={
        'cash':run_backtest(test['x'],test['returns'],lambda o,i:0),
        'constant_10pct':run_backtest(test['x'],test['returns'],lambda o,i:4),
        'trend':run_backtest(test['x'],test['returns'],lambda o,i:4 if raw[i,ix]>0 else 0),
        'mean_reversion':run_backtest(test['x'],test['returns'],lambda o,i:4 if raw[i,iz]<-1 else 0),
    }
    incumbent=None
    if incumbent_entry:
        ib,im=load_bundle(incumbent_entry['bundle'])
        if ib['names']!=data['names']: raise RuntimeError('Incompatible incumbent schema; keep incumbent and consumed holdout')
        oldscaler=Scaler(**ib['scaler'])
        oldtest=dict(test,x=oldscaler.transform(test['raw']).tolist())
        incumbent=evaluate(im,oldtest,'cpu')
    checks=promotion_checks(candidate,stress,baselines,incumbent)
    # Bundled dates were already inspected during earlier research. They cannot
    # justify a newly validated champion. New data must start AFTER that boundary.
    checks['fresh_holdout_dates']=test['dates'][0]>'2026-10-02'
    report=dict(fingerprint=data['fingerprint'],candidate_sha256=winner['sha256'],candidate=candidate,stress=stress,
        baselines=baselines,incumbent=incumbent,checks=checks,passed=all(checks.values()),
        news_coverage=data['coverage'],math_only=data['math_only'],test_start=test['dates'][0],test_end=test['label_ends'][-1],
        limitations=['Bundled historical dates have been inspected; promotion requires test dates after 2026-10-02.',
          'Daily bars and fixed transaction costs; no order book, intraday path, fill simulation, dividends or interest.',
          'Frozen Qwen is not fine-tuned; historical text can be affected by pretrained-model knowledge.',
          'Validation selects a finite candidate set. One short holdout cannot establish profitability.',
          'IEX historical sample; not all market data modalities. No live or paper orders are submitted.'])
    dump(reportfile,report)
    promoted=False
    if cfg['auto_promote_research']:
        promoted=promote(registry,bundle,report)
    print(json.dumps(report,indent=2)); print('Research registry promotion:',promoted,'; broker orders: NONE',flush=True)


if __name__=='__main__': main()
