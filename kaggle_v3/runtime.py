"""Inference proposal only. No credential access and no submit_order function.
Call propose with fresh features, quote age and reconciled paper-account state.
The existing repo's supervised paper runner is NOT wired to this PPO bundle.
"""
from pathlib import Path
import torch
from core import *
from evaluate import load_bundle


def propose(work, raw_features, *, equity, position_weight, peak_equity, day_return,
            quote_age_seconds, features_as_of, decision_as_of,
            open_orders=False, position_reconciled=True, registry_name='champion.json'):
    if registry_name not in ('champion.json','paper_candidate.json'): raise ValueError('Invalid registry')
    work=Path(work); halt=work/'risk_halt.json'; registry=work/registry_name
    try:
        champion=verify_champion(registry)
    except Exception as exc:
        # Bounded recovery: previous known artifact, then remain halted for review.
        if registry.exists():
            try: rollback(registry,halt,'champion_integrity_failure')
            except Exception: dump(halt,dict(halted=True,reason='rollback_failed'))
        else: dump(halt,dict(halted=True,reason='no_approved_champion'))
        return dict(allowed=False,reasons=['champion_unavailable'],target_notional=None)
    try:
        b,model=load_bundle(champion['bundle'])
        values=np.array([raw_features[n] for n in b['names']],dtype=float)
        if set(raw_features)!=set(b['names']) or not np.isfinite(values).all(): raise ValueError('schema')
        age=(utc(decision_as_of)-utc(features_as_of)).total_seconds()
        # Weekend/holiday allowance. Schedule/exchange calendar still belongs to execution adapter.
        if not 0<=age<=96*3600: raise ValueError('stale_features')
        if b['config']['require_news'] and raw_features['news_missing']>0: raise ValueError('missing_news')
        if not (math.isfinite(equity) and math.isfinite(peak_equity) and 0<equity<=peak_equity and 0<=position_weight<=1):
            raise ValueError('account')
        scaler=Scaler(**b['scaler']); dd=1-equity/peak_equity
        obs=np.concatenate([scaler.transform(values),np.array([position_weight/.1,
            math.log(equity/POLICY['initial_equity']),dd,float(halt.exists())],dtype=np.float32)])
        with torch.no_grad(): logits,_=model(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0))
        action=int(logits.argmax(-1).item())
        proposal=guard_decision(action,equity,day_return,dd,quote_age_seconds,scaler.drift(values),halt,
                                open_orders,position_reconciled)
        proposal.update(action_index=action,champion_sha256=champion['sha256'],mode='proposal_only')
        return proposal
    except Exception:
        dump(halt,dict(halted=True,reason='invalid_runtime_input'))
        return dict(allowed=False,reasons=['invalid_runtime_input'],target_notional=None)
