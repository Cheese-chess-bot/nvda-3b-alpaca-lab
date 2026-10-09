"""Purged expanding windows, stress traces and an explicitly approximate DSR."""
import math
from statistics import NormalDist
import numpy as np
from core import MarketEnv, POLICY, Scaler, metrics


def subset(part, indices):
    return {key: [values[i] for i in indices] for key, values in part.items()
            if isinstance(values, list) and len(values)==len(part['dates']) and key!='x'}


def development(data):
    # Never include test returns or features in a research fold.
    a, b = (data['partitions'][name] for name in ('train', 'validation'))
    return {key: a[key]+b[key] for key in ('raw', 'returns', 'dates', 'label_ends', 'as_of')}


def folds(part, count=3):
    n = len(part['dates'])
    if n < 240: raise ValueError('At least 240 development sessions required for walk-forward research')
    edges = np.linspace(n//2, n, count+1, dtype=int)
    result = []
    for start, end in zip(edges, edges[1:]):
        fit = [i for i in range(start) if part['label_ends'][i] < part['dates'][start]]
        val_start = fit[int(len(fit)*.8)]
        train = [i for i in fit if i<val_start and part['label_ends'][i]<part['dates'][val_start]]
        val = [i for i in fit if i>=val_start]
        if min(len(train), len(val), end-start)<20: raise ValueError('Fold too short')
        packs = dict(train=subset(part, train), validation=subset(part, val))
        scaler = Scaler.fit(packs['train']['raw'])
        for pack in packs.values(): pack['x'] = scaler.transform(pack['raw']).tolist()
        test = subset(part, list(range(start, end))); test['x'] = scaler.transform(test['raw']).tolist()
        result.append((packs, test, scaler.export()))
    return result


def trace(part, choose, cost=POLICY['cost_bps'], delay=0):
    env = MarketEnv(part['x'], part['returns'], cost); obs = env.reset(); infos=[]; pending=0
    while True:
        proposed = int(choose(obs, env.i)); action = pending if delay else proposed; pending=proposed
        obs, _, done, info = env.step(action); infos.append(info)
        if done: break
    return infos


def joined_metrics(traces):
    # Flat start/end per fold, costs paid on each close; reset risk state per fold.
    infos=[]; equity=POLICY['initial_equity']; peak=equity
    for group in traces:
        for source in group:
            row=dict(source); equity*=1+row['net_return']; peak=max(peak,equity)
            row.update(equity=equity,drawdown=1-equity/peak); infos.append(row)
    result=metrics(infos)
    result.update(daily_win_rate=float(np.mean([i['net_return']>0 for i in infos])),
                  rebalance_days=sum(i['turnover']>1e-9 for i in infos))
    return result


def deflated_sharpe(returns, trials, trial_sharpes=()):
    """Bailey/Lopez de Prado Eq. 2. Daily SR; Pearson (not excess) kurtosis.
    Raw checkpoint count is used as N; serial/adaptive dependence isn't solved.
    A null sampling-variance floor is used when empirical variance is too small.
    This is a research diagnostic, not a calibrated probability of profitability.
    """
    r=np.asarray(returns,dtype=float); n=len(r)
    if n<3 or not np.isfinite(r).all() or trials<1: raise ValueError('Invalid DSR inputs')
    std=float(r.std(ddof=1))
    if std<1e-12:
        return dict(probability=0., threshold_daily=0., sharpe_daily=0., trials=trials, degenerate=True)
    sr=float(r.mean()/std); centered=r-r.mean(); second=float(np.mean(centered**2))
    skew=float(np.mean(centered**3)/second**1.5); kurt=float(np.mean(centered**4)/second**2)
    sample=[float(s) for s in trial_sharpes if math.isfinite(s)]
    variance=max(1/(n-1),float(np.var(sample,ddof=1)) if len(sample)>1 else 0.)
    normal=NormalDist(); gamma=.5772156649015329
    threshold=0. if trials==1 else math.sqrt(variance)*(
        (1-gamma)*normal.inv_cdf(1-1/trials)+gamma*normal.inv_cdf(1-1/(trials*math.e)))
    denominator=1-skew*sr+(kurt-1)*sr*sr/4
    probability=normal.cdf((sr-threshold)*math.sqrt(n-1)/math.sqrt(denominator)) if denominator>0 else 0.
    return dict(probability=probability,threshold_daily=threshold,sharpe_daily=sr,
                trials=trials,skewness=skew,pearson_kurtosis=kurt,trial_variance=variance,
                approximation='raw trial count; variance floor; adaptive/serial dependence unresolved')


def worst_windows(part, length=20, count=5):
    r=np.asarray(part['returns']); ranked=sorted(range(0,len(r)-length+1),
        key=lambda i:float(np.prod(1+r[i:i+length])-1))
    chosen=[]
    for i in ranked:
        if all(i+length<=old or i>=old+length for old in chosen): chosen.append(i)
        if len(chosen)==count: break
    return [(i,i+length) for i in chosen]


def diagnostics(model, part, names):
    from ppo import choose_model
    choose=choose_model(model,'cpu')
    normal=trace(part,choose); stress=trace(part,choose,POLICY['stress_bps']); delayed=trace(part,choose,delay=1)
    raw=np.asarray(part['raw']); trend=names.index('NVDA_r20'); reversion=names.index('NVDA_z20')
    baselines={name:trace(part,fn) for name,fn in {
        'cash':lambda o,i:0,'constant_10pct':lambda o,i:4,
        'trend':lambda o,i:4 if raw[i,trend]>0 else 0,
        'mean_reversion':lambda o,i:4 if raw[i,reversion]<-1 else 0}.items()}
    worst=[]
    for start,end in worst_windows(part):
        p=subset(part,list(range(start,end))); p['x']=part['x'][start:end]
        worst.append(dict(start=p['dates'][0],end=p['label_ends'][-1],
                          metrics=joined_metrics([trace(p,choose,POLICY['stress_bps'],delay=1)])))
    return dict(normal=normal,stress=stress,delayed=delayed,baselines=baselines,worst=worst)


def report(fold_results, trials, trial_sharpes=()):
    normal=joined_metrics([f['normal'] for f in fold_results])
    stress=joined_metrics([f['stress'] for f in fold_results]); delayed=joined_metrics([f['delayed'] for f in fold_results])
    baselines={name:joined_metrics([f['baselines'][name] for f in fold_results]) for name in fold_results[0]['baselines']}
    dsr=deflated_sharpe([r['net_return'] for f in fold_results for r in f['normal']],trials,trial_sharpes)
    stable=sum(joined_metrics([f['normal']])['total_return']>0 for f in fold_results)
    checks=dict(positive=normal['total_return']>0,drawdown=normal['max_drawdown']<=.05,
        sharpe=normal['sharpe']>=.75,dsr=dsr['probability']>=.95,
        enough_data=normal['days']>=90,exposure=normal['exposed_days']>=30,
        daily_win_rate=normal['daily_win_rate']>=.50,
        stress_positive=stress['total_return']>0,delayed_positive=delayed['total_return']>0,
        stable_folds=stable>=math.ceil(2*len(fold_results)/3),
        beats_baselines=normal['total_return']>max(x['total_return'] for x in baselines.values()),
        worst_period_risk=all(w['metrics']['max_drawdown']<=.05 for f in fold_results for w in f['worst']))
    return dict(candidate=normal,stress=stress,delayed=delayed,baselines=baselines,dsr=dsr,
        folds=[dict(normal=joined_metrics([f['normal']]),worst=f['worst']) for f in fold_results],
        checks=checks,passed=all(checks.values()),
        limitations=['Development windows are adaptive research, never a final test.',
         'DSR is approximate under correlated returns and repeated PPO selection.',
         'Fixed daily-bar costs and a one-session action delay do not model real fills.',
         'Daily win rate and rebalance days are not completed-trade win rate/count.'])
