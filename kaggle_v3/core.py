"""Pure NumPy research core. Broker access and model-generated code are forbidden here."""
import hashlib, json, math, os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np

VERSION = 'nvda-news-ppo-v3.0'
SYMBOLS = ('NVDA', 'QQQ', 'SPY', 'SOXX')
EVENTS = ('earnings', 'product', 'regulation', 'macro', 'other')
POLICY = dict(max_weight=.10, max_notional=1000., initial_equity=10000.,
              day_loss=.02, max_drawdown=.08, cost_bps=10., stress_bps=20.)
ACTIONS = np.array([0., .025, .05, .075, .10], dtype=np.float64)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def utc(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timezone required')
    return d.astimezone(timezone.utc)


def normalize_article(a):
    """User JSONL: id, available_at (with timezone), headline, summary, source.
    Alpaca revisions use max(created_at, updated_at), never creation time alone.
    """
    times = [utc(a[k]) for k in ('available_at', 'created_at', 'updated_at') if a.get(k)]
    if not times or not a.get('id') or not a.get('source'):
        raise ValueError('News requires id, source and a timezone-qualified availability timestamp')
    headline = str(a.get('headline', '')).strip()
    summary = str(a.get('summary', '')).strip()
    if not headline and not summary:
        raise ValueError('Empty news')
    result = dict(id=str(a['id']), available_at=max(times).isoformat(),
                  source=str(a['source']), headline=headline[:1500], summary=summary[:5000])
    result['key'] = digest(result)
    return result


def select_articles(articles, as_of, limit=4, hours=72):
    t = utc(as_of); earliest = t - timedelta(hours=hours)
    latest = {}
    for a in articles:
        when = utc(a['available_at'])
        if earliest <= when <= t:
            identity = (a['source'], a['id'])
            if identity not in latest or when > utc(latest[identity]['available_at']):
                latest[identity] = a
    return sorted(latest.values(), key=lambda a: (a['available_at'], a['key']), reverse=True)[:limit]


def parse_signal(text):
    """Strict bounded JSON. Invalid text is missing, never fabricated neutral news."""
    try:
        s = text.strip()
        if s.startswith('```'):
            s = '\n'.join(s.splitlines()[1:-1]).strip()
        d = json.loads(s)
        if set(d) != {'sentiment', 'relevance', 'uncertainty', 'event'}:
            return None
        for k, lo, hi in [('sentiment', -1, 1), ('relevance', 0, 1), ('uncertainty', 0, 1)]:
            if isinstance(d[k], bool) or not isinstance(d[k], (int, float)):
                return None
            if not math.isfinite(d[k]) or not lo <= d[k] <= hi:
                return None
        if d['event'] not in EVENTS:
            return None
        return d
    except (ValueError, TypeError):
        return None


def news_features(selected, signals, as_of):
    good = [(a, signals.get(a['key'])) for a in selected]
    good = [(a,s) for a,s in good if s is not None]
    f = dict(news_count=len(selected), news_valid=len(good), news_missing=float(not good),
             news_sentiment=0., news_relevance=0., news_uncertainty=1., news_age_days=3.)
    f.update({'event_'+x: 0. for x in EVENTS})
    if good:
        weights = np.array([max(.01, s['relevance']) for a,s in good])
        f['news_sentiment'] = float(np.average([s['sentiment'] for a,s in good], weights=weights))
        f['news_relevance'] = float(np.mean([s['relevance'] for a,s in good]))
        f['news_uncertainty'] = float(np.mean([s['uncertainty'] for a,s in good]))
        f['news_age_days'] = min((utc(as_of)-utc(a['available_at'])).total_seconds()/86400 for a,s in good)
        for name in EVENTS:
            f['event_'+name] = sum(s['event']==name for a,s in good)/len(good)
    return f


def ema(x, span):
    v = float(x[0]); alpha = 2/(span+1)
    for a in x[1:]: v = alpha*float(a)+(1-alpha)*v
    return v


def math_features(row, bars):
    """Only bars dated <= decision day; price-scale-invariant indicators."""
    f = {}; returns = {}
    for sym in SYMBOLS:
        hist = [b for b in bars[sym] if b['t'][:10] <= row['date']][-61:]
        if len(hist) < 61 or hist[-1]['t'][:10] != row['date']:
            raise ValueError('Missing 61-bar history: '+sym+' '+row['date'])
        c,h,l,v = [np.array([b[k] for b in hist],dtype=float) for k in ('c','h','l','v')]
        if np.any(c<=0) or not all(np.isfinite(x).all() for x in (c,h,l,v)):
            raise ValueError('Invalid bars')
        r = c[1:]/c[:-1]-1; returns[sym] = r[-20:]
        gains=np.maximum(np.diff(c)[-14:],0).mean(); losses=np.maximum(-np.diff(c)[-14:],0).mean()
        rsi=50. if gains+losses==0 else 100*gains/(gains+losses)
        tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
        values = {f'r{n}': float(c[-1]/c[-1-n]-1) for n in (1,5,20,60)}
        values.update(vol20=float(r[-20:].std()),z20=float((c[-1]-c[-20:].mean())/max(c[-20:].std(),1e-12)),
                      range=float((h[-1]-l[-1])/c[-1]),body=float(c[-1]/hist[-1].get('o',c[-1])-1),
                      volume_ratio=float(v[-1]/max(v[-20:].mean(),1.)),
                      vwap_gap=float(c[-1]/hist[-1].get('vw',c[-1])-1))
        values.update(rsi14=rsi/100, atr14=float(tr[-14:].mean()/c[-1]),
                      macd=float((ema(c,12)-ema(c,26))/c[-1]),
                      ma20_gap=float(c[-1]/c[-20:].mean()-1),
                      ma60_gap=float(c[-1]/c[-60:].mean()-1),
                      vol5=float(r[-5:].std()), downside20=float(np.minimum(r[-20:],0).std()),
                      drawdown60=float(c[-1]/c.max()-1),
                      volume_trend=float(v[-5:].mean()/max(1.,v[-20:].mean())-1))
        f.update({sym+'_'+k: float(x) for k,x in values.items()})
    for sym in SYMBOLS[1:]:
        a,b=returns['NVDA'],returns[sym]; var=float(np.var(b))
        f['beta20_'+sym]=float(np.mean((a-a.mean())*(b-b.mean()))/max(var,1e-12))
        f['corr20_'+sym]=float(np.mean((a-a.mean())*(b-b.mean()))/max(a.std()*b.std(),1e-12))
        f['relative20_'+sym]=f['NVDA_r20']-f[sym+'_r20']
    if not all(math.isfinite(x) for x in f.values()): raise ValueError('Non-finite features')
    return f


class Scaler:
    def __init__(self, mean, std):
        self.mean=np.asarray(mean); self.std=np.asarray(std)
    @classmethod
    def fit(cls, train):
        x=np.asarray(train,dtype=float)
        return cls(x.mean(0), np.maximum(x.std(0),1e-6))
    def transform(self,x):
        return np.clip((np.asarray(x)-self.mean)/self.std,-10,10).astype(np.float32)
    def drift(self,x):
        return float(np.mean(abs((np.asarray(x)-self.mean)/self.std)>6))
    def export(self):
        return dict(mean=self.mean.tolist(),std=self.std.tolist())


class MarketEnv:
    """Daily open-to-open, long-only, no leverage. Fractional target weights.
    Risk is checked once per day: gaps can overshoot loss limits. Cash halt lasts
    to episode end. Rebalancing and mandatory terminal exit each incur costs.
    Observation: prior-close features + account state marked at the execution open.
    Same-open execution is an idealization; these are not intraday fill predictions.
    """
    def __init__(self,x,returns,cost_bps=POLICY['cost_bps']):
        self.x=np.asarray(x,dtype=np.float32); self.returns=np.asarray(returns,dtype=float)
        if len(x)!=len(returns) or not np.isfinite(self.x).all() or not np.isfinite(self.returns).all() or np.any(self.returns<=-1):
            raise ValueError('Invalid environment data')
        self.cost=cost_bps/10000; self.reset()
    def reset(self,start=0,length=None):
        self.i=int(start); self.end=min(len(self.x),self.i+(length or len(self.x)))
        if not 0<=self.i<self.end: raise ValueError('Empty episode')
        self.equity=POLICY['initial_equity']; self.peak=self.equity; self.weight=0.; self.halted=False
        return self.obs()
    def obs(self):
        idx=min(self.i,len(self.x)-1)
        return np.concatenate([self.x[idx],np.array([self.weight/.1,math.log(self.equity/POLICY['initial_equity']),
                  1-self.equity/self.peak,float(self.halted)],dtype=np.float32)]).astype(np.float32)
    def step(self,action):
        if isinstance(action,bool) or int(action)!=action or not 0<=int(action)<len(ACTIONS):
            raise ValueError('Invalid action')
        oldeq=self.equity; olddd=1-oldeq/self.peak; oldweight=self.weight
        target=0. if self.halted else min(float(ACTIONS[int(action)]), POLICY['max_weight'], POLICY['max_notional']/oldeq)
        # Exact target fraction after transaction costs: f = c*abs(w*(1-f)-w_old).
        if target>=self.weight:
            fee=self.cost*(target-self.weight)/(1+self.cost*target)
        else:
            fee=self.cost*(self.weight-target)/(1-self.cost*target)
        after=oldeq*(1-fee); r=float(self.returns[self.i])
        gross=1+target*r; self.equity=after*gross
        self.weight=target*(1+r)/gross
        turnover=fee/self.cost if self.cost else abs(target-oldweight)
        self.i+=1; done=self.i>=self.end
        if done:
            exit_fee=self.cost*self.weight; self.equity*=1-exit_fee
            turnover+=self.weight*(after*gross)/oldeq; self.weight=0.
        net=self.equity/oldeq-1; self.peak=max(self.peak,self.equity)
        dd=1-self.equity/self.peak
        if net<=-POLICY['day_loss'] or dd>=POLICY['max_drawdown']:
            self.halted=True
        # Net returns already include trading costs. Extra turnover penalty is explicit.
        reward=100*(math.log(self.equity/oldeq)-.10*max(0.,dd-olddd)-.0001*turnover)
        info=dict(net_return=net,equity=self.equity,drawdown=dd,exposure=target,turnover=turnover,halted=self.halted)
        return self.obs(), float(reward), done, info


def gae(rewards,values,dones,next_value,gamma=.99,lam=.95):
    adv=np.zeros(len(rewards),dtype=np.float32); last=0.; nxt=float(next_value)
    for i in reversed(range(len(rewards))):
        mask=1.-float(dones[i]); delta=rewards[i]+gamma*nxt*mask-values[i]
        last=delta+gamma*lam*mask*last; adv[i]=last; nxt=values[i]
    return adv, adv+np.asarray(values,dtype=np.float32)


def metrics(infos):
    r=np.array([i['net_return'] for i in infos]); eq=np.array([POLICY['initial_equity']]+[i['equity'] for i in infos])
    return dict(total_return=float(eq[-1]/eq[0]-1),sharpe=float(np.sqrt(252)*r.mean()/max(r.std(),1e-12)),
                max_drawdown=float(np.max(1-eq/np.maximum.accumulate(eq))),
                exposed_days=int(sum(i['exposure']>0 for i in infos)),
                turnover=float(sum(i['turnover'] for i in infos)), days=len(infos))


def score(m):
    return m['total_return']-2*m['max_drawdown']


def run_backtest(x,returns,choose,cost=POLICY['cost_bps']):
    env=MarketEnv(x,returns,cost); obs=env.reset(); infos=[]
    while True:
        obs,_,done,info=env.step(int(choose(obs,env.i))); infos.append(info)
        if done: break
    return metrics(infos)


def claim_holdout(path,start,end,run_id):
    """Single-controller ledger. Crash after reservation consumes the holdout.
    Keep this file across sessions; deleting it invalidates the research protocol.
    """
    rows=read(path) if Path(path).exists() else []
    if any(not (end<r['start'] or start>r['end']) for r in rows):
        raise RuntimeError('Holdout overlaps an already consumed interval. Collect fresh dates.')
    rows.append(dict(start=start,end=end,run_id=run_id,status='consumed_before_evaluation'))
    dump(path,rows)


def promotion_checks(candidate,stress,baselines,incumbent=None):
    return dict(positive=candidate['total_return']>0, sharpe=candidate['sharpe']>=.75,
                drawdown=candidate['max_drawdown']<=.05, exposure=candidate['exposed_days']>=30,
                stress_positive=stress['total_return']>0,
                beats_baselines=candidate['total_return']>max(m['total_return'] for m in baselines.values()),
                beats_incumbent=incumbent is None or score(candidate)>score(incumbent))


def promote(registry,bundle,report):
    registry=Path(registry)
    if not report['checks'] or not all(report['checks'].values()):
        return False
    new=dict(bundle=str(Path(bundle).resolve()),sha256=filehash(bundle),policy_hash=digest(POLICY),report=report)
    if registry.exists(): new['previous']=read(registry)
    dump(registry,new); return True


def verify_champion(registry):
    entry=read(registry)
    if entry['policy_hash']!=digest(POLICY) or filehash(entry['bundle'])!=entry['sha256']:
        raise RuntimeError('Champion policy/artifact integrity mismatch; halt')
    return entry


def rollback(registry,halt_path,reason):
    """Restore previous immutable artifact. The halt remains latched."""
    dump(halt_path,dict(halted=True,reason=reason))
    current=read(registry); previous=current.get('previous')
    if not previous: return False
    if previous['policy_hash']!=digest(POLICY) or filehash(previous['bundle'])!=previous['sha256']:
        raise RuntimeError('Previous champion is invalid; remain halted')
    dump(registry,previous); return True


def guard_decision(action, equity, day_return, drawdown, data_age_seconds, drift,
                   halt_path, open_orders=False, position_reconciled=True):
    """Broker-independent pre-trade gate; requires REAL current account/quote inputs.
    Returns a proposal only. Broker-side checks/reconciliation are still required.
    """
    reasons=[]
    if Path(halt_path).exists(): reasons.append('persistent_halt')
    inputs=[equity,day_return,drawdown,data_age_seconds,drift]
    if not all(math.isfinite(x) for x in inputs) or equity<=0 or not 0<=drawdown<=1 or not 0<=drift<=1 or day_return<=-1: reasons.append('invalid_state')
    if isinstance(action,bool) or not isinstance(action,(int,np.integer)) or not 0<=action<len(ACTIONS): reasons.append('invalid_action')
    if day_return<=-POLICY['day_loss'] or drawdown>=POLICY['max_drawdown']: reasons.append('loss_limit')
    if data_age_seconds<0 or data_age_seconds>30: reasons.append('stale_quote')
    if drift>.20: reasons.append('feature_drift')
    if open_orders or not position_reconciled: reasons.append('reconciliation_required')
    if reasons:
        dump(halt_path,dict(halted=True,reasons=reasons))
        return dict(allowed=False,reasons=reasons,target_notional=None)
    return dict(allowed=True,reasons=[],target_notional=min(equity*float(ACTIONS[action]),POLICY['max_notional']))
