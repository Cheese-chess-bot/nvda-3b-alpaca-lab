"""Independent bounded research memos and parameter proposals. Never generates code."""
import json
import numpy as np
from core import digest, dump

SPACE = [dict(learning_rate=3e-4, entropy_coef=.01, gamma=.99),
         dict(learning_rate=1e-4, entropy_coef=.005, gamma=.95),
         dict(learning_rate=5e-4, entropy_coef=.02, gamma=.99)]
PREDICTION = dict(expected_net_return_min=0., expected_daily_win_rate_min=.50,
                  failure_drawdown=.05, minimum_sharpe=.75, minimum_dsr=.95,
                  minimum_exposed_days=30, minimum_oos_days=90,
                  stress_cost_multiplier=2, delayed_fill_sessions=1)


def memos(data, ledger, directory):
    """Each analyst gets only its columns + past memory, not another analyst's memo.
    Sector ETFs are macro proxies; no filings/fundamentals data is invented.
    """
    train = data['partitions']['train']; names = data['names']
    raw = np.asarray(train['raw']); memory = ledger.search()
    def mean(name): return float(raw[:, names.index(name)].mean())
    regime = ('high_volatility' if mean('NVDA_vol20')>.04 else
              'trending_up' if mean('NVDA_r20')>.02 else
              'trending_down' if mean('NVDA_r20')<-.02 else 'choppy')
    views = {
        'price': dict(angle='price and volume', r20=mean('NVDA_r20'), volatility=mean('NVDA_vol20')),
        'macro_proxy': dict(angle='sector and index proxies', qqq_r20=mean('QQQ_r20'), soxx_r20=mean('SOXX_r20')),
        'news': dict(angle='independent frozen model signals', coverage=data['coverage']['train'],
                     disagreement=mean('analyst_disagreement')),
        'skeptic': dict(angle='failure conditions', blockers=['cost sensitivity', 'short samples',
                  'pretrained historical knowledge', 'adaptive selection bias', 'daily fill idealization']),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for role, facts in views.items():
        dump(directory/(role+'.json'), dict(role=role, facts=facts, regime=regime,
             memory=memory, prediction=PREDICTION, maximum_ideas=1,
             hypothesis='Causal daily features may support a cost-aware long-only PPO policy.',
             falsified_by='Failure of any fixed development, stress or fresh holdout gate.'))
    dump(directory/'sources.json', dict(fundamentals='unavailable; no filings reader is claimed',
                                      analyst_type='deterministic feature specialists; Qwen/Gemma supply news signals'))
    return regime


def propose(used, failures):
    """Remember the failure type and choose one unused, predeclared alternative."""
    order = [0, 1, 2]
    if failures and any(not f.get('stress_positive', True) or not f.get('drawdown', True) for f in failures):
        order = [1, 0, 2]
    for index in order:
        candidate = dict(SPACE[index], seed=42+index*10000)
        if digest(candidate) not in used: return candidate
    return None
