"""Keep individual analyst signals, strict consensus and explicit disagreement."""
from core import news_features, utc
from models import REGISTRY, selected_models


def combine(signals):
    if not signals or any(s is None for s in signals):
        return None
    return dict(sentiment=sum(s['sentiment'] for s in signals)/len(signals),
        relevance=sum(s['relevance'] for s in signals)/len(signals),
        uncertainty=max(s['uncertainty'] for s in signals),
        event=signals[0]['event'] if len({s['event'] for s in signals}) == 1 else 'other')


def features(selected, by_model, as_of, active):
    active = selected_models(active)
    if any(utc(a['available_at']) > utc(as_of) for a in selected):
        raise ValueError('Future news cannot enter decision features')
    merged = {a['key']: combine([by_model[n].get(a['key']) for n in active]) for a in selected}
    result = news_features(selected, merged, as_of)
    # Stable schema even for explicitly requested single-model or math-only ablations.
    for name in REGISTRY:
        individual = news_features(selected, by_model.get(name, {}), as_of)
        result.update({name + '_' + key: value for key, value in individual.items()})
    pairs = [(by_model.get('qwen', {}).get(a['key']), by_model.get('gemma', {}).get(a['key'])) for a in selected]
    pairs = [(q, g) for q, g in pairs if q is not None and g is not None]
    result['analyst_disagreement'] = sum(abs(q['sentiment']-g['sentiment'])/2 for q,g in pairs)/max(1,len(pairs))
    result['analyst_pair_fraction'] = len(pairs)/max(1,len(selected))
    result['analyst_pair_missing'] = float(not pairs)
    return result
