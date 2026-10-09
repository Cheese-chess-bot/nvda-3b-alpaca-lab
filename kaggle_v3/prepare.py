"""Point-in-time market/news ingestion and dual-analyst feature preparation."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from core import *

WORK=Path(os.environ['NVDA_WORK']); PROJECT=Path(os.environ['NVDA_PROJECT'])
CONFIG=read(WORK/'config.json')
DATA=Path(CONFIG.get('dataset_dir') or str(PROJECT/'data'))
from ensemble import features as analyst_features
from sentiment import cache_tag, cached_signal


def collect():
    dataset=read(DATA/'dataset.json'); bars=read(DATA/'bars.json')['bars']
    partitions=['train','validation','test']; allrows=[]
    for part in partitions:
        rows=dataset[part]
        if not rows or any(a['date']>=b['date'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Rows must be chronological and unique')
        allrows+=rows
    for left,right in zip(partitions,partitions[1:]):
        if max(r['label_end'] for r in dataset[left])>=dataset[right][0]['date']:
            raise ValueError('Unpurged split')
    prices=bars['NVDA']; positions={b['t'][:10]:i for i,b in enumerate(prices)}
    for symbol in SYMBOLS:
        if any(a['t']>=b['t'] for a,b in zip(bars[symbol],bars[symbol][1:])):
            raise ValueError('Unsorted or duplicate bars')
    for row in allrows:
        i=positions[row['date']]
        if i+2>=len(prices): raise ValueError('Unresolved return target')
        expected=prices[i+2]['o']/prices[i+1]['o']-1
        if abs(expected-row['return'])>1e-10 or row['label_end']!=prices[i+2]['t'][:10]:
            raise ValueError('Misaligned open-to-open target')
        # Feature cutoff must precede the first execution open (09:30 New York).
        from zoneinfo import ZoneInfo
        next_open=datetime.fromisoformat(prices[i+1]['t'][:10]+'T09:30:00').replace(tzinfo=ZoneInfo('America/New_York'))
        close=datetime.fromisoformat(row['date']+'T16:00:00').replace(tzinfo=ZoneInfo('America/New_York'))
        if not close<=utc(row['as_of'])<next_open:
            raise ValueError('Feature cutoff is not before execution')
    newsfile=WORK/'raw_news.json'
    if newsfile.exists():
        articles=read(newsfile)
    else:
        rows=[]; source=CONFIG['news_jsonl']
        if source:
            with open(source) as f:
                rows=[json.loads(line) for line in f if line.strip()]
        elif CONFIG['use_alpaca_news']:
            # Imports the repo's fixed market-data-only GET client, with bounded retries.
            from nvda_lab import Alpaca
            api=Alpaca(); seen=set(); token=None
            start=(utc(allrows[0]['as_of'])-timedelta(days=3)).isoformat()
            end=allrows[-1]['as_of']
            for page in range(CONFIG['news_page_limit']):
                params=dict(symbols='NVDA',start=start,end=end,limit=50,sort='asc',include_content='false')
                if token: params['page_token']=token
                result=api.call('/v1beta1/news',params=params,market=True)
                rows.extend(result.get('news',[])); token=result.get('next_page_token')
                if not token: break
                if token in seen: raise RuntimeError('Repeated news page token')
                seen.add(token)
                if page%25==0: print('News pages:',page+1,'articles:',len(rows),flush=True)
            else: raise RuntimeError('News page budget reached; no partial corpus accepted. Raise NEWS_PAGE_LIMIT or attach JSONL.')
        elif CONFIG['require_news']:
            raise RuntimeError('News required. Set NEWS_JSONL or USE_ALPACA_NEWS=True with both Kaggle Secrets. Set REQUIRE_NEWS=False only for an explicitly labelled math-only ablation.')
        articles=list({a['key']:a for a in map(normalize_article,rows)}.values())
        dump(newsfile,articles)
    chosen={}; needed={}
    for row in allrows:
        selected=select_articles(articles,row['as_of'])
        chosen[row['date']]=[a['key'] for a in selected]
        needed.update({a['key']:a for a in selected})
    if len(needed)>CONFIG['max_news_articles']:
        raise RuntimeError(f'{len(needed)} selected articles exceed MAX_NEWS_ARTICLES; raise the explicit inference budget.')
    if CONFIG['require_news'] and not needed: raise RuntimeError('No timestamp-eligible news')
    raw_coverage={p:sum(bool(chosen[r['date']]) for r in dataset[p])/len(dataset[p]) for p in partitions}
    if CONFIG['require_news'] and min(raw_coverage.values())<CONFIG['min_news_coverage']:
        raise RuntimeError('Insufficient news coverage by split: '+str(raw_coverage))
    plan=dict(articles=sorted(needed.values(),key=lambda a:a['key']),chosen=chosen,
              raw_coverage=raw_coverage,news_hash=digest(articles),
              bars_hash=filehash(DATA/'bars.json'),dataset_hash=filehash(DATA/'dataset.json'))
    old=WORK/'news_plan.json'
    if old.exists() and digest(read(old))!=digest(plan): raise RuntimeError('Inputs changed inside a resumable run')
    dump(old,plan)
    print('Selected news:',len(needed),'coverage:',raw_coverage,flush=True)


def assemble():
    plan=read(WORK/'news_plan.json'); dataset=read(DATA/'dataset.json')
    if plan['bars_hash']!=filehash(DATA/'bars.json') or plan['dataset_hash']!=filehash(DATA/'dataset.json'):
        raise RuntimeError('Market source changed')
    bars=read(DATA/'bars.json')['bars']; articles={a['key']:a for a in plan['articles']}
    active=CONFIG['news_models']; signals={name:{} for name in active}
    for name in active:
        for key in articles:
            signals[name][key]=cached_signal(WORK/'news_cache'/name/(key+'.json'),key,cache_tag(CONFIG,name))
    packs={}; names=None; coverage={}; model_coverage={name:{} for name in active}
    for part in ('train','validation','test'):
        rows=dataset[part]; matrix=[]; cov=[]; model_cov={name:[] for name in active}
        for row in rows:
            selected=[articles[k] for k in plan['chosen'][row['date']]]
            nf=analyst_features(selected,signals,row['as_of'],active); cov.append(nf['news_missing']==0)
            for name in active: model_cov[name].append(nf[name+'_news_missing']==0)
            f=math_features(row,bars); f.update(nf)
            if names is None: names=sorted(f)
            if sorted(f)!=names: raise ValueError('Feature schema changed')
            matrix.append([f[n] for n in names])
        packs[part]=dict(raw=matrix,returns=[r['return'] for r in rows],dates=[r['date'] for r in rows],
                         label_ends=[r['label_end'] for r in rows],as_of=[r['as_of'] for r in rows])
        coverage[part]=float(np.mean(cov))
        for name in active: model_coverage[name][part]=float(np.mean(model_cov[name]))
    if CONFIG['require_news'] and min(coverage.values())<CONFIG['min_news_coverage']:
        raise RuntimeError('Valid complete-ensemble coverage below required fraction: '+str(coverage))
    scaler=Scaler.fit(packs['train']['raw'])
    for p in packs.values(): p['x']=scaler.transform(p['raw']).tolist()
    payload=dict(schema=VERSION,names=names,scaler=scaler.export(),partitions=packs,
                 coverage=coverage,source_hash=digest(plan),config_hash=digest(CONFIG),
                 policy_hash=digest(POLICY),model_coverage=model_coverage,math_only=not any(s is not None for group in signals.values() for s in group.values()))
    payload['fingerprint']=digest(payload)
    previous=WORK/'prepared.json'
    if previous.exists() and read(previous)['fingerprint']!=payload['fingerprint']:
        raise RuntimeError('Prepared data changed; checkpoints cannot be reused')
    dump(previous,payload)
    print('Prepared',len(names),'features; complete-ensemble news coverage:',coverage,flush=True)


if __name__=='__main__':
    command=argparse.ArgumentParser(); command.add_argument('mode',choices=['collect','assemble'])
    globals()[command.parse_args().mode]()
