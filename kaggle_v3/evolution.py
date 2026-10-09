"""Finite, resumable research loop: memory -> hypothesis -> PPO -> breaker -> lesson."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from core import POLICY, digest, dump, filehash, read, score
from ledger import Ledger
from research import PREDICTION, SPACE, memos, propose
from validation import development, diagnostics, folds, report


def immutable(path, value):
    if path.exists():
        if digest(read(path))!=digest(value): raise RuntimeError('Immutable research artifact changed: '+path.name)
    else: dump(path,value)


def train_job(directory, cfg, prepared, env):
    from train_box import launch
    directory.mkdir(parents=True,exist_ok=True)
    prepared=dict(prepared,config_hash=digest(cfg),policy_hash=digest(POLICY))
    prepared['fingerprint']=digest(prepared)
    immutable(directory/'config.json',cfg); immutable(directory/'prepared.json',prepared)
    jobenv=dict(env,NVDA_WORK=str(directory))
    launch(sys.executable,'ppo.py',[],jobenv,cfg['hardware']['workers'],directory/'ppo.log')
    chosen=read(directory/'selection.json')['winner']
    if filehash(directory/chosen['bundle'])!=chosen['sha256']: raise RuntimeError('Training artifact mismatch')
    return chosen,prepared


def run(work, root, env):
    work,root=Path(work),Path(root); ledger=Ledger(root)
    cfg=read(work/'config.json'); data=read(work/'prepared.json')
    name=work.name; output=work/'evolution'; output.mkdir(exist_ok=True)
    try:
        ledger.import_history()
        spec=dict(config_hash=digest(cfg),data_fingerprint=data['fingerprint'],space=SPACE,
                  prediction=PREDICTION,revisions=cfg['evolution_revisions'],folds=3,
                  updates=cfg['updates'],risk_policy=POLICY)
        campaign=ledger.campaign(name,spec)
        deadline=__import__('datetime').datetime.fromisoformat(campaign['created']).timestamp()+cfg['budget_minutes']*60
        env=dict(env,NVDA_DEADLINE=str(min(deadline,float(env.get('NVDA_DEADLINE',deadline)))))
        completed=output/'summary.json'
        if completed.exists():
            summary=read(completed)
            selected=work/summary['winner']['bundle']
            if filehash(selected)!=summary['winner']['sha256']: raise RuntimeError('Completed policy artifact mismatch')
            print('Resuming completed evolution campaign; no new trials.',flush=True)
            return summary
        regime=memos(data,ledger,output/'research')
        idea='Daily causal math and '+','.join(cfg['news_models'])+' signals with a long-only PPO policy'
        hypothesis=ledger.hypothesis(idea,regime)
        old=list(ledger.db.execute('SELECT * FROM attempts WHERE hypothesis=? ORDER BY created,id',(hypothesis,)))
        used={digest(json.loads(r['parameters'])) for r in old if r['campaign']!=name}
        failures=[json.loads(r['result'])['checks'] for r in old if r['result']]
        windows=folds(development(data)); results=[]
        # Count all potential validation checkpoints before testing, plus final refit.
        ledger.reserve_trials('campaign:'+name,(cfg['evolution_revisions']*len(windows)+1)*cfg['updates'])
        for revision in range(cfg['evolution_revisions']):
            identity=name+':revision:'+str(revision)
            saved=ledger.db.execute('SELECT * FROM attempts WHERE id=?',(identity,)).fetchone()
            parameters=json.loads(saved['parameters']) if saved else propose(used,failures)
            if parameters is None: break
            if time.time()>=float(env['NVDA_DEADLINE']): raise TimeoutError('Evolution elapsed-time budget exhausted')
            ledger.reserve(identity,hypothesis,name,parameters,PREDICTION,cap=3)
            used.add(digest(parameters))
            directory=output/f'revision_{revision}'; directory.mkdir(exist_ok=True)
            immutable(directory/'prediction.json',dict(parameters=parameters,prediction=PREDICTION))
            if saved and saved['result']:
                result=json.loads(saved['result'])
            else:
                fold_results=[]; trial_sharpes=[]
                for index,(parts,oos,scaler) in enumerate(windows):
                    config=dict(cfg,candidates=1,learning_rates=[parameters['learning_rate']],
                                seed=parameters['seed'],entropy_coef=parameters['entropy_coef'],gamma=parameters['gamma'])
                    prepared=dict(data,partitions=parts,scaler=scaler); prepared.pop('fingerprint',None)
                    folder=directory/f'fold_{index}'
                    chosen,_=train_job(folder,config,prepared,env)
                    from evaluate import load_bundle
                    _,model=load_bundle(folder/chosen['bundle'])
                    fold_results.append(diagnostics(model,oos,data['names']))
                    trial_sharpes += [x['validation']['sharpe']/(252**.5) for x in read(folder/'candidate_0_history.json')]
                    del model
                result=report(fold_results,ledger.trial_count(),trial_sharpes)
                result.update(parameters=parameters,revision=revision,hypothesis=hypothesis)
                ledger.finish(identity,result)
            immutable(directory/'report.json',result)
            lessons=[key for key,value in result['checks'].items() if not value]
            dump(directory/'post_mortem.json',dict(failures=lessons,next_action='one unused bounded proposal' if lessons else 'retain for fresh holdout',
                 changes_allowed=['learning_rate','entropy_coef','gamma','seed'],risk_changes_allowed=False))
            results.append(result); failures.append(result['checks'])
            print('Evolution revision',revision,'passed:',result['passed'],'failed checks:',lessons,flush=True)
        if not results: raise RuntimeError('No unused hypothesis revisions remain; inspect the ledger')
        eligible=[r for r in results if r['passed']]
        best=max(eligible or results,key=lambda r:score(r['candidate']))
        p=best['parameters']; final=output/'refit'
        finalcfg=dict(cfg,candidates=1,learning_rates=[p['learning_rate']],seed=p['seed'],
                      entropy_coef=p['entropy_coef'],gamma=p['gamma'])
        refit=dict(data,partitions={k:data['partitions'][k] for k in ('train','validation')}); refit.pop('fingerprint',None)
        chosen,_=train_job(final,finalcfg,refit,env)
        # Retain a review artifact even when every revision failed. It is NOT eligible for paper.
        target=work/'evolved_policy.pt'; shutil.copyfile(final/chosen['bundle'],target)
        winner=dict(chosen,bundle=target.name,sha256=filehash(target),research_passed=bool(eligible))
        dump(work/'selection.json',dict(fingerprint=data['fingerprint'],winner=winner,
             candidates=[dict(revision=r['revision'],score=score(r['candidate']),passed=r['passed']) for r in results],
             selection_rule='development walk-forward gates, then development score; no test outcomes'))
        summary=dict(schema='nvda-evolution-1',hypothesis=hypothesis,campaign=name,
            status='awaiting_fresh_holdout' if eligible else 'failed_research',passed=bool(eligible),
            winner=winner,best_report=best,attempts=len(results),counted_trials=ledger.trial_count(),
            fresh_holdout_after=ledger.fresh_after,broker_orders=False,code_rewriting=False)
        dump(completed,summary)
        ledger.db.execute('UPDATE campaigns SET status=? WHERE id=?',(summary['status'],name))
        ledger.event('campaign_complete',name,dict(status=summary['status'],trials=ledger.trial_count()))
        return summary
    except BaseException as exc:
        ledger.event('research_paused',name,dict(error_type=type(exc).__name__,action='resume identical inputs; do not erase memory'))
        raise
    finally: ledger.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',default=os.environ.get('NVDA_STATE',''))
    parser.add_argument('--work',default=os.environ.get('NVDA_WORK',''))
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if not args.state: parser.error('--state required')
    if args.status:
        ledger=Ledger(args.state)
        try: print(json.dumps(dict(hypotheses=ledger.search(),trials=ledger.trial_count(),fresh_after=ledger.fresh_after),indent=2))
        finally: ledger.close()
    else:
        if not args.work: parser.error('--work required')
        run(args.work,args.state,os.environ.copy())


if __name__=='__main__': main()
