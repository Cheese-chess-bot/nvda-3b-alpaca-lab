for filename,content in FILES.items():
    target=SOURCE/filename
    if target.exists() and target.read_text()!=content and (WORK/'config.json').exists():
        raise RuntimeError('Code changed inside an existing run; restore the matching notebook version.')
    target.write_text(content,encoding='utf-8')
source_hash=hashlib.sha256(json.dumps(FILES,sort_keys=True).encode()).hexdigest()
data_dir=Path(DATASET_DIR) if DATASET_DIR else PROJECT/'data'
market_hashes={name:hashlib.sha256((data_dir/name).read_bytes()).hexdigest() for name in ('bars.json','dataset.json')}
news_input_hash=hashlib.sha256(Path(NEWS_JSONL).read_bytes()).hexdigest() if NEWS_JSONL else None
packages=json.loads(subprocess.check_output([PYTHON,'-c',
    'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ["torch","numpy","transformers","accelerate","bitsandbytes","huggingface-hub","safetensors"]}))'],env=env,text=True))
requested=dict(packages=packages,schema='nvda-news-ppo-v2.0',dataset_dir=DATASET_DIR,market_hashes=market_hashes,seed=SEED,candidates=CANDIDATES,updates=PPO_UPDATES,
    learning_rates=[3e-4,1e-4,5e-4],rollout_steps=256,ppo_epochs=4,minibatch=64,patience=12,
    news_jsonl=NEWS_JSONL,news_input_hash=news_input_hash,use_alpaca_news=USE_ALPACA_NEWS,
    require_news=REQUIRE_NEWS,min_news_coverage=MIN_NEWS_COVERAGE,max_news_articles=MAX_NEWS_ARTICLES,
    news_page_limit=NEWS_PAGE_LIMIT,auto_promote_research=AUTO_PROMOTE_RESEARCH,
    repo_commit=COMMIT,source_hash=source_hash,torch_version=runtime['torch'])
config_file=WORK/'config.json'
if config_file.exists():
    config=json.loads(config_file.read_text())
    if any(config.get(k)!=v for k,v in requested.items()):
        raise RuntimeError('Configuration/input changed. Resume with original settings; do not reset the holdout ledger.')
else:
    revision=subprocess.check_output([PYTHON,'-c',
        'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen2.5-3B-Instruct").sha)'],env=env,text=True).strip().splitlines()[-1]
    config=dict(requested,model_revision=revision)
    atomic_json(config_file,config)
print('Pinned Qwen revision:',config['model_revision'])
run([PYTHON,'-c','import torch,transformers,bitsandbytes; print("Dependencies imported")'],env=env)
run([PYTHON,'-m','unittest','discover','-s',SOURCE,'-p','test_core.py','-q'],env=env)
run([PYTHON,'-m','unittest','discover','-s','tests','-q'],cwd=PROJECT,env=env)
run([PYTHON,SOURCE/'prepare.py','collect'],env=ingest_env,log=WORK/'ingestion.log')
# Explicitly discard the only copy of credentials created by this notebook.
for name in ('APCA_API_KEY_ID','APCA_API_SECRET_KEY'): ingest_env.pop(name,None)
del ingest_env
plan=json.loads((WORK/'news_plan.json').read_text())
if plan['articles']:
    run([PYTHON,'-c',
        'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-3B-Instruct", revision='+repr(config['model_revision'])+', allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","LICENSE*"])'],env=env,retries=2)
    run([PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
         SOURCE/'prepare.py','analyze'],env=env,log=WORK/'news.log',retries=1)
else:
    print('EXPLICIT MATH-ONLY ABLATION: no news supplied; Qwen inference skipped.')
run([PYTHON,SOURCE/'prepare.py','assemble'],env=env,log=WORK/'preparation.log')
# News subprocesses exit first and release their Qwen weights before PPO starts.
run([PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
     SOURCE/'ppo.py'],env=env,log=WORK/'ppo.log',retries=1)
if RUN_FINAL_TEST:
    run([PYTHON,SOURCE/'evaluate.py'],env=env,log=WORK/'evaluation.log')
else:
    print('Final test not run in this notebook. Selection is based on validation only.')

# Save an inference/review bundle. FULL folder outputs are necessary to resume.
archive=WORK/'nvda_news_math_ppo_bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in SOURCE.glob('*.py'): z.write(path,'system/'+path.name)
    for pattern in ('candidate_*.pt','candidate_*_history.json','config.json','selection.json',
                    'holdout_report.json'):
        for path in WORK.glob(pattern): z.write(path,path.name)
    for name in ('holdout_ledger.json','champion.json','risk_halt.json'):
        path=BASE/name
        if path.exists(): z.write(path,name)
    prepared=json.loads((WORK/'prepared.json').read_text())
    metadata={k:prepared[k] for k in ('schema','names','scaler','coverage','fingerprint','policy_hash','math_only')}
    z.writestr('feature_metadata.json',json.dumps(metadata,indent=2))
print('\nDONE:',archive)
print('Frozen Qwen news extraction + numerical features + trained PPO. No broker orders submitted.')
print('Save Version WITH OUTPUTS. Retain the ENTIRE nvda_news_ppo_v2 folder, including checkpoints and holdout ledger.')
print('A new evolution cycle needs genuinely fresh holdout dates; do not delete the ledger to retry this test.')
print('runtime.propose(BASE, ...) emits gated proposals only; the old supervised Alpaca runner does not load this PPO policy.')
from IPython.display import FileLink,display
display(FileLink(str(archive.relative_to('/kaggle/working'))))
