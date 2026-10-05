"""One command for setup -> device probe -> data/news -> PPO -> optional test.
Called by kaggle-box.ipynb; modular stages remain separately editable.
"""
import argparse, hashlib, importlib.util, json, os, re, shutil, signal
import subprocess, sys, time, zipfile
from pathlib import Path

SOURCE=Path(__file__).resolve().parent


def read(path): return json.loads(Path(path).read_text())

def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def filehash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def run(command,env=None,cwd=None,log=None,retries=0):
    for attempt in range(retries+1):
        handle=open(log,'a',encoding='utf-8') if log else None
        process=subprocess.Popen(list(map(str,command)),env=env,cwd=cwd,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=os.name!='nt')
        tail=[]
        try:
            for line in process.stdout:
                print(line,end='',flush=True); tail=(tail+[line])[-80:]
                if handle: handle.write(line); handle.flush()
            code=process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                else:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
            raise
        finally:
            if handle: handle.close()
        if code==0: return
        retry=any(w in ''.join(tail).lower() for w in ('connection reset','timed out','temporary failure','nccl error'))
        if not retry or attempt>=retries:
            raise RuntimeError(f'Process exited {code}. Details are above; completed checkpoints are retained.')
        print('Retrying transient failure from saved state',attempt+1,flush=True); time.sleep(3)


def query(python,code,env=None):
    result=subprocess.run([str(python),'-c',code],env=env,text=True,capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr[-4000:] or result.stdout[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def read_keys(mode,existing_config=None):
    if mode=='off': return {}
    keys={k:os.environ[k] for k in ('APCA_API_KEY_ID','APCA_API_SECRET_KEY') if os.environ.get(k)}
    if len(keys)<2 and importlib.util.find_spec('kaggle_secrets'):
        from kaggle_secrets import UserSecretsClient
        client=UserSecretsClient()
        for name in ('APCA_API_KEY_ID','APCA_API_SECRET_KEY'):
            if name not in keys:
                try: keys[name]=client.get_secret(name)
                except Exception: pass  # Never print secret-service payloads.
    if len(keys)==1: raise RuntimeError('Only one Alpaca credential is enabled. Supply both, or explicitly select --news off.')
    return keys


def effective_news(mode,jsonl,keys):
    if mode=='off' and jsonl: raise ValueError('A news file conflicts with --news off')
    enabled=mode!='off' and bool(jsonl or len(keys)==2)
    if mode=='required' and not enabled:
        raise RuntimeError('News required: set NEWS_JSONL or both Alpaca Kaggle Secrets.')
    return dict(enabled=enabled,news_jsonl=jsonl if enabled else '',use_alpaca_news=enabled and not bool(jsonl))


def setup_python(base,no_install):
    if no_install: return Path(sys.executable)
    environment=(Path('/kaggle/tmp/nvda-box-env') if Path('/kaggle/working').exists() else base/'environment')
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists(): run([sys.executable,'-m','venv','--system-site-packages',environment])
    found=query(python,'import importlib.util,json; print(json.dumps(bool(importlib.util.find_spec("torch"))))')
    if not found:
        print('No PyTorch found: installing the CPU build. GPU use requires an official accelerator build and working drivers.',flush=True)
        run([python,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cpu'])
    version=query(python,'import torch,json; print(json.dumps(torch.__version__))')
    if tuple(map(int,version.split('+')[0].split('.')[:2]))<(2,6):
        raise RuntimeError('PyTorch >=2.6 required. Install the official build matching your CPU/CUDA/ROCm/XPU platform; existing GPU builds are not replaced automatically.')
    constraint=base/'torch-constraint.txt'; constraint.write_text('torch=='+version+'\n')
    run([python,'-m','pip','install','--quiet','-c',constraint,'numpy>=1.26,<3','tzdata'])
    return python


def launch(python,module,args,env,workers,log):
    command=[python]
    if workers>1:
        command+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(workers)]
    run(command+[SOURCE/module]+args,env=env,log=log,retries=1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default=str(SOURCE.parent))
    parser.add_argument('--work-dir',default='/kaggle/working/nvda-box' if Path('/kaggle/working').exists() else str(Path.cwd()/'nvda-box-output'))
    parser.add_argument('--cycle',default='cycle_001')
    parser.add_argument('--device',choices=['auto','cpu','cuda','rocm','xpu'],default='auto')
    parser.add_argument('--max-gpus',type=int,default=2); parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--news',choices=['auto','required','off'],default='auto')
    parser.add_argument('--news-jsonl',default=''); parser.add_argument('--dataset-dir',default='')
    parser.add_argument('--updates',type=int,default=0); parser.add_argument('--candidates',type=int,default=0)
    parser.add_argument('--max-news-articles',type=int,default=6000)
    parser.add_argument('--final-test',action='store_true'); parser.add_argument('--no-install',action='store_true',help='Use the current preconfigured Python; intended for development/CI')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.cycle): raise ValueError('Invalid cycle name')
    if args.max_gpus<1 or args.threads<1 or args.updates<0 or not 0<=args.candidates<=3 or args.max_news_articles<1:
        raise ValueError('Invalid training budget')
    project=Path(args.project).resolve(); base=Path(args.work_dir).resolve(); base.mkdir(parents=True,exist_ok=True)
    work=base/'cycles'/args.cycle; work.mkdir(parents=True,exist_ok=True)
    # One controller per output root; prevents concurrent ledger/checkpoint writes.
    lock=base/'controller.lock'
    try: fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError: raise RuntimeError('A controller lock exists. Stop the other run; after a crash verify it is stopped before removing controller.lock.') from None
    with os.fdopen(fd,'w') as f: json.dump(dict(pid=os.getpid(),cycle=args.cycle),f)
    try:
        python=setup_python(base,args.no_install)
        env=os.environ.copy()
        for key in ('APCA_API_KEY_ID','APCA_API_SECRET_KEY','LOCAL_RANK','RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
            env.pop(key,None)
        env.update(NVDA_WORK=str(work),NVDA_STATE=str(base),NVDA_PROJECT=str(project),NVDA_CPU_THREADS=str(args.threads),
            PYTHONPATH=str(SOURCE)+os.pathsep+str(project)+os.pathsep+os.environ.get('PYTHONPATH',''),
            TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',OMP_NUM_THREADS=str(args.threads),
            HF_HOME=str(Path('/kaggle/tmp/nvda-box-model-cache') if Path('/kaggle/working').exists() else base/'model-cache'))
        plan=query(python,'import hardware,json; print(json.dumps(hardware.probe('+repr(args.device)+','+str(args.max_gpus)+')))',env)
        env['NVDA_DEVICE']=plan['kind']
        if plan['kind'] in ('cuda','rocm'):
            env.update(NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
            if all('T4' in name for name in plan['names']): env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
        print('DEVICE PLAN:',json.dumps(plan,indent=2),flush=True)
        old=read(work/'config.json') if (work/'config.json').exists() else None
        news_path=str(Path(args.news_jsonl).resolve()) if args.news_jsonl else ''
        if news_path and not Path(news_path).is_file(): raise FileNotFoundError('NEWS_JSONL does not exist')
        # A cached run can resume without credentials; this does not re-enable or change its news mode.
        cached_news=old is not None and (work/'raw_news.json').exists()
        keys={} if cached_news or news_path else read_keys(args.news)
        news=effective_news(args.news,news_path,keys) if not cached_news else dict(
            enabled=old['require_news'],news_jsonl=old['news_jsonl'],use_alpaca_news=old['use_alpaca_news'])
        if cached_news and (args.news!=old.get('news_mode','auto') or news_path!=old['news_jsonl']):
            raise RuntimeError('News settings changed within a saved cycle')
        print('NEWS MODE: Qwen sentiment enabled.' if news['enabled'] else
              'NEWS MODE: MATH-ONLY ABLATION. No news source supplied; Qwen is not loaded. PPO trains on price/volume features with explicit missing-news flags.',flush=True)
        if news['enabled'] and not args.no_install:
            dependencies=['transformers==4.51.3','accelerate==1.6.0','huggingface-hub>=0.30,<1','safetensors>=0.5,<1']
            if plan['kind']=='cuda': dependencies.append('bitsandbytes==0.48.1')
            run([python,'-m','pip','install','--quiet','-c',base/'torch-constraint.txt']+dependencies,env=env)
        names=['torch','numpy']+(['transformers','accelerate','huggingface-hub','safetensors'] if news['enabled'] else [])
        if news['enabled'] and plan['kind']=='cuda': names.append('bitsandbytes')
        packages=query(python,'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in '+repr(names)+'}))',env)
        version=packages['torch']
        if tuple(map(int,version.split('+')[0].split('.')[:2]))<(2,6): raise RuntimeError('PyTorch >=2.6 required')
        revision=subprocess.run(['git','rev-parse','HEAD'],cwd=project,text=True,capture_output=True)
        source_revision=revision.stdout.strip() if revision.returncode==0 else 'local-unversioned'
        data=Path(args.dataset_dir).resolve() if args.dataset_dir else project/'data'
        requested=dict(schema='nvda-news-ppo-v2.0',entrypoint='kaggle-box-v1',seed=42,hardware=plan,
            updates=args.updates or (16 if plan['kind']=='cpu' else 40),
            candidates=args.candidates or (1 if plan['kind']=='cpu' else 3),
            learning_rates=[3e-4,1e-4,5e-4],rollout_steps=128 if plan['kind']=='cpu' else 256,
            ppo_epochs=4,minibatch=64,patience=12,news_mode=args.news,
            news_jsonl=news['news_jsonl'],news_input_hash=filehash(news_path) if news_path else None,
            use_alpaca_news=news['use_alpaca_news'],require_news=news['enabled'],min_news_coverage=.5,
            max_news_articles=args.max_news_articles,news_page_limit=2000,auto_promote_research=True,
            dataset_dir=str(data),market_hashes={n:filehash(data/n) for n in ('bars.json','dataset.json')},
            repo_commit=source_revision,source_hash=digest({p.name:filehash(p) for p in sorted(SOURCE.glob('*.py'))}),
            packages=packages,news_execution=plan['kind']+('-nf4-fp16' if plan['kind']=='cuda' else '-fp32' if plan['kind']=='cpu' else '-fp16'))
        if old:
            if any(old.get(k)!=v for k,v in requested.items()):
                raise RuntimeError('Saved cycle config/source/device/data changed. Restore its original settings; use a new cycle for new work and retain the global holdout ledger.')
            config=old
        else:
            model_revision=query(python,'from huggingface_hub import HfApi; import json; print(json.dumps(HfApi().model_info("Qwen/Qwen2.5-3B-Instruct").sha))',env) if news['enabled'] else 'not-used-math-only'
            config=dict(requested,model_revision=model_revision); dump(work/'config.json',config)
        dump(work/'hardware.json',plan)
        run([python,'-m','unittest','discover','-s',SOURCE,'-p','test_core.py','-q'],env=env)
        run([python,'-m','unittest','discover','-s',SOURCE,'-p','test_hardware.py','-q'],env=env)
        ingest_env=dict(env,**keys)
        run([python,SOURCE/'prepare.py','collect'],env=ingest_env,log=work/'ingestion.log')
        keys.clear(); del ingest_env
        news_plan=read(work/'news_plan.json')
        if news_plan['articles']:
            code='from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-3B-Instruct",revision='+repr(config['model_revision'])+',allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","LICENSE*"])'
            run([python,'-c',code],env=env,retries=2)
            launch(python,'prepare.py',['analyze'],env,plan['workers'],work/'news.log')
        run([python,SOURCE/'prepare.py','assemble'],env=env,log=work/'preparation.log')
        launch(python,'ppo.py',[],env,plan['workers'],work/'ppo.log')
        if args.final_test: run([python,SOURCE/'evaluate.py'],env=env,log=work/'evaluation.log')
        else: print('Final holdout not evaluated. Enable RUN_FINAL_TEST only when ready.',flush=True)
        selection=read(work/'selection.json'); prepared=read(work/'prepared.json')
        summary=dict(status='training_complete',hardware=plan,news_coverage=prepared['coverage'],
                     math_only=prepared['math_only'],winner=selection['winner'],final_test=args.final_test,
                     work_dir=str(work),repo_commit=source_revision)
        dump(work/'box_result.json',summary)
        archive=work/'kaggle-box-policy.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(SOURCE.glob('*.py')): z.write(p,'kaggle_v2/'+p.name)
            for pattern in ('candidate_*.pt','candidate_*_history.json','config.json','selection.json','hardware.json','box_result.json','holdout_report.json'):
                for p in work.glob(pattern): z.write(p,p.name)
        print(json.dumps(summary,indent=2)); print('Policy download:',archive,flush=True)
        print('Keep the full nvda-box output directory to resume, including the global ledger and checkpoints. No orders submitted.',flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    if os.name!='nt':
        def interrupted(signum,frame): raise KeyboardInterrupt('Training controller interrupted')
        signal.signal(signal.SIGTERM,interrupted)
    main()
