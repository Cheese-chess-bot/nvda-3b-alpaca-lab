# KAGGLE SETTINGS: GPU T4 x2, Internet ON. Use a fresh notebook session.
# This pipeline trains a PPO policy. Frozen Qwen extracts news; no orders are sent.
CYCLE_ID = 'cycle_001'           # increment ONLY when supplying genuinely fresh test dates
SEED = 42
CANDIDATES = 3                   # bounded evolution: max 3 seeds/learning rates
PPO_UPDATES = 40                 # each update: 256 transitions/GPU, 4 PPO epochs
RUN_FINAL_TEST = True            # ONE winner, ONE reserved test partition
AUTO_PROMOTE_RESEARCH = True     # only if every independent holdout gate passes
DATASET_DIR = ''                 # optional attached directory: bars.json + dataset.json
NEWS_JSONL = ''                  # e.g. /kaggle/input/nvda-news/news.jsonl
USE_ALPACA_NEWS = False          # True: read BOTH Alpaca keys from Kaggle Secrets
REQUIRE_NEWS = True              # False explicitly allows a math-only ablation
MIN_NEWS_COVERAGE = 0.50         # eligible, valid news on >=50% of days in EACH split
MAX_NEWS_ARTICLES = 6000         # explicit frozen-Qwen inference budget
NEWS_PAGE_LIMIT = 2000           # paginated Alpaca GET budget; no partial acceptance
RESTORE_FROM = ''                # saved nvda_news_ppo_v2 folder for full-state resume

import os, sys, json, shutil, subprocess, signal, hashlib, time, zipfile
from pathlib import Path

BASE = Path('/kaggle/working/nvda_news_ppo_v2')
if not CYCLE_ID or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in CYCLE_ID):
    raise ValueError('CYCLE_ID must use letters, digits, underscore or hyphen')
WORK = BASE/'cycles'/CYCLE_ID
PROJECT = BASE/'project'
SOURCE = WORK/'system'
REPO = 'https://github.com/Cheese-chess-bot/nvda-3b-alpaca-lab.git'
COMMIT = '706ae9f5861eb88347f977d74533aaed058b5cd9'


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def run(command, *, cwd=None, env=None, log=None, retries=0):
    """Bounded transient retries; interrupts terminate the entire GPU process group."""
    for attempt in range(retries+1):
        handle=open(log,'a',encoding='utf-8') if log else None
        process=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        tail=[]
        try:
            for line in process.stdout:
                print(line,end='',flush=True); tail=(tail+[line])[-80:]
                if handle: handle.write(line); handle.flush()
            code=process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
            raise
        finally:
            if handle: handle.close()
        if code==0: return
        transient=any(word in ''.join(tail).lower() for word in
                      ['connection reset','connection aborted','timed out','temporary failure','nccl error'])
        if attempt>=retries or not transient:
            raise RuntimeError(f'Process exited {code}; stopped with checkpoints retained. Read the error above.')
        print('Transient failure; retrying from atomic caches/checkpoint:',attempt+1,flush=True)
        time.sleep(3*(attempt+1))


if not Path('/kaggle/working').exists():
    raise RuntimeError('Open this notebook in Kaggle; choose GPU T4 x2 and Internet ON.')
if not 1<=CANDIDATES<=3 or PPO_UPDATES<1: raise ValueError('CANDIDATES must be 1..3 and PPO_UPDATES positive')
if not 0<=MIN_NEWS_COVERAGE<=1: raise ValueError('MIN_NEWS_COVERAGE must be 0..1')
if RESTORE_FROM:
    source=Path(RESTORE_FROM)
    if BASE.exists(): raise RuntimeError('Restore target exists; do not overwrite an active run.')
    if not (source/'cycles').is_dir(): raise RuntimeError('Restore the complete saved nvda_news_ppo_v2 folder.')
    shutil.copytree(source,BASE)
WORK.mkdir(parents=True,exist_ok=True); SOURCE.mkdir(exist_ok=True)
if REQUIRE_NEWS and not NEWS_JSONL and not USE_ALPACA_NEWS and not (WORK/'raw_news.json').exists():
    raise RuntimeError('Add news first: set NEWS_JSONL, or enable USE_ALPACA_NEWS with Kaggle Secrets. REQUIRE_NEWS=False is an explicit math-only ablation.')
if NEWS_JSONL and not Path(NEWS_JSONL).is_file():
    raise RuntimeError('NEWS_JSONL must name an attached readable JSONL file.')

probe=subprocess.run([sys.executable,'-c','''
import torch,json
assert torch.cuda.device_count()==2, 'Select GPU T4 x2 in Kaggle settings'
assert tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2]) >= (2,6), 'Requires PyTorch >=2.6'
assert all('T4' in torch.cuda.get_device_name(i) for i in range(2)), 'Expected two T4 GPUs'
print(json.dumps({'torch':torch.__version__,'gpus':[torch.cuda.get_device_name(i) for i in range(2)]}))
'''],text=True,capture_output=True)
if probe.returncode: raise RuntimeError(probe.stderr or probe.stdout)
runtime=json.loads(probe.stdout.strip().splitlines()[-1]); print('Detected:',runtime)
VENV=Path('/kaggle/tmp/nvda_news_ppo_venv')
if not (VENV/'bin/python').exists(): run([sys.executable,'-m','venv','--system-site-packages',VENV])
PYTHON=VENV/'bin/python'; constraint=WORK/'torch-constraint.txt'
constraint.write_text('torch=='+runtime['torch']+'\n')
run([PYTHON,'-m','pip','install','--quiet','--upgrade','-c',constraint,
     'transformers==4.51.3','accelerate==1.6.0','bitsandbytes==0.48.1',
     'huggingface-hub>=0.30,<1','safetensors>=0.5,<1','numpy>=1.26,<3','tzdata'])
if not PROJECT.exists():
    run(['git','clone',REPO,PROJECT],retries=1)
    run(['git','checkout','--detach',COMMIT],cwd=PROJECT)
else:
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    if head!=COMMIT: raise RuntimeError('Source revision changed; refusing an incompatible resume')

env=os.environ.copy()
env.update(HF_HOME='/kaggle/tmp/nvda_hf_cache',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
    NCCL_DEBUG='WARN',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
    PYTHONUNBUFFERED='1',NVDA_WORK=str(WORK),NVDA_STATE=str(BASE),NVDA_PROJECT=str(PROJECT),
    PYTHONPATH=str(SOURCE)+os.pathsep+str(PROJECT))
# Credentials go only into the ingestion subprocess environment. Never write them to outputs.
ingest_env=env.copy()
if USE_ALPACA_NEWS and not (WORK/'raw_news.json').exists():
    try:
        from kaggle_secrets import UserSecretsClient
        secrets=UserSecretsClient()
        for name in ('APCA_API_KEY_ID','APCA_API_SECRET_KEY'): ingest_env[name]=secrets.get_secret(name)
    except Exception:
        raise RuntimeError('Enable both APCA_API_KEY_ID and APCA_API_SECRET_KEY in Kaggle Add-ons > Secrets.') from None
for name in ('APCA_API_KEY_ID','APCA_API_SECRET_KEY'):
    env.pop(name,None)

# The next embedded mapping installs the complete, reviewable pipeline source.
