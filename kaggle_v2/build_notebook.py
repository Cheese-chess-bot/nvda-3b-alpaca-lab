import ast,json
from pathlib import Path
root=Path(__file__).resolve().parent
names=['hardware.py','core.py','prepare.py','ppo.py','evaluate.py','runtime.py','test_core.py']
files={name:(root/name).read_text() for name in names}
for name,source in files.items(): ast.parse(source,filename=name)
cell=(root/'launcher_prefix.py').read_text()+'\nFILES = '+repr(files)+'\n\n'+(root/'launcher_suffix.py').read_text()
ast.parse(cell,filename='kaggle_cell.py')
(root/'kaggle_cell.py').write_text(cell)
intro='''# NVDA: Qwen news + math + PPO • dual T4

Updated executable research notebook. **One code cell** writes the full Python modules, checks dependencies, ingests news, runs Qwen on both GPUs, then trains a separate PPO actor-critic on both GPUs.

| Component | What runs |
|---|---|
| Qwen base | Frozen `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4 on each T4; structured article analysis |
| News signals | Sentiment, relevance, uncertainty, event categories, age, coverage/missingness |
| Mathematical features | 85 causal price/volume/relative-market features from NVDA, QQQ, SPY and SOXX; 97 features including news |
| Reinforcement learning | Small neural actor-critic trained from scratch with clipped PPO, GAE and synchronous DDP |
| Position choices | Cash or 2.5%, 5%, 7.5%, 10% NVDA; independent $1,000 notional cap |
| Bounded evolution | At most three candidate policies; validation-only selection; one reserved test per cycle; gated research champion |
| Recovery | Bounded transient retries, article cache, atomic PPO checkpoints, provenance checks, champion rollback and persistent halt |

**Qwen stays frozen in this version. PPO is the component being trained.** These modules replace the earlier supervised return-classification notebook workflow. They do not claim to use every data type or every trading method. The news schema can be extended later with genuinely available fundamentals, filings, macro and options data.

The notebook submits **no orders**. `runtime.py` exports risk-gated proposals; the public repo's older supervised Alpaca runner does not load this PPO policy. No profitability guarantee is implied.
'''
setup='''## Before running

1. Import this notebook into Kaggle. Select **GPU T4 x2** and enable **Internet**. Use a fresh session.
2. Supply news through one of the following options, then edit the constants at the top of the code cell.
   - Set `NEWS_JSONL` to an attached JSONL file. Each line needs `id`, `source`, `available_at` with a timezone, and `headline` and/or `summary`. Content must be the version actually available at that timestamp. If an article was revised, use its revision time.
   - Or set `USE_ALPACA_NEWS=True` and enable the Kaggle Secrets `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. Your Alpaca connection in ChatGPT does not automatically expose credentials to Kaggle. This path calls only Alpaca's news-data GET endpoint.
3. Run the single code cell. The default `REQUIRE_NEWS=True` stops if news is absent or valid news coverage falls below 50% in any split. `REQUIRE_NEWS=False` explicitly permits an ablation; it does not create fake news signals.
4. Use **Save Version with outputs** when finished or before the session expires.

JSONL schema example (replace the sample with real timestamped data; do not train on this illustrative row):
```json
{"id":"article-id","source":"publisher","available_at":"2026-01-02T14:00:00Z","headline":"Actual published headline","summary":"Actual published summary"}
```

At most four eligible articles per decision date are analyzed from the preceding 72 hours. Revisions are joined point-in-time. The defaults cap Qwen analysis at 6,000 unique articles and Alpaca retrieval at 2,000 pages. Budget exhaustion stops the run rather than silently truncating the corpus. Invalid Qwen JSON is recorded as missing. News can take much longer than PPO training.

The starter price data is pinned to public repo commit `706ae9f5861eb88347f977d74533aaed058b5cd9`: 1,128 training days, 124 validation days and 126 test days. It is an IEX historical sample through October 2, 2026. `DATASET_DIR` can point to an attached replacement directory with the same `bars.json` and `dataset.json` schemas; all split boundaries, timestamps and target-return alignments are checked. Numerical features are recomputed from bars, and the scaler is fitted on training data only.
'''
operations='''## What the training and recovery actually do

**News stage:** two independent processes each load a quantized Qwen model and analyze half of the selected articles. Model weights are pinned to an exact Hugging Face revision, and each cache entry records the article and prompt/model identity. Qwen receives article text, not return labels. A pretrained model can still contain historical knowledge; timestamp filtering cannot remove that limitation.

**RL stage:** the Qwen processes exit to release GPU memory. A small actor-critic then uses two-GPU DistributedDataParallel. Each worker collects 256 historical transitions per update; PPO uses clipped probability ratios, a value loss, entropy regularization, generalized advantage estimates and clipped gradients. Candidate learning rates and seeds are fixed before training. Candidate selection uses validation return minus twice validation drawdown. This uses both T4s, but the small PPO network does not require their combined memory and is not promised to train twice as fast.

**Reward and simulation:** reward is net log return with additional drawdown-increase and turnover penalties. Each trade pays 10 basis points per side, with a 20-basis-point stress evaluation. Holdings drift with prices; rebalances and terminal liquidation are charged. Prior-close features drive an idealized next-open trade, with account state marked at that open. This is daily-bar historical replay, with no intraday fills, order book, market impact curve, dividend cashflows or cash interest. Loss-limit gaps may overshoot the configured threshold.

**Independent gates:** a maximum 10% position and $1,000 notional, 2% daily portfolio-loss halt and 8% drawdown halt live in fixed code outside PPO. A halt latches for the remainder of a simulated episode. Runtime proposals additionally require reconciled positions, no outstanding orders, fresh quotes, valid feature timestamps and acceptable feature drift. Recovery never automatically clears a risk halt, and a halt alone does not liquidate an existing real position.

**Evolution:** each `CYCLE_ID` has its own immutable config, caches and checkpoints; cycles share one persistent holdout ledger and research champion registry. A maximum of three candidates is trained. The winner is tested once against cash, constant 10% exposure, trend and mean-reversion baselines. Promotion requires positive net and stressed returns, Sharpe at least 0.75, drawdown at most 5%, at least 30 exposed days, beating all baselines and an incumbent when one exists. These are research thresholds, not proof of future profitability.

The bundled historical sample has already been inspected. **Its results cannot promote a new champion:** promotion also requires test dates strictly after **2026-10-02**. A later cycle must supply genuinely fresh, nonoverlapping dates through `DATASET_DIR`, keep the global ledger, and use a new `CYCLE_ID`. Do not reuse a seen test partition to tune candidate settings. The ledger is reserved before evaluation; an interrupted evaluation consumes that interval. The notebook does not autonomously collect future datasets or rewrite its own code.

**Recovery:** completed article results and PPO updates are written atomically. A resumed candidate restores its model, optimizer, validation history and best policy; per-update/per-rank seeding reconstructs rollouts. Transient process failures get a bounded retry. Invalid input, incompatible code/configuration, NaN losses or exhausted retries stop. `runtime.propose(BASE, ...)` verifies the champion hash; integrity failure attempts to restore the previous champion and retains a persistent halt for review. This is bounded recovery, not unrestricted self-modifying software.

## Outputs and resume

Outputs are under `/kaggle/working/nvda_news_ppo_v2/`:

- `cycles/<CYCLE_ID>/nvda_news_math_ppo_bundle.zip`: Python modules, policy weights, feature metadata and reports.
- `cycles/<CYCLE_ID>/checkpoints/`: complete PPO optimizer/model state.
- `cycles/<CYCLE_ID>/news_cache/`, `prepared.json`, `config.json`: reusable preprocessing and exact provenance.
- `holdout_ledger.json`, optional `champion.json` and `risk_halt.json`: persistent research and recovery state shared by cycles.

For a new Kaggle session, attach saved outputs as a dataset and set `RESTORE_FROM` to the **entire** saved `nvda_news_ppo_v2` directory. Keep the original settings and reattach any input datasets at their original paths. The compact ZIP alone cannot resume training. Do not publish provider-licensed news/data outputs or credentials with a public notebook; the code itself contains no keys.

## Validation of this deliverable

Passed locally: 20 new NumPy/unit tests, the repo's 24 existing tests, module and notebook syntax validation, and an integration check across all 1,378 historical samples. The checks cover cost accounting, weight drift, caps, halts, feature causality, missing/revised news, training-only normalization, GAE, holdout reuse, promotion and rollback.

**Not run here:** Qwen generation, PyTorch PPO optimization, NCCL communication, Kaggle package installation or dual-T4 execution. Their hardware/runtime validation remains part of the Kaggle run. No model-performance result is claimed.

References: [PPO paper](https://arxiv.org/abs/1707.06347), [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html), [Qwen model and its license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Alpaca news endpoint](https://docs.alpaca.markets/us/reference/news-3).
'''
def markdown(text): return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.11'),kaggle=dict(accelerator='gpu',isInternetEnabled=True,language='python',sourceType='notebook')),
    cells=[markdown(intro),markdown(setup),dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(keepends=True)),markdown(operations)])
for i,c in enumerate(notebook['cells']): c['id']=f'nvda-ppo-{i}'
path=root.parent/'NVDA_Kaggle_Dual_T4.ipynb'
path.write_text(json.dumps(notebook,indent=1))
print('Built',path,'bytes',path.stat().st_size,'embedded modules',len(files),'code cells',1)
