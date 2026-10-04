# NVDA 3B Research Lab

## New: dual-T4 Qwen news + math + PPO pipeline

Open [NVDA_Kaggle_Dual_T4.ipynb](NVDA_Kaggle_Dual_T4.ipynb) in Kaggle with **GPU T4 x2** and Internet enabled. The notebook contains one executable cell and the full setup/resume instructions. Reviewable Python source is in [kaggle_v2/](kaggle_v2/).

- Frozen Qwen2.5-3B extracts timestamped news sentiment, relevance, uncertainty and event categories.
- 85 numerical market features plus news signals produce 97 combined features, with training-only normalization.
- A separate actor-critic trains with actual PPO/GAE using both GPUs.
- Up to three candidates compete on validation data, with a single-test ledger, independent risk limits and gated research promotion.
- Atomic checkpoints, cached news extraction, bounded retries and integrity-triggered rollback provide recovery.

Supply news through an attached JSONL file or Alpaca Kaggle Secrets. Qwen stays frozen; PPO learns the trading policy. The new policy produces proposals and is **not yet connected to the older Alpaca order-execution runner below**. Historical starter dates are already inspected, so promotion requires genuinely fresh test dates after 2026-10-02.

Validation: 20 new CPU tests plus the 24 existing tests pass, and feature preparation was checked on all 1,378 rows. GPU training and Kaggle execution have not been run in the delivery environment. No trained model or profitability result is included.

The remaining sections document the original supervised LoRA pipeline.

Runnable Python research and Alpaca **paper-only** execution code. Includes actual Alpaca-plugin daily data, a dataset builder, Qwen2.5-3B LoRA/QLoRA classification training, cost-aware evaluation, model promotion/rollback, and a durable order journal. No trained weights or proven trading edge are included. No orders were placed to build this package.

## What this version does

- Uses split-adjusted OHLCV, VWAP, range, volume, momentum, volatility and mean-reversion features for NVDA, QQQ, SPY and SOXX.
- Fetches paginated Alpaca news. News revisions become usable only at their `updated_at` timestamp.
- Accepts filings, fundamentals, earnings, macro releases, transcripts, options summaries, microstructure summaries and alternative data as versioned, timestamped events. These feeds are **not automatically sourced** by this project.
- Fine-tunes the pretrained **Qwen2.5-3B-Instruct** backbone with LoRA adapters and a three-class prediction head: DOWN, FLAT, UP. This is not 3B-parameter training from scratch. Inputs are numeric features serialized as text plus bounded event text, not raw images/audio/video/order-book streams.
- Targets next-session-open to following-session-open NVDA returns. Labels use a fixed +/-0.3% neutral band. The strategy is long or cash, never short, with uncertainty and high-volatility abstention.
- Uses chronological training/validation/test partitions and purges samples whose label periods overlap the next partition. Chooses epoch, temperature and action threshold on validation only.
- Tests against cash, NVDA buy/hold, trend and mean-reversion baselines at matching nominal 10% exposure, plus double-cost stress. A candidate must pass every fixed gate to become a paper champion.
- Implements bounded GET retry/backoff, single-process locks, atomic JSON writes, adapter checksums, rollback, broker reconciliation, and a SQLite intent committed before any order submission.
- Never retries an ambiguous POST. Client order IDs are deterministic per session. A lost response halts execution until reconciliation. A crash before a POST can leave an unresolved intent: the bot deliberately misses that trade rather than risking duplication.

## Included data and verified work

`data/bars.json` contains 1,444 daily IEX bars per symbol for NVDA, QQQ, SPY and SOXX, January 2021 through October 2, 2026, retrieved using the connected Alpaca Paper Trading plugin. These are provider-returned observations, not independently audited exchange records. The included dataset has 1,128 training, 124 validation and 126 test samples after purging; initial event coverage is empty. Missing modalities are explicitly represented as empty lists.

The core dataset builder and baseline backtest were executed. The unit/integration suite tests causality, chronology, risk limits, order idempotency and recovery. See `VALIDATION.md` for actual results. CUDA training, model accuracy, authenticated standalone API access, and real broker fills have **not** been validated here. The standalone program does not inherit the ChatGPT plugin's authentication.

## Quick start: inspect the included data, no credentials needed

Use Python 3.11 or 3.12. Run from the extracted `nvda_lab` directory:

```bash
python -m pip install tzdata
python nvda_lab.py build
python nvda_lab.py baseline
python -m unittest discover -s tests -v
```

Core data processing and tests otherwise use only the standard library. `tzdata` is particularly needed on Windows. The baseline output is a simulation, not the 3B model's performance.

## Refresh data and add news

Generate **paper** API keys in your Alpaca dashboard. Set them in your local shell; do not paste keys into chat, source files, notebooks you share, or version control.

PowerShell:

```powershell
$env:APCA_API_KEY_ID = Read-Host 'Alpaca paper key ID'
$secret = Read-Host 'Alpaca paper secret' -AsSecureString
$env:APCA_API_SECRET_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
python nvda_lab.py fetch --start 2021-01-01 --news
python nvda_lab.py build
```

Bash:

```bash
read -r -p 'Alpaca paper key ID: ' APCA_API_KEY_ID
read -r -s -p 'Alpaca paper secret: ' APCA_API_SECRET_KEY
export APCA_API_KEY_ID APCA_API_SECRET_KEY
python nvda_lab.py fetch --start 2021-01-01 --news
python nvda_lab.py build
```

History/news access depends on account entitlements. HTTP errors stop the requested fetch; missing paid data is not synthesized. IEX is the default feed. `--feed sip` requires appropriate access and should be used consistently across training and execution. IEX covers one exchange; it is not consolidated whole-market volume or a full order book.

Fetching several years of news may be slow. For daily refresh, fetch bars with the same historical start; request news separately through the same `fetch --news` command if your model uses it. `fetch` replaces the historical bar snapshot; passing a short `--start` shrinks the training history. News is replaced only when `--news` is supplied. Keep event feeds refreshed and rebuild/retrain intentionally; the included program cannot verify a third-party feed's completeness.

## Add other data types

Create a JSON array or JSONL file with this schema. This example is illustrative, not included training evidence:

```json
{
  "id": "your-provider-unique-event-and-vintage-id",
  "kind": "fundamental",
  "symbol": "NVDA",
  "available_at": "2026-08-27T21:00:00+00:00",
  "source": "your licensed source URL or document identifier",
  "text": "Your factual numeric summary of the release, with units and reporting period."
}
```

```bash
python nvda_lab.py import-events your_events.jsonl
python nvda_lab.py build
```

Supported `kind` values: `news`, `filing`, `fundamental`, `macro`, `options`, `microstructure`, `earnings`, `transcript`, `alternative`. Symbols: NVDA, QQQ, SPY, SOXX, or `*` for market-wide events.

`available_at` means the instant this specific version was publicly usable, including vendor latency if known. It is **not** a fiscal period end, economic reference month, option expiry or document creation time. Use original filing acceptance times and release vintages; never put a restatement under an earlier timestamp. Keep multiple versions under the same `kind`/`id` and distinct availability timestamps. The latest eligible version is selected at each cutoff. The code checks schema and time ordering, not truthfulness or data licensing.

Context is limited to the latest two events per modality within 100 days, with 450 characters per event and an overall 1,024-token cap. Truncation can omit later modalities. Summarize long filings externally from the original dated document; this version is not a full document retrieval engine. Historical options, L2, estimates and alternative data usually need separately licensed sources. This does not provide "all market data."

## Train the 3B model

Recommended starting environment: Linux/WSL2 or a CUDA notebook with one NVIDIA GPU and roughly 16–24 GB VRAM for QLoRA, plus model-download disk space. This is a planning estimate, not a measured requirement. Batch size is one, sequence length is 1,024, and eight microbatches are accumulated. Full LoRA uses more memory. Training from scratch would require a radically larger dataset and compute budget.

Install a CUDA-enabled PyTorch 2.6 build appropriate for your driver, then:

```bash
python -m pip install -r requirements-training.txt
python nvda_lab.py train --run candidate_001 --qlora --epochs 3
python nvda_lab.py evaluate --run candidate_001 --promote
```

The training command resolves and records an immutable Hugging Face model revision. Optional `--revision COMMIT_SHA` selects a known revision. The classifier head starts randomly and is saved with the LoRA adapter; adapter weights alone are not a complete foundation model. Model downloads may require accepting the publisher's license. Review Qwen's model card and model-specific license before commercial use; no model license is granted by this package.

No GPU was available in the delivery environment. The training path is implemented and syntax-checked but not a tested claim of GPU compatibility or profitability. The pinned library API set is intentional; validate dependency updates before changing it. A GPU OOM or nonfinite loss aborts the run and cannot promote an incomplete candidate. Completed epoch-best adapter weights are saved; mid-epoch optimizer resume is not implemented.

## Evaluation and promotion

Defaults reserve 126 sessions for validation and 126 for the final test. At least 250 training and 63 test/validation sessions are required. Two boundary samples are removed where necessary because of the forward label horizon. Final test windows are consumed durably **before** scoring; a crash still consumes the window.

Promotion requires all of:

- Positive net return at 10 bps per side and at 20 bps stress cost.
- Zero-risk-free-rate annualized Sharpe at least 0.75.
- Maximum portfolio drawdown no worse than -5%.
- At least 30 invested test sessions.
- Better Sharpe than the fixed baseline family and the incumbent on the same test dates.

Failing a gate leaves the prior champion in place, or leaves trading disabled if no champion exists. The candidate report lives at `runs/<run>/report.json`. Avoid tuning this policy based on a failed test; wait for new data. These simple gates are screening criteria, not statistical proof. They do not include confidence intervals, multiple-testing correction, or a guarantee of significant alpha.

`evaluation_ledger.json` prevents another evaluation with overlapping test dates. With a 126-session test, a new automatic promotion attempt needs a genuinely fresh block. Do not delete the ledger to optimize against the same test. The research design is rolling chronological evaluation across runs, not a claim that multiple historical GPU walk-forward folds have already been trained.

## Start paper trading

A passing candidate must have been promoted first. Use a dedicated, empty NVDA-only paper account. The base URL is hard-coded to `https://paper-api.alpaca.markets`; there is no live-money switch.

```bash
python nvda_lab.py paper
python nvda_lab.py paper --execute --loop
```

The first command is a dry run; the second explicitly enables simulated broker orders. Both need local Alpaca keys. The loop checks every 30 seconds, uses Alpaca's market calendar, and rebalances only 5–20 minutes after the session open. Fetch completed daily data before that window. A closed market produces no order. The latest completed bar must match the prior market-calendar session, including holidays.

Policy:

- Long/cash NVDA only; whole shares, no leverage, options or shorts.
- Target exposure at most 10% of equity **and** $1,000 notional, bounded by cash.
- One submitted IOC limit order per session; canceled or partial fills are not chased that day.
- Quotes at most 30 seconds old; maximum spread 25 bps; limit price collar around the observed quote.
- Reject open orders, other symbols, fractional/short positions, blocked accounts and changed account identity.
- Halt new orders at 2% day-over-day account equity loss, 8% observed account peak drawdown, repeated failures, model-integrity failure or a `HALT` file.

These are entry/rebalance controls, not guaranteed stop-losses. **A halt does not liquidate positions or cancel existing orders.** Overnight gaps, a process outage or a failed sell can leave exposure. Exposure can also drift above the target as prices change. The process must be running for its checks to occur. Review and manage positions in the paper dashboard when halted. Never use the same paper account with another bot or manual concurrent trading; external account changes can race the local checks.

Backtests assume daily rebalancing at the session open with a simple turnover cost model. Paper execution is 5–20 minutes after open, whole-share, cash/notional capped and IOC, so it can differ materially. Baseline backtests report a theoretical 10% sleeve without the fixed $1,000 cap, taxes, dividends, interest, queue simulation or actual fill liquidity. Forward-paper measurement is necessary before drawing conclusions.

## Recovery and bounded evolution

```bash
python nvda_lab.py status
python nvda_lab.py reconcile
python nvda_lab.py reconcile --resume
python nvda_lab.py rollback
```

`reconcile` reads the broker by client order ID and updates the journal. `--resume` explicitly clears a persistent halt only when orders are resolved, no orders remain open, the account is active, equity checks pass and no `HALT` file exists. Unresolved `not_found` intents are never automatically erased. Inspect broker records and preserve the journal when diagnosing them; contact support if necessary. Rollback restores the prior checksummed champion but retains the risk halt. There is no code that bypasses a breached drawdown limit to get back into the market.

A complete retraining cycle is exposed for an external scheduler/controller:

```bash
python nvda_lab.py evolve --run candidate_002 --start 2021-01-01 --news --qlora --promote
```

This refreshes data, rebuilds labels, checks holdout freshness, trains, evaluates and conditionally promotes. Schedule it separately from paper execution, after collecting enough new holdout sessions. Run names must be unique. The existing champion remains in use unless the candidate passes. Do not overlap dataset updates/research with order execution: the program has separate process locks but is not a distributed workflow engine.

"Self-healing" here means bounded transport retries, crash-safe journaling, integrity checks, reconciliation and an explicit rollback path. "Self-evolution" means a fixed retraining/evaluation/promotion pipeline. It does not autonomously edit source code, relax risk settings, buy data, rent GPUs, or try every known strategy. An external controller should call these commands and parse the JSON reports; it should not let model output run shell commands.

## Research limits that matter

No system can ensure profit. One stock has relatively few independent daily samples, and a 3B model may overfit badly or lose to a simple baseline. A pretrained model can have memorized historical financial events, so historical testing cannot establish contamination-free skill. Only prospective evaluation after the model is frozen can address that concern. Split-adjusted prices and vendor news histories may also contain corrections that were unavailable at the historical decision time; this is not a certified point-in-time database.

Paper fills are simulated. Alpaca documents omissions such as market impact, latency slippage, queue position and regulatory fees. IEX quotes differ from consolidated NBBO. Do not interpret a passing backtest or paper profit as a promise of live returns.

## Sources

- Alpaca paper trading: https://docs.alpaca.markets/us/docs/paper-trading
- Alpaca historical bars: https://docs.alpaca.markets/us/reference/stockbars
- Alpaca news: https://docs.alpaca.markets/us/reference/news-3
- Alpaca orders: https://docs.alpaca.markets/us/reference/postorder
- Qwen 3B model card and license: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- PEFT: https://huggingface.co/docs/peft/

Never commit secrets, execution.sqlite, downloaded model weights or account logs to a public repository.
