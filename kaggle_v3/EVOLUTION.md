# Bounded evolution, recovery and paper observation

V3 implements a finite research controller inspired by the supplied guide. It evolves **PPO parameters**, remembers failed experiments, and restores verified checkpoints. Qwen and Gemma remain frozen. It does not rewrite source code, invent missing fundamentals, or guarantee income.

## Run on Kaggle

Use the updated `kaggle-box.ipynb`, a **new cycle ID**, Internet enabled, and Run All. New defaults:

```python
RUN_EVOLUTION = True
EVOLUTION_REVISIONS = 3
BUDGET_MINUTES = 240
RUN_FINAL_TEST = False
```

Without news data/credentials this is explicitly math-only PPO. To use both analysts, set `NEWS_MODE="required"`, supply timestamped news and enable `HF_TOKEN` with Gemma access. The notebook never starts the paper adapter. Keep the **entire output directory** with Save Version WITH OUTPUTS.

```bash
python kaggle_v3/train_box.py --evolve --evolution-revisions 3 --budget-minutes 240
# Small wiring check, not a useful trained trading model:
python kaggle_v3/train_box.py --evolve --device cpu --news off --updates 2 --evolution-revisions 2 --cycle smoke_001
python kaggle_v3/evolution.py --state nvda-box-v3-output --status
```

## Research cycle

1. **Read memory; write independent memos.** Deterministic price, sector/index proxy, news and skeptic specialists receive only their assigned features plus prior memory, not each other's memos. Qwen/Gemma supply news signals. There is no fundamentals/filings feed; `sources.json` records the gap.
2. **Preregister.** SQLite commits the hypothesis, parameters, prediction and failure conditions before training. Same-hypothesis revisions are capped at three across campaigns. Changing the cycle ID or regime label cannot reset this cap.
3. **Walk forward.** Three expanding development windows each have an earlier training split, purged internal validation split, and later out-of-sample segment. Fit scaling on each fold's training data. Fold jobs contain no final-test partition.
4. **Break the candidate.** Evaluate twice the costs, a one-session action delay, and five non-overlapping worst 20-session asset windows per fold where possible. Compare cash, fixed 10% exposure, trend and mean-reversion baselines. Risk state starts flat per fold, with terminal liquidation costs.
5. **Count selection trials.** Reserve every potential inspected PPO update checkpoint across planned folds/revisions and final refit, including work skipped by early stopping/crashes. Import older regular runs under the same root when possible. The DSR diagnostic uses daily Sharpe, skewness, Pearson kurtosis, raw trial count and cross-trial Sharpe variance with a sampling-variance floor. Correlated/adaptive trials and serial dependence remain unresolved; this is approximate, not a calibrated probability of profitability.
6. **Remember, then revise.** Save failed checks and lessons. Cost/drawdown failures prioritize an unused lower-learning-rate/entropy preset. Only learning rate, entropy, discount and seed can vary. Code, risk limits, holdouts and broker commands cannot evolve.
7. **Refit one artifact.** Passing revisions rank first. If all fail, export the best result as `failed_research` for inspection, without paper eligibility. Refit uses training/validation only.
8. **Require a fresh holdout.** Explicit final evaluation requires passed development gates and dates later than the ledger's initial freshness boundary. Reserve the interval before evaluation; a crash consumes it. SQLite rejects overlap; the previous JSON ledger is imported and mirrored.

The initial freshness boundary is the later of `2026-10-02` and the UTC date when memory is first created. Bundled history cannot qualify. Collect fresh data over time and preserve the same memory. Retain prior ledgers across all versions; unrelated/deleted state cannot be detected across machines. A short historical backtest is not evidence of reliable income. Pretrained models may already know historical events.

## Modules and outputs

| Module | Responsibility |
|---|---|
| `research.py` | Independent memos, immutable predictions, bounded proposals |
| `ledger.py` | SQLite hypotheses, attempts, trial counts, holdouts, decisions and order IDs |
| `validation.py` | Purged folds, stress traces, baselines, approximate DSR |
| `evolution.py` | Train/evaluate/remember/revise controller |
| `recovery.py` | Kernel locks, deadlines, transient retry and artifact rollback |
| `risk.py` | Independent pre-order limits, using fractional percentages |
| `paper.py` | Separate opt-in Alpaca paper adapter and critic report |
| `test_evolution.py` | Offline failure-path tests; fake broker APIs |

Memory lives in `memory/ledger.db` at the output root. Cycle outputs include `evolution/research/`, immutable `revision_*/prediction.json`, fold checkpoints, `report.json`, `post_mortem.json`, `refit/` and `summary.json`. `evolved_policy.pt`, `selection.json`, `box_result.json` and `run_manifest.json` identify the review artifact. Paper events are in SQLite; critic exports appear under `journal/`.

SQLite uses WAL and FULL synchronization. Stop writers before copying the entire output directory. Copying only `ledger.db` while active can omit WAL transactions. Restore the entire directory, including WAL files. Old backups or deleted memory invalidate unseen-test claims. These local files are auditable, not tamper-proof against their owner.

## Self-healing

- Kernel locks release on process death; live locks are never stolen. Stop an old-version controller before migrating its lock.
- Atomic checkpoints have SHA-256 sidecars and one verified predecessor. Corruption restores the predecessor only with a matching data fingerprint. Trial/revision counts remain consumed.
- Allowlisted temporary network/distributed failures get a bounded retry. OOM, non-finite loss, changed data, poor coverage and integrity faults stop the run. Recovery cannot switch models/devices inside an immutable experiment.
- Quiet/stuck subprocesses terminate at the deadline. A campaign retains its elapsed budget across resume. Kaggle shutdowns still require saved outputs and a notebook rerun; it is not an always-on server.
- Damaged paper artifacts roll back to a verified predecessor where available and **keep the risk halt latched**. No automatic risk reset or uncertain-order resend exists.

## Optional paper adapter

Training never imports or executes the adapter. It uses the Alpaca SDK's fixed `paper=True` environment, with no live-money switch. Install `alpaca-py` in the execution environment using its official documentation. Supply paper keys through environment variables, never source code. Use a **dedicated, unleveraged NVDA paper account** without unrelated positions/orders.

Only after research and a fresh holdout pass:

```bash
python kaggle_v3/paper.py --state /path/to/nvda-box-v3 --stage /path/to/nvda-box-v3/cycles/approved_cycle
# Dry run, using CURRENT upstream features:
python kaggle_v3/paper.py --state /path/to/nvda-box-v3 --features /path/to/features.json
# Explicit one-decision paper submission:
python kaggle_v3/paper.py --state /path/to/nvda-box-v3 --features /path/to/features.json --submit-paper
python kaggle_v3/paper.py --state /path/to/nvda-box-v3 --review
```

`features.json` requires `as_of` with a timezone and `raw_features` with the exact 124 names/finite values in the bundle. A current point-in-time feature producer is required; do not use training rows as live inputs. The adapter does not reconstruct live news/features for you. Features may be at most 96 hours old for weekends/holidays; quotes/account snapshots at most 30 seconds. Broker state is checked again before submission.

Risk defaults are fractions: `.02` daily loss, `.08` drawdown, `.10` position cap, plus `$1,000` absolute position value. These are example engineering defaults, not investment advice. Orders are whole-share DAY limits. No shorts/leverage; small/fractional residuals are not automatically liquidated. Halts block new submissions; **they do not cancel orders or flatten positions**. Inspect and manage the broker explicitly when a halt occurs.

Order identity is reserved before submission, allowing at most one order per account/session across candidate versions. A timeout records `unknown` and latches a halt. Reconcile the recorded client ID; never blindly resend:

```bash
python kaggle_v3/paper.py --state /path/to/nvda-box-v3 --reconcile nv3-RECORDED_CLIENT_ID
```

Reconciliation never clears a halt. A broker lookup failure stays unresolved. There is no reset command: fix and review the cause before manually clearing a halt.

The critic separates shadow from paper observations, summarizes the last week's events, and writes lessons to memory. After 20 observed paper sessions it requests human review, **without automatic promotion**. Account transfers, fills and attribution must be reviewed. Snapshots are first eligible observations per session, not official closing NAV. Weekly scheduling is external; no Sunday automation/daemon is installed. There is no paper-to-live transition.

## Verification and limits

Run `python -m unittest discover -s kaggle_v3 -p 'test_*.py' -v` with NumPy/PyTorch installed. Tests cover immutable predictions, global revision caps, overlap rejection, trial counts, purged scaling, DSR direction, deadlines, lock exclusion, verified checkpoint recovery, rollback with a latched halt, order limits, dry runs and uncertain-submission idempotence.

A local CPU check executes actual PPO folds/refit and completed-run resume. Broker tests use fakes; no Alpaca orders are placed during development. Full frozen-model inference, dual-T4 execution and real broker connectivity still require target-environment validation. Daily fixed costs/delayed actions do not model real order-book fills. Daily win rate and rebalance days are not completed-trade win rate/count.

Primary sources: [Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [Alpaca-py paper trading](https://alpaca.markets/sdks/python/trading.html), [order requests](https://alpaca.markets/sdks/python/api_reference/trading/requests.html).
