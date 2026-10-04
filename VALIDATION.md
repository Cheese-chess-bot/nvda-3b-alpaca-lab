# Validation performed — October 3, 2026

- Python syntax compilation: PASS.
- Standard-library unit and mocked integration suite: **24 tests passed**.
- Included provider data: 1,444 daily bars each for NVDA, QQQ, SPY, SOXX.
- Chronological dataset: 1,128 train, 124 validation, 126 test samples.
- Real-data baseline backtests executed; results in data/baseline_report.json.
- Tests cover future-bar invariance, news revision cutoff, missing sessions, label alignment, purged partitions, cost accounting, budget/cash limits, stale/crossed/wide quotes, uncertainty abstention, durable submission intent, lost-response reconciliation, duplicate prevention, model tampering, dry run, mocked paper submission/restart, loss halt, manual halt, and foreign-position rejection.

## Not performed

- No CUDA GPU, PyTorch, Transformers or PEFT available in the delivery environment.
- No 3B weights downloaded or trained; no candidate model evaluated or promoted.
- No actual paper orders sent. All order-writing tests used a fake broker in temporary directories.
- Standalone API keys were not accessed; price retrieval used the connected Alpaca plugin.
- Historical news/filings/macro/options feeds have not been populated in the included dataset.
- No live-money testing, profitability finding, execution latency measurement, or full dependency compatibility validation.

The project is a research implementation, not an audited production trading system.
