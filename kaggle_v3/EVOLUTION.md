# Next stage: bounded self-evolution

Status: design contract for the next release. V3 supplies the two-model analysis pipeline and a real `run_manifest.json`; it does not run an automatic evolution scheduler, edit its own code, deploy policies or place orders.

The existing checkpoint/retry/cache behavior provides bounded recovery. A controller can later call `train_box.py` and consume these artifacts:

- `run_manifest.json`: schema, model identities, configuration/data fingerprints, analyst coverage and selected-policy checksum.
- `selection.json`: finite validation-ranked candidates, with no test-based selection.
- `holdout_report.json`: created only by explicit final evaluation, with freshness and risk checks.
- Atomic checkpoints and separate model caches: resume the same immutable run after interruption.

## Intended workflow

1. Acquire versioned data with reliable availability timestamps and preserve source/model revisions.
2. Propose a bounded number of challengers using a predefined parameter space. Set explicit compute, elapsed-time and failure budgets.
3. Train and select on training/validation only. Compare Qwen-only, Gemma-only, combined and math-only approaches as predeclared experiments.
4. Reserve a genuinely fresh chronological holdout before scoring. Use one central durable ledger across versions and experiments. A new folder, model, schema or random seed does not reset test history.
5. Compare the selected challenger with fixed baselines and the incumbent at the same costs and risk budget. Handle incompatible feature schemas explicitly using each model's own feature preparation; do not silently remove the incumbent comparison.
6. Retain the incumbent unless all fixed checks pass. Require a staged paper-observation period before activating a replacement; preserve the previous artifact and its provenance for rollback.
7. Monitor missing data, stale inputs, drift, model corruption and drawdown. Retry only bounded transient failures. Halt on an unresolved invariant violation; rollback must retain the halt for review.

Research selection cannot relax the independent exposure, notional, loss or drawdown limits. Language-model output must remain structured data, never executable code, shell commands or broker instructions. A proposed strategy change must pass the same validation and fresh-holdout process.

The next implementation needs a persistent scheduler, global experiment/holdout registry, budget enforcement, explicit candidate state machine, paper-observation metrics and promotion/rollback transactions. None of those should be inferred from a successful V3 training run.
