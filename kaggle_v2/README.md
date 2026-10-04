# Kaggle news, mathematical features and PPO

Use the complete [dual-T4 notebook](../NVDA_Kaggle_Dual_T4.ipynb) for setup, training, evaluation and resume. It embeds these sources and writes them into each Kaggle cycle directory.

- `core.py`: causal features, news schema, portfolio simulator, GAE, risk and promotion gates.
- `prepare.py`: Alpaca/JSONL news ingestion, two-GPU frozen-Qwen extraction, point-in-time feature preparation.
- `ppo.py`: clipped PPO, actor-critic, synchronous DDP and resumable optimizer checkpoints.
- `evaluate.py`: one selected policy, reserved holdout, baselines, stress test and gated promotion.
- `runtime.py`: integrity-checked, risk-gated trade proposals; does not submit Alpaca orders.
- `test_core.py`: 20 CPU tests.
- `launcher_prefix.py`, `launcher_suffix.py`, `build_notebook.py`: notebook assembly.
- `kaggle_cell.py`: the complete generated single-cell script.

From repository root, run `python -m unittest discover -s kaggle_v2 -p test_core.py -v` after installing NumPy. Rebuild the notebook with `python kaggle_v2/build_notebook.py`.

Qwen remains frozen. PPO trains a smaller numerical policy. Supply timestamped news or explicitly select a math-only ablation. Keep saved outputs, the global evaluation ledger and risk-halt state across sessions. GPU execution remains unverified in the delivery environment; inspect the notebook's limitations before interpreting results.
