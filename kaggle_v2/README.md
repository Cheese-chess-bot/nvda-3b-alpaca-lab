# Start here: Kaggle Box

Open [kaggle-box.ipynb](kaggle-box.ipynb), enable Internet, select an accelerator if available, and **Run All**. It clones this repository, sets up dependencies while preserving your installed PyTorch build, detects working hardware, and starts PPO training.

Defaults require no keys: the bundled market history starts a clearly labelled math-only run. Supply a timestamped news JSONL or both Alpaca secrets to enable Qwen sentiment, or set `NEWS_MODE="required"` to enforce news. API errors do not silently disable an explicitly enabled news source.

Supported runtime paths: NVIDIA CUDA, AMD ROCm, Intel XPU, and Intel/AMD CPU. Multiple CUDA/ROCm GPUs use NCCL; multiple Intel GPUs use XCCL when available. Otherwise one device is used. A usable vendor driver and matching PyTorch build are required for GPU acceleration; the notebook cannot install hardware or drivers into Kaggle. If PyTorch is absent, the setup installs a CPU build.

New entry-point files:

- `kaggle-box.ipynb`: small notebook that imports the repo and calls the modular runner.
- `train_box.py`: dependency setup, configuration, data/news pipeline, training, checkpoints and outputs.
- `hardware.py`: backend detection and a real allocation/forward/backward/optimizer probe.
- `test_hardware.py`: CPU-testable backend routing and news-mode checks.

From a cloned repository, the same default run is available with:

```bash
python kaggle_v2/train_box.py
```

Use `python kaggle_v2/train_box.py --help` for device, news and budget overrides. The default final holdout is left unevaluated; `--final-test` consumes it through the existing ledger. Source changes require a new cycle name; keep the global ledger. Save the entire output folder for resume. Model weights are generated during training, not supplied as a performance claim.

Validation: 32 core/device-routing tests and 24 original repository tests pass. A real CPU run, dependency-environment setup, completed-checkpoint resume and policy reload were exercised. The local two-process check timed out; distributed execution and NVIDIA/AMD/Intel GPU execution remain unverified on physical hardware.

# Kaggle news, mathematical features and PPO

Use the complete [dual-T4 notebook](../NVDA_Kaggle_Dual_T4.ipynb) for setup, training, evaluation and resume. It embeds these sources and writes them into each Kaggle cycle directory.

- `core.py`: causal features, news schema, portfolio simulator, GAE, risk and promotion gates.
- `prepare.py`: Alpaca/JSONL news ingestion, device-aware frozen-Qwen extraction, point-in-time feature preparation.
- `ppo.py`: clipped PPO, actor-critic, synchronous DDP and resumable optimizer checkpoints.
- `evaluate.py`: one selected policy, reserved holdout, baselines, stress test and gated promotion.
- `runtime.py`: integrity-checked, risk-gated trade proposals; does not submit Alpaca orders.
- `test_core.py`: 20 CPU tests.
- `launcher_prefix.py`, `launcher_suffix.py`, `build_notebook.py`: notebook assembly.
- `kaggle_cell.py`: the complete generated single-cell script.

From repository root, run `python -m unittest discover -s kaggle_v2 -p test_core.py -v` after installing NumPy. Rebuild the notebook with `python kaggle_v2/build_notebook.py`.

Qwen remains frozen. PPO trains a smaller numerical policy. Supply timestamped news or explicitly select a math-only ablation. Keep saved outputs, the global evaluation ledger and risk-halt state across sessions. GPU execution remains unverified in the delivery environment; inspect the notebook's limitations before interpreting results.
