# Kaggle v3: Qwen + Gemma analysts

Open [kaggle-box.ipynb](kaggle-box.ipynb) in Kaggle, enable Internet, select T4 x2 when available, and Run All. The notebook clones this repo and invokes the modular runner.

This version adds `google/gemma-3-4b-it` alongside `Qwen/Qwen2.5-3B-Instruct`. Each independently analyzes the same timestamped article. Both language models remain frozen; a separate PPO policy trains on numerical features and the resulting signals. This release uses text input only, even though Gemma's architecture also supports images.

## Enable both models

1. Review and accept the Gemma model terms on [its Hugging Face page](https://huggingface.co/google/gemma-3-4b-it) using your account.
2. Enable a Kaggle Secret named `HF_TOKEN` with read access to that checkpoint. The program does not accept terms or create tokens.
3. Supply `NEWS_JSONL` or enable both `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` for Alpaca news. Use `NEWS_MODE="required"` to require a news-enabled run. The bundled dataset does not include a historical news corpus.
4. Leave `NEWS_MODELS="qwen,gemma"`. `qwen` or `gemma` is available as an explicit ablation; access errors never silently change the selected model set.

With `NEWS_MODE="auto"` and no source, the default trains a clearly labelled math-only PPO baseline. It does not download or run either language model. A source that fails fetch, access, timestamp or coverage checks stops the run.

Standalone equivalent, from repository root:

```bash
python kaggle_v3/train_box.py --models qwen,gemma --news required --news-jsonl /path/to/news.jsonl
# Credential-free pipeline check:
python kaggle_v3/train_box.py --news off --device cpu --updates 2 --candidates 1
```

JSONL records require `id`, `source`, a timezone-qualified `available_at`, and `headline` and/or `summary`. Availability must reflect the specific published revision. Alpaca records use the later of creation/update timestamps. Never backdate a revised article.

## Files and features

| File | Role |
|---|---|
| `qwen.py`, `gemma.py` | Separate model adapters |
| `models.py` | Model allowlist, immutable revisions, gated-file preflight and downloads |
| `model_runtime.py` | Shared frozen text generation and strict structured-output prompt |
| `sentiment.py` | Sharded per-model analysis, isolated caches and provenance checks |
| `ensemble.py` | Independent signals, conservative consensus and disagreement features |
| `prepare.py`, `core.py` | Causal ingestion, numerical features, train-only scaling and simulator |
| `ppo.py` | PPO/GAE actor-critic with resumable checkpoints |
| `hardware.py` | CPU/CUDA/ROCm/XPU probing and routing |
| `evaluate.py`, `runtime.py` | Reserved-test reporting and proposal-only risk checks |
| `train_box.py` | Setup, stage orchestration, outputs and run manifest |
| `EVOLUTION.md` | Planned controller contract and promotion workflow |

The policy receives 124 features: 85 market features, 12 Qwen signals, 12 Gemma signals, 12 consensus signals, and 3 disagreement/pair-coverage signals. Invalid model output stays missing. Consensus for an article requires every enabled analyst; disagreement is the mean absolute sentiment difference divided by two. Event disagreement maps to `other`, and consensus uncertainty uses the maximum. Weights are fixed, not tuned on the test set. Single-model and math-only ablations retain the same schema and explicit missing flags.

Coverage is checked on each chronological split. A two-model run needs valid paired analysis on at least 50% of sessions in each split. Per-model and paired coverage are reported separately. Token truncation, model errors and unavailable news can reduce coverage.

## Hardware and resume

Model stages run sequentially, releasing memory between Qwen, Gemma and PPO. On two supported GPUs each analyst stage shards articles across workers; the PPO stage uses distributed training. Both full language models are not kept resident together.

NVIDIA uses NF4 weights. Qwen computes in FP16; Gemma computes in FP32, including on T4, to avoid Gemma's FP16 path. AMD/Intel use unquantized Qwen FP16 and Gemma FP32; CPU uses FP32. Unquantized Gemma weights alone are roughly 16 GB, with additional activation/cache overhead. Supported hardware, drivers and the matching PyTorch build are required. No claim of physical GPU validation is made.

Outputs use `nvda-box-v3/cycles/<cycle>/`. Keep the whole output folder for resume: model-specific news caches, checkpoints, configuration, ledger and selected weights. `kaggle-box-policy.zip` is a portable review/inference bundle, not a complete resumable run. V2 and V3 policy schemas differ; V2 checkpoints cannot be resumed as V3. Code/device/data/model-set changes require a new cycle, never deletion of the evaluation ledger.

`run_manifest.json` records both model revisions, coverage, configuration/data fingerprints and the selected policy checksum for the later evolution controller. The default runs training/validation only. `--final-test` explicitly consumes the holdout; automatic promotion is disabled in V3. The existing Alpaca order runner is not connected to these policies. No orders are submitted.

## Verification and limits

Validated locally: 43 V3 tests pass. A real two-update CPU PPO run completed with 124 features; completed-checkpoint resume preserved identical policy bytes and the saved bundle reloaded. A clearly labelled synthetic two-analyst cache also passed full feature assembly across all chronological rows. The pinned Gemma3 class also generated text from a tiny random test configuration; this checks the API, not the downloaded 4B model.

Run `python -m unittest discover -s kaggle_v3 -p 'test_*.py' -v` with NumPy installed. Tests cover chronology, risk, device routing, analyst disagreement, missing signals, cache provenance and model precision selection. Physical GPU execution and full Qwen/Gemma downloads/inference require validation in the target environment. A synthetic fixture verifies wiring only; it is never evidence of model quality or trading performance.

Historical text may be present in pretrained-model knowledge. Adding Gemma does not establish an edge, and using two models does not guarantee independent errors. Current execution is daily-bar simulation with fixed costs; see the existing research limitations in the repository README. Self-evolution is a planned next stage, not an active autonomous service.

Sources: [Gemma 3 model card](https://huggingface.co/google/gemma-3-4b-it), [pinned Transformers API](https://huggingface.co/docs/transformers/v4.51.3/model_doc/gemma3), [upstream FP16 issue](https://github.com/huggingface/transformers/issues/36822).
