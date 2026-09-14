# CLAUDE.md

This is a fork of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (the *NanoGPT speedrun*).
Remotes: `origin` = this fork (`claudeformanek/modded-nanogpt`), `upstream` = the original repo.
Pull upstream changes with `git fetch upstream && git merge upstream/master`.

## Two entry points

- **`train_gpt.py`** -- the real speedrun script. Requires CUDA + 8xH100 (or at least one CUDA
  GPU). Uses Triton kernels, FP8 matmuls, Flash-Attention-3, and the Muon/NorMuon optimizer.
  Run via `./run.sh` (`torchrun --standalone --nproc_per_node=1 train_gpt.py` -- currently set to
  1 GPU, was 8 originally).
- **`train_gpt_cpu.py`** -- a from-scratch minimal debug script added in this fork so the project
  can be run on a CUDA-less machine (e.g. a Mac laptop). Run via `./run_cpu.sh` or
  `python train_gpt_cpu.py` -- no `torchrun` needed, single-process only.

## Why `train_gpt_cpu.py` exists and what it deliberately is not

`train_gpt.py` cannot run without CUDA at all -- not just slowly. It hard-asserts
`torch.cuda.is_available()`, hardcodes NCCL, and its core ops (Muon's Newton-Schulz
orthogonalization, the MLP and cross-entropy fused kernels, the "DC attention correction") are
raw Triton/CUDA-C with no CPU fallback. Triton itself has no Apple/Metal backend, so
`pip install triton` doesn't fix it either.

Rather than porting every CUDA kernel to a plain-torch equivalent (which was scoped and is
~1-2 days of work, see git history / PR description), the chosen approach was to write a
**standard, from-scratch decoder-only transformer** and drop the speedrun-specific machinery
entirely, since most of it exists purely for 8xH100 training speed, not for being
understandable:

- Dropped: Muon/NorMuon optimizer (replaced with plain `torch.optim.AdamW`), FP8 everywhere,
  Flash-Attention-3, the "DC attention" correction term (a custom cross-head-mixing kernel on
  one layer), bigram-hash embeddings + their sparse cross-rank gradient comms, MuDD gating,
  paired-head attention, YaRN window scheduling, the multi-stage batch/seq-len training
  schedule, `torch.distributed` entirely.
- Kept standard: learned position embeddings, pre-LN transformer blocks, causal self-attention
  via `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`, GELU MLP.
- **Reused as-is** (see module docstring in `train_gpt_cpu.py` for exact diffs): the data
  loading pipeline (`_load_data_shard` / `Shard` / the BOS-aligned document-packing generator,
  renamed `data_generator`) and the shape of the periodic validation-loss loop. Reads the same
  `data/fineweb10B/*.bin` shards as the real script.

Known trade-off from this: attention in the CPU script is plain causal with **no
document-boundary masking**, so a packed sequence containing multiple short documents lets
tokens attend across document boundaries. Fine for debugging; just don't compare loss curves
directly against a real `train_gpt.py` run.

### Gotcha discovered while building this

`torch.empty(..., pin_memory=True)` (used in the original data-loading code for
CUDA-transfer speed) does **not** error on CPU-only Apple Silicon -- it silently allocates the
tensor on the `mps` device instead, which then breaks `.numpy()` downstream with a confusing
`can't convert mps:0 device type tensor to numpy` error. `train_gpt_cpu.py` just never passes
`pin_memory=True`. Worth knowing if you see that error anywhere else in this codebase on a Mac.

## `train_gpt_cpu.py` config knobs

All at the top of the file: `SEQ_LEN`, `BATCH_SIZE`, `NUM_STEPS`, `LOG_EVERY`, `VAL_EVERY`,
`VAL_STEPS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `LR`, `WEIGHT_DECAY`, `BETAS`.

**Heads up:** `NUM_STEPS` is currently `200000`, set manually and left as-is at the user's
choice -- the file's own comment ("finishes in well under a minute") is now stale and describes
the smaller value (200) it was tested with, not the current setting. At the batch size in the
file, expect roughly tens of hours for a full 200k-step run on CPU. Adjust `NUM_STEPS` down if
you just want a quick smoke test.

Epoch fraction (logged as `progress/epoch`) is computed from total tokens seen so far divided
by the total token count across all `TRAIN_FILES` shards (read once at startup, via a
header-only read, not by loading every shard). `1.5` means one and a half passes over the
training data.

## Logging

Uses TensorBoard (`tensorboard` is in `requirements.txt`, install into the venv if missing).
Each run writes to a fresh timestamped dir under `runs/` (gitignored). View with:

```bash
tensorboard --logdir runs
```

Scalars logged every step: `loss/train`, `progress/step`, `progress/epoch`,
`progress/tokens_seen`, `time/step_seconds`. `loss/val` is logged only on validation steps
(every `VAL_EVERY`).

## Environment notes

- Dev machine is a CUDA-less Mac (Apple Silicon, has MPS available via PyTorch but
  `train_gpt_cpu.py` defaults to `DEVICE = "cpu"`; swapping to `"mps"` would likely be faster
  and hasn't been tried/validated yet).
- venv is at `.venv/`, gitignored.
