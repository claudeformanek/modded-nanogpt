"""
Minimal CPU-friendly GPT training script, for local debugging on machines with no CUDA GPU
(e.g. a Mac). Not fast -- just meant to run anywhere and be easy to read end to end.

What's reused from train_gpt.py, essentially as-is:
  - The data pipeline: _load_data_shard / Shard / distributed_data_generator. Reads the same
    fineweb .bin shards, does the same BOS-aligned document packing. Two changes only:
      * dropped `pin_memory=True` -- on Apple Silicon this silently allocates on the MPS
        device instead of erroring, which then breaks the `.numpy()` call that reads the file
        into the tensor. It's a CUDA-transfer-speed optimization anyway, irrelevant on CPU.
      * dropped bigram-hash computation from the generator's yield -- it fed the bigram
        embedding table, which this model doesn't have (see below).
  - The evaluation loop shape: periodically build a fresh val loader, average loss with
    model.eval() + no_grad over a handful of steps, print it, resume training.

What's NOT reused -- the actual model and optimizer. train_gpt.py's GPT is a heavily
CUDA/Triton-specific speedrun architecture (custom FP8 kernels, Flash-Attention-3, a
bigram-hash embedding table with sparse cross-rank gradient comms, MuDD gating, paired-head
attention, a "DC" attention correction term, YaRN window-size scheduling, a multi-stage
batch/seq-len training schedule, and the Muon/NorMuon optimizer with a custom Newton-Schulz
orthogonalization step) -- none of which has a CPU fallback and all of which exists purely
for training speed on 8xH100, not for being understandable. Below is instead a standard
decoder-only transformer (learned position embeddings, pre-LN blocks, causal self-attention
via torch's built-in scaled_dot_product_attention, GELU MLP) trained with plain AdamW.

Trade-offs from picking "standard and simple" over "faithful to the speedrun":
  - Attention is plain causal (is_causal=True) with no document-boundary masking, so on
    steps where a packed sequence contains multiple short documents, tokens can attend across
    document boundaries. Harmless for debugging; just be aware the loss isn't directly
    comparable to a real run's.
  - Position ids run continuously across a packed sequence rather than resetting per document.

Run with: python train_gpt_cpu.py  (no torchrun needed -- this is single-process only)
"""
import glob
import math
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cpu"
torch.manual_seed(0)

# -----------------------------------------------------------------------------
# Config -- deliberately small, so a full run finishes in well under a minute on a laptop CPU.

TRAIN_FILES = "data/fineweb10B/fineweb_train_*.bin"
VAL_FILES = "data/fineweb10B/fineweb_val_*.bin"
LOG_DIR = "runs/cpu_debug_" + time.strftime("%Y%m%d_%H%M%S")  # `tensorboard --logdir runs` to view

SEQ_LEN = 512       # tokens per sequence; also caps document length when packing
BATCH_SIZE = 8      # sequences per training step
NUM_STEPS = 200000
LOG_EVERY = 10
VAL_EVERY = 50
VAL_STEPS = 5       # how many val batches to average per evaluation

VOCAB_SIZE = 50257  # gpt2 tokenizer, matches the tokenization already baked into the .bin files
MODEL_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6

LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)

# -----------------------------------------------------------------------------
# Data pipeline (copied from train_gpt.py -- see module docstring for the two changes made)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def _shard_num_tokens(file: Path) -> int:
    """Reads just the shard header (not the token body) to get its token count -- used to size
    an epoch, without loading every training shard into memory up front."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    return int(header[2])


def count_epoch_tokens(filename_pattern: str) -> int:
    files = sorted(glob.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    return sum(_shard_num_tokens(Path(f)) for f in files)


BOS_ID = 50256
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 49152: 128}


def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n


class Shard:
    """Tracks BOS-token positions in a token shard so batches can be packed as whole documents."""

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens
        self.size = tokens.numel()
        self.i = 0

        # Partial index now, full index async (scanning a whole shard for BOS tokens is slow)
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).numpy()
        self._full_idx = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).numpy()
        self._ready.set()

    def _maybe_switch(self):
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens: int, max_seq_len: int):
        """Pack whole documents (each truncated to max_seq_len) until `num_tokens` are covered."""
        self._maybe_switch()
        n = len(self.bos_idx)
        starts, ends = [], []
        idx = self.i
        cur_len = 0
        while cur_len <= num_tokens:
            if idx >= n:
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            cur = self.bos_idx[idx]
            starts.append(cur)
            idx += 1
            end = min(self.bos_idx[idx] if idx < n else self.size,
                      cur + max_seq_len,
                      cur + num_tokens - cur_len + 1)
            ends.append(end)
            cur_len += end - cur
        assert cur_len == num_tokens + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path):
        """Returns a getter function for async shard loading."""
        result = {}
        ready = threading.Event()

        def load():
            result["shard"] = Shard(_load_data_shard(file))
            ready.set()

        threading.Thread(target=load).start()

        def get():
            ready.wait()
            return result["shard"]

        return get


def data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, align_to_bos: bool = True):
    """Yields (inputs, targets) 1D token tensors of length `num_tokens`, packed from whole
    documents (align_to_bos=True) or a plain contiguous slice (align_to_bos=False, used for
    validation, where document packing isn't needed). Call get_batch() to stack several of
    these into a (batch_size, num_tokens) batch."""
    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens)
        next_shard_getter = Shard.load_async(next(file_iter)) if len(files) > 1 else None
    else:
        pos = 0

    while True:
        if align_to_bos:
            try:
                starts, ends = shard.next_batch(num_tokens, max_seq_len)
            except StopIteration:
                if next_shard_getter is None:
                    raise StopIteration("Ran out of shards.")
                shard = next_shard_getter()
                try:
                    next_shard_getter = Shard.load_async(next(file_iter))
                except StopIteration:
                    next_shard_getter = None
                continue
            buf = torch.cat([tokens[i:j] for i, j in zip(starts, ends)])
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            buf = tokens[pos: pos + num_tokens + 1]
            pos += num_tokens

        inputs = buf[:-1].to(dtype=torch.int64, device=DEVICE)
        targets = buf[1:].to(dtype=torch.int64, device=DEVICE)
        yield inputs, targets


def get_batch(loader, batch_size: int):
    """Pulls `batch_size` independently-packed sequences from the generator and stacks them
    into a (batch_size, seq_len) batch -- each call to `loader` always returns exactly
    `num_tokens` tokens, so the lengths always match."""
    inputs, targets = zip(*(next(loader) for _ in range(batch_size)))
    return torch.stack(inputs), torch.stack(targets)


# -----------------------------------------------------------------------------
# Model: a standard decoder-only transformer (see module docstring for what this drops
# relative to the real speedrun architecture)

class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.fc = nn.Linear(model_dim, 4 * model_dim, bias=False)
        self.proj = nn.Linear(4 * model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_embed = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)
        self.blocks = nn.ModuleList(Block(model_dim, num_heads) for _ in range(num_layers))
        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Training loop

def main():
    model = GPT(VOCAB_SIZE, MODEL_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)

    train_loader = data_generator(TRAIN_FILES, SEQ_LEN, SEQ_LEN, align_to_bos=True)

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    epoch_tokens = count_epoch_tokens(TRAIN_FILES)  # size of one pass over the training shards

    writer = SummaryWriter(LOG_DIR)
    print(f"tensorboard logdir: {LOG_DIR}  (run `tensorboard --logdir runs` to view)")

    t0 = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        step_t0 = time.perf_counter()
        inputs, targets = get_batch(train_loader, BATCH_SIZE)
        model.train()
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_time = time.perf_counter() - step_t0

        tokens_seen = step * tokens_per_step
        epoch = tokens_seen / epoch_tokens

        writer.add_scalar("loss/train", loss.item(), step)
        writer.add_scalar("progress/step", step, step)
        writer.add_scalar("progress/epoch", epoch, step)
        writer.add_scalar("progress/tokens_seen", tokens_seen, step)
        writer.add_scalar("time/step_seconds", step_time, step)

        if step == 1 or step % LOG_EVERY == 0:
            print(f"step {step}/{NUM_STEPS}  epoch {epoch:.2f}  train_loss {loss.item():.4f}  "
                  f"tokens {tokens_seen}  step_time {step_time*1000:.0f}ms  elapsed {time.perf_counter() - t0:.1f}s")

        if step % VAL_EVERY == 0 or step == NUM_STEPS:
            model.eval()
            val_loader = data_generator(VAL_FILES, SEQ_LEN, SEQ_LEN, align_to_bos=False)
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(VAL_STEPS):
                    v_inputs, v_targets = get_batch(val_loader, BATCH_SIZE)
                    _, v_loss = model(v_inputs, v_targets)
                    val_loss += v_loss.item()
            val_loss /= VAL_STEPS
            writer.add_scalar("loss/val", val_loss, step)
            print(f"step {step}/{NUM_STEPS}  val_loss {val_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
