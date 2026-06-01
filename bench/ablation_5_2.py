"""Sprint 5.2 kill-gate - does concept memory lower bits-per-byte?

Trains two GenerativeTHCM variants on the SAME train split with identical byte-
decoder config and optimizer — one conditioned on the concept buffer
(use_memory=True), one a plain causal byte LM (use_memory=False) — then reports
held-out bpb for each. If memory doesn't beat the baseline, the concept
architecture doesn't earn its place in a generative model.

Run with the 3.12 ROCm venv interpreter:
    python bench/ablation_5_2.py --corpus enwik8 --steps 800

NOT a pytest gate (it trains; minutes-scale and environment-sensitive).
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "src")

from thcm.data.dataloader import ByteWindowDataset, CudaPrefetcher, collate_bytes  # noqa: E402
from thcm.models.code_model import GenerativeTHCM  # noqa: E402
from thcm.training.trainer import enable_speed  # noqa: E402
from thcm.utils.device import require_accelerated  # noqa: E402


def loaders(corpus, seq_len, batch, device, val_frac=0.05):
    full = ByteWindowDataset(corpus, seq_len)
    n = len(full)
    n_val = max(batch, int(n * val_frac))
    train = Subset(full, range(0, n - n_val - seq_len))
    val = Subset(full, range(n - n_val, n))

    def mk(ds, shuffle):
        dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4,
                        pin_memory=True, collate_fn=collate_bytes, drop_last=True)
        return CudaPrefetcher(dl, device)
    return train, val, mk


@torch.no_grad()
def eval_bpb(model, val_ds, mk, n_batches):
    model.eval()
    tot, k = 0.0, 0
    for i, batch in enumerate(mk(val_ds, True)):
        if i >= n_batches:
            break
        _, bpb = model.loss(batch)
        tot += float(bpb); k += 1
    model.train()
    return tot / max(1, k)


def train_variant(use_memory, args, device, train_ds, val_ds, mk):
    torch.manual_seed(0)
    m = GenerativeTHCM(embed_dim=args.embed_dim, use_memory=use_memory,
                       byte_layers=args.byte_layers, concept_layers=args.concept_layers,
                       encoder_blocks=args.encoder_blocks).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    tag = "memory" if use_memory else "baseline"
    step = 0
    while step < args.steps:
        for batch in mk(train_ds, True):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _ = m.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step >= args.steps:
                bpb = eval_bpb(m, val_ds, mk, args.eval_batches)
                print(f"  [{tag:8s}] step {step:5d}  val bpb {bpb:.4f}")
            if step >= args.steps:
                break
    return eval_bpb(m, val_ds, mk, args.eval_batches), n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="enwik8")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--byte-layers", type=int, default=4)
    p.add_argument("--concept-layers", type=int, default=2)
    p.add_argument("--encoder-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    args = p.parse_args()

    enable_speed()
    device = require_accelerated()
    print(f"device: {torch.cuda.get_device_name(0)}  corpus={args.corpus}")
    print(f"config: D={args.embed_dim} seq={args.seq_len} batch={args.batch_size} "
          f"steps={args.steps}\n")
    train_ds, val_ds, mk = loaders(args.corpus, args.seq_len, args.batch_size, device)

    print("training BASELINE (plain causal byte LM)…")
    base_bpb, base_p = train_variant(False, args, device, train_ds, val_ds, mk)
    print("\ntraining MEMORY (byte LM + concept buffer)…")
    mem_bpb, mem_p = train_variant(True, args, device, train_ds, val_ds, mk)

    print("\n================ KILL-GATE RESULT ================")
    print(f"baseline (no memory) : {base_bpb:.4f} bpb   ({base_p/1e6:.2f}M params)")
    print(f"concept memory       : {mem_bpb:.4f} bpb   ({mem_p/1e6:.2f}M params)")
    delta = base_bpb - mem_bpb
    verdict = ("MEMORY HELPS" if delta > 0.02 else
               "INCONCLUSIVE" if abs(delta) <= 0.02 else "MEMORY HURTS")
    print(f"delta                : {delta:+.4f} bpb  ->  {verdict}")
    print("(preliminary: short run on text; a full verdict needs a longer run on code)")


if __name__ == "__main__":
    main()
