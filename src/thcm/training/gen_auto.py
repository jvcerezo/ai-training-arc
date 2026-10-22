"""Sprint 5.3 - autonomous training for the generative byte LM (GenerativeTHCM).

Same unattended control policy as `thcm.training.auto` (held-out val, best
checkpoint, plateau LR decay, divergence recovery, crash resume, live dashboard),
but the objective is next-byte prediction scored in **bits-per-byte** — the
number that says whether this can become a coding model. The Phase-3 run
optimized next-concept InfoNCE, which never trains the model to emit bytes.

Dashboard mapping: `val_loss` / `train_loss` are bpb, `val_acc` is next-byte
top-1 accuracy.

Run (detached, unattended) with the 3.12 ROCm venv interpreter:

    .\\.venv\\Scripts\\python.exe -m thcm.training.gen_auto --corpus enwik8 --resume
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from thcm.data.fetch import ensure_corpus
from thcm.models.code_model import GenerativeTHCM
from thcm.models.generator import byte_lm_loss
from thcm.training.auto import (AutoConfig, _make_logger, _make_metrics, _prefetcher,
                                autotrain)
from thcm.training.trainer import enable_speed
from thcm.utils.device import preflight


@dataclass
class GenStepStats:
    total: float           # train bits-per-byte (what the loop logs + checks for NaN)


class GenTrainer:
    """One-step-at-a-time wrapper around GenerativeTHCM: AdamW, bf16, warmup, clip."""

    def __init__(self, model: GenerativeTHCM, device: str, *, lr: float = 3e-4,
                 warmup: int = 1_000, precision: str = "bf16", clip: float = 1.0) -> None:
        self.model = model.to(device)
        self.device = device
        self.lr, self.warmup, self.clip = lr, warmup, clip
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        if device == "cpu":
            self.amp_dtype = None
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                     weight_decay=0.01)
        self.steps = 0

    @property
    def params(self):
        return self.model.parameters()

    def _forward_loss(self, batch: torch.Tensor):
        with torch.autocast(device_type="cuda" if self.device != "cpu" else "cpu",
                            dtype=self.amp_dtype or torch.float32,
                            enabled=self.amp_dtype is not None):
            return self.model.loss(batch)

    def step(self, batch: torch.Tensor) -> GenStepStats:
        if self.steps < self.warmup:                       # linear warmup to base lr
            for g in self.opt.param_groups:
                g["lr"] = self.lr * (self.steps + 1) / self.warmup
        loss, bpb = self._forward_loss(batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        self.steps += 1
        return GenStepStats(total=float(bpb))


@torch.no_grad()
def evaluate_gen(trainer: GenTrainer, val_ds, batch_size: int, device: str,
                 workers: int, n_batches: int) -> tuple[float, float]:
    """Mean held-out bits-per-byte + next-byte top-1 accuracy."""
    trainer.model.eval()
    bpbs, accs = [], []
    for i, batch in enumerate(_prefetcher(val_ds, batch_size, device, workers)):
        if i >= n_batches:
            break
        with torch.autocast(device_type="cuda" if device != "cpu" else "cpu",
                            dtype=trainer.amp_dtype or torch.float32,
                            enabled=trainer.amp_dtype is not None):
            logits = trainer.model(batch)
        logits = logits.float()
        _, bpb = byte_lm_loss(logits, batch)
        acc = float((logits[:, :-1].argmax(-1) == batch[:, 1:].long()).float().mean())
        bpbs.append(float(bpb)); accs.append(acc)
    trainer.model.train()
    n = max(1, len(bpbs))
    return sum(bpbs) / n, sum(accs) / n


def _gen_checkpoint(trainer: GenTrainer, step: int, best_val: float,
                    patience_ctr: int, lr: float) -> dict:
    return {
        "step": step, "best_val": best_val, "patience_ctr": patience_ctr, "lr": lr,
        "model": trainer.model.state_dict(), "optimizer": trainer.opt.state_dict(),
        "model_config": trainer.model_config,
    }


def _gen_load(trainer: GenTrainer, ckpt: dict) -> None:
    trainer.model.load_state_dict(ckpt["model"])
    trainer.opt.load_state_dict(ckpt["optimizer"])
    trainer.steps = max(trainer.steps, ckpt["step"])     # no re-warmup on resume


def build_gen_trainer(args: argparse.Namespace, device: str) -> GenTrainer:
    config = dict(embed_dim=args.embed_dim, use_memory=not args.no_memory,
                  byte_layers=args.byte_layers, byte_heads=args.num_heads,
                  concept_layers=args.concept_layers, concept_heads=args.num_heads,
                  encoder_blocks=args.encoder_blocks, threshold_k=args.threshold_k)
    trainer = GenTrainer(GenerativeTHCM(**config), device, lr=args.lr,
                         warmup=args.warmup, precision=args.precision)
    trainer.model_config = config                          # saved so checkpoints self-describe
    return trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous generative (bits-per-byte) training.")
    p.add_argument("--corpus", required=True, help="path to a raw byte corpus (e.g. enwik8)")
    p.add_argument("--resume", action="store_true", help="resume from <ckpt-dir>/last.pt")
    p.add_argument("--max-steps", type=int, default=AutoConfig.max_steps)
    p.add_argument("--eval-interval", type=int, default=AutoConfig.eval_interval)
    p.add_argument("--eval-batches", type=int, default=AutoConfig.eval_batches)
    p.add_argument("--ckpt-interval", type=int, default=AutoConfig.ckpt_interval)
    p.add_argument("--heartbeat", type=int, default=AutoConfig.heartbeat)
    p.add_argument("--patience", type=int, default=AutoConfig.patience)
    p.add_argument("--ckpt-dir", default="checkpoints_gen")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--byte-layers", type=int, default=8)
    p.add_argument("--concept-layers", type=int, default=4)
    p.add_argument("--encoder-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--threshold-k", type=float, default=1.0)
    p.add_argument("--no-memory", action="store_true", help="plain byte-LM baseline")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--workers", type=int, default=AutoConfig.workers)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_speed()
    report = preflight()
    cfg = AutoConfig(
        max_steps=args.max_steps, eval_interval=args.eval_interval,
        eval_batches=args.eval_batches, ckpt_interval=args.ckpt_interval,
        heartbeat=args.heartbeat, patience=args.patience,
        workers=args.workers, ckpt_dir=args.ckpt_dir,
    )
    log = _make_logger(cfg.ckpt_dir)
    metrics = _make_metrics(cfg.ckpt_dir)
    log(f"preflight: backend={report.backend} device={report.device_str} "
        f"accelerated={report.accelerated}")
    if not report.accelerated:
        log("WARNING: no GPU backend resolved — training on CPU will be slow.")

    corpus = ensure_corpus(args.corpus, log=log)
    trainer = build_gen_trainer(args, report.device_str)
    n_params = sum(p.numel() for p in trainer.params)
    log(f"objective=bpb memory={not args.no_memory} params={n_params/1e6:.2f}M "
        f"precision={args.precision} batch={args.batch_size}x{args.seq_len} "
        f"max_steps={args.max_steps}")
    autotrain(corpus, report.device_str, trainer, cfg,
              batch_size=args.batch_size, seq_len=args.seq_len, resume=args.resume,
              log=log, metrics=metrics, evaluate_fn=evaluate_gen,
              checkpoint_fn=_gen_checkpoint, load_fn=_gen_load)


if __name__ == "__main__":
    main()
