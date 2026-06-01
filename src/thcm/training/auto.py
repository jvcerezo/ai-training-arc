"""Autonomous, unsupervised training loop for the T-HCM engine.

Trains the full stack on a raw byte corpus with no human in the loop, and — the
whole point — keeps a held-out validation signal so "is it still improving?" is
answered on generalization, not on the training batch it just saw. The control
policy:

  * Train/val split. The tail `val_fraction` of the corpus is held out (with a
    one-window gap so train and val never overlap). Validation loss + next-concept
    accuracy are measured every `eval_interval` steps over `eval_batches` batches.

  * Keep-improving / best checkpoint. When val loss improves by at least
    `min_delta`, the model is the new best and `best.pt` is saved. When it stalls
    for `patience` consecutive evals, the learning rate is decayed by `lr_decay`
    (ReduceLROnPlateau). When the LR falls below `min_lr` while still stalled, the
    run has converged and stops — it will not spin forever pretending to improve.

  * Divergence recovery. If a step produces a non-finite loss (fp16 overflow, a
    bad batch), the best checkpoint is reloaded and the LR is decayed, so an
    unattended run heals instead of corrupting itself.

  * Crash resume. `last.pt` is written every `ckpt_interval` steps; `--resume`
    restores model + optimizer + control state and continues, so the loop
    survives reboots and pre-emptions.

Run (detached, unattended) with the 3.12 ROCm venv interpreter:

    .\\.venv\\Scripts\\python.exe -m thcm.training.auto --corpus enwik8 --resume

Progress is printed and appended to `<ckpt-dir>/training.log`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Subset

from thcm.config import ModelDims
from thcm.data.dataloader import ByteWindowDataset, CudaPrefetcher, collate_bytes
from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.losses import THCMLoss
from thcm.training.trainer import THCMTrainer, TrainConfig
from thcm.utils.device import preflight


@dataclass
class AutoConfig:
    """Control policy for the autonomous loop."""

    max_steps: int = 200_000
    eval_interval: int = 500
    eval_batches: int = 20
    ckpt_interval: int = 1_000
    val_fraction: float = 0.05
    patience: int = 5            # evals without improvement before an LR decay
    lr_decay: float = 0.5
    min_lr: float = 1e-6
    min_delta: float = 1e-3      # smallest val-loss drop that counts as improvement
    workers: int = 4
    ckpt_dir: str = "checkpoints"


@dataclass
class TrainingSummary:
    """What the run achieved — returned so callers/tests can assert on it."""

    steps: int
    best_val: float
    converged: bool
    val_history: list[tuple[int, float, float]] = field(default_factory=list)  # (step, loss, acc)


def _split_datasets(corpus: str, seq_len: int, val_fraction: float):
    """Disjoint train / val views over one memmapped corpus (val is the tail)."""
    full = ByteWindowDataset(corpus, seq_len)
    n = len(full)                                   # number of valid window starts
    n_val = max(1, int(n * val_fraction))
    train_end = n - n_val - seq_len                 # gap so windows never overlap
    assert train_end > 0, "corpus too small for the requested split"
    # range objects (not lists) keep this O(1) even for a 100M-offset corpus.
    return Subset(full, range(0, train_end)), Subset(full, range(n - n_val, n))


def _prefetcher(dataset, batch_size: int, device: str, workers: int) -> CudaPrefetcher:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=(device != "cpu"), persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_bytes, drop_last=True,
    )
    return CudaPrefetcher(loader, device)


def _set_lr(trainer: THCMTrainer, lr: float) -> None:
    for group in trainer.opt.param_groups:
        group["lr"] = lr


def _get_lr(trainer: THCMTrainer) -> float:
    return float(trainer.opt.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(trainer: THCMTrainer, val_ds, batch_size: int, device: str,
             workers: int, n_batches: int) -> tuple[float, float]:
    """Mean val loss + next-concept accuracy over `n_batches` held-out batches."""
    trainer.encoder.eval(); trainer.decoder.eval(); trainer.loss_fn.eval()
    losses, accs = [], []
    for i, batch in enumerate(_prefetcher(val_ds, batch_size, device, workers)):
        if i >= n_batches:
            break
        traj, wave = trainer.encoder(batch)
        packed = trainer.patcher(traj, wave)
        out = trainer.decoder(packed)
        h, z, m = out.buffer.float(), packed.vectors.float(), packed.mask
        losses.append(float(trainer.loss_fn(h, z, m).total))
        accs.append(trainer.loss_fn.accuracy(h, z, m)[0])
    trainer.encoder.train(); trainer.decoder.train(); trainer.loss_fn.train()
    n = max(1, len(losses))
    return sum(losses) / n, sum(accs) / n


def _checkpoint(trainer: THCMTrainer, step: int, best_val: float,
                patience_ctr: int, lr: float) -> dict:
    return {
        "step": step, "best_val": best_val, "patience_ctr": patience_ctr, "lr": lr,
        "encoder": trainer.encoder.state_dict(),
        "decoder": trainer.decoder.state_dict(),
        "loss_fn": trainer.loss_fn.state_dict(),
        "optimizer": trainer.opt.state_dict(),
    }


def _load_checkpoint(trainer: THCMTrainer, ckpt: dict) -> None:
    trainer.encoder.load_state_dict(ckpt["encoder"])
    trainer.decoder.load_state_dict(ckpt["decoder"])
    trainer.loss_fn.load_state_dict(ckpt["loss_fn"])
    trainer.opt.load_state_dict(ckpt["optimizer"])


def autotrain(corpus: str, device: str, trainer: THCMTrainer, cfg: AutoConfig, *,
              batch_size: int, seq_len: int, resume: bool, log,
              metrics=lambda rec: None) -> TrainingSummary:
    """Run the autonomous loop and return what it achieved.

    `metrics` is called with a flat dict per eval (and at finish) for the live
    dashboard; defaults to a no-op so the loop is usable without one.
    """
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_path = os.path.join(cfg.ckpt_dir, "best.pt")
    last_path = os.path.join(cfg.ckpt_dir, "last.pt")
    train_ds, val_ds = _split_datasets(corpus, seq_len, cfg.val_fraction)

    step, best_val, patience_ctr = 0, float("inf"), 0
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        _load_checkpoint(trainer, ck)
        step, best_val, patience_ctr = ck["step"], ck["best_val"], ck["patience_ctr"]
        _set_lr(trainer, ck["lr"])
        log(f"resumed from {last_path}: step {step}, best_val {best_val:.4f}, lr {ck['lr']:g}")

    history: list[tuple[int, float, float]] = []
    converged = False
    t_prev, step_prev, last_train = time.perf_counter(), step, float("nan")
    while step < cfg.max_steps and not converged:
        for batch in _prefetcher(train_ds, batch_size, device, cfg.workers):
            stats = trainer.step(batch)
            step += 1

            # --- divergence recovery: heal instead of corrupting the run ---
            if not (stats.total == stats.total) or abs(stats.total) == float("inf"):
                lr = max(cfg.min_lr, _get_lr(trainer) * cfg.lr_decay)
                if os.path.exists(best_path):
                    _load_checkpoint(trainer, torch.load(best_path, map_location=device))
                _set_lr(trainer, lr)
                log(f"step {step}: NON-FINITE loss -> restored best, lr -> {lr:g}")
                metrics({"event": "diverge", "step": step, "lr": lr, "best_val": best_val})
                continue
            last_train = stats.total

            # --- periodic validation: the unsupervised improvement signal ---
            if step % cfg.eval_interval == 0:
                val_loss, val_acc = evaluate(trainer, val_ds, batch_size, device,
                                             cfg.workers, cfg.eval_batches)
                history.append((step, val_loss, val_acc))
                lr = _get_lr(trainer)
                now = time.perf_counter()
                sps = (step - step_prev) / max(1e-6, now - t_prev)
                t_prev, step_prev = now, step
                event = "stall"
                if val_loss < best_val - cfg.min_delta:
                    event = "improve"
                    log(f"step {step}: val {val_loss:.4f} (acc {val_acc:.3f}) "
                        f"improved from {best_val:.4f} -> saving best  [lr {lr:g}]")
                    best_val, patience_ctr = val_loss, 0
                    torch.save(_checkpoint(trainer, step, best_val, patience_ctr, lr), best_path)
                else:
                    patience_ctr += 1
                    log(f"step {step}: val {val_loss:.4f} (acc {val_acc:.3f}) "
                        f"no improvement ({patience_ctr}/{cfg.patience})  [lr {lr:g}]")
                    if patience_ctr >= cfg.patience:
                        new_lr = lr * cfg.lr_decay
                        if new_lr < cfg.min_lr:
                            log(f"step {step}: CONVERGED (lr {new_lr:g} < min {cfg.min_lr:g}) "
                                f"— stopping. best_val {best_val:.4f}")
                            converged = True
                        else:
                            _set_lr(trainer, new_lr)
                            patience_ctr = 0
                            event = "plateau"
                            log(f"step {step}: plateau -> lr {lr:g} -> {new_lr:g}")
                metrics({"event": "converged" if converged else event, "step": step,
                         "train_loss": last_train, "val_loss": val_loss, "val_acc": val_acc,
                         "lr": _get_lr(trainer), "best_val": best_val, "steps_per_sec": sps})
                if converged:
                    break

            if step % cfg.ckpt_interval == 0:
                torch.save(_checkpoint(trainer, step, best_val, patience_ctr,
                                       _get_lr(trainer)), last_path)

            if step >= cfg.max_steps:
                break

    torch.save(_checkpoint(trainer, step, best_val, patience_ctr, _get_lr(trainer)), last_path)
    log(f"finished: {step} steps, best_val {best_val:.4f}, converged={converged}")
    metrics({"event": "finish", "step": step, "best_val": best_val,
             "lr": _get_lr(trainer), "converged": converged})
    return TrainingSummary(steps=step, best_val=best_val, converged=converged,
                           val_history=history)


def _make_logger(ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "training.log")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def _make_metrics(ckpt_dir: str):
    """Append one JSON record per call to <ckpt-dir>/metrics.jsonl for the dashboard."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "metrics.jsonl")

    def emit(record: dict) -> None:
        record = {"t": time.time(), **record}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    return emit


def build_trainer(args: argparse.Namespace, device: str) -> THCMTrainer:
    d = args.embed_dim
    return THCMTrainer(
        encoder=ByteEncoder(embed_dim=d, num_blocks=4, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=args.threshold_k),
        decoder=ConceptDecoder(embed_dim=d, num_heads=args.num_heads, num_layers=args.num_layers),
        loss_fn=THCMLoss(embed_dim=d),
        device=device,
        config=TrainConfig(precision=args.precision, lr=args.lr),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous, unsupervised T-HCM training.")
    p.add_argument("--corpus", required=True, help="path to a raw byte corpus (e.g. enwik8)")
    p.add_argument("--resume", action="store_true", help="resume from <ckpt-dir>/last.pt")
    p.add_argument("--max-steps", type=int, default=AutoConfig.max_steps)
    p.add_argument("--eval-interval", type=int, default=AutoConfig.eval_interval)
    p.add_argument("--eval-batches", type=int, default=AutoConfig.eval_batches)
    p.add_argument("--ckpt-interval", type=int, default=AutoConfig.ckpt_interval)
    p.add_argument("--patience", type=int, default=AutoConfig.patience)
    p.add_argument("--ckpt-dir", default=AutoConfig.ckpt_dir)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--embed-dim", type=int, default=ModelDims().embed_dim)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--threshold-k", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=AutoConfig.workers)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = preflight()
    cfg = AutoConfig(
        max_steps=args.max_steps, eval_interval=args.eval_interval,
        eval_batches=args.eval_batches, ckpt_interval=args.ckpt_interval,
        patience=args.patience, workers=args.workers, ckpt_dir=args.ckpt_dir,
    )
    log = _make_logger(cfg.ckpt_dir)
    metrics = _make_metrics(cfg.ckpt_dir)
    log(f"preflight: backend={report.backend} device={report.device_str} "
        f"accelerated={report.accelerated}")
    if not report.accelerated:
        log("WARNING: no GPU backend resolved — training on CPU will be slow.")

    trainer = build_trainer(args, report.device_str)
    n_params = sum(p.numel() for p in trainer.params)
    log(f"params={n_params/1e6:.2f}M precision={args.precision} "
        f"batch={args.batch_size}x{args.seq_len} max_steps={args.max_steps}")
    autotrain(args.corpus, report.device_str, trainer, cfg,
              batch_size=args.batch_size, seq_len=args.seq_len, resume=args.resume,
              log=log, metrics=metrics)


if __name__ == "__main__":
    main()
