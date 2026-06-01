"""Training entry point — drive the full T-HCM stack over a raw byte corpus.

Wires the Sprint 1.1 byte loader to the Sprint 3.2 trainer: memory-maps a corpus,
streams fixed-length uint8 windows to the GPU, and runs forward/backward/step
through encoder -> DEP -> decoder -> InfoNCE loss. The loader is single-epoch
(the prefetcher exhausts after one pass), so this rebuilds it each epoch until the
step budget is met, logging loss/accuracy and checkpointing periodically.

Run with the 3.12 ROCm venv interpreter (package is installed via `pip install -e`):

    .\\.venv\\Scripts\\python.exe -m thcm.training.run --corpus path\\to\\corpus.txt

Always confirm the preflight reports accelerated:True first (a silent CPU
fallback would train ~2x slower and is a false pass for the GPU path).
"""

from __future__ import annotations

import argparse

import torch

from thcm.config import LoaderConfig, ModelDims
from thcm.data.dataloader import build_loader
from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.losses import THCMLoss
from thcm.training.trainer import THCMTrainer, TrainConfig
from thcm.utils.device import preflight


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Tokenless T-HCM engine.")
    p.add_argument("--corpus", required=True, help="path to a raw byte corpus file")
    p.add_argument("--steps", type=int, default=10_000, help="total optimizer steps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--embed-dim", type=int, default=ModelDims().embed_dim)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--threshold-k", type=float, default=1.0, help="DEP boundary sensitivity")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--ckpt", default="thcm_ckpt.pt", help="checkpoint output path")
    return p.parse_args()


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


def save_checkpoint(trainer: THCMTrainer, step: int, path: str) -> None:
    torch.save(
        {
            "step": step,
            "encoder": trainer.encoder.state_dict(),
            "decoder": trainer.decoder.state_dict(),
            "loss_fn": trainer.loss_fn.state_dict(),
            "optimizer": trainer.opt.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    report = preflight()
    print(f"backend={report.backend} device={report.device_str} "
          f"accelerated={report.accelerated} — {report.detail}")
    if not report.accelerated:
        print("WARNING: no GPU backend resolved; training on CPU will be slow.")

    device = report.device_str
    trainer = build_trainer(args, device)
    cfg = LoaderConfig(corpus_path=args.corpus, seq_len=args.seq_len,
                       batch_size=args.batch_size, device=device)

    n_params = sum(p.numel() for p in trainer.params)
    print(f"params={n_params/1e6:.2f}M  precision={args.precision}  "
          f"batch={args.batch_size}x{args.seq_len}  target_steps={args.steps}")

    step, epoch = 0, 0
    while step < args.steps:
        epoch += 1
        for batch in build_loader(cfg):           # fresh single-epoch iterator
            stats = trainer.step(batch)
            step += 1
            if step % args.log_interval == 0:
                na, sa = trainer.loss_fn.accuracy(
                    *_eval_inputs(trainer, batch))
                print(f"epoch {epoch:>3} step {step:>6}/{args.steps}  "
                      f"loss {stats.total:6.3f}  next {stats.next_concept:6.3f}  "
                      f"con {stats.contrastive:6.3f}  next_acc {na:.3f}  "
                      f"grad {stats.grad_norm:6.2f}  scale {stats.scale:g}")
            if step % args.ckpt_interval == 0:
                save_checkpoint(trainer, step, args.ckpt)
                print(f"  checkpoint -> {args.ckpt} (step {step})")
            if step >= args.steps:
                break

    save_checkpoint(trainer, step, args.ckpt)
    print(f"done: {step} steps over {epoch} epoch(s); final checkpoint -> {args.ckpt}")


@torch.no_grad()
def _eval_inputs(trainer: THCMTrainer, batch: torch.Tensor):
    """Recompute (contextual, concepts, mask) for an accuracy read-out."""
    trainer.encoder.eval(); trainer.decoder.eval(); trainer.loss_fn.eval()
    traj, wave = trainer.encoder(batch)
    packed = trainer.patcher(traj, wave)
    out = trainer.decoder(packed)
    trainer.encoder.train(); trainer.decoder.train(); trainer.loss_fn.train()
    return out.buffer.float(), packed.vectors.float(), packed.mask


if __name__ == "__main__":
    main()
