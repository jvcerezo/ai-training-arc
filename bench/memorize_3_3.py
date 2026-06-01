"""Sprint 3.3 memorization curve - overfit a tiny batch, watch loss -> 0.

Trains on a single fixed byte batch and prints the loss components and retrieval
accuracy as they converge. The point to *see*: within a few dozen steps the
InfoNCE loss collapses toward zero and next-concept accuracy hits 100% — the
whole stack has the capacity to memorize.

Run with the 3.12 ROCm venv interpreter:  python bench/memorize_3_3.py
This is a profiling/demo script, NOT a pytest gate.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from thcm.models.encoder import ByteEncoder  # noqa: E402
from thcm.models.patcher import DynamicEntropyPatcher  # noqa: E402
from thcm.models.transformer import ConceptDecoder  # noqa: E402
from thcm.training.losses import THCMLoss  # noqa: E402
from thcm.training.trainer import THCMTrainer, TrainConfig  # noqa: E402
from thcm.utils.device import require_accelerated  # noqa: E402

D, STEPS = 64, 300


def main() -> None:
    device = require_accelerated()
    torch.manual_seed(0)
    tr = THCMTrainer(
        encoder=ByteEncoder(embed_dim=D, num_blocks=2, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=1.0),
        decoder=ConceptDecoder(embed_dim=D, num_heads=4, num_layers=2),
        loss_fn=THCMLoss(embed_dim=D),
        device=device,
        config=TrainConfig(precision="fp32", lr=3e-3),
    )
    batch = torch.randint(0, 256, (2, 64), dtype=torch.uint8, device=device)
    print(f"device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"{'step':>5} {'total':>8} {'next':>8} {'contrast':>9} {'next_acc':>9} {'self_acc':>9}")

    for i in range(STEPS):
        s = tr.step(batch)
        if i % 25 == 0 or i == STEPS - 1:
            tr.encoder.eval(); tr.decoder.eval(); tr.loss_fn.eval()
            with torch.no_grad():
                traj, wave = tr.encoder(batch)
                packed = tr.patcher(traj, wave)
                out = tr.decoder(packed)
                na, sa = tr.loss_fn.accuracy(out.buffer, packed.vectors, packed.mask)
            print(f"{i:>5} {s.total:>8.4f} {s.next_concept:>8.4f} "
                  f"{s.contrastive:>9.4f} {na:>9.3f} {sa:>9.3f}")

    print("\nLoss collapses toward zero and accuracy saturates at 1.0 -> memorized.")


if __name__ == "__main__":
    main()
