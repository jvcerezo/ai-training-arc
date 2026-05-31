"""Sprint 3.2 training-throughput profile (ROCm mixed precision).

Runs a short training burst through the full pipeline and reports steps/sec,
byte throughput, the loss trajectory, and the live GradScaler scale for each
precision mode. Useful for confirming the fp16 scaler stabilizes and bf16 runs
scaler-free on the RX 9060 XT.

Run with the 3.12 ROCm venv interpreter:  python bench/train_3_2.py
This is a profiling script, NOT a pytest gate (perf is environment-sensitive).
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "src")

from thcm.models.encoder import ByteEncoder  # noqa: E402
from thcm.models.patcher import DynamicEntropyPatcher  # noqa: E402
from thcm.models.transformer import ConceptDecoder  # noqa: E402
from thcm.training.losses import THCMLoss  # noqa: E402
from thcm.training.trainer import THCMTrainer, TrainConfig  # noqa: E402
from thcm.utils.device import require_accelerated  # noqa: E402

B, L, D = 16, 2048, 256
STEPS = 40


def build(device: str, precision: str) -> THCMTrainer:
    return THCMTrainer(
        encoder=ByteEncoder(embed_dim=D, num_blocks=4, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=1.0),
        decoder=ConceptDecoder(embed_dim=D, num_heads=8, num_layers=4),
        loss_fn=THCMLoss(embed_dim=D),
        device=device,
        config=TrainConfig(precision=precision, lr=3e-4),
    )


def run(device: str, precision: str) -> None:
    tr = build(device, precision)
    batch = torch.randint(0, 256, (B, L), dtype=torch.uint8, device=device)

    for _ in range(3):                       # warmup (compile/allocs)
        tr.step(batch)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    first = last = scale = 0.0
    for i in range(STEPS):
        s = tr.step(batch)
        if i == 0:
            first = s.total
        last, scale = s.total, s.scale
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    sps = STEPS / dt
    bytes_per_s = sps * B * L
    print(f"[{precision:>4}]  {sps:6.2f} steps/s  {bytes_per_s/1e6:7.2f} MB/s  "
          f"loss {first:5.2f} -> {last:5.2f}  scale={scale:g}")


def main() -> None:
    device = require_accelerated()
    print(f"device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"config: B={B} L={L} D={D}, {STEPS} steps/mode\n")
    for precision in ("fp32", "bf16", "fp16"):
        run(device, precision)


if __name__ == "__main__":
    main()
