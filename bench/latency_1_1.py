"""Sprint 1.1 latency gate - validate "no pipeline stalling" on real GPU.

Demonstrates the two design claims:
  1. Invariant B payoff: shipping (B, L) uint8 across the bus is ~256x cheaper
     than shipping the (B, L, 256) float one-hot. We measure both.
  2. The CudaPrefetcher overlaps H2D transfer with compute, so end-to-end time
     approaches max(transfer, compute) rather than their sum.

Run with the 3.12 ROCm venv interpreter:  python bench/latency_1_1.py
This is a profiling script, NOT a pytest gate (perf is environment-sensitive).
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "src")

from thcm.config import LoaderConfig  # noqa: E402
from thcm.data.dataloader import build_loader, to_onehot  # noqa: E402
from thcm.utils.device import require_accelerated  # noqa: E402

B, L = 32, 2048
ITERS = 50


def _sync() -> None:
    torch.cuda.synchronize()


def bench_transfer(device: str) -> None:
    uint8_cpu = torch.randint(0, 256, (B, L), dtype=torch.uint8).pin_memory()
    float_cpu = torch.rand(B, L, 256, dtype=torch.float32).pin_memory()

    # warmup
    uint8_cpu.to(device, non_blocking=True)
    float_cpu.to(device, non_blocking=True)
    _sync()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        uint8_cpu.to(device, non_blocking=True)
    _sync()
    t_uint8 = (time.perf_counter() - t0) / ITERS * 1e3

    t0 = time.perf_counter()
    for _ in range(ITERS):
        float_cpu.to(device, non_blocking=True)
    _sync()
    t_float = (time.perf_counter() - t0) / ITERS * 1e3

    uint8_mb = uint8_cpu.numel() / 1e6
    float_mb = float_cpu.numel() * 4 / 1e6
    print(f"  uint8 (B,L)      {uint8_mb:6.3f} MB  ->  {t_uint8:.4f} ms/batch")
    print(f"  float (B,L,256)  {float_mb:6.3f} MB  ->  {t_float:.4f} ms/batch")
    print(f"  Invariant B saves ~{t_float / max(t_uint8, 1e-6):.0f}x transfer time")


def bench_prefetch(device: str) -> None:
    # synthetic corpus on disk
    import tempfile

    import numpy as np

    corpus = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    np.random.randint(0, 256, size=4_000_000, dtype=np.uint8).tofile(corpus.name)
    corpus.close()

    cfg = LoaderConfig(corpus_path=corpus.name, seq_len=L, batch_size=B,
                       num_workers=2, device=device)

    def fake_compute(x: torch.Tensor) -> torch.Tensor:
        z = to_onehot(x)                        # (B, L, 256) on-device
        return (z @ z.transpose(1, 2)).sum()    # some real GPU work

    loader = build_loader(cfg)
    # warmup
    for i, batch in enumerate(loader):
        fake_compute(batch)
        if i >= 3:
            break
    _sync()

    loader = build_loader(cfg)
    t0 = time.perf_counter()
    n = 0
    for batch in loader:
        fake_compute(batch)
        n += 1
        if n >= ITERS:
            break
    _sync()
    dt = (time.perf_counter() - t0) / n * 1e3
    print(f"  overlapped load+transfer+compute: {dt:.3f} ms/batch over {n} batches")


def main() -> None:
    device = require_accelerated()
    print(f"device: {device}  ({torch.cuda.get_device_name(0)})")
    print("[transfer cost: uint8 vs float one-hot]")
    bench_transfer(device)
    print("[prefetcher: overlapped pipeline]")
    bench_prefetch(device)


if __name__ == "__main__":
    main()
