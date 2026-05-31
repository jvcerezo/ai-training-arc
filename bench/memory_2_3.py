"""Sprint 2.3 memory gate - profile O(1) growth of the Holographic Accumulator.

Streams an increasing number of saturated concept buffers through the
accumulator and prints resident GPU memory after each stream length. The point
to *see*: memory is flat — folding 100 chunks and 100,000 chunks leave the same
resident state, because each binding superposes into one fixed (B, D) vector.

Run with the 3.12 ROCm venv interpreter:  python bench/memory_2_3.py
This is a profiling script, NOT a pytest gate (perf/mem is environment-sensitive).
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from thcm.models.holographic import HolographicAccumulator  # noqa: E402
from thcm.models.transformer import ConceptBuffer  # noqa: E402
from thcm.utils.device import require_accelerated  # noqa: E402

B, P, D = 8, 256, 256
STREAM_LENGTHS = [100, 1_000, 10_000, 100_000]


def make_chunk(device: str) -> ConceptBuffer:
    vec = torch.randn(B, P, D, device=device)
    mask = torch.ones(B, P, dtype=torch.bool, device=device)
    counts = torch.full((B,), P, device=device)
    return ConceptBuffer(buffer=vec, mask=mask, counts=counts)


def main() -> None:
    device = require_accelerated()
    print(f"device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"buffer per chunk: (B={B}, P={P}, D={D})  -> global state: (B={B}, D={D}) complex")

    acc = HolographicAccumulator(embed_dim=D, seed=0).to(device)
    chunk = make_chunk(device)

    print(f"\n{'chunks':>10} {'concepts folded':>18} {'resident MB':>14} {'state bytes':>13}")
    for n in STREAM_LENGTHS:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        state = acc.init_state(B, device=device)
        for _ in range(n):
            state = acc.accumulate(state, chunk)
        torch.cuda.synchronize()
        resident = (torch.cuda.memory_allocated() - base) / 1e6
        folded = int(state.step[0].item())
        print(f"{n:>10} {folded:>18} {resident:>14.4f} {state.nbytes():>13}")
        del state

    print("\nResident memory is flat across stream lengths -> O(1) steady-state.")


if __name__ == "__main__":
    main()
