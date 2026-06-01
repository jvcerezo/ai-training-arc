"""Sprint 4.1 VRAM audit - flat streaming vs quadratic full attention.

Two curves on the real GPU:

  * STREAMING — the T-HCM path: stream an increasing number of byte windows
    through encoder -> DEP -> decoder -> holographic accumulator. Peak VRAM is
    flat: each window is bounded by CONTEXT_CAP and folds into one O(1) state.

  * FULL ATTENTION — the baseline: a single decoder pass over a growing patch
    count. Peak VRAM climbs ~quadratically (the B*heads*P*P attention scores) —
    the wall T-HCM is built to avoid.

Reports torch.cuda.max_memory_allocated (process-scoped, exact) and, if the tool
is on PATH, a one-shot rocm-smi VRAM line for the device-level view.

Run with the 3.12 ROCm venv interpreter:  python bench/vram_4_1.py
This is a profiling script, NOT a pytest gate (memory is environment-sensitive).
"""

from __future__ import annotations

import subprocess
import sys

import torch

sys.path.insert(0, "src")

from thcm.eval.vram import audit_full_attention, audit_streaming  # noqa: E402
from thcm.models.encoder import ByteEncoder  # noqa: E402
from thcm.models.holographic import HolographicAccumulator  # noqa: E402
from thcm.models.patcher import DynamicEntropyPatcher  # noqa: E402
from thcm.models.transformer import ConceptDecoder  # noqa: E402
from thcm.utils.device import require_accelerated  # noqa: E402

B, LW, D = 4, 1024, 256


def rocm_smi_vram() -> str:
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=10)
        lines = [ln.strip() for ln in out.stdout.splitlines() if "VRAM" in ln.upper()]
        return lines[0] if lines else "(rocm-smi returned no VRAM line)"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "(rocm-smi not available on PATH)"


def main() -> None:
    device = require_accelerated()
    print(f"device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"rocm-smi: {rocm_smi_vram()}\n")

    enc = ByteEncoder(embed_dim=D, num_blocks=4, kernel_size=5).to(device).eval()
    dep = DynamicEntropyPatcher(threshold_k=1.0).to(device)
    dec = ConceptDecoder(embed_dim=D, num_heads=8, num_layers=4).to(device).eval()
    acc = HolographicAccumulator(embed_dim=D, seed=0).to(device)
    window = torch.randint(0, 256, (B, LW), dtype=torch.uint8, device=device)

    print("STREAMING (T-HCM): peak VRAM vs windows streamed")
    print(f"{'windows':>10} {'bytes':>14} {'peak MB':>12}")
    for r in audit_streaming(enc, dep, dec, acc, window, [10, 100, 1000, 10000], device):
        print(f"{r.units:>10} {r.units * B * LW:>14} {r.peak_bytes / 1e6:>12.3f}")

    print("\nFULL ATTENTION (baseline): peak VRAM vs patch count (single pass)")
    print(f"{'patches':>10} {'peak MB':>12}")
    for r in audit_full_attention(dec, B, D, [128, 256, 512, 1024, 2048], device):
        print(f"{r.units:>10} {r.peak_bytes / 1e6:>12.3f}")

    print("\nStreaming peak is flat; full-attention peak grows ~P^2.")


if __name__ == "__main__":
    main()
