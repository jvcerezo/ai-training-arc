"""Test Gate 4.1 - VRAM audit: steady-state memory does not grow with length.

The architecture's reason to exist. Verifies the peak-memory harness actually
measures growth, that streaming an order of magnitude more windows leaves peak
VRAM flat (the O(1) claim), and that the stream still processes every window
(no silent truncation hiding the memory win).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.eval.vram import audit_streaming, peak_memory, stream
from thcm.models.encoder import ByteEncoder
from thcm.models.holographic import HolographicAccumulator
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def report():
    return preflight()


@pytest.fixture(scope="module")
def device(report) -> str:
    return report.device_str


def _stack(device: str, d: int = 64):
    enc = ByteEncoder(embed_dim=d, num_blocks=2, kernel_size=5).to(device).eval()
    dep = DynamicEntropyPatcher(threshold_k=1.0).to(device)
    dec = ConceptDecoder(embed_dim=d, num_heads=4, num_layers=2).to(device).eval()
    acc = HolographicAccumulator(embed_dim=d, seed=0).to(device)
    return enc, dep, dec, acc


def test_peak_memory_detects_growth(device: str, report) -> None:
    if not report.accelerated:
        pytest.skip("peak VRAM counter requires an accelerated backend")

    def alloc(n: int):
        x = torch.empty(n, device=device)
        x.add_(1.0)                       # force the allocation to be live

    big = peak_memory(lambda: alloc(8_000_000), device)
    small = peak_memory(lambda: alloc(1_000), device)
    assert big > small
    assert big - small >= 8_000_000 * 4 * 0.5    # ~32 MB tensor dominates


def test_streaming_processes_every_window(device: str) -> None:
    """Functional: the stream folds in all windows (step advances correctly),
    so the flat-memory result isn't an artifact of dropping work."""
    torch.manual_seed(0)
    enc, dep, dec, acc = _stack(device)
    window = torch.randint(0, 256, (2, 128), dtype=torch.uint8, device=device)

    # Patches produced by one window (deterministic for a fixed window/encoder).
    traj, wave = enc(window)
    per_window = dep(traj, wave).counts.clone()

    state = stream(enc, dep, dec, acc, window, n_windows=40)
    assert torch.equal(state.step, per_window * 40)
    assert state.freq.shape == (2, 64)               # O(1) state, unchanged


def test_streaming_vram_is_flat(device: str, report) -> None:
    if not report.accelerated:
        pytest.skip("VRAM audit requires an accelerated backend")
    torch.manual_seed(1)
    enc, dep, dec, acc = _stack(device)
    window = torch.randint(0, 256, (2, 256), dtype=torch.uint8, device=device)

    reports = audit_streaming(enc, dep, dec, acc, window,
                              window_counts=[50, 2000], device_str=device)
    short, long = reports[0].peak_bytes, reports[1].peak_bytes
    assert short > 0 and long > 0
    # 40x more windows must not grow peak VRAM beyond per-window working set noise.
    assert long <= short * 1.05 + (1 << 20), f"VRAM grew with length: {short} -> {long}"
