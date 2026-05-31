"""Test Gate 2.3 - Holographic Accumulator (HRR) correctness + O(1) memory.

Verifies the FFT circular-convolution primitives against an independent direct
reference, the unitary-key property that makes long streams stable, exact
single-item round-trip plus above-chance retrieval from a superposition, and the
headline claim: state size and resident memory do NOT grow with stream length.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.models.holographic import (
    HolographicAccumulator,
    HoloState,
    circular_conv,
    circular_corr,
    unitary_vector,
)
from thcm.models.transformer import ConceptBuffer
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def report():
    return preflight()


@pytest.fixture(scope="module")
def device(report) -> str:
    return report.device_str


def _buffer(b: int, p: int, d: int, counts: list[int], device: str) -> ConceptBuffer:
    vec = torch.randn(b, p, d, device=device)
    counts_t = torch.tensor(counts, device=device)
    mask = torch.arange(p, device=device).unsqueeze(0) < counts_t.unsqueeze(1)
    return ConceptBuffer(buffer=vec * mask.unsqueeze(-1), mask=mask, counts=counts_t)


def _ref_circular_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Direct O(D^2) circular convolution: c[k] = sum_j a[j] b[(k-j) mod D]."""
    d = a.shape[-1]
    out = torch.zeros_like(a)
    for k in range(d):
        for j in range(d):
            out[..., k] += a[..., j] * b[..., (k - j) % d]
    return out


def test_circular_conv_matches_direct_reference(device: str) -> None:
    torch.manual_seed(0)
    a = torch.randn(2, 8, device=device)
    b = torch.randn(2, 8, device=device)
    assert torch.allclose(circular_conv(a, b), _ref_circular_conv(a, b), atol=1e-5)


def test_base_is_unitary(device: str) -> None:
    vec = unitary_vector(64, device=device)
    spectrum_mag = torch.fft.fft(vec, dim=-1).abs()
    assert torch.allclose(spectrum_mag, torch.ones_like(spectrum_mag), atol=1e-4)
    # Unitary keys give an EXACT single-item round-trip: corr(k, k(x)v) == v.
    torch.manual_seed(1)
    v = torch.randn(64, device=device)
    bound = circular_conv(vec, v)
    recovered = circular_corr(vec, bound)
    assert torch.allclose(recovered, v, atol=1e-4)


def test_accumulate_is_order_aware_and_retrievable(device: str) -> None:
    """A few concepts superposed; unbinding each position recovers the right one."""
    torch.manual_seed(2)
    d, n = 256, 4
    acc = HolographicAccumulator(embed_dim=d, seed=7).to(device)
    concepts = torch.randn(1, n, d, device=device)
    buf = ConceptBuffer(
        buffer=concepts,
        mask=torch.ones(1, n, dtype=torch.bool, device=device),
        counts=torch.tensor([n], device=device),
    )
    state = acc.accumulate(acc.init_state(1, device=device), buf)

    # Each retrieved slot should be most similar to the TRUE concept at that slot.
    for pos in range(n):
        approx = acc.retrieve(state, pos)[0]                  # (D,)
        sims = torch.cosine_similarity(approx[None, :], concepts[0], dim=-1)  # (n,)
        assert int(sims.argmax()) == pos, f"pos {pos}: argmax {int(sims.argmax())}"


def test_streaming_continuity_matches_single_shot(device: str) -> None:
    """Folding in two chunks == folding the concatenation in one shot."""
    torch.manual_seed(3)
    d = 128
    acc = HolographicAccumulator(embed_dim=d, seed=1).to(device)
    full = torch.randn(1, 6, d, device=device)

    one = ConceptBuffer(full, torch.ones(1, 6, dtype=torch.bool, device=device),
                        torch.tensor([6], device=device))
    s_single = acc.accumulate(acc.init_state(1, device=device), one)

    c1 = ConceptBuffer(full[:, :4], torch.ones(1, 4, dtype=torch.bool, device=device),
                       torch.tensor([4], device=device))
    c2 = ConceptBuffer(full[:, 4:], torch.ones(1, 2, dtype=torch.bool, device=device),
                       torch.tensor([2], device=device))
    s_stream = acc.accumulate(acc.accumulate(acc.init_state(1, device=device), c1), c2)

    assert torch.allclose(s_single.freq, s_stream.freq, atol=1e-4)
    assert int(s_stream.step.item()) == 6


def test_o1_memory_growth_across_stream_length(device: str, report) -> None:
    """The core gate: state size is constant and resident memory does NOT grow
    with how many chunks have been streamed."""
    d = 256
    acc = HolographicAccumulator(embed_dim=d, seed=0).to(device)
    chunk = _buffer(8, 32, d, [32, 31, 20, 16, 8, 4, 2, 1], device)

    def stream(n_chunks: int) -> HoloState:
        st = acc.init_state(8, device=device)
        for _ in range(n_chunks):
            st = acc.accumulate(st, chunk)
        return st

    short = stream(64)
    long = stream(8192)
    # Structural O(1): identical state shape and byte size regardless of length.
    assert short.freq.shape == long.freq.shape == (8, d)
    assert short.nbytes() == long.nbytes()
    assert int(long.step[0].item()) == 8192 * 32   # positions kept advancing

    # Measured O(1) on the GPU: resident memory after a long stream is no larger
    # than after a short one (per-chunk working tensors are freed each step).
    if report.accelerated and report.device_str == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base_mem = torch.cuda.memory_allocated()
        s1 = stream(64)
        torch.cuda.synchronize()
        mem_after_short = torch.cuda.memory_allocated() - base_mem
        del s1
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base_mem2 = torch.cuda.memory_allocated()
        s2 = stream(8192)
        torch.cuda.synchronize()
        mem_after_long = torch.cuda.memory_allocated() - base_mem2
        del s2
        # 128x more chunks must not cost meaningfully more resident memory.
        assert mem_after_long <= mem_after_short + 64 * 1024, (
            f"memory grew with stream length: {mem_after_short} -> {mem_after_long} bytes"
        )
