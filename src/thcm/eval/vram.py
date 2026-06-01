"""Sprint 4.1 - VRAM auditing under extreme inputs.

The whole point of the architecture is that memory does NOT scale with input
length. A vanilla Transformer pays O(L^2) attention memory, so a long enough
document simply will not fit. T-HCM bounds it two ways:

  * the Concept Decoder attends over at most CONTEXT_CAP patches (a fixed
    window), so its attention cost is constant, not a function of total length;
  * the Holographic Accumulator folds each finished window into one O(1) global
    state vector, so nothing accumulates across windows.

This module provides the streaming runner that wires those two together for
arbitrarily long byte streams, plus a peak-VRAM measurement harness. The audit
demonstrates the claim empirically: peak `torch.cuda.max_memory_allocated` is
flat as the stream grows by orders of magnitude.

Streaming step (per window, no_grad, working tensors discarded):
    bytes (B, Lw) uint8
      -> encoder -> trajectory, waveform
      -> DEP     -> PatchedBatch
      -> decoder -> ConceptBuffer (<= CONTEXT_CAP patches)
      -> accumulator.accumulate(state, buffer)   # O(1) fold
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thcm.models.encoder import ByteEncoder
from thcm.models.holographic import HolographicAccumulator, HoloState
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder


@dataclass
class VramReport:
    """Peak device memory for one audited workload size."""

    label: str
    units: int          # the scaled quantity (e.g. windows streamed, or patches)
    peak_bytes: int     # torch.cuda.max_memory_allocated during the workload


@torch.no_grad()
def stream_step(encoder: ByteEncoder, patcher: DynamicEntropyPatcher,
                decoder: ConceptDecoder, accumulator: HolographicAccumulator,
                state: HoloState, byte_window: torch.Tensor) -> HoloState:
    """Process one byte window and fold it into the running global state."""
    trajectory, waveform = encoder(byte_window)
    packed = patcher(trajectory, waveform)
    buffer = decoder(packed)
    return accumulator.accumulate(state, buffer)


@torch.no_grad()
def stream(encoder: ByteEncoder, patcher: DynamicEntropyPatcher,
           decoder: ConceptDecoder, accumulator: HolographicAccumulator,
           byte_window: torch.Tensor, n_windows: int) -> HoloState:
    """Stream `n_windows` byte windows through the pipeline into one O(1) state."""
    batch = byte_window.shape[0]
    state = accumulator.init_state(batch, device=byte_window.device)
    for _ in range(n_windows):
        state = stream_step(encoder, patcher, decoder, accumulator, state, byte_window)
    return state


def peak_memory(fn, device_str: str) -> int:
    """Run `fn` and return peak bytes allocated on the device during it.

    Returns -1 on a non-accelerated device, where the peak counter is undefined.
    """
    if "cuda" not in device_str:
        return -1
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def audit_streaming(encoder, patcher, decoder, accumulator, byte_window,
                    window_counts: list[int], device_str: str) -> list[VramReport]:
    """Peak VRAM for streaming each of `window_counts` window totals (should be flat)."""
    reports: list[VramReport] = []
    for n in window_counts:
        peak = peak_memory(
            lambda n=n: stream(encoder, patcher, decoder, accumulator, byte_window, n),
            device_str,
        )
        reports.append(VramReport(label="stream", units=n, peak_bytes=peak))
    return reports


@torch.no_grad()
def audit_full_attention(decoder: ConceptDecoder, batch: int, embed_dim: int,
                         patch_counts: list[int], device_str: str) -> list[VramReport]:
    """Peak VRAM for a single decoder pass over a growing patch count (the
    quadratic baseline: attention memory scales with the window, not folded)."""
    from thcm.models.patcher import PatchedBatch

    reports: list[VramReport] = []
    for p in patch_counts:
        def run(p=p):
            vectors = torch.randn(batch, p, embed_dim, device=device_str)
            mask = torch.ones(batch, p, dtype=torch.bool, device=device_str)
            counts = torch.full((batch,), p, device=device_str)
            seg = torch.zeros(batch, 1, dtype=torch.long, device=device_str)
            decoder(PatchedBatch(vectors=vectors, mask=mask, counts=counts, segment_id=seg))

        reports.append(VramReport(label="full_attn", units=p,
                                  peak_bytes=peak_memory(run, device_str)))
    return reports
