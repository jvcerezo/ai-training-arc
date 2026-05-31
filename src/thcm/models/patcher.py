"""Sprint 2.1 - Dynamic Entropy Patching (DEP) slicing engine.

Slices a continuous trajectory (B, L, D) into a *variable* number of semantic
Concept Vectors per sequence, driven by the (smoothed) transition waveform: a
patch boundary opens wherever the waveform spikes past an adaptive per-sequence
threshold. The ragged result is packed into a dense padded batch + mask so it
can flow into the context-capped Transformer (Sprint 2.2).

Pipeline:
    trajectory (B, L, D), waveform (B, L)
      -> adaptive threshold      (B, 1)      mean + k*std, per sequence
      -> boundary markers        (B, L) bool  position 0 always opens a patch
      -> segment ids             (B, L) int   cumsum(boundaries) - 1
      -> scatter mean-pool        (B, P, D)   one Concept Vector per patch
      -> mask + counts           (B, P), (B,) valid-patch bookkeeping

Fully vectorized (no per-sequence Python loop): segmentation is a cumulative
sum, pooling is a batched scatter_add. Runs on CPU or ROCm/CUDA unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PatchedBatch:
    """Dense packing of a ragged set of per-sequence Concept Vectors."""

    vectors: torch.Tensor   # (B, P, D) - pooled concept vectors, zero-padded
    mask: torch.Tensor      # (B, P) bool - True where a patch is real
    counts: torch.Tensor    # (B,) int - number of real patches per sequence
    segment_id: torch.Tensor  # (B, L) int - patch index each byte-position fell in

    def num_patches(self) -> int:
        """Padded patch dimension P."""
        return int(self.vectors.shape[1])


def adaptive_threshold(waveform: torch.Tensor, k: float) -> torch.Tensor:
    """Per-sequence threshold = mean + k * std over the waveform. (B, L) -> (B, 1)."""
    assert waveform.dim() == 2, f"expected (B, L), got {tuple(waveform.shape)}"
    mean = waveform.mean(dim=1, keepdim=True)
    std = waveform.std(dim=1, keepdim=True)
    return mean + k * std


def detect_boundaries(waveform: torch.Tensor, k: float) -> torch.Tensor:
    """Boolean (B, L): True opens a new patch. Position 0 always opens one."""
    thresh = adaptive_threshold(waveform, k)
    boundaries = waveform > thresh                 # (B, L) bool
    boundaries[:, 0] = True                         # every sequence starts a patch
    return boundaries


def pack_patches(trajectory: torch.Tensor, boundaries: torch.Tensor) -> PatchedBatch:
    """Mean-pool the trajectory within each patch and pad into a dense batch."""
    assert trajectory.dim() == 3, f"expected (B, L, D), got {tuple(trajectory.shape)}"
    b, length, d = trajectory.shape
    assert boundaries.shape == (b, length), boundaries.shape

    # Segment id of each position = how many boundaries opened at-or-before it, -1.
    segment_id = torch.cumsum(boundaries.long(), dim=1) - 1   # (B, L), 0-based
    counts = segment_id[:, -1] + 1                            # (B,) patches per row
    p_max = int(counts.max().item())

    # Batched scatter mean-pool: sum trajectory rows sharing a segment id, / size.
    idx = segment_id.unsqueeze(-1).expand(-1, -1, d)         # (B, L, D)
    summed = torch.zeros(b, p_max, d, dtype=trajectory.dtype, device=trajectory.device)
    summed.scatter_add_(1, idx, trajectory)                  # (B, P, D)

    sizes = torch.zeros(b, p_max, dtype=trajectory.dtype, device=trajectory.device)
    ones = torch.ones(b, length, dtype=trajectory.dtype, device=trajectory.device)
    sizes.scatter_add_(1, segment_id, ones)                  # (B, P) patch sizes
    vectors = summed / sizes.clamp(min=1.0).unsqueeze(-1)    # mean; empty slots stay 0

    patch_idx = torch.arange(p_max, device=trajectory.device).unsqueeze(0)  # (1, P)
    mask = patch_idx < counts.unsqueeze(1)                   # (B, P) bool
    vectors = vectors * mask.unsqueeze(-1)                   # zero out padded slots

    return PatchedBatch(vectors=vectors, mask=mask, counts=counts, segment_id=segment_id)


class DynamicEntropyPatcher(nn.Module):
    """DEP: waveform -> adaptive boundaries -> packed variable-length patches.

    `k` sets sensitivity: higher k -> fewer boundaries -> longer patches.
    """

    def __init__(self, threshold_k: float = 1.0) -> None:
        super().__init__()
        self.threshold_k = threshold_k

    def forward(self, trajectory: torch.Tensor, waveform: torch.Tensor) -> PatchedBatch:
        boundaries = detect_boundaries(waveform, self.threshold_k)
        packed = pack_patches(trajectory, boundaries)
        # Structural integrity assertions at the packing boundary.
        b, length, d = trajectory.shape
        assert packed.vectors.shape == (b, packed.num_patches(), d)
        assert packed.mask.shape == (b, packed.num_patches())
        assert packed.counts.shape == (b,)
        assert int(packed.mask.sum().item()) == int(packed.counts.sum().item())
        return packed
