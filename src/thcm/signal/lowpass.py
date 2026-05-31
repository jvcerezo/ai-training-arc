"""Sprint 1.3 - waveform smoothing low-pass filter (typo resilience).

A minor spelling anomaly creates a sharp, localized spike in the transition
waveform. If DEP sliced on the raw waveform, that spike could split a
consolidated semantic patch. A low-pass filter flattens such spikes so patch
boundaries stay stable under typos.

Both filters operate on a (B, L) waveform and preserve length via reflect
padding (so boundary positions are not biased toward zero).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _depthwise_smooth(waveform: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 1D kernel along L with reflect padding. (B, L), (K,) -> (B, L)."""
    assert waveform.dim() == 2, f"expected (B, L), got {tuple(waveform.shape)}"
    k = kernel.shape[0]
    assert k % 2 == 1, "odd kernel keeps output length-aligned"
    pad = k // 2
    x = waveform.unsqueeze(1)                                  # (B, 1, L)
    x = F.pad(x, (pad, pad), mode="reflect")
    weight = kernel.to(x.dtype).view(1, 1, k)
    out = F.conv1d(x, weight)                                  # (B, 1, L)
    out = out.squeeze(1)
    assert out.shape == waveform.shape, (out.shape, waveform.shape)
    return out


def gaussian_kernel(sigma: float, radius: int | None = None) -> torch.Tensor:
    """Normalized 1D Gaussian kernel. radius defaults to ceil(3*sigma)."""
    assert sigma > 0
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(xs**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_lowpass(waveform: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian-blur low-pass filter over the entropy waveform."""
    kernel = gaussian_kernel(sigma).to(waveform.device)
    return _depthwise_smooth(waveform, kernel)


def moving_average(waveform: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Simple moving-average (boxcar) low-pass filter."""
    assert window % 2 == 1, "odd window keeps the filter symmetric"
    kernel = torch.full((window,), 1.0 / window, device=waveform.device)
    return _depthwise_smooth(waveform, kernel)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity of two (B, L) waveforms -> (B,)."""
    return F.cosine_similarity(a, b, dim=-1)
