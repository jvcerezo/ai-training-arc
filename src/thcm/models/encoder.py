"""Sprint 1.2 - 1D CNN parallel byte encoder & continuous trajectory mapping.

The encoder maps a raw byte batch (B, L) onto a continuous latent trajectory
(B, L, D) by convolving over the one-hot byte channels. The convolution sweeps
the whole sequence in parallel on the GPU (no character-by-character loop),
which is the "parallel ingestion" functional requirement.

It also emits a continuous transition *waveform* (B, L): a per-position scalar
measuring how fast the latent trajectory is moving, w_t = ||z_t - z_{t-1}||_2.
This is the entropy/boundary precursor that Sprint 1.3 smooths and Phase-2 DEP
slices on.

Shape contract (boundaries 5 -> 7):
    (B, L) uint8
      -> one-hot         (B, L, 256) float32
      -> channel-first   (B, 256, L)
      -> Conv1d stack     (B, D, L)        ('same' padding preserves L)
      -> channel-last    (B, L, D)         continuous trajectory
      -> transition norm  (B, L)           waveform

Causal mode (Sprint 5.3). The default encoder is bidirectional: symmetric 'same'
padding lets z_t see ~kernel//2 future bytes per layer, and GroupNorm(1, D)
normalizes over the whole sequence. Fine for representation (Phases 1-4), fatal
for a generative model — the concept memory would leak the bytes being
predicted. `causal=True` left-pads every conv (z_t depends only on bytes <= t)
and swaps GroupNorm for a per-position LayerNorm over channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from thcm.config import VOCAB_SIZE
from thcm.data.dataloader import to_onehot


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over D at each position of a (B, D, L) tensor — no cross-time stats."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _ConvBlock(nn.Module):
    """Residual 1D conv block, length-preserving ('same' or causal left padding)."""

    def __init__(self, dim: int, kernel_size: int, causal: bool = False) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "odd kernel keeps 'same' length symmetric"
        self.causal_pad = kernel_size - 1 if causal else 0
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=0 if causal else kernel_size // 2)
        # GroupNorm: batch-size-agnostic; deterministic. Causal mode can't pool over L.
        self.norm = _ChannelLayerNorm(dim) if causal else nn.GroupNorm(1, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D, L) -> (B, D, L)
        residual = x
        x = self.conv(F.pad(x, (self.causal_pad, 0)))
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class ByteEncoder(nn.Module):
    """Parallel 1D-CNN byte encoder producing a continuous latent trajectory."""

    def __init__(
        self,
        embed_dim: int = 256,        # D - latent width (NOT the vocab; see Invariant A)
        num_blocks: int = 4,
        kernel_size: int = 5,
        causal: bool = False,        # True: z_t sees bytes <= t only (generative use)
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        # kernel_size=... first conv mixes local byte context: this is the
        # learned projection W in z_t = sum_j W_j e_{x_{t+j}} + c.
        self.stem_pad = kernel_size - 1 if causal else 0
        self.stem = nn.Conv1d(VOCAB_SIZE, embed_dim, kernel_size,
                              padding=0 if causal else kernel_size // 2)
        self.blocks = nn.ModuleList(
            _ConvBlock(embed_dim, kernel_size, causal) for _ in range(num_blocks)
        )

    def forward(self, byte_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L) uint8 -> (trajectory (B, L, D), waveform (B, L))."""
        assert byte_batch.dim() == 2, f"expected (B, L), got {tuple(byte_batch.shape)}"
        b, length = byte_batch.shape

        onehot = to_onehot(byte_batch)                 # (B, L, 256)
        x = onehot.transpose(1, 2)                     # (B, 256, L) channel-first
        assert x.shape == (b, VOCAB_SIZE, length), x.shape

        x = self.stem(F.pad(x, (self.stem_pad, 0)))    # (B, D, L)
        assert x.shape == (b, self.embed_dim, length), x.shape
        for block in self.blocks:
            x = block(x)
            assert x.shape == (b, self.embed_dim, length), x.shape

        trajectory = x.transpose(1, 2)                 # (B, L, D)
        assert trajectory.shape == (b, length, self.embed_dim), trajectory.shape

        waveform = self._transition_waveform(trajectory)  # (B, L)
        assert waveform.shape == (b, length), waveform.shape
        return trajectory, waveform

    @staticmethod
    def _transition_waveform(trajectory: torch.Tensor) -> torch.Tensor:
        """w_t = ||z_t - z_{t-1}||_2, with w_0 = 0. Continuous transition signal."""
        delta = trajectory[:, 1:, :] - trajectory[:, :-1, :]  # (B, L-1, D)
        mag = torch.linalg.vector_norm(delta, dim=-1)         # (B, L-1)
        zero = torch.zeros(mag.shape[0], 1, dtype=mag.dtype, device=mag.device)
        return torch.cat([zero, mag], dim=1)                  # (B, L)
