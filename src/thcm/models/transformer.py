"""Sprint 2.2 - Context-capped, causally-masked Concept Decoder (buffer build).

Consumes the dense, padded set of Concept Vectors emitted by the DEP slicing
engine (Sprint 2.1) and contextualizes them with *causal* self-attention. This
is the "local reasoning" stage: a decoder-only Transformer operating not on raw
bytes but on the variable-length semantic patches, under a HARD window cap. Its
output is the contextualized concept *buffer* that Phase 3's Holographic
Accumulator will compress once it saturates.

Decoder-only is realized the canonical PyTorch way: an encoder *layer* stack fed
a causal (lower-triangular) attention mask. No cross-attention, no memory.

Three invariants distinguish this from a vanilla encoder:

  * Causality. Concept i may attend to concepts <= i only. The Gate-2.2 test
    pins this: perturbing patch k leaves every output at positions < k
    bit-for-bit unchanged.

  * Padding invariance. The patcher left-aligns real patches and zero-pads the
    rest. Padded slots must never influence a real patch's output. Padded
    patches are excluded as attention *keys* (key-padding mask) and padded
    *query* rows are re-zeroed so garbage never leaks downstream. Because DEP
    forces position 0 open and patches are left-aligned, the causal + padding
    masks together never fully mask a row -> no softmax-over-empty NaN.

  * Context cap. Attention is capped at CONTEXT_CAP patches; the sinusoidal
    positional table has exactly that many slots. Exceeding it is a loud
    contract violation (ValueError), never a silent truncation — overflow
    routing is the Holographic Accumulator's job (Phase 3).

Shape contract:
    PatchedBatch.vectors (B, P, D), .mask (B, P) bool, P <= CONTEXT_CAP
      -> + sinusoidal positional encoding on real slots   (B, P, D)
      -> TransformerEncoder(causal src_mask, ~mask key-pad) (B, P, D)
      -> re-zero padded query rows                         (B, P, D)
      -> ConceptBuffer(buffer, mask, counts)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from thcm.config import CONTEXT_CAP
from thcm.models.patcher import PatchedBatch


@dataclass
class ConceptBuffer:
    """Contextualized Concept Vectors plus the carried-through validity mask."""

    buffer: torch.Tensor    # (B, P, D) - causally-attended concepts, padded slots zero
    mask: torch.Tensor      # (B, P) bool - True where a patch is real
    counts: torch.Tensor    # (B,) int - real patches per sequence

    def num_patches(self) -> int:
        """Padded patch dimension P."""
        return int(self.buffer.shape[1])


def sinusoidal_encoding(num_pos: int, dim: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Classic (Vaswani) sinusoidal positional table. (num_pos, dim).

    Deterministic and parameter-free, so it never perturbs the padding-invariance
    gate and adds no state to load/save. Even dims use sin, odd dims use cos.
    """
    assert dim % 2 == 0, f"sinusoidal dim must be even, got {dim}"
    position = torch.arange(num_pos, device=device, dtype=dtype).unsqueeze(1)  # (P, 1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )  # (D/2,)
    pe = torch.zeros(num_pos, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class ConceptDecoder(nn.Module):
    """Context-capped, causally-masked, padding-aware decoder over Concept Vectors.

    `context_cap` is the hard attention window (defaults to the architectural
    CONTEXT_CAP). Patch counts beyond it raise rather than silently truncate.
    """

    def __init__(
        self,
        embed_dim: int = 256,        # D - must match the patcher / encoder width.
        num_heads: int = 8,
        num_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        context_cap: int = CONTEXT_CAP,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.context_cap = context_cap

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, P, D)
            norm_first=True,    # pre-LN: stabler, and a no-op on zeroed pad rows
        )
        # Encoder-layer stack + causal mask == decoder-only self-attention.
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Cache the full positional table once; slice [:P] per forward.
        pe = sinusoidal_encoding(context_cap, embed_dim)
        self.register_buffer("positional", pe, persistent=False)

    def forward(self, packed: PatchedBatch) -> ConceptBuffer:
        """(B, P, D) padded concept vectors -> (B, P, D) causally contextualized."""
        vectors, mask = packed.vectors, packed.mask
        assert vectors.dim() == 3, f"expected (B, P, D), got {tuple(vectors.shape)}"
        b, p, d = vectors.shape
        assert d == self.embed_dim, f"width {d} != model dim {self.embed_dim}"
        assert mask.shape == (b, p), mask.shape

        if p > self.context_cap:
            raise ValueError(
                f"{p} patches exceed CONTEXT_CAP={self.context_cap}; overflow must "
                "be routed to the Holographic Accumulator (Phase 3), not truncated."
            )

        # Positional signal only on real patches (padded rows are already 0).
        pos = self.positional[:p].to(vectors.dtype).unsqueeze(0)   # (1, P, D)
        x = vectors + pos * mask.unsqueeze(-1)                     # (B, P, D)

        # Causal bool mask (P, P): True == ignore. Upper triangle (strictly above
        # the diagonal) is masked, so concept i attends to <= i only. Bool (not
        # float) to match key_padding_mask's dtype — PyTorch deprecates mixing them.
        causal = torch.triu(
            torch.ones(p, p, dtype=torch.bool, device=vectors.device), diagonal=1
        )
        # key_padding_mask: True == ignore this key. DEP forces position 0 open and
        # patches are left-aligned, so causal + padding never fully mask a row.
        key_padding_mask = ~mask                                  # (B, P) bool
        x = self.decoder(x, mask=causal, src_key_padding_mask=key_padding_mask)

        # Re-zero padded query rows so downstream stages never see attention garbage.
        buffer = x * mask.unsqueeze(-1)

        assert buffer.shape == (b, p, d), buffer.shape
        return ConceptBuffer(buffer=buffer, mask=mask, counts=packed.counts)
