"""Sprint 5.1 - generative byte head + bits-per-byte (the coding-model objective).

Everything a coding tool does — completion, generation, eventually chat — requires
the engine to *emit bytes*. The representation stack (encoder -> DEP -> concept
decoder) only recognizes the right next concept; it cannot produce code. This adds
the generative half: a strictly-causal byte-level autoregressive decoder that
predicts the next byte, trained with cross-entropy and scored in **bits-per-byte**
(bpb) — the standard, benchmarkable yardstick (random = 8 bpb; lower = better).

Per the chosen design, the generator can be *conditioned on the concept buffer* as
compressed long-range memory: each byte attends to its prior patch's contextual
concept (strictly causal — patches before the current one only). Sprint 5.2 will
A/B whether that memory actually lowers bpb versus a plain byte LM; if it does, the
novel architecture earns its place in a generative model.

Causality contract (the gate): the logit for position t depends only on bytes <= t.

    bytes (B, L) uint8
      -> embed + positional                         (B, L, D)
      -> [+ per-byte concept memory]                (B, L, D)   optional, causal
      -> causal Transformer (lower-triangular mask)  (B, L, D)
      -> linear head                                 (B, L, 256) next-byte logits
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from thcm.config import VOCAB_SIZE
from thcm.models.transformer import sinusoidal_encoding


def memory_from_concepts(buffer: torch.Tensor, segment_id: torch.Tensor) -> torch.Tensor:
    """Per-byte memory = the contextual concept of each byte's PRIOR patch.

    buffer (B, P, D), segment_id (B, L) -> (B, L, D). Using patch p-1 (not p)
    keeps it strictly causal: patch p-1 summarizes only bytes that precede the
    current patch. Bytes in patch 0 get a zero memory (no prior context yet).
    """
    b, p, d = buffer.shape
    prior = (segment_id - 1).clamp(min=0)                       # (B, L)
    mem = torch.gather(buffer, 1, prior.unsqueeze(-1).expand(-1, -1, d))
    return mem * (segment_id > 0).unsqueeze(-1).to(buffer.dtype)


class CausalByteDecoder(nn.Module):
    """Strictly-causal byte LM head, optionally conditioned on concept memory."""

    def __init__(self, embed_dim: int = 256, num_layers: int = 4, num_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, context_cap: int = 4096) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide num_heads"
        self.embed_dim = embed_dim
        self.context_cap = context_cap
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, VOCAB_SIZE)
        self.register_buffer("positional", sinusoidal_encoding(context_cap, embed_dim),
                             persistent=False)

    def forward(self, byte_batch: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """(B, L) uint8 -> (B, L, 256) next-byte logits."""
        assert byte_batch.dim() == 2, f"expected (B, L), got {tuple(byte_batch.shape)}"
        b, length = byte_batch.shape
        if length > self.context_cap:
            raise ValueError(f"{length} bytes exceed context_cap={self.context_cap}")

        x = self.embed(byte_batch.long()) + self.positional[:length].unsqueeze(0)
        if memory is not None:
            assert memory.shape == (b, length, self.embed_dim), memory.shape
            x = x + memory                                       # causal concept context
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=byte_batch.device), diagonal=1)
        x = self.decoder(x, mask=causal)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_bytes: int = 256, *,
                 temperature: float = 1.0, greedy: bool = False) -> torch.Tensor:
        """Autoregressively extend a (B, L0) uint8 prompt by max_new_bytes."""
        self.eval()
        seq = prompt.long()
        for _ in range(max_new_bytes):
            logits = self.forward(seq[:, -self.context_cap:])[:, -1]   # (B, 256)
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq.to(prompt.dtype)


def byte_lm_loss(logits: torch.Tensor, byte_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-byte cross-entropy. Returns (loss_nats, bits_per_byte).

    logits[:, t] predicts byte[:, t+1], so the last position has no target.
    bits-per-byte = nats / ln(2) — random guessing over 256 bytes is 8.0 bpb.
    """
    assert logits.dim() == 3 and logits.shape[-1] == VOCAB_SIZE, tuple(logits.shape)
    v = logits.shape[-1]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, v), byte_batch[:, 1:].reshape(-1).long())
    return ce, ce / math.log(2)
