"""Sprint 3.1 - Training objective: contrastive + next-concept-prediction CE.

The engine is tokenless: concepts are continuous vectors, not entries in a fixed
vocabulary, so "next-concept-prediction cross-entropy" cannot be a softmax over a
vocab dict. The continuous analog is InfoNCE (van den Oord 2018): cross-entropy
over a *contrastive candidate set* of real concept vectors, where the positive is
the true target and the negatives are the other concepts in the batch. Two terms
share that machinery:

  * next-concept (autoregressive). From the causal decoder output h_i (which has
    attended to concepts <= i) predict the NEXT concept z_{i+1}. Positive is the
    true z_{i+1}; negatives are every other real concept in the batch. This is the
    tokenless replacement for next-token cross-entropy.

  * contrastive (alignment / anti-collapse). The projected output h_i must
    re-identify its OWN concept z_i among all concepts in the batch. A collapsed
    encoder (all concepts -> one vector) cannot discriminate here, so this term
    punishes representational collapse and gives a clean early gradient.

Both are masked: padded slots never enter the candidate bank or the query set,
and the last real position of each sequence has no next-target so it is excluded
from the next-concept term.

Shape contract:
    contextual h (B, P, D), concepts z (B, P, D), mask (B, P) bool
      -> bank = z[real]                         (M, D)   negatives + positives
      -> next queries  predictor(h_i)[valid]    (Qn, D), positives = idx of z_{i+1}
      -> self queries  projector(h_i)[real]     (M,  D), positives = idx of z_i
      -> InfoNCE cross-entropy on each           scalars
      -> total = w_next * next + w_contrastive * contrastive
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LossConfig:
    """Weights and temperature for the combined training objective."""

    temperature: float = 0.1
    w_next: float = 1.0
    w_contrastive: float = 0.5


@dataclass
class LossOutput:
    """Combined loss plus its components and the per-term query counts."""

    total: torch.Tensor        # scalar - weighted sum, the thing to .backward()
    next_concept: torch.Tensor  # scalar - InfoNCE next-concept term (unweighted)
    contrastive: torch.Tensor  # scalar - InfoNCE self-identification term (unweighted)
    n_next: int                # number of valid next-concept queries
    n_contrastive: int         # number of real concepts (self queries)


def info_nce(queries: torch.Tensor, bank: torch.Tensor,
             positives: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-entropy over cosine-similarity logits. queries (Q,D), bank (M,D).

    `positives[q]` is the row of `bank` that is the correct match for query q.
    Returns a scalar; an empty query set returns a differentiable 0 so callers
    never special-case degenerate batches.
    """
    if queries.numel() == 0:
        return bank.new_zeros(())
    q = F.normalize(queries, dim=-1)
    k = F.normalize(bank, dim=-1)
    logits = (q @ k.t()) / temperature           # (Q, M)
    return F.cross_entropy(logits, positives)


class THCMLoss(nn.Module):
    """Contrastive + next-concept-prediction objective over the concept buffer.

    Owns a `predictor` (forecasts the next concept) and a `projector` (maps the
    contextual output into the concept space for self-identification). Both are
    small MLPs so the objective has learnable capacity and the gradient path is
    real for the training loop (Sprint 3.2).
    """

    def __init__(self, embed_dim: int = 256, config: LossConfig | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config or LossConfig()
        self.predictor = self._head(embed_dim)
        self.projector = self._head(embed_dim)

    @staticmethod
    def _head(dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, contextual: torch.Tensor, concepts: torch.Tensor,
                mask: torch.Tensor) -> LossOutput:
        assert contextual.shape == concepts.shape, (contextual.shape, concepts.shape)
        b, p, d = concepts.shape
        assert d == self.embed_dim, f"width {d} != loss dim {self.embed_dim}"
        assert mask.shape == (b, p), mask.shape

        # Candidate bank = every real concept, flattened. bank_index maps each
        # (b, p) slot to its row in the bank (-1 for padded slots, never queried).
        flat_mask = mask.reshape(-1)                       # (B*P,)
        bank = concepts.reshape(-1, d)[flat_mask]          # (M, D)
        m = bank.shape[0]
        bank_index = torch.full((b * p,), -1, dtype=torch.long, device=concepts.device)
        bank_index[flat_mask] = torch.arange(m, device=concepts.device)
        bank_index = bank_index.reshape(b, p)              # (B, P)

        # --- next-concept: position i predicts z_{i+1}; both i and i+1 real. ---
        valid_next = mask[:, :-1] & mask[:, 1:]            # (B, P-1)
        next_q = self.predictor(contextual[:, :-1])[valid_next]   # (Qn, D)
        next_pos = bank_index[:, 1:][valid_next]                  # (Qn,) into bank
        loss_next = info_nce(next_q, bank, next_pos, self.config.temperature)

        # --- contrastive: every real h_i re-identifies its own z_i. ---
        self_q = self.projector(contextual)[mask]          # (M, D)
        self_pos = bank_index[mask]                        # (M,) == arange(M)
        loss_contrastive = info_nce(self_q, bank, self_pos, self.config.temperature)

        total = self.config.w_next * loss_next + self.config.w_contrastive * loss_contrastive
        return LossOutput(
            total=total,
            next_concept=loss_next,
            contrastive=loss_contrastive,
            n_next=int(valid_next.sum().item()),
            n_contrastive=m,
        )
