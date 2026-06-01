"""Sprint 5.2 - GenerativeTHCM: the byte LM, with concept memory on a switch.

This is the model we train toward a coding tool, and the vehicle for the key
kill-gate: does conditioning byte prediction on the concept architecture's output
actually lower bits-per-byte versus a plain causal byte LM? `use_memory` toggles
exactly that, with everything else held fixed, so an A/B run answers it cleanly.

  use_memory=True:  bytes -> encoder -> DEP -> concept decoder -> buffer
                    -> memory_from_concepts (prior-patch, causal) ----+
                    bytes ------------------------------------------> byte decoder (+memory) -> logits
  use_memory=False: bytes -> byte decoder -> logits          (plain causal byte LM baseline)

If the concept path doesn't beat the baseline on held-out bpb, the novel
machinery doesn't earn its place in a generative model — and we learn that for the
price of one short run instead of a full training campaign.
"""

from __future__ import annotations

import torch
from torch import nn

from thcm.models.encoder import ByteEncoder
from thcm.models.generator import CausalByteDecoder, byte_lm_loss, memory_from_concepts
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder


class GenerativeTHCM(nn.Module):
    """Causal byte LM, optionally conditioned on the T-HCM concept buffer."""

    def __init__(
        self,
        embed_dim: int = 256,
        *,
        use_memory: bool = True,
        byte_layers: int = 4,
        byte_heads: int = 8,
        concept_layers: int = 4,
        concept_heads: int = 8,
        encoder_blocks: int = 4,
        threshold_k: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_memory = use_memory
        self.byte_decoder = CausalByteDecoder(
            embed_dim=embed_dim, num_layers=byte_layers, num_heads=byte_heads, dropout=dropout)
        if use_memory:
            self.encoder = ByteEncoder(embed_dim=embed_dim, num_blocks=encoder_blocks, kernel_size=5)
            self.patcher = DynamicEntropyPatcher(threshold_k=threshold_k)
            self.concept_decoder = ConceptDecoder(
                embed_dim=embed_dim, num_heads=concept_heads, num_layers=concept_layers, dropout=dropout)

    def _memory(self, byte_batch: torch.Tensor) -> torch.Tensor:
        trajectory, waveform = self.encoder(byte_batch)
        packed = self.patcher(trajectory, waveform)
        buf = self.concept_decoder(packed)
        return memory_from_concepts(buf.buffer, packed.segment_id)   # (B, L, D)

    def forward(self, byte_batch: torch.Tensor) -> torch.Tensor:
        """(B, L) uint8 -> (B, L, 256) next-byte logits."""
        memory = self._memory(byte_batch) if self.use_memory else None
        return self.byte_decoder(byte_batch, memory=memory)

    def loss(self, byte_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-byte (loss_nats, bits_per_byte)."""
        return byte_lm_loss(self.forward(byte_batch), byte_batch)
