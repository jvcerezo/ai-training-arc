"""Global dimensional constants and configuration for the T-HCM engine.

Invariant A (architecture decision): VOCAB_SIZE (V) and EMBED_DIM (D) are
DIFFERENT quantities and must never be conflated, even if they happen to share
a numeric value. V is the fixed raw-byte space; D is a tunable latent width.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Fixed architecture constants -------------------------------------------
VOCAB_SIZE: int = 256  # V — raw byte vocabulary. IMMUTABLE. Never == "D by luck".


@dataclass(frozen=True)
class LoaderConfig:
    """Configuration for the byte-window data pipeline (Sprint 1.1)."""

    corpus_path: str
    seq_len: int = 2048          # L — matches the local Transformer window cap.
    batch_size: int = 32         # B
    num_workers: int = 4
    prefetch_factor: int = 4
    device: str = "cuda"         # ROCm/HIP exposes the AMD GPU as "cuda".


@dataclass(frozen=True)
class ModelDims:
    """Latent dimensions consumed downstream (Sprint 1.2+). Declared early so the
    V-vs-D separation is explicit from the first commit."""

    embed_dim: int = 256         # D — latent width. Tunable; NOT tied to V.
    vocab_size: int = VOCAB_SIZE  # echo of V for assert sites.
