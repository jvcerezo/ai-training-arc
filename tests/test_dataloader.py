"""Test Gate 1.1 — Byte Data Engineering.

Pass criteria:
  1. A known byte string round-trips to a (B, L) uint8 tensor with byte-exact values.
  2. to_onehot yields (B, L, 256) float32 on the active device, sums to 1 along
     the vocab axis, and argmax recovers the original bytes.
  3. The transferred batch is uint8 (Invariant B: no float crossed the bus).

These run on whatever backend preflight resolves (CPU is a valid correctness
substrate; the latency gate is a separate, GPU-only check in Sprint 4).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thcm.config import VOCAB_SIZE, LoaderConfig
from thcm.data.dataloader import (
    ByteWindowDataset,
    build_loader,
    collate_bytes,
    to_onehot,
)
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


@pytest.fixture()
def corpus(tmp_path) -> str:
    """A deterministic 4 KB byte corpus covering all 256 values."""
    path = tmp_path / "corpus.bin"
    data = np.arange(4096, dtype=np.int64) % 256
    data.astype(np.uint8).tofile(path)
    return str(path)


def test_window_is_byte_exact(corpus: str) -> None:
    seq_len = 64
    ds = ByteWindowDataset(corpus, seq_len)
    window = ds[10]
    assert window.shape == (seq_len,)
    assert window.dtype == torch.uint8
    # corpus[i] == i % 256, so window[j] == (10 + j) % 256.
    expected = torch.tensor([(10 + j) % 256 for j in range(seq_len)], dtype=torch.uint8)
    assert torch.equal(window, expected)


def test_collate_stays_uint8(corpus: str) -> None:
    ds = ByteWindowDataset(corpus, 32)
    batch = collate_bytes([ds[0], ds[1], ds[2]])
    assert batch.shape == (3, 32)
    assert batch.dtype == torch.uint8  # Invariant B


def test_onehot_shape_and_recovery(corpus: str, device: str) -> None:
    ds = ByteWindowDataset(corpus, 32)
    batch = collate_bytes([ds[0], ds[1]]).to(device)
    onehot = to_onehot(batch)
    assert onehot.shape == (2, 32, VOCAB_SIZE)
    assert onehot.dtype == torch.float32
    assert str(onehot.device).split(":")[0] == device.split(":")[0]
    # Exactly one hot per position.
    assert torch.all(onehot.sum(dim=-1) == 1.0)
    # argmax recovers the original bytes.
    recovered = onehot.argmax(dim=-1).to(torch.uint8).cpu()
    assert torch.equal(recovered, batch.cpu())


def test_build_loader_yields_uint8_batches(corpus: str) -> None:
    cfg = LoaderConfig(
        corpus_path=corpus,
        seq_len=32,
        batch_size=4,
        num_workers=0,        # in-process for deterministic test
        device=preflight().device_str,
    )
    prefetcher = build_loader(cfg)
    batch = next(iter(prefetcher))
    assert batch.shape == (4, 32)
    assert batch.dtype == torch.uint8
