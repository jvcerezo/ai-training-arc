"""Sprint 1.1 — Byte Data Engineering.

Tokenless ingestion: bytes ARE the vocabulary. The pipeline memory-maps a raw
byte corpus, samples fixed-length windows, and ships them to the GPU as `uint8`
(Invariant B: never inflate to float before the bus). The one-hot / projection
materializes ON-DEVICE via `to_onehot`, which is also the seam into Sprint 1.2.

Boundaries covered (see shape contract): (1) corpus -> (2) window -> (3) batch
-> (4) GPU transfer -> (5) on-device one-hot (B, L, 256).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from thcm.config import VOCAB_SIZE, LoaderConfig


class ByteWindowDataset(Dataset):
    """Memory-maps a raw byte corpus; yields fixed-length ``uint8`` windows.

    No vocabulary dict, no tokenizer model — operates directly on the 256-byte
    space. ``np.memmap`` keeps an arbitrarily large corpus off-RAM.
    """

    def __init__(self, path: str, seq_len: int) -> None:
        self._data = np.memmap(path, dtype=np.uint8, mode="r")
        self._seq_len = seq_len
        assert self._data.shape[0] > seq_len, (
            f"corpus ({self._data.shape[0]} bytes) shorter than one window "
            f"({seq_len})"
        )

    def __len__(self) -> int:
        # Number of valid start offsets for a full window.
        return int(self._data.shape[0]) - self._seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self._data[idx : idx + self._seq_len]   # (L,) uint8 view
        # .copy() detaches from the read-only mmap so torch owns contiguous bytes.
        tensor = torch.from_numpy(window.copy())          # (L,) uint8
        assert tensor.shape == (self._seq_len,), tensor.shape
        assert tensor.dtype == torch.uint8
        return tensor


def collate_bytes(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack windows into a ``(B, L)`` ``uint8`` batch — kept tiny for the PCIe hop."""
    out = torch.stack(batch, dim=0)
    assert out.dtype == torch.uint8, "Invariant B: batch must stay uint8"
    assert out.dim() == 2, f"expected (B, L), got {tuple(out.shape)}"
    return out


class CudaPrefetcher:
    """Double-buffers the H2D copy on a side stream to hide transfer latency.

    This is the mechanism behind the "no pipeline stalling" gate: the next
    batch's copy is launched on a side stream while the current batch is being
    consumed, so the transfer overlaps compute instead of blocking on it.
    """

    def __init__(self, loader: DataLoader, device: str) -> None:
        self._it = iter(loader)
        self._device = device
        self._stream = torch.cuda.Stream() if device != "cpu" else None
        self._next: torch.Tensor | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            cpu_batch = next(self._it)                    # (B, L) uint8, pinned
        except StopIteration:
            self._next = None
            return
        if self._stream is None:                          # CPU fallback path
            self._next = cpu_batch.to(self._device)
            return
        with torch.cuda.stream(self._stream):
            self._next = cpu_batch.to(self._device, non_blocking=True)

    def __iter__(self) -> "CudaPrefetcher":
        return self

    def __next__(self) -> torch.Tensor:
        if self._next is None:
            raise StopIteration
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next                                # (B, L) uint8 on device
        self._preload()
        return batch


def build_loader(cfg: LoaderConfig) -> CudaPrefetcher:
    """Assemble the full Sprint 1.1 pipeline: dataset -> loader -> prefetcher."""
    dataset = ByteWindowDataset(cfg.corpus_path, cfg.seq_len)
    pin = cfg.device != "cpu"
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        collate_fn=collate_bytes,
        drop_last=True,
    )
    return CudaPrefetcher(loader, cfg.device)


def to_onehot(byte_batch: torch.Tensor) -> torch.Tensor:
    """``(B, L)`` uint8 on device -> ``(B, L, 256)`` float32 on the SAME device.

    This is the Test-Gate (B, L, 256) reference tensor AND the seam into the
    Sprint 1.2 conv encoder. Runs on-device so no float ever crosses the bus.
    """
    assert byte_batch.dtype == torch.uint8, (
        "Invariant B violated: transfer must stay uint8 until on-device"
    )
    assert byte_batch.dim() == 2, f"expected (B, L), got {tuple(byte_batch.shape)}"
    onehot = torch.nn.functional.one_hot(byte_batch.long(), VOCAB_SIZE)
    onehot = onehot.to(torch.float32)                     # (B, L, 256)
    b, length = byte_batch.shape
    assert onehot.shape == (b, length, VOCAB_SIZE), onehot.shape
    return onehot
