"""Test Gate 2.1 - dynamic batch packing of variable-length patches.

Verifies the DEP slicing engine against an independent reference: correct
segment ids, mean-pooled concept vectors, padding/mask bookkeeping, and the
two degenerate regimes (one patch for the whole sequence; one patch per byte).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import (
    DynamicEntropyPatcher,
    detect_boundaries,
    pack_patches,
)
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _ref_pack(traj: torch.Tensor, boundaries: torch.Tensor):
    """Reference: split each row at boundary indices, mean-pool each segment."""
    b, length, _ = traj.shape
    rows = []
    for r in range(b):
        starts = [i for i in range(length) if bool(boundaries[r, i])]
        segs = []
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else length
            segs.append(traj[r, s:e].mean(dim=0))
        rows.append(segs)
    return rows


def test_pack_matches_reference(device: str) -> None:
    torch.manual_seed(0)
    b, length, d = 3, 40, 8
    traj = torch.randn(b, length, d, device=device)
    # Hand-built irregular boundaries (col 0 forced on inside detect/pack).
    boundaries = torch.zeros(b, length, dtype=torch.bool, device=device)
    boundaries[0, [0, 5, 20]] = True
    boundaries[1, [0]] = True                       # single patch
    boundaries[2, list(range(length))] = True       # patch per position

    packed = pack_patches(traj, boundaries)
    ref = _ref_pack(traj, boundaries)

    assert packed.counts.tolist() == [3, 1, length]
    assert packed.num_patches() == length           # P_max from row 2
    assert int(packed.mask.sum()) == 3 + 1 + length
    for r in range(b):
        for p in range(int(packed.counts[r])):
            assert torch.allclose(packed.vectors[r, p], ref[r][p], atol=1e-5)


def test_padding_is_zero_and_masked(device: str) -> None:
    torch.manual_seed(1)
    traj = torch.randn(2, 16, 4, device=device)
    boundaries = torch.zeros(2, 16, dtype=torch.bool, device=device)
    boundaries[0, [0, 8]] = True        # 2 patches
    boundaries[1, [0]] = True           # 1 patch -> 1 padded slot
    packed = pack_patches(traj, boundaries)
    assert packed.num_patches() == 2
    # Row 1's second slot is padding: masked False and exactly zero.
    assert not bool(packed.mask[1, 1])
    assert torch.all(packed.vectors[1, 1] == 0.0)


def test_threshold_sensitivity_controls_patch_count(device: str) -> None:
    torch.manual_seed(2)
    # A waveform with a few sharp spikes over a calm baseline.
    waveform = torch.full((1, 50), 0.1, device=device)
    waveform[0, [10, 25, 40]] = 5.0
    high_k = detect_boundaries(waveform, k=3.0)      # only spikes survive
    low_k = detect_boundaries(waveform, k=0.1)       # more positions cross
    assert int(high_k.sum()) <= int(low_k.sum())
    assert bool(high_k[0, 0]) and bool(low_k[0, 0])  # position 0 always opens


def test_end_to_end_encoder_to_patcher(device: str) -> None:
    torch.manual_seed(3)
    enc = ByteEncoder(embed_dim=32, num_blocks=2, kernel_size=5).to(device).eval()
    dep = DynamicEntropyPatcher(threshold_k=1.0).to(device)
    bytes_in = torch.randint(0, 256, (4, 128), dtype=torch.uint8, device=device)
    with torch.no_grad():
        traj, wave = enc(bytes_in)
        packed = dep(traj, wave)
    assert packed.vectors.shape[0] == 4
    assert packed.vectors.shape[2] == 32
    assert torch.all(packed.counts >= 1)
    assert int(packed.mask.sum()) == int(packed.counts.sum())
    assert torch.isfinite(packed.vectors).all()
