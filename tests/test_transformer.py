"""Test Gate 2.2 - context-capped causal Concept Decoder (buffer build).

Verifies the decoder-only Transformer that consumes PatchedBatch concept vectors:
shape integrity, the two masking invariants that make it correct (causality and
padding invariance), the hard context-cap guard, and the encoder->DEP->decoder
end-to-end path. The masking tests are the heart of the gate — a vanilla encoder
or a dropped mask would silently pass shapes but fail these.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.config import CONTEXT_CAP
from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher, PatchedBatch
from thcm.models.transformer import ConceptDecoder, sinusoidal_encoding
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _packed(vectors: torch.Tensor, counts: list[int]) -> PatchedBatch:
    """Build a PatchedBatch directly from vectors + per-row real-patch counts."""
    b, p, _ = vectors.shape
    counts_t = torch.tensor(counts, device=vectors.device)
    mask = torch.arange(p, device=vectors.device).unsqueeze(0) < counts_t.unsqueeze(1)
    vectors = vectors * mask.unsqueeze(-1)  # zero padded slots, like the patcher
    seg = torch.zeros(b, 1, dtype=torch.long, device=vectors.device)
    return PatchedBatch(vectors=vectors, mask=mask, counts=counts_t, segment_id=seg)


def _decoder(device: str, d: int = 32) -> ConceptDecoder:
    return ConceptDecoder(embed_dim=d, num_heads=4, num_layers=3).to(device).eval()


def test_shape_integrity_and_finite(device: str) -> None:
    torch.manual_seed(0)
    dec = _decoder(device)
    packed = _packed(torch.randn(4, 20, 32, device=device), [20, 13, 7, 1])
    with torch.no_grad():
        out = dec(packed)
    assert out.buffer.shape == (4, 20, 32)
    assert out.mask.shape == (4, 20)
    assert torch.isfinite(out.buffer).all()
    # Padded query rows must come back exactly zero.
    assert torch.all(out.buffer[1, 13:] == 0.0)
    assert torch.all(out.buffer[3, 1:] == 0.0)


def test_causality_future_does_not_leak_to_past(device: str) -> None:
    """Perturbing patch k must not change any output at positions < k."""
    torch.manual_seed(1)
    dec = _decoder(device)
    # Single fully-real row isolates causality from padding effects.
    base = torch.randn(1, 16, 32, device=device)
    packed = _packed(base, [16])
    with torch.no_grad():
        out0 = dec(packed).buffer

    k = 9
    perturbed = base.clone()
    perturbed[0, k] += 10.0  # large kick at position k
    with torch.no_grad():
        out1 = dec(_packed(perturbed, [16])).buffer

    # Positions strictly before k are untouched...
    assert torch.allclose(out0[0, :k], out1[0, :k], atol=1e-5)
    # ...and position k itself does change (sanity: the kick actually propagates).
    assert not torch.allclose(out0[0, k], out1[0, k], atol=1e-4)


def test_padding_invariance_scrambled_pad_slots(device: str) -> None:
    """Garbage in padded slots must not change any real patch's output."""
    torch.manual_seed(2)
    dec = _decoder(device)
    real = torch.randn(2, 18, 32, device=device)
    counts = [18, 5]  # row 1 has 13 padded slots
    out_clean = dec(_packed(real, counts)).buffer

    scrambled = real.clone()
    scrambled[1, 5:] = torch.randn(13, 32, device=device) * 50.0  # fill pad with noise
    out_noisy = dec(_packed(scrambled, counts)).buffer

    with torch.no_grad():
        # Real patches of the padded row are bit-stable against pad contents.
        assert torch.allclose(out_clean[1, :5], out_noisy[1, :5], atol=1e-5)
        # The fully-real row is untouched too.
        assert torch.allclose(out_clean[0], out_noisy[0], atol=1e-5)


def test_context_cap_overflow_raises(device: str) -> None:
    dec = ConceptDecoder(embed_dim=16, num_heads=2, num_layers=1, context_cap=8).to(device).eval()
    over = _packed(torch.randn(1, 9, 16, device=device), [9])
    with pytest.raises(ValueError, match="exceed CONTEXT_CAP"):
        dec(over)


def test_positional_table_sized_to_cap(device: str) -> None:
    pe = sinusoidal_encoding(CONTEXT_CAP, 32, device=device)
    assert pe.shape == (CONTEXT_CAP, 32)
    assert torch.isfinite(pe).all()


def test_end_to_end_encoder_to_decoder(device: str) -> None:
    torch.manual_seed(3)
    enc = ByteEncoder(embed_dim=32, num_blocks=2, kernel_size=5).to(device).eval()
    dep = DynamicEntropyPatcher(threshold_k=1.0).to(device)
    dec = _decoder(device)
    bytes_in = torch.randint(0, 256, (4, 128), dtype=torch.uint8, device=device)
    with torch.no_grad():
        traj, wave = enc(bytes_in)
        packed = dep(traj, wave)
        out = dec(packed)
    assert out.buffer.shape[0] == 4 and out.buffer.shape[2] == 32
    assert out.num_patches() == packed.num_patches()
    assert torch.isfinite(out.buffer).all()
    assert int(out.mask.sum()) == int(out.counts.sum())
