"""Test Gate 4.2 - robustness: semantic recovery across corrupted inputs.

Verifies the corruptor hits its target rate, the pipeline degrades gracefully
(monotonic, high at low corruption, no collapse), the recovered patch structure
survives surface corruption far better than the raw bytes, and the pooled
document embedding stays stable. Closes Phase 4 / the project.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.eval.robustness import (
    byte_match,
    corrupt_substitution,
    robustness_curve,
)
from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.utils.device import preflight

RATES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


@pytest.fixture(scope="module")
def stack(device: str):
    torch.manual_seed(0)
    d = 64
    enc = ByteEncoder(embed_dim=d, num_blocks=2, kernel_size=5).to(device).eval()
    dep = DynamicEntropyPatcher(threshold_k=1.0).to(device)
    dec = ConceptDecoder(embed_dim=d, num_heads=4, num_layers=2).to(device).eval()
    return enc, dep, dec


@pytest.fixture(scope="module")
def curve(device: str, stack):
    enc, dep, dec = stack
    clean = torch.randint(0, 256, (4, 512), dtype=torch.uint8, device=device)
    gen = torch.Generator(device=device).manual_seed(123)
    return robustness_curve(enc, dep, dec, clean, RATES, gen)


def test_corruption_rate_is_respected(device: str) -> None:
    torch.manual_seed(1)
    clean = torch.randint(0, 256, (4, 1024), dtype=torch.uint8, device=device)
    gen = torch.Generator(device=device).manual_seed(7)
    assert byte_match(clean, corrupt_substitution(clean, 0.0, gen)) == 1.0
    changed = 1.0 - byte_match(clean, corrupt_substitution(clean, 0.2, gen))
    # ~0.2 of positions flip (minus the ~1/256 that randomly land on themselves).
    assert 0.15 < changed < 0.22


def test_clean_input_is_identity(curve) -> None:
    zero = curve[0]
    assert zero.rate == 0.0
    assert zero.byte_match == 1.0
    assert zero.boundary_agreement == 1.0
    assert zero.embedding_cosine > 0.999


def test_graceful_degradation(curve) -> None:
    """Boundary recovery is monotonic, high at low corruption, never collapses."""
    agree = [p.boundary_agreement for p in curve]
    for prev, nxt in zip(agree, agree[1:]):
        assert nxt <= prev + 0.01           # non-increasing (small numerical slack)
    by_rate = {p.rate: p.boundary_agreement for p in curve}
    assert by_rate[0.05] >= 0.90            # 5% corruption barely moves boundaries
    assert by_rate[0.40] > 0.70             # even 40% corruption: no collapse


def test_structure_survives_surface_corruption(curve) -> None:
    """The recovery claim: at heavy corruption the patch structure is preserved
    much better than the raw byte surface."""
    for p in curve:
        if p.rate >= 0.2:
            assert p.boundary_agreement > p.byte_match, (
                f"rate {p.rate}: boundaries {p.boundary_agreement:.3f} "
                f"not above bytes {p.byte_match:.3f}")


def test_global_embedding_is_stable(curve) -> None:
    """The pooled document vector degrades gracefully — stays near the clean one."""
    for p in curve:
        assert p.embedding_cosine >= 0.95
