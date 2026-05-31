"""Test Gate 3.1 - contrastive + next-concept-prediction objective.

Verifies the InfoNCE core actually rewards correct matches, the combined loss is
a finite scalar with the right weighting, padded slots never influence it, the
per-term query masking is correct, and — the gate that de-risks Sprint 3.2 — the
loss is minimizable: it drops under optimization, and gradients flow end-to-end
through the real encoder -> DEP -> decoder -> loss path.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.losses import LossConfig, THCMLoss, info_nce
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _masked(b: int, p: int, d: int, counts: list[int], device: str):
    h = torch.randn(b, p, d, device=device)
    z = torch.randn(b, p, d, device=device)
    counts_t = torch.tensor(counts, device=device)
    mask = torch.arange(p, device=device).unsqueeze(0) < counts_t.unsqueeze(1)
    return h * mask.unsqueeze(-1), z * mask.unsqueeze(-1), mask


def test_info_nce_rewards_correct_match(device: str) -> None:
    torch.manual_seed(0)
    m, d = 6, 16
    bank = torch.randn(m, d, device=device)
    positives = torch.arange(m, device=device)

    perfect = info_nce(bank.clone(), bank, positives, temperature=0.1)   # query == positive
    scrambled = info_nce(torch.randn(m, d, device=device), bank, positives, 0.1)

    assert perfect < scrambled                  # matching query beats random
    assert float(perfect) < 0.1                 # near the InfoNCE floor
    assert float(scrambled) > 0.5               # random query is confused
    # Empty query set is a differentiable zero, not a crash.
    empty = info_nce(torch.empty(0, d, device=device), bank, torch.empty(0, dtype=torch.long, device=device), 0.1)
    assert float(empty) == 0.0


def test_components_finite_and_weighted(device: str) -> None:
    torch.manual_seed(1)
    loss = THCMLoss(embed_dim=16, config=LossConfig(w_next=1.0, w_contrastive=0.5)).to(device)
    h, z, mask = _masked(4, 10, 16, [10, 7, 3, 1], device)
    out = loss(h, z, mask)
    assert out.total.shape == () and torch.isfinite(out.total)
    assert torch.isfinite(out.next_concept) and torch.isfinite(out.contrastive)
    expected = 1.0 * out.next_concept + 0.5 * out.contrastive
    assert torch.allclose(out.total, expected, atol=1e-6)
    # n_next counts only positions with a real successor: sum(max(c-1, 0)).
    assert out.n_next == (9 + 6 + 2 + 0)
    assert out.n_contrastive == (10 + 7 + 3 + 1)   # every real concept


def test_padding_invariance(device: str) -> None:
    torch.manual_seed(2)
    loss = THCMLoss(embed_dim=16).to(device)
    h, z, mask = _masked(3, 12, 16, [12, 5, 8], device)
    out_clean = loss(h, z, mask)

    h2, z2 = h.clone(), z.clone()
    pad = ~mask
    h2[pad] = torch.randn_like(h2[pad]) * 30.0      # garbage into padded slots
    z2[pad] = torch.randn_like(z2[pad]) * 30.0
    out_noisy = loss(h2, z2, mask)

    assert torch.allclose(out_clean.total, out_noisy.total, atol=1e-5)


def test_loss_decreases_under_optimization(device: str) -> None:
    """The objective is well-formed and minimizable: a learnable contextual
    representation + heads can be driven to predict a fixed concept sequence."""
    torch.manual_seed(3)
    b, p, d = 2, 16, 32
    concepts = torch.randn(b, p, d, device=device)
    mask = torch.ones(b, p, dtype=torch.bool, device=device)
    contextual = torch.zeros(b, p, d, device=device, requires_grad=True)
    loss = THCMLoss(embed_dim=d).to(device)

    opt = torch.optim.Adam([contextual, *loss.parameters()], lr=5e-2)
    first = float(loss(contextual, concepts, mask).total)
    for _ in range(50):
        opt.zero_grad()
        out = loss(contextual, concepts, mask)
        out.total.backward()
        opt.step()
    last = float(loss(contextual, concepts, mask).total)

    assert last < first - 0.5, f"loss did not drop: {first:.3f} -> {last:.3f}"
    assert last < math.log(b * p)   # beats the uniform-guess baseline


def test_end_to_end_pipeline_backward(device: str) -> None:
    """Gradients flow through encoder -> DEP -> decoder -> loss with no NaNs."""
    torch.manual_seed(4)
    enc = ByteEncoder(embed_dim=32, num_blocks=2, kernel_size=5).to(device)
    dep = DynamicEntropyPatcher(threshold_k=1.0).to(device)
    dec = ConceptDecoder(embed_dim=32, num_heads=4, num_layers=2).to(device)
    loss = THCMLoss(embed_dim=32).to(device)

    bytes_in = torch.randint(0, 256, (4, 128), dtype=torch.uint8, device=device)
    traj, wave = enc(bytes_in)
    packed = dep(traj, wave)
    out = dec(packed)
    result = loss(out.buffer, packed.vectors, packed.mask)

    assert torch.isfinite(result.total)
    result.total.backward()
    # Encoder stem and a loss head must receive finite, non-trivial gradients.
    g_enc = enc.stem.weight.grad
    g_head = loss.predictor[0].weight.grad
    assert g_enc is not None and torch.isfinite(g_enc).all()
    assert g_head is not None and torch.isfinite(g_head).all()
    assert float(g_enc.abs().sum()) > 0.0
