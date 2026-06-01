"""Test Gate 5.2 - GenerativeTHCM (byte LM with optional concept memory).

Verifies both modes produce valid next-byte logits, the no-memory baseline is
strictly causal, gradients flow end-to-end through the full concept pipeline into
the byte head when memory is on, and both modes are trainable (bpb drops). The
actual with-vs-without-memory bpb comparison is the job of bench/ablation_5_2.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.models.code_model import GenerativeTHCM
from thcm.models.generator import byte_lm_loss
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _model(device: str, use_memory: bool, d: int = 32) -> GenerativeTHCM:
    return GenerativeTHCM(embed_dim=d, use_memory=use_memory, byte_layers=2, byte_heads=4,
                          concept_layers=2, concept_heads=4, encoder_blocks=2).to(device)


@pytest.mark.parametrize("use_memory", [False, True])
def test_forward_shapes_and_finite(device: str, use_memory: bool) -> None:
    torch.manual_seed(0)
    m = _model(device, use_memory).eval()
    bytes_in = torch.randint(0, 256, (3, 96), dtype=torch.uint8, device=device)
    with torch.no_grad():
        logits = m(bytes_in)
        _, bpb = m.loss(bytes_in)
    assert logits.shape == (3, 96, 256)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(bpb) and float(bpb) > 0


def test_no_memory_baseline_is_causal(device: str) -> None:
    torch.manual_seed(1)
    m = _model(device, use_memory=False).eval()
    bytes_in = torch.randint(0, 256, (1, 40), dtype=torch.uint8, device=device)
    with torch.no_grad():
        base = m(bytes_in)
    k = 22
    pert = bytes_in.clone()
    pert[0, k] = (int(pert[0, k]) + 3) % 256
    with torch.no_grad():
        out = m(pert)
    assert torch.allclose(base[0, :k], out[0, :k], atol=1e-5)


def test_memory_mode_backward_reaches_encoder(device: str) -> None:
    """End-to-end differentiability: byte loss flows through DEP+decoder into the
    concept encoder, so the concept path is actually trained by the LM signal."""
    torch.manual_seed(2)
    m = _model(device, use_memory=True)
    bytes_in = torch.randint(0, 256, (4, 96), dtype=torch.uint8, device=device)
    loss, _ = m.loss(bytes_in)
    loss.backward()
    g_enc = m.encoder.stem.weight.grad
    g_byte = m.byte_decoder.head.weight.grad
    assert g_enc is not None and torch.isfinite(g_enc).all() and float(g_enc.abs().sum()) > 0
    assert g_byte is not None and torch.isfinite(g_byte).all()


@pytest.mark.parametrize("use_memory", [False, True])
def test_trainable_bpb_drops(device: str, use_memory: bool) -> None:
    torch.manual_seed(3)
    m = _model(device, use_memory, d=48)
    bytes_in = torch.randint(0, 256, (2, 64), dtype=torch.uint8, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    _, bpb0 = m.loss(bytes_in)
    for _ in range(60):
        opt.zero_grad()
        loss, _ = m.loss(bytes_in)
        loss.backward()
        opt.step()
    _, bpb1 = m.loss(bytes_in)
    assert float(bpb1) < float(bpb0) - 1.0          # clearly learning to generate
