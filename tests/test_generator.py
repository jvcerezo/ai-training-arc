"""Test Gate 5.1 - generative byte head + bits-per-byte.

The foundation of any coding tool: the model must emit bytes, causally, and be
trainable toward low bits-per-byte. Verifies strict causality (no future byte
leaks into a logit), the bpb metric (random = 8), that optimization drives bpb
far below random (it genuinely generates/memorizes), the concept-memory
conditioning path, and autoregressive generation.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from thcm.models.generator import (
    CausalByteDecoder,
    byte_lm_loss,
    memory_from_concepts,
)
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def test_byte_lm_loss_random_is_eight_bpb(device: str) -> None:
    # Uniform logits => cross-entropy ln(256) => exactly 8 bits per byte.
    logits = torch.zeros(2, 32, 256, device=device)
    bytes_in = torch.randint(0, 256, (2, 32), dtype=torch.uint8, device=device)
    _, bpb = byte_lm_loss(logits, bytes_in)
    assert abs(float(bpb) - 8.0) < 1e-4


def test_causality_no_future_leak(device: str) -> None:
    torch.manual_seed(0)
    dec = CausalByteDecoder(embed_dim=32, num_layers=3, num_heads=4).to(device).eval()
    bytes_in = torch.randint(0, 256, (1, 24), dtype=torch.uint8, device=device)
    with torch.no_grad():
        base = dec(bytes_in)
    k = 13
    perturbed = bytes_in.clone()
    perturbed[0, k] = (int(perturbed[0, k]) + 7) % 256
    with torch.no_grad():
        out = dec(perturbed)
    # Logits for positions < k cannot depend on the byte at k (strict causality).
    assert torch.allclose(base[0, :k], out[0, :k], atol=1e-5)
    assert not torch.allclose(base[0, k], out[0, k], atol=1e-4)


def test_optimization_drives_bpb_down(device: str) -> None:
    """A tiny fixed batch should be memorizable: bpb falls from ~8 toward 0."""
    torch.manual_seed(1)
    dec = CausalByteDecoder(embed_dim=64, num_layers=2, num_heads=4).to(device)
    bytes_in = torch.randint(0, 256, (2, 64), dtype=torch.uint8, device=device)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)

    _, bpb0 = byte_lm_loss(dec(bytes_in), bytes_in)
    for _ in range(120):
        opt.zero_grad()
        loss, _ = byte_lm_loss(dec(bytes_in), bytes_in)
        loss.backward()
        opt.step()
    _, bpb1 = byte_lm_loss(dec(bytes_in), bytes_in)

    assert float(bpb0) > 7.0                 # starts near random
    assert float(bpb1) < 1.5                 # learns to predict the next byte


def test_memory_from_concepts_is_prior_patch(device: str) -> None:
    # 1 sequence, 3 patches, D=4; bytes 0-1 -> patch0, 2-4 -> patch1, 5 -> patch2.
    buffer = torch.tensor([[[1., 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]], device=device)
    seg = torch.tensor([[0, 0, 1, 1, 1, 2]], device=device)
    mem = memory_from_concepts(buffer, seg)            # (1, 6, 4)
    assert torch.all(mem[0, 0] == 0.0)                 # patch 0: no prior -> zero
    assert torch.all(mem[0, 2] == 1.0)                 # patch 1's bytes see patch 0
    assert torch.all(mem[0, 5] == 2.0)                 # patch 2's byte sees patch 1


def test_memory_conditioning_runs(device: str) -> None:
    torch.manual_seed(2)
    dec = CausalByteDecoder(embed_dim=16, num_layers=2, num_heads=4).to(device).eval()
    bytes_in = torch.randint(0, 256, (2, 20), dtype=torch.uint8, device=device)
    memory = torch.randn(2, 20, 16, device=device)
    with torch.no_grad():
        logits = dec(bytes_in, memory=memory)
    assert logits.shape == (2, 20, 256)
    assert torch.isfinite(logits).all()


def test_generate_extends_prompt(device: str) -> None:
    torch.manual_seed(3)
    dec = CausalByteDecoder(embed_dim=32, num_layers=2, num_heads=4).to(device).eval()
    prompt = torch.randint(0, 256, (1, 8), dtype=torch.uint8, device=device)
    out = dec.generate(prompt, max_new_bytes=16, greedy=True)
    assert out.shape == (1, 24)
    assert torch.equal(out[:, :8], prompt)             # prompt preserved as prefix
    # Greedy is deterministic.
    out2 = dec.generate(prompt, max_new_bytes=16, greedy=True)
    assert torch.equal(out, out2)
