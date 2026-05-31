"""Test Gate 1.2 - shape integrity through the convolutional blocks.

Verifies the byte encoder preserves sequence length (length-preserving 'same'
convolutions), emits the (B, L, D) trajectory and (B, L) waveform with exact
shapes, and is device-agnostic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.models.encoder import ByteEncoder
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


@pytest.mark.parametrize("b,length,d", [(2, 128, 64), (1, 33, 16), (4, 2048, 32)])
def test_encoder_shape_integrity(b: int, length: int, d: int, device: str) -> None:
    torch.manual_seed(0)
    enc = ByteEncoder(embed_dim=d, num_blocks=3, kernel_size=5).to(device).eval()
    bytes_in = torch.randint(0, 256, (b, length), dtype=torch.uint8, device=device)
    with torch.no_grad():
        trajectory, waveform = enc(bytes_in)
    assert trajectory.shape == (b, length, d)
    assert waveform.shape == (b, length)
    assert trajectory.dtype == torch.float32
    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(waveform).all()


def test_waveform_first_position_is_zero(device: str) -> None:
    torch.manual_seed(0)
    enc = ByteEncoder(embed_dim=32, num_blocks=2).to(device).eval()
    bytes_in = torch.randint(0, 256, (3, 64), dtype=torch.uint8, device=device)
    with torch.no_grad():
        _, waveform = enc(bytes_in)
    assert torch.all(waveform[:, 0] == 0.0)
    assert torch.all(waveform[:, 1:] >= 0.0)  # norms are non-negative


def test_encoder_is_deterministic(device: str) -> None:
    bytes_in = torch.randint(0, 256, (2, 96), dtype=torch.uint8, device=device)
    torch.manual_seed(7)
    enc_a = ByteEncoder(embed_dim=48, num_blocks=2).to(device).eval()
    torch.manual_seed(7)
    enc_b = ByteEncoder(embed_dim=48, num_blocks=2).to(device).eval()
    with torch.no_grad():
        traj_a, _ = enc_a(bytes_in)
        traj_b, _ = enc_b(bytes_in)
    assert torch.allclose(traj_a, traj_b)
