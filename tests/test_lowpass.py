"""Test Gate 1.3 - low-pass filter & typo stress test.

Gate: injecting intentional typos into a string preserves waveform cosine
similarity above 0.85 against the clean profile. We additionally show the
low-pass filter is non-vacuous (it measurably helps under heavier corruption,
and the raw metric is responsive to corruption).
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from thcm.models.encoder import ByteEncoder
from thcm.signal.lowpass import (
    cosine_similarity,
    gaussian_kernel,
    gaussian_lowpass,
    moving_average,
)
from thcm.utils.device import preflight

GATE = 0.85
SIGMA = 2.5
CLEAN = b"The quick brown fox jumps over the lazy dog while the engine ingests raw bytes."


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


@pytest.fixture(scope="module")
def encoder(device: str) -> ByteEncoder:
    torch.manual_seed(0)
    return ByteEncoder(embed_dim=64, num_blocks=3, kernel_size=5).to(device).eval()


def _waveform(encoder: ByteEncoder, s: bytes, device: str) -> torch.Tensor:
    t = torch.tensor(list(s), dtype=torch.uint8, device=device).unsqueeze(0)
    with torch.no_grad():
        _, w = encoder(t)
    return w


def _corrupt(s: bytes, n: int, seed: int) -> bytes:
    rnd = random.Random(seed)
    b = bytearray(s)
    for i in rnd.sample(range(len(b)), n):
        b[i] = rnd.randint(97, 122)  # random lowercase substitution
    return bytes(b)


# --- filter unit properties -------------------------------------------------

def test_gaussian_kernel_normalized() -> None:
    k = gaussian_kernel(2.0)
    assert k.shape[0] % 2 == 1
    assert torch.isclose(k.sum(), torch.tensor(1.0))


def test_filters_preserve_length_and_dc(device: str) -> None:
    x = torch.ones(2, 50, device=device) * 3.0  # constant signal
    for smoothed in (gaussian_lowpass(x, SIGMA), moving_average(x, 7)):
        assert smoothed.shape == x.shape
        # A low-pass filter must pass a DC (constant) signal through unchanged.
        assert torch.allclose(smoothed, x, atol=1e-5)


# --- Test Gate 1.3: typo resilience ----------------------------------------

def test_single_typo_preserves_similarity(encoder: ByteEncoder, device: str) -> None:
    wc = gaussian_lowpass(_waveform(encoder, CLEAN, device), SIGMA)
    for seed in range(10):
        wt = gaussian_lowpass(_waveform(encoder, _corrupt(CLEAN, 1, seed), device), SIGMA)
        assert cosine_similarity(wc, wt).item() >= GATE


def test_multi_typo_preserves_similarity(encoder: ByteEncoder, device: str) -> None:
    wc_raw = _waveform(encoder, CLEAN, device)
    wc = gaussian_lowpass(wc_raw, SIGMA)
    worst = 1.0
    for seed in range(20):
        wt = gaussian_lowpass(_waveform(encoder, _corrupt(CLEAN, 12, seed), device), SIGMA)
        worst = min(worst, cosine_similarity(wc, wt).item())
    assert worst >= GATE, f"worst smoothed cosine {worst:.3f} < {GATE}"


def test_smoothing_is_non_vacuous(encoder: ByteEncoder, device: str) -> None:
    """Filter must (a) actually be responsive and (b) not degrade similarity."""
    wc_raw = _waveform(encoder, CLEAN, device)
    wc_sm = gaussian_lowpass(wc_raw, SIGMA)
    raw_sims, sm_sims = [], []
    for seed in range(20):
        wt_raw = _waveform(encoder, _corrupt(CLEAN, 12, seed), device)
        raw_sims.append(cosine_similarity(wc_raw, wt_raw).item())
        sm_sims.append(cosine_similarity(wc_sm, gaussian_lowpass(wt_raw, SIGMA)).item())
    raw_mean = sum(raw_sims) / len(raw_sims)
    sm_mean = sum(sm_sims) / len(sm_sims)
    # (a) corruption actually moves the raw metric (not a vacuous 1.000 test).
    assert min(raw_sims) < 0.999, "corruption did not perturb the waveform"
    # (b) smoothing does not hurt and on average helps under heavy corruption.
    assert sm_mean >= raw_mean
