"""Test Gate 3.2 - training loop + ROCm-tuned mixed precision.

Verifies a step actually updates parameters, the fp16 GradScaler is active and
the bf16 path runs without one, and — the headline — the real pipeline trains:
the loss drops over steps on a fixed batch and a multi-step run stays finite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.losses import THCMLoss
from thcm.training.trainer import StepStats, THCMTrainer, TrainConfig
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def report():
    return preflight()


@pytest.fixture(scope="module")
def device(report) -> str:
    return report.device_str


def _trainer(device: str, precision: str, d: int = 32, layers: int = 2,
             lr: float = 1e-3) -> THCMTrainer:
    return THCMTrainer(
        encoder=ByteEncoder(embed_dim=d, num_blocks=2, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=1.0),
        decoder=ConceptDecoder(embed_dim=d, num_heads=4, num_layers=layers),
        loss_fn=THCMLoss(embed_dim=d),
        device=device,
        config=TrainConfig(precision=precision, lr=lr),
    )


def _batch(device: str, b: int = 4, length: int = 128) -> torch.Tensor:
    return torch.randint(0, 256, (b, length), dtype=torch.uint8, device=device)


def test_step_updates_parameters_fp32(device: str) -> None:
    torch.manual_seed(0)
    tr = _trainer(device, "fp32")
    before = tr.encoder.stem.weight.detach().clone()
    stats = tr.step(_batch(device))
    after = tr.encoder.stem.weight.detach()

    assert isinstance(stats, StepStats)
    assert all(map(torch.isfinite, [torch.tensor(stats.total),
                                    torch.tensor(stats.next_concept),
                                    torch.tensor(stats.contrastive)]))
    assert stats.scale == 1.0 and not stats.skipped     # no scaler in fp32
    assert not torch.allclose(before, after)            # weights actually moved


def test_fp16_gradscaler_active(device: str, report) -> None:
    if not report.accelerated:
        pytest.skip("fp16 GradScaler path requires an accelerated backend")
    torch.manual_seed(1)
    tr = _trainer(device, "fp16")
    assert tr.scaler.is_enabled()
    stats = tr.step(_batch(device))
    assert stats.scale >= 2.0 ** 10                     # large loss scale in effect
    # Parameters stay finite after a scaled step.
    assert all(torch.isfinite(p).all() for p in tr.params)


def test_bf16_runs_without_scaler(device: str, report) -> None:
    if not report.accelerated:
        pytest.skip("bf16 autocast requires an accelerated backend")
    torch.manual_seed(2)
    tr = _trainer(device, "bf16")
    assert not tr.scaler.is_enabled()                   # bf16 needs no scaling
    stats = tr.step(_batch(device))
    assert stats.scale == 1.0
    assert all(torch.isfinite(p).all() for p in tr.params)


def test_loss_decreases_on_fixed_batch(device: str) -> None:
    """The real pipeline learns: overfitting one batch drives the loss down."""
    torch.manual_seed(3)
    tr = _trainer(device, "fp32", lr=2e-3)
    fixed = _batch(device)
    history = tr.fit((fixed for _ in range(60)), max_steps=60)

    totals = [s.total for s in history]
    start = sum(totals[:3]) / 3                          # smooth over DEP jitter
    best_late = min(totals[-10:])
    assert best_late < start - 0.3, f"loss did not drop: {start:.3f} -> {best_late:.3f}"
    assert all(s.total == s.total for s in history)      # no NaN anywhere


def test_multi_step_run_is_finite(device: str) -> None:
    torch.manual_seed(4)
    tr = _trainer(device, "fp32")
    history = tr.fit((_batch(device) for _ in range(10)), max_steps=10)
    assert len(history) == 10
    for s in history:
        assert s.total == s.total and abs(s.total) < 1e6
        assert s.grad_norm == s.grad_norm               # finite (not NaN)
