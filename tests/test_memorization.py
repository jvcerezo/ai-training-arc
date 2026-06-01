"""Test Gate 3.3 - memorization check (overfit a tiny dataset, drive loss -> 0).

The capacity proof for the whole stack: a fixed tiny byte batch must be
memorizable end-to-end, driving the InfoNCE loss to ~0 and next-concept
retrieval accuracy to 100%. Also pins the accuracy metric itself with a
deterministic identity-head case so the headline test can't pass on a broken
metric.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.losses import THCMLoss
from thcm.training.trainer import THCMTrainer, TrainConfig
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _trainer(device: str, d: int = 64) -> THCMTrainer:
    return THCMTrainer(
        encoder=ByteEncoder(embed_dim=d, num_blocks=2, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=1.0),
        decoder=ConceptDecoder(embed_dim=d, num_heads=4, num_layers=2),
        loss_fn=THCMLoss(embed_dim=d),
        device=device,
        config=TrainConfig(precision="fp32", lr=3e-3),
    )


def _eval_accuracy(tr: THCMTrainer, batch: torch.Tensor) -> tuple[float, float]:
    tr.encoder.eval(); tr.decoder.eval(); tr.loss_fn.eval()
    with torch.no_grad():
        traj, wave = tr.encoder(batch)
        packed = tr.patcher(traj, wave)
        out = tr.decoder(packed)
        return tr.loss_fn.accuracy(out.buffer, packed.vectors, packed.mask)


def test_accuracy_metric_identity_heads(device: str) -> None:
    """With identity heads and contextual == concepts, every concept identifies
    itself (self acc = 1.0) but argmax(z_i) != z_{i+1}, so next acc is low. A
    deterministic sanity check on accuracy() independent of training."""
    torch.manual_seed(0)
    loss = THCMLoss(embed_dim=8).to(device)
    loss.predictor = nn.Identity()
    loss.projector = nn.Identity()
    z = torch.randn(1, 5, 8, device=device)               # distinct concepts
    mask = torch.ones(1, 5, dtype=torch.bool, device=device)
    next_acc, self_acc = loss.accuracy(z, z, mask)
    assert self_acc == 1.0          # each concept is its own nearest neighbour
    assert next_acc < 0.5           # z_i does not point at z_{i+1}


def test_overfits_tiny_dataset_to_zero(device: str) -> None:
    torch.manual_seed(0)
    tr = _trainer(device)
    batch = torch.randint(0, 256, (2, 64), dtype=torch.uint8, device=device)

    # Untrained baseline: retrieval is near chance, not already solved.
    pre_next, pre_self = _eval_accuracy(tr, batch)
    assert pre_next < 0.6 and pre_self < 0.6

    totals: list[float] = []
    best_next = best_self = 0.0
    for _ in range(150):
        s = tr.step(batch)
        totals.append(s.total)
        na, sa = _eval_accuracy(tr, batch)
        best_next, best_self = max(best_next, na), max(best_self, sa)

    assert totals[0] > 2.0                  # started genuinely untrained
    assert min(totals) < 0.05               # loss driven to ~zero
    assert best_next == 1.0                 # perfect next-concept retrieval
    assert best_self == 1.0                 # perfect self-identification
