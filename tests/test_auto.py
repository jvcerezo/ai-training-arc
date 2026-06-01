"""Gate for the autonomous trainer — the control logic that runs unsupervised.

Verifies the loop checkpoints its best model and records a validation history,
resumes from the last checkpoint and continues the step count, and actually stops
(converged) once the LR has decayed past the floor while stalled — so an
unattended run neither loses progress nor spins forever.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.auto import AutoConfig, autotrain
from thcm.training.losses import THCMLoss
from thcm.training.trainer import THCMTrainer, TrainConfig
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _corpus(path, n_bytes: int, structured: bool) -> str:
    if structured:
        pattern = (b"the quick brown fox jumps over the lazy dog. " * 64)
        data = (pattern * (n_bytes // len(pattern) + 1))[:n_bytes]
        np.frombuffer(data, dtype=np.uint8).tofile(path)
    else:
        np.random.RandomState(0).randint(0, 256, size=n_bytes, dtype=np.uint8).tofile(path)
    return str(path)


def _trainer(device: str, lr: float = 1e-3) -> THCMTrainer:
    d = 32
    return THCMTrainer(
        encoder=ByteEncoder(embed_dim=d, num_blocks=2, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=1.0),
        decoder=ConceptDecoder(embed_dim=d, num_heads=4, num_layers=2),
        loss_fn=THCMLoss(embed_dim=d),
        device=device,
        config=TrainConfig(precision="fp32", lr=lr),
    )


def test_autotrain_checkpoints_best_and_logs_history(device: str, tmp_path) -> None:
    torch.manual_seed(0)
    corpus = _corpus(tmp_path / "c.bin", 120_000, structured=True)
    ckpt_dir = tmp_path / "ck"
    cfg = AutoConfig(max_steps=30, eval_interval=10, eval_batches=2, ckpt_interval=10,
                     patience=50, workers=0, ckpt_dir=str(ckpt_dir))
    summary = autotrain(corpus, device, _trainer(device), cfg,
                        batch_size=4, seq_len=64, resume=False, log=lambda m: None)

    assert (ckpt_dir / "best.pt").exists()          # best model was saved
    assert (ckpt_dir / "last.pt").exists()          # crash-resume checkpoint
    assert len(summary.val_history) == 3            # one eval per eval_interval
    assert summary.best_val < float("inf")          # a real best was recorded
    assert summary.best_val <= summary.val_history[0][1]  # never worse than first eval


def test_resume_continues_step_count(device: str, tmp_path) -> None:
    torch.manual_seed(1)
    corpus = _corpus(tmp_path / "c.bin", 120_000, structured=True)
    ckpt_dir = tmp_path / "ck"
    cfg = AutoConfig(max_steps=20, eval_interval=10, eval_batches=2, ckpt_interval=10,
                     patience=50, workers=0, ckpt_dir=str(ckpt_dir))
    first = autotrain(corpus, device, _trainer(device), cfg,
                      batch_size=4, seq_len=64, resume=False, log=lambda m: None)
    assert first.steps == 20

    cfg2 = AutoConfig(max_steps=40, eval_interval=10, eval_batches=2, ckpt_interval=10,
                      patience=50, workers=0, ckpt_dir=str(ckpt_dir))
    resumed = autotrain(corpus, device, _trainer(device), cfg2,
                        batch_size=4, seq_len=64, resume=True, log=lambda m: None)
    assert resumed.steps == 40                      # continued from 20, not restarted


def test_heartbeat_emits_progress_between_evals(device: str, tmp_path) -> None:
    """Frequent heartbeats keep step/throughput flowing without waiting for an
    eval — what stops the dashboard from looking 'idle' mid-training."""
    from thcm.training.auto import _make_metrics
    from thcm.training.dashboard import read_metrics

    torch.manual_seed(5)
    corpus = _corpus(tmp_path / "c.bin", 120_000, structured=True)
    cfg = AutoConfig(max_steps=20, eval_interval=1000, ckpt_interval=1000,
                     heartbeat=5, patience=50, workers=0, ckpt_dir=str(tmp_path))
    autotrain(corpus, device, _trainer(device), cfg, batch_size=4, seq_len=64,
              resume=False, log=lambda m: None, metrics=_make_metrics(str(tmp_path)))

    snap = read_metrics(str(tmp_path))
    # No eval happened (interval 1000 > 20 steps), yet step advanced via heartbeats.
    assert snap["status"]["n_evals"] == 0
    assert snap["status"]["step"] == 20
    assert snap["status"]["steps_per_sec"] is not None


def test_plateau_drives_convergence_and_stops(device: str, tmp_path) -> None:
    """No real improvement + a high min_delta => LR decays past the floor and the
    run halts itself instead of training forever."""
    torch.manual_seed(2)
    corpus = _corpus(tmp_path / "c.bin", 120_000, structured=False)
    ckpt_dir = tmp_path / "ck"
    cfg = AutoConfig(max_steps=200, eval_interval=5, eval_batches=2, ckpt_interval=100,
                     patience=1, lr_decay=0.1, min_lr=5e-4, min_delta=1e3,
                     workers=0, ckpt_dir=str(ckpt_dir))
    summary = autotrain(corpus, device, _trainer(device, lr=1e-3), cfg,
                        batch_size=4, seq_len=64, resume=False, log=lambda m: None)

    assert summary.converged                        # stopped on its own
    assert summary.steps < cfg.max_steps            # well before the hard cap
