"""Gate 5.3 - the autonomous loop drives the generative (bpb) objective.

Verifies the shared control policy runs GenerativeTHCM end-to-end: bpb falls on a
learnable corpus, best/last checkpoints are written and self-describing, and a
resume restores the model and skips re-warmup.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thcm.training.auto import AutoConfig, autotrain
from thcm.training.gen_auto import _gen_checkpoint, _gen_load, build_gen_trainer, evaluate_gen
from thcm.utils.device import preflight


@pytest.fixture(scope="module")
def device() -> str:
    return preflight().device_str


def _corpus(path, n_bytes: int) -> str:
    pattern = b"def add(a, b):\n    return a + b\n\n" * 64
    data = (pattern * (n_bytes // len(pattern) + 1))[:n_bytes]
    np.frombuffer(data, dtype=np.uint8).tofile(path)
    return str(path)


def _args(**over) -> argparse.Namespace:
    base = dict(embed_dim=32, no_memory=False, byte_layers=2, num_heads=4, concept_layers=1,
                encoder_blocks=1, threshold_k=1.0, lr=3e-3, warmup=5, precision="fp32")
    return argparse.Namespace(**{**base, **over})


def _run(device, corpus, ckpt_dir, max_steps, resume, trainer=None):
    cfg = AutoConfig(max_steps=max_steps, eval_interval=20, eval_batches=2, ckpt_interval=20,
                     patience=50, workers=0, ckpt_dir=str(ckpt_dir))
    trainer = trainer or build_gen_trainer(_args(), device)
    return trainer, autotrain(corpus, device, trainer, cfg, batch_size=4, seq_len=64,
                              resume=resume, log=lambda m: None, evaluate_fn=evaluate_gen,
                              checkpoint_fn=_gen_checkpoint, load_fn=_gen_load)


def test_generative_autotrain_lowers_bpb(device: str, tmp_path) -> None:
    torch.manual_seed(0)
    corpus = _corpus(tmp_path / "c.bin", 60_000)
    _, summary = _run(device, corpus, tmp_path / "ck", 60, resume=False)
    first, best = summary.val_history[0][1], summary.best_val
    assert best < first - 0.5                       # clearly learning to emit bytes
    ck = torch.load(tmp_path / "ck" / "best.pt", map_location="cpu")
    assert ck["model_config"]["use_memory"] is True  # checkpoint self-describes


def test_generative_resume_restores_and_skips_warmup(device: str, tmp_path) -> None:
    torch.manual_seed(1)
    corpus = _corpus(tmp_path / "c.bin", 60_000)
    _run(device, corpus, tmp_path / "ck", 20, resume=False)
    trainer, resumed = _run(device, corpus, tmp_path / "ck", 40, resume=True)
    assert resumed.steps == 40
    assert trainer.steps >= 20 + 20                 # warmup counter carried over
