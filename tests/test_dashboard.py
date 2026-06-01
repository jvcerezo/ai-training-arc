"""Gate for the training dashboard's data layer + metrics emission.

The HTTP serving is thin; the logic worth pinning is read_metrics — it must parse
the JSONL the trainer writes, tolerate a half-written trailing line, surface the
latest snapshot + best val, and detect checkpoints. Also confirms the autonomous
loop actually emits a metrics.jsonl the dashboard can read.
"""

from __future__ import annotations

import json

import pytest

from thcm.training.dashboard import read_log, read_metrics


def _write(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_read_metrics_empty_dir(tmp_path) -> None:
    snap = read_metrics(str(tmp_path))
    assert snap["status"]["step"] is None
    assert snap["status"]["n_evals"] == 0
    assert snap["checkpoints"]["best"]["exists"] is False
    assert snap["series"]["step"] == []


def test_read_metrics_parses_and_summarizes(tmp_path) -> None:
    _write(tmp_path / "metrics.jsonl", [
        {"t": 1.0, "event": "improve", "step": 10, "train_loss": 5.0, "val_loss": 4.8,
         "val_acc": 0.1, "lr": 3e-4, "best_val": 4.8, "steps_per_sec": 9.0},
        {"t": 2.0, "event": "improve", "step": 20, "train_loss": 4.2, "val_loss": 4.0,
         "val_acc": 0.2, "lr": 3e-4, "best_val": 4.0, "steps_per_sec": 9.5},
        {"t": 3.0, "event": "stall", "step": 30, "train_loss": 4.1, "val_loss": 4.3,
         "val_acc": 0.18, "lr": 3e-4, "best_val": 4.0, "steps_per_sec": 9.4},
    ])
    (tmp_path / "best.pt").write_bytes(b"x" * 16)

    snap = read_metrics(str(tmp_path))
    assert snap["status"]["step"] == 30                 # latest eval
    assert snap["status"]["last_event"] == "stall"
    assert snap["status"]["best_val"] == 4.0            # min over evals, not latest
    assert snap["status"]["n_evals"] == 3
    assert snap["series"]["val_loss"] == [4.8, 4.0, 4.3]
    assert snap["checkpoints"]["best"]["exists"] is True
    assert snap["checkpoints"]["best"]["size"] == 16


def test_read_metrics_exposes_progress_and_log(tmp_path) -> None:
    _write(tmp_path / "metrics.jsonl", [
        {"t": 100.0, "event": "start", "step": 0, "max_steps": 5000, "lr": 3e-4},
        {"t": 110.0, "event": "improve", "step": 500, "train_loss": 4.0, "val_loss": 3.9,
         "val_acc": 0.3, "lr": 3e-4, "best_val": 3.9, "steps_per_sec": 8.0},
    ])
    (tmp_path / "training.log").write_text("line one\nstep 500: val improved\n", encoding="utf-8")

    snap = read_metrics(str(tmp_path))
    assert snap["status"]["max_steps"] == 5000        # surfaced for the progress bar
    assert snap["status"]["step"] == 500              # latest step across all records
    assert snap["status"]["elapsed"] == 10.0          # last_t - first_t
    assert snap["status"]["last_event"] == "improve"
    assert "step 500: val improved" in snap["log"]    # log tail included in payload


def test_read_log_tail_and_missing(tmp_path) -> None:
    assert read_log(str(tmp_path)) == []              # no file -> empty
    (tmp_path / "training.log").write_text("\n".join(f"line {i}" for i in range(50)),
                                           encoding="utf-8")
    tail = read_log(str(tmp_path), max_lines=10)
    assert tail[-1] == "line 49" and len(tail) == 10


def test_read_metrics_tolerates_partial_last_line(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(path, [{"t": 1.0, "event": "improve", "step": 10, "val_loss": 4.8, "val_acc": 0.1}])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"t": 2.0, "step": 20, "val_los')   # truncated mid-write
    snap = read_metrics(str(tmp_path))
    assert snap["status"]["n_evals"] == 1               # the good record only
    assert snap["status"]["step"] == 10


def test_autotrain_emits_metrics_jsonl(tmp_path) -> None:
    """End-to-end: the autonomous loop writes a metrics file the dashboard reads."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from thcm.models.encoder import ByteEncoder
    from thcm.models.patcher import DynamicEntropyPatcher
    from thcm.models.transformer import ConceptDecoder
    from thcm.training.auto import AutoConfig, _make_metrics, autotrain
    from thcm.training.losses import THCMLoss
    from thcm.training.trainer import THCMTrainer, TrainConfig
    from thcm.utils.device import preflight

    device = preflight().device_str
    (np.frombuffer(b"the quick brown fox. " * 6000, dtype=np.uint8)).tofile(tmp_path / "c.bin")
    d = 32
    trainer = THCMTrainer(
        encoder=ByteEncoder(embed_dim=d, num_blocks=2, kernel_size=5),
        patcher=DynamicEntropyPatcher(threshold_k=1.0),
        decoder=ConceptDecoder(embed_dim=d, num_heads=4, num_layers=2),
        loss_fn=THCMLoss(embed_dim=d), device=device,
        config=TrainConfig(precision="fp32", lr=1e-3),
    )
    cfg = AutoConfig(max_steps=20, eval_interval=10, eval_batches=2, ckpt_interval=10,
                     patience=50, workers=0, ckpt_dir=str(tmp_path))
    autotrain(str(tmp_path / "c.bin"), device, trainer, cfg, batch_size=4, seq_len=64,
              resume=False, log=lambda m: None, metrics=_make_metrics(str(tmp_path)))

    snap = read_metrics(str(tmp_path))
    assert snap["status"]["n_evals"] >= 2               # two evals at steps 10, 20
    assert snap["status"]["best_val"] < float("inf")
    assert snap["checkpoints"]["best"]["exists"] is True
