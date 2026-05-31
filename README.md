# Tokenless T-HCM (Transformer-Holographic Concept-Manifold)

A novel, tokenless hybrid AI engine: ingests raw bytes, dynamically compresses
predictable patterns into variable-length semantic patches via real-time entropy
evaluation, processes localized sequences with a context-capped Transformer, and
hands off overflow context into a continuous Holographic Concept-Manifold (HRR +
circular convolution) for **O(1) steady-state memory** at scale.

## Architecture (high level)

```
Raw bytes (0-255)
  -> 1D CNN parallel ingestion        (B, L) uint8 -> (B, L, D) trajectory
  -> low-pass waveform smoothing       typo-resilient entropy curve
  -> Dynamic Entropy Patching (DEP)    variable-length Concept Vectors
  -> context-capped Transformer        local execution, hard window (2048)
  -> Holographic Accumulator (FFT/HRR) buffer saturation -> single global vector
  -> Hopfield / LSH clean-up           noise mitigation on retrieval
```

See `docs/` (added per phase) for the full blueprint.

## Status

| Phase / Sprint | Item | State |
|---|---|---|
| 1.1 | Byte Data Engineering (`dataloader.py`) | structural gate **GREEN** (CPU); latency gate **BLOCKED on GPU backend** |

## Setup

```bash
python -m pip install -e ".[dev]"     # numpy, scipy, pytest
python -m pip install torch           # backend-specific, see below
```

### GPU backend (AMD RX 9060 XT / RDNA4)

The target accelerator is AMD ROCm. **ROCm PyTorch wheels are Linux-only** — on
native Windows, PyPI's `torch` is CPU-only. Resolve one of:

- **WSL2 + ROCm** (recommended on this Windows host) — Ubuntu under WSL2, then
  `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`.
- **Native Linux** (dual-boot) — same ROCm index. Best raw performance.
- **DirectML** (`pip install torch-directml`) — native-Windows GPU path, but
  partial op coverage and lags Python versions.
- **CPU-only** — valid for Sprint 1.1 correctness; fails all latency/VRAM gates.

> RDNA4 (gfx12xx) ROCm support is recent — if the GPU isn't enumerated, set
> `HSA_OVERRIDE_GFX_VERSION` to a supported nearby target.

Always run the preflight first — a green test on a silent CPU fallback is a
**false pass**:

```bash
PYTHONPATH=src python src/thcm/utils/device.py
```

## Test

```bash
PYTHONPATH=src python -m pytest -q
```

## Branching

`main` (protected, per-Phase merges) <- `dev` (integration, all tests pass) <-
`feature/sprint-X.Y-*` (per-sprint work).
