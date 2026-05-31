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
| 1.2 | 1D CNN encoder + transition waveform (`encoder.py`) | shape-integrity gate **GREEN** (CPU) |
| 1.3 | Low-pass filter + typo stress test (`lowpass.py`) | typo cosine ≥ 0.85 gate **GREEN** (CPU) |
| 2.1 | Dynamic Entropy Patching slicing engine (`patcher.py`) | pack-vs-reference gate **GREEN** (GPU) |
| 2.2 | Context-capped causal Concept Decoder (`transformer.py`) | shape + causal/pad-mask gate **GREEN** (GPU) |

Phase-1 mathematical verification is complete (14/14). Phase-2 structural
integration is verified on the native-Windows ROCm GPU (24/24 tests total).
The Phase-4 VRAM gate is deferred to its sprint.

## Setup

```bash
python -m pip install -e ".[dev]"     # numpy, scipy, pytest
python -m pip install torch           # backend-specific, see below
```

### GPU backend (AMD RX 9060 XT / RDNA4) — native Windows ROCm

As of **ROCm 7.2.1**, AMD ships **native Windows PyTorch wheels** for the RX 9060
XT (RDNA4) — no WSL, no Linux required. This is the chosen backend.

Requirements:
- **Python 3.12** (the wheels are `cp312`; the default `python` here is 3.14 —
  use the `.venv` 3.12 environment for all GPU work).
- **AMD Adrenalin graphics driver >= 26.2.2** (required by the 7.2.1 release).

Install into a 3.12 venv (`repo.radeon.com`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
.\.venv\Scripts\python.exe -m pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
```

> ROCm/HIP exposes the AMD GPU through the `"cuda"` device string. If the GPU is
> not enumerated, confirm the Adrenalin driver is >= 26.2.2.

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
