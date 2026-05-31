"""Hardware preflight and device guard.

Run this BEFORE trusting any data/compute path. A green test on a silent CPU
fallback is a FALSE PASS — this module makes the backend explicit and loud.

Target hardware: AMD Radeon RX 9060 XT (RDNA4 / gfx12xx) via ROCm/HIP, with
DirectML and CPU as fallbacks. The correct backend is environment-dependent;
this preflight reports exactly what was resolved so we never guess.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceReport:
    backend: str          # "rocm" | "cuda" | "directml" | "cpu" | "none"
    device_str: str       # what to pass to .to(...)
    torch_version: str | None
    hip_version: str | None
    accelerated: bool     # True iff a real GPU path is active
    detail: str


def preflight() -> DeviceReport:
    """Resolve and report the active compute backend without raising."""
    try:
        import torch
    except ModuleNotFoundError:
        return DeviceReport(
            backend="none",
            device_str="cpu",
            torch_version=None,
            hip_version=None,
            accelerated=False,
            detail="PyTorch not installed. Resolve a backend wheel first "
                   "(see pyproject.toml notes).",
        )

    tv = torch.__version__
    hip = getattr(torch.version, "hip", None)

    # ROCm/CUDA path (HIP exposes AMD as 'cuda').
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        backend = "rocm" if hip else "cuda"
        return DeviceReport(
            backend=backend,
            device_str="cuda",
            torch_version=tv,
            hip_version=hip,
            accelerated=True,
            detail=f"GPU visible: {name} (hip={hip}).",
        )

    # DirectML fallback (common on Windows + AMD when ROCm is unavailable).
    try:
        import torch_directml  # type: ignore

        dml = torch_directml.device()
        return DeviceReport(
            backend="directml",
            device_str=str(dml),
            torch_version=tv,
            hip_version=hip,
            accelerated=True,
            detail="ROCm not available; using torch-directml.",
        )
    except ModuleNotFoundError:
        pass

    return DeviceReport(
        backend="cpu",
        device_str="cpu",
        torch_version=tv,
        hip_version=hip,
        accelerated=False,
        detail=f"No GPU backend resolved on {platform.system()}. "
               "Running on CPU - acceptable for Sprint 1.1 correctness, NOT "
               "for the latency gate.",
    )


def require_accelerated() -> str:
    """Return the device string, raising if no GPU path is active.

    Use in profiling/latency gates where a CPU fallback must be a hard failure.
    """
    report = preflight()
    if not report.accelerated:
        raise RuntimeError(f"GPU backend required but not available. {report.detail}")
    return report.device_str


if __name__ == "__main__":
    r = preflight()
    print("=== T-HCM Device Preflight ===")
    print(f"  backend       : {r.backend}")
    print(f"  device_str    : {r.device_str}")
    print(f"  torch_version : {r.torch_version}")
    print(f"  hip_version   : {r.hip_version}")
    print(f"  accelerated   : {r.accelerated}")
    print(f"  detail        : {r.detail}")
