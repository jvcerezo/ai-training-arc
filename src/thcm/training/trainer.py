"""Sprint 3.2 - Training loop with ROCm-tuned mixed precision.

Wires the full differentiable path into one optimization step:

    bytes (B, L) uint8
      -> ByteEncoder        -> trajectory (B, L, D), waveform (B, L)
      -> DynamicEntropyPatcher -> PatchedBatch (concept vectors)
      -> ConceptDecoder     -> contextualized buffer (B, P, D)
      -> THCMLoss           -> InfoNCE total
      -> scaler.scale(loss).backward(); clip; scaler.step; scaler.update()

Mixed precision is tuned for the native-Windows ROCm backend (exposed as the
"cuda" device):

  * fp16 needs a GradScaler. Half precision has only ~10 exponent bits, so small
    gradients underflow to zero; the scaler multiplies the loss by a large factor
    before backward and divides it out before the step, dynamically backing the
    factor off whenever an inf/NaN gradient is detected. `init_scale` /
    `growth_interval` are the knobs.
  * bf16 needs NO scaler. It keeps fp32's 8 exponent bits (just fewer mantissa
    bits), so gradients stay in range — enabling the scaler would be wasted work.
    On RDNA4 this is the simpler, recommended default.
  * fp32 is the deterministic fallback (and the only path on a CPU box).

Clipping uses clip_grad_norm_ on UNSCALED gradients (scaler.unscale_ first), so
the clip threshold is in real gradient units regardless of the loss scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from thcm.models.encoder import ByteEncoder
from thcm.models.patcher import DynamicEntropyPatcher
from thcm.models.transformer import ConceptDecoder
from thcm.training.losses import THCMLoss

_AMP_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}


def enable_speed() -> None:
    """Turn on backend autotuning for the training path (call once at startup).

    The encoder feeds the conv stack a fixed (B, L) input shape, so cuDNN/MIOpen
    benchmark mode can pick the fastest algorithm once and reuse it; tf32 matmuls
    speed up the Transformer. Kept out of THCMTrainer.__init__ so the test suite
    (many small, shape-varying models) stays deterministic.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer + mixed-precision hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: str = "bf16"          # "fp32" | "fp16" | "bf16"
    init_scale: float = 2.0 ** 16    # fp16 GradScaler starting loss scale
    growth_interval: int = 2000      # fp16 GradScaler scale-up cadence


@dataclass
class StepStats:
    """Per-step telemetry for logging / the training-loop gate."""

    total: float
    next_concept: float
    contrastive: float
    grad_norm: float
    scale: float           # active GradScaler scale (1.0 when no scaler)
    skipped: bool          # True if the scaler vetoed the step (inf/NaN grads)


class THCMTrainer:
    """Owns the model stack + optimizer + GradScaler and runs one step at a time."""

    def __init__(
        self,
        encoder: ByteEncoder,
        patcher: DynamicEntropyPatcher,
        decoder: ConceptDecoder,
        loss_fn: THCMLoss,
        device: str,
        config: TrainConfig | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = device
        self.device_type = "cuda" if "cuda" in device else "cpu"

        self.encoder = encoder.to(device)
        self.patcher = patcher.to(device)
        self.decoder = decoder.to(device)
        self.loss_fn = loss_fn.to(device)

        # Mixed precision only on an accelerated device; fp32 everywhere on CPU.
        precision = self.config.precision
        if self.device_type != "cuda":
            precision = "fp32"
        self.precision = precision
        self.use_amp = precision in _AMP_DTYPE
        self.amp_dtype = _AMP_DTYPE.get(precision)
        # Scaler is meaningful ONLY for fp16 (bf16 keeps fp32 exponent range).
        self.scaler = torch.amp.GradScaler(
            device,
            init_scale=self.config.init_scale,
            growth_interval=self.config.growth_interval,
            enabled=(precision == "fp16"),
        )

        self.params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.loss_fn.parameters())
        )  # DEP has no learnable parameters.
        self.opt = torch.optim.AdamW(
            self.params, lr=self.config.lr, betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

    def _train_mode(self) -> None:
        self.encoder.train()
        self.decoder.train()
        self.loss_fn.train()

    def step(self, byte_batch: torch.Tensor) -> StepStats:
        """One forward/backward/optimizer step on a (B, L) uint8 batch."""
        assert byte_batch.dim() == 2, f"expected (B, L), got {tuple(byte_batch.shape)}"
        byte_batch = byte_batch.to(self.device, non_blocking=True)
        self._train_mode()
        self.opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype,
                            enabled=self.use_amp):
            trajectory, waveform = self.encoder(byte_batch)
            packed = self.patcher(trajectory, waveform)
            out = self.decoder(packed)
            # Concept targets are the patch vectors; keep the loss math in fp32.
            result = self.loss_fn(out.buffer.float(), packed.vectors.float(), packed.mask)

        self.scaler.scale(result.total).backward()
        # Unscale before clipping so grad_clip is in true gradient units (no-op
        # when the scaler is disabled).
        self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.config.grad_clip)

        scale_before = self.scaler.get_scale()
        self.scaler.step(self.opt)       # skips the step if grads held inf/NaN
        self.scaler.update()
        # A drop in scale after update() means the step was vetoed and backed off.
        skipped = self.scaler.get_scale() < scale_before

        return StepStats(
            total=float(result.total.detach()),
            next_concept=float(result.next_concept.detach()),
            contrastive=float(result.contrastive.detach()),
            grad_norm=float(grad_norm),
            scale=scale_before,
            skipped=bool(skipped),
        )

    def fit(self, byte_batches, max_steps: int) -> list[StepStats]:
        """Run up to `max_steps` optimization steps over an iterable of batches."""
        history: list[StepStats] = []
        for batch in byte_batches:
            history.append(self.step(batch))
            if len(history) >= max_steps:
                break
        return history
