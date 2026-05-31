"""Sprint 2.3 - Holographic Accumulator (HRR via FFT circular convolution).

The context-capped decoder (Sprint 2.2) can only hold CONTEXT_CAP concept
vectors. To process arbitrarily long streams without unbounded memory, saturated
buffers are compressed into a SINGLE fixed-size global state vector using
Holographic Reduced Representations (Plate, 1995): circular convolution *binds*
a positional key to a concept, and superposition (sum) *bundles* the bindings.

    global_state = sum_t  key_t (x) concept_t          # (x) = circular conv

This is the engine's O(1) steady-state memory claim: no matter how many patches
have streamed, the state is one (B, D) complex vector. We keep the state in the
frequency domain (where convolution is a pointwise product), so accumulation is
just an additive update.

Why *unitary* positional keys. We index position with convolution powers of one
fixed key: key_t = base^(x)t. If `base` is unitary (every Fourier component has
unit modulus), then |fft(base)^t| = 1 for ALL t — the keys never decay or blow
up across an arbitrarily long stream, and single-item unbinding is EXACT
(corr(k, k (x) v) = v when |fft(k)| = 1). A generic random key would make the
powers drift in magnitude and the stream numerically unstable.

Shape contract:
    ConceptBuffer.buffer (B, P, D) real, .mask (B, P), .counts (B,)
      -> per-patch fft                       (B, P, D) complex
      -> bind positional key, mask, sum_P     (B, D)   complex   (the update)
      -> add into running state.freq          (B, D)   complex   (O(1))
    read(state): ifft(state.freq).real        (B, D)   real
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from thcm.models.transformer import ConceptBuffer


# --- HRR primitives ---------------------------------------------------------

def circular_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution a (x) b along the last dim (binding). FFT-accelerated."""
    assert a.shape[-1] == b.shape[-1], (a.shape, b.shape)
    n = a.shape[-1]
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1).real


def circular_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular correlation a (#) b (unbinding): recovers v from a=key, b=key(x)v.

    Correlation = convolution with the involution of `a`, i.e. conj in frequency.
    """
    assert a.shape[-1] == b.shape[-1], (a.shape, b.shape)
    fa = torch.fft.fft(a, dim=-1)
    fb = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(torch.conj(fa) * fb, dim=-1).real


def unitary_vector(dim: int, *, generator: torch.Generator | None = None,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """A unitary HRR vector: real, with every Fourier component of unit modulus.

    Built by assigning random phases under conjugate symmetry so the inverse FFT
    is real. Convolution powers of a unitary vector stay unit-modulus forever,
    which is what makes long-stream positional binding numerically stable.
    """
    assert dim % 2 == 0, f"dim must be even for the symmetry construction, got {dim}"
    half = dim // 2
    phases = torch.rand(half - 1, generator=generator, device=device, dtype=dtype) * (2 * torch.pi)
    spectrum = torch.ones(dim, device=device, dtype=torch.complex64)
    spectrum[0] = 1.0                                   # DC: real
    spectrum[half] = 1.0                                # Nyquist: real
    pos = torch.exp(1j * phases.to(torch.complex64))    # (half-1,)
    spectrum[1:half] = pos
    spectrum[half + 1:] = torch.conj(torch.flip(pos, dims=[0]))  # conjugate-symmetric
    vec = torch.fft.ifft(spectrum, dim=-1).real
    return vec.to(dtype)


# --- Streaming accumulator --------------------------------------------------

@dataclass(frozen=True)
class HoloState:
    """O(1) holographic memory: one frequency-domain vector + a position counter."""

    freq: torch.Tensor   # (B, D) complex - superposed bindings in the Fourier domain
    step: torch.Tensor   # (B,) long - absolute count of concepts folded in so far

    def nbytes(self) -> int:
        """Resident byte size of the state (the O(1) memory claim, measured)."""
        return self.freq.element_size() * self.freq.nelement() + \
            self.step.element_size() * self.step.nelement()


class HolographicAccumulator(nn.Module):
    """Folds streamed ConceptBuffers into a single fixed-size global state.

    Holds only the fixed unitary key (no per-stream state) so one instance can
    accumulate many independent streams. State is threaded functionally via
    HoloState, which keeps the O(1) property structurally explicit.
    """

    def __init__(self, embed_dim: int = 256, *, seed: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        gen = torch.Generator().manual_seed(seed)
        base = unitary_vector(embed_dim, generator=gen)
        # Cache the per-component phase of fft(base): key_t in freq = exp(i*t*phase).
        phase = torch.angle(torch.fft.fft(base, dim=-1))   # (D,) real
        self.register_buffer("base", base, persistent=True)
        self.register_buffer("phase", phase, persistent=True)

    def init_state(self, batch: int, *, device=None) -> HoloState:
        device = device or self.phase.device
        return HoloState(
            freq=torch.zeros(batch, self.embed_dim, dtype=torch.complex64, device=device),
            step=torch.zeros(batch, dtype=torch.long, device=device),
        )

    def accumulate(self, state: HoloState, buffer: ConceptBuffer) -> HoloState:
        """Bind each concept to its absolute-position key and superpose into state."""
        v, mask, counts = buffer.buffer, buffer.mask, buffer.counts
        assert v.dim() == 3 and v.shape[-1] == self.embed_dim, tuple(v.shape)
        b, p, _ = v.shape
        assert state.freq.shape == (b, self.embed_dim), state.freq.shape

        vf = torch.fft.fft(v, dim=-1)                                  # (B, P, D) complex
        # Absolute position of each slot: continues across chunks via state.step.
        t = state.step[:, None] + torch.arange(p, device=v.device)[None, :]  # (B, P) long
        # key_t in frequency = exp(i * t * phase); unit modulus for all t.
        kf = torch.exp(1j * (t[..., None].to(self.phase.dtype) * self.phase[None, None, :]))
        bound = (kf * vf) * mask[..., None]                            # zero padded slots
        delta = bound.sum(dim=1)                                       # (B, D) complex
        # O(1): state stays (B, D) no matter how long the stream is.
        return HoloState(freq=state.freq + delta, step=state.step + counts)

    def read(self, state: HoloState) -> torch.Tensor:
        """Decode the global state back to a real (B, D) concept-space vector."""
        g = torch.fft.ifft(state.freq, dim=-1).real
        assert g.shape == state.freq.shape
        return g

    def retrieve(self, state: HoloState, position: int) -> torch.Tensor:
        """Unbind the concept stored at absolute `position` (noisy if superposed)."""
        # key_position in time domain = ifft(exp(i*position*phase)).
        kf = torch.exp(1j * (position * self.phase))                   # (D,) complex
        key = torch.fft.ifft(kf, dim=-1).real                          # (D,)
        approx = circular_corr(key.expand_as(state.freq.real), torch.fft.ifft(state.freq, dim=-1).real)
        return approx
