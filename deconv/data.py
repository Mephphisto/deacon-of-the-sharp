"""Synthetic objects, the noisy forward model, and real-image loading.

Everything runs on the GPU inside the training loop: there is no dataset on disk
and no DataLoader. The forward model is only FFTs, so synthesising a fresh batch
is cheaper than reading one, and the CPU never becomes the bottleneck.

Real images come from ``skimage.data``. They are *not* bundled with the library
and are fetched via ``pooch`` on first use, then cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from deconv.optics import N_COEFFS, Optics, center_crop, convolve_fft, otf_from_psf, psf_from_coeffs

#: Coefficient sampling range in waves. Defocus (index 0, Noll j=4) is sampled
#: non-negative: a single in-focus intensity PSF cannot resolve its sign, so
#: allowing both would put an irreducible floor on the loss.
COEFF_RANGE: float = 0.5

#: Real images reserved for the transfer test. These must never enter training.
HELD_OUT: tuple[str, ...] = ("human_mitosis", "kidney")

#: Real images allowed in training (Run 3 only).
TRAIN_REAL: tuple[str, ...] = ("cells3d",)


@dataclass(frozen=True)
class NoiseConfig:
    """Randomised per sample, so the model cannot key on a fixed noise level."""

    min_photons: float = 50.0
    max_photons: float = 2000.0
    min_read_sigma: float = 1.0
    max_read_sigma: float = 5.0


def sample_coeffs(
    batch: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    coeff_range: float = COEFF_RANGE,
) -> torch.Tensor:
    """Uniform Zernike coefficients in ``[-range, +range]``, with defocus >= 0."""
    coeffs = (
        torch.rand(batch, N_COEFFS, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0
    ) * coeff_range
    coeffs[:, 0] = coeffs[:, 0].abs()
    return coeffs


def normalise_coeffs(coeffs: torch.Tensor, coeff_range: float = COEFF_RANGE) -> torch.Tensor:
    """Scale coefficients to ~unit range so every mode contributes equally to the loss.

    Without this, whichever mode happens to have the largest numeric spread
    dominates the gradient and the others train poorly.
    """
    return coeffs / coeff_range


def denormalise_coeffs(coeffs: torch.Tensor, coeff_range: float = COEFF_RANGE) -> torch.Tensor:
    return coeffs * coeff_range


# --------------------------------------------------------------------------
# Synthetic objects
# --------------------------------------------------------------------------


def make_beads(
    batch: int,
    size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    min_beads: int = 30,
    max_beads: int = 150,
) -> torch.Tensor:
    """Sparse sub-diffraction point emitters.

    A point source *is* the PSF, which makes aberration directly visible. Beads
    are placed at integer pixels with log-uniform intensity over about a decade.
    """
    images = torch.zeros(batch, size, size, device=device)
    counts = torch.randint(min_beads, max_beads + 1, (batch,), generator=generator, device=device)

    for b in range(batch):
        n = int(counts[b].item())
        ys = torch.randint(0, size, (n,), generator=generator, device=device)
        xs = torch.randint(0, size, (n,), generator=generator, device=device)
        amps = 10.0 ** torch.rand(n, generator=generator, device=device)
        images[b].index_put_((ys, xs), amps, accumulate=True)

    return images


def make_extended(
    batch: int,
    size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    max_blobs: int = 12,
    max_filaments: int = 8,
    filament_steps: int = 60,
) -> torch.Tensor:
    """Filaments, blobs and discs - crude mimics of cytoskeleton and nuclei.

    Beads alone are a poor proxy for biology: sparse impulses have completely
    different image statistics from dense extended texture, and a bead-only model
    transfers badly. These cost nothing to generate and close much of that gap.

    Fully vectorised over the batch. A per-image Python implementation runs ~100x
    slower and, on GPU, forces a host sync per scalar read, which would leave the
    GPU idle and make data synthesis dominate the training budget.
    """
    images = torch.zeros(batch, size, size, device=device)

    coords = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, device=device)

    # --- blobs: one vectorised pass per blob slot, batched over images ---
    centres = rand(batch, max_blobs, 2) * size
    radii = 2.0 + rand(batch, max_blobs) * 8.0
    amps = 0.3 + rand(batch, max_blobs)

    n_blobs = torch.randint(3, max_blobs + 1, (batch, 1), generator=generator, device=device)
    amps = amps * (torch.arange(max_blobs, device=device)[None, :] < n_blobs)

    for k in range(max_blobs):
        cy = centres[:, k, 0].view(-1, 1, 1)
        cx = centres[:, k, 1].view(-1, 1, 1)
        radius = radii[:, k].view(-1, 1, 1)
        amp = amps[:, k].view(-1, 1, 1)
        r2 = (yy[None] - cy) ** 2 + (xx[None] - cx) ** 2
        images += amp * torch.exp(-r2 / (2.0 * radius**2))

    # --- filaments: random walks stepped in parallel across batch and filament ---
    n_fil = torch.randint(2, max_filaments + 1, (batch, 1), generator=generator, device=device)
    active = torch.arange(max_filaments, device=device)[None, :] < n_fil

    y = rand(batch, max_filaments) * size
    x = rand(batch, max_filaments) * size
    angle = rand(batch, max_filaments) * (2.0 * np.pi)
    amp = (0.5 + rand(batch, max_filaments)) * active

    batch_index = (
        torch.arange(batch, device=device).view(-1, 1).expand(batch, max_filaments).reshape(-1)
    )
    flat_amp = amp.reshape(-1)

    for _ in range(filament_steps):
        angle = angle + (rand(batch, max_filaments) - 0.5) * 0.4
        y = (y + torch.sin(angle)) % size
        x = (x + torch.cos(angle)) % size
        images.index_put_(
            (batch_index, y.long().reshape(-1), x.long().reshape(-1)),
            flat_amp,
            accumulate=True,
        )

    return images


def make_objects(
    batch: int,
    size: int,
    bead_fraction: float = 0.5,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    real_pool: torch.Tensor | None = None,
    real_fraction: float = 0.0,
) -> torch.Tensor:
    """Mixed object batch: beads, extended structures, and optionally real crops.

    ``real_pool`` is only supplied for Run 3. Runs 1 and 2 train fully
    synthetically so the sim-to-real gap can be measured rather than hidden.
    """
    n_real = int(round(batch * real_fraction)) if real_pool is not None else 0
    n_synth = batch - n_real
    n_beads = int(round(n_synth * bead_fraction))
    n_ext = n_synth - n_beads

    parts: list[torch.Tensor] = []
    if n_beads > 0:
        parts.append(make_beads(n_beads, size, generator=generator, device=device))
    if n_ext > 0:
        parts.append(make_extended(n_ext, size, generator=generator, device=device))
    if n_real > 0 and real_pool is not None:
        parts.append(random_crops(real_pool, n_real, size, generator=generator, device=device))

    objects = torch.cat(parts, dim=0)
    perm = torch.randperm(batch, generator=generator, device=device)
    return objects[perm]


def random_crops(
    pool: torch.Tensor,
    batch: int,
    size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random ``size x size`` crops from a stack of real images."""
    n, height, width = pool.shape
    if height < size or width < size:
        raise ValueError(f"pool images {height}x{width} smaller than crop {size}")

    idx = torch.randint(0, n, (batch,), generator=generator, device=pool.device)
    tops = torch.randint(0, height - size + 1, (batch,), generator=generator, device=pool.device)
    lefts = torch.randint(0, width - size + 1, (batch,), generator=generator, device=pool.device)

    crops = torch.stack(
        [
            pool[
                int(idx[b]),
                int(tops[b]) : int(tops[b]) + size,
                int(lefts[b]) : int(lefts[b]) + size,
            ]
            for b in range(batch)
        ]
    )
    return crops.to(device) if device is not None else crops


# --------------------------------------------------------------------------
# Forward model
# --------------------------------------------------------------------------


def add_noise(
    image: torch.Tensor,
    peak_photons: torch.Tensor | float,
    read_sigma: torch.Tensor | float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Scale to photon counts, apply Poisson shot noise, then Gaussian read noise.

    Poisson must act on photon counts, not on a normalised image, or the noise
    statistics are meaningless.
    """
    peak = image.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    if isinstance(peak_photons, torch.Tensor):
        peak_photons = peak_photons.reshape(-1, 1, 1)
    photons = image / peak * peak_photons

    noisy = torch.poisson(photons.clamp(min=0.0), generator=generator)

    if isinstance(read_sigma, torch.Tensor):
        read_sigma = read_sigma.reshape(-1, 1, 1)
    noisy = noisy + torch.randn(noisy.shape, generator=generator, device=noisy.device) * read_sigma
    return noisy.clamp(min=0.0)


def normalise_image(image: torch.Tensor) -> torch.Tensor:
    """Per-image scaling to [0, 1]."""
    lo = image.amin(dim=(-2, -1), keepdim=True)
    hi = image.amax(dim=(-2, -1), keepdim=True)
    return (image - lo) / (hi - lo).clamp(min=1e-8)


def log_power_spectrum(image: torch.Tensor) -> torch.Tensor:
    """``log(1 + |FFT|^2)``, fftshifted and per-image standardised.

    The OTF's signature - especially its zero rings - is far more directly
    visible here than in the image. The DC term dominates by orders of
    magnitude, hence the log.
    """
    spectrum = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))
    power = torch.log1p(spectrum.real**2 + spectrum.imag**2)
    mean = power.mean(dim=(-2, -1), keepdim=True)
    std = power.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return (power - mean) / std


def synth_batch(
    batch: int,
    size: int,
    optics: Optics,
    basis: torch.Tensor,
    mask: torch.Tensor,
    noise: NoiseConfig | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    bead_fraction: float = 0.5,
    real_pool: torch.Tensor | None = None,
    real_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One training batch: ``(blurred_normalised, coeffs, clean_object)``.

    The returned coefficients are the *raw* values in waves; normalise them for
    the loss with :func:`normalise_coeffs`.
    """
    noise = noise or NoiseConfig()

    coeffs = sample_coeffs(batch, generator=generator, device=device, dtype=basis.dtype)
    psf = center_crop(psf_from_coeffs(coeffs, basis, mask), size)
    otf = otf_from_psf(psf)

    objects = make_objects(
        batch,
        size,
        bead_fraction=bead_fraction,
        generator=generator,
        device=device,
        real_pool=real_pool,
        real_fraction=real_fraction,
    )

    blurred = convolve_fft(objects, otf)

    photons = torch.empty(batch, device=device).uniform_(
        noise.min_photons, noise.max_photons, generator=generator
    )
    sigma = torch.empty(batch, device=device).uniform_(
        noise.min_read_sigma, noise.max_read_sigma, generator=generator
    )
    noisy = add_noise(blurred, photons, sigma, generator=generator)

    return normalise_image(noisy), coeffs, objects


# --------------------------------------------------------------------------
# Real images
# --------------------------------------------------------------------------


def load_real_slices(name: str) -> torch.Tensor:
    """Load a real microscopy stack from ``skimage.data`` as normalised 2D slices.

    Returns ``(n_slices, height, width)`` in ``[0, 1]``. Requires ``pooch``; the
    download happens once and is cached by scikit-image.

    Note: these images are *already blurred* by the microscope that acquired
    them. Treating them as sharp ground truth means the synthetic PSF is an
    additional blur on top of an unknown existing one.
    """
    from skimage import data

    if name == "cells3d":
        stack = data.cells3d()  # (z, c, y, x)
        slices = stack.reshape(-1, stack.shape[-2], stack.shape[-1])
    elif name == "kidney":
        stack = data.kidney()  # (z, y, x, c)
        slices = np.moveaxis(stack, -1, 1).reshape(-1, stack.shape[1], stack.shape[2])
    elif name == "human_mitosis":
        slices = data.human_mitosis()[None]
    elif name == "cell":
        slices = data.cell()[None]
    else:
        raise ValueError(f"unknown dataset {name!r}")

    tensor = torch.from_numpy(np.ascontiguousarray(slices)).to(torch.float32)
    return normalise_image(tensor)


def load_training_pool() -> torch.Tensor:
    """Real images permitted in training (Run 3 only)."""
    return torch.cat([load_real_slices(name) for name in TRAIN_REAL], dim=0)


def load_held_out(name: str) -> torch.Tensor:
    """Real images reserved for the transfer test."""
    if name not in HELD_OUT:
        raise ValueError(f"{name!r} is not a held-out dataset; expected one of {HELD_OUT}")
    return load_real_slices(name)
