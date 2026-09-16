"""Image-quality metrics, including a Fourier Ring Correlation implementation.

PSNR and SSIM answer "how close is this to the truth". FRC answers "what is the
resolution, in nanometres", which is the number a microscopist actually wants.

scikit-image has no FRC, so it is implemented here.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from deconv.optics import Optics


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().float().numpy()


def psnr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB, averaged over a batch.

    Identical images are handled explicitly: skimage would divide by a zero MSE
    and emit a warning on its way to the mathematically correct infinity.
    """
    est, ref = _to_numpy(estimate), _to_numpy(reference)
    if est.ndim == 2:
        est, ref = est[None], ref[None]

    values = []
    for e, r in zip(est, ref, strict=True):
        if np.array_equal(e, r):
            values.append(np.inf)
        else:
            values.append(peak_signal_noise_ratio(r, e, data_range=1.0))
    return float(np.mean(values))


def ssim(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Structural similarity, averaged over a batch."""
    est, ref = _to_numpy(estimate), _to_numpy(reference)
    if est.ndim == 2:
        est, ref = est[None], ref[None]
    return float(
        np.mean(
            [structural_similarity(r, e, data_range=1.0) for e, r in zip(est, ref, strict=True)]
        )
    )


def frc(image_a: torch.Tensor, image_b: torch.Tensor, n_bins: int | None = None) -> torch.Tensor:
    """Fourier Ring Correlation between two images.

    FRC compares two *independent* measurements of the same scene and reports the
    correlation as a function of spatial frequency. Where it falls below a
    threshold, the two no longer agree, which is the resolution limit.

    Important: the two inputs must be independent noise realisations of the same
    object. Correlating a reconstruction against its own ground truth is not an
    FRC and does not measure resolution.

    Equally important, and easy to get wrong: FRC measures the **noise-limited**
    resolution. Where two noise draws decorrelate is set by the photon budget,
    not by how accurate the PSF used to reconstruct them was. Deconvolving with
    the true PSF and with a deliberately wrong one yields the same FRC
    resolution. To judge PSF accuracy, compare against ground truth with
    :func:`psnr` or :func:`ssim` instead.
    """
    if image_a.shape != image_b.shape:
        raise ValueError("FRC requires two images of the same shape")

    a = torch.fft.fftshift(torch.fft.fft2(image_a.float()), dim=(-2, -1))
    b = torch.fft.fftshift(torch.fft.fft2(image_b.float()), dim=(-2, -1))

    size = a.shape[-1]
    coords = torch.arange(size, device=a.device, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(xx * xx + yy * yy).round().long().flatten()

    n_bins = n_bins or size // 2
    radius = radius.clamp(max=n_bins - 1)

    cross = (a * b.conj()).flatten()
    power_a = (a.real**2 + a.imag**2).flatten()
    power_b = (b.real**2 + b.imag**2).flatten()

    num = torch.zeros(n_bins, dtype=torch.float32, device=a.device)
    den_a = torch.zeros(n_bins, dtype=torch.float32, device=a.device)
    den_b = torch.zeros(n_bins, dtype=torch.float32, device=a.device)

    num.index_add_(0, radius, cross.real)
    den_a.index_add_(0, radius, power_a)
    den_b.index_add_(0, radius, power_b)

    return num / torch.sqrt(den_a * den_b).clamp(min=1e-12)


def frc_resolution_nm(curve: torch.Tensor, optics: Optics, threshold: float = 1.0 / 7.0) -> float:
    """Resolution in nm where an FRC curve first drops below ``threshold``.

    The 1/7 threshold is the common convention in single-molecule and
    super-resolution microscopy.
    """
    below = (curve < threshold).nonzero()
    if below.numel() == 0:
        return float(2.0 * optics.pixel_nm)  # resolved to the sampling limit

    bin_index = int(below[0].item())
    if bin_index == 0:
        return float("inf")

    size = 2 * len(curve)
    frequency_per_px = bin_index / size / optics.pixel_nm
    return float(1.0 / frequency_per_px)


def frequency_axis_per_um(curve: torch.Tensor, optics: Optics) -> torch.Tensor:
    """Spatial-frequency axis for an FRC curve, in inverse micrometres."""
    size = 2 * len(curve)
    bins = torch.arange(len(curve), dtype=torch.float32)
    return bins / size / optics.pixel_nm * 1000.0
