"""Fourier optics: Zernike wavefronts, pupils, PSFs, OTFs and regularized inverses.

Everything here is closed-form physics. No learning happens in this module.

Conventions:
  - Zernike indices follow **Noll (1976)**. Mixing this up with OSA/ANSI silently
    corrupts every label while training still looks healthy, so
    ``test_optics.py`` asserts orthonormality and per-mode symmetry.
  - Wavefronts are in **waves** (multiples of lambda), not radians.
  - PSFs are normalised to sum to 1, so ``OTF[0, 0] == 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

NOLL_INDICES: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11)
N_COEFFS: int = len(NOLL_INDICES)

NOLL_NAMES: dict[int, str] = {
    4: "defocus",
    5: "oblique astigmatism",
    6: "vertical astigmatism",
    7: "vertical coma",
    8: "horizontal coma",
    9: "vertical trefoil",
    10: "oblique trefoil",
    11: "primary spherical",
}

#: Azimuthal order m per Noll index. Modes with m == 0 are rotationally
#: symmetric; the rest rotate by m*theta under image rotation, which is why
#: geometric augmentation is not label-preserving.
NOLL_AZIMUTHAL: dict[int, int] = {4: 0, 5: 2, 6: 2, 7: 1, 8: 1, 9: 3, 10: 3, 11: 0}


@dataclass(frozen=True)
class Optics:
    """Optical configuration for a high-NA fluorescence microscope.

    Defaults are a realistic oil-immersion objective imaging GFP onto an sCMOS
    sensor at 100x (6.5 um / 100 = 65 nm pixels).
    """

    na: float = 1.4
    wavelength_nm: float = 520.0
    pixel_nm: float = 65.0
    grid: int = 256

    @property
    def pupil_radius_px(self) -> float:
        """Pupil radius in grid samples.

        The pupil edge sits at spatial frequency NA/lambda. Sampling the radius
        with R points gives df = NA/(lambda*R), and an N-point FFT satisfies
        dx*df = 1/N, hence R = N*dx*NA/lambda.
        """
        return self.grid * self.pixel_nm * self.na / self.wavelength_nm

    @property
    def airy_radius_nm(self) -> float:
        """Radius of the first dark ring of the Airy disc."""
        return 0.61 * self.wavelength_nm / self.na

    @property
    def airy_radius_px(self) -> float:
        return self.airy_radius_nm / self.pixel_nm

    @property
    def fwhm_nm(self) -> float:
        return 0.51 * self.wavelength_nm / self.na

    @property
    def fwhm_px(self) -> float:
        return self.fwhm_nm / self.pixel_nm

    @property
    def cutoff_per_um(self) -> float:
        """Incoherent OTF cutoff 2*NA/lambda, in inverse micrometres."""
        return 2000.0 * self.na / self.wavelength_nm

    @property
    def nyquist_pixel_nm(self) -> float:
        """Largest pixel that still samples the intensity PSF at Nyquist."""
        return self.wavelength_nm / (4.0 * self.na)

    @property
    def sampling_factor(self) -> float:
        """How far inside Nyquist we sit. Must be >= 1."""
        return self.nyquist_pixel_nm / self.pixel_nm


def zernike_basis(
    optics: Optics,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the Noll j=4..11 Zernike basis on the pupil grid.

    Returns ``(basis, mask)`` where ``basis`` has shape ``(8, grid, grid)`` and
    ``mask`` is the circular aperture. Modes are zero outside the aperture.

    Normalisation is Noll's: the mean square of each mode over the unit disc is
    1, so ``mean(Zi * Zj)`` over the aperture is the identity matrix.
    """
    n = optics.grid
    radius = optics.pupil_radius_px

    coords = torch.arange(n, device=device, dtype=dtype) - n // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    rho = torch.sqrt(xx * xx + yy * yy) / radius
    theta = torch.atan2(yy, xx)
    mask = (rho <= 1.0).to(dtype)

    r2 = rho * rho
    r3 = r2 * rho
    r4 = r2 * r2

    s3 = math.sqrt(3.0)
    s5 = math.sqrt(5.0)
    s6 = math.sqrt(6.0)
    s8 = math.sqrt(8.0)

    modes = [
        s3 * (2.0 * r2 - 1.0),  # j=4  defocus
        s6 * r2 * torch.sin(2.0 * theta),  # j=5  oblique astigmatism
        s6 * r2 * torch.cos(2.0 * theta),  # j=6  vertical astigmatism
        s8 * (3.0 * r3 - 2.0 * rho) * torch.sin(theta),  # j=7  vertical coma
        s8 * (3.0 * r3 - 2.0 * rho) * torch.cos(theta),  # j=8  horizontal coma
        s8 * r3 * torch.sin(3.0 * theta),  # j=9  vertical trefoil
        s8 * r3 * torch.cos(3.0 * theta),  # j=10 oblique trefoil
        s5 * (6.0 * r4 - 6.0 * r2 + 1.0),  # j=11 primary spherical
    ]

    basis = torch.stack(modes, dim=0) * mask
    return basis, mask


def pupil_from_coeffs(
    coeffs: torch.Tensor, basis: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Complex pupil ``P = A * exp(i*2*pi*W)`` from Zernike coefficients.

    ``coeffs`` has shape ``(batch, 8)`` in waves; returns ``(batch, grid, grid)``.
    """
    wavefront = (coeffs[..., None, None] * basis).sum(dim=-3)
    phase = 2.0 * math.pi * wavefront
    return mask * torch.complex(torch.cos(phase), torch.sin(phase))


def psf_from_pupil(pupil: torch.Tensor) -> torch.Tensor:
    """Intensity PSF ``|FFT(P)|^2``, normalised to sum to 1.

    The ifftshift/fftshift pair keeps a centred pupil mapping to a centred PSF.
    """
    shifted = torch.fft.ifftshift(pupil, dim=(-2, -1))
    amplitude = torch.fft.fftshift(torch.fft.fft2(shifted), dim=(-2, -1))
    psf = amplitude.real**2 + amplitude.imag**2
    return psf / psf.sum(dim=(-2, -1), keepdim=True)


def psf_from_coeffs(coeffs: torch.Tensor, basis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convenience: Zernike coefficients straight to a normalised intensity PSF."""
    return psf_from_pupil(pupil_from_coeffs(coeffs, basis, mask))


def center_crop(psf: torch.Tensor, size: int) -> torch.Tensor:
    """Centre-crop a PSF to ``size`` and renormalise.

    The PSF is computed on the pupil grid (typically 256) so the pupil is well
    sampled, then cropped to the image size. Aberrated PSFs stay within ~20 px
    of centre at the coefficient ranges used here, so nothing meaningful is lost.
    """
    n = psf.shape[-1]
    if size > n:
        raise ValueError(f"crop size {size} exceeds PSF grid {n}")
    start = n // 2 - size // 2
    cropped = psf[..., start : start + size, start : start + size]
    return cropped / cropped.sum(dim=(-2, -1), keepdim=True)


def otf_from_psf(psf: torch.Tensor) -> torch.Tensor:
    """Optical transfer function, unshifted so ``OTF[..., 0, 0] == 1``."""
    return torch.fft.fft2(torch.fft.ifftshift(psf, dim=(-2, -1)))


def convolve_fft(image: torch.Tensor, otf: torch.Tensor) -> torch.Tensor:
    """Circular convolution of an image with a PSF, given that PSF's OTF."""
    return torch.fft.ifft2(torch.fft.fft2(image) * otf).real


def wiener_filter(otf: torch.Tensor, reg: float) -> torch.Tensor:
    """Wiener restoration filter ``conj(H) / (|H|^2 + reg)``.

    ``reg`` plays the role of inverse SNR. It is what makes the inverse exist at
    all: no exact inverse kernel does, because the OTF vanishes identically above
    2*NA/lambda and has interior zeros when aberrated.
    """
    power = otf.real**2 + otf.imag**2
    return otf.conj() / (power + reg)


def apply_wiener(image: torch.Tensor, otf: torch.Tensor, reg: float) -> torch.Tensor:
    """Deconvolve with a regularized (Wiener) inverse."""
    return torch.fft.ifft2(torch.fft.fft2(image) * wiener_filter(otf, reg)).real


def naive_inverse(image: torch.Tensor, otf: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unregularized inverse filtering. Deliberately catastrophic.

    This exists to be plotted, not used: dividing by an OTF that is ~0 above the
    cutoff amplifies noise without bound. ``eps`` only prevents 0/0 producing
    NaN, which would break the figure rather than illustrate the point.
    """
    return torch.fft.ifft2(torch.fft.fft2(image) / (otf + eps)).real


def radial_profile(image: torch.Tensor) -> torch.Tensor:
    """Azimuthally averaged profile of a centred 2D image, indexed by radius in px."""
    n = image.shape[-1]
    coords = torch.arange(n, device=image.device, dtype=torch.float32) - n // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy).round().long().flatten()
    values = image.flatten().to(torch.float32)

    n_bins = int(r.max().item()) + 1
    total = torch.zeros(n_bins, device=image.device, dtype=torch.float32)
    count = torch.zeros(n_bins, device=image.device, dtype=torch.float32)
    total.index_add_(0, r, values)
    count.index_add_(0, r, torch.ones_like(values))
    return total / count.clamp(min=1.0)
