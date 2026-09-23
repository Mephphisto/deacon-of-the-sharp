"""Tests for the Fourier-optics core.

These are the highest-value tests in the project. A Zernike convention error or
an FFT normalisation slip produces plausible-looking but wrong results that
survive all the way to the final figures, so the physics is pinned numerically
against closed-form values here.
"""

from __future__ import annotations

import math

import pytest
import torch

from deconv.optics import (
    N_COEFFS,
    NOLL_AZIMUTHAL,
    NOLL_INDICES,
    Optics,
    apply_wiener,
    center_crop,
    convolve_fft,
    naive_inverse,
    otf_from_psf,
    psf_from_coeffs,
    radial_profile,
    wiener_filter,
    zernike_basis,
)

OPTICS = Optics()
DTYPE = torch.float64  # tighter tolerances than float32 for the physics checks


@pytest.fixture(scope="module")
def basis_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    return zernike_basis(OPTICS, dtype=DTYPE)


def zeros_coeffs(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, N_COEFFS, dtype=DTYPE)


# --------------------------------------------------------------------------
# Optical configuration
# --------------------------------------------------------------------------


def test_pupil_radius_matches_derivation() -> None:
    """R = N * dx * NA / lambda."""
    expected = 256 * 65.0 * 1.4 / 520.0
    assert OPTICS.pupil_radius_px == pytest.approx(expected)
    assert OPTICS.pupil_radius_px == pytest.approx(44.8, abs=0.1)


def test_nyquist_is_satisfied() -> None:
    """The chosen 65 nm pixel must sample the intensity PSF at better than Nyquist."""
    assert OPTICS.pixel_nm < OPTICS.nyquist_pixel_nm
    assert OPTICS.sampling_factor > 1.0
    assert OPTICS.sampling_factor == pytest.approx(1.43, abs=0.02)


def test_psf_is_resolved_by_at_least_two_pixels() -> None:
    """RESEARCH.md requires >= 2-3 px across the FWHM."""
    assert OPTICS.fwhm_px >= 2.0
    assert OPTICS.fwhm_px == pytest.approx(2.91, abs=0.05)


def test_derived_optical_quantities() -> None:
    assert OPTICS.airy_radius_nm == pytest.approx(226.6, abs=0.5)
    assert OPTICS.cutoff_per_um == pytest.approx(5.385, abs=0.01)


# --------------------------------------------------------------------------
# Zernike basis - the convention guard
# --------------------------------------------------------------------------


def test_basis_shape_and_indices() -> None:
    basis, mask = zernike_basis(OPTICS, dtype=DTYPE)
    assert basis.shape == (N_COEFFS, OPTICS.grid, OPTICS.grid)
    assert mask.shape == (OPTICS.grid, OPTICS.grid)
    assert NOLL_INDICES == (4, 5, 6, 7, 8, 9, 10, 11)
    assert len(NOLL_AZIMUTHAL) == N_COEFFS


def test_zernike_orthonormality(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    """<Zi, Zj> = delta_ij over the unit disc.

    This is the single most important assertion in the project: it pins the Noll
    convention and the normalisation simultaneously. If it fails, every training
    label is wrong while the loss curve still looks healthy.
    """
    basis, mask = basis_and_mask
    n_inside = mask.sum()

    flat = basis.reshape(N_COEFFS, -1)
    gram = (flat @ flat.T) / n_inside

    # Tolerance is set by discretisation, not by correctness. The aperture is a
    # pixelated circle of radius ~45 px containing ~6300 samples, so the
    # continuous integral is approximated to O(1/R) ~ 1e-2. A genuine convention
    # error (wrong Noll ordering, wrong normalisation) produces O(1) deviations,
    # which this still catches by a wide margin.
    assert torch.allclose(gram, torch.eye(N_COEFFS, dtype=DTYPE), atol=1e-2)

    diagonal = gram.diag()
    off_diagonal = (gram - torch.diag(diagonal)).abs().max()
    assert off_diagonal < 0.05 * diagonal.min(), "modes are not mutually orthogonal"


def test_modes_are_zero_outside_aperture(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    basis, mask = basis_and_mask
    outside = basis * (1.0 - mask)
    assert torch.all(outside == 0.0)


@pytest.mark.parametrize("idx,noll", list(enumerate(NOLL_INDICES)))
def test_rotational_symmetry_matches_azimuthal_order(
    idx: int, noll: int, basis_and_mask: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """A mode with azimuthal order m is invariant under rotation by 2*pi/m.

    Verified by rotating the *sampling grid* rather than the image, which avoids
    interpolation error. m = 0 modes must be invariant under any rotation.
    """
    basis, mask = basis_and_mask
    m = NOLL_AZIMUTHAL[noll]
    mode = basis[idx]

    n = OPTICS.grid
    coords = torch.arange(n, dtype=DTYPE) - n // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rho = torch.sqrt(xx * xx + yy * yy) / OPTICS.pupil_radius_px
    theta = torch.atan2(yy, xx)
    inside = rho <= 1.0

    angle = 2.0 * math.pi / m if m > 0 else 1.0  # arbitrary angle for m == 0
    rotated = _evaluate_mode(noll, rho, theta + angle) * inside

    assert torch.allclose(mode[inside], rotated[inside], atol=1e-9)


def _evaluate_mode(noll: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Closed-form Noll polynomials, written independently of the module.

    Deliberately duplicated rather than imported: a test that calls the code it
    is testing cannot catch a wrong formula.
    """
    r2 = rho * rho
    r3 = r2 * rho
    r4 = r2 * r2
    table = {
        4: math.sqrt(3.0) * (2.0 * r2 - 1.0),
        5: math.sqrt(6.0) * r2 * torch.sin(2.0 * theta),
        6: math.sqrt(6.0) * r2 * torch.cos(2.0 * theta),
        7: math.sqrt(8.0) * (3.0 * r3 - 2.0 * rho) * torch.sin(theta),
        8: math.sqrt(8.0) * (3.0 * r3 - 2.0 * rho) * torch.cos(theta),
        9: math.sqrt(8.0) * r3 * torch.sin(3.0 * theta),
        10: math.sqrt(8.0) * r3 * torch.cos(3.0 * theta),
        11: math.sqrt(5.0) * (6.0 * r4 - 6.0 * r2 + 1.0),
    }
    return table[noll]


# --------------------------------------------------------------------------
# PSF
# --------------------------------------------------------------------------


def test_psf_normalised_and_centred(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    basis, mask = basis_and_mask
    psf = psf_from_coeffs(zeros_coeffs(), basis, mask)[0]

    assert psf.sum() == pytest.approx(1.0, abs=1e-10)
    assert torch.all(psf >= 0.0)

    n = psf.shape[-1]
    assert psf.argmax().item() == (n // 2) * n + n // 2

    # Centring is checked by 180-degree symmetry about the centre pixel rather
    # than by a centroid. On an even grid the index range is [-n/2, n/2 - 1],
    # which is itself asymmetric, so a centroid of a perfectly centred PSF is
    # not zero. Row/column 0 is the unpaired edge and is excluded.
    paired = psf[1:, 1:]
    assert torch.allclose(paired, torch.flip(paired, dims=(-2, -1)), atol=1e-12)


def test_airy_first_zero_matches_analytic_value(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Unaberrated PSF must have its first dark ring at 0.61*lambda/NA.

    In pixels that is 0.61 * grid / pupil_radius, which for the pinned optics is
    ~3.5 px. This catches FFT scaling and pupil-sizing errors, which otherwise
    produce a PSF of entirely the wrong physical size.
    """
    basis, mask = basis_and_mask
    psf = psf_from_coeffs(zeros_coeffs(), basis, mask)[0]
    profile = radial_profile(psf)

    # first local minimum of the radial profile
    first_zero = next(
        r
        for r in range(1, len(profile) - 1)
        if profile[r] <= profile[r - 1] and profile[r] <= profile[r + 1]
    )

    expected = 0.61 * OPTICS.grid / OPTICS.pupil_radius_px
    assert expected == pytest.approx(OPTICS.airy_radius_px, rel=1e-6)
    assert first_zero == pytest.approx(expected, abs=1.0)


def test_aberration_spreads_the_psf(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Any nonzero aberration must lower the Strehl ratio (peak intensity)."""
    basis, mask = basis_and_mask
    perfect = psf_from_coeffs(zeros_coeffs(), basis, mask)[0]

    for idx in range(N_COEFFS):
        coeffs = zeros_coeffs()
        coeffs[0, idx] = 0.3
        aberrated = psf_from_coeffs(coeffs, basis, mask)[0]
        assert aberrated.max() < perfect.max(), f"mode {NOLL_INDICES[idx]} did not spread the PSF"


def test_defocus_sign_degeneracy(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    """+Z4 and -Z4 give identical intensity PSFs.

    This is why RESEARCH.md restricts Z4 >= 0: the sign is not recoverable from a
    single in-focus intensity image, and leaving it free puts an irreducible
    floor on the loss that looks like a training failure.
    """
    basis, mask = basis_and_mask
    plus, minus = zeros_coeffs(), zeros_coeffs()
    plus[0, 0] = 0.4
    minus[0, 0] = -0.4

    a = psf_from_coeffs(plus, basis, mask)[0]
    b = psf_from_coeffs(minus, basis, mask)[0]
    assert torch.allclose(a, b, atol=1e-12)


def test_center_crop_preserves_normalisation(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    basis, mask = basis_and_mask
    psf = psf_from_coeffs(zeros_coeffs(), basis, mask)
    cropped = center_crop(psf, 128)

    assert cropped.shape == (1, 128, 128)
    assert cropped.sum() == pytest.approx(1.0, abs=1e-10)

    # Airy rings decay slowly, so "almost nothing" is ~1%, not ~0.1%. Encircled
    # energy is E(v) = 1 - J0(v)^2 - J1(v)^2, and for large v the residual goes
    # as 2/(pi*v). The crop half-width of 64 px is ~18 Airy radii, giving
    # v ~ 70 and a predicted ~0.9% outside. Measured: ~0.7%.
    enclosed = psf[0, 64:192, 64:192].sum()
    assert enclosed > 0.99
    assert enclosed < 1.0


def test_center_crop_rejects_upscale(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    basis, mask = basis_and_mask
    psf = psf_from_coeffs(zeros_coeffs(), basis, mask)
    with pytest.raises(ValueError):
        center_crop(psf, 512)


# --------------------------------------------------------------------------
# OTF
# --------------------------------------------------------------------------


def test_otf_dc_is_unity(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    basis, mask = basis_and_mask
    psf = center_crop(psf_from_coeffs(zeros_coeffs(), basis, mask), 128)
    otf = otf_from_psf(psf)
    assert otf[0, 0, 0].real == pytest.approx(1.0, abs=1e-10)
    assert otf[0, 0, 0].imag == pytest.approx(0.0, abs=1e-10)


def test_otf_magnitude_never_exceeds_one(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """|OTF| <= OTF(0) = 1 everywhere, for any aberration."""
    basis, mask = basis_and_mask
    coeffs = torch.zeros(1, N_COEFFS, dtype=DTYPE)
    coeffs[0, 2] = 0.4
    coeffs[0, 5] = -0.3
    psf = center_crop(psf_from_coeffs(coeffs, basis, mask), 128)
    assert otf_from_psf(psf).abs().max() <= 1.0 + 1e-9


def test_otf_vanishes_beyond_cutoff(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    """The incoherent OTF has finite support, cutting off at 2*NA/lambda.

    Beyond that the microscope transmitted nothing, which is precisely why no
    exact inverse kernel exists.
    """
    basis, mask = basis_and_mask
    size = 128
    psf = center_crop(psf_from_coeffs(zeros_coeffs(), basis, mask), size)
    otf = torch.fft.fftshift(otf_from_psf(psf)[0]).abs()

    # cutoff in cycles/pixel is 2*NA*dx/lambda; in FFT bins, times `size`
    cutoff_bins = 2.0 * OPTICS.na * OPTICS.pixel_nm / OPTICS.wavelength_nm * size
    profile = radial_profile(otf)

    inside = profile[: int(cutoff_bins * 0.5)]
    outside = profile[int(cutoff_bins * 1.2) :]
    assert inside.min() > 1e-3
    assert outside.abs().max() < 1e-3


# --------------------------------------------------------------------------
# Convolution and inversion
# --------------------------------------------------------------------------


def test_delta_psf_is_identity() -> None:
    """Convolving with a delta function must return the image unchanged."""
    size = 64
    psf = torch.zeros(1, size, size, dtype=DTYPE)
    psf[0, size // 2, size // 2] = 1.0
    image = torch.rand(1, size, size, dtype=DTYPE)

    out = convolve_fft(image, otf_from_psf(psf))
    assert torch.allclose(out, image, atol=1e-10)


def test_convolution_preserves_total_intensity(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A PSF summing to 1 conserves photons."""
    basis, mask = basis_and_mask
    psf = center_crop(psf_from_coeffs(zeros_coeffs(), basis, mask), 64)
    image = torch.rand(1, 64, 64, dtype=DTYPE)

    out = convolve_fft(image, otf_from_psf(psf))
    assert out.sum() == pytest.approx(image.sum().item(), rel=1e-9)


def test_wiener_of_delta_psf_is_identity() -> None:
    size = 64
    psf = torch.zeros(1, size, size, dtype=DTYPE)
    psf[0, size // 2, size // 2] = 1.0
    image = torch.rand(1, size, size, dtype=DTYPE)

    out = apply_wiener(image, otf_from_psf(psf), reg=1e-8)
    assert torch.allclose(out, image, atol=1e-6)


def test_wiener_recovers_noiseless_blur(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """With no noise and small regularisation, Wiener should undo the blur well."""
    basis, mask = basis_and_mask
    psf = center_crop(psf_from_coeffs(zeros_coeffs(), basis, mask), 128)
    otf = otf_from_psf(psf)

    image = torch.zeros(1, 128, 128, dtype=DTYPE)
    image[0, 40:80, 50:70] = 1.0

    blurred = convolve_fft(image, otf)
    restored = apply_wiener(blurred, otf, reg=1e-6)

    assert (restored - image).abs().mean() < (blurred - image).abs().mean()


def test_wiener_approaches_naive_as_reg_goes_to_zero() -> None:
    """Wiener with reg -> 0 is the unregularized inverse."""
    size = 32
    psf = torch.zeros(1, size, size, dtype=DTYPE)
    psf[0, size // 2, size // 2] = 0.6
    psf[0, size // 2 + 1, size // 2] = 0.4
    otf = otf_from_psf(psf)
    image = torch.rand(1, size, size, dtype=DTYPE)

    tight = apply_wiener(image, otf, reg=1e-14)
    naive = naive_inverse(image, otf, eps=0.0)
    assert torch.allclose(tight, naive, atol=1e-6)


def test_more_regularisation_is_smoother(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Larger reg must suppress the restored high frequencies."""
    basis, mask = basis_and_mask
    psf = center_crop(psf_from_coeffs(zeros_coeffs(), basis, mask), 128)
    otf = otf_from_psf(psf)

    torch.manual_seed(0)
    noisy = convolve_fft(torch.rand(1, 128, 128, dtype=DTYPE), otf)
    noisy = noisy + 0.01 * torch.randn_like(noisy)

    light = apply_wiener(noisy, otf, reg=1e-4)
    heavy = apply_wiener(noisy, otf, reg=1e-1)
    assert heavy.std() < light.std()


def test_wiener_filter_is_finite_where_otf_vanishes() -> None:
    """The +reg term is what makes the inverse exist at all."""
    otf = torch.zeros(4, 4, dtype=torch.complex128)
    filt = wiener_filter(otf, reg=1e-6)
    assert torch.all(torch.isfinite(filt.real))
    assert torch.all(torch.isfinite(filt.imag))
    assert torch.all(filt.abs() == 0.0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_radial_profile_of_constant_is_constant() -> None:
    img = torch.full((32, 32), 3.0, dtype=DTYPE)
    profile = radial_profile(img)
    assert torch.allclose(profile[:10], torch.full((10,), 3.0, dtype=torch.float32), atol=1e-5)
