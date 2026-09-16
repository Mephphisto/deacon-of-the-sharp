"""Tests for PSNR, SSIM and the hand-rolled Fourier Ring Correlation."""

from __future__ import annotations

import pytest
import torch

from deconv.metrics import frc, frc_resolution_nm, frequency_axis_per_um, psnr, ssim
from deconv.optics import Optics

OPTICS = Optics()
SIZE = 64


@pytest.fixture
def image() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(SIZE, SIZE)


# --------------------------------------------------------------------------
# PSNR / SSIM
# --------------------------------------------------------------------------


def test_psnr_of_identical_images_is_large(image: torch.Tensor) -> None:
    assert psnr(image, image) > 100.0


def test_psnr_decreases_with_noise(image: torch.Tensor) -> None:
    torch.manual_seed(1)
    mild = (image + 0.01 * torch.randn_like(image)).clamp(0, 1)
    harsh = (image + 0.20 * torch.randn_like(image)).clamp(0, 1)
    assert psnr(mild, image) > psnr(harsh, image)


def test_ssim_of_identical_images_is_one(image: torch.Tensor) -> None:
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_ssim_decreases_with_noise(image: torch.Tensor) -> None:
    torch.manual_seed(2)
    noisy = (image + 0.2 * torch.randn_like(image)).clamp(0, 1)
    assert ssim(noisy, image) < ssim(image, image)


def test_metrics_accept_batches() -> None:
    torch.manual_seed(3)
    batch = torch.rand(4, SIZE, SIZE)
    assert psnr(batch, batch) > 100.0
    assert ssim(batch, batch) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# FRC
# --------------------------------------------------------------------------


def test_frc_of_image_with_itself_is_unity(image: torch.Tensor) -> None:
    curve = frc(image, image)
    assert torch.all(curve[:-1] > 0.99), "self-correlation must be ~1 at all frequencies"


def test_frc_of_independent_noise_is_near_zero() -> None:
    torch.manual_seed(4)
    a = torch.randn(SIZE, SIZE)
    b = torch.randn(SIZE, SIZE)
    curve = frc(a, b)
    # DC bin aside, uncorrelated fields should hover around zero
    assert curve[1:].abs().mean() < 0.3


def test_frc_decreases_when_noise_is_added(image: torch.Tensor) -> None:
    """Two noisy copies of one scene correlate worse than two clean ones."""
    torch.manual_seed(5)
    a = image + 0.3 * torch.randn_like(image)
    b = image + 0.3 * torch.randn_like(image)

    clean = frc(image, image)[1:].mean()
    noisy = frc(a, b)[1:].mean()
    assert noisy < clean


def test_frc_rejects_mismatched_shapes(image: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        frc(image, torch.rand(32, 32))


def test_frc_curve_length() -> None:
    torch.manual_seed(6)
    curve = frc(torch.rand(SIZE, SIZE), torch.rand(SIZE, SIZE))
    assert curve.shape == (SIZE // 2,)


def test_resolution_is_finite_for_a_decaying_curve() -> None:
    curve = torch.linspace(1.0, 0.0, 32)
    resolution = frc_resolution_nm(curve, OPTICS)
    assert resolution > 0.0
    assert resolution < float("inf")


def test_better_correlation_means_finer_resolution() -> None:
    """A curve that stays high longer must report a smaller (finer) resolution."""
    poor = torch.cat([torch.ones(4), torch.zeros(28)])
    good = torch.cat([torch.ones(16), torch.zeros(16)])
    assert frc_resolution_nm(good, OPTICS) < frc_resolution_nm(poor, OPTICS)


def test_resolution_of_fully_correlated_curve_hits_sampling_limit() -> None:
    curve = torch.ones(32)
    assert frc_resolution_nm(curve, OPTICS) == pytest.approx(2.0 * OPTICS.pixel_nm)


def test_frequency_axis_matches_curve_and_reaches_nyquist() -> None:
    curve = torch.ones(32)
    axis = frequency_axis_per_um(curve, OPTICS)

    assert axis.shape == curve.shape
    assert axis[0].item() == 0.0
    # the last bin approaches the sampling Nyquist frequency 1/(2*dx)
    nyquist_per_um = 1000.0 / (2.0 * OPTICS.pixel_nm)
    assert axis[-1].item() == pytest.approx(nyquist_per_um, rel=0.05)
