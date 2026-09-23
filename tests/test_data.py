"""Tests for object generation, the noisy forward model and real-image loading.

The critical assertion here is :func:`test_held_out_never_in_training_pool`: if
held-out images leak into training, the transfer test measures nothing.
"""

from __future__ import annotations

import pytest
import torch

from deconv.data import (
    COEFF_RANGE,
    HELD_OUT,
    TRAIN_REAL,
    NoiseConfig,
    add_noise,
    denormalise_coeffs,
    load_held_out,
    load_real_slices,
    log_power_spectrum,
    make_beads,
    make_extended,
    make_objects,
    normalise_coeffs,
    normalise_image,
    random_crops,
    sample_coeffs,
    synth_batch,
)
from deconv.optics import (
    N_COEFFS,
    Optics,
    center_crop,
    convolve_fft,
    otf_from_psf,
    psf_from_coeffs,
    zernike_basis,
)

OPTICS = Optics()
SIZE = 64


@pytest.fixture(scope="module")
def basis_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    return zernike_basis(OPTICS)


@pytest.fixture
def gen() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


# --------------------------------------------------------------------------
# Coefficient sampling
# --------------------------------------------------------------------------


def test_sample_coeffs_shape_and_range(gen: torch.Generator) -> None:
    coeffs = sample_coeffs(32, generator=gen)
    assert coeffs.shape == (32, N_COEFFS)
    assert coeffs.abs().max() <= COEFF_RANGE


def test_defocus_is_non_negative(gen: torch.Generator) -> None:
    """Z4 sign is not identifiable from a single in-focus image, so it is restricted."""
    coeffs = sample_coeffs(256, generator=gen)
    assert torch.all(coeffs[:, 0] >= 0.0)
    # the other modes must still be free to go negative
    assert torch.any(coeffs[:, 1:] < 0.0)


def test_coefficient_normalisation_roundtrip(gen: torch.Generator) -> None:
    coeffs = sample_coeffs(16, generator=gen)
    assert torch.allclose(denormalise_coeffs(normalise_coeffs(coeffs)), coeffs, atol=1e-6)
    assert normalise_coeffs(coeffs).abs().max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------
# Object generators
# --------------------------------------------------------------------------


def test_beads_are_sparse_and_positive(gen: torch.Generator) -> None:
    beads = make_beads(4, SIZE, generator=gen, min_beads=30, max_beads=150)
    assert beads.shape == (4, SIZE, SIZE)
    assert torch.all(beads >= 0.0)

    for b in range(4):
        occupied = int((beads[b] > 0).sum().item())
        assert 1 <= occupied <= 150
        assert occupied < SIZE * SIZE * 0.1, "beads should be sparse"


def test_extended_structures_are_non_degenerate(gen: torch.Generator) -> None:
    ext = make_extended(4, SIZE, generator=gen)
    assert ext.shape == (4, SIZE, SIZE)
    assert torch.all(ext >= 0.0)

    for b in range(4):
        assert ext[b].max() > 0.0, "generated an empty field"
        assert ext[b].std() > 0.0, "generated a constant field"
        occupied = (ext[b] > 0.01 * ext[b].max()).float().mean()
        assert occupied > 0.01, "extended structures should cover more than beads do"


def test_extended_are_denser_than_beads(gen: torch.Generator) -> None:
    """The whole point of adding extended objects is different image statistics."""
    beads = make_beads(8, SIZE, generator=gen)
    ext = make_extended(8, SIZE, generator=gen)

    bead_fill = (beads > 0).float().mean()
    ext_fill = (ext > 0.01 * ext.amax()).float().mean()
    assert ext_fill > bead_fill


def test_make_objects_respects_mix(gen: torch.Generator) -> None:
    objects = make_objects(16, SIZE, bead_fraction=0.5, generator=gen)
    assert objects.shape == (16, SIZE, SIZE)
    assert torch.all(objects >= 0.0)


def test_make_objects_with_real_pool(gen: torch.Generator) -> None:
    pool = torch.rand(5, 128, 128)
    objects = make_objects(20, SIZE, generator=gen, real_pool=pool, real_fraction=0.25, device=None)
    assert objects.shape == (20, SIZE, SIZE)


def test_random_crops_shape_and_bounds(gen: torch.Generator) -> None:
    pool = torch.rand(3, 100, 120)
    crops = random_crops(pool, 8, SIZE, generator=gen)
    assert crops.shape == (8, SIZE, SIZE)
    assert crops.min() >= 0.0 and crops.max() <= 1.0


def test_random_crops_rejects_oversized_crop(gen: torch.Generator) -> None:
    pool = torch.rand(2, 32, 32)
    with pytest.raises(ValueError):
        random_crops(pool, 4, 64, generator=gen)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def test_poisson_noise_preserves_mean(gen: torch.Generator) -> None:
    """Shot noise is unbiased: E[Poisson(k)] = k."""
    image = torch.ones(64, 64, 64) * 0.5
    noisy = add_noise(image, peak_photons=1000.0, read_sigma=0.0, generator=gen)
    assert noisy.mean().item() == pytest.approx(1000.0, rel=0.02)


def test_noise_increases_variance(gen: torch.Generator) -> None:
    image = torch.ones(8, 64, 64) * 0.5
    noisy = add_noise(image, peak_photons=500.0, read_sigma=2.0, generator=gen)
    assert noisy.std() > 0.0


def test_fewer_photons_means_lower_snr(gen: torch.Generator) -> None:
    """Poisson SNR goes as sqrt(N), so fewer photons is relatively noisier."""
    image = torch.ones(16, 64, 64) * 0.5

    bright = add_noise(image, peak_photons=5000.0, read_sigma=0.0, generator=gen)
    dim = add_noise(image, peak_photons=50.0, read_sigma=0.0, generator=gen)

    assert (bright.std() / bright.mean()) < (dim.std() / dim.mean())


def test_noise_output_is_non_negative(gen: torch.Generator) -> None:
    image = torch.rand(4, 32, 32)
    noisy = add_noise(image, peak_photons=20.0, read_sigma=5.0, generator=gen)
    assert torch.all(noisy >= 0.0)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def test_normalise_image_maps_to_unit_range() -> None:
    image = torch.randn(4, 32, 32) * 10.0 + 3.0
    out = normalise_image(image)
    for b in range(4):
        assert out[b].min().item() == pytest.approx(0.0, abs=1e-6)
        assert out[b].max().item() == pytest.approx(1.0, abs=1e-6)


def test_normalise_image_handles_constant_input() -> None:
    """A flat image must not produce NaN."""
    out = normalise_image(torch.full((2, 16, 16), 5.0))
    assert torch.all(torch.isfinite(out))


def test_log_power_spectrum_is_standardised() -> None:
    image = torch.rand(4, 32, 32)
    spec = log_power_spectrum(image)
    assert spec.shape == image.shape
    assert torch.all(torch.isfinite(spec))
    for b in range(4):
        assert spec[b].mean().item() == pytest.approx(0.0, abs=1e-5)
        assert spec[b].std().item() == pytest.approx(1.0, abs=1e-3)


def test_log_power_spectrum_is_blur_sensitive() -> None:
    """A blurred image must have visibly less high-frequency power.

    This is the premise of the image+spectrum input variant.
    """
    torch.manual_seed(0)
    sharp = torch.rand(1, 64, 64)

    kernel = torch.zeros(1, 64, 64)
    kernel[0, 30:35, 30:35] = 1.0 / 25.0
    otf = torch.fft.fft2(torch.fft.ifftshift(kernel, dim=(-2, -1)))
    blurred = torch.fft.ifft2(torch.fft.fft2(sharp) * otf).real

    corner = (slice(None), slice(0, 8), slice(0, 8))  # high frequency after fftshift
    assert log_power_spectrum(blurred)[corner].mean() < log_power_spectrum(sharp)[corner].mean()


# --------------------------------------------------------------------------
# Full forward model
# --------------------------------------------------------------------------


def test_synth_batch_shapes(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor], gen: torch.Generator
) -> None:
    basis, mask = basis_and_mask
    images, coeffs, objects = synth_batch(
        8, SIZE, OPTICS, basis, mask, generator=gen, noise=NoiseConfig()
    )

    assert images.shape == (8, SIZE, SIZE)
    assert coeffs.shape == (8, N_COEFFS)
    assert objects.shape == (8, SIZE, SIZE)
    assert torch.all(torch.isfinite(images))
    assert images.min() >= 0.0 and images.max() <= 1.0


def test_synth_batch_is_reproducible(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    basis, mask = basis_and_mask

    g1 = torch.Generator()
    g1.manual_seed(42)
    a_img, a_c, _ = synth_batch(4, SIZE, OPTICS, basis, mask, generator=g1)

    g2 = torch.Generator()
    g2.manual_seed(42)
    b_img, b_c, _ = synth_batch(4, SIZE, OPTICS, basis, mask, generator=g2)

    assert torch.allclose(a_c, b_c)
    assert torch.allclose(a_img, b_img)


def test_convolution_reduces_high_frequency_content(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Blurring must attenuate high frequencies - the defining act of the forward model.

    Tested on the *noiseless* convolution. See the companion test below for why
    the same claim is false once noise is added.
    """
    basis, mask = basis_and_mask
    g = torch.Generator()
    g.manual_seed(3)

    coeffs = sample_coeffs(8, generator=g)
    psf = center_crop(psf_from_coeffs(coeffs, basis, mask), SIZE)
    objects = make_extended(8, SIZE, generator=g)
    blurred = convolve_fft(objects, otf_from_psf(psf))

    sharp_hf = log_power_spectrum(normalise_image(objects))[:, :8, :8].mean()
    blur_hf = log_power_spectrum(normalise_image(blurred))[:, :8, :8].mean()
    assert blur_hf < sharp_hf


def test_noise_raises_the_high_frequency_floor(
    basis_and_mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Shot noise is white, so it refills the band that blurring emptied.

    This is physically correct and worth pinning: it means the OTF's signature in
    the power spectrum sits on a noise floor, and the ``image+spectrum`` variant
    has to read the mid-frequency rolloff rather than the cutoff itself.
    """
    basis, mask = basis_and_mask
    g = torch.Generator()
    g.manual_seed(4)

    coeffs = sample_coeffs(8, generator=g)
    psf = center_crop(psf_from_coeffs(coeffs, basis, mask), SIZE)
    objects = make_extended(8, SIZE, generator=g)
    blurred = convolve_fft(objects, otf_from_psf(psf))
    noisy = add_noise(blurred, peak_photons=500.0, read_sigma=2.0, generator=g)

    clean_hf = log_power_spectrum(normalise_image(blurred))[:, :8, :8].mean()
    noisy_hf = log_power_spectrum(normalise_image(noisy))[:, :8, :8].mean()
    assert noisy_hf > clean_hf


# --------------------------------------------------------------------------
# Real data - the split that makes the transfer test meaningful
# --------------------------------------------------------------------------


def test_held_out_never_in_training_pool() -> None:
    """The transfer test is only meaningful if these sets are disjoint."""
    assert set(HELD_OUT).isdisjoint(set(TRAIN_REAL))
    assert "human_mitosis" in HELD_OUT
    assert "kidney" in HELD_OUT
    assert "cells3d" in TRAIN_REAL


def test_load_held_out_rejects_training_images() -> None:
    with pytest.raises(ValueError):
        load_held_out("cells3d")


def test_load_real_slices_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        load_real_slices("not_a_dataset")


@pytest.mark.parametrize(
    "name,expected_slices,expected_size",
    [("cells3d", 120, 256), ("human_mitosis", 1, 512), ("kidney", 48, 512)],
)
def test_real_loaders_return_documented_shapes(
    name: str, expected_slices: int, expected_size: int
) -> None:
    """Shapes are pinned because PLAN.md's split arithmetic depends on them.

    Requires pooch and a one-time network fetch; skipped if unavailable.
    """
    pytest.importorskip("pooch")
    try:
        slices = load_real_slices(name)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not fetch {name}: {exc}")

    assert slices.shape == (expected_slices, expected_size, expected_size)
    assert slices.dtype == torch.float32
    assert slices.min() >= 0.0 and slices.max() <= 1.0
