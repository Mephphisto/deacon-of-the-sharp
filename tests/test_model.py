"""Tests for the CNN and training loop.

These never train to convergence and never touch a GPU. Whether the science
works is established by the overfit-one-batch gate and the full runs in the
notebook; these tests only establish that the machinery is sound.
"""

from __future__ import annotations

import pytest
import torch

from deconv.data import NoiseConfig
from deconv.model import (
    INPUT_VARIANTS,
    PSFNet,
    TrainConfig,
    build_input,
    coefficient_errors,
    count_parameters,
    n_input_channels,
    predict_coeffs,
    train,
)
from deconv.optics import N_COEFFS, Optics, zernike_basis

OPTICS = Optics()
SIZE = 32


@pytest.fixture(scope="module")
def basis_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    return zernike_basis(OPTICS)


# --------------------------------------------------------------------------
# Input assembly
# --------------------------------------------------------------------------


def test_input_channel_counts() -> None:
    assert n_input_channels("image") == 1
    assert n_input_channels("image+spectrum") == 2


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError):
        n_input_channels("image+magic")


@pytest.mark.parametrize("variant", INPUT_VARIANTS)
def test_build_input_shape(variant: str) -> None:
    image = torch.rand(4, SIZE, SIZE)
    x = build_input(image, variant)
    assert x.shape == (4, n_input_channels(variant), SIZE, SIZE)
    assert torch.all(torch.isfinite(x))


def test_spectrum_channel_differs_from_image() -> None:
    """The second channel must actually carry different information."""
    image = torch.rand(2, SIZE, SIZE)
    x = build_input(image, "image+spectrum")
    assert not torch.allclose(x[:, 0], x[:, 1])


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", INPUT_VARIANTS)
def test_forward_shape(variant: str) -> None:
    model = PSFNet(variant)
    out = model(build_input(torch.rand(3, SIZE, SIZE), variant))
    assert out.shape == (3, N_COEFFS)
    assert torch.all(torch.isfinite(out))


def test_parameter_count_in_planned_range() -> None:
    """PLAN.md budgets 2-4 M parameters; the runtime estimate depends on it."""
    n = count_parameters(PSFNet("image"))
    assert 2_000_000 < n < 4_000_000, f"unexpected parameter count {n}"


def test_model_is_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(0)
    a = PSFNet("image")
    torch.manual_seed(0)
    b = PSFNet("image")

    x = build_input(torch.rand(2, SIZE, SIZE), "image")
    a.eval()
    b.eval()
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_accepts_multiple_input_sizes() -> None:
    """Global average pooling should make the head size-independent."""
    model = PSFNet("image").eval()
    for size in (32, 64):
        out = model(build_input(torch.rand(1, size, size), "image"))
        assert out.shape == (1, N_COEFFS)


def test_predict_coeffs_runs_in_eval_mode() -> None:
    model = PSFNet("image")
    model.train()
    out = predict_coeffs(model, torch.rand(2, SIZE, SIZE))
    assert out.shape == (2, N_COEFFS)
    assert not model.training


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def test_one_step_decreases_loss_on_a_fixed_batch() -> None:
    """A few optimizer steps on one fixed batch must reduce the loss.

    This is the cheap CPU analogue of the notebook's overfit-one-batch gate: if
    the loss does not move here, the training wiring is broken.
    """
    torch.manual_seed(0)
    model = PSFNet("image")
    x = build_input(torch.rand(8, SIZE, SIZE), "image")
    target = torch.rand(8, N_COEFFS) * 2.0 - 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = torch.nn.functional.mse_loss(model(x), target).item()
    last = first

    for _ in range(10):
        loss = torch.nn.functional.mse_loss(model(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = loss.item()

    assert last < first


def test_train_runs_and_logs(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    basis, mask = basis_and_mask
    cfg = TrainConfig(steps=6, batch=2, size=SIZE, warmup=2, log_every=3, noise=NoiseConfig())

    result = train(PSFNet("image"), OPTICS, basis, mask, cfg, progress=False)

    assert result.steps == [3, 6]
    assert len(result.losses) == 2
    assert all(loss > 0.0 for loss in result.losses)
    assert result.final_loss == result.losses[-1]


def test_train_is_reproducible(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Same seed, same losses - required for the Run 3 comparison to be controlled."""
    basis, mask = basis_and_mask
    cfg = TrainConfig(steps=4, batch=2, size=SIZE, warmup=1, log_every=2, seed=7)

    torch.manual_seed(0)
    first = train(PSFNet("image"), OPTICS, basis, mask, cfg, progress=False)
    torch.manual_seed(0)
    second = train(PSFNet("image"), OPTICS, basis, mask, cfg, progress=False)

    assert first.losses == pytest.approx(second.losses, rel=1e-5)


def test_coefficient_errors_shape(basis_and_mask: tuple[torch.Tensor, torch.Tensor]) -> None:
    basis, mask = basis_and_mask
    per_mode, overall = coefficient_errors(
        PSFNet("image"), OPTICS, basis, mask, batch=4, size=SIZE, seed=1
    )

    assert per_mode.shape == (N_COEFFS,)
    assert torch.all(per_mode >= 0.0)
    assert overall.item() == pytest.approx(per_mode.mean().item(), rel=1e-5)
