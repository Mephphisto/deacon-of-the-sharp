"""Plotting helpers.

These are deliberately kept out of the notebook: matplotlib boilerplate is not
physics, and burying the narrative in axis configuration would defeat the point.
Every function returns its Figure so the notebook stays one line per figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from deconv.metrics import frequency_axis_per_um
from deconv.optics import NOLL_INDICES, NOLL_NAMES, Optics, radial_profile

_PSF_CMAP = "inferno"
_IMG_CMAP = "gray"


def _show(ax: Axes, image: torch.Tensor, title: str = "", cmap: str = _IMG_CMAP) -> None:
    ax.imshow(image.detach().cpu().float().numpy(), cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _strip(panels: list[tuple[torch.Tensor, str]], emphasise: int | None = None) -> Figure:
    """One row of images with titles; ``emphasise`` bolds a single panel."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (image, title) in zip(axes, panels, strict=True):
        _show(ax, image, title)
    if emphasise is not None:
        axes[emphasise].set_title(axes[emphasise].get_title(), fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_zernike_gallery(basis: torch.Tensor, mask: torch.Tensor) -> Figure:
    """The 8 Noll modes used as the label vector."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        noll = NOLL_INDICES[idx]
        mode = (basis[idx] * mask).detach().cpu().numpy()
        ax.imshow(mode, cmap="RdBu_r")
        ax.set_title(f"$Z_{{{noll}}}$ - {NOLL_NAMES[noll]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Zernike basis (Noll indexing)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_psf_and_profile(psf: torch.Tensor, optics: Optics, zoom: int = 32) -> Figure:
    """A PSF alongside its radial profile, with the Airy radius marked."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    n = psf.shape[-1]
    lo, hi = n // 2 - zoom, n // 2 + zoom
    _show(left, psf[lo:hi, lo:hi], "PSF (log scale)", cmap=_PSF_CMAP)
    left.images[0].set_data(torch.log10(psf[lo:hi, lo:hi] + 1e-12).cpu().numpy())

    profile = radial_profile(psf).cpu()
    radii = torch.arange(len(profile)).float() * optics.pixel_nm
    right.semilogy(radii[:zoom], profile[:zoom].clamp(min=1e-12))
    right.axvline(
        optics.airy_radius_nm,
        color="crimson",
        linestyle="--",
        label=f"Airy radius {optics.airy_radius_nm:.0f} nm",
    )
    right.set_xlabel("radius (nm)")
    right.set_ylabel("intensity")
    right.set_title("Radial profile")
    right.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_psf_gallery(psfs: torch.Tensor, titles: list[str], zoom: int = 24) -> Figure:
    """A row of PSFs, one per aberration mode."""
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    size = psfs.shape[-1]
    lo, hi = size // 2 - zoom, size // 2 + zoom
    for idx, ax in enumerate(axes if n > 1 else [axes]):
        patch = psfs[idx, lo:hi, lo:hi]
        ax.imshow(torch.log10(patch + 1e-12).cpu().numpy(), cmap=_PSF_CMAP)
        ax.set_title(titles[idx], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_otf(otf: torch.Tensor, optics: Optics) -> Figure:
    """OTF magnitude with the 2*NA/lambda cutoff ring marked."""
    magnitude = torch.fft.fftshift(otf).abs().cpu()
    size = magnitude.shape[-1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    extent_per_um = 1000.0 / (2.0 * optics.pixel_nm)
    left.imshow(
        torch.log10(magnitude + 1e-12).numpy(),
        cmap="viridis",
        extent=(-extent_per_um, extent_per_um, -extent_per_um, extent_per_um),
    )
    ring = Circle(
        (0, 0), optics.cutoff_per_um, fill=False, color="crimson", linestyle="--", linewidth=1.5
    )
    left.add_patch(ring)
    left.set_title("$\\log_{10}|OTF|$ with cutoff $2NA/\\lambda$", fontsize=10)
    left.set_xlabel("spatial frequency (1/$\\mu$m)")

    profile = radial_profile(magnitude)
    freq = torch.arange(len(profile)).float() / size / optics.pixel_nm * 1000.0
    right.semilogy(freq, profile.clamp(min=1e-12))
    right.axvline(
        optics.cutoff_per_um,
        color="crimson",
        linestyle="--",
        label=f"cutoff {optics.cutoff_per_um:.2f} /$\\mu$m",
    )
    right.set_xlim(0, optics.cutoff_per_um * 1.6)
    right.set_xlabel("spatial frequency (1/$\\mu$m)")
    right.set_ylabel("|OTF|")
    right.set_title("Radially averaged OTF")
    right.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_forward_model(obj: torch.Tensor, blurred: torch.Tensor, noisy: torch.Tensor) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _show(axes[0], obj, "object")
    _show(axes[1], blurred, "blurred")
    _show(axes[2], noisy, "blurred + noise (what the microscope gives you)")
    fig.tight_layout()
    return fig


def plot_lambda_sweep(images: list[torch.Tensor], regs: list[float], scores: list[float]) -> Figure:
    """The bias/variance strip: under-regularised is noisy, over-regularised is blurry."""
    fig, axes = plt.subplots(1, len(images), figsize=(2.4 * len(images), 3.0))
    for ax, image, reg, score in zip(axes, images, regs, scores, strict=True):
        _show(ax, image, f"$\\lambda$={reg:.0e}\n{score:.1f} dB")
    fig.suptitle("Wiener regularisation sweep", fontsize=12)
    fig.tight_layout()
    return fig


def plot_loss_curves(results: dict[str, tuple[list[int], list[float]]]) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (steps, losses) in results.items():
        ax.semilogy(steps, losses, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE on normalised coefficients")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_mode_error(errors: dict[str, torch.Tensor]) -> Figure:
    """Grouped bars of per-Zernike-mode error, in waves."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    labels = [f"$Z_{{{j}}}$\n{NOLL_NAMES[j].split()[0]}" for j in NOLL_INDICES]
    positions = torch.arange(len(labels)).float()
    width = 0.8 / max(1, len(errors))

    for offset, (label, values) in enumerate(errors.items()):
        ax.bar(positions + offset * width, values.cpu().numpy(), width, label=label)

    ax.set_xticks(positions + width * (len(errors) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean absolute error (waves)")
    ax.set_title("Per-mode coefficient error")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_three_way(
    blurred: torch.Tensor,
    naive_psf: torch.Tensor,
    predicted: torch.Tensor,
    oracle: torch.Tensor,
    truth: torch.Tensor,
    scores: dict[str, float],
) -> Figure:
    """The money plot.

    The gap between `predicted` and `oracle` is what the model costs you; the gap
    between `predicted` and `naive` is what it buys you.
    """
    return _strip(
        [
            (blurred, f"blurred\n{scores['blurred']:.2f} dB"),
            (naive_psf, f"wrong PSF\n{scores['naive']:.2f} dB"),
            (predicted, f"PREDICTED PSF\n{scores['predicted']:.2f} dB"),
            (oracle, f"true PSF (oracle)\n{scores['oracle']:.2f} dB"),
            (truth, "ground truth"),
        ],
        emphasise=2,
    )


def plot_real_data_comparison(
    blurred: torch.Tensor,
    synthetic_only: torch.Tensor,
    with_real: torch.Tensor,
    oracle: torch.Tensor,
    truth: torch.Tensor,
    scores: dict[str, float],
) -> Figure:
    """Does 10% real data in training show up in the reconstructed image?

    The object, the PSF that blurred it, the noise draw and the Wiener step are
    identical across every panel. Only the training mix behind the *predicted*
    PSF differs between panels 2 and 3, so whatever separates them is
    attributable to the data and to nothing else.
    """
    return _strip(
        [
            (blurred, f"blurred\n{scores['blurred']:.2f} dB"),
            (synthetic_only, f"synthetic only\n{scores['synthetic only']:.2f} dB"),
            (with_real, f"+10% REAL\n{scores['+10% real']:.2f} dB"),
            (oracle, f"true PSF (oracle)\n{scores['oracle']:.2f} dB"),
            (truth, "ground truth"),
        ],
        emphasise=2,
    )


def plot_frc(curves: dict[str, torch.Tensor], optics: Optics, threshold: float = 1 / 7) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, curve in curves.items():
        ax.plot(frequency_axis_per_um(curve, optics).numpy(), curve.cpu().numpy(), label=label)
    ax.axhline(threshold, color="crimson", linestyle="--", label=f"{threshold:.3f} threshold")
    ax.axvline(optics.cutoff_per_um, color="grey", linestyle=":", label="OTF cutoff")
    ax.set_xlabel("spatial frequency (1/$\\mu$m)")
    ax.set_ylabel("FRC")
    ax.set_title("Fourier Ring Correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gain_vs_photons(photons: list[float], gains: list[float]) -> Figure:
    """Deconvolution gain against photon budget.

    At low photon counts even a perfect PSF buys nothing, because the
    information is not in the data. Worth showing rather than hiding.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(photons, gains, marker="o")
    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("peak photons")
    ax.set_ylabel("PSNR gain from deconvolution (dB)")
    ax.set_title("Deconvolution only helps when you have photons")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
