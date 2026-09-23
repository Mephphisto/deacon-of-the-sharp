"""The CNN that estimates Zernike coefficients, and the training loop.

This is the only learned component in the project. Everything it feeds - PSF
reconstruction, Wiener filtering, deconvolution - is closed-form physics.

Aberration is a *global* property of the image, so the network reduces to a
global average pool before the regression head rather than predicting anything
spatially resolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from deconv.data import NoiseConfig, log_power_spectrum, normalise_coeffs, synth_batch
from deconv.optics import N_COEFFS, Optics

INPUT_VARIANTS: tuple[str, ...] = ("image", "image+spectrum")


def n_input_channels(variant: str) -> int:
    if variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown input variant {variant!r}; expected one of {INPUT_VARIANTS}")
    return 1 if variant == "image" else 2


def build_input(image: torch.Tensor, variant: str) -> torch.Tensor:
    """Assemble network input ``(batch, channels, h, w)`` from a normalised image.

    The ``image+spectrum`` variant adds the log power spectrum, where the OTF's
    signature - especially its zero rings - is far more directly visible than in
    the image itself.
    """
    channels = [image]
    if n_input_channels(variant) == 2:
        channels.append(log_power_spectrum(image))
    return torch.stack(channels, dim=1)


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PSFNet(nn.Module):
    """Predicts the 8 normalised Noll coefficients from a blurred image."""

    def __init__(self, variant: str = "image", width: int = 32) -> None:
        super().__init__()
        self.variant = variant
        in_ch = n_input_channels(variant)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _block(width, width * 2),
            _block(width * 2, width * 4),
            _block(width * 4, width * 8),
            _block(width * 8, width * 8),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 8, width * 4),
            nn.ReLU(inplace=True),
            nn.Linear(width * 4, N_COEFFS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        pooled = h.mean(dim=(-2, -1))
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainConfig:
    """Hyperparameters from RESEARCH.md section 5.3."""

    steps: int = 30_000
    batch: int = 64
    size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 500
    bead_fraction: float = 0.5
    real_fraction: float = 0.0
    seed: int = 0
    log_every: int = 100
    noise: NoiseConfig = field(default_factory=NoiseConfig)


def _lr_scale(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay."""
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainResult:
    steps: list[int]
    losses: list[float]

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def train(
    model: PSFNet,
    optics: Optics,
    basis: torch.Tensor,
    mask: torch.Tensor,
    cfg: TrainConfig | None = None,
    real_pool: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    progress: bool = True,
) -> TrainResult:
    """Train on freshly synthesised batches. One function, reused for every run.

    Using a single ``train`` for all three runs is what makes the Run 3
    comparison controlled: only the data mix differs.

    Autocast covers the network only. The forward optical model stays in fp32,
    because FFTs in bf16 lose enough precision to distort the PSF.
    """
    cfg = cfg or TrainConfig()
    model = model.to(device).train()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_scale(s, cfg))

    use_amp = torch.device(device).type == "cuda"
    steps_log: list[int] = []
    losses: list[float] = []
    running = 0.0

    iterator = range(cfg.steps)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=f"train[{model.variant}]")
        except ImportError:
            pass

    for step in iterator:
        with torch.no_grad():  # the forward model is physics, not something to learn
            images, coeffs, _ = synth_batch(
                cfg.batch,
                cfg.size,
                optics,
                basis,
                mask,
                noise=cfg.noise,
                generator=generator,
                device=device,
                bead_fraction=cfg.bead_fraction,
                real_pool=real_pool,
                real_fraction=cfg.real_fraction,
            )
            inputs = build_input(images, model.variant)
            targets = normalise_coeffs(coeffs)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            predicted = model(inputs)
            loss = nn.functional.mse_loss(predicted.float(), targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running += loss.item()
        if (step + 1) % cfg.log_every == 0:
            steps_log.append(step + 1)
            losses.append(running / cfg.log_every)
            running = 0.0

    return TrainResult(steps_log, losses)


@torch.no_grad()
def predict_coeffs(
    model: PSFNet, images: torch.Tensor, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Predict *normalised* coefficients for a batch of normalised images."""
    model = model.to(device).eval()
    return model(build_input(images.to(device), model.variant))


@torch.no_grad()
def coefficient_errors(
    model: PSFNet,
    optics: Optics,
    basis: torch.Tensor,
    mask: torch.Tensor,
    batch: int = 256,
    size: int = 128,
    seed: int = 1234,
    device: torch.device | str = "cpu",
    noise: NoiseConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-mode absolute error in waves on a fresh validation batch.

    Returns ``(mean_abs_error_per_mode, overall_mae)``. Reported in waves rather
    than in normalised units so the numbers are physically meaningful.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    images, coeffs, _ = synth_batch(
        batch, size, optics, basis, mask, noise=noise, generator=generator, device=device
    )
    predicted = predict_coeffs(model, images, device=device)

    from deconv.data import denormalise_coeffs

    error = (denormalise_coeffs(predicted.cpu()) - coeffs.cpu()).abs()
    return error.mean(dim=0), error.mean()
