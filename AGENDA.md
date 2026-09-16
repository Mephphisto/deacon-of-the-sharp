# Agenda — ML Microscopy Deconvolution Demo

## Goal

Build a Jupyter notebook (`deconvolution_demo.ipynb`) that demonstrates training and
evaluating a machine-learning model for the deconvolution of 2D microscopy images.

A CNN observes a blurred image and predicts the **parameters of the point-spread
function (PSF)** that produced the blur. From those parameters a **regularized
(Wiener) inverse filter** is constructed in closed form and applied to the blurred
image, sharpening it.

The pipeline therefore has two stages, and only the first is learned:

> blurred image → `[CNN]` → PSF parameters → `[closed-form Wiener inverse]` → sharpened image

The notebook is **didactic, written for a physicist rather than a programmer**, and
follows the physics narrative:

> forward model → why inversion is ill-posed → classical attempt →
> "but we don't know the PSF" → learn it → close the loop

Scope: **proof of concept / demo**, not product development.

### Why the model predicts the forward PSF, and why its inverse must be regularized

An exact inverse kernel does not exist — for physical rather than numerical reasons:

- The OTF has a **hard cutoff** at `2·NA/λ`, above which it is exactly zero. Those
  spatial frequencies were destroyed by the microscope; no inverse recovers them.
- Aberrated and defocused OTFs have additional **interior zeros** inside the passband.
- Where the OTF is small but nonzero, `Y = K·X + N` inverts to `Y/K = X + N/K`, so
  **noise is amplified without bound**.

Only regularized inverses (Wiener, Tikhonov) are well defined, which is why the
learned quantity is the **forward** PSF: it is bounded, low-dimensional and
physically meaningful, and the regularized inverse is then derived from it with λ
chosen at inference. It also feeds Richardson–Lucy directly, which needs only the
forward PSF.

Note that sharpening applies the **regularized inverse**, not the kernel itself —
applying the kernel would blur the image further. Both points are core teaching
moments of the notebook rather than incidental details.

---

## Locked decisions

| Topic | Decision |
|---|---|
| Dimensionality | 2D only |
| PSF model | Scalar diffraction: `P = A(ρ)·exp(i2π·W)`, `W = Σ cⱼZⱼ`, `PSF = \|FFT(P)\|²` |
| Parametrization | Zernike **Z4–Z11** (8 coefficients): defocus, astigmatism ×2, coma ×2, trefoil ×2, spherical |
| Excluded terms | Piston (invisible) and tip/tilt (pure image shift — would make labels shift-ambiguous) |
| Airy disk | The `all cⱼ = 0` special case of the same generator — used as sanity check |
| Coefficient range | ±0.5λ nominal (±1λ optional for a more dramatic demo) |
| Architecture | **CNN** (~6–8 conv blocks, ~2–5M params) → global average pooling → MLP head → 8 coefficients |
| Transformer | Rejected — ViT is data-hungry and would underperform at this scale, yielding a conclusion that is an artifact of the compute budget |
| Learning approach | **Approach A**: supervised regression, loss = `‖ĉ − c_true‖²` |
| Rejected losses | Differentiable-deconvolution loss (ill-conditioned near OTF zeros, PSF signal diluted by object content, and discards a perfect free label) |
| Comparison axis | **image-only vs. image + log-power-spectrum** as input representation (2 training runs) |
| Image noise | Poisson shot + Gaussian read noise, **randomized** across samples |
| Wavefront noise | None — adds unidentifiable degrees of freedom, no pedagogical gain |
| PSF assumption | Spatially invariant |
| Training data | Synthetic bead fields, generated **on GPU** inside the training loop (`torch.fft`) |
| Real data | EPFL Biomedical Imaging Group — used as **transfer test**, not training corpus |
| Working resolution | 128² for training runs; single 256² run for final figures |
| Framework | PyTorch |
| Classical baselines | `scikit-image` — Wiener, `unsupervised_wiener`, Richardson–Lucy |
| Hardware target | RTX 5070 Ti + i9-9900K + 32 GB |
| Runtime target | Full notebook < 4 h end-to-end; demo may be shown from committed outputs without live re-run |
| Reproducibility | Fixed seeds, pinned `requirements.txt`, notebooks committed **with executed outputs** |
| Relevant skills | `show-me` (diagrams), `ponytail` (minimal didactic code) |

### Why noise randomization is free under Approach A

Because the label is the Zernike coefficient vector rather than an inverse kernel,
the label no longer depends on SNR. Under inverse-kernel regression the "correct"
inverse would vary per-sample with the noise level, making labels ambiguous.
Approach A dissolves that problem.

### Why the CPU is the bottleneck, not the GPU

The i9-9900K (8C/16T) would starve the GPU if PSF synthesis and convolution ran
per-sample in a CPU DataLoader. Doing the whole forward model on GPU via
`torch.fft` gives effectively infinite non-repeating data, zero disk I/O, and no
dataloader stalls — and is only a few lines, since the forward model is just FFTs.

---

## Phase 1 — Research

- [ ] Set up skills for the next phase (`show-me`, `ponytail`)
- [ ] Topic research: Fourier optics, PSF/OTF relationship, Zernike basis,
      Wiener vs. Richardson–Lucy, deep-learning wavefront sensing / adaptive optics
- [ ] Library research: PyTorch, `scikit-image`, Zernike basis generation,
      Gibson–Lanni PSF references
- [ ] **Verify the EPFL dataset**: availability, current host, format, size, licence
      (redistributable or download-script-only?)
- [ ] Fix the regularized-inverse convention: Wiener vs. Tikhonov, and how λ is chosen
- [ ] Confirm PSF sampling meets Nyquist (≥ 2–3 px across the Airy FWHM)
- [ ] Hyperparameter starting points (optimizer, LR schedule, batch size, augmentation)
- [ ] Write `RESEARCH.md`

**GATE: manual verification of `RESEARCH.md`**

---

## Phase 2 — Plan

- [ ] Mermaid graph of the ML pipeline and the physics forward/inverse model
- [ ] Software architecture and module layout
- [ ] Didactic notebook layout following the physics narrative (not the usual
      data → model → train → eval ordering)
- [ ] Train/test discipline: **disjoint source images** between splits so the model
      cannot memorize image content
- [ ] Decide data handling: commit sample images vs. download script + cache
- [ ] Write `PLAN.md`

**GATE: manual verification of `PLAN.md`**

---

## Phase 3 — Build

- [ ] GPU-side PSF generator and forward model (Zernike → pupil → PSF → convolve → noise)
- [ ] Classical baseline first: Richardson–Lucy / Wiener with a **known** PSF
      *(cheap sanity gate — if this doesn't visibly sharpen, nothing downstream will)*
- [ ] CNN model and training loop
- [ ] Both input variants: image-only and image + log-power-spectrum
- [ ] `deconvolution_demo.ipynb`
- [ ] `requirements.txt` (pinned)

**GATE: manual verification of `deconvolution_demo.ipynb`**

---

## Phase 4 — Evaluate

- [ ] **Overfit-one-batch smoke test** — gate: near-zero loss on ~8 samples,
      otherwise there is a bug
- [ ] Full training runs, both input variants. Stopping / hyperparameter-tuning
      criterion: the supervised coefficient loss (sufficient for a PoC)
- [ ] Compare image-only vs. image + spectrum result quality
- [ ] **Real-world verification section** — evaluate the actual image improvement,
      not just regression quality:
  - deconvolution using the **predicted** PSF
  - vs. the **true** PSF (oracle ceiling)
  - vs. a **default/wrong** PSF (naive baseline)
  - metrics: PSNR, SSIM, optionally Fourier Ring Correlation for a resolution
    figure in nm
- [ ] **Transfer test** on EPFL real confocal data — trained on beads, does it
      work on real biology? Report the sim-to-real gap honestly.

**GATE: manual verification of results**

---

## Open risks

| Risk | Contingency |
|---|---|
| EPFL dataset too small to train on | Already the plan — beads-primary, EPFL as transfer test only |
| EPFL licence prohibits redistribution | Download script + local cache; commit few or no images |
| Bead-trained model doesn't transfer to real biology | This is a **finding to report**, not a failure. The sim-to-real gap is part of the demo's honesty. |
| 4 h budget overrun | 128² working resolution keeps both runs at ~15–20 min each; 256² reserved for final figures only |
| Model collapses to a degenerate PSF | Mitigated by construction: bounded 8-dim parametrization with no shift/scale ambiguity |
