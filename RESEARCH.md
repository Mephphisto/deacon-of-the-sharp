# Research — ML Microscopy Deconvolution Demo

Phase 1 output. Resolves the open parameters from `AGENDA.md` and records the
physics, conventions, libraries and starting hyperparameters for Phase 2 (Plan).

---

## 0. Gate decisions — RESOLVED

| # | Decision | Resolution |
|---|---|---|
| **D1** | **Real-data source.** EPFL is egress-blocked here, almost entirely **3D** (only the Siemens star is 2D), and partly email-gated (§6.2). | **`skimage.data` bundled CC-0 microscopy.** EPFL Siemens star optional manual download. |
| **D2** | **Training data mix**, and whether real data belongs in training at all. | **Two runs.** Runs 1–2 train fully synthetic (50 % beads / 50 % extended) and *measure* the sim-to-real gap on held-out real data. Run 3 adds 10 % real `cells3d()` slices and shows whether the gap closes. |
| **D3** | **Geometric augmentation** — flips/rotations are not label-preserving (§8.3). | **None.** |

### Train/test split (fixed by D2)

Keeping these disjoint is what makes the transfer test mean anything:

| Split | Source |
|---|---|
| Train — synthetic | Bead fields + extended structures, generated fresh on GPU |
| Train — real (**Run 3 only**) | `skimage.data.cells3d()` — 60 z × 2 channels = **120 distinct 256² slices** |
| **Held out** | `skimage.data.human_mitosis()` (512²), `skimage.data.kidney()` (16 z × 3 ch = 48 slices, 512²) |

The three runs:

| Run | Input variant | Data | Measures |
|---|---|---|---|
| 1 | image-only | 100 % synthetic | Baseline + sim-to-real gap |
| 2 | image + spectrum | 100 % synthetic | Whether the Fourier channel helps |
| 3 | *winner of 1 vs 2* | + 10 % real `cells3d()` | Whether real data closes the gap |

Run 3 differs from its own baseline **only** in the data mix, so the comparison
isolates one variable.

### Caveat to state in the notebook

Real microscopy images are **already blurred by the microscope that acquired
them**. Using them as "sharp" sources means the synthetic PSF is an *additional*
blur on top of an unknown existing one. This is standard practice in the
literature and fine for a PoC, but a physicist will notice immediately — so say
it rather than let it be caught.

---

## 1. Physics foundation

### 1.1 Forward model

Incoherent imaging of a fluorescent sample is, under the spatially-invariant
assumption, a convolution:

```
y(x) = (h * o)(x) + n(x)
```

- `o` — the true object (fluorophore distribution)
- `h` — the intensity PSF
- `n` — noise (Poisson shot + Gaussian read)

In Fourier space this becomes a multiplication, which is the whole reason the
problem is tractable:

```
Y(f) = H(f)·O(f) + N(f)
```

`H(f)` is the **optical transfer function (OTF)**, normalised so `H(0) = 1`.

### 1.2 PSF from the pupil

Scalar diffraction theory, which is the standard model and entirely adequate here:

```
Pupil:      P(ρ,θ) = A(ρ) · exp(i·2π·W(ρ,θ))
Aperture:   A(ρ) = 1 for ρ ≤ 1, else 0
Wavefront:  W(ρ,θ) = Σⱼ cⱼ · Zⱼ(ρ,θ)          [units: waves]
Amplitude:  a = FFT(P)
Intensity:  h = |a|² ,  normalised so Σh = 1
OTF:        H = FFT(h)
```

With all `cⱼ = 0` this produces the **Airy disc** exactly — so the aberration-free
case is the zero-coefficient special case of the same generator, not a separate
code path. This is the natural sanity check.

### 1.3 Why inversion is ill-posed

Three distinct reasons, all physical:

1. **Hard cutoff.** The intensity OTF has finite support, vanishing identically
   above `f_c = 2·NA/λ`. The intensity PSF is `|a|²`, so its transform is the
   autocorrelation of the pupil — which is why the cutoff is `2·NA/λ` and not
   `NA/λ`. Above `f_c` the microscope transmitted nothing; no algorithm recovers it.
2. **Interior zeros.** Defocused and aberrated OTFs oscillate and cross zero
   *inside* the passband. Additional bands are lost.
3. **Noise amplification.** Where `H` is small but nonzero,
   `Y/H = O + N/H`, and the noise term diverges.

The notebook should *show* all three: plot the OTF with its cutoff, plot an
aberrated OTF with visible interior zeros, and show naive inverse filtering
producing garbage. That failure is the motivation for everything after it.

---

## 2. Zernike parametrization

### 2.1 Convention: Noll, and pin it with a test

**The web is genuinely unreliable here.** During this research, sources returned
mutually contradictory Noll tables within a single page — one listing `j=4`
as defocus, another listing `j=3` as defocus and `j=8` as spherical. The OSA/ANSI
scheme numbers these differently again (it is 0-indexed and orders `m` differently).

Mixing conventions silently corrupts **every label in the dataset** while training
loss still looks plausible. This is the highest-value place in the project for an
assertion.

**Canonical Noll (1976) ordering**, which is what `AGENDA.md` specifies:

| Noll `j` | `(n, m)` | Aberration |
|---|---|---|
| 1 | (0, 0) | Piston — *excluded* (constant phase, invisible) |
| 2 | (1, 1) | Tip / x-tilt — *excluded* (pure image shift) |
| 3 | (1, −1) | Tilt / y-tilt — *excluded* (pure image shift) |
| **4** | (2, 0) | **Defocus** |
| **5** | (2, −2) | **Oblique astigmatism** |
| **6** | (2, 2) | **Vertical astigmatism** |
| **7** | (3, −1) | **Vertical coma** |
| **8** | (3, 1) | **Horizontal coma** |
| **9** | (3, −3) | **Vertical trefoil** |
| **10** | (3, 3) | **Oblique trefoil** |
| **11** | (4, 0) | **Primary spherical** |

Ordering rule: ascending radial order `n`, then ascending `|m|`; even `j` takes the
`cos(mθ)` term, odd `j` the `sin(|m|θ)` term.

**Label vector: `c ∈ ℝ⁸`, Noll j = 4…11.**

### 2.2 Required unit test (Phase 3)

Assert orthonormality over the unit disc and verify each mode's identity:

- `⟨Zᵢ, Zⱼ⟩ = δᵢⱼ` over `ρ ≤ 1` (Noll's normalisation)
- `Z4` is rotationally symmetric; `Z5/Z6` have 2-fold symmetry; `Z9/Z10` 3-fold
- A pure `+Z4` and a pure `−Z4` must produce **identical** PSFs (defocus is
  symmetric in sign for the intensity PSF) — a useful degeneracy to be aware of,
  see §2.3

### 2.3 Known degeneracy — worth stating in the notebook

For a *single in-focus intensity* image, the PSF is `|FFT(P)|²`. This has a parity
degeneracy: wavefronts `W(ρ,θ)` and `−W(−ρ,−θ)` produce the same intensity PSF.
In practice this makes the **sign of even-symmetry modes (notably defocus `Z4`)
weakly identifiable** from one in-focus image.

Mitigations, in increasing order of effort:
- Accept it, and **restrict `Z4` to one sign** in the generator (simplest; recommended for a PoC)
- Report per-coefficient error and show `Z4` is the weak one (honest, and a good teaching moment)
- Use phase diversity — two images, one deliberately defocused (this is what the
  literature does; out of scope here)

Recommendation: **restrict `Z4 ≥ 0`** and note the reason. It removes an
irreducible ambiguity that would otherwise put a floor on the loss and look like
a training failure.

---

## 3. Optical parameters (pinned)

Realistic high-NA fluorescence configuration:

| Parameter | Value | Note |
|---|---|---|
| NA | **1.4** | Oil immersion, high-end objective |
| λ (emission) | **520 nm** | GFP |
| Pixel size | **65 nm** | 100× onto 6.5 µm sCMOS — very standard |
| Image grid | **128²** (train), 256² (final figures) | |
| Pupil grid | **256²** | |
| Pupil radius | **45 px** | Derived below |

### 3.1 Derived quantities

```
Airy radius (1st zero)  r  = 0.61·λ/NA = 0.61 × 520/1.4 = 226.6 nm
PSF FWHM                    ≈ 0.51·λ/NA = 0.51 × 520/1.4 = 189.4 nm
OTF cutoff              f_c = 2·NA/λ    = 5.385 µm⁻¹
Smallest period         1/f_c                            = 185.7 nm
Nyquist pixel limit     λ/(4·NA)                          =  92.9 nm
```

### 3.2 Nyquist check — passes

```
Chosen pixel 65 nm  ≤  92.9 nm limit            ✓  oversampled 1.43×
FWHM / pixel = 189.4 / 65 = 2.91 px across FWHM ✓  meets the ≥2–3 px criterion
Airy radius  = 226.6 / 65 = 3.49 px
```

Field of view: 128 × 65 nm = **8.3 µm** (cell-scale), 256² → 16.6 µm.

### 3.3 Pupil grid sizing — derivation

The pupil edge sits at spatial frequency `NA/λ`. Sampling the pupil radius with
`R` grid points gives `Δf = NA/(λ·R)`. An `N`-point FFT satisfies `Δx·Δf = 1/N`, so

```
Δx = λ·R / (N·NA)     ⟹     R/N = Δx·NA/λ
```

For `Δx` = 65 nm, NA = 1.4, λ = 520 nm:

```
R/N = 65 × 1.4 / 520 = 0.175     ⟹     N = 256  →  R ≈ 45 px
```

Sanity cross-check: critical sampling (`Δx = λ/4NA`) gives `R/N = 0.25`, i.e. pupil
diameter exactly half the grid — the familiar rule. We sit inside that, consistent
with being 1.43× oversampled. A pupil radius of 45 px is comfortably enough to
represent Zernikes up to `n = 4` without aliasing the high-order modes.

---

## 4. Regularized inverse convention

### 4.1 Wiener (primary)

```
G(f) = conj(H(f)) / ( |H(f)|² + λ_reg )
ô    = IFFT( G · Y )
```

The `+λ_reg` term is what makes this well-defined — it bounds the gain where
`H → 0` and removes any division-by-zero path. Chosen over Tikhonov because it is
the MMSE linear estimator and `λ_reg` has a direct reading as inverse SNR.

**λ_reg selection.** For the demo, set it from the known simulated noise level:

```
λ_reg ≈ 1 / SNR  ,  estimated as σ_noise² / σ_signal²
```

Sweep `λ_reg` over a decade in the notebook and show the bias/variance trade-off
— under-regularised is noisy, over-regularised is blurry. This is a good figure
and costs one line.

### 4.2 Richardson–Lucy (secondary baseline)

Worth including because it is what microscopists actually use:

- Needs only the **forward** PSF — another argument for predicting the forward model
- Enforces non-negativity and assumes Poisson statistics, which matches the real
  noise model
- Iteration count *is* the regularisation; it diverges (amplifies noise) if run too long

`skimage.restoration.richardson_lucy(image, psf, num_iter=50, clip=True,
filter_epsilon=None)`. Note the parameter is `num_iter` in current versions
(older code uses `iterations`).

---

## 5. Model architecture and hyperparameters

### 5.1 Architecture

```
input  (1 or 2 × 128 × 128)
  → 6–8 conv blocks [Conv3×3 → BatchNorm → ReLU], channels 32→64→128→256, stride-2 downsampling
  → Global Average Pooling
  → MLP head (256 → 128 → 8)
output c ∈ ℝ⁸
```

Global average pooling is the right choice: the aberration is a *global* property
of the image, so the head should see a whole-image summary, not a local patch.
Parameter count lands around 2–4 M.

### 5.2 Input variants (the comparison axis)

| Variant | Channels |
|---|---|
| **A: image-only** | normalised image |
| **B: image + spectrum** | normalised image, `log(1 + \|FFT(y)\|²)` fftshifted |

Rationale for B: the OTF's signature — especially its zero rings — is far more
directly visible in Fourier space than in the image. Same intuition as cepstral
blur identification. Handle the DC spike (it dominates by orders of magnitude):
log-scale, then per-image standardise.

### 5.3 Starting hyperparameters

| Setting | Value |
|---|---|
| Optimizer | AdamW, `lr = 3e-4`, `weight_decay = 1e-4` |
| Schedule | Cosine annealing, ~500-step linear warmup |
| Batch size | 64 at 128² |
| Steps | ~30 k (data is generated fresh, so think in steps, not epochs) |
| Precision | bf16 autocast (Blackwell handles this well) |
| Loss | MSE on **normalised** coefficients — see below |
| Augmentation | **None** (see D3 / §8.3) |

**Coefficient normalisation matters.** Divide each `cⱼ` by its sampling range so
all 8 modes contribute comparably to the loss. Without this, whichever mode has
the largest numeric range dominates the gradient and the rest train poorly.

### 5.4 Runtime estimate

~3 M params, 128², batch 64, bf16 on a 5070 Ti → roughly 20–30 ms/step, so
**30 k steps ≈ 10–15 min per run**. Two runs ≈ 30 min. The 256² figure run adds
maybe 15 min. Comfortably inside the 4 h budget with room for the sweep in §4.1.

---

## 6. Training and evaluation data

### 6.1 Synthetic sources (primary)

**Bead fields** — sub-diffraction point emitters:

| Property | Value |
|---|---|
| Beads per 128² field | 30–150, log-uniform |
| Bead size | Sub-diffraction (single pixel, or σ ≪ PSF) |
| Intensity | Log-uniform over ~1 decade |
| Placement | Uniform random, sub-pixel positions |
| Background | Small constant offset |

Beads are pedagogically ideal — a point source *is* the PSF, so the aberration is
directly visible to the eye.

**Resolved per D2.** A bead-only model sees image statistics utterly unlike real
biology (sparse impulses vs. dense extended texture) and would likely transfer
badly. The literature reflects this concern — recent work explicitly addresses
Zernike prediction from "PSFs **and extended images**". Training is therefore
**50 % beads / 50 % synthetic extended structures** (random filaments, blobs,
discs). It is free — all synthetic — and substantially de-risks the transfer test.

**Noise model** (applied after convolution, randomized per sample):

```
photons   ~ log-uniform peak signal, ~50–2000
shot      : Poisson
read       : Gaussian, σ ≈ 1–5 ADU
```

### 6.2 Real data — EPFL findings

Investigated per the agenda. Three problems:

1. **Network-blocked here.** `bigwww.epfl.ch` is blocked by this container's
   egress proxy, so I could not fetch, verify or download it. (I attempted to check
   the proxy status for a per-domain fix; that command was denied by the permission
   classifier as credential exploration, and I did not work around it. If you want
   this pursued, it needs your approval.)
2. **Almost entirely 3D.** The collection is built around the ISBI 3D
   deconvolution challenge:
   - *Cube of spherical beads* — 128³ synthetic, **3D**
   - *Microtubules* — synthetic GFP-channel network, **3D**
   - *C. elegans embryo* — real, three stacks, **3D**
   - *Sinusoidal Siemens star* — **2D** ← the only 2D item
3. **Partly email-gated.** Ground truth, PSF and results are described as
   available *on request by email* for some datasets — not scriptable, and the
   licence terms are not stated publicly.

**This does not disqualify 3D data in principle** — since we synthesise the blur
ourselves, we only need *sharp 2D sources*, and an XY slice from a clean 3D stack
serves fine. The blockers are access and licence, not dimensionality.

**Recommended (D1): `skimage.data`**, which ships real fluorescence microscopy
bundled with the library:

- `skimage.data.human_mitosis()` — human cells in mitosis, nuclear DNA stain,
  **CC-0**, credited to David Root via CellProfiler
- `skimage.data.cell()`, `skimage.data.kidney()` also available

Zero download, zero egress problem, zero licence ambiguity, already a pinned
dependency. For a PoC transfer test this is strictly better engineering than a
gated benchmark archive.

**Optional extra:** the EPFL **Siemens star** is a genuinely superb didactic
object — a resolution test target whose converging spokes make the resolution
limit *visible*, and deconvolution visibly pushes the unresolvable radius inward.
If you download it manually, it earns its place as a figure. It is not needed for
the pipeline to work.

---

## 7. Evaluation metrics

| Metric | Space | Purpose |
|---|---|---|
| Per-coefficient MAE (in waves) | Coefficient | Training/stopping criterion; shows *which* modes are hard |
| PSNR | Image | Standard restoration quality |
| SSIM | Image | Perceptual/structural quality |
| FRC | Image | Resolution in nm — the physicist-friendly number |

**The money plot** — deconvolve three ways and compare:

1. **Predicted** PSF ← what the model achieves
2. **True** PSF ← oracle ceiling
3. **Default/wrong** PSF (e.g. aberration-free Airy) ← naive baseline

The gap between 1 and 2 is the model's real cost; the gap between 1 and 3 is its
real value.

**FRC caveat.** Proper Fourier Ring Correlation needs *two independent noisy
realisations* of the same scene. With simulation this is easy — generate the same
object and PSF twice with independent noise draws. Do not compute FRC between a
reconstruction and its own ground truth and call it a resolution measurement.

---

## 8. Libraries

### 8.1 Stack

| Package | Role | Note |
|---|---|---|
| `torch` | Model, and the **whole forward model on GPU** via `torch.fft` | Needs a CUDA build matching the 5070 Ti (Blackwell, sm_120) |
| `numpy` | Array basics | |
| `scipy` | Misc numerics | |
| `scikit-image` | Classical baselines, and bundled real images | 0.26.x current |
| `matplotlib` | Figures | |
| `jupyter` / `notebook` | The deliverable | |
| `tqdm` | Progress | |

**Explicitly not needed:** no Zernike library. The 8 Noll polynomials for
`n ≤ 4` are short closed-form expressions — writing them out directly is fewer
lines than a dependency, avoids importing someone else's indexing convention (the
exact hazard in §2.1), and lets a physicist *read the formula* in the notebook.
This is the right call pedagogically as well as practically.

`microscPSF` / Gibson–Lanni is likewise unnecessary — that model's value is in 3D
depth-dependent aberration, and we locked 2D.

### 8.2 Environment caveat

This container has **Python 3.11.15 and none of the scientific stack installed,
and no GPU**. Code can be written and CPU-smoke-tested here after installing
dependencies, but the real training runs must happen on the 5070 Ti. Plan phase
should decide whether to CPU-verify at tiny scale here first.

Pin the CUDA build to one that supports Blackwell — this needs checking against
the user's driver at build time.

### 8.3 Augmentation hazard (D3)

Standard flips/rotations are **not label-preserving**. Rotating the image by θ
rotates the wavefront, which maps the Zernike coefficients onto each other:
`Z5/Z6` mix, `Z7/Z8` mix, `Z9/Z10` mix, while `Z4` and `Z11` (rotationally
symmetric) are invariant. Flips additionally flip the sign of the sine modes.

Applying naive augmentation therefore **corrupts labels silently** — training
proceeds, loss decreases, and the model learns a smeared average.

Since synthesis is free and unlimited, there is no reason to augment. **Use none.**
(If it were ever wanted, the correct approach is to rotate the *wavefront* in the
generator, which is already handled by sampling coefficients freshly.)

---

## 9. References

- Noll, R. J. (1976), *Zernike polynomials and atmospheric turbulence*, JOSA 66(3) — the indexing convention
- Sage et al., *DeconvolutionLab2: An open-source software for deconvolution microscopy*, Methods (2017) — [bigwww.epfl.ch/deconvolution/](https://bigwww.epfl.ch/deconvolution/)
- *Deep learning wavefront sensing*, Optics Express 27(1), 240 — [opg.optica.org](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-1-240&id=403519)
- *Direct Zernike Coefficient Prediction from Point Spread Functions and Extended Images using Deep Learning* — [arXiv:2404.15231](https://arxiv.org/pdf/2404.15231) — directly relevant to the bead→extended-object transfer question (D2)
- *Focal Plane Wavefront Sensing using Machine Learning: Performance of CNNs compared to Fundamental Limits* — [arXiv:2106.04456](https://arxiv.org/pdf/2106.04456)
- Weigert et al. (2018), *Content-aware image restoration* (CARE), Nature Methods — the direct image→image alternative we deliberately did not take
- scikit-image restoration API — [scikit-image.org](https://scikit-image.org/docs/stable/api/skimage.restoration.html)
- `skimage.data` catalogue — [scikit-image.org](https://scikit-image.org/docs/stable/api/skimage.data.html)

---

## 10. Changes applied to `AGENDA.md`

D1–D3 accepted; `AGENDA.md` updated accordingly:

- **Real data** row → `skimage.data` bundled images, held-out set named, EPFL Siemens star optional
- **Training data** row → 50 % beads / 50 % synthetic extended structures
- **New locked row** → Zernike convention = Noll `j = 4…11`, `Z4 ≥ 0`
- **New locked row** → no geometric augmentation
- **New locked row** → optics pinned (NA 1.4, 520 nm, 65 nm px, 256² pupil, R = 45 px)
- **Phase 1** → all items checked, gate marked passed
- **Phase 4** → Run 3 added; transfer test retargeted to the held-out `skimage` images
- **Open risks** → EPFL rows dropped; added Zernike-convention corruption, defocus-sign
  degeneracy, and the already-blurred-real-images caveat
