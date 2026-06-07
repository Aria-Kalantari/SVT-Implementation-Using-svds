# Low-Rank Matrix Completion via Singular Value Thresholding (SVT)

**A numerical-linear-algebra project that reconstructs images from incomplete pixel data by solving a low-rank matrix-completion problem with Singular Value Thresholding, accelerated with a sparse SVD and an adaptive rank schedule.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-numerics-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-svds-8CAAE6?logo=scipy&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-PSNR%2FSSIM-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

---

## Overview

Many real signals — images especially — are approximately low-rank, which means a corrupted or partially observed version can be recovered by finding the lowest-rank matrix consistent with the pixels we *do* see. This project implements that idea directly: it treats image reconstruction as **low-rank matrix completion** and solves it with the **Singular Value Thresholding (SVT)** algorithm, a proximal/iterative-thresholding method for the nuclear-norm relaxation of the rank-minimization problem.

Two engineering choices make it efficient and robust:

- **Sparse SVD (`scipy.sparse.linalg.svds`)** computes only the leading singular triplets each iteration, instead of a full dense SVD.
- **Adaptive rank search with a "kicking" initialization** (`Y₀ = δ · P_Ω(M)`) grows the working rank only as needed and accelerates convergence on heavily masked inputs.

Reconstruction quality is measured with standard image metrics (**PSNR** and **SSIM** via scikit-image) across multiple datasets and missing-data levels.

---

## Results (verified from the notebook)

Reconstruction quality on 50 images per dataset, at 30 / 40 / 50% missing pixels:

| Dataset | 30% missing | 40% missing | 50% missing |
|---|---|---|---|
| **CIFAR-10** | 25.78 dB · 0.904 SSIM | 24.00 dB · 0.858 SSIM | 18.97 dB · 0.651 SSIM |
| **BSDS500** | 27.48 dB · 0.858 SSIM | 26.76 dB · 0.835 SSIM | 24.26 dB · 0.751 SSIM |

As expected for low-rank completion, quality degrades gracefully as the observed fraction shrinks, and the higher-resolution BSDS500 images retain more structure (higher PSNR) than the small 32×32 CIFAR-10 images at the same mask density.

---

## Method

1. **Problem.** Given an image matrix `M` observed only on an index set `Ω`, find a low-rank `X` minimizing the nuclear norm subject to agreeing with `M` on `Ω`.
2. **SVT iteration.** Alternate a singular-value soft-threshold (shrink small singular values to zero, enforcing low rank) with a data step that re-imposes the observed entries.
3. **Sparse SVD.** Use `svds` to compute only the top singular values/vectors needed for the threshold each step.
4. **Adaptive rank + kicking init.** Start from `Y₀ = δ · P_Ω(M)` and increase the retained rank adaptively to converge faster and avoid over/under-shooting the true rank.
5. **Evaluation.** Compare reconstructions to ground truth with PSNR and SSIM across datasets and mask densities.

---

## Tech stack

Python · NumPy · SciPy (`sparse.linalg.svds`) · scikit-image (PSNR/SSIM) · Matplotlib · datasets: CIFAR-10 and BSDS500.

## Repository contents

```text
SVT-Implementation-Using-svds/
└── LRMC_SVT.ipynb      # full implementation, experiments, and figures
```

> The implementation, experiments, and result figures live in `LRMC_SVT.ipynb`. Open it in Jupyter to reproduce the reconstructions and metric tables above.

```bash
pip install numpy scipy scikit-image matplotlib jupyter
jupyter notebook LRMC_SVT.ipynb
```

---

## Skills demonstrated

Numerical linear algebra (SVD, nuclear-norm / rank minimization) · matrix completion & convex relaxation · iterative optimization (singular-value thresholding) · sparse/truncated SVD for efficiency · image reconstruction · quantitative evaluation with PSNR/SSIM · scientific Python (NumPy/SciPy/scikit-image).

---

*Author: Arya Kalantari · [github.com/Aria-Kalantari](https://github.com/Aria-Kalantari)*
