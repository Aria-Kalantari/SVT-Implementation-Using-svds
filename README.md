# Low-Rank Matrix Completion via Singular Value Thresholding (SVT)

Implements the Singular Value Thresholding (SVT) algorithm using **Sparse SVD** and an
**adaptive rank-search** strategy to recover low-rank matrices from incomplete data, applied
to image reconstruction. Bachelor's thesis project.

## Why This Project Matters

Many real-world matrices (images, recommender data, sensor readings) are approximately
low-rank, so missing entries can be recovered by finding the lowest-rank matrix consistent
with the observed values. SVT solves a nuclear-norm relaxation of this problem iteratively;
using a sparse/truncated SVD and searching for the right rank each iteration keeps the
computation tractable at higher dimensions.

## Tech Stack

- **Language:** Python (Jupyter Notebook)
- **Methods:** Singular Value Thresholding, Sparse/truncated SVD, adaptive rank search, low-rank matrix completion, image reconstruction
- **Libraries:** NumPy, SciPy *(confirm exact imports in the notebook)*

## Key Features

- SVT iteration built on a **sparse SVD** rather than a full dense SVD.
- **Adaptive rank-search** strategy to track the effective rank across iterations.
- Reconstruction of images from partially observed pixels (masked entries).
- Study of convergence behavior under varying mask densities (amount of missing data).

## Repository Structure

```text
SVT-Implementation-Using-svds/
└── LRMC_SVT.ipynb   # SVT implementation, experiments, and figures
```

## How to Run

```bash
# Open the notebook in Jupyter
pip install numpy scipy matplotlib jupyter   # confirm against notebook imports
jupyter notebook LRMC_SVT.ipynb
```

## Example Output / Results

To be added after verification. *(Resume references PSNR/SSIM on standard image datasets at
30–50% missing data — add the exact datasets, metrics, and reconstructed-image figures here
once confirmed from the notebook outputs.)*

## What I Learned

- Applying nuclear-norm / SVT optimization to a practical recovery problem.
- Using sparse/truncated SVD to make iterative low-rank methods scale.
- Reasoning about rank, thresholding, and convergence under different observation rates.

## Future Improvements

- Add a quantitative results table (datasets, missing-data ratios, PSNR/SSIM) once verified.
- Refactor the notebook into a small reusable module + a script entry point.
- Add a `requirements.txt` and a couple of saved example figures.

## Limitations

- Research/coursework prototype, not an optimized library.
- Results are currently inside the notebook; summary metrics still to be surfaced here.
