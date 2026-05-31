# Tatyana V3

Residual MLP surrogate model for GENE linear gyrokinetic stability.

Given seven local equilibrium parameters, Tatyana V3 predicts the dominant linear growth rate, mode frequency, and species-resolved linear outputs — all quantities required for quasilinear transport modelling via saturation rules such as SAT3.

Feel free to contact me via flyawaypencil480@gmail.com 

---

## Overview

| | |
|---|---|
| **Inputs** | `kymin, trpeps, shat, q0, omt_i, omt_e, omn` |
| **Outputs** | `gamma, omega, gamma_e, omega_e, gamma_i, omega_i` |
| **Architecture** | Residual MLP — 6 blocks × 256 hidden, SiLU + LayerNorm |
| **Parameters** | ~830 k |
| **Training data** | ~67 k GENE linear samples (ITG + TEM, unstable only) |
| **Loss** | Huber, AdamW + cosine annealing, 600 epochs |

The species-resolved outputs (`gamma_e/omega_e`, `gamma_i/omega_i`) serve as proxies for the SAT3 linear energy weights W_eL and W_iL. The electron-to-ion ratio R = |γ_e / γ_i| distinguishes ITG from TEM-dominated regimes and is intended to feed a future mode-switching function M(R) connecting to the SAT3 nonlinear saturation amplitude.

Peak mode quantities (k_max, γ_max) are not regressed directly; instead `find_peak()` scans the surrogate over a ky grid at inference time, avoiding the discretisation artefacts of a grid-argmax-trained head.

---

## Validation

Evaluated on a held-out 15% split (~10 k samples):

| Output | RMSE | Median Rel. Error |
|---|---|---|
| γ | 0.0235 | 3.38% |
| ω | 0.0631 | 3.01% |
| γ_e | 0.0218 | 3.63% |
| ω_e | 0.0554 | 3.55% |
| γ_i | 0.0223 | 3.40% |
| ω_i | 0.0549 | 3.41% |

![Parity plots](tatyana_v3_eval.png)

---

## Repository structure

```
.
├── tatyana_v3.py          # model definition, training, inference helpers
├── requirements.txt
└── README.md
```

Weight file (`tatyana_v3.pt`) and dataset (`df_clean_reconstructed.tsv`) are not included in this repository.

---

## Training

Place `your_dataset.tsv` in the repo root, then:

```bash
python tatyana_v3.py
```

Expects columns: `kymin, trpeps, shat, q0, omt_i, omt_e, omn, gamma, omega, gamma_e, omega_e, gamma_i, omega_i, is_unstable`.
Saves `tatyana_v3.pt` and `tatyana_v3_scalers.pkl`.

---

## Inference

```python
from tatyana_v3 import load_tatyana_v3, predict, find_peak, mode_ratio
import numpy as np

model, sx, sy = load_tatyana_v3("tatyana_v3.pt", "tatyana_v3_scalers.pkl")

# Single point: [kymin, trpeps, shat, q0, omt_i, omt_e, omn]
X = np.array([[0.3, 0.18, 0.8, 1.4, 6.5, 6.5, 2.5]])
out = predict(model, sx, sy, X)
# out: [[gamma, omega, gamma_e, omega_e, gamma_i, omega_i]]

# Peak mode over ky for given equilibrium [trpeps, shat, q0, omt_i, omt_e, omn]
eq = np.array([0.18, 0.8, 1.4, 6.5, 6.5, 2.5])
kmax, gamma_max = find_peak(model, sx, sy, eq)

# Mode ratio R = |gamma_e / gamma_i| — ITG: R << 1, TEM: R >> 1
R = mode_ratio(model, sx, sy, X)
```

---

## Roadmap afterwards (will be updated)

- [ ] Item 3 — mode-switching function M(R): smooth interpolation between SAT3 ITG (C=3.3) and TEM (C=12.7) coefficients using R = |γ_e / γ_i|
- [ ] Item 4 — add ρ_unit or ion mass number A as input (pending confirmation of dataset availability)
- [ ] Item 5 — physics-constrained spectral shape: predict σ_ky, enforce −2.42 power-law flux integral

---

## Author

Tingyi Chen — NTU Plasma Theory | BSc Physics Jilin University
`flyawaypencil480@gmail.com`
