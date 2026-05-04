"""
Sample rankings + Bayesian latent-space fit (Plackett–Luce) following:

  • Gormley & Murphy / “paper17” latent geometry + Metropolis–Hastings +
    Procrustean post-processing (MAP reference, orthogonal rotation after centering).

  • Main (11) / oboe-multiphonics extension: per-(participant, trial) ideal points
    y_sr, item locations x_j, baseline appeals c_j, sensitivities b_s > 0,
    η_sr,j = c_j − b_s * ‖y_sr − x_j‖² with ‖·‖² the GM06 mean-squared distance
    (divide squared Euclidean by D), priors N(0,I) on locations, N(0,1) on c_j,
    Gamma(25, 1/24) on b_s, and log-scale random-walk proposals for b_s.

Run this file: it writes multiphonics_rankings.csv, fits the model for D in {1,2,3},
prints posterior draw tables, and opens interactive plots (dimension pair + traces).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Simulate toy ranking data (same structure as before)
# -----------------------------------------------------------------------------
# np.random.seed(42)

# participants = 15
# trials = 15
# items = 7

# rows = []
# for p in range(1, participants + 1):
#     for t in range(1, trials + 1):
#         ranking = np.random.permutation(np.arange(1, items + 1))
#         rows.append([p, t] + ranking.tolist())

# df = pd.DataFrame(rows, columns=["participant", "trial"] + [f"m{i}" for i in range(1, 8)])
# df.to_csv("multiphonics_rankings.csv", index=False)



# -----------------------------------------------------------------------------
# 1. Load provided ranking data (REAL DATA)
# -----------------------------------------------------------------------------

import numpy as np

raw_data = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [2, 1, 5, 6, 3, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [1, 5, 3, 4, 2, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 7, 1, 4, 3, 6, 5],
    [2, 4, 1, 7, 3, 5, 6],
    [6, 5, 4, 1, 2, 3, 7],
    [5, 6, 2, 3, 4, 7, 1],
    [4, 3, 6, 2, 5, 1, 7],
    [4, 2, 6, 5, 3, 1, 7],
    [5, 6, 3, 1, 2, 4, 7],
    [4, 5, 6, 1, 2, 3, 7],
    [2, 3, 1, 7, 4, 5, 6],
    [5, 7, 3, 1, 2, 6, 4],
    [3, 6, 2, 5, 4, 7, 1],
    [4, 1, 5, 6, 2, 3, 7],
    [4, 1, 5, 6, 3, 2, 7],
    [2, 4, 3, 6, 1, 5, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 7, 1, 5, 3, 6, 4],
    [1, 4, 2, 5, 3, 6, 7],
    [4, 6, 5, 1, 3, 2, 7],
    [3, 6, 2, 5, 4, 7, 1],
    [5, 3, 6, 2, 4, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [3, 6, 4, 1, 2, 5, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 7, 1, 2, 3, 6, 5],
    [4, 6, 2, 5, 3, 7, 1],
    [4, 1, 5, 6, 3, 2, 7],
    [4, 1, 6, 5, 2, 3, 7],
    [3, 4, 2, 5, 1, 6, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [3, 6, 1, 5, 2, 7, 4],
    [1, 4, 2, 7, 3, 6, 5],
    [3, 6, 5, 1, 4, 2, 7],
    [3, 7, 2, 5, 4, 6, 1],
    [5, 2, 6, 4, 3, 1, 7],
    [5, 2, 4, 6, 3, 1, 7],
    [3, 5, 6, 1, 2, 4, 7],
    [4, 5, 6, 2, 3, 1, 7],
    [1, 4, 3, 6, 2, 5, 7],
    [3, 7, 4, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [2, 1, 5, 6, 4, 3, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 6, 1, 5, 3, 7, 4],
    [1, 4, 2, 5, 3, 7, 6],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 6, 2, 5, 4, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 4, 2, 6, 3, 5, 7],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 4, 2, 7],
    [4, 1, 5, 6, 3, 2, 7],
    [3, 6, 2, 4, 1, 5, 7],
    [2, 6, 1, 5, 4, 7, 3],
    [3, 7, 1, 4, 2, 6, 5],
    [1, 4, 2, 5, 3, 6, 7],
    [5, 4, 6, 1, 2, 3, 7],
    [4, 6, 2, 3, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [6, 5, 4, 1, 2, 3, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 4, 2, 6, 3, 5, 7],
    [5, 7, 3, 1, 2, 6, 4],
    [3, 6, 2, 5, 4, 7, 1],
    [4, 1, 5, 6, 3, 2, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [1, 6, 2, 5, 3, 4, 7],
    [2, 5, 1, 7, 3, 6, 4],
    [2, 7, 1, 5, 4, 6, 3],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 5, 6, 1, 2, 3, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 3, 6, 4, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 5, 6, 1, 2, 3, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 4, 2, 6, 3, 5, 7],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 7, 2, 5, 4, 6, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 5, 1, 7, 4, 6, 3],
    [2, 7, 1, 5, 3, 6, 4],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 6, 5, 1, 3, 2, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 4, 1, 2, 3, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 7, 3, 1, 2, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [3, 6, 1, 5, 4, 7, 2],
    [1, 4, 2, 5, 3, 7, 6],
    [4, 6, 5, 1, 2, 3, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 4, 1, 2, 3, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 3, 2, 5, 4, 7, 6],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 4, 5, 7, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 1, 4, 3, 6, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 6, 1, 4, 3, 7, 5],
    [1, 4, 2, 5, 3, 6, 7],
    [5, 6, 4, 1, 2, 3, 7],
    [4, 6, 2, 3, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 5, 6, 3, 2, 7],
    [5, 6, 3, 1, 2, 4, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 3, 2, 7, 4, 5, 6],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 4, 5, 7, 1],
    [2, 1, 4, 6, 5, 3, 7],
    [4, 1, 5, 6, 3, 2, 7],
    [1, 5, 3, 4, 2, 6, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 7, 1, 4, 3, 6, 5],
    [1, 4, 2, 6, 3, 7, 5],
    [4, 6, 5, 1, 2, 3, 7],
    [3, 6, 2, 5, 4, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [4, 2, 6, 5, 3, 1, 7],
    [3, 6, 5, 2, 1, 4, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [3, 7, 4, 1, 2, 5, 6],
    [3, 6, 2, 4, 5, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [5, 2, 6, 4, 3, 1, 7],
    [1, 6, 3, 4, 2, 5, 7],
    [2, 6, 1, 5, 4, 7, 3],
    [3, 7, 1, 4, 2, 6, 5],
    [2, 4, 1, 5, 3, 7, 6],
    [4, 5, 6, 1, 3, 2, 7],
    [4, 6, 2, 3, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 4, 1, 3, 2, 7],
    [6, 5, 4, 1, 3, 2, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [3, 7, 2, 1, 4, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 7, 1, 5, 3, 6, 4],
    [1, 4, 2, 5, 3, 6, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 7, 2, 4, 5, 6, 1],
    [5, 3, 6, 4, 2, 1, 7],
    [4, 1, 5, 6, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 7, 1, 4, 3, 6, 5],
    [1, 4, 2, 6, 3, 5, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [3, 1, 5, 6, 2, 4, 7],
    [1, 4, 2, 5, 3, 6, 7],
    [2, 5, 1, 6, 4, 7, 3],
    [4, 7, 1, 5, 3, 6, 2],
    [2, 4, 1, 5, 3, 6, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 2, 6, 3, 4, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [2, 4, 1, 7, 3, 5, 6],
    [4, 7, 3, 1, 2, 6, 5],
    [4, 7, 2, 5, 3, 6, 1],
    [4, 1, 5, 6, 3, 2, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 1, 4, 3, 6, 7],
    [2, 5, 1, 6, 4, 7, 3],
    [3, 6, 1, 4, 5, 7, 2],
    [1, 4, 2, 5, 3, 6, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [3, 7, 2, 5, 4, 6, 1],
    [4, 5, 6, 3, 2, 1, 7],
    [4, 3, 6, 5, 2, 1, 7],
    [3, 6, 5, 1, 2, 4, 7],
    [5, 4, 6, 1, 2, 3, 7],
    [1, 2, 3, 6, 4, 5, 7],
    [4, 7, 3, 1, 2, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 2, 6, 5, 1, 3, 7],
    [1, 6, 3, 4, 2, 5, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 4, 1, 5, 3, 7, 6],
    [5, 6, 4, 1, 3, 2, 7],
    [3, 7, 2, 4, 5, 6, 1],
    [5, 3, 6, 4, 2, 1, 7],
    [4, 2, 5, 6, 3, 1, 7],
    [4, 5, 6, 1, 2, 3, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 4, 2, 5, 3, 6, 7],
    [4, 7, 2, 1, 3, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [2, 1, 5, 6, 3, 4, 7],
    [1, 6, 3, 4, 2, 5, 7],
    [2, 5, 1, 6, 4, 7, 3],
    [4, 6, 1, 5, 2, 7, 3],
    [1, 4, 2, 5, 3, 6, 7],
    [5, 6, 4, 1, 3, 2, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 4, 1, 2, 3, 7],
    [5, 4, 6, 2, 3, 1, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 3, 1, 2, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [3, 1, 6, 5, 4, 2, 7],
    [2, 6, 3, 4, 1, 5, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 7, 1, 4, 3, 6, 5],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 6, 5, 1, 3, 2, 7],
    [4, 6, 2, 3, 5, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [3, 6, 5, 1, 2, 4, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [2, 1, 5, 6, 4, 3, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [1, 5, 2, 4, 3, 6, 7],
    [2, 5, 1, 6, 4, 7, 3],
    [2, 7, 1, 5, 3, 6, 4],
    [1, 4, 2, 5, 3, 7, 6],
    [4, 6, 5, 1, 2, 3, 7],
    [3, 6, 2, 5, 4, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [3, 1, 6, 5, 4, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 4, 2, 7, 3, 5, 6],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [4, 1, 5, 6, 3, 2, 7],
    [4, 1, 6, 5, 2, 3, 7],
    [1, 5, 3, 4, 2, 6, 7],
    [2, 5, 1, 6, 4, 7, 3],
    [2, 6, 1, 5, 3, 7, 4],
    [1, 4, 2, 5, 3, 7, 6],
    [4, 6, 5, 1, 3, 2, 7],
    [3, 7, 2, 4, 5, 6, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [5, 4, 6, 2, 3, 1, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 4, 5, 7, 1],
    [2, 1, 5, 6, 4, 3, 7],
    [4, 1, 5, 6, 3, 2, 7],
    [1, 5, 3, 4, 2, 7, 6],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 7, 1, 4, 3, 6, 5],
    [2, 4, 1, 6, 3, 7, 5],
    [4, 6, 5, 1, 3, 2, 7],
    [3, 6, 2, 5, 4, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [2, 6, 5, 1, 3, 4, 7],
    [4, 5, 6, 2, 3, 1, 7],
    [1, 4, 2, 7, 3, 6, 5],
    [3, 7, 4, 1, 2, 5, 6],
    [3, 7, 2, 4, 5, 6, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 6, 5, 2, 3, 7],
    [1, 6, 2, 4, 3, 5, 7],
    [3, 5, 1, 6, 4, 7, 2],
    [2, 6, 1, 4, 3, 7, 5],
    [2, 4, 1, 5, 3, 6, 7],
    [5, 6, 4, 1, 3, 2, 7],
    [3, 7, 2, 5, 4, 6, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [5, 1, 4, 6, 3, 2, 7],
    [4, 6, 3, 2, 1, 5, 7],
    [4, 5, 6, 2, 3, 1, 7],
    [2, 4, 1, 7, 3, 6, 5],
    [4, 7, 3, 1, 2, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [6, 1, 4, 5, 2, 3, 7],
    [1, 7, 2, 4, 3, 5, 6],
    [2, 5, 1, 6, 4, 7, 3],
    [2, 7, 1, 3, 4, 5, 6],
    [2, 3, 1, 7, 4, 6, 5],
    [5, 4, 6, 1, 2, 3, 7],
    [3, 7, 2, 4, 5, 6, 1],
    [6, 4, 5, 3, 2, 1, 7],
    [4, 2, 5, 6, 3, 1, 7],
    [4, 6, 5, 2, 1, 3, 7],
    [5, 4, 6, 1, 3, 2, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [2, 7, 3, 1, 4, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 6, 1, 5, 4, 7, 3],
    [2, 7, 1, 3, 4, 6, 5],
    [1, 4, 2, 5, 3, 7, 6],
    [4, 6, 5, 1, 2, 3, 7],
    [3, 7, 2, 5, 4, 6, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 3, 1, 2, 4, 7],
    [4, 5, 6, 2, 3, 1, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 3, 1, 2, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [1, 5, 2, 4, 3, 6, 7],
    [2, 5, 1, 6, 4, 7, 3],
    [3, 6, 1, 5, 4, 7, 2],
    [2, 4, 1, 5, 3, 7, 6],
    [5, 4, 6, 1, 2, 3, 7],
    [4, 6, 2, 3, 5, 7, 1],
    [6, 4, 5, 2, 3, 1, 7],
    [3, 1, 6, 5, 4, 2, 7],
    [6, 5, 4, 1, 2, 3, 7],
    [6, 4, 5, 1, 2, 3, 7],
    [1, 3, 2, 7, 4, 6, 5],
    [4, 7, 2, 1, 3, 6, 5],
    [4, 6, 2, 5, 3, 7, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [4, 1, 6, 5, 2, 3, 7],
    [1, 5, 3, 4, 2, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [3, 6, 1, 5, 2, 7, 4],
    [2, 4, 1, 5, 3, 7, 6],
    [5, 6, 4, 1, 3, 2, 7],
    [4, 6, 2, 3, 5, 7, 1],
    [5, 4, 6, 3, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [6, 4, 5, 1, 3, 2, 7],
    [1, 3, 2, 7, 4, 5, 6],
    [4, 7, 3, 1, 2, 5, 6],
    [3, 7, 2, 4, 5, 6, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 5, 6, 2, 3, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 6, 1, 5, 4, 7, 3],
    [1, 4, 2, 6, 3, 5, 7],
    [4, 5, 6, 1, 2, 3, 7],
    [4, 7, 2, 3, 5, 6, 1],
    [5, 3, 6, 4, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 4, 1, 2, 3, 7],
    [5, 4, 6, 2, 3, 1, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 2, 1, 3, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [3, 5, 2, 4, 1, 6, 7],
    [2, 6, 1, 5, 3, 7, 4],
    [2, 7, 1, 4, 3, 6, 5],
    [1, 4, 2, 6, 3, 5, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 7, 2, 4, 5, 6, 1],
    [5, 3, 6, 4, 2, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [5, 6, 4, 1, 2, 3, 7],
    [4, 5, 6, 2, 3, 1, 7],
    [1, 4, 2, 6, 3, 5, 7],
    [4, 7, 3, 1, 2, 5, 6],
    [3, 6, 2, 5, 4, 7, 1],
    [2, 1, 5, 6, 3, 4, 7],
    [4, 1, 6, 5, 2, 3, 7],
    [2, 5, 3, 4, 1, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 6, 1, 4, 3, 7, 5],
    [1, 4, 2, 5, 3, 6, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 6, 2, 4, 5, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [4, 2, 6, 5, 3, 1, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [5, 4, 6, 2, 3, 1, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 2, 1, 3, 6, 5],
    [3, 6, 2, 5, 4, 7, 1],
    [3, 1, 5, 6, 2, 4, 7],
    [4, 1, 6, 5, 2, 3, 7],
    [1, 5, 3, 4, 2, 6, 7],
    [2, 5, 1, 6, 3, 7, 4],
    [2, 6, 1, 4, 3, 7, 5],
    [1, 4, 2, 5, 3, 6, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [3, 6, 2, 5, 4, 7, 1],
    [5, 4, 6, 2, 3, 1, 7],
    [4, 1, 6, 5, 3, 2, 7],
    [4, 6, 5, 1, 2, 3, 7],
    [4, 5, 6, 1, 3, 2, 7],
    [1, 3, 2, 6, 4, 5, 7],
    [4, 7, 3, 1, 2, 6, 5]
])

# Infer structure
n_obs = raw_data.shape[0]
items = raw_data.shape[1]

# Choose a participant/trial structure
participants = 15
trials = n_obs // participants

if participants * trials != n_obs:
    raise ValueError("Data size not divisible by participants — adjust participants count.")

rows = []
idx = 0
for p in range(1, participants + 1):
    for t in range(1, trials + 1):
        rows.append([p, t] + raw_data[idx].tolist())
        idx += 1

df = pd.DataFrame(rows, columns=["participant", "trial"] + [f"m{i}" for i in range(1, items + 1)])
df.to_csv("multiphonics_rankings.csv", index=False)

print(f"Loaded real dataset: {participants} participants × {trials} trials × {items} items")



# =============================================================================
# SECTION A — Data → Plackett–Luce ranking tensors
# =============================================================================
# Each row: participant s, trial r, and m_j = rank position of multiphonic j
# (1 = most preferred). We convert to k_sr = (item at rank 1, …, item at rank N)
# with items coded 0 … N−1 internally (paper uses 1 … N).


def rankings_from_csv(path: str | Path) -> np.ndarray:
    """
    Load CSV produced above. Returns rankings[s, r, :] : int, item indices 0..N-1
    (best → worst). We stack explicitly so NumPy does not silently ragged-promote.
    """
    table = pd.read_csv(path)
    s_ids = sorted(table["participant"].astype(int).unique().tolist())
    S = len(s_ids)
    s_map = {sid: i for i, sid in enumerate(s_ids)}
    trials_by_s = [set() for _ in range(S)]
    for _, row in table.iterrows():
        trials_by_s[s_map[int(row["participant"])]].add(int(row["trial"]))
    R = max(len(trials_by_s[i]) for i in range(S))
    N = len([c for c in table.columns if c.startswith("m")])

    out: list[list[np.ndarray | None]] = [[None] * R for _ in range(S)]

    for _, row in table.iterrows():
        sid = int(row["participant"])
        tid = int(row["trial"])
        si, ri = s_map[sid], tid - 1
        rank_vals = np.array([int(row[f"m{j}"]) for j in range(1, N + 1)], dtype=float)
        order = np.argsort(rank_vals).astype(int)
        out[si][ri] = order

    for si in range(S):
        for ri in range(R):
            if out[si][ri] is None:
                raise ValueError(f"Missing ranking for participant index {si}, trial {ri + 1}")

    return np.stack([[np.asarray(out[si][ri], dtype=int) for ri in range(R)] for si in range(S)])


# =============================================================================
# SECTION B — GM06 / paper17 mean squared distance in R^D
# =============================================================================
# d(y, x_j) = (1/D) * sum_d (y_d - x_{j,d})^2  (Main 11 text, eq. after GM06 cite)


def mean_sq_distances(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    y : (D,), X : (N, D)
    Returns d of length N with d[j] = (1/D) * ||y - x_j||^2 (row-wise).
    """
    D = y.shape[0]
    diff = X - y  # (N, D)
    return np.sum(diff * diff, axis=1) / float(D)


# =============================================================================
# SECTION C — Plackett–Luce log-likelihood for one complete ranking
# =============================================================================
# P(k | η) = ∏_t exp(η_{k_t}) / sum_{u≥t} exp(η_{k_u})   (sequential choice)


def log_plackett_luce_one(ordered_items: np.ndarray, eta: np.ndarray) -> float:
    """
    ordered_items: length N, indices 0..N-1, best → worst.
    At stage t the t-th chosen item is ordered_items[t]; denominator sums exp(eta)
    over all items not yet chosen (standard Plackett–Luce sequential construction).
    """
    order = ordered_items.astype(int)
    remaining = set(order.tolist())
    logp = 0.0
    for t in range(len(order)):
        j = int(order[t])
        rem = np.array(list(remaining), dtype=int)
        et = eta[rem]
        m = np.max(et)
        log_den = m + np.log(np.sum(np.exp(et - m)))
        logp += eta[j] - log_den
        remaining.remove(j)
    return logp


def eta_sr(y: np.ndarray, X: np.ndarray, c: np.ndarray, b_s: float) -> np.ndarray:
    """η_sr,j = c_j − b_s * d(y, x_j) with mean-squared distance d."""
    dvec = mean_sq_distances(y, X)
    return c - b_s * dvec


# =============================================================================
# SECTION D — Log-prior (Main 11, Appendix A.3; Gaussian + Gamma)
# =============================================================================
# y_sr, x_j ~ N(0, I_D), c_j ~ N(0,1), b_s ~ Gamma(shape=25, scale=1/24)


def logpdf_normal(x: np.ndarray, sigma2: float) -> float:
    """Independent N(0, sigma2) components."""
    return -0.5 * np.sum(x * x) / sigma2 - 0.5 * x.size * np.log(2 * np.pi * sigma2)


def logpdf_gamma_scalar(b: float, shape: float = 25.0, scale: float = 1.0 / 24.0) -> float:
    if b <= 0:
        return -np.inf
    k, theta = shape, scale
    return (k - 1.0) * math.log(b) - b / theta - k * math.log(theta) - math.lgamma(k)


def logpdf_gamma(b: np.ndarray, shape: float = 25.0, scale: float = 1.0 / 24.0) -> float:
    """Independent Gamma(shape, scale) components (Main 11, shape–scale)."""
    if np.any(b <= 0):
        return -np.inf
    return float(sum(logpdf_gamma_scalar(float(t), shape, scale) for t in b))


def log_prior(Y: np.ndarray, X: np.ndarray, c: np.ndarray, b: np.ndarray) -> float:
    """Joint log-prior for all latent vectors and scalars."""
    lp = 0.0
    S, R, D = Y.shape
    for s in range(S):
        for r in range(R):
            lp += logpdf_normal(Y[s, r], 1.0)
    lp += logpdf_normal(X.reshape(-1), 1.0)
    lp += logpdf_normal(c, 1.0)
    lp += logpdf_gamma(b)
    return lp


# =============================================================================
# SECTION E — Full log-posterior kernel (Main 11, Appendix A.4)
# =============================================================================


def log_likelihood_sr(
    Y: np.ndarray,
    X: np.ndarray,
    c: np.ndarray,
    b: np.ndarray,
    rankings: np.ndarray,
    s: int,
    r: int,
) -> float:
    """Contribution of ranking (s, r) to the log-likelihood (Main 11 factorization)."""
    return log_likelihood_sr_yvec(Y[s, r], X, c, float(b[s]), rankings[s, r])


def log_likelihood_sr_yvec(y: np.ndarray, X: np.ndarray, c: np.ndarray, b_s: float, rank_vec: np.ndarray) -> float:
    """Same as log_likelihood_sr but with an explicit y vector (avoids full Y copies in MH)."""
    eta = eta_sr(y, X, c, b_s)
    return log_plackett_luce_one(rank_vec, eta)


def log_likelihood_all(Y: np.ndarray, X: np.ndarray, c: np.ndarray, b: np.ndarray, rankings: np.ndarray) -> float:
    S, R, _ = Y.shape
    return sum(
        log_likelihood_sr(Y, X, c, b, rankings, s, r) for s in range(S) for r in range(R)
    )


def log_likelihood_participant(
    Y: np.ndarray, X: np.ndarray, c: np.ndarray, b: np.ndarray, rankings: np.ndarray, s: int
) -> float:
    """Sum of log-likelihood terms involving participant s (used for b_s updates)."""
    return sum(log_likelihood_sr(Y, X, c, b, rankings, s, r) for r in range(Y.shape[1]))


def log_posterior(
    Y: np.ndarray,
    X: np.ndarray,
    c: np.ndarray,
    b: np.ndarray,
    rankings: np.ndarray,
) -> float:
    """
    log p(Y, X, c, b | data) up to additive constant:
      likelihood × priors
    rankings.shape = (S, R, N): item indices best → worst.
    """
    return log_likelihood_all(Y, X, c, b, rankings) + log_prior(Y, X, c, b)


# =============================================================================
# SECTION F — Stack latent coordinates for Procrustes (paper17 §4.1)
# =============================================================================
# C = [all y_sr rows; all x_j rows], shape ((S*R + N), D)


def stack_configuration(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    S, R, D = Y.shape
    N = X.shape[0]
    rows = [Y[s, r] for s in range(S) for r in range(R)]
    rows.extend([X[j] for j in range(N)])
    return np.vstack(rows)


def unstack_configuration(C: np.ndarray, S: int, R: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    sr = S * R
    Y = C[:sr].reshape(S, R, -1)
    X = C[sr:].reshape(N, -1)
    return Y, X


# =============================================================================
# SECTION G — Orthogonal Procrustes: match centered Ĉ to centered C_R (paper17)
# =============================================================================
# Minimize ||C_R - Ĉ Q||_F with Q orthogonal ⇒ Q = U V' for SVD(Ĉ' C_R) after
# column centering; then aligned Ĉ* = Ĉ Q (rows are points).


def center_rows(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = C.mean(axis=0)
    return C - mu, mu


def orthogonal_procrustes(C_ref: np.ndarray, C_tilt: np.ndarray) -> np.ndarray:
    """
    Find orthogonal Q (D×D) minimizing ||C_ref - C_tilt @ Q||_F
    after both are row-centered (caller should center; we center again defensively).
    Returns Q.
    """
    A, _ = center_rows(C_ref)
    B, _ = center_rows(C_tilt)
    M = B.T @ A  # D×D
    U, _, Vt = np.linalg.svd(M, full_matrices=True)
    Q = U @ Vt
    # guard against reflection if det(Q)<0 (optional: paper allows rotation)
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q


def align_to_reference(Y: np.ndarray, X: np.ndarray, CR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Translate + rotate (Y, X) stacked as paper17 so the stacked matrix best
    matches fixed reference CR (already centered in our pipeline).
    """
    S, R, D = Y.shape
    N = X.shape[0]
    C_hat = stack_configuration(Y, X)
    mu_hat, _ = center_rows(C_hat)
    # reference is already centered around origin in our construction
    Q = orthogonal_procrustes(CR, C_hat)
    C_aligned = (C_hat - C_hat.mean(0)) @ Q
    return unstack_configuration(C_aligned, S, R, N)


# =============================================================================
# SECTION H — Metropolis–Hastings (Main 11 Appendix A.5 + paper17 MAP phase)
# =============================================================================
# One sweep:
#   (1) each y_sr — Gaussian RW, acceptance with (2) from paper17 (symmetric q).
#   (2) each x_j
#   (3) each c_j
#   (4) each b_s — log-scale RW, Hastings factor b*/b (see text).
#
# Phase MAP (paper17 §4.1): for n_map_iter iterations accept only if log-posterior
# increases (strict uphill). The centered stacked MAP configuration becomes C_R.
#
# After burn-in on the sampling phase, each stored iteration applies Procrustes
# to C_R (Main 11 Appendix A.6: draw-by-draw alignment).


def mcmc_latent_pl(
    rankings: np.ndarray,
    D: int,
    *,
    # Short defaults keep a full (S,R)=(20,15) grid tractable; raise toward paper values.
    n_iter: int = 80,
    n_map: int = 45,
    burn_in: int = 25,
    sigma_prop_y: float = 0.25,
    sigma_prop_x: float = 0.08,
    sigma_prop_c: float = 0.12,
    sigma_prop_log_b: float = 0.12,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    S, R = rankings.shape[0], rankings.shape[1]
    N = int(rankings.shape[2])

    # Initialization (diffuse but consistent with priors)
    Y = rng.normal(size=(S, R, D))
    X = rng.normal(size=(N, D))
    c = rng.normal(size=N)
    b = rng.gamma(25.0, 1.0 / 24.0, size=S)

    # -------- MAP / uphill phase (paper17): build reference configuration --------
    for it in range(n_map):
        if it % 20 == 0:
            print(f"    MAP phase {it}/{n_map} (D={D})", flush=True)
        # (1) y_sr — only ranking (s,r) depends on y_sr; compare local likelihood + N(0,I) prior.
        for s in range(S):
            for r in range(R):
                prop = Y[s, r] + rng.normal(scale=sigma_prop_y, size=D)
                log_old = log_likelihood_sr_yvec(Y[s, r], X, c, float(b[s]), rankings[s, r]) + logpdf_normal(
                    Y[s, r], 1.0
                )
                log_new = log_likelihood_sr_yvec(prop, X, c, float(b[s]), rankings[s, r]) + logpdf_normal(prop, 1.0)
                if log_new > log_old:
                    Y[s, r] = prop
        # (2) x_j — likelihood depends on all rankings; recompute full likelihood (still cheap vs old y-loop).
        for j in range(N):
            prop_row = X[j] + rng.normal(scale=sigma_prop_x, size=D)
            Xp = X.copy()
            Xp[j] = prop_row
            log_old = log_likelihood_all(Y, X, c, b, rankings) + logpdf_normal(X[j], 1.0)
            log_new = log_likelihood_all(Y, Xp, c, b, rankings) + logpdf_normal(prop_row, 1.0)
            if log_new > log_old:
                X[j] = prop_row
        # (3) c_j
        for j in range(N):
            cp = c.copy()
            cp[j] = c[j] + rng.normal(scale=sigma_prop_c)
            log_old = log_likelihood_all(Y, X, c, b, rankings) + logpdf_normal(np.array([c[j]]), 1.0)
            log_new = log_likelihood_all(Y, X, cp, b, rankings) + logpdf_normal(np.array([cp[j]]), 1.0)
            if log_new > log_old:
                c = cp
        # (4) b_s — only rankings for participant s; MAP compares posterior density (no MH Jacobian).
        for s in range(S):
            bp = b.copy()
            m = np.exp(rng.normal(scale=sigma_prop_log_b))
            bp[s] = b[s] * m
            log_old = log_likelihood_participant(Y, X, c, b, rankings, s) + logpdf_gamma_scalar(float(b[s]))
            log_new = log_likelihood_participant(Y, X, c, bp, rankings, s) + logpdf_gamma_scalar(float(bp[s]))
            if log_new > log_old:
                b = bp

    C_map = stack_configuration(Y, X)
    CR, _ = center_rows(C_map)

    # -------- Sampling phase (standard MH) --------
    samples_Y: list[np.ndarray] = []
    samples_X: list[np.ndarray] = []
    samples_c: list[np.ndarray] = []
    samples_b: list[np.ndarray] = []

    for it in range(n_iter):
        if it % max(1, n_iter // 4) == 0:
            print(f"    MH sampling {it}/{n_iter} (D={D})", flush=True)
        for s in range(S):
            for r in range(R):
                prop = Y[s, r] + rng.normal(scale=sigma_prop_y, size=D)
                log_a = (
                    log_likelihood_sr_yvec(prop, X, c, float(b[s]), rankings[s, r])
                    + logpdf_normal(prop, 1.0)
                    - log_likelihood_sr_yvec(Y[s, r], X, c, float(b[s]), rankings[s, r])
                    - logpdf_normal(Y[s, r], 1.0)
                )
                if np.log(rng.random()) < min(0.0, log_a):
                    Y[s, r] = prop
        for j in range(N):
            prop_row = X[j] + rng.normal(scale=sigma_prop_x, size=D)
            Xp = X.copy()
            Xp[j] = prop_row
            log_a = (
                log_likelihood_all(Y, Xp, c, b, rankings)
                - log_likelihood_all(Y, X, c, b, rankings)
                + logpdf_normal(prop_row, 1.0)
                - logpdf_normal(X[j], 1.0)
            )
            if np.log(rng.random()) < min(0.0, log_a):
                X[j] = prop_row
        for j in range(N):
            cp = c.copy()
            cp[j] = c[j] + rng.normal(scale=sigma_prop_c)
            log_a = (
                log_likelihood_all(Y, X, cp, b, rankings)
                - log_likelihood_all(Y, X, c, b, rankings)
                + logpdf_normal(np.array([cp[j]]), 1.0)
                - logpdf_normal(np.array([c[j]]), 1.0)
            )
            if np.log(rng.random()) < min(0.0, log_a):
                c = cp
        for s in range(S):
            bp = b.copy()
            m = np.exp(rng.normal(scale=sigma_prop_log_b))
            bp[s] = b[s] * m
            log_a = (
                log_likelihood_participant(Y, X, c, bp, rankings, s)
                - log_likelihood_participant(Y, X, c, b, rankings, s)
                + logpdf_gamma_scalar(float(bp[s]))
                - logpdf_gamma_scalar(float(b[s]))
                + np.log(bp[s] / b[s])
            )
            if np.log(rng.random()) < min(0.0, log_a):
                b = bp

        # Procrustean alignment to C_R after each iteration (paper17 §4.2; Main 11 A.6)
        Y, X = align_to_reference(Y, X, CR)

        if it >= burn_in:
            samples_Y.append(Y.copy())
            samples_X.append(X.copy())
            samples_c.append(c.copy())
            samples_b.append(b.copy())

    return {
        "samples_Y": np.stack(samples_Y, axis=0),  # (T, S, R, D)
        "samples_X": np.stack(samples_X, axis=0),  # (T, N, D)
        "samples_c": np.stack(samples_c, axis=0),  # (T, N)
        "samples_b": np.stack(samples_b, axis=0),  # (T, S)
        "CR": CR,
        "D": D,
    }


# =============================================================================
# SECTION I — Run fits for several D; print draws; interactive plots
# =============================================================================


def summarize_draws(fit: dict, n_tail: int = 8) -> None:
    """Print the last few posterior draws (aligned) as numeric tables."""
    SY = fit["samples_Y"]
    SX = fit["samples_X"]
    Sc = fit["samples_c"]
    Sb = fit["samples_b"]
    T = SY.shape[0]
    sl = slice(max(0, T - n_tail), T)
    print("\n--- Posterior tail (last", n_tail, "iterations), D =", fit["D"], "---")
    print("samples_b (rows=iter, cols=participant):\n", Sb[sl])
    print("samples_c (rows=iter, cols=item):\n", Sc[sl])
    print("samples_X mean over items last iter:\n", SX[-1].mean(axis=0))
    print("one y_sr (participant 0, trial 0) last draws:\n", SY[sl, 0, 0, :])


def plot_latent_panel(fit: dict, rankings: np.ndarray) -> None:
    """Slider: pick 2 dims to plot when D>2; traces for b_1, c_1."""
    D = int(fit["D"])
    SY = fit["samples_Y"]
    SX = fit["samples_X"]
    X_mean = SX.mean(axis=0)
    Y_mean = SY.mean(axis=0)
    S, R, _ = Y_mean.shape
    N = X_mean.shape[0]

    fig = plt.figure(figsize=(11, 7))
    ax_scatter = fig.add_axes([0.08, 0.35, 0.45, 0.55])
    ax_radio = fig.add_axes([0.62, 0.55, 0.15, 0.35])
    ax_dim1 = fig.add_axes([0.62, 0.38, 0.25, 0.03])
    ax_dim2 = fig.add_axes([0.62, 0.32, 0.25, 0.03])
    ax_trace = fig.add_axes([0.08, 0.08, 0.85, 0.18])

    if D >= 2:
        d1, d2 = 0, 1
    else:
        d1, d2 = 0, 0

    def redraw_scatter():
        ax_scatter.clear()
        if D == 1:
            ax_scatter.axvline(X_mean[:, 0].mean(), color="steelblue", lw=2, label="items mean")
            ax_scatter.scatter(X_mean[:, 0], np.zeros(N), c="tab:blue", s=80, label="items", zorder=3)
            ax_scatter.scatter(Y_mean[..., 0].ravel(), np.zeros(S * R), c="tab:orange", s=12, alpha=0.4, label="y_sr")
            ax_scatter.set_yticks([])
            ax_scatter.set_xlabel("Dimension 1")
        else:
            ax_scatter.scatter(
                X_mean[:, d1], X_mean[:, d2], c="tab:blue", s=100, label="multiphonics", zorder=3
            )
            for j in range(N):
                ax_scatter.annotate(str(j + 1), (X_mean[j, d1], X_mean[j, d2]), xytext=(4, 4), textcoords="offset points")
            ax_scatter.scatter(
                Y_mean[..., d1].ravel(),
                Y_mean[..., d2].ravel(),
                c="tab:orange",
                s=14,
                alpha=0.35,
                label="y_sr (participant×trial)",
            )
            ax_scatter.set_xlabel(f"Dim {d1 + 1}")
            ax_scatter.set_ylabel(f"Dim {d2 + 1}")
        ax_scatter.set_title(f"Posterior mean latent space (D={D}, Procrustes-aligned draws)")
        ax_scatter.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    redraw_scatter()

    if D >= 2:
        s1 = Slider(ax_dim1, "x-axis dim", 0, D - 1, valinit=0, valstep=1)
        s2 = Slider(ax_dim2, "y-axis dim", 0, D - 1, valinit=min(1, D - 1), valstep=1)

        def on_slider(_val):
            nonlocal d1, d2
            d1, d2 = int(s1.val), int(s2.val)
            if d1 == d2 and D > 1:
                d2 = (d1 + 1) % D
                s2.set_val(d2)
            redraw_scatter()

        s1.on_changed(on_slider)
        s2.on_changed(on_slider)

    # Trace for first participant sensitivity and first item appeal
    ax_trace.clear()
    ax_trace.plot(fit["samples_b"][:, 0], lw=0.6, label="b_1 trace")
    ax_trace.plot(fit["samples_c"][:, 0], lw=0.6, label="c_1 trace")
    ax_trace.legend(fontsize=7, ncol=2)
    ax_trace.set_title("Posterior traces (aligned chain)")
    ax_trace.set_xlabel("Saved iteration")

    # Radio only useful when multiple fits passed — here single fit; still show D
    rax = ax_radio
    rax.clear()
    rax.text(0.0, 0.5, f"Model\nD = {D}", transform=rax.transAxes, fontsize=10)
    rax.set_xticks([])
    rax.set_yticks([])

    plt.show()


def run_all_dimensions(rankings: np.ndarray, dims: tuple[int, ...] = (1, 2, 3)) -> dict[int, dict]:
    """Fit separate models for each D (model selection / comparison)."""
    fits: dict[int, dict] = {}
    for j, D in enumerate(dims):
        print(f"\n=== MCMC for latent dimension D = {D} ===")
        fits[D] = mcmc_latent_pl(rankings, D, seed=100 + j)
        summarize_draws(fits[D], n_tail=6)
    return fits


def plot_latent_panel_static(fit: dict, rankings: np.ndarray, fname: str) -> None:
    """Non-interactive version: posterior mean scatter + traces, saved to fname."""
    D = int(fit["D"])
    SY, SX = fit["samples_Y"], fit["samples_X"]
    X_mean, Y_mean = SX.mean(axis=0), SY.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes
    if D == 1:
        ax0.scatter(X_mean[:, 0], np.zeros(fit["samples_X"].shape[1]), c="tab:blue", s=80, label="items")
        ax0.scatter(Y_mean[..., 0].ravel(), np.zeros(Y_mean[..., 0].size), c="tab:orange", s=10, alpha=0.35, label="y_sr")
        ax0.set_yticks([])
        ax0.set_xlabel("Dim 1")
    else:
        d1, d2 = 0, 1
        ax0.scatter(X_mean[:, d1], X_mean[:, d2], c="tab:blue", s=90, zorder=3, label="items")
        for j in range(X_mean.shape[0]):
            ax0.annotate(str(j + 1), (X_mean[j, d1], X_mean[j, d2]), xytext=(3, 3), textcoords="offset points")
        ax0.scatter(
            Y_mean[..., d1].ravel(),
            Y_mean[..., d2].ravel(),
            c="tab:orange",
            s=12,
            alpha=0.35,
            label="y_sr",
        )
        ax0.set_xlabel("Dim 1")
        ax0.set_ylabel("Dim 2")
    ax0.set_title(f"Posterior mean (D={D})")
    ax0.legend(fontsize=7)
    ax1.plot(fit["samples_b"][:, 0], lw=0.7, label="b_1")
    ax1.plot(fit["samples_c"][:, 0], lw=0.7, label="c_1")
    ax1.legend(fontsize=7)
    ax1.set_title("Traces")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main driver: load generated CSV, fit D ∈ {1,2,3}, plots + interactive choice
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    rankings_np = rankings_from_csv("multiphonics_rankings.csv")
    S, R, N = rankings_np.shape

    # Fit each D separately (paper17 §4.3 / Main 11 §5.4 model comparison over dimension).
    all_fits = run_all_dimensions(rankings_np, dims=(1, 2, 3))

    for d, fit in all_fits.items():
        np.savez_compressed(
            f"posterior_draws_D{d}.npz",
            samples_Y=fit["samples_Y"],
            samples_X=fit["samples_X"],
            samples_c=fit["samples_c"],
            samples_b=fit["samples_b"],
            CR=fit["CR"],
        )
    print("Saved posterior_draws_D1.npz, D2, D3 (aligned chains).")

    # Interactive GUI, or save static figures when MPLBACKEND=Agg (e.g. CI / servers).
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        for d, fit in all_fits.items():
            plot_latent_panel_static(fit, rankings_np, f"latent_space_D{d}.png")
        print("Saved latent_space_D1.png, latent_space_D2.png, latent_space_D3.png")
    else:
        fig_pick = plt.figure(figsize=(5, 3))
        axp = fig_pick.add_axes([0.15, 0.2, 0.7, 0.65])
        labels = [f"D = {d}" for d in all_fits]
        radio = RadioButtons(axp, labels, active=1)

        def onselect(sel: str):
            d = int(str(sel).split("=")[1].strip())
            plt.close(fig_pick)
            plot_latent_panel(all_fits[d], rankings_np)

        radio.on_clicked(onselect)
        plt.suptitle("Choose latent dimensionality to plot (close window to skip)")
        plt.show()