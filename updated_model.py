"""
Bayesian latent-space Plackett-Luce model for oboe-multiphonic ranking data.

MODEL (per the manuscript "main", Sections 4-5 and Appendix A; replaces the
Gormley & Murphy 2006 voter/candidate parameterization):

  Latent variables
    y_r  in R^D : location of orchestral target sound r      (SHARED across participants)
    x_j  in R^D : location of oboe multiphonic j
    c_j  in R   : baseline appeal of multiphonic j
    b_s  > 0    : sensitivity of participant s

  Support score (squared Euclidean distance -- NO division by D):
    eta_srj = c_j - b_s * ||y_r - x_j||^2

  Plackett-Luce likelihood of a complete ranking k_sr = (k_sr1, ..., k_srN):
    P(k_sr) = prod_{t=1}^{N} exp(eta_sr,k_srt) / sum_{u=t}^{N} exp(eta_sr,k_sru)

  Priors:
    y_r ~ N(0, I_D),  x_j ~ N(0, I_D),  c_j ~ N(0, 1),
    b_s ~ Gamma(shape=25, scale=1/24)    [mean ~= 1.042, sd ~= 0.208]

  MCMC: block Metropolis-Hastings.
    y_r, x_j, c_j : Gaussian random walk (symmetric proposal, no Hastings term).
    b_s           : log-scale random walk with Jacobian correction  + log(b*/b).
  MAP/uphill phase first (GM06 / paper17 Sec. 4.1) to build the Procrustes
  reference configuration C_R; sampling draws are aligned draw-by-draw.
  Procrustes acts ONLY on the stacked latent coordinates {y_r} u {x_j};
  c and b are location/rotation invariant and are never transformed.

DATA STRUCTURE (inferred empirically, see infer_structure()):
  450 rows = 30 participants x 15 targets, participant-major
  (rows 15*(s-1) .. 15*s - 1 are participant s's rankings of targets 1..15;
  within-target Kendall-tau agreement ~0.85 vs ~0.19 for random row pairs).
  NOTE: the manuscript states R = 14 targets; the present dataset contains 15.
  Reconcile before publication -- set R_TASKS below if the design changes.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from pathlib import Path

from rankdata import raw_data  # (450, 7) rank matrix: entry = rank of multiphonic j (1 = best)

# -----------------------------------------------------------------------------
# 0. Experimental design constants
# -----------------------------------------------------------------------------
R_TASKS = 15  # number of orchestral target sounds actually present in the data
              # (manuscript Section 3 says 14 -- see module docstring)

# -----------------------------------------------------------------------------
# 1. Load ranking data -> long CSV with correct (participant, target) labels
# -----------------------------------------------------------------------------


def build_dataframe(raw: np.ndarray, r_tasks: int = R_TASKS) -> pd.DataFrame:
    n_obs, items = raw.shape
    if n_obs % r_tasks != 0:
        raise ValueError(f"{n_obs} rows not divisible by R = {r_tasks} targets.")
    participants = n_obs // r_tasks
    rows = []
    idx = 0
    for s in range(1, participants + 1):
        for r in range(1, r_tasks + 1):  # participant-major, target = row mod R
            rows.append([s, r] + raw[idx].tolist())
            idx += 1
    return pd.DataFrame(
        rows, columns=["participant", "target"] + [f"m{j}" for j in range(1, items + 1)]
    )


# =============================================================================
# SECTION A -- Data -> Plackett-Luce ranking tensors
# =============================================================================
# rankings[s, r, :] = item indices (0..N-1) ordered best -> worst for
# participant s (0-based) and orchestral target r (0-based).


def rankings_from_csv(path: str | Path) -> np.ndarray:
    table = pd.read_csv(path)
    s_ids = sorted(table["participant"].astype(int).unique().tolist())
    r_ids = sorted(table["target"].astype(int).unique().tolist())
    S, R = len(s_ids), len(r_ids)
    s_map = {v: i for i, v in enumerate(s_ids)}
    r_map = {v: i for i, v in enumerate(r_ids)}
    N = len([col for col in table.columns if col.startswith("m")])

    out = np.full((S, R, N), -1, dtype=int)
    for _, row in table.iterrows():
        si, ri = s_map[int(row["participant"])], r_map[int(row["target"])]
        rank_vals = np.array([int(row[f"m{j}"]) for j in range(1, N + 1)])
        out[si, ri] = np.argsort(rank_vals)  # best -> worst item indices
    if np.any(out < 0):
        raise ValueError("Missing (participant, target) ranking cells.")
    return out


# =============================================================================
# SECTION B -- Squared Euclidean distances (NEW: no 1/D factor)
# =============================================================================


def sq_dist_row(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """d[j] = ||y - x_j||^2 (plain squared Euclidean; GM06's 1/D factor removed)."""
    diff = X - y
    return np.einsum("jd,jd->j", diff, diff)


def sq_dist_matrix(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Dm[r, j] = ||y_r - x_j||^2, shape (R, N). Cached and updated incrementally."""
    diff = Y[:, None, :] - X[None, :, :]
    return np.einsum("rjd,rjd->rj", diff, diff)


# =============================================================================
# SECTION C -- Plackett-Luce log-likelihood (vectorized, log-sum-exp stable)
# =============================================================================
# For eta (…, N) and orders (…, N) [item indices best -> worst]:
#   log P = sum_t eta[k_t] - sum_t logsumexp(eta[k_t], ..., eta[k_N])
# The stage-t denominator is a *suffix* logsumexp of eta re-ordered by rank,
# computed in one pass with np.logaddexp.accumulate on the reversed axis.


def log_pl_batch(eta: np.ndarray, orders: np.ndarray) -> float:
    """
    eta    : (..., N) support scores
    orders : (..., N) integer item indices, best -> worst
    Returns the SUM of Plackett-Luce log-probabilities over all leading axes.
    """
    e = np.take_along_axis(eta, orders, axis=-1)          # eta sorted by rank position
    suffix_lse = np.logaddexp.accumulate(e[..., ::-1], axis=-1)  # lse over suffixes
    return float(np.sum(e) - np.sum(suffix_lse))


def log_pl_naive(order: np.ndarray, eta: np.ndarray) -> float:
    """Reference O(N^2) implementation (validation only)."""
    remaining = list(order.astype(int))
    logp = 0.0
    for t in range(len(remaining)):
        rem = np.array(remaining[t:], dtype=int)
        m = np.max(eta[rem])
        logp += eta[order[t]] - (m + np.log(np.sum(np.exp(eta[rem] - m))))
    return logp


# =============================================================================
# SECTION D -- Log-priors
# =============================================================================


def logpdf_normal(x: np.ndarray, sigma2: float = 1.0) -> float:
    return -0.5 * np.sum(x * x) / sigma2 - 0.5 * x.size * math.log(2 * math.pi * sigma2)


GAMMA_SHAPE, GAMMA_SCALE = 25.0, 1.0 / 24.0
_GAMMA_LOGNORM = -GAMMA_SHAPE * math.log(GAMMA_SCALE) - math.lgamma(GAMMA_SHAPE)


def logpdf_gamma_scalar(bval: float) -> float:
    if bval <= 0:
        return -np.inf
    return (GAMMA_SHAPE - 1.0) * math.log(bval) - bval / GAMMA_SCALE + _GAMMA_LOGNORM


def log_prior(Y: np.ndarray, X: np.ndarray, c: np.ndarray, b: np.ndarray) -> float:
    return (
        logpdf_normal(Y)                     # y_r ~ N(0, I_D), r = 1..R
        + logpdf_normal(X)                   # x_j ~ N(0, I_D)
        + logpdf_normal(c)                   # c_j ~ N(0, 1)
        + float(sum(logpdf_gamma_scalar(float(t)) for t in b))
    )


# =============================================================================
# SECTION E -- Log-likelihood blocks (locality-aware, using cached Dm)
# =============================================================================
# eta for participant s, target r:  eta_sr = c - b_s * Dm[r]
# y_r enters the rankings of ALL participants for target r (column r);
# b_s enters ALL targets of participant s (row s); x_j, c_j enter everything.


def ll_target(Dm_r: np.ndarray, c: np.ndarray, b: np.ndarray, orders_r: np.ndarray) -> float:
    """Sum over participants of log P(k_sr) for one target r.
    Dm_r: (N,) distances; orders_r: (S, N)."""
    eta = c[None, :] - b[:, None] * Dm_r[None, :]          # (S, N)
    return log_pl_batch(eta, orders_r)


def ll_participant(Dm: np.ndarray, c: np.ndarray, b_s: float, orders_s: np.ndarray) -> float:
    """Sum over targets of log P(k_sr) for one participant s.
    Dm: (R, N); orders_s: (R, N)."""
    eta = c[None, :] - b_s * Dm                            # (R, N)
    return log_pl_batch(eta, orders_s)


def ll_all(Dm: np.ndarray, c: np.ndarray, b: np.ndarray, rankings: np.ndarray) -> float:
    """Full log-likelihood: rankings (S, R, N)."""
    eta = c[None, None, :] - b[:, None, None] * Dm[None, :, :]   # (S, R, N)
    return log_pl_batch(eta, rankings)


def log_posterior(Y, X, c, b, rankings) -> float:
    return ll_all(sq_dist_matrix(Y, X), c, b, rankings) + log_prior(Y, X, c, b)


# =============================================================================
# SECTION F -- Configuration stacking for Procrustes
# =============================================================================
# C = [y_1; ...; y_R; x_1; ...; x_N], shape (R + N, D).
# c_j and b_s are NOT part of the configuration (rotation/translation invariant).


def stack_configuration(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.vstack([Y, X])


def unstack_configuration(C: np.ndarray, R: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    return C[:R].copy(), C[R:].copy()


# =============================================================================
# SECTION G -- Orthogonal Procrustes alignment
# =============================================================================
# min_Q ||C_ref - C_hat Q||_F over orthogonal Q after row-centering both.
# Reflections are permitted: the squared-Euclidean likelihood is invariant to
# translation, rotation AND reflection, so the unconstrained orthogonal
# solution Q = U V' (SVD of C_hat' C_ref) is used without a det(Q) > 0 guard.


def center_rows(C: np.ndarray) -> np.ndarray:
    return C - C.mean(axis=0)


def orthogonal_procrustes(C_ref: np.ndarray, C_hat: np.ndarray) -> np.ndarray:
    A, B = center_rows(C_ref), center_rows(C_hat)
    U, _, Vt = np.linalg.svd(B.T @ A, full_matrices=True)
    return U @ Vt


def align_to_reference(Y: np.ndarray, X: np.ndarray, CR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C_hat = stack_configuration(Y, X)
    Q = orthogonal_procrustes(CR, C_hat)
    C_aligned = center_rows(C_hat) @ Q
    return unstack_configuration(C_aligned, Y.shape[0], X.shape[0])


# =============================================================================
# SECTION H -- Metropolis-Hastings sampler
# =============================================================================
# One sweep: (1) each y_r, (2) each x_j, (3) each c_j -- Gaussian RW, symmetric
# proposals (Hastings ratio 1); (4) each b_s -- log-scale RW, acceptance
# includes the Jacobian term log(b*/b).
# MAP phase: strict uphill moves on the posterior density (Jacobian not
# included -- MAP maximizes the density of b itself, not of log b).
# Sampling phase: standard MH; after every sweep the configuration is
# Procrustes-aligned to the MAP reference C_R; post-burn-in draws stored.


def mcmc_latent_pl(
    rankings: np.ndarray,
    D: int,
    *,
    n_iter: int = 800,
    n_map: int = 150,
    burn_in: int = 200,
    sigma_prop_y: float = 0.25,
    sigma_prop_x: float = 0.08,
    sigma_prop_c: float = 0.12,
    sigma_prop_log_b: float = 0.12,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    S, R, N = rankings.shape

    # Initialization from the priors
    Y = rng.normal(size=(R, D))          # <-- (R, D): one point per target, SHARED
    X = rng.normal(size=(N, D))
    c = rng.normal(size=N)
    b = rng.gamma(GAMMA_SHAPE, GAMMA_SCALE, size=S)

    Dm = sq_dist_matrix(Y, X)            # cached (R, N) squared distances
    orders_by_target = [rankings[:, r, :] for r in range(R)]   # each (S, N)

    accept = {"y": 0, "x": 0, "c": 0, "b": 0}
    total = {"y": 0, "x": 0, "c": 0, "b": 0}

    def sweep(uphill: bool) -> None:
        nonlocal Y, X, c, b, Dm
        # (1) y_r : column-r likelihood + N(0, I) prior
        for r in range(R):
            prop = Y[r] + rng.normal(scale=sigma_prop_y, size=D)
            d_new = sq_dist_row(prop, X)
            log_a = (
                ll_target(d_new, c, b, orders_by_target[r]) + logpdf_normal(prop)
                - ll_target(Dm[r], c, b, orders_by_target[r]) - logpdf_normal(Y[r])
            )
            total["y"] += 1
            if (log_a > 0) if uphill else (np.log(rng.random()) < min(0.0, log_a)):
                Y[r] = prop
                Dm[r] = d_new
                accept["y"] += 1
        # (2) x_j : affects every ranking; only column j of Dm changes
        for j in range(N):
            prop = X[j] + rng.normal(scale=sigma_prop_x, size=D)
            col_new = np.einsum("rd,rd->r", Y - prop, Y - prop)   # (R,)
            Dm_new = Dm.copy()
            Dm_new[:, j] = col_new
            log_a = (
                ll_all(Dm_new, c, b, rankings) + logpdf_normal(prop)
                - ll_all(Dm, c, b, rankings) - logpdf_normal(X[j])
            )
            total["x"] += 1
            if (log_a > 0) if uphill else (np.log(rng.random()) < min(0.0, log_a)):
                X[j] = prop
                Dm = Dm_new
                accept["x"] += 1
        # (3) c_j : affects every ranking; distances unchanged
        for j in range(N):
            cp = c.copy()
            cp[j] = c[j] + rng.normal(scale=sigma_prop_c)
            log_a = (
                ll_all(Dm, cp, b, rankings) + logpdf_normal(np.array([cp[j]]))
                - ll_all(Dm, c, b, rankings) - logpdf_normal(np.array([c[j]]))
            )
            total["c"] += 1
            if (log_a > 0) if uphill else (np.log(rng.random()) < min(0.0, log_a)):
                c = cp
                accept["c"] += 1
        # (4) b_s : row-s likelihood; log-scale RW with Jacobian in MH phase
        for s in range(S):
            m = math.exp(rng.normal(scale=sigma_prop_log_b))
            b_new = b[s] * m
            log_a = (
                ll_participant(Dm, c, b_new, rankings[s]) + logpdf_gamma_scalar(b_new)
                - ll_participant(Dm, c, float(b[s]), rankings[s]) - logpdf_gamma_scalar(float(b[s]))
            )
            if not uphill:
                log_a += math.log(b_new / b[s])   # Jacobian of theta -> log theta
            total["b"] += 1
            if (log_a > 0) if uphill else (np.log(rng.random()) < min(0.0, log_a)):
                b[s] = b_new
                accept["b"] += 1

    # -------- MAP / uphill phase: build the Procrustes reference C_R --------
    for it in range(n_map):
        if verbose and it % max(1, n_map // 3) == 0:
            print(f"    MAP phase {it}/{n_map} (D={D})", flush=True)
        sweep(uphill=True)
    CR = center_rows(stack_configuration(Y, X))

    # -------- Sampling phase --------
    samples_Y, samples_X, samples_c, samples_b = [], [], [], []
    for it in range(n_iter):
        if verbose and it % max(1, n_iter // 4) == 0:
            print(f"    MH sampling {it}/{n_iter} (D={D})", flush=True)
        sweep(uphill=False)
        Y, X = align_to_reference(Y, X, CR)   # draw-by-draw Procrustes (Y, X only)
        Dm = sq_dist_matrix(Y, X)             # distances invariant in theory; recompute
        if it >= burn_in:                     # to preclude numerical drift
            samples_Y.append(Y.copy())
            samples_X.append(X.copy())
            samples_c.append(c.copy())
            samples_b.append(b.copy())

    rates = {k: accept[k] / max(1, total[k]) for k in accept}
    if verbose:
        print(f"    acceptance rates (D={D}):",
              {k: round(v, 3) for k, v in rates.items()})

    return {
        "samples_Y": np.stack(samples_Y),   # (T, R, D)
        "samples_X": np.stack(samples_X),   # (T, N, D)
        "samples_c": np.stack(samples_c),   # (T, N)
        "samples_b": np.stack(samples_b),   # (T, S)
        "CR": CR,
        "D": D,
        "acceptance": rates,
    }


# =============================================================================
# SECTION I -- Validation, summaries, plots, driver
# =============================================================================


def validate_likelihood(rng_seed: int = 1) -> None:
    """Check vectorized suffix-LSE Plackett-Luce against the naive version."""
    rng = np.random.default_rng(rng_seed)
    N = 7
    eta = rng.normal(scale=2.0, size=N)
    order = rng.permutation(N)
    fast = log_pl_batch(eta[None, :], order[None, :])
    slow = log_pl_naive(order, eta)
    assert abs(fast - slow) < 1e-10, (fast, slow)
    # a full ranking over all N! permutations must sum to probability 1 (small N check)
    from itertools import permutations
    tot = sum(math.exp(log_pl_naive(np.array(p), eta[:4])) for p in permutations(range(4)))
    assert abs(tot - 1.0) < 1e-10, tot
    print("Likelihood validation passed (suffix-LSE == naive; permutations sum to 1).")


def summarize_draws(fit: dict, n_tail: int = 6) -> None:
    Sb, Sc = fit["samples_b"], fit["samples_c"]
    SX, SY = fit["samples_X"], fit["samples_Y"]
    T = Sb.shape[0]
    sl = slice(max(0, T - n_tail), T)
    print(f"\n--- Posterior tail (last {n_tail} of {T} draws), D = {fit['D']} ---")
    print("b posterior mean (per participant):\n", np.round(Sb.mean(0), 3))
    print("c posterior mean (per multiphonic):\n", np.round(Sc.mean(0), 3))
    print("last draws of b (rows=iter):\n", np.round(Sb[sl], 3))
    print("multiphonic locations, posterior mean:\n", np.round(SX.mean(0), 3))
    print("orchestral target locations, posterior mean:\n", np.round(SY.mean(0), 3))


def plot_latent_panel_static(fit: dict, fname: str) -> None:
    D = int(fit["D"])
    X_mean, Y_mean = fit["samples_X"].mean(0), fit["samples_Y"].mean(0)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    if D == 1:
        ax0.scatter(X_mean[:, 0], np.zeros(len(X_mean)), c="tab:blue", s=90, label="multiphonics", zorder=3)
        ax0.scatter(Y_mean[:, 0], np.zeros(len(Y_mean)), c="tab:orange", s=45, marker="s", label="orchestral targets")
        ax0.set_yticks([])
        ax0.set_xlabel("Dim 1")
    else:
        ax0.scatter(X_mean[:, 0], X_mean[:, 1], c="tab:blue", s=110, zorder=3, label="multiphonics")
        for j in range(len(X_mean)):
            ax0.annotate(f"M{j+1}", X_mean[j, :2], xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax0.scatter(Y_mean[:, 0], Y_mean[:, 1], c="tab:orange", s=55, marker="s", label="orchestral targets")
        for r in range(len(Y_mean)):
            ax0.annotate(f"T{r+1}", Y_mean[r, :2], xytext=(4, -8), textcoords="offset points", fontsize=7, color="darkorange")
        ax0.set_xlabel("Dim 1")
        ax0.set_ylabel("Dim 2")
    ax0.set_title(f"Posterior mean latent space (D={D})")
    ax0.legend(fontsize=8)
    ax1.plot(fit["samples_b"][:, 0], lw=0.7, label="b_1")
    ax1.plot(fit["samples_c"][:, 0], lw=0.7, label="c_1")
    ax1.legend(fontsize=8)
    ax1.set_title("Traces (aligned chain)")
    ax1.set_xlabel("Saved iteration")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_latent_panel(fit: dict) -> None:
    """Interactive: dimension-pair sliders (D>2) + traces."""
    D = int(fit["D"])
    X_mean, Y_mean = fit["samples_X"].mean(0), fit["samples_Y"].mean(0)
    N, R = len(X_mean), len(Y_mean)

    fig = plt.figure(figsize=(11, 7))
    ax_scatter = fig.add_axes([0.08, 0.35, 0.52, 0.55])
    ax_dim1 = fig.add_axes([0.68, 0.55, 0.25, 0.03])
    ax_dim2 = fig.add_axes([0.68, 0.48, 0.25, 0.03])
    ax_trace = fig.add_axes([0.08, 0.08, 0.85, 0.18])
    d = [0, min(1, D - 1)]

    def redraw():
        ax_scatter.clear()
        if D == 1:
            ax_scatter.scatter(X_mean[:, 0], np.zeros(N), c="tab:blue", s=90, label="multiphonics", zorder=3)
            ax_scatter.scatter(Y_mean[:, 0], np.zeros(R), c="tab:orange", s=45, marker="s", label="orchestral targets")
            ax_scatter.set_yticks([])
        else:
            ax_scatter.scatter(X_mean[:, d[0]], X_mean[:, d[1]], c="tab:blue", s=110, zorder=3, label="multiphonics")
            for j in range(N):
                ax_scatter.annotate(f"M{j+1}", (X_mean[j, d[0]], X_mean[j, d[1]]),
                                    xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax_scatter.scatter(Y_mean[:, d[0]], Y_mean[:, d[1]], c="tab:orange", s=55, marker="s", label="orchestral targets")
            for r in range(R):
                ax_scatter.annotate(f"T{r+1}", (Y_mean[r, d[0]], Y_mean[r, d[1]]),
                                    xytext=(4, -8), textcoords="offset points", fontsize=7, color="darkorange")
            ax_scatter.set_xlabel(f"Dim {d[0] + 1}")
            ax_scatter.set_ylabel(f"Dim {d[1] + 1}")
        ax_scatter.set_title(f"Posterior mean latent space (D={D}, Procrustes-aligned)")
        ax_scatter.legend(loc="best", fontsize=8)
        fig.canvas.draw_idle()

    redraw()
    if D >= 2:
        s1 = Slider(ax_dim1, "x dim", 0, D - 1, valinit=0, valstep=1)
        s2 = Slider(ax_dim2, "y dim", 0, D - 1, valinit=min(1, D - 1), valstep=1)

        def on_slider(_):
            d[0], d[1] = int(s1.val), int(s2.val)
            if d[0] == d[1] and D > 1:
                d[1] = (d[0] + 1) % D
                s2.set_val(d[1])
            redraw()

        s1.on_changed(on_slider)
        s2.on_changed(on_slider)

    ax_trace.plot(fit["samples_b"][:, 0], lw=0.6, label="b_1 trace")
    ax_trace.plot(fit["samples_c"][:, 0], lw=0.6, label="c_1 trace")
    ax_trace.legend(fontsize=7, ncol=2)
    ax_trace.set_title("Posterior traces (aligned chain)")
    ax_trace.set_xlabel("Saved iteration")
    plt.show()


def run_all_dimensions(rankings: np.ndarray, dims: tuple[int, ...] = (1, 2, 3), **kw) -> dict[int, dict]:
    fits: dict[int, dict] = {}
    for j, D in enumerate(dims):
        print(f"\n=== MCMC for latent dimension D = {D} ===")
        fits[D] = mcmc_latent_pl(rankings, D, seed=100 + j, **kw)
        summarize_draws(fits[D])
    return fits


if __name__ == "__main__":
    import os

    validate_likelihood()

    df = build_dataframe(raw_data)
    df.to_csv("multiphonics_rankings.csv", index=False)
    rankings_np = rankings_from_csv("multiphonics_rankings.csv")
    S, R, N = rankings_np.shape
    print(f"Loaded dataset: {S} participants x {R} orchestral targets x {N} multiphonics")

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

    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        for d, fit in all_fits.items():
            plot_latent_panel_static(fit, f"latent_space_D{d}.png")
        print("Saved latent_space_D1.png, latent_space_D2.png, latent_space_D3.png")
    else:
        fig_pick = plt.figure(figsize=(5, 3))
        axp = fig_pick.add_axes([0.15, 0.2, 0.7, 0.65])
        radio = RadioButtons(axp, [f"D = {d}" for d in all_fits], active=1)

        def onselect(sel: str):
            d = int(str(sel).split("=")[1].strip())
            plt.close(fig_pick)
            plot_latent_panel(all_fits[d])

        radio.on_clicked(onselect)
        plt.suptitle("Choose latent dimensionality to plot")
        plt.show()
