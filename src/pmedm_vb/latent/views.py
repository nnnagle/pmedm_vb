"""The latent-view PMEDM: the exact PMEDM with each unit's constraint-cell
counts replaced by their expected values given its latent position.

For unit i with category-probability features h(φ_i) (the calibrated
decoder's, here every attribute's) and constraint cells c,

    x̂_ic = E[x_ic | φ_i] ≈ [1, h(φ_i)]′ g_c,

``G = [g_c]`` fitted by design-weighted least squares of each cell's per-unit
count on ``[1, h]`` over the statewide PUMS (no zone labels, no truth).
Fitted values may be negative: the cells are linear *views* of the latent
features. The PMEDM is then solved as it stands (the existing dual solver, the
regional baseline's settings) with ``X`` replaced by ``X̂ = [1, h] G``: tilt
and constraint features are the same, so it is the usual convex maximum
entropy dual, now over latent factors, with zone b's tilt
``η_ib = [1, h_i]′ G λ_b`` (zone latent coefficients ``θ_b = G λ_b``).
Allocations use the real records; only the constraint features change.

**Approximation error.** The tables are fitted with ``X̂`` but are sums of
``X``. With the true allocation (unit i in its zone b with weight w_i), the
zone total's error is ``Σ_{i∈b} w_i e_ic``, ``e = x − x̂``, of variance about
``Σ_{i∈b} w_i² s_c² ≈ N_b w̄ s_c²``, with ``s_c²`` the cell's statewide
design-weighted residual variance, ``N_b`` the zone's unit total and
``w̄ = Σ w_i² / Σ w_i`` over the area. ``κ`` times this is added to each
cell's variance (``sigma_v``): a diagonal approximation; the residuals'
covariance between cells is not carried. κ = 0 is the plain latent-view
PMEDM.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.embed.puma_check import CheckData
from pmedm_vb.latent.regions import RegionProblem


@dataclass
class Views:
    """Least-squares views of one table set's cells on ``[1, h]``."""

    G: np.ndarray              # (1 + features) x cells
    cells: list[str]
    residual_var: np.ndarray   # per cell, design-weighted, statewide
    r2: pd.DataFrame           # per table: weighted R²


def design(F: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(F)), np.asarray(F, dtype=np.float64)])


def fit_views(data: CheckData, tables: tuple[str, ...], F: np.ndarray,
              chunk: int = 20000) -> Views:
    """Design-weighted least squares of every cell of ``tables`` on ``[1, F]``
    over all statewide units."""
    X = sp.hstack([data.counts[t] for t in tables]).tocsr().astype(np.float64)
    cells = [f"{t}:{c.name}" for t in tables for c in data.tables[t].categories]
    table_of = np.array([c.split(":", 1)[0] for c in cells])
    w = np.asarray(data.weight, dtype=np.float64)
    D = design(F)
    gram = D.T @ (D * w[:, None])
    cross = np.asarray((X.T @ (D * w[:, None])).T)
    G = np.linalg.lstsq(gram, cross, rcond=1e-12)[0]
    mean = np.asarray(X.T @ w).ravel() / w.sum()
    sst = np.asarray(X.multiply(X).T @ w).ravel() - w.sum() * mean**2
    ssr = np.zeros(X.shape[1])
    for start in range(0, X.shape[0], chunk):
        rows = slice(start, start + chunk)
        e = X[rows].toarray() - D[rows] @ G
        ssr += w[rows] @ np.square(e)
    r2 = (pd.DataFrame({"table": table_of, "ssr": ssr, "sst": sst})
          .groupby("table", sort=False)[["ssr", "sst"]].sum())
    r2["r2"] = 1.0 - r2["ssr"] / r2["sst"]
    r2["cells"] = pd.Series(table_of).value_counts().reindex(r2.index).to_numpy()
    return Views(G=G, cells=cells, residual_var=ssr / w.sum(),
                 r2=r2.reset_index()[["table", "cells", "r2"]])


def approximation_variance(problem: RegionProblem, views: Views) -> np.ndarray:
    """``N_b w̄ s_c²`` in the stacked order (cell-major, zone fastest)."""
    w = problem.inputs.units["weight"].to_numpy(dtype=np.float64)
    w_bar = float(np.square(w).sum() / w.sum())
    return np.outer(views.residual_var, problem.zone_weight * w_bar).ravel()


def latent_view_inputs(problem: RegionProblem, views: Views, F_units: np.ndarray,
                       kappa: float = 0.0, X: sp.spmatrix | np.ndarray | None = None
                       ) -> PMEDMInputs:
    """The region's inputs with ``X_B = [1, h] G`` (or ``X`` if given, for the
    check that ``X̂ = X`` reproduces the PMEDM) and ``κ`` times the
    approximation variance added to ``sigma_v``."""
    if views.cells != problem.cells:
        raise ValueError("the views' cells are not the problem's")
    X_hat = sp.csr_matrix(design(F_units) @ views.G if X is None else X)
    sigma_v = problem.inputs.sigma_v
    if kappa:
        sigma_v = sigma_v + float(kappa) * approximation_variance(problem, views)
    inputs = replace(problem.inputs, X_B=X_hat, sigma_v=sigma_v)
    inputs.validate()
    return inputs


__all__ = ["Views", "approximation_variance", "design", "fit_views", "latent_view_inputs"]
