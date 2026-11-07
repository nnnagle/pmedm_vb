"""Held-out table scores of a posterior's draws, shared by ``compare_support.py`` and ``compare_variants.py``.

The held-out tables (``run_map.py heldout``; :class:`pmedm_vb.assemble.heldout.HeldOut`)
are published estimates no fit sees. A posterior draw ``W`` (persons per block
group and record) predicts each held-out cell: ``X_b' W`` per block group, and
per tract the block groups' ``X_t' W`` summed by ``A_T``
(:class:`HeldoutPredictor`). Over the draws, each cell is scored against the
published estimate ``Y`` and its sampling variance ``v``
(:func:`score_cells`): the error of the posterior mean, the predictive
``z = (mean - Y) / sqrt(v + sd^2)`` (the published value's sampling error plus
the posterior's own spread), whether ``|z| <= 1.645`` (90% predictive
coverage) and the Gaussian log score ``log N(Y; mean, v + sd^2)``. A cell with
no spread at all (zero variance and sd) has no score (NaN).

Cells are grouped by level and by their relation to the fitted tables
(``HeldOut.relation``) and, as their own group, the race x poverty cells
(``B17010A-I``, :data:`RACE_POVERTY`): the joint structure the prior, not the
tables, decides (:func:`score_groups`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

#: The extra group of held-out cells reported on their own: B17010A-I, families
#: by poverty status by race of householder.
RACE_POVERTY = ("B17010", "B17010 race x poverty")
#: The per-cell measures of :func:`score_cells`.
MEASURES = ("mean", "sd", "error", "pred_z", "covers", "log_score")
LEVELS = ("block group", "tract")


class HeldoutPredictor:
    """A population's held-out cells: ``(block group cells (zones, k_b), tract
    cells (tracts, k_t))`` for ``W`` (zones, rows)."""

    def __init__(self, inputs, heldout):
        self.h_b, self.h_t = (sp.csr_matrix(heldout.X[level]) for level in LEVELS)
        self.A_T = sp.csr_matrix(inputs.A_T)

    def __call__(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.h_b.T @ W.T).T, self.A_T @ (self.h_t.T @ W.T).T


def predict(inputs, heldout, lam: np.ndarray) -> dict[str, np.ndarray]:
    """Each column of ``lam`` (``(m, draws)``, the multipliers the data see) made
    a population ``W = N p(lambda)``; its held-out cells per level, ``(draws,
    areas, k)`` float32."""
    from pmedm_vb.solvers.base import ConstraintOperator

    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    held = HeldoutPredictor(inputs, heldout)
    d = lam.shape[1]
    out = {"block group": np.empty((d, inputs.n_zones, held.h_b.shape[1]), dtype=np.float32),
           "tract": np.empty((d, held.A_T.shape[0], held.h_t.shape[1]), dtype=np.float32)}
    for k in range(d):
        logits = log_q - op.adjoint(lam[:, k])
        W = inputs.N * np.exp(logits - logsumexp(logits))
        out["block group"][k], out["tract"][k] = held(W)
    return out


def score_cells(draws: np.ndarray, Y: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Per cell of ``(draws, areas, k)`` predictions against published ``Y`` and
    variance ``v``: mean, sd, error, predictive z, 90% predictive coverage and a
    Gaussian log score. Cells with no spread at all (zero variance and sd) are NaN."""
    mean, sd = draws.mean(0).astype(float), draws.std(0, ddof=1).astype(float)
    spread = np.sqrt(np.asarray(v, float) + sd**2)
    ok = spread > 0
    safe = np.where(ok, spread, 1.0)
    pred_z = np.where(ok, (mean - Y) / safe, np.nan)
    return {"mean": mean, "sd": sd, "error": mean - Y, "pred_z": pred_z,
            "covers": np.where(ok, np.abs(pred_z) <= 1.645, np.nan),
            "log_score": np.where(ok, -0.5 * np.log(2 * np.pi * safe**2) - 0.5 * pred_z**2, np.nan)}


def area_names(inputs) -> dict[str, list[str]]:
    """The held-out cells' areas per level: block groups (zones) and tracts."""
    return {"block group": list(inputs.zones["block_group_geoid"]),
            "tract": list(inputs.tracts.iloc[:, 0].astype(str))}


def level_frame(heldout, level: str, areas: list[str],
                scores: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """One row per held-out cell of ``level`` (area-major): its area, name,
    relation, published value and SE, and ``<tag>_<measure>`` for each
    ``scores[tag]`` (:func:`score_cells`)."""
    Y, v = heldout.Y[level], heldout.v[level]
    names, relation = heldout.names[level], heldout.relation(level)
    return pd.DataFrame({
        "level": level, "area": np.repeat(areas, len(names)),
        "cell": np.tile(names, len(areas)),
        "relation": np.tile(relation, len(areas)),
        "published": Y.ravel(), "se": np.sqrt(v).ravel(),
        **{f"{tag}_{m}": vals.ravel() for tag, sc in scores.items() for m, vals in sc.items()},
    })


def score_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """One level's cells grouped by relation, then the race x poverty cells
    (:data:`RACE_POVERTY`) as a group of their own when there are any."""
    groups = list(frame.groupby("relation"))
    prefix, label = RACE_POVERTY
    race_poverty = frame[frame["cell"].str.startswith(prefix)]
    if len(race_poverty):
        groups.append((label, race_poverty))
    return groups


def group_summary(group: pd.DataFrame, tag: str) -> dict:
    """One posterior's scores (columns ``<tag>_<measure>``) over a group of cells:
    the scored cells, mean |error| and RMSE of the posterior mean, the mean and
    mean square of the predictive z, 90% predictive coverage and the mean
    Gaussian log score."""
    ok = group.dropna(subset=[f"{tag}_pred_z"])
    error, z = ok[f"{tag}_error"], ok[f"{tag}_pred_z"]
    return {"cells": len(ok), "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt((error**2).mean())), "pred_z_mean": float(z.mean()),
            "pred_z2": float((z**2).mean()), "cover90": float(ok[f"{tag}_covers"].mean()),
            "log_score": float(ok[f"{tag}_log_score"].mean())}
