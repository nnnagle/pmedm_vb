"""Scores of a region's fitted distribution against the PUMS truth.

Per zone (a PUMA of the region), the prediction is ``N p_b``: the region's
units weighted by their mass in that zone. It is scored with the
leave-one-PUMA-out check's own functions
(:func:`~pmedm_vb.embed.puma_check.score_tables`, ``rare_groups``,
``misfit``) against the zone's own units with their design weights, so the
numbers are on the check's scale. Two scores are new here:

* **Allocation accuracy:** the design-weighted mean, over the region's units,
  of the share of each unit's mass placed in its true zone,
  ``p_{t(i), i} / Σ_b p_{b, i}``. The design distribution ``q`` spreads every
  unit evenly, scoring ``1 / Z``.
* **Spread of rare groups across zones:** for each rare group, the RMS over
  zones of ``log(prediction / truth)`` (zones where both are positive), and
  the share of the group's predicted total placed in the zone holding most of
  its truth, beside that zone's true share. A model that spreads a rare group
  too evenly scores a low predicted share where the truth is concentrated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmedm_vb.embed.puma_check import CheckData, RARE, misfit, rare_groups, score_tables
from pmedm_vb.latent.regions import RegionProblem


def allocation_accuracy(problem: RegionProblem, p: np.ndarray) -> float:
    mass = p.sum(axis=0)
    own = p[problem.zone_of_unit, np.arange(p.shape[1])]
    share = np.where(mass > 0, own / np.maximum(mass, 1e-300), 0.0)
    w = problem.inputs.units["weight"].to_numpy()
    return float(w @ share / w.sum())


def score_region(data: CheckData, problem: RegionProblem, p: np.ndarray, model: str,
                 se_cells: np.ndarray) -> dict[str, pd.DataFrame]:
    """Per-zone table scores, rare groups and misfit, and the region's
    allocation accuracy and rare-group spread, for one model's ``p``."""
    region = problem.region
    N = problem.inputs.N
    n_all = len(data.weight)
    scores, rare, fit = [], [], []
    for b, puma in enumerate(region.pumas):
        p_full = np.zeros(n_all)
        p_full[problem.units] = p[b]
        in_p = np.zeros(n_all, dtype=bool)
        in_p[problem.units[problem.zone_of_unit == b]] = True
        floor = float(data.weight[in_p].mean())
        base = {"region": region.id, "zone": puma, "model": model}
        scores.extend({**base, **r} for r in score_tables(data, p_full, N, in_p, 0.5 * floor))
        rare.extend({**base, **r} for r in rare_groups(data, p_full, N, in_p, None, puma))
        X = problem.inputs.X_B
        y = problem.inputs.Y_B[b]
        se = se_cells.reshape(len(problem.cells), len(region.pumas))[:, b]
        fit.append({**base, **misfit(X, y, se, p[b], N),
                    "pred_units": float(N * p[b].sum()),
                    "true_units": float(problem.zone_weight[b])})
    rare_frame = pd.DataFrame(rare)
    spread = []
    for (table, category), group in rare_frame.groupby(["table", "category"]):
        both = group[(group["truth"] > 0) & (group["pred"] > 0)]
        rms = float(np.sqrt(np.mean(np.log(both["pred"] / both["truth"]) ** 2))) if len(both) else np.nan
        total_truth, total_pred = group["truth"].sum(), group["pred"].sum()
        top = group.loc[group["truth"].idxmax()]
        spread.append({
            "region": region.id, "model": model, "table": table, "category": category,
            "truth": float(total_truth), "pred": float(total_pred),
            "rms_log_ratio": rms, "zones_scored": int(len(both)),
            "top_zone_true_share": float(top["truth"] / total_truth) if total_truth > 0 else np.nan,
            "top_zone_pred_share": float(top["pred"] / total_pred) if total_pred > 0 else np.nan,
        })
    summary = {"region": region.id, "model": model,
               "allocation_accuracy": allocation_accuracy(problem, p),
               "uniform_accuracy": 1.0 / len(region.pumas)}
    return {"scores": pd.DataFrame(scores), "rare": rare_frame, "misfit": pd.DataFrame(fit),
            "spread": pd.DataFrame(spread), "summary": pd.DataFrame([summary])}


def transfer_misfit(other: RegionProblem, p: np.ndarray, model: str, fit_set: str,
                    scored_set: str) -> pd.DataFrame:
    """Misfit of ``p`` to another table set's cells, per zone: the transfer
    test. ``other`` is the region's problem for the scored set (same units,
    in the same order); its SEs are the replicate SEs of those cells without
    the PMEDM's prior."""
    region = other.region
    N = other.inputs.N
    se = np.sqrt(other.inputs.sigma_v).reshape(len(other.cells), len(region.pumas))
    rows = []
    for b, puma in enumerate(region.pumas):
        rows.append({"region": region.id, "zone": puma, "model": model, "fit_set": fit_set,
                     "scored_set": scored_set,
                     **misfit(other.inputs.X_B, other.inputs.Y_B[b], se[:, b], p[b], N)})
    return pd.DataFrame(rows)


__all__ = ["RARE", "allocation_accuracy", "score_region", "transfer_misfit"]
