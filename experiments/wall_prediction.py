"""Can the walls be predicted before drawing? Scores every cell, checks against events.

A cell (block group ``b``, record ``i``) reaches a share ``t`` of ``N`` when its
logit rises, relative to the MAP, by ``c_bi = log t - log p*_bi``. The rise is
``s = -a' delta`` for ``a`` the cell's wall normal (``wall_attribution.py``)
and ``delta = lambda - lambda*``. Under a Gaussian ``N(lambda* + m, C)``,
``s ~ N(-a' m, a' C a)``, so

.. math::

    P(\\text{event at } bi) \\approx \\Phi\\big((-a^\\top m - c_{bi}) / \\sqrt{a^\\top C a}\\big),

ignoring the change in the normaliser, which is small while no cell holds
much of ``p``. Three scores, from cheapest to closest to VB:

- ``structural``: the record's largest loading on a category whose cell in
  that block group is published within 2 SE of zero (floored SE), plus a
  thousandth of household size to break ties. From X and the tables alone.
- ``laplace``: the probability above with ``C = (n H)^{-1}``, the MAP
  Laplace approximation, ``m = 0``. Available right after the MAP solve.
- ``vb``: the same with the fitted family's Gaussian part (``m`` its mean
  less ``lambda*``, ``C`` its covariance), ignoring its skew layers.

``a' C a = ||G^{-1} a||^2`` for ``C = (G G')^{-1}``,
``G = blockdiag(L_t)(I + W V')``: one triangular solve per block and a
Woodbury correction, for every cell at once.

Each score ranks every cell (units that load on something, ``q > 0``). The
cells that actually had events in ``wall_attribution.py``'s draws
(``--events``, its ``events.csv``) are the truth. Reported per fit and score:
AUC (event cells against the rest), and the share of distinct event cells
and of events captured by the top ``K`` cells for each ``--top``.

Also per fit: the median marginal sd ``sqrt(a' C a)`` along the wall normals
of event and of other cells, under Laplace and under VB -- how far draws
spread along a wall, which is what matters, as against the curvature along
it, which ``wall_attribution.py`` reports; the VB score's *calibration*,
``--draws`` times the summed per-cell probabilities against the events
observed; and how *concentrated* that predicted risk is, the number of cells
holding 50% and 90% of it -- roughly how many cells a wall-aware family would
have to cover.

Writes ``<out>/prediction.csv`` (one row per fit, score and K),
``<out>/fit_stats.csv`` (one row per fit: the spread, calibration and
concentration figures),
``<out>/top_cells.csv`` (each score's top ``max(--top)`` cells per fit, with
all three scores and the observed events) and ``<out>/summary.txt``::

    $CONDA_PREFIX/bin/python experiments/wall_prediction.py --variance-floor zero \\
        --events $RUNS/wall_attribution/events.csv \\
        --fit 4701502 1.0 $RUNS/6277113 ... --out $RUNS/wall_prediction
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.progress import logger, record_run
from pmedm_vb.solvers.base import ConstraintOperator
from pmedm_vb.solvers.vb import StructuredGaussian

from laplace_diagnostic import load_vb

SCORES = ("structural", "laplace", "vb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fit", nargs=3, action="append", required=True,
                        metavar=("PUMA", "ALPHA", "VB_RUN"))
    parser.add_argument("--events", type=Path, required=True, help="wall_attribution.py events.csv")
    parser.add_argument("--taper", default="tract")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--variance-floor", default="zero")
    parser.add_argument("--threshold", type=float, default=0.01, help="share of N, as in the events")
    parser.add_argument("--top", type=int, nargs="+", default=[100, 300, 1000, 3000])
    parser.add_argument("--draws", type=int, default=4000,
                        help="draws behind --events, for the calibration check")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def floor_spec(text: str):
    return None if text == "none" else "zero" if text == "zero" else float(text)


def normal_quadratic(inputs: PMEDMInputs, g: StructuredGaussian) -> np.ndarray:
    """``(n_zones, n_units)``: ``a' C a`` for every cell's wall normal ``a``, with
    ``C = (G G')^{-1}`` the covariance of ``g``'s Gaussian part."""
    n_t, n_z = inputs.Y_T.shape[0], inputs.n_zones
    k_t, k_b = inputs.Y_T.shape[1], inputs.Y_B.shape[1]
    split = inputs.Y_T.size
    zone_tract = inputs.zone_tracts()
    XT = np.asarray(sp.csr_matrix(inputs.X_T).T.todense())  # (k_t, units)
    XB = np.asarray(sp.csr_matrix(inputs.X_B).T.todense())  # (k_b, units)
    r = g.W.shape[1]
    norm2 = np.zeros((n_z, inputs.n_units))
    VY = np.zeros((n_z, r, inputs.n_units))
    WY = np.zeros((n_z, r, inputs.n_units))
    for rows, block in zip(g.rows, g.blocks):
        position = {row: j for j, row in enumerate(rows)}
        for b in range(n_z):
            t_rows = np.arange(k_t) * n_t + zone_tract[b]
            b_rows = split + np.arange(k_b) * n_z + b
            in_t = [(k, position[row]) for k, row in enumerate(t_rows) if row in position]
            in_b = [(k, position[row]) for k, row in enumerate(b_rows) if row in position]
            if not in_t and not in_b:
                continue
            M = np.zeros((len(rows), inputs.n_units))
            for k, j in in_t:
                M[j] += XT[k]
            for k, j in in_b:
                M[j] += XB[k]
            Y = solve_triangular(block, M, lower=True)  # L^{-1} a, this block's part
            norm2[b] += (Y * Y).sum(0)
            VY[b] += g.V[rows].T @ Y
            WY[b] += g.W[rows].T @ Y
    # G^{-1} a = (I + W V')^{-1} y = y - W S^{-1} V' y,  S = I + V' W.
    S = np.eye(r) + g.V.T @ g.W
    WtW = g.W.T @ g.W
    z = np.linalg.solve(S, VY.transpose(1, 0, 2).reshape(r, -1)).reshape(r, n_z, -1).transpose(1, 0, 2)
    return norm2 - 2 * (z * WY).sum(1) + (z * np.einsum("ij,bju->biu", WtW, z)).sum(1)


def structural_score(inputs: PMEDMInputs, floor) -> np.ndarray:
    """Largest loading on a category zero-equivalent in the cell's own block group."""
    split = inputs.Y_T.size
    se = np.sqrt(inputs.floored_variances(floor)[split:]).reshape(inputs.Y_B.shape, order="F")
    zero_equiv = (np.abs(inputs.Y_B) < 2 * se).astype(float)  # (zones, k_b)
    X = sp.csr_matrix(inputs.X_B)
    out = np.zeros((inputs.n_zones, inputs.n_units))
    for b in range(inputs.n_zones):
        out[b] = X.multiply(zero_equiv[b][None, :]).max(axis=1).toarray().ravel()
    age_sex = [k for k, c in enumerate(inputs.bg_constraints) if c.startswith("B01001.")]
    size = np.asarray(X[:, age_sex].sum(axis=1)).ravel()
    return out + 1e-3 * size[None, :]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.sum()
    neg = label.size - pos
    if pos == 0 or neg == 0:
        return np.nan
    ranks = rankdata(score)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    floor = floor_spec(args.variance_floor)
    events = pd.read_csv(args.events, dtype={"block_group": str, "serialno": str, "puma": str})
    rows, tops, lines, stats = [], [], [], []
    for puma, alpha, vb_run in args.fit:
        alpha = float(alpha)
        started = time.perf_counter()
        name = f"{puma}_{args.taper}_a{alpha:g}"
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma)
        q, _, _ = load_vb(Path(vb_run) / f"{name}.npz")
        with np.load(Path(vb_run) / f"{name}.npz") as saved:
            lam_star = saved["map_lam"]
        op = ConstraintOperator(inputs)
        with np.errstate(divide="ignore"):
            logits = np.log(inputs.q) - op.adjoint(lam_star)
        log_p = logits - logsumexp(logits)
        need = np.log(args.threshold) - log_p  # rise in logit needed, per cell

        taper = None if args.taper == "none" else args.taper
        laplace = StructuredGaussian.laplace(inputs, SimpleNamespace(
            lam=lam_star, alpha=alpha, taper=taper, variance_floor=floor))
        scores = {"structural": structural_score(inputs, floor)}
        sd_l = np.sqrt(np.maximum(normal_quadratic(inputs, laplace), 0))
        scores["laplace"] = norm.logsf(need / np.where(sd_l > 0, sd_l, np.inf))
        sd_v = np.sqrt(np.maximum(normal_quadratic(inputs, q), 0))
        shift = -op.adjoint(q.mean - lam_star)  # mean rise in each cell's logit under q
        scores["vb"] = norm.logsf((need - shift) / np.where(sd_v > 0, sd_v, np.inf))

        loads = (np.asarray(abs(sp.csr_matrix(inputs.X_T)).sum(1)).ravel()
                 + np.asarray(abs(sp.csr_matrix(inputs.X_B)).sum(1)).ravel()) > 0
        live = (inputs.q > 0) & loads[None, :]
        zones = inputs.zones.iloc[:, 0].astype(str).to_numpy()
        serials = inputs.units["SERIALNO"].astype(str).to_numpy()
        ev = events[(events.puma == str(puma)) & np.isclose(events.alpha, alpha)]
        counts = np.zeros((inputs.n_zones, inputs.n_units))
        zi = {z: k for k, z in enumerate(zones)}
        ui = {u: k for k, u in enumerate(serials)}
        for (b, u), c in ev.groupby(["block_group", "serialno"]).size().items():
            if b in zi and u in ui:
                counts[zi[b], ui[u]] += c
        label = counts[live] > 0
        weight = counts[live]
        lines.append(f"== {puma} alpha {alpha:g}: {int(live.sum()):,} cells, "
                     f"{int(label.sum())} event cells, {int(weight.sum())} events "
                     f"({time.perf_counter() - started:.0f}s)")
        flat_zone, flat_unit = np.nonzero(live)
        risk = np.exp(scores["vb"][live])
        order_risk = np.sort(risk)[::-1]
        cum = np.cumsum(order_risk) / order_risk.sum()
        stat = {
            "puma": puma, "alpha": alpha, "cells": int(live.sum()), "event_cells": int(label.sum()),
            "events_observed": int(weight.sum()),
            "events_expected_vb": float(args.draws * risk.sum()),
            "cells_for_50pct_risk": int(np.searchsorted(cum, 0.5) + 1),
            "cells_for_90pct_risk": int(np.searchsorted(cum, 0.9) + 1),
        }
        for tag, sd in (("laplace", sd_l), ("vb", sd_v)):
            values = sd[live]
            stat[f"sd_{tag}_event"] = float(np.median(values[label])) if label.any() else np.nan
            stat[f"sd_{tag}_other"] = float(np.median(values[~label]))
        ratio = sd_v[live] / np.where(sd_l[live] > 0, sd_l[live], np.nan)
        stat["sd_vb_over_laplace_event"] = float(np.nanmedian(ratio[label])) if label.any() else np.nan
        stat["sd_vb_over_laplace_other"] = float(np.nanmedian(ratio[~label]))
        stats.append(stat)
        lines.append(
            f"   marginal sd along the wall normal, median, event / other cells: "
            f"Laplace {stat['sd_laplace_event']:.3g} / {stat['sd_laplace_other']:.3g}, "
            f"VB {stat['sd_vb_event']:.3g} / {stat['sd_vb_other']:.3g}; "
            f"VB/Laplace {stat['sd_vb_over_laplace_event']:.3g} / {stat['sd_vb_over_laplace_other']:.3g}")
        lines.append(
            f"   VB score calibration: {stat['events_expected_vb']:,.0f} events expected in "
            f"{args.draws:,} draws, {stat['events_observed']:,} observed; predicted risk: 50% in "
            f"{stat['cells_for_50pct_risk']:,} cells, 90% in {stat['cells_for_90pct_risk']:,}")
        for score_name in SCORES:
            s = scores[score_name][live]
            order = np.argsort(-s, kind="stable")
            row = {"puma": puma, "alpha": alpha, "score": score_name, "auc": auc(s, label)}
            parts = [f"   {score_name:<10} AUC {row['auc']:.3f}"]
            for K in args.top:
                top = order[:K]
                row[f"cells_top{K}"] = float(label[top].sum() / max(label.sum(), 1))
                row[f"events_top{K}"] = float(weight[top].sum() / max(weight.sum(), 1))
                parts.append(f"top {K}: {100 * row[f'cells_top{K}']:.0f}% cells "
                             f"{100 * row[f'events_top{K}']:.0f}% events")
            rows.append(row)
            lines.append("  ".join(parts))
            for rank, j in enumerate(order[:max(args.top)], start=1):
                b, u = flat_zone[j], flat_unit[j]
                tops.append({"puma": puma, "alpha": alpha, "ranked_by": score_name, "rank": rank,
                             "block_group": zones[b], "serialno": serials[u],
                             **{f"score_{k}": float(scores[k][b, u]) for k in SCORES},
                             "log_p_map": float(log_p[b, u]), "events": int(counts[b, u])})
        logger.info("%s a=%g scored in %.0fs", puma, alpha, time.perf_counter() - started)

    pd.DataFrame(rows).to_csv(args.out / "prediction.csv", index=False)
    pd.DataFrame(stats).to_csv(args.out / "fit_stats.csv", index=False)
    pd.DataFrame(tops).to_csv(args.out / "top_cells.csv", index=False)
    text = ("Wall prediction: share of observed event cells (and events) captured by each score's "
            "top K cells\n\n" + "\n".join(lines) + "\n")
    (args.out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
