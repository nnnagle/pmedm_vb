"""What holds the wide directions: the error covariance, the data, or only the walls?

Along a direction ``d`` of the sampling coordinates, the curvature of HMC's
potential ``n f`` is the sum of three parts (``f`` as in
:class:`~pmedm_vb.solvers.vb._DualTarget`):

- **Sigma**: ``n c zeta(d)' Sigma zeta(d)``, ``c = n / N^2``, ``zeta`` the
  hierarchy's extended multipliers -- the measurement-error term;
- **data**: ``n d' (grad^2 log Z) d``, the variance across (zone, unit) cells of
  the tilt change ``X lambda(d)`` under the cell shares ``p`` -- averaged over
  ``--draws`` posterior draws, since it changes with position;
- **ridge**: the hierarchy's ridge on its dropped directions, if any.

For ``d`` a one-posterior-sd move along a whitened coordinate (as in
``wide_directions.py``), a Gaussian direction has total curvature about 1;
far below 1 means the direction is held by neither term in the bulk, only by
whatever lies in its tails. Also, per direction:

- **Sigma ratio** ``zeta' Sigma zeta / zeta' diag(Sigma) zeta``: about 1 where
  ``Sigma`` acts like its diagonal; about ``alpha`` where the direction lies
  outside the replicates' span, so that only the ``alpha`` share of each
  variance is left -- the signature of a near-singular ``Sigma``;
- **wall distance**: from the posterior mean, the smallest move along ``d``
  (in posterior sds, either way) at which some excluded cell (weight under
  ``--excluded`` people) reaches one person, holding ``log Z`` fixed; and that
  cell (unit, zone).

Directions: whitened coordinates grouped by posterior sd (widest over
``--wide-sd``, moderate, bulk; up to ``--per-group`` each), plus any listed
with ``--coordinates``, e.g. ``exp02_nullspace:6629,6720 exp01_baseline:12895``
(a bare list applies to every experiment). Whitening is the Gaussian-only
fit's by default (``--whitening``). Finally, each multiplier's own posterior
sd (data layout): how many a prior of scale 1, 3 or 10 would bind.

**Tilts.** A cell's tilt is its weight over its prior allocation,
``W_ij / (N q_ij) = p_ij / q_ij``; a unit's is the same summed over zones,
``sum_j p_ij / sum_j q_ij``. For each bound ``K`` in ``--bounds``, over
``--tilt-draws`` posterior draws: the cells and units tilted above ``K`` or
below ``1/K`` in more than half the draws and in any draw, and the people
(``N p``) those cells hold, averaged over the draws -- what a prior bounding
the tilt in both directions would act on. Units are the PUMA's own (a
statewide support's added rows have a tiny prior by construction). Also the
excluded cells' tilt at the posterior mean, and for each direction's wall
cell its prior allocation ``N q`` and its tilt now and at one person.

Writes ``<out>/flat_directions.csv`` (medians per group), ``..._detail.csv``
(every direction) and ``flat_directions.txt``. Needs the assembled inputs::

    $PY experiments/flat_directions.py --root $PAPER --alpha 0.01 --pumas 4701501 \\
        --experiments exp01_baseline exp02_nullspace \\
        --coordinates exp02_nullspace:6629,6720,1199,794 exp01_baseline:12895,12916 \\
        --out $PAPER/oracle_scale/flat_directions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.progress import logger, record_run

from laplace_diagnostic import constraint_table
from run_experiment import fit_name
from wide_directions import Tilt, load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=["exp01_baseline", "exp02_nullspace"])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--pumas", nargs="+", default=None)
    parser.add_argument("--kind", default="ref", help="the HMC run read, hmc_<kind>")
    parser.add_argument("--whitening", choices=["skewed", "gaussian"], default="gaussian")
    parser.add_argument("--wide-sd", type=float, default=3.0)
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--coordinates", nargs="+", default=[],
                        help="EXPERIMENT:j,j,... or j ... (every experiment)")
    parser.add_argument("--draws", type=int, default=8, help="posterior draws for the data curvature")
    parser.add_argument("--excluded", type=float, default=1e-6,
                        help="a cell under this many people counts as excluded")
    parser.add_argument("--bounds", type=float, nargs="+", default=[1e3, 3e3, 1e4],
                        help="tilt bounds K, applied as K and 1/K")
    parser.add_argument("--tilt-draws", type=int, default=200,
                        help="posterior draws for the tilt counts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def listed_for(items: list[str], experiment: str) -> list[int]:
    out = []
    for item in items:
        if ":" in item:
            name, coords = item.split(":", 1)
            if name == experiment:
                out += [int(j) for j in coords.split(",") if j]
        else:
            out += [int(j) for j in item.split(",") if j]
    return out


def log_shares(inputs, tilt: Tilt, lam: np.ndarray) -> tuple[np.ndarray, float]:
    """``log p`` over (zone, unit) cells (``-inf`` off the support) and ``log Z``."""
    with np.errstate(divide="ignore"):
        logits = np.log(inputs.q) - tilt(lam)
    logits = np.where(tilt.support, logits, -np.inf)
    top = logits[np.isfinite(logits)].max()
    log_z = top + np.log(np.exp(logits - top).sum())
    return logits - log_z, float(log_z)


def own_units(inputs) -> np.ndarray:
    """Units holding at least one of the PUMA's own records (all of them
    without a statewide support)."""
    members = inputs.unit_members
    n_units = inputs.q.shape[1]
    if members is None:
        return np.ones(n_units, bool)
    own = np.zeros(n_units, bool)
    own[members.loc[members["in_puma"].to_numpy(bool), "row"].to_numpy(int)] = True
    return own


def tilt_counts(args, inputs, tilt: Tilt, lam_flat: np.ndarray, log_q: np.ndarray,
                own: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Per bound ``K``: cells and units tilted beyond ``K`` either way, over posterior draws."""
    picks = rng.choice(lam_flat.shape[0], min(args.tilt_draws, lam_flat.shape[0]), replace=False)
    log_k = np.log(np.asarray(args.bounds))
    cell_up = np.zeros((log_k.size,) + log_q.shape, np.int32)
    cell_down = np.zeros_like(cell_up)
    unit_up = np.zeros((log_k.size, own.size), np.int32)
    unit_down = np.zeros_like(unit_up)
    people_up, people_down = np.zeros(log_k.size), np.zeros(log_k.size)
    log_q_unit = np.log(inputs.q.sum(axis=0))
    for i in picks:
        log_p, _ = log_shares(inputs, tilt, lam_flat[i])
        p = np.exp(log_p)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(tilt.support, log_p - log_q, 0.0)            # log tilt, 0 off the support
            t_unit = np.log(p.sum(axis=0)) - log_q_unit
        for b, lk in enumerate(log_k):
            up, down = t > lk, t < -lk
            cell_up[b] += up
            cell_down[b] += down
            unit_up[b] += t_unit > lk
            unit_down[b] += t_unit < -lk
            people_up[b] += inputs.N * p[up].sum()
            people_down[b] += inputs.N * p[down].sum()
    half = picks.size / 2
    rows = []
    for b, k in enumerate(args.bounds):
        rows.append({
            "bound": k, "draws": int(picks.size),
            "cells_up_half": int((cell_up[b] > half).sum()), "cells_up_any": int((cell_up[b] > 0).sum()),
            "cells_down_half": int((cell_down[b] > half).sum()),
            "cells_down_any": int((cell_down[b] > 0).sum()),
            "people_up": people_up[b] / picks.size, "people_down": people_down[b] / picks.size,
            "units_up_half": int((unit_up[b][own] > half).sum()),
            "units_up_any": int((unit_up[b][own] > 0).sum()),
            "units_down_half": int((unit_down[b][own] > half).sum()),
            "units_down_any": int((unit_down[b][own] > 0).sum()),
            "own_units": int(own.sum()),
        })
    return rows


def one_fit(args, spec: dict, exp: Path, puma: str, listed: list[int],
            rng: np.random.Generator) -> tuple[list[dict], list[dict], dict, list[dict]]:
    inputs, sigma, q, _, hierarchy = load(spec, exp, puma, args.alpha, args.whitening)
    name = fit_name(spec, puma, args.alpha)
    with np.load(exp / puma / f"hmc_{args.kind}" / f"{name}_trace.npz") as saved:
        xi_draws = saved["xi"] if "xi" in saved.files else saved["lam"]
        lam_draws = saved["lam"]
    kept, chains, m = xi_draws.shape
    xi_flat = xi_draws.reshape(-1, m)
    sd = q.whiten((xi_flat - q.mean).T).T.std(axis=0)
    lam_flat = lam_draws.reshape(kept * chains, -1)
    lam_sd = lam_flat.std(axis=0)
    lam_mean = lam_flat.mean(axis=0)

    tilt = Tilt(inputs)
    table = constraint_table(inputs)
    zones = inputs.block_groups.iloc[:, 0].to_numpy()
    unit_ids = inputs.units.iloc[:, 0].to_numpy()
    n, N = inputs.n, inputs.N
    c = n / N**2
    diag_sigma = sigma.diagonal()

    picks = rng.choice(lam_flat.shape[0], min(args.draws, lam_flat.shape[0]), replace=False)
    shares = [np.exp(log_shares(inputs, tilt, lam_flat[i])[0]) for i in picks]
    mean_log_p, _ = log_shares(inputs, tilt, lam_mean)
    target = np.log(1.0 / N)                                         # one person, as a log share
    excluded = tilt.support & (mean_log_p < np.log(args.excluded / N))
    with np.errstate(divide="ignore"):
        log_q = np.where(tilt.support, np.log(inputs.q), -np.inf)
    ln10 = np.log(10.0)

    groups = {"widest": np.flatnonzero(sd > args.wide_sd),
              "moderate": np.flatnonzero((sd >= 1.5) & (sd <= args.wide_sd)),
              "bulk": np.flatnonzero((sd > 0) & (sd < 1.5))}
    groups = {g: (idx if idx.size <= args.per_group
                  else np.sort(rng.choice(idx, args.per_group, replace=False)))
              for g, idx in groups.items()}
    if listed:
        groups["listed"] = np.asarray([j for j in listed if 0 <= j < m], int)

    details = []
    for group, idx in groups.items():
        if not idx.size:
            continue
        E = np.zeros((m, idx.size))
        E[idx, np.arange(idx.size)] = sd[idx]
        moves = q.gaussian_part(E)                                   # one posterior sd each
        for i, j in enumerate(idx):
            d = moves[:, i]
            if hierarchy.is_trivial:
                zeta, lam_d, ridge = d, d, 0.0
            else:
                zeta, lam_d = hierarchy.zeta(d), hierarchy.lambda_data(d)
                ridge_gradient = getattr(hierarchy, "ridge_gradient", None)
                ridge = float(n * d @ ridge_gradient(d)) if ridge_gradient else 0.0
            s_zeta = sigma.matvec(zeta)
            sigma_curv = float(n * c * zeta @ s_zeta)
            ratio = float(zeta @ s_zeta / (zeta @ (diag_sigma * zeta)))
            delta = tilt(lam_d)                                       # tilt change per unit move
            data_curv = float(np.mean([n * ((p * delta * delta).sum() - (p * delta).sum() ** 2)
                                       for p in shares]))
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.where(excluded & (delta != 0), (target - mean_log_p) / np.abs(delta), np.inf)
            k = int(np.argmin(dist))
            z, u = np.unravel_index(k, dist.shape)
            top = int(np.argmax(np.abs(lam_d)))
            details.append({
                "group": group, "coordinate": int(j), "sd": float(sd[j]),
                "sigma_curv": sigma_curv, "data_curv": data_curv, "ridge_curv": ridge,
                "total_curv": sigma_curv + data_curv + ridge, "sigma_ratio": ratio,
                "wall_distance_sd": float(dist[z, u]), "wall_unit": unit_ids[u], "wall_zone": zones[z],
                "wall_d_eta": float(delta[z, u]),
                "wall_prior_people": float(N * inputs.q[z, u]),
                "wall_log10_tilt_now": float((mean_log_p[z, u] - log_q[z, u]) / ln10),
                "wall_log10_tilt_at_one": float((target - log_q[z, u]) / ln10),
                "top_constraint": f"{table['level'].iat[top]} {table['geoid'].iat[top]} "
                                  f"{table['constraint'].iat[top]}",
                "top_move": float(lam_d[top]),
            })

    frame = pd.DataFrame(details)
    rows = []
    for group in groups:
        f = frame[frame["group"] == group]
        if f.empty:
            continue
        rows.append({"group": group, "n": len(f), "sd": f["sd"].median(),
                     "sigma_curv": f["sigma_curv"].median(), "data_curv": f["data_curv"].median(),
                     "ridge_curv": f["ridge_curv"].median(), "total_curv": f["total_curv"].median(),
                     "sigma_ratio": f["sigma_ratio"].median(),
                     "sigma_ratio_p10": f["sigma_ratio"].quantile(0.1),
                     "wall_distance_sd": f["wall_distance_sd"].median(),
                     "wall_distance_sd_p10": f["wall_distance_sd"].quantile(0.1)})
    excluded_tilt = (mean_log_p[excluded] - log_q[excluded]) / ln10
    excluded_q = (np.quantile(excluded_tilt, [0.0, 0.1, 0.5, 0.9]) if excluded_tilt.size
                  else [np.nan] * 4)
    multipliers = {"multipliers": int(lam_sd.size),
                   **{f"lam_sd_over_{s:g}": int((lam_sd > s).sum()) for s in (1, 3, 10)},
                   "lam_sd_p50": float(np.median(lam_sd)), "lam_sd_p99": float(np.quantile(lam_sd, 0.99)),
                   "lam_sd_max": float(lam_sd.max()), "alpha": args.alpha,
                   "excluded_cells": int(excluded.sum()), "support_cells": int(tilt.support.sum()),
                   **{f"excluded_log10_tilt_{name}": float(v)
                      for name, v in zip(("min", "p10", "p50", "p90"), excluded_q)}}
    logger.info("tilt counts over %d draws", min(args.tilt_draws, lam_flat.shape[0]))
    tilts = tilt_counts(args, inputs, tilt, lam_flat, log_q, own_units(inputs), rng)
    return rows, details, multipliers, tilts


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    rng = np.random.default_rng(args.seed)

    summary, detail_rows, extra, tilt_rows = [], [], [], []
    for experiment in args.experiments:
        exp = args.root / experiment
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            name = fit_name(spec, puma, args.alpha)
            if not (exp / puma / f"hmc_{args.kind}" / f"{name}_trace.npz").exists():
                logger.info("%s %s: no %s trace; skipped", experiment, puma, args.kind)
                continue
            logger.info("%s %s", experiment, puma)
            rows, details, multipliers, tilts = one_fit(args, spec, exp, puma,
                                                 listed_for(args.coordinates, experiment), rng)
            ident = {"experiment": experiment, "puma": puma}
            summary += [{**ident, **r} for r in rows]
            detail_rows += [{**ident, **d} for d in details]
            extra.append({**ident, **multipliers})
            tilt_rows += [{**ident, **t} for t in tilts]
            pd.DataFrame(summary).to_csv(args.out / "flat_directions.csv", index=False)
            pd.DataFrame(detail_rows).to_csv(args.out / "flat_directions_detail.csv", index=False)
            pd.DataFrame(tilt_rows).to_csv(args.out / "flat_directions_tilts.csv", index=False)
    if not summary:
        raise SystemExit("no traces found")

    fmt = lambda v: f"{v:,.3g}"  # noqa: E731
    detail = pd.DataFrame(detail_rows)
    listed = detail[detail["group"] == "listed"]
    lines = [f"What holds the wide directions, alpha {args.alpha:g}, {args.whitening} whitening, "
             f"from {args.root}",
             "Curvatures of n f along a one-posterior-sd move (about 1 in total for a Gaussian "
             "direction): Sigma term, data (log Z) term, ridge. sigma_ratio: zeta' Sigma zeta over "
             "zeta' diag(Sigma) zeta (about alpha = outside the replicates' span). wall_distance_sd: "
             "posterior sds from the mean until an excluded cell reaches one person.", "",
             "Medians by group", pd.DataFrame(summary).to_string(index=False, float_format=fmt), ""]
    if not listed.empty:
        lines += ["Listed coordinates", listed.drop(columns=["group"]).to_string(index=False, float_format=fmt), ""]
    lines += ["Multipliers (data layout): how many have a posterior sd over 1, 3, 10; excluded "
              "cells' log10 tilt (weight over prior allocation) at the posterior mean",
              pd.DataFrame(extra).to_string(index=False, float_format=fmt), "",
              "Tilts beyond K (up) or 1/K (down): cells and own units, in more than half the draws "
              "and in any draw; people = N p held by those cells, averaged over draws",
              pd.DataFrame(tilt_rows).to_string(index=False, float_format=fmt)]
    text = "\n".join(lines)
    (args.out / "flat_directions.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
