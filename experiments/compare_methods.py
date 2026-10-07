"""Score one method on one (PUMA, alpha) for the end-to-end comparison.

Every method is reduced to draws of the weights ``W`` -- one draw for the
raking baselines, 4,000 for the rest -- and every draw is scored the same way.
The result is appended to one long CSV, one row per
``(method, puma, alpha, subset, metric, stat)``, for tables and figures in R.

Methods and where their draws come from (``--run`` is the results directory):

- ``ipf``, ``sinkhorn``: ``<run>/<puma>_<method>.npz`` (``run_map.py rake``); a
  single ``W``, and no alpha (write ``--alpha`` as the cell being compared to).
- ``map_laplace``: ``<run>/<name>.npz`` from ``run_map.py solve``; draws from
  ``N(lambda*, (n H)^-1)`` (``StructuredGaussian.laplace``).
- ``vb_gaussian``, ``vb_skewed``, ``vb_sumdiff``: ``<run>/<name>.npz`` from
  ``run_map.py vb``; draws from the fitted family.
- ``hmc_short``, ``hmc_ref``: ``<run>/<name>_trace.npz`` from
  ``run_mcmc.py sample``; kept draws, thinned evenly to ``--draws``.

``<name>`` is ``<puma>_<taper>_a<alpha>`` (``--taper``, default ``tract``), with
``_h<level>`` appended under a hierarchy. ``--reference`` (an HMC reference run
for the same cell) adds the comparisons against it, and ``--whiten`` (the
skewed VB run whose Gaussian part whitened HMC) adds the whitened-coordinate
statistics. The reference's own summaries are cached in the reference
directory (``<name>_refsummary.npz``) and reused.

**What is scored** (``subset`` in the CSV):

- Outcomes ``A W X`` against the published tables: constrained block group
  and tract cells, held-out cells split into related and less related, and
  the race x income cross-tabulation (block group; no published value). Per
  cell: mean, sd, 5/50/95% quantiles; ``z = (mean - Y) / SE``; the 90%
  half-width over the published MOE (``1.645 SE``); whether ``Y`` lies in the
  90% interval; the share of draws beyond ``Y +- 2 SE`` and ``+- 3 SE``; the
  predictive ``pred_z = (mean - Y) / sqrt(SE^2 + sd^2)``, 90% predictive
  coverage ``pred_covers`` (normal) and ``post_var_over_se2 = (sd / SE)^2``; and,
  with ``--reference``, the median and mean difference in reference sds (sd
  floored at ``compare.SD_FLOOR``), the sd ratio and the 90% width ratio. Each
  is summarised as p1/p10/p50/p90/p99 over the subset's cells (``mean`` too).
- ``constrained_puma``: the PUMA total of each tract category against the
  tract estimates summed, with the SDR SE of the summed tract replicates --
  scored for every method, so how loosely each holds the PUMA totals shows.
- With ``--hierarchy tract|puma`` the MAP, VB and HMC results are the
  ``_h<level>`` fits of :mod:`pmedm_vb.assemble.hierarchy`: draws are in the
  solver's coordinates ``xi``, W comes from ``lambda_data(xi)``, and PSIS and
  the whitened comparison use ``xi``. The reference must be fitted at the same
  level. Raking ignores the option.
- ``p``: per draw, over every (block group, record) cell --
  ``p/q > R`` for ``R`` in ``--ratio``; ``W`` over a share of its block
  group's total weight in that draw, for shares in ``--bg-share``; a household
  record's ``W`` over the household weight of its block group in that draw (GQ
  units left out of both), for the same shares, over the block groups with more
  than ``--hh-min`` published occupied households (``B25003``); over the same
  block groups, each one's effective number of household records,
  ``n_eff = 1 / sum_i s_i^2`` with ``s_i`` a record's share of its households
  (Kish's effective sample size within the block group), as the smallest, p10
  and median over block groups and the count under each ``--neff-floor``; and
  the expected number of distinct records among the block group's ``H``
  published households drawn from those shares, ``D = sum_i 1 - (1 - s_i)^H``
  (at most ``H``), as the smallest ``D``, the smallest and median ``D / H`` and the
  count of block groups with ``D / H`` under each ``--distinct-floor``; ``W`` over
  the reference's largest value for that cell (``--reference``); the largest
  cell over 1% and 10% of ``N``; and the p99.9 and max of ``log(p/q)``. Each
  per-draw count or value is summarised over draws: the share of draws with at
  least one cell over, and p50/p90/p99/max.
- ``joint``: PSIS k-hat (Laplace and VB, which have a density), and with
  ``--whiten`` and ``--reference`` the whitened-coordinate sd ratio and the
  tract/block group pair statistics.
- With ``--by-table``, a second long CSV, one row per ``(method, puma,
  alpha, subset, table, cells, metric, stat)``: the outcome metrics against
  the published values (``z``, ``abs_z``, ``covers_published``,
  ``halfwidth_over_moe`` and ``sd_over_se``, the draws' sd over the published
  SE) per ACS table, over all its cells (``cells = all``) and over the
  ``sampled`` ones only -- a nonzero estimate whose variance is a sampling
  variance, not a zero cell's modelled one and not below the zero-count
  floor (see :func:`outcome_sets`). The same CSV also carries, per table, how
  the posterior mean fits the published table as a distribution over its
  categories (total variation, split into published-zero and nonzero cells,
  with the SDR replicates' own total variation as the sampling benchmark), the
  mean ``z^2`` over sampled cells (``q``) and the geometric-mean half-width
  over MOE, each weighted by published count and not; see
  :func:`distribution_rows`. Per table, too, the error decomposition of
  :func:`error_decomposition` (bias, scatter, the posterior's claimed variance
  and their ratio) and the across-area shrinkage tests of
  :func:`shrinkage_rows` (calibration slope, smoothing ratio, coverage by
  distance from the PUMA-wide share). Held-out PUMA totals (the tract estimates
  summed; no SE) are scored by total variation only.
- ``--score-area`` scores against another area's tables with the same units
  and zones: a fit on ``knox-min-2024-5yr`` (``drop_small_cells.py``) against
  the full ``knox-2024-5yr`` tables, so its dropped cells are scored too.
- ``timing``: seconds -- fit, MAP part, drawing -- and for HMC the sampler's
  seconds, its warmup part and seconds per 1,000 bulk ESS (minimum and median
  over coordinates of the kept ``lambda``).

The share-of-block-group rule measures a record against the draw's own block
group total, which is available for every method; the published block group
population is not in the inputs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

from pmedm_vb import compare
from pmedm_vb.assemble.heldout import HeldOut
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.data.variance import SDR_FACTOR
from pmedm_vb.progress import logger, record_run
from pmedm_vb.solvers.base import ConstraintOperator

from laplace_diagnostic import batched_f, load_vb

METHODS = ("ipf", "sinkhorn", "map_laplace", "vb_gaussian", "vb_skewed", "vb_sumdiff",
           "vb_gaussian_trunc", "vb_skewed_trunc", "vb_sumdiff_trunc",
           "hmc_short", "hmc_ref")
#: 90% interval half-width in standard errors.
Z90 = 1.645
CELL_STATS = (0.01, 0.10, 0.50, 0.90, 0.99)
DRAW_STATS = (0.50, 0.90, 0.99, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--run", type=Path, required=True, help="the method's results directory")
    parser.add_argument("--reference", type=Path, default=None,
                        help="HMC reference run_mcmc.py --out directory for this cell")
    parser.add_argument("--whiten", type=Path, default=None,
                        help="skewed VB results directory whose Gaussian part whitened HMC")
    parser.add_argument("--out", type=Path, required=True, help="long CSV to append to")
    parser.add_argument("--by-table", type=Path, default=None,
                        help="also append per-table summaries to this CSV (see module docstring)")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--score-area", default=None,
                        help="score against this area's tables instead of --area's: same units "
                             "and zones, e.g. the full tables for a fit with cells dropped")
    parser.add_argument("--variance-floor", default="zero")
    parser.add_argument("--taper", choices=["tract", "none"], default="tract",
                        help="the fits' Sigma taper: selects their result names and the Sigma "
                             "for the Laplace start and PSIS")
    parser.add_argument("--hierarchy", choices=["none", "tract", "puma", "nullspace"], default="none",
                        help="the fits' hierarchy level (pmedm_vb.assemble.hierarchy): selects "
                             "the _h<level> results and scores lambda_data; raking ignores it")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--ratio", type=float, nargs="+", default=[10.0, 100.0])
    parser.add_argument("--bg-share", type=float, nargs="+", default=[0.05, 0.25])
    parser.add_argument("--trunc-distinct", type=float, default=0.1,
                        help="vb_*_trunc: reject a draw with a block group (over --hh-min "
                             "published households) whose D / H is under this")
    parser.add_argument("--trunc-share", type=float, default=0.01,
                        help="vb_*_trunc: reject a draw with any cell over this share of N")
    parser.add_argument("--trunc-max-factor", type=int, default=20,
                        help="vb_*_trunc: give up after drawing this many times --draws")
    parser.add_argument("--hh-min", type=float, default=100,
                        help="the household-share counts use block groups with more than this "
                             "many published occupied households (B25003)")
    parser.add_argument("--neff-floor", type=float, nargs="+", default=[20, 50],
                        help="count, per draw, the block groups whose effective number of "
                             "household records is under each of these")
    parser.add_argument("--distinct-floor", type=float, nargs="+", default=[0.1, 0.2],
                        help="count, per draw, the block groups whose expected distinct records "
                             "per published household, D / H, is under each of these")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collapse", type=float, default=None,
                        help="collapse each area's zero cells into one and merge small positive "
                             "cells until each holds more than this many (pmedm_vb.assemble."
                             "collapse); the hierarchy level becomes <level>-c<threshold>")
    parser.add_argument("--rollup", default=None,
                        help="the structure-aware roll-up (pmedm_vb.assemble.rollup) with this "
                             "threshold, PERSONS or PERSONShHOUSEHOLDS (e.g. 15h10); the "
                             "hierarchy level becomes <level>-r<spec>")
    parser.add_argument("--share-cap", type=float, default=None,
                        help="the soft cap on each cell's share of the population, w / N "
                             "(pmedm_vb.assemble.sharecap), e.g. 0.005; the hierarchy level gains "
                             "+share<share>x<strength>")
    parser.add_argument("--cap-strength", type=float, default=None,
                        help="the cap's strength tau (required with --share-cap)")
    args = parser.parse_args()
    from pmedm_vb.assemble.hierarchy import level_name
    from pmedm_vb.assemble.sharecap import cap_level

    if getattr(args, "hierarchy", None) is not None:
        args.hierarchy = cap_level(level_name(args.hierarchy, args.collapse, args.rollup),
                                   args.share_cap, args.cap_strength)
    return args


def floor_spec(text: str):
    return None if text == "none" else "zero" if text == "zero" else float(text)


def fit_name(puma: str, alpha: float, hierarchy: str = "none", taper: str = "tract") -> str:
    suffix = "" if hierarchy == "none" else f"_h{hierarchy}"
    return f"{puma}_{taper}_a{alpha:g}{suffix}"


def taper_spec(text: str) -> str | None:
    return None if text == "none" else text


def thinned(kept: np.ndarray | None, draws: int) -> np.ndarray | None:
    """``(m, <= draws)`` evenly spaced draws from a ``(kept, chains, m)`` trace."""
    if kept is None:
        return None
    flat = kept.reshape(-1, kept.shape[-1])
    return flat[np.unique(np.linspace(0, len(flat) - 1, min(draws, len(flat))).astype(int))].T


def with_lambda(h, xi: np.ndarray, q) -> dict:
    """A draw source from draws in the solver's coordinates: ``lam`` in today's
    layout for W, and ``xi`` for densities and whitening."""
    return {"lam": xi if h.is_trivial else h.lambda_data(xi), "xi": xi, "q": q}


def check_level(path: Path, saved, level: str) -> None:
    found = str(saved["hierarchy"]) if "hierarchy" in saved.files else "none"
    if found != level:
        raise SystemExit(f"{path} was fitted with hierarchy={found}, not {level}")


# -- draws --------------------------------------------------------------------


def truncated_draws(args, inputs: PMEDMInputs, rng, h, q) -> tuple[np.ndarray, dict]:
    """``vb_*_trunc``: draws from ``q`` with those the HMC reference never makes
    turned away -- any cell over ``--trunc-share`` of N, or any block group over
    ``--hh-min`` published households whose expected distinct household records
    per household, ``D / H``, is under ``--trunc-distinct``. Drawn in batches until
    ``--draws`` are kept or ``--trunc-max-factor`` times that many are spent.
    Returns the kept draws (solver coordinates) and the acceptance counts."""
    op = ConstraintOperator(inputs)
    tenure = [k for k, c in enumerate(inputs.bg_constraints) if c.startswith("B25003.")]
    hh = ~inputs.units["is_group_quarters"].to_numpy(bool)
    published = inputs.Y_B[:, tenure].sum(axis=1)
    zones = published > args.hh_min
    H = published[zones]
    log_share = np.log(args.trunc_share)
    kept, tried, by_share, by_distinct = [], 0, 0, 0
    batch = max(64, args.draws // 4)
    while sum(k.shape[1] for k in kept) < args.draws and tried < args.trunc_max_factor * args.draws:
        xi = q.sample(rng, batch)
        lam = xi if h.is_trivial else h.lambda_data(xi)
        keep = np.ones(batch, bool)
        for d in range(batch):
            log_p = log_p_of(inputs, op, lam[:, d])
            if log_p.max() > log_share:
                keep[d] = False
                by_share += 1
                continue
            if zones.any():
                w = np.exp(log_p[np.ix_(zones, hh)])
                s = w / w.sum(axis=1, keepdims=True)
                if np.any(expected_distinct(s, H) / H < args.trunc_distinct):
                    keep[d] = False
                    by_distinct += 1
        kept.append(xi[:, keep])
        tried += batch
    draws = np.concatenate(kept, axis=1)[:, :args.draws]
    if draws.shape[1] == 0:
        raise SystemExit(f"truncated VB kept none of {tried} draws: every draw has a cell over "
                         f"{args.trunc_share:g} of N or a block group under D/H "
                         f"{args.trunc_distinct:g}")
    if draws.shape[1] < args.draws:
        logger.warning("truncated VB kept %d of %d draws after %d tries", draws.shape[1],
                       args.draws, tried)
    counts = {"trunc_tried": float(tried), "trunc_kept": float(draws.shape[1]),
              "trunc_acceptance": float(sum(k.shape[1] for k in kept) / tried),
              "trunc_rejected_share": float(by_share), "trunc_rejected_distinct": float(by_distinct)}
    return draws, counts


def method_draws(args, inputs: PMEDMInputs, rng, h) -> tuple[dict, dict]:
    """``{"W": (zones, units)}`` or ``{"lam": (m, draws), "xi": ..., "q": density
    or None}``, and the method's timing and settings. ``lam`` is always in
    today's layout; ``xi`` is the solver's coordinates (the same without a
    hierarchy)."""
    name = fit_name(args.puma, args.alpha, args.hierarchy, args.taper)
    timing: dict[str, float] = {}
    if args.method in ("ipf", "sinkhorn"):
        from pmedm_vb.rake import rake_name

        with np.load(args.run / f"{rake_name(args.puma, args.method, args.rollup)}.npz") as saved:
            timing["fit_seconds"] = float(saved["seconds"])
            timing["converged"] = float(saved["converged"])
            timing["n_sweeps"] = float(saved["n_sweeps"])
            timing["max_residual"] = float(saved["max_residual"])
            return {"W": saved["W"]}, timing
    if args.method == "map_laplace":
        from pmedm_vb.solvers.vb import StructuredGaussian

        with np.load(args.run / f"{name}.npz") as saved:
            check_level(args.run / f"{name}.npz", saved, args.hierarchy)
            lam = saved["lam"]
            xi = saved["xi"] if "xi" in saved.files else None
            timing["fit_seconds"] = float(saved["seconds"])
        start = time.perf_counter()
        result = SimpleNamespace(lam=lam, alpha=args.alpha, taper=taper_spec(args.taper),
                                 variance_floor=floor_spec(args.variance_floor),
                                 hierarchy=args.hierarchy, xi=xi)
        q = StructuredGaussian.laplace(inputs, result)
        timing["laplace_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        draws = q.sample(rng, args.draws)
        timing["draw_seconds"] = time.perf_counter() - start
        return with_lambda(h, draws, q), timing
    if args.method.startswith("vb_"):
        path = args.run / f"{name}.npz"
        q, _, _ = load_vb(path)
        with np.load(path) as saved:
            check_level(path, saved, args.hierarchy)
            timing["fit_seconds"] = float(saved["seconds"])
            if "map_seconds" in saved.files:
                timing["map_seconds"] = float(saved["map_seconds"])
                timing["vb_seconds"] = timing["fit_seconds"] - timing["map_seconds"]
            timing["converged"] = float(saved["converged"])
        start = time.perf_counter()
        if args.method.endswith("_trunc"):
            draws, counts = truncated_draws(args, inputs, rng, h, q)
            timing.update(counts)
        else:
            draws = q.sample(rng, args.draws)
        timing["draw_seconds"] = time.perf_counter() - start
        return with_lambda(h, draws, q), timing
    # HMC
    import arviz as az

    with np.load(args.run / f"{name}_trace.npz") as saved:
        kept = saved["lam"]
        kept_xi = saved["xi"] if "xi" in saved.files else None
        for key in ("seconds", "warmup_seconds"):
            if key in saved.files:
                timing[f"hmc_{key}"] = float(saved[key])
        timing["iterations"] = float(saved["log_pi"].shape[0])
        timing["divergent"] = float(saved["divergent"][int(saved["warmup"]):].sum())
    ess = az.ess(az.convert_to_dataset({"lam": np.transpose(kept, (1, 0, 2))}),
                 method="bulk")["lam"].values
    timing["ess_bulk_min"], timing["ess_bulk_median"] = float(ess.min()), float(np.median(ess))
    if "hmc_seconds" in timing:
        timing["seconds_per_1000_ess_min"] = 1000 * timing["hmc_seconds"] / ess.min()
        timing["seconds_per_1000_ess_median"] = 1000 * timing["hmc_seconds"] / np.median(ess)
    lam = thinned(kept, args.draws)
    return {"lam": lam, "xi": lam if kept_xi is None else thinned(kept_xi, args.draws),
            "q": None}, timing


# -- outcomes -----------------------------------------------------------------


def outcome_sets(inputs: PMEDMInputs, heldout: HeldOut | None) -> list[dict]:
    """Every scored cell set: loadings, level, published values and SEs.

    Sets with published values also carry ``names`` (``"{table}.{category}"``
    per column) and ``sampled``, the cells whose published variance is a
    sampling variance: a nonzero estimate, a variance at or above its area's
    zero-count floor (:meth:`PMEDMInputs.zero_cell_variances`, so the floor
    would not bind), and, for constrained cells, a nonzero replicate spread.
    Held-out tables carry no replicates, so their test is the first two.
    """
    split = inputs.Y_T.size
    floor = inputs.zero_cell_variances()
    replicate = SDR_FACTOR * np.square(inputs.sigma_l).sum(axis=1)
    sets = []
    for level, rows, Y, names in (
        ("tract", slice(0, split), inputs.Y_T, inputs.tract_constraints),
        ("block group", slice(split, None), inputs.Y_B, inputs.bg_constraints),
    ):
        v, fl, s = (a[rows].reshape(Y.shape, order="F") for a in (inputs.sigma_v, floor, replicate))
        # replicate deviations Y_r - Y as (areas, cells, replicates)
        rep = inputs.sigma_l[rows].reshape(Y.shape[1], Y.shape[0], -1).transpose(1, 0, 2)
        sets.append(dict(subset=f"constrained_{level.replace(' ', '_')}", level=level,
                         X=sp.csc_matrix(inputs.X_T if level == "tract" else inputs.X_B),
                         Y=Y, se=np.sqrt(v), names=list(names), rep=rep,
                         sampled=(Y > 0) & (s > 0) & (v >= fl)))
    if heldout is not None:
        for level, rows, n_areas in (("tract", slice(0, split), inputs.Y_T.shape[0]),
                                     ("block group", slice(split, None), inputs.Y_B.shape[0])):
            area_floor = floor[rows][:n_areas]  # the floor is per area: any column's copy
            relation = np.array(heldout.relation(level))
            for kind in ("related", "less_related"):
                cols = np.flatnonzero(relation == kind)
                Y, v = heldout.Y[level][:, cols], heldout.v[level][:, cols]
                sets.append(dict(
                    subset=f"heldout_{kind}_{level.replace(' ', '_')}", level=level,
                    X=sp.csc_matrix(heldout.X[level])[:, cols], Y=Y, se=np.sqrt(v),
                    names=[heldout.names[level][c] for c in cols],
                    sampled=(Y > 0) & (v > 0) & (v >= area_floor[:, None]),
                ))
    matrix, meta = compare.outcome_matrix(inputs)
    cross = np.flatnonzero(meta["kind"].to_numpy() == "crosstab")
    sets.append(dict(subset="crosstab_block_group", level="block group",
                     X=sp.csc_matrix(matrix)[:, cross], Y=None, se=None))
    # PUMA totals of each tract category: the tract estimates and replicates summed.
    n_tracts, c_t = inputs.Y_T.shape
    summed = np.stack([inputs.sigma_l[k * n_tracts + np.arange(n_tracts)].sum(axis=0)
                       for k in range(c_t)])
    s_p = SDR_FACTOR * np.square(summed).sum(axis=1)
    # A category with no replicate spread in any tract (a published zero everywhere)
    # takes its tract variances summed -- the modelled zero-cell ones -- as the
    # model's PUMA rows do, rather than an SE of 0 (whose z is unbounded).
    tract_v = inputs.sigma_v[:split].reshape(inputs.Y_T.shape, order="F").sum(axis=0)
    v_p = np.where(s_p > 0, s_p, tract_v)
    Y_p = inputs.Y_T.sum(axis=0)[None, :]
    sets.append(dict(subset="constrained_puma", level="puma", X=sp.csc_matrix(inputs.X_T),
                     Y=Y_p, se=np.sqrt(v_p)[None, :], names=list(inputs.tract_constraints),
                     rep=summed[None], sampled=(Y_p > 0) & (s_p[None, :] > 0)))
    # Held-out PUMA totals: the tract estimates summed. No replicates, so no SE:
    # scored by the distribution measures of ``--by-table`` only.
    if heldout is not None:
        relation = np.array(heldout.relation("tract"))
        for kind in ("related", "less_related"):
            cols = np.flatnonzero(relation == kind)
            Y = heldout.Y["tract"][:, cols].sum(axis=0)[None, :]
            sets.append(dict(subset=f"heldout_{kind}_puma", level="puma",
                             X=sp.csc_matrix(heldout.X["tract"])[:, cols], Y=Y, se=None,
                             names=[heldout.names["tract"][c] for c in cols],
                             sampled=np.zeros(Y.shape, bool)))
    return sets


class Accumulator:
    """Per-draw outcomes and ``p`` statistics, one draw of ``W`` at a time."""

    def __init__(self, inputs, sets, ratios, shares, ref_max_log_share=None, hh_min=100,
                 neff_floors=(20, 50), distinct_floors=(0.1, 0.2)):
        self.inputs, self.sets = inputs, sets
        self.neff_floors, self.distinct_floors = neff_floors, distinct_floors
        # Household records, and the block groups with more than hh_min published
        # occupied households, for each household's share of its block group's
        # households. None without B25003 among the block group constraints.
        tenure = [k for k, c in enumerate(inputs.bg_constraints) if c.startswith("B25003.")]
        if tenure and "is_group_quarters" in inputs.units:
            self.hh_units = ~inputs.units["is_group_quarters"].to_numpy(bool)
            published = inputs.Y_B[:, tenure].sum(axis=1)
            self.hh_zones = published > hh_min
            self.hh_H = published[self.hh_zones]
        else:
            self.hh_units = self.hh_zones = self.hh_H = None
        self.A_T = sp.csr_matrix(inputs.A_T)
        self.X_all = sp.hstack([s["X"] for s in sets]).tocsc()
        self.bounds = np.cumsum([0] + [s["X"].shape[1] for s in sets])
        self.outcomes = [[] for _ in sets]
        with np.errstate(divide="ignore"):
            self.log_q = np.log(inputs.q)
        self.log_ratio_bounds = np.log(ratios)
        self.ratios, self.shares = ratios, shares
        self.ref = ref_max_log_share
        self.p_rows = []
        self.max_log_share = None  # elementwise max over draws, for a reference

    def add(self, log_p: np.ndarray, W: np.ndarray | None = None) -> None:
        """``log_p``: ``(zones, units)`` log of ``p``, summing to one. ``W``
        defaults to ``N p``; a raked ``W`` is passed as it is, since its total
        need not be ``N``."""
        if W is None:
            W = self.inputs.N * np.exp(log_p)
        by_zone = np.asarray((self.X_all.T @ W.T).T)
        for i, s in enumerate(self.sets):
            block = by_zone[:, self.bounds[i]:self.bounds[i + 1]]
            if s["level"] == "tract":
                block = self.A_T @ block
            elif s["level"] == "puma":
                block = block.sum(axis=0, keepdims=True)
            self.outcomes[i].append(block.astype(np.float32))
        log_pq = log_p - self.log_q
        finite = np.isfinite(log_pq)
        row = {}
        for R, bound in zip(self.ratios, self.log_ratio_bounds):
            row[f"count_p_over_q_gt_{R:g}"] = int((log_pq[finite] > bound).sum())
        row["max_log_p_over_q"] = float(log_pq[finite].max())
        row["q999_log_p_over_q"] = float(np.quantile(log_pq[finite], 0.999))
        share_bg = W / W.sum(axis=1, keepdims=True)
        for share in self.shares:
            row[f"count_bg_share_gt_{share:g}"] = int((share_bg > share).sum())
        row["max_bg_share"] = float(share_bg.max())
        if self.hh_units is not None and self.hh_zones.any():
            W_hh = W[np.ix_(self.hh_zones, self.hh_units)]
            share_hh = W_hh / W_hh.sum(axis=1, keepdims=True)
            for share in self.shares:
                row[f"count_hh_share_gt_{share:g}"] = int((share_hh > share).sum())
            row["max_hh_share"] = float(share_hh.max())
            n_eff = 1.0 / np.square(share_hh).sum(axis=1)
            row["min_bg_n_eff"] = float(n_eff.min())
            row["p10_bg_n_eff"] = float(np.quantile(n_eff, 0.10))
            row["median_bg_n_eff"] = float(np.median(n_eff))
            for floor in self.neff_floors:
                row[f"count_bg_n_eff_lt_{floor:g}"] = int((n_eff < floor).sum())
            distinct = expected_distinct(share_hh, self.hh_H)
            per_hh = distinct / self.hh_H
            row["min_bg_distinct"] = float(distinct.min())
            row["min_bg_distinct_per_hh"] = float(per_hh.min())
            row["median_bg_distinct_per_hh"] = float(np.median(per_hh))
            for floor in self.distinct_floors:
                row[f"count_bg_distinct_per_hh_lt_{floor:g}"] = int((per_hh < floor).sum())
        row["max_share_of_N"] = float(W.max() / self.inputs.N)
        if self.ref is not None:
            row["count_over_reference_max"] = int((log_p > self.ref).sum())
            row["max_log_excess_over_reference"] = float((log_p - self.ref).max())
        self.p_rows.append(row)
        self.max_log_share = log_p if self.max_log_share is None else np.maximum(
            self.max_log_share, log_p)

    def draws(self, i: int) -> np.ndarray:
        return np.stack(self.outcomes[i])  # (draws, areas, cells)


def expected_distinct(shares: np.ndarray, H: np.ndarray) -> np.ndarray:
    """``D_j = sum_i 1 - (1 - s_ij)^H_j``: the expected number of distinct records
    among ``H_j`` households drawn with replacement from block group ``j``'s
    shares (rows of ``shares``). At most ``H_j`` and the number of records."""
    with np.errstate(divide="ignore"):
        return -np.expm1(H[:, None] * np.log1p(-np.minimum(shares, 1.0))).sum(axis=1)


def log_p_of(inputs, op, lam):
    with np.errstate(divide="ignore"):
        logits = np.log(inputs.q) - op.adjoint(lam)
    return logits - logsumexp(logits)


def run_draws(inputs, source, sets, ratios, shares, ref_max=None, hh_min=100,
              neff_floors=(20, 50), distinct_floors=(0.1, 0.2)) -> Accumulator:
    acc = Accumulator(inputs, sets, ratios, shares, ref_max, hh_min, neff_floors, distinct_floors)
    if "W" in source:
        with np.errstate(divide="ignore"):
            acc.add(np.log(source["W"] / source["W"].sum()), W=source["W"])
        return acc
    op = ConstraintOperator(inputs)
    for d in range(source["lam"].shape[1]):
        acc.add(log_p_of(inputs, op, source["lam"][:, d]))
    return acc


def cell_summary(draws: np.ndarray) -> dict[str, np.ndarray]:
    q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    sd = draws.std(0, ddof=1) if draws.shape[0] > 1 else np.full(draws.shape[1:], np.nan)
    return {"mean": draws.mean(0), "sd": sd, "q05": q05, "q50": q50, "q95": q95}


def outcome_metrics(draws: np.ndarray, s: dict, ref: dict | None) -> dict[str, np.ndarray]:
    """Per-cell metrics for one subset, flattened over (area, cell)."""
    summary = cell_summary(draws)
    out = {k: v.ravel() for k, v in summary.items()}
    if s["Y"] is not None and s["se"] is not None:  # a published value with an SE
        Y, se = s["Y"], np.maximum(s["se"], 1e-9)
        out["z"] = ((summary["mean"] - Y) / se).ravel()
        out["abs_z"] = np.abs(out["z"])
    if s["Y"] is not None and s["se"] is not None and draws.shape[0] > 1:  # intervals need >1 draw
        Y, se = s["Y"], np.maximum(s["se"], 1e-9)
        out["halfwidth_over_moe"] = ((summary["q95"] - summary["q05"]) / 2 / (Z90 * se)).ravel()
        out["covers_published"] = ((summary["q05"] <= Y) & (Y <= summary["q95"])).astype(float).ravel()
        dev = np.abs(draws - Y[None]) / se[None]
        out["share_beyond_2se"] = (dev > 2).mean(0).ravel()
        out["share_beyond_3se"] = (dev > 3).mean(0).ravel()
        # The published value is the true count plus survey error, so with the
        # posterior's own uncertainty it should sit within sqrt(SE^2 + sd^2) of
        # the mean: a predictive z and (normal) 90% predictive coverage.
        pred_sd = np.sqrt(np.square(se) + np.square(summary["sd"]))
        out["pred_z"] = ((summary["mean"] - Y) / pred_sd).ravel()
        out["pred_covers"] = (np.abs(summary["mean"] - Y) <= Z90 * pred_sd).astype(float).ravel()
        out["post_var_over_se2"] = np.square(summary["sd"] / se).ravel()
    return out if ref is None else _ref_metrics(out, summary, draws, ref)


def _ref_metrics(out: dict, summary: dict, draws: np.ndarray, ref: dict) -> dict:
    """Add the comparisons against the reference's cell summaries to ``out``."""
    scale = np.maximum(ref["sd"], compare.SD_FLOOR)
    out["median_diff_ref_sd"] = np.abs((summary["q50"] - ref["q50"]) / scale).ravel()
    if draws.shape[0] == 1:
        return out
    width_ref = np.maximum(ref["q95"] - ref["q05"], compare.SD_FLOOR)
    out["mean_diff_ref_sd"] = np.abs((summary["mean"] - ref["mean"]) / scale).ravel()
    out["sd_ratio_ref"] = (summary["sd"] / scale).ravel()
    out["width_ratio_ref"] = ((summary["q95"] - summary["q05"]) / width_ref).ravel()
    return out


# -- reference ----------------------------------------------------------------


def reference_summaries(args, inputs, sets):
    """The reference's per-subset cell summaries, its per-cell max ``log p``, and
    its thinned ``lambda`` and ``xi`` draws; cached beside the reference trace
    (and recomputed if the cache predates a scored subset)."""
    name = fit_name(args.puma, args.alpha, args.hierarchy, args.taper)
    tag = "" if args.score_area in (None, args.area) else f"_{args.score_area}"
    cache = args.reference / f"{name}{tag}_refsummary.npz"
    with np.load(args.reference / f"{name}_trace.npz") as saved:
        lam = thinned(saved["lam"], args.draws)
        xi = thinned(saved["xi"], args.draws) if "xi" in saved.files else lam
    if cache.exists():
        with np.load(cache) as saved:
            if f"{len(sets) - 1}_mean" in saved.files:
                summaries = [{k: saved[f"{i}_{k}"] for k in ("mean", "sd", "q05", "q50", "q95")}
                             for i in range(len(sets))]
                return summaries, saved["max_log_share"], lam, xi
    logger.info("summarising the reference %s (cached for next time)", args.reference)
    acc = run_draws(inputs, {"lam": lam}, sets, args.ratio, args.bg_share, hh_min=args.hh_min,
                    neff_floors=args.neff_floor, distinct_floors=args.distinct_floor)
    summaries = [cell_summary(acc.draws(i)) for i in range(len(sets))]
    np.savez(cache, max_log_share=acc.max_log_share,
             **{f"{i}_{k}": v for i, s in enumerate(summaries) for k, v in s.items()})
    return summaries, acc.max_log_share, lam, xi


# -- output -------------------------------------------------------------------


def rows_for(base: dict, subset: str, metric: str, values: np.ndarray, stats) -> list[dict]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return []
    out = [dict(base, subset=subset, metric=metric, stat="mean", value=float(values.mean())),
           dict(base, subset=subset, metric=metric, stat="n", value=float(values.size))]
    out += [dict(base, subset=subset, metric=metric, stat=f"p{100 * s:g}",
                 value=float(np.quantile(values, s))) for s in stats]
    return out


#: Metrics against the published values that ``--by-table`` breaks down.
TABLE_METRICS = ("z", "abs_z", "covers_published", "halfwidth_over_moe", "sd_over_se",
                 "pred_z", "pred_covers", "post_var_over_se2")


def by_table_rows(base: dict, s: dict, metrics: dict[str, np.ndarray]) -> list[dict]:
    """Per-table summaries of one subset's per-cell metrics, all and sampled cells."""
    k = len(s["names"])
    table = np.array([n.split(".")[0] for n in s["names"]])[np.tile(np.arange(k), s["Y"].shape[0])]
    sampled = s["sampled"].ravel()
    per_cell = dict(metrics)
    if "sd" in per_cell and s["se"] is not None:
        per_cell["sd_over_se"] = per_cell["sd"] / np.maximum(s["se"], 1e-9).ravel()
    out = []
    for name in np.unique(table):
        for cells, mask in (("all", table == name), ("sampled", (table == name) & sampled)):
            row_base = dict(base, table=name, cells=cells)
            for metric in TABLE_METRICS:
                if metric in per_cell:
                    out += rows_for(row_base, s["subset"], metric, per_cell[metric][mask], CELL_STATS)
    return out


#: Per-area statistics of the distribution measures: the mean weighted by the
#: area's published table total, then the plain mean and quantiles.
AREA_STATS = (0.50, 0.90, 1.00)


def distribution_rows(base: dict, s: dict, mean: np.ndarray,
                      metrics: dict[str, np.ndarray]) -> list[dict]:
    """Per-table fit of the posterior mean to the published table.

    Per area, the table's published cells ``Y`` and fitted means ``m`` are each
    normalised to a distribution over the table's categories, ``p`` and
    ``p_hat``, and compared by total variation: ``tvd = 1/2 sum |p_hat - p|``,
    split into ``tvd_zero`` (the fitted mass in published-zero cells) and
    ``tvd_pos`` (the mismatch among the nonzero ones). With replicates, the
    same is computed for each replicate table ``Y + 2 (Y_r - Y)`` (the factor
    2 gives each deviation one sampling variance under SDR's ``4/80``; clipped
    at 0) against ``p``: ``tvd_rep`` is its mean over replicates, and
    ``tvd_within_rep`` whether the fit's tvd is at most the replicates' 90th
    percentile. Areas with a published total of 0 are skipped; so are
    one-category tables, whose tvd is 0 by construction.

    Over the table's sampled cells, pooled across areas: ``q`` is the mean of
    ``z^2``, weighted by the published count (``stat = weighted``) and not
    (``unweighted``) -- 1 if the fit were off by sampling error alone -- and
    ``hw_over_moe_gmean`` the geometric mean of the draws' 90% half-width over
    the published MOE, weighted and not.
    """
    Y = s["Y"]
    table = np.array([n.split(".")[0] for n in s["names"]])
    rep = s.get("rep")
    out = []
    for name in np.unique(table):
        cols = np.flatnonzero(table == name)
        row_base = dict(base, table=name)
        Yt, Mt = Y[:, cols], mean[:, cols]
        total, fitted = Yt.sum(axis=1), Mt.sum(axis=1)
        keep = (total > 0) & (fitted > 0)
        if cols.size > 1 and keep.any():
            p, ph = Yt[keep] / total[keep, None], Mt[keep] / fitted[keep, None]
            zero = p == 0
            per_area = {"tvd_zero": 0.5 * np.where(zero, ph, 0).sum(axis=1),
                        "tvd_pos": 0.5 * np.where(zero, 0, np.abs(ph - p)).sum(axis=1)}
            per_area["tvd"] = per_area["tvd_zero"] + per_area["tvd_pos"]
            if rep is not None:
                Yr = np.maximum(Yt[keep][:, :, None] + 2 * rep[keep][:, cols], 0)
                pr = Yr / np.maximum(Yr.sum(axis=1, keepdims=True), 1e-12)
                tvd_r = 0.5 * np.abs(pr - p[:, :, None]).sum(axis=1)  # (areas, replicates)
                per_area["tvd_rep"] = tvd_r.mean(axis=1)
                per_area["tvd_within_rep"] = (
                    per_area["tvd"] <= np.quantile(tvd_r, 0.9, axis=1)).astype(float)
            weight = total[keep]
            for metric, values in per_area.items():
                rb = dict(row_base, cells="all", subset=s["subset"], metric=metric)
                out.append(dict(rb, stat="wmean", value=float(np.average(values, weights=weight))))
                out.append(dict(rb, stat="mean", value=float(values.mean())))
                out.append(dict(rb, stat="n", value=float(values.size)))
                out += [dict(rb, stat=f"p{100 * q:g}", value=float(np.quantile(values, q)))
                        for q in AREA_STATS]
        if s["se"] is None:
            continue
        mask = s["sampled"][:, cols]
        if not mask.any():
            continue
        rb = dict(row_base, cells="sampled", subset=s["subset"])
        w = Yt[mask]
        z2 = np.square((Mt[mask] - Yt[mask]) / np.maximum(s["se"][:, cols][mask], 1e-9))
        out += [dict(rb, metric="q", stat="weighted", value=float(np.average(z2, weights=w))),
                dict(rb, metric="q", stat="unweighted", value=float(z2.mean())),
                dict(rb, metric="q", stat="n", value=float(z2.size))]
        out += error_decomposition(rb, Yt, Mt, s["se"][:, cols], mask, metrics, Y.shape, cols)
        out += shrinkage_rows(row_base, s["subset"], Yt, Mt, s["se"][:, cols], metrics, Y.shape, cols)
        if "halfwidth_over_moe" in metrics:
            ratio = metrics["halfwidth_over_moe"].reshape(Y.shape)[:, cols][mask]
            ok = ratio > 0
            if ok.any():
                log_r = np.log(ratio[ok])
                out += [dict(rb, metric="hw_over_moe_gmean", stat="weighted",
                             value=float(np.exp(np.average(log_r, weights=w[ok])))),
                        dict(rb, metric="hw_over_moe_gmean", stat="unweighted",
                             value=float(np.exp(log_r.mean())))]
    return out


def error_decomposition(rb: dict, Yt, Mt, se, mask, metrics, shape, cols) -> list[dict]:
    """Over a table's sampled cells, in SE units (``z = (mean - Y) / SE``):
    ``E z^2 = 1 + ((mean - truth) / SE)^2`` when the survey error is independent
    of the fit (exactly so for held-out cells), so the model's own squared error
    is ``z2 - 1``; it splits into a systematic part ``bias2 = (mean z)^2`` and
    the rest, ``scatter``. ``claimed`` is the posterior's own variance, the mean
    of ``(sd / SE)^2`` (and its median), and ``calib_ratio = (z2 - 1) / claimed``:
    about 1 if the posterior is as uncertain as its error, above 1 if it is too
    sure. ``pred_q`` is the mean predictive ``z^2``, 1 when calibrated."""
    z = (Mt[mask] - Yt[mask]) / np.maximum(se[mask], 1e-9)
    if z.size == 0:
        return []
    z2, zbar = float(np.mean(z * z)), float(z.mean())
    rows = [("z_mean", zbar), ("z2", z2), ("bias2", zbar * zbar), ("scatter", z2 - 1 - zbar * zbar)]
    if "post_var_over_se2" in metrics:
        claimed = metrics["post_var_over_se2"].reshape(shape)[:, cols][mask]
        pred = metrics["pred_z"].reshape(shape)[:, cols][mask]
        rows += [("claimed", float(claimed.mean())), ("claimed_p50", float(np.median(claimed))),
                 ("calib_ratio", (z2 - 1) / float(claimed.mean()) if claimed.mean() > 0 else np.nan),
                 ("calib_ratio_p50", (z2 - 1) / float(np.median(claimed))
                  if np.median(claimed) > 0 else np.nan),
                 ("pred_q", float(np.mean(pred * pred)))]
    return [dict(rb, metric="decomp", stat=name, value=float(v)) for name, v in rows] + [
        dict(rb, metric="decomp", stat="n", value=float(z.size))]


#: Fewest areas a category needs for its across-area slope and variance ratio.
MIN_AREAS = 5


def shrinkage_rows(row_base: dict, subset: str, Yt, Mt, se, metrics, shape, cols) -> list[dict]:
    """Does the fit smooth a table across areas? Per category, over the areas
    with a nonzero published (and fitted) table total, in shares of that total:

    - ``calib_slope``: the slope of the published share on the fitted share.
      The survey error is on the published side only (and independent of the
      fit for held-out cells), so a correct fit gives 1; a fit pulled toward
      the PUMA-wide share gives a slope above 1.
    - ``smooth_ratio``: the variance of the fitted shares over the true
      between-area variance, estimated as the published shares' variance less
      their mean sampling variance. Below 1: smoother than the truth.

    Summarised over the table's categories (median, mean, and the mean weighted
    by each category's published total). ``covers_by_distance`` and
    ``pred_covers_by_distance``: coverage of the published value by the 90%
    posterior and predictive intervals, for cells grouped by how far their
    published share lies from the PUMA-wide share, in SEs (``lt1``, ``1to2``,
    ``ge2``)."""
    total, fitted = Yt.sum(axis=1), Mt.sum(axis=1)
    keep = (total > 0) & (fitted > 0)
    if keep.sum() < MIN_AREAS or Yt.shape[1] < 2:
        return []
    y = Yt[keep] / total[keep, None]
    m = Mt[keep] / fitted[keep, None]
    s2 = np.square(se[keep] / total[keep, None])
    slopes, ratios, weights = [], [], []
    for k in range(Yt.shape[1]):
        vm = m[:, k].var(ddof=1)
        if vm <= 0:
            continue
        slopes.append(np.cov(y[:, k], m[:, k])[0, 1] / vm)
        true_var = y[:, k].var(ddof=1) - s2[:, k].mean()
        ratios.append(vm / true_var if true_var > 0 else np.nan)
        weights.append(Yt[keep][:, k].sum())
    out = []
    rb = dict(row_base, cells="all", subset=subset)
    for metric, values in (("calib_slope", slopes), ("smooth_ratio", ratios)):
        v, w = np.asarray(values, float), np.asarray(weights, float)
        ok = np.isfinite(v)
        if not ok.any():
            continue
        out += [dict(rb, metric=metric, stat="p50", value=float(np.median(v[ok]))),
                dict(rb, metric=metric, stat="mean", value=float(v[ok].mean())),
                dict(rb, metric=metric, stat="wmean",
                     value=float(np.average(v[ok], weights=w[ok])) if w[ok].sum() > 0 else np.nan),
                dict(rb, metric=metric, stat="n", value=float(ok.sum()))]
    puma_share = Yt[keep].sum(axis=0) / total[keep].sum()
    distance = np.abs(y - puma_share[None]) / np.sqrt(np.maximum(s2, 1e-18))
    bins = {"lt1": distance < 1, "1to2": (distance >= 1) & (distance < 2), "ge2": distance >= 2}
    for metric in ("covers_published", "pred_covers"):
        if metric not in metrics:
            continue
        values = metrics[metric].reshape(shape)[:, cols][keep]
        for name, sel in bins.items():
            if sel.any():
                out += [dict(rb, metric=f"{metric}_by_distance", stat=name,
                             value=float(values[sel].mean())),
                        dict(rb, metric=f"{metric}_by_distance", stat=f"n_{name}",
                             value=float(sel.sum()))]
    return out


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record_run(args.out.parent, args)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    path = processed_dir() / "inputs" / args.area / args.puma
    inputs = PMEDMInputs.load(path)
    score_path = processed_dir() / "inputs" / (args.score_area or args.area) / args.puma
    scored = inputs if score_path == path else PMEDMInputs.load(score_path)
    if scored is not inputs and not (scored.units.equals(inputs.units)
                                     and scored.zones.equals(inputs.zones)):
        raise SystemExit(f"{score_path} and {path} have different units or zones")
    heldout = HeldOut.load(score_path) if HeldOut.exists(score_path) else None
    if heldout is None:
        logger.warning("no held-out tables under %s; scoring without them", score_path)
    sets = outcome_sets(scored, heldout)

    from pmedm_vb.assemble.hierarchy import Hierarchy

    hierarchy = Hierarchy.build(inputs, args.hierarchy, alpha=args.alpha,
                                taper=taper_spec(args.taper),
                                variance_floor=floor_spec(args.variance_floor))
    source, timing = method_draws(args, inputs, rng, hierarchy)
    ref_summaries = ref_max = ref_lam = ref_xi = None
    if args.reference is not None and args.method != "hmc_ref":
        ref_summaries, ref_max, ref_lam, ref_xi = reference_summaries(args, inputs, sets)
    acc = run_draws(inputs, source, sets, args.ratio, args.bg_share, ref_max, args.hh_min,
                    args.neff_floor, args.distinct_floor)

    base = dict(method=args.method, puma=args.puma, alpha=args.alpha,
                n_draws=len(acc.p_rows))
    rows = [dict(base, subset="timing", metric=k, stat="value", value=float(v))
            for k, v in timing.items()]

    table_rows = []
    for i, s in enumerate(sets):
        metrics = outcome_metrics(acc.draws(i), s, ref_summaries[i] if ref_summaries else None)
        for metric, values in metrics.items():
            rows += rows_for(base, s["subset"], metric, values, CELL_STATS)
        if args.by_table is not None and s["Y"] is not None:
            table_rows += by_table_rows(base, s, metrics)
            table_rows += distribution_rows(base, s, metrics["mean"].reshape(s["Y"].shape),
                                            metrics)

    p = pd.DataFrame(acc.p_rows)
    for column in p.columns:
        values = p[column].to_numpy(float)
        if column.startswith("count_"):
            rows.append(dict(base, subset="p", metric=column, stat="share_of_draws_any",
                             value=float((values > 0).mean())))
        # A diversity measure's bad side is its low end: its lower quantiles too.
        low = (column.endswith(("_n_eff", "_distinct", "_distinct_per_hh"))
               and not column.startswith("count_"))
        rows += rows_for(base, "p", column, values, (0.0, 0.01, 0.10) + DRAW_STATS if low
                         else DRAW_STATS)
    for share in (0.01, 0.10):
        rows.append(dict(base, subset="p", metric=f"largest_cell_gt_{share:g}_of_N",
                         stat="share_of_draws", value=float((p["max_share_of_N"] > share).mean())))

    if "lam" in source:
        from pmedm_vb.solvers.vb import _DualTarget

        lam = source["lam"]
        if source.get("q") is not None:
            import torch

            torch.set_num_threads(max(1, torch.get_num_threads()))
            sigma = hierarchy.sigma(inputs, args.alpha, taper_spec(args.taper),
                                    floor_spec(args.variance_floor))
            target = _DualTarget(inputs, sigma, "cpu", hierarchy)
            xi = source["xi"]
            log_ratio = -inputs.n * batched_f(target, xi, 100) - source["q"].log_density(xi)
            rows.append(dict(base, subset="joint", metric="psis_khat", stat="value",
                             value=compare.psis_khat(log_ratio)))
        if ref_lam is not None:
            pairs = compare.pair_comparison(inputs, ref_lam, lam)
            for column in ("sd_sum_ratio", "sd_diff_ratio"):
                rows += rows_for(base, "joint", f"pair_{column}", pairs[column], CELL_STATS)
            for tag in ("ref", "alt"):
                for column in ("corr", "skew_sum", "skew_diff"):
                    rows.append(dict(base, subset="joint", metric=f"pair_{column}_{tag}",
                                     stat="p50", value=float(pairs[f"{column}_{tag}"].median())))
            if args.whiten is not None:
                q_white, _, _ = load_vb(
                    args.whiten / f"{fit_name(args.puma, args.alpha, args.hierarchy, args.taper)}.npz")
                coords = compare.coordinate_comparison(compare.whitened(q_white, ref_xi),
                                                       compare.whitened(q_white, source["xi"]))
                rows += rows_for(base, "joint", "whitened_sd_ratio", coords["sd_ratio"], CELL_STATS)

    rows.append(dict(base, subset="timing", metric="scoring_seconds", stat="value",
                     value=time.perf_counter() - started))
    frame = pd.DataFrame(rows)[["method", "puma", "alpha", "n_draws", "subset", "metric",
                                "stat", "value"]]
    frame.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
    if args.by_table is not None:
        tables = pd.DataFrame(table_rows)[["method", "puma", "alpha", "n_draws", "subset", "table",
                                          "cells", "metric", "stat", "value"]]
        tables.to_csv(args.by_table, mode="a", header=not args.by_table.exists(), index=False)
    logger.info("%s %s a=%g: %d rows appended to %s", args.method, args.puma, args.alpha,
                len(frame), args.out)


if __name__ == "__main__":
    main()
