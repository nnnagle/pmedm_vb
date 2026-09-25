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

``<name>`` is ``<puma>_tract_a<alpha>``. ``--reference`` (an HMC reference run
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
  90% interval; the share of draws beyond ``Y +- 2 SE`` and ``+- 3 SE``; and,
  with ``--reference``, the median and mean difference in reference sds (sd
  floored at ``compare.SD_FLOOR``), the sd ratio and the 90% width ratio. Each
  is summarised as p1/p10/p50/p90/p99 over the subset's cells (``mean`` too).
- ``p``: per draw, over every (block group, record) cell --
  ``p/q > R`` for ``R`` in ``--ratio``; ``W`` over a share of its block
  group's total weight in that draw, for shares in ``--bg-share``; ``W`` over
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
  floor (see :func:`outcome_sets`).
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
    parser.add_argument("--variance-floor", default="zero")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--ratio", type=float, nargs="+", default=[10.0, 100.0])
    parser.add_argument("--bg-share", type=float, nargs="+", default=[0.05, 0.25])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def floor_spec(text: str):
    return None if text == "none" else "zero" if text == "zero" else float(text)


def fit_name(puma: str, alpha: float) -> str:
    return f"{puma}_tract_a{alpha:g}"


def thinned(kept: np.ndarray, draws: int) -> np.ndarray:
    """``(m, <= draws)`` evenly spaced draws from a ``(kept, chains, m)`` trace."""
    flat = kept.reshape(-1, kept.shape[-1])
    return flat[np.unique(np.linspace(0, len(flat) - 1, min(draws, len(flat))).astype(int))].T


# -- draws --------------------------------------------------------------------


def method_draws(args, inputs: PMEDMInputs, rng) -> tuple[dict, dict]:
    """``{"W": (zones, units)}`` or ``{"lam": (m, draws), "q": density or None}``,
    and the method's timing and settings."""
    name = fit_name(args.puma, args.alpha)
    timing: dict[str, float] = {}
    if args.method in ("ipf", "sinkhorn"):
        with np.load(args.run / f"{args.puma}_{args.method}.npz") as saved:
            timing["fit_seconds"] = float(saved["seconds"])
            timing["converged"] = float(saved["converged"])
            timing["n_sweeps"] = float(saved["n_sweeps"])
            timing["max_residual"] = float(saved["max_residual"])
            return {"W": saved["W"]}, timing
    if args.method == "map_laplace":
        from pmedm_vb.solvers.vb import StructuredGaussian

        with np.load(args.run / f"{name}.npz") as saved:
            lam = saved["lam"]
            timing["fit_seconds"] = float(saved["seconds"])
        start = time.perf_counter()
        result = SimpleNamespace(lam=lam, alpha=args.alpha, taper="tract",
                                 variance_floor=floor_spec(args.variance_floor))
        q = StructuredGaussian.laplace(inputs, result)
        timing["laplace_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        draws = q.sample(rng, args.draws)
        timing["draw_seconds"] = time.perf_counter() - start
        return {"lam": draws, "q": q}, timing
    if args.method.startswith("vb_"):
        path = args.run / f"{name}.npz"
        q, _, _ = load_vb(path)
        with np.load(path) as saved:
            timing["fit_seconds"] = float(saved["seconds"])
            if "map_seconds" in saved.files:
                timing["map_seconds"] = float(saved["map_seconds"])
                timing["vb_seconds"] = timing["fit_seconds"] - timing["map_seconds"]
            timing["converged"] = float(saved["converged"])
        start = time.perf_counter()
        draws = q.sample(rng, args.draws)
        timing["draw_seconds"] = time.perf_counter() - start
        return {"lam": draws, "q": q}, timing
    # HMC
    import arviz as az

    with np.load(args.run / f"{name}_trace.npz") as saved:
        kept = saved["lam"]
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
    return {"lam": thinned(kept, args.draws), "q": None}, timing


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
        sets.append(dict(subset=f"constrained_{level.replace(' ', '_')}", level=level,
                         X=sp.csc_matrix(inputs.X_T if level == "tract" else inputs.X_B),
                         Y=Y, se=np.sqrt(v), names=list(names),
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
    return sets


class Accumulator:
    """Per-draw outcomes and ``p`` statistics, one draw of ``W`` at a time."""

    def __init__(self, inputs, sets, ratios, shares, ref_max_log_share=None):
        self.inputs, self.sets = inputs, sets
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
            self.outcomes[i].append(
                (self.A_T @ block if s["level"] == "tract" else block).astype(np.float32))
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
        row["max_share_of_N"] = float(W.max() / self.inputs.N)
        if self.ref is not None:
            row["count_over_reference_max"] = int((log_p > self.ref).sum())
            row["max_log_excess_over_reference"] = float((log_p - self.ref).max())
        self.p_rows.append(row)
        self.max_log_share = log_p if self.max_log_share is None else np.maximum(
            self.max_log_share, log_p)

    def draws(self, i: int) -> np.ndarray:
        return np.stack(self.outcomes[i])  # (draws, areas, cells)


def log_p_of(inputs, op, lam):
    with np.errstate(divide="ignore"):
        logits = np.log(inputs.q) - op.adjoint(lam)
    return logits - logsumexp(logits)


def run_draws(inputs, source, sets, ratios, shares, ref_max=None) -> Accumulator:
    acc = Accumulator(inputs, sets, ratios, shares, ref_max)
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
    if s["Y"] is not None:
        Y, se = s["Y"], np.maximum(s["se"], 1e-9)
        out["z"] = ((summary["mean"] - Y) / se).ravel()
        out["abs_z"] = np.abs(out["z"])
    if s["Y"] is not None and draws.shape[0] > 1:  # intervals need more than one draw
        Y, se = s["Y"], np.maximum(s["se"], 1e-9)
        out["halfwidth_over_moe"] = ((summary["q95"] - summary["q05"]) / 2 / (Z90 * se)).ravel()
        out["covers_published"] = ((summary["q05"] <= Y) & (Y <= summary["q95"])).astype(float).ravel()
        dev = np.abs(draws - Y[None]) / se[None]
        out["share_beyond_2se"] = (dev > 2).mean(0).ravel()
        out["share_beyond_3se"] = (dev > 3).mean(0).ravel()
    if ref is not None:
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


def reference_summaries(args, inputs, sets) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """The reference's per-subset cell summaries, its per-cell max ``log p``, and
    its thinned ``lambda`` draws; cached beside the reference trace."""
    name = fit_name(args.puma, args.alpha)
    cache = args.reference / f"{name}_refsummary.npz"
    with np.load(args.reference / f"{name}_trace.npz") as saved:
        lam = thinned(saved["lam"], args.draws)
    if cache.exists():
        with np.load(cache) as saved:
            summaries = [{k: saved[f"{i}_{k}"] for k in ("mean", "sd", "q05", "q50", "q95")}
                         for i in range(len(sets))]
            return summaries, saved["max_log_share"], lam
    logger.info("summarising the reference %s (cached for next time)", args.reference)
    acc = run_draws(inputs, {"lam": lam}, sets, args.ratio, args.bg_share)
    summaries = [cell_summary(acc.draws(i)) for i in range(len(sets))]
    np.savez(cache, max_log_share=acc.max_log_share,
             **{f"{i}_{k}": v for i, s in enumerate(summaries) for k, v in s.items()})
    return summaries, acc.max_log_share, lam


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
TABLE_METRICS = ("z", "abs_z", "covers_published", "halfwidth_over_moe", "sd_over_se")


def by_table_rows(base: dict, s: dict, metrics: dict[str, np.ndarray]) -> list[dict]:
    """Per-table summaries of one subset's per-cell metrics, all and sampled cells."""
    k = len(s["names"])
    table = np.array([n.split(".")[0] for n in s["names"]])[np.tile(np.arange(k), s["Y"].shape[0])]
    sampled = s["sampled"].ravel()
    per_cell = dict(metrics)
    if "sd" in per_cell:
        per_cell["sd_over_se"] = per_cell["sd"] / np.maximum(s["se"], 1e-9).ravel()
    out = []
    for name in np.unique(table):
        for cells, mask in (("all", table == name), ("sampled", (table == name) & sampled)):
            row_base = dict(base, table=name, cells=cells)
            for metric in TABLE_METRICS:
                if metric in per_cell:
                    out += rows_for(row_base, s["subset"], metric, per_cell[metric][mask], CELL_STATS)
    return out


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record_run(args.out.parent, args)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    path = processed_dir() / "inputs" / args.area / args.puma
    inputs = PMEDMInputs.load(path)
    heldout = HeldOut.load(path) if HeldOut.exists(path) else None
    if heldout is None:
        logger.warning("no held-out tables under %s; scoring without them", path)
    sets = outcome_sets(inputs, heldout)

    source, timing = method_draws(args, inputs, rng)
    ref_summaries = ref_max = ref_lam = None
    if args.reference is not None and args.method != "hmc_ref":
        ref_summaries, ref_max, ref_lam = reference_summaries(args, inputs, sets)
    acc = run_draws(inputs, source, sets, args.ratio, args.bg_share, ref_max)

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

    p = pd.DataFrame(acc.p_rows)
    for column in p.columns:
        values = p[column].to_numpy(float)
        if column.startswith("count_"):
            rows.append(dict(base, subset="p", metric=column, stat="share_of_draws_any",
                             value=float((values > 0).mean())))
        rows += rows_for(base, "p", column, values, DRAW_STATS)
    for share in (0.01, 0.10):
        rows.append(dict(base, subset="p", metric=f"largest_cell_gt_{share:g}_of_N",
                         stat="share_of_draws", value=float((p["max_share_of_N"] > share).mean())))

    if "lam" in source:
        from pmedm_vb.solvers.vb import _DualTarget

        lam = source["lam"]
        if source.get("q") is not None:
            import torch

            torch.set_num_threads(max(1, torch.get_num_threads()))
            sigma = inputs.sigma(args.alpha, "tract", floor_spec(args.variance_floor))
            target = _DualTarget(inputs, sigma, "cpu")
            log_ratio = -inputs.n * batched_f(target, lam, 100) - source["q"].log_density(lam)
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
                q_white, _, _ = load_vb(args.whiten / f"{fit_name(args.puma, args.alpha)}.npz")
                coords = compare.coordinate_comparison(compare.whitened(q_white, ref_lam),
                                                       compare.whitened(q_white, lam))
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
