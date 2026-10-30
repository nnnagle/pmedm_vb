"""The paper's tables for experiments 1-3, every statistic pooled over the PUMAs.

Reads, for each experiment, the scoring ``<root>/<experiment>/scores/<tag>``:
``detail/*/<method>_cells.parquet`` (every scored cell),
``detail/*/<method>_draws.parquet`` (every draw) and ``all_scores.csv``
(timings and PSIS k-hat), and writes five tables, plus the tract versions of
Tables 2, 3 and 4, and two tables by ACS table (6 and 7), each as LaTeX (a bare booktabs ``tabular`` per experiment,
``table<key>_<experiment>.tex``, for a document to caption and label), CSV
(long: one row per experiment, alpha, method and statistic, unrounded, with
``n`` the cells, draws or PUMAs behind it) and Markdown (all in ``tables.md``,
with the notes); ``settings.csv``, the settings behind the numbers the paper
quotes (each experiment's recorded spec, flattened to ``<experiment>.<path>``
keys, its scoring's draws, and these tables' options); and ``draws_sample.csv``, up
to 1,000 random draws of each (experiment, method, alpha, PUMA) with their
block-group D/H and n_eff and largest cell, for the figures of the draws::

    sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_python.sbatch \\
        paper_tables.py --root $PAPER --tag paper1 --out $PAPER/tables

Rows are methods, one panel per alpha, columns each statistic for each
experiment. Pooling:

- Over cells (Tables 2-4): every leaf cell of every PUMA together (a table's
  published categories; its totals and subtotals, sums of them, left out), so
  quantiles are quantiles of the pooled cells. z and coverage use the *sampled* cells only
  (a nonzero published estimate whose variance is a sampling variance, not a
  zero cell's modelled one). Coverage, in Tables 2 and 3, is whether the
  posterior 90% interval (the draws' 5th to 95th percentiles) contains the
  published value; it has no nominal level, since the interval is for the true
  count and the published value also carries sampling error.
- By ACS table (Tables 6 and 7): the same pooled block-group leaf cells, one row
  per ACS table, columns the methods at each alpha; each value is followed (in
  parentheses) by the same statistic over the table's published zeros, whose
  SE is the modelled zero-cell one.
- Over draws (Table 5): the draws of every PUMA together; each draw is one
  PUMA's, so "any block group" means any in that PUMA.
- Not poolable: PSIS k-hat, one value per fit, as the median [min, max] over
  PUMAs; and cost, the mean over PUMAs per PUMA (the county total, the sum, is
  in the CSV).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pmedm_vb.assemble.allcells import SUBTOTAL_MARK  # noqa: E402
from provenance import TABLES_SCRIPTS, write_stamp  # noqa: E402

#: (folder, label)
EXPERIMENTS = [("exp01_baseline", "Baseline"), ("exp02_nullspace", "Null space"),
               ("exp03_rollup", "Roll-up")]

#: (method, label), in the tables' order.
METHODS = [("ipf", "IPF"), ("ipf_multinomial", "IPF, multinomial"),
           ("map_laplace", "MAP + Laplace"),
           ("vb_gaussian", "VB Gaussian"), ("vb_skewed", "VB skewed"),
           ("vb_gaussian_trunc", "VB Gaussian, truncated"),
           ("vb_skewed_trunc", "VB skewed, truncated"),
           ("hmc_short", "HMC short"), ("hmc_ref", "HMC reference")]

ALPHAS = (0.01, 0.1, 1.0)

#: Methods without an alpha (raking): one fit per PUMA, scored in every alpha
#: cell. They are shown once, in their own panel, except in Table 4, where they
#: are compared with each alpha's reference.
NO_ALPHA = ("ipf", "ipf_multinomial")

#: Table 5's statistics that are shares of draws: for a method without an alpha,
#: one draw per PUMA, shown as a count of draws.
SHARE_KEYS = ("walls", "dh_lt", "neff_lt", "hhz_gt")

#: Cell columns read from the detail files.
CELL_COLUMNS = ["method", "puma", "alpha", "subset", "sampled", "published", "abs_z",
                "covers_published",
                "halfwidth_over_moe", "median_diff_ref_sd", "width_ratio_ref"]
CELL_SUBSETS = ("constrained_block_group", "constrained_tract",
                "heldout_related_block_group", "heldout_less_related_block_group",
                "heldout_related_tract", "heldout_less_related_tract")
DRAW_COLUMNS = ["method", "puma", "alpha", "max_share_of_N", "min_bg_distinct_per_hh",
                "median_bg_distinct_per_hh", "min_bg_n_eff", "median_bg_n_eff",
                "max_abs_hh_z", "count_hh_abs_z_gt_5"]
#: Draws kept per (method, alpha, PUMA) in draws_sample.csv, for the figures
#: of the draws' distributions (experiments/figures/fig_draws.R).
SAMPLE_DRAWS = 1000
SAMPLE_COLUMNS = ["method", "puma", "alpha", "min_bg_distinct_per_hh", "median_bg_distinct_per_hh",
                  "min_bg_n_eff", "median_bg_n_eff", "max_share_of_N"]


def sample_draws(draws: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """At most SAMPLE_DRAWS draws of each (method, alpha, PUMA), at random."""
    have = [c for c in SAMPLE_COLUMNS if c in draws]
    shuffled = draws[have].sample(frac=1.0, random_state=seed)
    return (shuffled.groupby(["method", "alpha", "puma"]).head(SAMPLE_DRAWS)
            .sort_values(["method", "alpha", "puma"]).reset_index(drop=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--tag", default="paper1", help="the scoring tag of every experiment")
    parser.add_argument("--experiments", nargs="+", default=[e for e, _ in EXPERIMENTS],
                        help="experiment folders, in column order")
    parser.add_argument("--wall", type=float, default=0.01, help="a wall: a cell over this share of N")
    parser.add_argument("--dh-floor", type=float, default=0.04,
                        help="Table 5's D/H floor: below the smallest any HMC draw of the "
                             "paper_v3 runs reached (0.054), so the posterior never fails it")
    parser.add_argument("--neff-floor", type=float, default=10)
    parser.add_argument("--hh-z", type=float, default=5)
    parser.add_argument("--pumas", nargs="+", default=None,
                        help="pool only these PUMAs (default: every PUMA scored)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="report only these alphas (default: every alpha scored)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


# -- reading ------------------------------------------------------------------


def read_detail(folder: Path, kind: str, columns: list[str], subsets=None) -> pd.DataFrame:
    """Every ``<method>_<kind>.parquet`` under ``folder``, with the columns it has.

    Cells are leaf cells only: an all-cell table's totals and subtotals (names
    ending in ``":"``) are sums of its leaves, so counting them as cells would
    count the same people again and weight the large aggregates. Cells also
    get ``acs_table``, the ACS table of their name ``"{table}.{category}"``."""
    frames = []
    for path in sorted(folder.glob(f"*/*_{kind}.parquet")):
        names = pq.read_schema(path).names
        have = [c for c in columns if c in names]
        leaves_only = kind == "cells" and "cell" in names
        frame = pd.read_parquet(path, columns=have + (["cell"] if leaves_only else []))
        if leaves_only:
            cell = frame.pop("cell").astype("category")
            names = cell.cat.categories
            codes = cell.cat.codes.to_numpy()
            acs = np.asarray(names.str.split(".", n=1).str[0])
            frame["acs_table"] = pd.Categorical(acs[codes])
            frame = frame[~np.asarray(names.str.endswith(SUBTOTAL_MARK))[codes]]
        if subsets is not None:
            frame = frame[frame.subset.isin(subsets)]
        for column in frame.columns:
            if frame[column].dtype == np.float64 and column != "alpha":
                frame[column] = frame[column].astype(np.float32)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no *_{kind}.parquet under {folder}")
    frame = pd.concat(frames, ignore_index=True)
    frame["alpha"] = frame.alpha.astype(float).round(12)
    if "acs_table" in frame:  # concat makes differing categories plain strings
        frame["acs_table"] = frame.acs_table.astype("category")
    return frame


def quantile(p: float):
    def fn(x: pd.Series) -> float:
        x = x.dropna()
        return float(np.quantile(x, p)) if len(x) else np.nan
    fn.__name__ = f"p{100 * p:g}"
    return fn


def mean(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean()) if len(x) else np.nan


# -- statistics, one long frame per experiment -----------------------------------
#
# Each statistic is (table, key, label, value) per (alpha, method).


#: A central 90% interval's z and the interval score's level.
Z90, LEVEL = 1.645, 0.10


def interval_score(cells: pd.DataFrame) -> pd.Series:
    """Each cell's interval score, in published SEs: the 90% predictive
    interval's width, plus 2 / 0.10 times how far the published value falls
    outside it (Gneiting & Raftery 2007, *JASA* 102). Lower is better; it
    rewards an interval that is narrow and contains the value, so it judges
    the centre and the spread together.

    The predictive interval is for the published value: the posterior's own
    90% interval (half-width h, its 5th to 95th percentiles) widened by the
    published value's sampling error, the two half-widths added in quadrature,
    sqrt(h^2 + (Z90 SE)^2), about the posterior mean. That is exact for normal
    draws; built from the draws' quantiles rather than their sd, it is not set
    by a few extreme draws. From the stored ``halfwidth_over_moe`` = h /
    (Z90 SE) and ``abs_z`` = |mean - Y| / SE, the half-width in SEs is
    Z90 sqrt(1 + halfwidth_over_moe^2).
    """
    half = Z90 * np.sqrt(1.0 + np.square(cells["halfwidth_over_moe"]))
    return 2 * half + 2 / LEVEL * np.maximum(cells["abs_z"] - half, 0.0)


def oracle_interval_score() -> float:
    """The expected score of a method that knows every true count exactly
    (no spread, so the interval is the true count +- Z90 SE): the published
    value is then off by a standard normal number of SEs."""
    from math import erf, exp, pi, sqrt

    density = exp(-Z90 * Z90 / 2) / sqrt(2 * pi)
    upper_tail = 0.5 * (1 - erf(Z90 / sqrt(2)))
    return 2 * Z90 + 2 / LEVEL * 2 * (density - Z90 * upper_tail)


def cell_statistics(cells: pd.DataFrame) -> list[dict]:
    rows = []
    if {"abs_z", "halfwidth_over_moe"} <= set(cells.columns):
        cells = cells.assign(interval_score=interval_score(cells))

    def add(table, key, label, subset, column, fn, sampled):
        part = cells[cells.subset.isin([subset] if isinstance(subset, str) else subset)]
        if sampled:
            part = part[part.sampled.fillna(False).astype(bool)]
        if column not in part:
            return
        for (alpha, method), group in part.groupby(["alpha", "method"]):
            rows.append(dict(table=table, key=key, label=label, alpha=alpha, method=method,
                             value=fn(group[column]), n=int(group[column].notna().sum())))

    for table, subset in (("2", "constrained_block_group"), ("2t", "constrained_tract")):
        add(table, "post_cover", "90% coverage", subset, "covers_published", mean, True)
        add(table, "absz_p90", "|z| p90", subset, "abs_z", quantile(0.90), True)
        add(table, "hw_p50", "Half-width / MOE", subset, "halfwidth_over_moe", quantile(0.50), True)
        add(table, "is", "Interval score / SE, mean", subset, "interval_score", mean, True)
    for table, level in (("3", "block_group"), ("3t", "tract")):
        held_out = (f"heldout_related_{level}", f"heldout_less_related_{level}")
        add(table, "all_cover", "90% coverage", held_out, "covers_published", mean, True)
        add(table, "all_absz_p90", "|z| p90", held_out, "abs_z", quantile(0.90), True)
        add(table, "all_is", "Interval score / SE, mean", held_out, "interval_score", mean,
            True)
    for table, subset in (("4", "constrained_block_group"), ("4t", "constrained_tract")):
        add(table, "mdiff_p50", "Median diff. p50", subset, "median_diff_ref_sd",
            quantile(0.50), False)
        add(table, "mdiff_p99", "Median diff. p99", subset, "median_diff_ref_sd",
            quantile(0.99), False)
        add(table, "width_p10", "Width ratio p10", subset, "width_ratio_ref", quantile(0.10),
            False)
        add(table, "width_p50", "Width ratio p50", subset, "width_ratio_ref", quantile(0.50),
            False)
    return rows


#: Tables 6 and 7's panels: (panel, label, subsets), block-group cells.
ACS_PANELS = [("constraint", "Constraint tables", ("constrained_block_group",)),
              ("heldout", "Held-out tables", ("heldout_related_block_group",
                                              "heldout_less_related_block_group"))]


#: (column, key with {} for the zeros' "0", table, label)
ACS_KEYS = [("tz_p50", "tz{}_p50", "6", "|z| p50"),
            ("tis", "tis{}", "7", "Interval score / SE, mean")]
#: Each table's statistics in one cell, joined by "/": the sampled cells', then
#: the published zeros'.
ACS_CELLS = {"6": (("tz_p50",), ("tz0_p50",)), "7": (("tis",), ("tis0",))}


def acs_table_statistics(cells: pd.DataFrame) -> list[dict]:
    """Tables 6 and 7, per ACS table: the median |z| and the mean interval
    score over the sampled leaf cells, and the same over the published zeros
    (keys ending ``0``)."""
    if not {"acs_table", "published", "abs_z"} <= set(cells.columns):
        return []
    if "halfwidth_over_moe" in cells:
        cells = cells.assign(interval_score=interval_score(cells))
    rows = []
    for panel, _, subsets in ACS_PANELS:
        part = cells[cells.subset.isin(subsets)]
        sampled = part.sampled.fillna(False).astype(bool)
        for suffix, mask in (("", sampled), ("0", part.published == 0)):
            group = part[mask].groupby(["acs_table", "alpha", "method"], observed=True)
            stats = pd.DataFrame({"tz_p50": group.abs_z.quantile(0.50),
                                  "n": group.abs_z.count()})
            if "interval_score" in part:
                stats["tis"] = group.interval_score.mean()
            for (acs, alpha, method), r in stats.iterrows():
                base = dict(alpha=alpha, method=method, n=int(r["n"]), acs_table=acs,
                            panel=panel)
                for column, key, table, label in ACS_KEYS:
                    if column in r and pd.notna(r[column]):
                        rows.append(dict(base, table=table, key=key.format(suffix),
                                         label=label + (", published zeros" if suffix else ""),
                                         value=float(r[column])))
    return rows


def draw_statistics(draws: pd.DataFrame, args) -> list[dict]:
    rows = []
    for (alpha, method), group in draws.groupby(["alpha", "method"]):
        base = dict(table="5", alpha=alpha, method=method, n=len(group))
        stats = [("walls", f"Draws with a cell > {args.wall:g} N",
                  float((group.max_share_of_N > args.wall).mean()))]
        if "min_bg_distinct_per_hh" in group:
            stats += [("dh_min_p50", "Smallest D/H, median",
                       float(group.min_bg_distinct_per_hh.median())),
                      ("dh_lt", f"Draws with D/H < {args.dh_floor:g}",
                       float((group.min_bg_distinct_per_hh < args.dh_floor).mean()))]
        if "min_bg_n_eff" in group:
            stats += [("neff_min_p50", "Smallest n_eff, median", float(group.min_bg_n_eff.median())),
                      ("neff_lt", f"Draws with n_eff < {args.neff_floor:g}",
                       float((group.min_bg_n_eff < args.neff_floor).mean()))]
        if "max_abs_hh_z" in group:
            stats += [("hhz_gt", f"Draws with households |z| > {args.hh_z:g}",
                       float((group.max_abs_hh_z > args.hh_z).mean())),
                      ("hhz_p50", "Largest households |z|, median",
                       float(group.max_abs_hh_z.median()))]
        rows += [dict(base, key=k, label=label, value=v) for k, label, v in stats]
    return rows


def scores_statistics(scores: pd.DataFrame, whiten: str = "vb_skewed") -> list[dict]:
    """Cost (Table 1) and k-hat (Table 4) from all_scores.csv; HMC's cost includes
    the fit that whitened it (``whiten``)."""
    scores = scores.assign(alpha=scores.alpha.astype(float).round(12))
    timing = scores[(scores.subset == "timing") & (scores.stat == "value")].pivot_table(
        index=["method", "puma", "alpha"], columns="metric", values="value", aggfunc="first")
    timing = timing.reset_index()
    vb_fit = timing[timing.method == whiten].set_index(["puma", "alpha"])["fit_seconds"]
    cost = []
    for _, row in timing.iterrows():
        method = row["method"]
        if method in ("hmc_short", "hmc_ref"):
            fit = vb_fit.get((row["puma"], row["alpha"]), np.nan)
            sampling = row.get("hmc_seconds", np.nan)
        elif method in ("ipf", "sinkhorn", "ipf_multinomial"):
            fit, sampling = row.get("fit_seconds", np.nan), np.nan
        else:
            fit = row.get("fit_seconds", np.nan)
            sampling = np.nansum([row.get("laplace_seconds", np.nan), row.get("draw_seconds", np.nan)])
        cost.append(dict(method=method, puma=row["puma"], alpha=row["alpha"],
                         fit=fit / 60, sampling=sampling / 60,
                         total=np.nansum([fit, sampling]) / 60))
    cost = pd.DataFrame(cost)
    rows = []
    for (alpha, method), group in cost.groupby(["alpha", "method"]):
        for key, label in (("fit", "Fit"), ("sampling", "Sampling"), ("total", "Total")):
            values = group[key]
            rows.append(dict(table="1", key=key, label=label, alpha=alpha, method=method,
                             value=float(values.mean()) if values.notna().any() else np.nan,
                             county=float(values.sum()) if values.notna().any() else np.nan,
                             n=int(values.notna().sum())))
    khat = scores[(scores.subset == "joint") & (scores.metric == "psis_khat")]
    for (alpha, method), group in khat.groupby(["alpha", "method"]):
        v = group.value.astype(float)
        rows.append(dict(table="4", key="khat", label="k-hat", alpha=alpha, method=method,
                         value=float(v.median()), low=float(v.min()), high=float(v.max()),
                         n=len(v)))
    return rows


# -- formatting -----------------------------------------------------------------

#: Decimals per statistic; "min" is whole minutes, "khat" median [min, max].
DIGITS = {"fit": "min", "sampling": "min", "total": "min",
          "post_cover": 2, "absz_p90": 2, "hw_p50": 2, "is": 2,
          "all_cover": 2, "all_absz_p90": 2, "all_is": 2,
          "tz_p50": 2, "tz0_p50": 2, "tis": 2, "tis0": 2,
          "mdiff_p50": 2, "mdiff_p99": 2, "width_p10": 2, "width_p50": 2, "khat": "khat",
          "walls": 3, "dh_min_p50": 3, "dh_lt": 3, "neff_min_p50": 1, "neff_lt": 3, "hhz_gt": 3, "hhz_p50": 2}


def fmt(row: pd.Series | None, key: str) -> str:
    if row is None or pd.isna(row["value"]):
        return "--"
    v, d = row["value"], DIGITS[key]
    if key in SHARE_KEYS and row["method"] in NO_ALPHA:
        return f"{round(v * row['n']):d}/{int(row['n'])}"
    if d == "min":
        return "<1" if v < 0.5 else f"{v:,.0f}"
    if d == "khat":
        return f"{v:.2f} [{row['low']:.2f}, {row['high']:.2f}]"
    return f"{v:.{d}f}"


TITLES = {
    "1": ("Cost", "Minutes per PUMA (mean over the four PUMAs). Fit: raking, MAP, or MAP + VB; "
                  "for HMC, the VB fit that whitens it (Gaussian or skewed, as the experiment "
                  "sets). Sampling: Laplace and drawing, "
                  "VB drawing (with rejection when truncated), or HMC. Fits ran on 48-core CPU "
                  "nodes, HMC on one GPU; county totals are in the CSV."),
    "2": ("Fit to the published block-group constraints",
          "All block-group constraint leaf cells of the four PUMAs pooled (published "
          "categories; totals and subtotals left out); sampled cells only "
          "(nonzero estimates with a sampling variance). 90% coverage: "
          "the share of cells whose published estimate lies between the 5th and "
          "95th percentiles of the method's draws. The published values were used in "
          "fitting, so this describes the fit and has no nominal level: even for the exact "
          "posterior it depends on how strongly each cell's own estimate determines its "
          "posterior. |z| = |posterior mean - published| / SE, 90th percentile over cells; "
          "half-width of the posterior 90% interval over the published MOE, median over "
          "cells. Interval score / SE as in Table 3; the published values were used in "
          "fitting, so here it too describes the fit. Calibration of the fitted cells is "
          "assessed against the reference posterior in Table 4."),
    "2t": ("Fit to the published tract constraints", "As Table 2, for tract cells."),
    "3": ("Held-out block-group accuracy",
          "Held-out ACS tables at block-group level; leaf cells of the four PUMAs pooled, "
          "sampled cells only. 90% coverage and |z| as in Table 2. The interval is for the true "
          "count, while the published value also carries sampling error, so coverage below "
          "0.90 is expected even for a correct model. Interval score / SE: the width of a "
          "90% predictive interval for the published value (about the posterior mean, the "
          "posterior 90% half-width and 1.645 SE added in quadrature) plus 20 times how far "
          "the published value falls outside it, in published SEs, mean over cells "
          "(Gneiting and Raftery 2007); lower is better, and a method that knew every true "
          "count would score about 4.1."),
    "3t": ("Held-out tract accuracy", "As Table 3, for tract cells."),
    "4": ("Agreement with the reference HMC",
          "Block-group constraint leaf cells of the four PUMAs pooled. Median diff.: |median - "
          "reference median| in reference sds; width ratio: 90% interval width over the "
          "reference's. k-hat: PSIS k-hat of the method's density against the posterior, one "
          "per fit, median [min, max] over PUMAs (methods with a density only)."),
    "4t": ("Tract agreement with the reference HMC",
           "As Table 4, for tract constraint leaf cells; k-hat, one per fit, is in Table 4."),
    "6": ("|z| by ACS table",
          "Block-group leaf cells of the four PUMAs pooled, by ACS table: constraint tables, "
          "then held-out tables. Each entry is the median |z| over the table's sampled cells "
          "(|z| as in Table 2); below it, in parentheses, the same over the table's published "
          "zeros, whose SE is the modelled zero-cell one, so their |z| is not on the "
          "sampled cells' scale."),
    "7": ("Interval score by ACS table",
          "Block-group leaf cells of the four PUMAs pooled, by ACS table, as in Table 6. "
          "Each entry is the mean interval score / SE (as in Table 3) over the table's "
          "sampled cells; below it, in parentheses, over its published zeros, with the "
          "modelled zero-cell SE. IPF, a single fit, has no interval."),
    "5": ("Usability of the draws",
          "Draws of the four PUMAs pooled (4,000 per PUMA; one for raking). A cell is one "
          "record in one block group. D/H: the expected number of distinct household records "
          "among a block group's H published households, drawn by the records' shares s, "
          "D = sum 1 - (1 - s)^H, over H; n_eff = 1 / sum s^2; both over block groups with "
          "more than 100 published households; "
          "households |z|: a block group's allocated occupied households against B25003, SE "
          "from its cells' variances with the zero-count floor."),
}


#: Tables written once per experiment in tables.md (the CSV stays one file, which
#: Figure C reads), the rest with the experiments side by side. LaTeX is always
#: per experiment, as bare tables (bare_tabular).
PER_EXPERIMENT = {"1", "6", "7"}

#: Tables by ACS table (rows ACS tables, columns methods at each alpha).
BY_ACS_TABLE = ("6", "7")

#: Short method labels for the by-ACS-table columns.
SHORT_METHODS = {"ipf": "IPF", "ipf_multinomial": "IPF mult.", "map_laplace": "Laplace",
                 "vb_gaussian": "VB G", "vb_skewed": "VB S", "vb_gaussian_trunc": "VB G tr.",
                 "vb_skewed_trunc": "VB S tr.", "hmc_short": "HMC short",
                 "hmc_ref": "HMC ref."}
SHORT_NOTE = (" Columns: IPF; IPF mult., IPF multinomial; Laplace, MAP + Laplace; VB G and "
              "VB S, VB Gaussian and skewed; tr., truncated; HMC short and reference.")

#: The sentence each table's note gains about the raking (no-alpha) rows.
RAKING_NOTES = {
    "default": " IPF has no alpha: one fit per PUMA, shown once.",
    "4": " IPF has no alpha; it is compared with each alpha's reference.",
    "4t": " IPF has no alpha; it is compared with each alpha's reference.",
    "5": " IPF has no alpha: one fit per PUMA, shown once; its shares of draws are counts "
         "of its four draws, one per PUMA.",
}


def wide(stats: pd.DataFrame, table: str, experiments: list[tuple[str, str]]) -> tuple[list, list, list]:
    """Column groups (key, label), the experiment labels, and the panels of rows."""
    part = stats[stats.table == table]
    keys = list(dict.fromkeys(part.key))
    labels = {k: part[part.key == k].label.iloc[0] for k in keys}
    index = {(r.experiment, r.alpha, r.method, r.key): r for _, r in part.iterrows()}
    panels = []
    per_alpha = table in ("4", "4t")
    if not per_alpha:
        rows = []
        for method, method_label in METHODS:
            if method not in NO_ALPHA:
                continue
            # The same fit in every alpha cell: read it from the first.
            values = [fmt(next((index[(e, a, method, k)] for a in ALPHAS
                                if (e, a, method, k) in index), None), k)
                      for k in keys for e, _ in experiments]
            if not all(v == "--" for v in values):
                rows.append((method_label, values))
        if rows:
            panels.append((None, rows))
    for alpha in ALPHAS:
        rows = []
        for method, method_label in METHODS:
            if method in NO_ALPHA and not per_alpha:
                continue
            values = [fmt(index.get((e, alpha, method, k)), k) for k in keys for e, _ in experiments]
            if all(v == "--" for v in values):
                continue
            rows.append((method_label, values))
        if rows:
            panels.append((alpha, rows))
    return [(k, labels[k]) for k in keys], [label for _, label in experiments], panels


#: Width of each value column of a bare table: its header wraps within it.
BARE_COLUMN = "1.6cm"


def bare_tabular(stats, table, experiment) -> str:
    """One experiment's table as a bare ``tabular`` (booktabs; no float, caption
    or note), for a document to wrap in its own table environment, caption and
    label: ``table<key>_<experiment>.tex``. Value columns are a fixed width, so
    long headers wrap; the document needs ``\\usepackage{booktabs,array}``."""
    groups, _, panels = wide(stats, table, [experiment])
    column = f">{{\\raggedleft\\arraybackslash}}p{{{BARE_COLUMN}}}"
    lines = ["\\begin{tabular}{l" + column * len(groups) + "}", "\\toprule",
             "Method & " + " & ".join(tex(label) for _, label in groups) + " \\\\", "\\midrule"]
    for i, (alpha, rows) in enumerate(panels):
        if i:
            lines.append("\\addlinespace")
        panel = "Raking (no $\\alpha$)" if alpha is None else f"$\\alpha = {alpha:g}$"
        lines.append(f"\\multicolumn{{{1 + len(groups)}}}{{l}}{{\\textit{{{panel}}}}} \\\\")
        lines += [f"{tex(label)} & " + " & ".join(tex(v) for v in values) + " \\\\"
                  for label, values in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def acs_wide(stats: pd.DataFrame, table: str) -> tuple[list, list]:
    """Tables 6 and 7's columns, ``(method, alpha or None)``, raking first, and
    their panels ``(label, [(acs table, values, zeros' values or None)])``."""
    part = stats[stats.table == table]
    index = {(r.panel, r.acs_table, r.alpha, r.method, r.key): r for r in part.itertuples()}
    have = set(zip(part.method, part.alpha))
    columns = [(m, None) for m, _ in METHODS if m in NO_ALPHA and any(m == h for h, _ in have)]
    columns += [(m, a) for a in ALPHAS for m, _ in METHODS
                if m not in NO_ALPHA and (m, a) in have]
    main, zeros = ACS_CELLS[table]

    def cell(panel, acs, method, alpha, keys):
        alphas = ALPHAS if alpha is None else (alpha,)  # raking: the same in every alpha
        found = [next((index[(panel, acs, a, method, k)] for a in alphas
                       if (panel, acs, a, method, k) in index), None) for k in keys]
        if all(f is None for f in found):
            return None
        return "/".join(fmt(None if f is None else pd.Series(f._asdict()), k)
                        for f, k in zip(found, keys))

    panels = []
    for panel, label, _ in ACS_PANELS:
        rows = []
        for acs in sorted(part[part.panel == panel].acs_table.unique()):
            values = [cell(panel, acs, m, a, main) or "--" for m, a in columns]
            zero = [cell(panel, acs, m, a, zeros) for m, a in columns]
            rows.append((acs, values, None if all(z is None for z in zero)
                         else [f"({z})" if z else "" for z in zero]))
        if rows:
            panels.append((label, rows))
    return columns, panels


def acs_tabular(stats, table) -> str:
    """Table 6 or 7 for one experiment as a bare ``tabular``: a row per ACS table,
    its published zeros' values on the row below; columns grouped by alpha."""
    columns, panels = acs_wide(stats, table)
    groups = [(alpha, sum(1 for _, a in columns if a == alpha))
              for alpha in dict.fromkeys(a for _, a in columns)]
    lines = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "r" * len(columns) + "}",
             "\\toprule"]
    header, rules, start = [""], [], 2
    for alpha, n in groups:
        name = "Raking" if alpha is None else f"$\\alpha = {alpha:g}$"
        header.append(f"\\multicolumn{{{n}}}{{c}}{{{name}}}")
        rules.append(f"\\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n
    lines += [" & ".join(header) + " \\\\", " ".join(rules),
              "ACS table & " + " & ".join(SHORT_METHODS.get(m, m) for m, _ in columns) + " \\\\",
              "\\midrule"]
    for i, (label, rows) in enumerate(panels):
        if i:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{{1 + len(columns)}}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for acs, values, zero in rows:
            lines.append(acs.replace("_", "\\_") + " & " + " & ".join(values) + " \\\\")
            if zero is not None:
                lines.append(" & " + " & ".join(zero) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def acs_markdown(stats, table, experiment) -> str:
    columns, panels = acs_wide(stats, table)
    title, note = TITLES[table]
    header = ["ACS table"] + [SHORT_METHODS.get(m, m) + ("" if a is None else f", α = {a:g}")
                              for m, a in columns]
    lines = [f"### Table {table}. {title}: {experiment[1]}".replace("|z|", "\\|z\\|"), "",
             "| " + " | ".join(header) + " |", "|" + "---|" + "---:|" * len(columns)]
    for label, rows in panels:
        lines.append(f"| *{label}* |" + " |" * len(columns))
        for acs, values, zero in rows:
            lines.append("| " + " | ".join([acs] + [v if zero is None or not z else f"{v} {z}"
                                                    for v, z in zip(values, zero or values)])
                         + " |")
    lines += ["", (note + SHORT_NOTE).replace("|z|", "\\|z\\|"), ""]
    return "\n".join(lines) + "\n"


def tex(text: str) -> str:
    text = text.replace("+-", "$\\pm$").replace("|z|", "$|z|$").replace("%", "\\%")
    text = text.replace("n_eff", "$n_{\\mathrm{eff}}$").replace("<1", "$<$1")
    text = re.sub(r"sqrt\(SE\^2 \+ sd\^2\)", r"$\\sqrt{\\mathrm{SE}^2 + \\mathrm{sd}^2}$", text)
    text = text.replace("sum s^2", "$\\sum s^2$").replace("SE^2", "SE$^2$")
    text = text.replace("no alpha", "no $\\alpha$").replace("each alpha's", "each $\\alpha$'s")
    return text.replace(" < ", " $<$ ").replace(" > ", " $>$ ")


def markdown(stats, table, experiments, part: str | None = None) -> str:
    groups, exps, panels = wide(stats, table, experiments)
    title, note = TITLES[table]
    note = (note + RAKING_NOTES.get(table, RAKING_NOTES["default"])).replace(
        "no alpha", "no $\\alpha$").replace("each alpha's", "each $\\alpha$'s")
    if part is not None:
        title = f"{title}: {exps[0]}"
    header = ["Method"] + [(label if len(exps) == 1 else f"{label}: {e}").replace("|", "\\|")
                           for _, label in groups for e in exps]
    lines = [f"### Table {table}. {title}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" + "---:|" * (len(header) - 1)]
    for alpha, rows in panels:
        label = "*Raking (no $\\alpha$)*" if alpha is None else f"$\\alpha = {alpha:g}$"
        lines.append(f"| {label} |" + " |" * (len(header) - 1))
        lines += ["| " + " | ".join([label] + values) + " |" for label, values in rows]
    lines += ["", note, ""]
    return "\n".join(lines) + "\n"


NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def word_notes(pumas: list[str]) -> None:
    """Word the notes for the PUMAs pooled, which they are written for four of."""
    n = len(pumas)
    if n == 4:
        return
    the = f"PUMA {pumas[0]}" if n == 1 else f"the {NUMBER_WORDS.get(n, n)} PUMAs"
    draws = ("its one draw" if n == 1 else f"its {NUMBER_WORDS.get(n, n)} draws, one per PUMA")
    for key, (title, note) in TITLES.items():
        note = (note.replace("mean over the four PUMAs", "mean over " + the if n > 1 else the)
                .replace("of the four PUMAs", "of " + the).replace("over PUMAs", "over " + (
                    "PUMAs" if n > 1 else "its one fit")))
        TITLES[key] = (title, note)
    for key, note in RAKING_NOTES.items():
        RAKING_NOTES[key] = note.replace("its four draws, one per PUMA", draws)


def flatten(value, prefix: str) -> list[tuple[str, str]]:
    """``(dotted key, text)`` of every leaf of a JSON value; lists as ``a;b``."""
    if isinstance(value, dict):
        return [row for key, v in value.items() for row in flatten(v, f"{prefix}.{key}")]
    if isinstance(value, list) and not any(isinstance(v, (dict, list)) for v in value):
        return [(prefix, ";".join(str(v) for v in value))]
    if isinstance(value, list):
        return [row for i, v in enumerate(value) for row in flatten(v, f"{prefix}.{i}")]
    return [(prefix, "" if value is None else str(value))]


def settings(args, experiments: list[tuple[str, str]]) -> pd.DataFrame:
    """``settings.csv``: what the paper quotes as settings rather than results."""
    from run_experiment import FITS_SLURM

    from pmedm_vb.data.variance import N_REPLICATES

    rows = [("tag", args.tag), ("wall", args.wall), ("dh_floor", args.dh_floor),
            ("neff_floor", args.neff_floor), ("hh_z", args.hh_z),
            ("n_replicates", N_REPLICATES),
            ("fit_cpus", next(o.split("=")[1] for o in FITS_SLURM if "cpus-per-task" in o))]
    source = (Path(__file__).resolve().parent / "compare_methods.py").read_text()
    hh_min = re.search(r'"--hh-min", type=float, default=([0-9.]+)', source)
    rows.append(("hh_min", hh_min.group(1) if hh_min else ""))
    for name, label in experiments:
        rows.append((f"{name}.label", label))
        record = args.root / name / "experiment.json"
        if record.exists():
            rows += flatten(json.loads(record.read_text()), name)
        scoring = args.root / name / "scores" / args.tag / "scoring.json"
        if scoring.exists():
            rows.append((f"{name}.scoring.draws", json.loads(scoring.read_text())["draws"]))
    return pd.DataFrame(rows, columns=["key", "value"])


def keep(frame: pd.DataFrame, args) -> pd.DataFrame:
    """The rows of the PUMAs and alphas asked for."""
    if args.pumas is not None:
        frame = frame[frame.puma.astype(str).isin(args.pumas)]
    if args.alphas is not None:
        alpha = frame.alpha.astype(float)
        frame = frame[np.isclose(alpha.to_numpy()[:, None], np.array(args.alphas)[None, :]).any(axis=1)]
    return frame


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = dict(EXPERIMENTS)
    experiments, stats, pooled, samples = [], [], set(), []
    for name in args.experiments:
        folder = args.root / name / "scores" / args.tag
        if not (folder / "all_scores.csv").exists():
            print(f"skipping {name}: no {folder / 'all_scores.csv'}", flush=True)
            continue
        experiments.append((name, labels.get(name, name)))
        cells = keep(read_detail(folder / "detail", "cells", CELL_COLUMNS, CELL_SUBSETS), args)
        rows = cell_statistics(cells) + acs_table_statistics(cells)
        del cells
        draws = keep(read_detail(folder / "detail", "draws", DRAW_COLUMNS), args)
        rows += draw_statistics(draws, args)
        samples.append(sample_draws(draws).assign(experiment=name))
        del draws
        scores = keep(pd.read_csv(folder / "all_scores.csv", dtype={"puma": str}), args)
        pooled |= set(scores.puma.astype(str))
        record = args.root / name / "experiment.json"
        whiten = (f"vb_{json.loads(record.read_text())['hmc'].get('whiten', 'skewed')}"
                  if record.exists() else "vb_skewed")
        rows += scores_statistics(scores, whiten)
        stats.append(pd.DataFrame(rows).assign(experiment=name))
        print(f"read {name}", flush=True)
    if not stats:
        raise SystemExit("no experiment has its scoring")
    stats = pd.concat(stats, ignore_index=True)
    word_notes(sorted(pooled))
    order = {m: i for i, (m, _) in enumerate(METHODS)}
    key_order = {k: i for i, k in enumerate(DIGITS)}
    stats = stats.assign(method_order=stats.method.map(order), key_order=stats.key.map(key_order))
    stats = stats.sort_values(["table", "key_order", "experiment", "alpha", "method_order"]).drop(
        columns=["method_order", "key_order"])
    md = [f"# Tables: experiments {', '.join(label for _, label in experiments)} "
          f"(scoring {args.tag})", ""]
    for table, (title, _) in TITLES.items():
        if not (stats.table == table).any():
            continue
        slug = f"table{table}_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        part = stats[stats.table == table]
        if table not in BY_ACS_TABLE:
            part = part.drop(columns=["acs_table", "panel"], errors="ignore")
        part.to_csv(args.out / f"{slug}.csv", index=False)
        for experiment in experiments:  # bare, per experiment: table2_baseline.tex
            short = re.sub(r"^exp\d+_", "", experiment[0])
            one = stats[stats.experiment == experiment[0]]
            (args.out / f"table{table}_{short}.tex").write_text(
                acs_tabular(one, table) if table in BY_ACS_TABLE
                else bare_tabular(one, table, experiment))
        # LaTeX only as the bare tables above: remove the floats older code wrote.
        for old in args.out.glob(f"{slug}*.tex"):
            old.unlink()
        if table not in PER_EXPERIMENT:
            md.append(markdown(stats, table, experiments))
            continue
        for experiment in experiments:
            one = stats[stats.experiment == experiment[0]]
            md.append(acs_markdown(one, table, experiment) if table in BY_ACS_TABLE
                      else markdown(one, table, [experiment], experiment[0]))
    (args.out / "tables.md").write_text("\n".join(md))
    pd.concat(samples, ignore_index=True).to_csv(args.out / "draws_sample.csv", index=False)
    settings(args, experiments).to_csv(args.out / "settings.csv", index=False)
    # Last, so that a run cut short leaves the folder unstamped (run_paper.py).
    write_stamp(args.out / "tables.md", TABLES_SCRIPTS, tag=args.tag, experiments=args.experiments)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
