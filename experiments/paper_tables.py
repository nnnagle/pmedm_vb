"""The paper's tables for experiments 1-3, every statistic pooled over the PUMAs.

Reads, for each experiment, the scoring ``<root>/<experiment>/scores/<tag>``:
``detail/*/<method>_cells.parquet`` (every scored cell),
``detail/*/<method>_draws.parquet`` (every draw) and ``all_scores.csv``
(timings and PSIS k-hat), and writes five tables, plus the tract versions of
Tables 2 and 3, each as LaTeX (booktabs), CSV (long: one row per experiment,
alpha, method and statistic, unrounded, with ``n`` the cells, draws or PUMAs
behind it) and Markdown (all in ``tables.md``); and ``draws_sample.csv``, up
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
CELL_COLUMNS = ["method", "puma", "alpha", "subset", "sampled", "abs_z",
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
    parser.add_argument("--dh-floor", type=float, default=0.025,
                        help="Table 5's D/H floor: about half the smallest any HMC draw of the "
                             "paper_v3 runs reached (0.054); the truncated VB's rule")
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
    count the same people again and weight the large aggregates."""
    frames = []
    for path in sorted(folder.glob(f"*/*_{kind}.parquet")):
        names = pq.read_schema(path).names
        have = [c for c in columns if c in names]
        leaves_only = kind == "cells" and "cell" in names
        frame = pd.read_parquet(path, columns=have + (["cell"] if leaves_only else []))
        if leaves_only:
            frame = frame[~frame.pop("cell").str.endswith(SUBTOTAL_MARK)]
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
        add(table, "post_cover", "Posterior 90% interval contains published", subset,
            "covers_published", mean, True)
        add(table, "absz_p90", "|z| p90", subset, "abs_z", quantile(0.90), True)
        add(table, "hw_p50", "Half-width / MOE", subset, "halfwidth_over_moe", quantile(0.50), True)
        add(table, "is", "Interval score / SE, mean", subset, "interval_score", mean, True)
    for table, level in (("3", "block_group"), ("3t", "tract")):
        held_out = (f"heldout_related_{level}", f"heldout_less_related_{level}")
        add(table, "all_cover", "All: posterior 90% interval contains published", held_out,
            "covers_published", mean, True)
        add(table, "all_absz_p90", "All: |z| p90", held_out, "abs_z", quantile(0.90), True)
        add(table, "all_is", "All: interval score / SE, mean", held_out, "interval_score", mean,
            True)
        for kind, subset in (("rel", held_out[0]), ("less", held_out[1])):
            name = "Related" if kind == "rel" else "Less related"
            add(table, f"{kind}_cover", f"{name}: posterior 90% interval contains published",
                subset, "covers_published", mean, True)
            add(table, f"{kind}_absz_p90", f"{name}: |z| p90", subset, "abs_z", quantile(0.90),
                True)
            add(table, f"{kind}_is", f"{name}: interval score / SE, mean", subset,
                "interval_score", mean, True)
    subset = "constrained_block_group"
    add("4", "mdiff_p50", "Median diff. p50", subset, "median_diff_ref_sd", quantile(0.50), False)
    add("4", "mdiff_p99", "Median diff. p99", subset, "median_diff_ref_sd", quantile(0.99), False)
    add("4", "width_p10", "Width ratio p10", subset, "width_ratio_ref", quantile(0.10), False)
    add("4", "width_p50", "Width ratio p50", subset, "width_ratio_ref", quantile(0.50), False)
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
          "rel_cover": 2, "rel_absz_p90": 2, "rel_is": 2,
          "less_cover": 2, "less_absz_p90": 2, "less_is": 2,
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
                  "for HMC, the skewed VB fit that whitens it. Sampling: Laplace and drawing, "
                  "VB drawing (with rejection when truncated), or HMC. Fits ran on 48-core CPU "
                  "nodes, HMC on one GPU; county totals are in the CSV."),
    "2": ("Fit to the published block-group constraints",
          "All block-group constraint leaf cells of the four PUMAs pooled (published "
          "categories; totals and subtotals left out); sampled cells only "
          "(nonzero estimates with a sampling variance). Posterior 90% interval contains "
          "published: the share of cells whose published estimate lies between the 5th and "
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
          "Held-out ACS tables at block-group level: all of them, then those related and "
          "less related to the constraints; leaf cells of the four PUMAs pooled, sampled "
          "cells only. Posterior 90% "
          "interval contains published, and |z|, as in Table 2. The interval is for the true "
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
    "5": ("Usability of the draws",
          "Draws of the four PUMAs pooled (4,000 per PUMA; one for raking). A cell is one "
          "record in one block group. D/H: the expected number of distinct household records "
          "among a block group's H published households, drawn by the records' shares s, "
          "D = sum 1 - (1 - s)^H, over H; n_eff = 1 / sum s^2; both over block groups with "
          "more than 100 published households; "
          "households |z|: a block group's allocated occupied households against B25003, SE "
          "from its cells' variances with the zero-count floor."),
}


#: The sentence each table's note gains about the raking (no-alpha) rows.
RAKING_NOTES = {
    "default": " IPF has no alpha: one fit per PUMA, shown once.",
    "4": " IPF has no alpha; it is compared with each alpha's reference.",
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
    per_alpha = table == "4"
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


def latex(stats, table, experiments) -> str:
    groups, exps, panels = wide(stats, table, experiments)
    n = len(exps)
    title, note = TITLES[table]
    note = note + RAKING_NOTES.get(table, RAKING_NOTES["default"])
    lines = ["% requires \\usepackage{booktabs}",
             "\\begin{table}", "\\centering", f"\\caption{{{title}}}",
             f"\\label{{tab:{table}}}",
             "\\begin{tabular}{l" + "".join(" " + "r" * n for _ in groups) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{tex(label)}}}" for _, label in groups)
             + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(groups))),
             "Method & " + " & ".join(tex(e) for _ in groups for e in exps) + " \\\\", "\\midrule"]
    for i, (alpha, rows) in enumerate(panels):
        if i:
            lines.append("\\addlinespace")
        label = "Raking (no $\\alpha$)" if alpha is None else f"$\\alpha = {alpha:g}$"
        lines.append(f"\\multicolumn{{{1 + n * len(groups)}}}{{l}}{{\\textit{{{label}}}}} \\\\")
        lines += [f"{tex(label)} & " + " & ".join(tex(v) for v in values) + " \\\\"
                  for label, values in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip",
              f"\\parbox{{\\linewidth}}{{\\footnotesize {tex(note)}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def tex(text: str) -> str:
    text = text.replace("+-", "$\\pm$").replace("|z|", "$|z|$").replace("%", "\\%")
    text = text.replace("n_eff", "$n_{\\mathrm{eff}}$").replace("<1", "$<$1")
    text = re.sub(r"sqrt\(SE\^2 \+ sd\^2\)", r"$\\sqrt{\\mathrm{SE}^2 + \\mathrm{sd}^2}$", text)
    text = text.replace("sum s^2", "$\\sum s^2$").replace("SE^2", "SE$^2$")
    text = text.replace("no alpha", "no $\\alpha$").replace("each alpha's", "each $\\alpha$'s")
    return text.replace(" < ", " $<$ ").replace(" > ", " $>$ ")


def markdown(stats, table, experiments) -> str:
    groups, exps, panels = wide(stats, table, experiments)
    title, note = TITLES[table]
    note = (note + RAKING_NOTES.get(table, RAKING_NOTES["default"])).replace(
        "no alpha", "no $\\alpha$").replace("each alpha's", "each $\\alpha$'s")
    header = ["Method"] + [f"{label}: {e}".replace("|", "\\|") for _, label in groups for e in exps]
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
        rows = cell_statistics(cells)
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
        stats[stats.table == table].to_csv(args.out / f"{slug}.csv", index=False)
        (args.out / f"{slug}.tex").write_text(latex(stats, table, experiments))
        md.append(markdown(stats, table, experiments))
    (args.out / "tables.md").write_text("\n".join(md))
    pd.concat(samples, ignore_index=True).to_csv(args.out / "draws_sample.csv", index=False)
    # Last, so that a run cut short leaves the folder unstamped (run_paper.py).
    write_stamp(args.out / "tables.md", TABLES_SCRIPTS, tag=args.tag, experiments=args.experiments)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
