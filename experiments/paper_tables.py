"""The paper's tables for experiments 1-3, every statistic pooled over the PUMAs.

Reads, for each experiment, the scoring ``<root>/<experiment>/scores/<tag>``:
``detail/*/<method>_cells.parquet`` (every scored cell),
``detail/*/<method>_draws.parquet`` (every draw) and ``all_scores.csv``
(timings and PSIS k-hat), and writes five tables, plus the tract version of
Table 2, each as LaTeX (booktabs), CSV (long: one row per experiment, alpha,
method and statistic, unrounded, with ``n`` the cells, draws or PUMAs behind
it) and Markdown (all in ``tables.md``)::

    sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_python.sbatch \\
        paper_tables.py --root $PAPER --tag paper1 --out $PAPER/tables

Rows are methods, one panel per alpha, columns each statistic for each
experiment. Pooling:

- Over cells (Tables 2-4): every cell of every PUMA together, so quantiles are
  quantiles of the pooled cells. z and coverage use the *sampled* cells only
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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

#: (folder, label)
EXPERIMENTS = [("exp01_baseline", "Baseline"), ("exp02_nullspace", "Null space"),
               ("exp03_rollup", "Roll-up")]

#: (method, label), in the tables' order.
METHODS = [("ipf", "IPF"), ("sinkhorn", "Sinkhorn"), ("map_laplace", "MAP + Laplace"),
           ("vb_gaussian", "VB Gaussian"), ("vb_skewed", "VB skewed"),
           ("vb_gaussian_trunc", "VB Gaussian, truncated"),
           ("vb_skewed_trunc", "VB skewed, truncated"),
           ("hmc_short", "HMC short"), ("hmc_ref", "HMC reference")]

ALPHAS = (0.01, 0.1, 1.0)

#: Cell columns read from the detail files.
CELL_COLUMNS = ["method", "puma", "alpha", "subset", "sampled", "abs_z",
                "covers_published",
                "halfwidth_over_moe", "median_diff_ref_sd", "width_ratio_ref"]
CELL_SUBSETS = ("constrained_block_group", "constrained_tract",
                "heldout_related_block_group", "heldout_less_related_block_group")
DRAW_COLUMNS = ["method", "puma", "alpha", "max_share_of_N", "min_bg_n_eff", "max_abs_hh_z",
                "count_hh_abs_z_gt_5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--tag", default="paper1", help="the scoring tag of every experiment")
    parser.add_argument("--experiments", nargs="+", default=[e for e, _ in EXPERIMENTS],
                        help="experiment folders, in column order")
    parser.add_argument("--wall", type=float, default=0.01, help="a wall: a cell over this share of N")
    parser.add_argument("--neff-floor", type=float, default=10)
    parser.add_argument("--hh-z", type=float, default=5)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


# -- reading ------------------------------------------------------------------


def read_detail(folder: Path, kind: str, columns: list[str], subsets=None) -> pd.DataFrame:
    """Every ``<method>_<kind>.parquet`` under ``folder``, with the columns it has."""
    frames = []
    for path in sorted(folder.glob(f"*/*_{kind}.parquet")):
        have = [c for c in columns if c in pq.read_schema(path).names]
        frame = pd.read_parquet(path, columns=have)
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


def cell_statistics(cells: pd.DataFrame) -> list[dict]:
    rows = []

    def add(table, key, label, subset, column, fn, sampled):
        part = cells[cells.subset == subset]
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
    for kind, subset in (("rel", "heldout_related_block_group"),
                         ("less", "heldout_less_related_block_group")):
        name = "Related" if kind == "rel" else "Less related"
        add("3", f"{kind}_cover", f"{name}: posterior 90% interval contains published", subset,
            "covers_published", mean, True)
        add("3", f"{kind}_absz_p90", f"{name}: |z| p90", subset, "abs_z", quantile(0.90), True)
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


def scores_statistics(scores: pd.DataFrame) -> list[dict]:
    """Cost (Table 1) and k-hat (Table 4) from all_scores.csv."""
    scores = scores.assign(alpha=scores.alpha.astype(float).round(12))
    timing = scores[(scores.subset == "timing") & (scores.stat == "value")].pivot_table(
        index=["method", "puma", "alpha"], columns="metric", values="value", aggfunc="first")
    timing = timing.reset_index()
    vb_fit = timing[timing.method == "vb_skewed"].set_index(["puma", "alpha"])["fit_seconds"]
    cost = []
    for _, row in timing.iterrows():
        method = row["method"]
        if method in ("hmc_short", "hmc_ref"):
            fit = vb_fit.get((row["puma"], row["alpha"]), np.nan)
            sampling = row.get("hmc_seconds", np.nan)
        elif method in ("ipf", "sinkhorn"):
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
          "post_cover": 2, "absz_p90": 2, "hw_p50": 2,
          "rel_cover": 2, "rel_absz_p90": 2, "less_cover": 2, "less_absz_p90": 2,
          "mdiff_p50": 2, "mdiff_p99": 2, "width_p10": 2, "width_p50": 2, "khat": "khat",
          "walls": 3, "neff_min_p50": 1, "neff_lt": 3, "hhz_gt": 3, "hhz_p50": 2}


def fmt(row: pd.Series | None, key: str) -> str:
    if row is None or pd.isna(row["value"]):
        return "--"
    v, d = row["value"], DIGITS[key]
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
          "All block-group constraint cells of the four PUMAs pooled; sampled cells only "
          "(nonzero estimates with a sampling variance). Posterior 90% interval contains "
          "published: the share of cells whose published estimate lies between the 5th and "
          "95th percentiles of the method's draws. The published values were used in "
          "fitting, so this describes the fit and has no nominal level: even for the exact "
          "posterior it depends on how strongly each cell's own estimate determines its "
          "posterior. |z| = |posterior mean - published| / SE, 90th percentile over cells; "
          "half-width of the posterior 90% interval over the published MOE, median over "
          "cells. Calibration of the fitted cells is assessed against the reference "
          "posterior in Table 4."),
    "2t": ("Fit to the published tract constraints", "As Table 2, for tract cells."),
    "3": ("Held-out block-group accuracy",
          "Held-out ACS tables at block-group level, related and less related to the "
          "constraints; cells of the four PUMAs pooled, sampled cells only. Posterior 90% "
          "interval contains published, and |z|, as in Table 2. The interval is for the true "
          "count, while the published value also carries sampling error, so coverage below "
          "0.90 is expected even for a correct model."),
    "4": ("Agreement with the reference HMC",
          "Block-group constraint cells of the four PUMAs pooled. Median diff.: |median - "
          "reference median| in reference sds; width ratio: 90% interval width over the "
          "reference's. k-hat: PSIS k-hat of the method's density against the posterior, one "
          "per fit, median [min, max] over PUMAs (methods with a density only)."),
    "5": ("Usability of the draws",
          "Draws of the four PUMAs pooled (4,000 per PUMA; one for raking). A cell is one "
          "record in one block group. n_eff = 1 / sum s^2 over the household records' shares "
          "of a block group's households, block groups over 100 published households; "
          "households |z|: a block group's allocated occupied households against B25003, SE "
          "from its cells' variances with the zero-count floor."),
}


def wide(stats: pd.DataFrame, table: str, experiments: list[tuple[str, str]]) -> tuple[list, list, list]:
    """Column groups (key, label), the experiment labels, and the panels of rows."""
    part = stats[stats.table == table]
    keys = list(dict.fromkeys(part.key))
    labels = {k: part[part.key == k].label.iloc[0] for k in keys}
    index = {(r.experiment, r.alpha, r.method, r.key): r for _, r in part.iterrows()}
    panels = []
    for alpha in ALPHAS:
        rows = []
        for method, method_label in METHODS:
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
        lines.append(f"\\multicolumn{{{1 + n * len(groups)}}}{{l}}{{\\textit{{$\\alpha = {alpha:g}$}}}} \\\\")
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
    return text.replace(" < ", " $<$ ").replace(" > ", " $>$ ")


def markdown(stats, table, experiments) -> str:
    groups, exps, panels = wide(stats, table, experiments)
    title, note = TITLES[table]
    header = ["Method"] + [f"{label}: {e}".replace("|", "\\|") for _, label in groups for e in exps]
    lines = [f"### Table {table}. {title}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" + "---:|" * (len(header) - 1)]
    for alpha, rows in panels:
        lines.append(f"| *α = {alpha:g}* |" + " |" * (len(header) - 1))
        lines += ["| " + " | ".join([label] + values) + " |" for label, values in rows]
    lines += ["", note, ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = dict(EXPERIMENTS)
    experiments, stats = [], []
    for name in args.experiments:
        folder = args.root / name / "scores" / args.tag
        if not (folder / "all_scores.csv").exists():
            print(f"skipping {name}: no {folder / 'all_scores.csv'}", flush=True)
            continue
        experiments.append((name, labels.get(name, name)))
        cells = read_detail(folder / "detail", "cells", CELL_COLUMNS, CELL_SUBSETS)
        rows = cell_statistics(cells)
        del cells
        rows += draw_statistics(read_detail(folder / "detail", "draws", DRAW_COLUMNS), args)
        rows += scores_statistics(pd.read_csv(folder / "all_scores.csv", dtype={"puma": str}))
        stats.append(pd.DataFrame(rows).assign(experiment=name))
        print(f"read {name}", flush=True)
    if not stats:
        raise SystemExit("no experiment has its scoring")
    stats = pd.concat(stats, ignore_index=True)
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
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
