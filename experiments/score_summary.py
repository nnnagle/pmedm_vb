"""Condense the comparison grid's long CSV into a few readable tables.

Reads ``compare_methods.py``'s combined CSV (``all_scores.csv``) and writes
``<out>/score_summary.txt``: for each alpha, one row per method and one column
per headline metric, each cell the median over PUMAs with its range. Also lists
the (subset, metric, stat) rows that some (method, PUMA, alpha) cells lack and
others have, so a short cell can be explained. The long CSV stays the source
for tables and figures in R::

    $CONDA_PREFIX/bin/python experiments/score_summary.py \\
        --scores $RUNS/scores/all_scores.csv --out $RUNS/scores

With ``--by-table`` (the per-table CSV of ``compare_methods.py --by-table``)
it writes ``<out>/table_summary.txt`` instead: per alpha and ACS table, the
bias and coverage against the published values and the ratio of the draws'
90% half-width to the published MOE, the last two over sampled cells only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["ipf", "sinkhorn", "map_laplace", "vb_gaussian", "vb_skewed", "vb_sumdiff",
         "hmc_short", "hmc_ref"]

#: (label, subset, metric, stat)
HEADLINE = [
    ("fit_s", "timing", "fit_seconds", "value"),
    ("hmc_s", "timing", "hmc_seconds", "value"),
    ("bg_cover", "constrained_block_group", "covers_published", "mean"),
    ("bg_absz_p90", "constrained_block_group", "abs_z", "p90"),
    ("tr_absz_p90", "constrained_tract", "abs_z", "p90"),
    ("bg_hw/moe", "constrained_block_group", "halfwidth_over_moe", "p50"),
    ("hr_cover", "heldout_related_block_group", "covers_published", "mean"),
    ("hr_z_mean", "heldout_related_block_group", "z", "mean"),
    ("hr_absz_p90", "heldout_related_block_group", "abs_z", "p90"),
    ("hl_cover", "heldout_less_related_block_group", "covers_published", "mean"),
    ("bg_mdiff_p50", "constrained_block_group", "median_diff_ref_sd", "p50"),
    ("bg_mdiff_p99", "constrained_block_group", "median_diff_ref_sd", "p99"),
    ("bg_width_p10", "constrained_block_group", "width_ratio_ref", "p10"),
    ("bg_width_p50", "constrained_block_group", "width_ratio_ref", "p50"),
    ("xt_width_p50", "crosstab_block_group", "width_ratio_ref", "p50"),
    ("gt1%N", "p", "largest_cell_gt_0.01_of_N", "share_of_draws"),
    ("gt10%N", "p", "largest_cell_gt_0.1_of_N", "share_of_draws"),
    ("bg25%_any", "p", "count_bg_share_gt_0.25", "share_of_draws_any"),
    ("maxN_p99", "p", "max_share_of_N", "p99"),
    ("khat", "joint", "psis_khat", "value"),
    ("pair_diff_p50", "joint", "pair_sd_diff_ratio", "p50"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--by-table", type=Path, default=None,
                        help="compare_methods.py --by-table CSV: write table_summary.txt instead")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def fmt(x: pd.Series) -> str:
    x = x.dropna()
    if not len(x):
        return "-"
    def one(v):
        if not np.isfinite(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"
    if len(x) == 1 or x.min() == x.max():
        return one(np.median(x))
    return f"{one(np.median(x))} [{one(x.min())},{one(x.max())}]"


#: (label, method, cells, metric, stat) for the per-table view.
TABLE_COLUMNS = [
    ("n_sampled", "hmc_ref", "sampled", "z", "n"),
    ("z_mean:ref", "hmc_ref", "all", "z", "mean"),
    ("z_mean:sumdiff", "vb_sumdiff", "all", "z", "mean"),
    ("z_mean:sinkhorn", "sinkhorn", "all", "z", "mean"),
    ("cover:ref", "hmc_ref", "sampled", "covers_published", "mean"),
    ("cover:sumdiff", "vb_sumdiff", "sampled", "covers_published", "mean"),
    ("hw/moe:ref", "hmc_ref", "sampled", "halfwidth_over_moe", "p50"),
    ("hw/moe:sumdiff", "vb_sumdiff", "sampled", "halfwidth_over_moe", "p50"),
    ("hw/moe:skewed", "vb_skewed", "sampled", "halfwidth_over_moe", "p50"),
    ("hw/moe:laplace", "map_laplace", "sampled", "halfwidth_over_moe", "p50"),
    ("hw/moe:ref_p10", "hmc_ref", "sampled", "halfwidth_over_moe", "p10"),
    ("hw/moe:ref_p90", "hmc_ref", "sampled", "halfwidth_over_moe", "p90"),
]


def table_view(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"puma": str})
    lines = ["Per ACS table (median over PUMAs [min,max]). z_mean over all cells; cover and",
             "hw/moe (90% half-width over the published MOE) over sampled cells only: nonzero",
             "estimates with a sampling variance at or above the zero-count floor.", ""]
    for alpha, block in frame.groupby("alpha"):
        rows = {}
        for (subset, table), group in block.groupby(["subset", "table"]):
            row = {}
            for label, method, cells, metric, stat in TABLE_COLUMNS:
                sel = group[(group.method == method) & (group.cells == cells)
                            & (group.metric == metric) & (group.stat == stat)]
                row[label] = fmt(sel.groupby("puma")["value"].first())
            rows[(subset.replace("_block_group", "_bg"), table)] = row
        table_frame = pd.DataFrame(rows).T
        lines.append(f"== alpha {alpha:g}")
        half = 6
        for cols in (list(table_frame.columns[:half]), list(table_frame.columns[half:])):
            lines += [table_frame[cols].to_string(), ""]
    return lines


def main() -> None:
    args = parse_args()
    if args.by_table is not None:
        text = "\n".join(table_view(args.by_table)) + "\n"
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "table_summary.txt").write_text(text)
        print(text)
        return
    scores = pd.read_csv(args.scores, dtype={"puma": str})
    lines = ["Median over PUMAs [min,max]. Held-out: hr = related, hl = less related (block group).",
             "mdiff / width: against the HMC reference, in reference sds / as a 90% width ratio.",
             "gt1%N, gt10%N: share of draws whose largest cell exceeds that share of N.", ""]
    for alpha, block in scores.groupby("alpha"):
        rows = {}
        for method, group in block.groupby("method"):
            row = {}
            for label, subset, metric, stat in HEADLINE:
                sel = group[(group.subset == subset) & (group.metric == metric) & (group.stat == stat)]
                row[label] = fmt(sel.groupby("puma")["value"].first())
            rows[method] = row
        table = pd.DataFrame(rows).T.reindex([m for m in ORDER if m in rows])
        lines.append(f"== alpha {alpha:g}")
        half = len(HEADLINE) // 2 + 1
        for cols in (list(table.columns[:half]), list(table.columns[half:])):
            lines += [table[cols].to_string(), ""]

    keys = ["subset", "metric", "stat"]
    present = scores.groupby(["method", "puma", "alpha"])[keys].apply(
        lambda g: set(map(tuple, g.to_numpy())))
    lines.append("Rows missing from some cells, relative to the same method at other PUMAs/alphas:")
    found = False
    for method, sets in present.groupby(level="method"):
        union = set().union(*sets)
        for (m, puma, alpha), have in sets.items():
            for key in sorted(union - have):
                lines.append(f"  {m} {puma} a{alpha:g}: {' / '.join(key)}")
                found = True
    if not found:
        lines.append("  none")
    text = "\n".join(lines) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "score_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
