"""Condense the comparison grid's long CSV into a few readable tables.

Reads ``compare_methods.py``'s combined CSV (``all_scores.csv``) and writes
``<out>/score_summary.txt``: for each alpha, one row per method and one column
per headline metric, each cell the median over PUMAs with its range. Also lists
the (subset, metric, stat) rows that some (method, PUMA, alpha) cells lack and
others have, so a short cell can be explained. The long CSV stays the source
for tables and figures in R::

    $CONDA_PREFIX/bin/python experiments/score_summary.py \\
        --scores $RUNS/scores/all_scores.csv --out $RUNS/scores
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
    parser.add_argument("--scores", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
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
