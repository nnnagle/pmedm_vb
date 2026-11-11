"""Which cells short HMC's intervals are much narrower than the reference's in.

Reads the scoring's per-cell detail (``compare_methods.py --detail``) of
``hmc_short`` and ``hmc_ref`` for one experiment, tag and alpha, every PUMA,
the constraint cells' leaves (totals and subtotals left out, as Table 4 does),
and calls a cell *narrow* when short HMC's 90% interval is under ``--narrow``
times the reference's (``width_ratio_ref``). Writes small CSVs to ``--out``
and prints them:

``width_overall.csv``   per level: cells, the share narrow, quantiles of the ratio
``width_by_table.csv``  per level and ACS table: cells, narrow cells, share narrow
``width_by_size.csv``   per level and published count (zero, 1-9, 10-49, 50-199, 200+)
``width_by_puma.csv``   per level and PUMA
``width_contrast.csv``  narrow cells against the rest: medians of the published
                        count, the reference's and short HMC's half-width over
                        MOE, the sd ratio and the median difference
``width_lines.csv``     the table lines with the most narrow cells

    python experiments/width_check.py --root $R --out $R/width_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["puma", "subset", "area", "cell"]
LEVELS = {"constrained_block_group": "block group", "constrained_tract": "tract"}
SIZES = [-np.inf, 0, 9, 49, 199, np.inf]
SIZE_LABELS = ["zero", "1-9", "10-49", "50-199", "200+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiment", default="exp01_baseline")
    parser.add_argument("--tag", default="paper2")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--narrow", type=float, default=0.1)
    parser.add_argument("--lines", type=int, default=40, help="lines in width_lines.csv")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read(detail: Path, method: str, alpha: float, columns: list[str]) -> pd.DataFrame:
    frames = []
    for folder in sorted(detail.glob(f"*_a{alpha:g}")):
        path = folder / f"{method}_cells.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=columns)
            frames.append(frame[frame.subset.isin(LEVELS)])
    if not frames:
        raise SystemExit(f"no {method}_cells.parquet under {detail}/*_a{alpha:g}")
    return pd.concat(frames, ignore_index=True)


def share(group: pd.DataFrame) -> pd.Series:
    return pd.Series({"cells": len(group), "narrow": int(group.narrow.sum()),
                      "share_narrow": group.narrow.mean()})


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detail = args.root / args.experiment / "scores" / args.tag / "detail"
    short = read(detail, "hmc_short", args.alpha,
                 KEYS + ["published", "sampled", "halfwidth_over_moe", "width_ratio_ref",
                         "sd_ratio_ref", "median_diff_ref_sd"])
    ref = read(detail, "hmc_ref", args.alpha, KEYS + ["halfwidth_over_moe"])
    cells = short.merge(ref.rename(columns={"halfwidth_over_moe": "ref_halfwidth_over_moe"}),
                        on=KEYS, how="left")
    cells["puma"] = cells.puma.astype(str)
    cells = cells[~cells.cell.astype(str).str.endswith(":")]          # leaves only
    cells["level"] = cells.subset.map(LEVELS)
    cells["table"] = cells.cell.astype(str).str.split(".").str[0]
    cells["size"] = pd.cut(cells.published, SIZES, labels=SIZE_LABELS)
    cells["narrow"] = cells.width_ratio_ref < args.narrow

    out = {}
    q = cells.groupby("level").width_ratio_ref.quantile([0.01, 0.05, 0.1, 0.25, 0.5]).unstack()
    q.columns = [f"ratio_p{int(100 * c)}" for c in q.columns]
    out["overall"] = cells.groupby("level").apply(share).join(q).reset_index()
    out["by_table"] = (cells.groupby(["level", "table"]).apply(share).reset_index()
                       .sort_values(["level", "share_narrow"], ascending=[True, False]))
    out["by_size"] = cells.groupby(["level", "size"], observed=True).apply(share).reset_index()
    out["by_puma"] = cells.groupby(["level", "puma"]).apply(share).reset_index()
    out["contrast"] = (cells.groupby(["level", "narrow"])
                       [["published", "ref_halfwidth_over_moe", "halfwidth_over_moe",
                         "sd_ratio_ref", "median_diff_ref_sd"]].median()
                       .join(cells.groupby(["level", "narrow"]).sampled.mean()
                             .rename("share_sampled"))
                       .reset_index())
    lines = cells.groupby(["level", "cell"]).apply(share).reset_index()
    out["lines"] = lines[lines.narrow > 0].sort_values("narrow", ascending=False).head(args.lines)

    print(f"{args.experiment}, tag {args.tag}, alpha {args.alpha:g}: narrow = width ratio "
          f"< {args.narrow:g}; {len(cells):,} leaf cells")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.max_rows", 200):
        for name, frame in out.items():
            frame = frame.astype({c: int for c in ("cells", "narrow") if c in frame})
            frame.to_csv(args.out / f"width_{name}.csv", index=False)
            print(f"\n== {name}\n{frame.round(3).to_string(index=False)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
