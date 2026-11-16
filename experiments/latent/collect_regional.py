"""Pool the regional run across regions and write a summary the user can copy.

    python experiments/latent/collect_regional.py --root $PMEDM_VB_RUNS/latent/regional

Reads ``<root>/<region>/*.csv`` and writes, under ``--root``:
``summary.txt`` (the tables below, as text), ``pooled_overall.csv``,
``pooled_tables.csv``, ``pooled_rare.csv`` and ``pooled_spread.csv``.

* **Overall, per model:** means over every (region, zone) of the scoring
  tables' TVD, KL and log-odds-ratio errors; misfit to the zone cells
  (RMS z, share within 1 SE, from the zone-level SEs without the prior);
  allocation accuracy (unit-weighted mean over regions); fit time and
  convergence.
* **Per scoring table:** the same errors by table.
* **Rare groups:** predicted over true totals, pooled over all zones; and the
  spread across zones: RMS log(pred/true) over zones, and for each group the
  mean over regions of the predicted share in the zone holding most of its
  truth against that zone's true share.
* **Per region:** allocation accuracy by model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.progress import write_text


def read(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"*/{name}.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    scores, misfit, rare = read(args.root, "scores"), read(args.root, "misfit"), read(args.root, "rare")
    spread, summary, fits = read(args.root, "spread"), read(args.root, "summary"), read(args.root, "fits")
    if scores.empty:
        raise SystemExit(f"no results under {args.root}")

    metrics = [c for c in ("tvd", "kl", "lor_mae", "lor_wrms") if c in scores]
    overall = scores.groupby("model")[metrics].mean()
    overall = overall.join(misfit.groupby("model")[["rms_z", "share_within_1se"]].mean())
    overall = overall.join(summary.groupby("model")["allocation_accuracy"].mean())
    overall = overall.join(fits.groupby("model").agg(
        converged=("converged", "mean"), seconds=("seconds", "mean")))
    tables = scores.groupby(["table", "model"])[metrics].mean().unstack("model")

    pooled = rare.groupby(["table", "category", "model"])[["truth", "pred"]].sum()
    pooled["pred_over_truth"] = pooled["pred"] / pooled["truth"]
    sp = spread.groupby(["table", "category", "model"]).agg(
        rms_log_ratio=("rms_log_ratio", "mean"),
        top_zone_true_share=("top_zone_true_share", "mean"),
        top_zone_pred_share=("top_zone_pred_share", "mean"))
    by_region = summary.pivot_table(index="region", columns="model", values="allocation_accuracy")

    n_regions = scores["region"].nunique()
    n_zones = scores.groupby("region")["zone"].nunique().sum()
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.max_rows", 500):
        text = [
            f"Regional run: {n_regions} regions, {n_zones} zones (PUMAs)\n",
            "1. Overall (means over zones and scoring tables)",
            overall.round(4).to_string(), "",
            "2. Per scoring table: log-odds-ratio error (mean abs)",
            tables["lor_mae"].dropna(how="all").round(3).to_string() if "lor_mae" in tables else "",
            "", "3. Per scoring table: TVD",
            tables["tvd"].round(3).to_string(), "",
            "4. Rare groups: pooled prediction / truth",
            pooled["pred_over_truth"].unstack("model").round(3).to_string(), "",
            "5. Rare groups across zones: RMS log(pred/truth)",
            sp["rms_log_ratio"].unstack("model").round(3).to_string(), "",
            "6. Rare groups: share of the predicted total in the zone holding most of the "
            "truth (true share in the first column)",
            pd.concat([sp["top_zone_true_share"].groupby(level=[0, 1]).first().rename("true"),
                       sp["top_zone_pred_share"].unstack("model")], axis=1).round(3).to_string(),
            "", "7. Allocation accuracy by region",
            by_region.round(3).to_string(), "",
        ]
    write_text(args.root / "summary.txt", "\n".join(text) + "\n")
    write_text(args.root / "pooled_overall.csv", overall.reset_index().to_csv(index=False))
    write_text(args.root / "pooled_tables.csv",
               scores.groupby(["table", "model"])[metrics].mean().reset_index().to_csv(index=False))
    write_text(args.root / "pooled_rare.csv", pooled.reset_index().to_csv(index=False))
    write_text(args.root / "pooled_spread.csv", sp.reset_index().to_csv(index=False))
    print(overall.round(4).to_string())
    print(f"\nfull summary in {args.root / 'summary.txt'}")


if __name__ == "__main__":
    main()
