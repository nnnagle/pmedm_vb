"""Pool the regional run across regions and write a summary the user can copy.

    python experiments/latent/collect_regional.py \
        --root $PMEDM_VB_RUNS/latent/regional $PMEDM_VB_RUNS/latent/lowrank

Reads every ``scores.csv``, ``misfit.csv``, ... below each ``--root`` (the
regional run's ``<region>/``, the low-rank run's ``<config>/<region>/``) and
writes, under ``--out`` (default: the first root): ``summary.txt`` (the tables
below, as text), ``pooled_overall.csv``, ``pooled_tables.csv``,
``pooled_rare.csv``, ``pooled_spread.csv`` and ``pooled_transfer.csv``.
Results without a ``fit_set`` column were fitted to table set A.

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
* **Transfer:** misfit (RMS z, share within 1 SE) of each model fitted to
  table set A or B, scored on the cells of A and of B; the off-diagonal
  entries are the transfer test.
* **Latent views:** R² per table of the latent-view PMEDM's views
  (``views_r2.csv`` of a root), if present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.progress import write_text


def read(roots: list[Path], name: str) -> pd.DataFrame:
    paths = sorted(p for root in roots for p in root.rglob(f"{name}.csv")
                   if p.parent != root)
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)
    if "model" in frame:
        frame = frame.drop_duplicates()
    if "fit_set" not in frame:
        frame["fit_set"] = "A"
    frame["fit_set"] = frame["fit_set"].fillna("A")
    return frame


def transfer_table(misfit: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """RMS z and share within 1 SE by model family, fitted set -> scored set."""
    own = misfit.assign(scored_set=misfit["fit_set"])
    both = pd.concat([own, transfer], ignore_index=True) if not transfer.empty else own
    both["family"] = both["model"].str.replace(r"@[A-Z]$", "", regex=True)
    both["fit -> scored"] = both["fit_set"] + " -> " + both["scored_set"]
    return both.groupby(["family", "fit -> scored"])[["rms_z", "share_within_1se"]].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root[0]
    roots = args.root
    scores, misfit, rare = read(roots, "scores"), read(roots, "misfit"), read(roots, "rare")
    spread, summary, fits = read(roots, "spread"), read(roots, "summary"), read(roots, "fits")
    transfer = read(roots, "transfer")
    if scores.empty:
        raise SystemExit(f"no results under {roots}")

    metrics = [c for c in ("tvd", "kl", "lor_mae", "lor_wrms") if c in scores]
    overall = scores.groupby("model")[metrics].mean()
    overall.insert(0, "fit_set", scores.groupby("model")["fit_set"].first())
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
    moved = transfer_table(misfit, transfer)
    view_paths = [r / "views_r2.csv" for r in roots if (r / "views_r2.csv").exists()]
    views = (pd.read_csv(view_paths[0]).pivot_table(index="table", columns="table_set",
                                                    values="r2", sort=False)
             if view_paths else None)

    n_regions = scores["region"].nunique()
    n_zones = scores.groupby("region")["zone"].nunique().sum()
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.max_rows", 500):
        text = [
            f"Regional run: {n_regions} regions, {n_zones} zones (PUMAs); "
            f"{scores['model'].nunique()} models\n",
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
            "8. Transfer: misfit to each table set's cells (RMS z; share within 1 SE), "
            "by the set fitted",
            moved["rms_z"].unstack("fit -> scored").round(3).to_string(), "",
            moved["share_within_1se"].unstack("fit -> scored").round(3).to_string(), "",
        ]
        if views is not None:
            text += ["9. Latent views: R² of each table's cells on the decoder features",
                     views.round(3).to_string(), ""]
    write_text(out / "summary.txt", "\n".join(text) + "\n")
    write_text(out / "pooled_overall.csv", overall.reset_index().to_csv(index=False))
    write_text(out / "pooled_tables.csv",
               scores.groupby(["table", "model"])[metrics].mean().reset_index().to_csv(index=False))
    write_text(out / "pooled_rare.csv", pooled.reset_index().to_csv(index=False))
    write_text(out / "pooled_spread.csv", sp.reset_index().to_csv(index=False))
    write_text(out / "pooled_transfer.csv", moved.reset_index().to_csv(index=False))
    print(overall.round(4).to_string())
    print(f"\nfull summary in {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
