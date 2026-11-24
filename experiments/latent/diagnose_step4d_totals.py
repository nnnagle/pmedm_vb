"""Step 4d review: where the zone totals' miss comes from (two diagnostics).

    python experiments/latent/diagnose_step4d_totals.py --root $PMEDM_VB_RUNS/latent/step4d \
        --step4c $PMEDM_VB_RUNS/latent/step4c

Reads only existing outputs: the ``posterior.csv.gz`` of every unit of FS
(step 4d) and of S and P0 (step 4c; S pooled as ``collect_step4c.py`` pools
it). All three share S's survey draws, so the direct estimates are the same.

1. **Direct coverage of the survey's zone totals.** The direct estimate is
   the sum of the zone's fitted ``unit_type:*`` cells (the Horvitz–Thompson
   unit total). Its variance is the sum of those cells' variances: under
   simulation S's Bernoulli design each unit is in exactly one unit type, so
   the estimated covariances between them are exactly zero. Coverage of
   ``±1.645`` and ``±0.674`` direct SEs against the true total.
2. **Decomposition of each arm's error.** For the posterior mean of the zone
   total, ``error = survey error + pull``, with ``survey error = direct -
   truth`` and ``pull = posterior mean - direct``; all relative to the truth.
   Reported: RMS of each, their correlation, the slope of the pull on the
   survey error, and the pull's persistence: the share of its variance that
   is between zones (zone means over replicates). Each replicate redraws the
   truth and the sample, so a pull that persists by zone is systematic: it
   comes from the zone itself (its composition against the model), not from
   that replicate's survey error.
3. **The pull split in two** (added after the first run): the direct
   estimates rescaled so that the region's zones sum to the known ``N_A``
   (the true region total in the simulation) are the constraint's own
   estimator, with no model in it. ``pull = constraint part (rescaled -
   direct) + model part (posterior mean - rescaled)``. The rescaled
   estimator's error is reported beside the direct one.

Writes ``<root>/diagnostics/totals_by_unit.csv`` and
``<root>/diagnostics/totals_summary.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_step4c import unit_dirs
from pmedm_vb.progress import write_text

Z90, Z50 = 1.645, 0.6745
COLUMNS = ("simulation", "region", "replicate", "zone", "cls", "functional", "truth", "mean",
           "sd", "cover90", "cover50", "published", "se")


def unit_rows(d: Path, label: str) -> pd.DataFrame:
    f = pd.read_csv(d / "posterior.csv.gz", dtype={"zone": str},
                    usecols=lambda c: c in COLUMNS)
    unit = f[f["functional"].str.startswith("unit_type:") & (f["cls"] == "fitted")]
    direct = unit.groupby("zone").agg(direct=("published", "sum"),
                                      direct_var=("se", lambda s: float(np.sum(s ** 2))))
    tot = f[f["functional"] == "units:total"].set_index("zone")
    out = tot[["region", "replicate", "truth", "mean", "sd", "cover90", "cover50"]].join(direct)
    out["arm"] = label
    return out.reset_index()


def persistence(frame: pd.DataFrame, column: str) -> float:
    """Share of ``column``'s variance between zones (zone means over replicates)."""
    zone_means = frame.groupby(["region", "zone"])[column].transform("mean")
    total = float(np.var(frame[column]))
    return float(np.var(zone_means)) / total if total > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="step 4d output root")
    parser.add_argument("--step4c", required=True, type=Path, help="step 4c output root")
    args = parser.parse_args()
    choice = json.loads((args.step4c / "choice.json").read_text())
    g4c = unit_dirs(args.step4c, choice)
    groups = {"FS": sorted(args.root.glob("FS/*/r*")), "S": g4c.get("S", []),
              "P0": g4c.get("P0", [])}
    frames = [unit_rows(d, label) for label, dirs in groups.items() for d in dirs
              if (d / "posterior.csv.gz").exists()]
    f = pd.concat(frames, ignore_index=True)
    f["direct_se"] = np.sqrt(f["direct_var"])
    f["survey_err"] = (f["direct"] - f["truth"]) / f["truth"]
    f["pull"] = (f["mean"] - f["direct"]) / f["truth"]
    f["err"] = (f["mean"] - f["truth"]) / f["truth"]
    f["sd_rel"] = f["sd"] / f["truth"]
    f["direct_se_rel"] = f["direct_se"] / f["truth"]
    f["direct_z"] = (f["direct"] - f["truth"]) / f["direct_se"]
    unit = f.groupby(["arm", "region", "replicate"])
    f["rescaled"] = f["direct"] * unit["truth"].transform("sum") / unit["direct"].transform("sum")
    f["rescaled_err"] = (f["rescaled"] - f["truth"]) / f["truth"]
    f["constraint_part"] = (f["rescaled"] - f["direct"]) / f["truth"]
    f["model_part"] = (f["mean"] - f["rescaled"]) / f["truth"]
    for col in ("cover90", "cover50"):
        f[col] = f[col].astype(bool)

    # 1. direct coverage: the survey draws are common to the arms; FS's copy
    d = f[f["arm"] == "FS"] if (f["arm"] == "FS").any() else f[f["arm"] == "S"]
    direct = pd.DataFrame({
        "n": [len(d)],
        "cover90": [float((np.abs(d["direct_z"]) <= Z90).mean())],
        "cover50": [float((np.abs(d["direct_z"]) <= Z50).mean())],
        "rms_z": [float(np.sqrt(np.mean(d["direct_z"] ** 2)))],
        "rms_rel_error": [float(np.sqrt(np.mean(d["survey_err"] ** 2)))],
        "mean_rel_se": [float(d["direct_se_rel"].mean())],
        "mean_rel_error": [float(d["survey_err"].mean())]}, index=["direct (HT)"])
    direct.loc["rescaled to N_A", ["n", "rms_rel_error", "mean_rel_error"]] = [
        len(d), float(np.sqrt(np.mean(d["rescaled_err"] ** 2))),
        float(d["rescaled_err"].mean())]
    by_region_direct = d.assign(inside=np.abs(d["direct_z"]) <= Z90,
                                z2=d["direct_z"] ** 2).groupby("region").agg(
        n=("inside", "size"), cover90=("inside", "mean"), mean_z2=("z2", "mean"))
    by_region_direct["rms_z"] = np.sqrt(by_region_direct.pop("mean_z2"))

    # 2. decomposition per arm
    rows, split = {}, {}
    for arm, g in f.groupby("arm"):
        slope = float(np.polyfit(g["survey_err"], g["pull"], 1)[0])
        rows[arm] = {
            "n": len(g), "cover90": float(g["cover90"].mean()),
            "cover50": float(g["cover50"].mean()),
            "rms_error": float(np.sqrt(np.mean(g["err"] ** 2))),
            "rms_survey_error": float(np.sqrt(np.mean(g["survey_err"] ** 2))),
            "rms_pull": float(np.sqrt(np.mean(g["pull"] ** 2))),
            "mean_pull": float(g["pull"].mean()),
            "corr_survey_pull": float(np.corrcoef(g["survey_err"], g["pull"])[0, 1]),
            "slope_pull_on_survey": slope,
            "pull_between_zone_share": persistence(g, "pull"),
            "survey_err_between_zone_share": persistence(g, "survey_err"),
            "mean_sd_rel": float(g["sd_rel"].mean()),
            "median_sd_over_direct_se": float((g["sd"] / g["direct_se"]).median())}
        split[arm] = {
            "rms_error": float(np.sqrt(np.mean(g["err"] ** 2))),
            "rms_rescaled_error": float(np.sqrt(np.mean(g["rescaled_err"] ** 2))),
            "rms_constraint_part": float(np.sqrt(np.mean(g["constraint_part"] ** 2))),
            "rms_model_part": float(np.sqrt(np.mean(g["model_part"] ** 2))),
            "corr_model_part_rescaled_err": float(np.corrcoef(g["rescaled_err"],
                                                              g["model_part"])[0, 1]),
            "slope_model_part_on_rescaled_err": float(
                np.polyfit(g["rescaled_err"], g["model_part"], 1)[0]),
            "model_part_between_zone_share": persistence(g, "model_part"),
            "rescaled_err_between_zone_share": persistence(g, "rescaled_err")}
    order = [a for a in ("FS", "S", "P0") if a in rows]
    decomposition = pd.DataFrame(rows).T.reindex(order)
    split_table = pd.DataFrame(split).T.reindex(order)

    out = args.root / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "totals_by_unit.csv", f.to_csv(index=False))
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 200):
        text = ["Step 4d review: zone totals", "",
                "1. Direct coverage of the survey's zone totals (HT, SE from the unit-type "
                "cells' variances)", direct.round(4).to_string(), "",
                "   by region", by_region_direct.round(3).to_string(), "",
                "2. Each arm's posterior mean: error = survey error + pull (relative to the "
                "truth); persistence = share of variance between zones",
                decomposition.round(4).to_string(), "",
                "3. The pull split: constraint part (direct rescaled to N_A - direct) and "
                "model part (posterior mean - rescaled)", split_table.round(4).to_string(), ""]
    write_text(out / "totals_summary.txt", "\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
