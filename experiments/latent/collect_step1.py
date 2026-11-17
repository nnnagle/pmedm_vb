"""Pool step 1's cross-fitted reproduction check and apply the §6 decision rules
(``PLAN_latent_step1.md``).

    python experiments/latent/collect_step1.py --root $PMEDM_VB_RUNS/latent/step1

Reads ``<root>/grid/<basis>/<puma>/{fits,cells,tables,rare}.csv`` (finished
units only), ``<root>/inputs/`` (atom, distance and basis checks) and, if
present, ``<root>/final/``. Writes under ``<root>``:

* ``summary.txt``: the inputs' checks; per setting (basis, γ, σ_θ, and the
  baselines) the §4.6 metrics -- RMS z and share within 1 SE by cell group
  (S1, B, T) and over all cells, TVD and log-odds-ratio error of the two-way
  scoring tables, the rare groups' RMS log ratio and the pooled AIAN and NH/PI
  ratios, Kish ESS overall and within the rare groups, share of ``q^P``
  outside the PUMA -- and the fits' diagnostics; the rare groups one by one at
  each basis's best setting and the baselines; the §6 decisions; timings; and
  the final fits with the secondary check when they exist.
* ``settings.csv`` (the per-setting table), ``rare_groups.csv`` (every
  setting and rare group), ``timing.csv`` and ``decisions.json`` (read by
  ``step1_reference.py final``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.latent.reference_check import RARE_STEP1, decide, pool_rare, pool_settings
from pmedm_vb.progress import write_text


def read(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(p / f"{name}.csv" for p in (root / "grid").glob("*/*")
                   if (p / "done.json").exists())
    frames = [pd.read_csv(p, dtype={"puma": str}) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    fits, cells = read(root, "fits"), read(root, "cells")
    tables, rare = read(root, "tables"), read(root, "rare")
    if fits.empty:
        raise SystemExit(f"no finished units under {root / 'grid'}")
    bases = {p.stem: json.loads(p.read_text()) for p in (root / "inputs" / "bases").glob("*.json")}
    K = {name: b["K"] for name, b in bases.items()}
    settings = pool_settings(fits, cells, tables, rare, K)
    per_rare, _ = pool_rare(rare)
    decision, decision_lines = decide(settings)

    done = [json.loads(p.read_text()) | {"basis": p.parent.parent.name, "puma": p.parent.name}
            for p in (root / "grid").glob("*/*/done.json")]
    units = pd.DataFrame(done)
    timing = fits.groupby(["basis", "gamma"], dropna=False)["seconds"].agg(
        ["count", "mean", "max", "sum"]).rename(columns={"count": "fits"})
    unit_time = units.groupby("basis")["seconds"].agg(["count", "mean", "sum"]).rename(
        columns={"count": "units"})
    n_pumas = fits["puma"].nunique()

    atoms = json.loads((root / "inputs" / "atoms.json").read_text())
    dist = json.loads((root / "inputs" / "distances.json").read_text())
    basis_table = pd.DataFrame([{k: b.get(k) for k in ("K", "K_raw", "n_dropped",
                                                        "condition_cov", "ell")}
                                | {"basis": name} for name, b in bases.items()]).set_index("basis")

    rare_ids = pd.MultiIndex.from_tuples(RARE_STEP1, names=["table", "category"])
    best = settings[~settings["basis"].str.startswith("baseline")]
    best = best.loc[best.groupby("basis")["rms_z_all"].idxmin()]
    shown = pd.concat([best, settings[settings["basis"].str.startswith("baseline")]])
    def label(basis, gamma, sigma) -> str:
        return basis if np.isnan(sigma) else f"{basis} g{gamma:g} s{sigma:g}"

    wanted = {label(r.basis, r.gamma, r.sigma) for r in shown.itertuples()}
    pr = per_rare.copy()
    pr["setting"] = [label(b, g, s) for b, g, s in zip(pr["basis"], pr["gamma"], pr["sigma"])]
    pr = pr[pr["setting"].isin(wanted)].set_index(["table", "category"])

    cols = ["basis", "K", "gamma", "sigma", "rms_z_all", "within_1_all", "rms_z_S1", "rms_z_B",
            "rms_z_T", "tvd", "lor_mae", "rms_log_ratio_rare", "rms_log_ratio_fold_rare", "pooled_nh_aian",
            "pooled_nh_nhpi", "ess", "rare_ess", "share_outside", "iterations", "converged",
            "seconds"]
    cols = [c for c in cols if c in settings]
    with pd.option_context("display.width", 250, "display.max_columns", 60,
                           "display.max_rows", 500):
        text = [
            f"Step 1 reproduction check: {n_pumas} PUMAs, {fits['fold'].nunique()} folds, "
            f"{len(units)} finished units, {len(fits):,} fits\n",
            "0. Inputs",
            f"  atoms: M = {atoms['M']:,} (d = {atoms['d']}); z weighted mean max |.| "
            f"{atoms['z_mean_max_abs']:.2e}; cov max |. - I| {atoms['z_cov_max_abs_dev_from_I']:.2e}",
            f"  weights: households {atoms['weight_household']:,.0f} "
            f"({atoms['n_household']:,} units), institutional GQ "
            f"{atoms['weight_institutional']:,.0f} ({atoms['n_institutional']:,}), "
            f"noninstitutional GQ {atoms['weight_noninstitutional']:,.0f} "
            f"({atoms['n_noninstitutional']:,})",
            f"  distances: {dist}",
            basis_table.to_string(), "",
            "1. Per setting (held-out fold scores pooled over PUMAs and folds)",
            settings[cols].round(4).to_string(index=False), "",
            "2. Rare groups at each basis's best setting and the baselines: pooled "
            "predicted / held-out",
            pr["pooled_ratio"].reset_index().pivot_table(
                index=["table", "category"], columns=pr["setting"].to_numpy(),
                values="pooled_ratio", sort=False).reindex(rare_ids).round(3).to_string(), "",
            "3. Rare groups: RMS over PUMAs of log(predicted / held-out)",
            pr["rms_log_ratio"].reset_index().pivot_table(
                index=["table", "category"], columns=pr["setting"].to_numpy(),
                values="rms_log_ratio", sort=False).reindex(rare_ids).round(3).to_string(), "",
            "3b. Rare groups: RMS over (PUMA, fold) of the per-fold log(predicted / held-out) "
            "(pairs with a zero left out)",
            pr["rms_log_ratio_fold"].reset_index().pivot_table(
                index=["table", "category"], columns=pr["setting"].to_numpy(),
                values="rms_log_ratio_fold", sort=False).reindex(rare_ids).round(3).to_string(),
            "",
            "4. Rare groups: mean Kish ESS of q^P within the group",
            pr["mean_ess"].reset_index().pivot_table(
                index=["table", "category"], columns=pr["setting"].to_numpy(),
                values="mean_ess", sort=False).reindex(rare_ids).round(1).to_string(), "",
            "5. Decisions (section 6)", *decision_lines, "",
            "6. Timing: seconds per fit by basis and gamma; units",
            timing.round(3).to_string(), unit_time.round(1).to_string(),
            f"  total fit CPU hours {fits['seconds'].sum() / 3600:.2f}; "
            f"total unit CPU hours {units['seconds'].sum() / 3600:.2f}", "",
        ]
        final = root / "final" / "fits.csv"
        if final.exists():
            ff = pd.read_csv(final, dtype={"puma": str})
            fr = pd.read_csv(root / "final" / "rare.csv", dtype={"puma": str})
            st = json.loads((root / "final" / "settings.json").read_text())
            g = fr.groupby(["table", "category"])
            fin = pd.DataFrame({"pooled_pred_over_truth": g["pred"].sum() / g["truth"].sum()})
            if "z_pred" in fr:
                fin["rms_z_pred_published"] = g["z_pred"].apply(
                    lambda z: float(np.sqrt(np.nanmean(np.square(z)))))
                fin["rms_moe_units_pred"] = g["moe_units_pred"].apply(
                    lambda z: float(np.sqrt(np.nanmean(np.square(z)))))
                fin["rms_z_truth_published"] = g["z_truth"].apply(
                    lambda z: float(np.sqrt(np.nanmean(np.square(z)))))
            text += [f"7. Final fits on all records: h = {st['h']}, gamma = {st['gamma']:g}, "
                     f"sigma = {st['sigma']:g}; secondary check "
                     f"{'run' if st.get('secondary_check') else 'skipped (tables not cached)'}",
                     ff[["iterations", "converged", "grad_max", "ess", "share_outside",
                         "max_moment_error", "seconds"]].describe().round(4).to_string(),
                     "  rare groups, in sample (and against the published PUMA tables, "
                     "SE and MOE units, when run)",
                     fin.reindex(pd.MultiIndex.from_tuples(
                         [x for x in fin.index], names=["table", "category"])).round(3).to_string(),
                     ""]
    write_text(root / "summary.txt", "\n".join(text) + "\n")
    write_text(root / "settings.csv", settings.to_csv(index=False))
    write_text(root / "rare_groups.csv", per_rare.to_csv(index=False))
    write_text(root / "timing.csv", timing.reset_index().to_csv(index=False))
    write_text(root / "decisions.json", json.dumps(decision, indent=1))
    print("\n".join(decision_lines))
    print(f"\nfull summary in {root / 'summary.txt'}")


if __name__ == "__main__":
    main()
