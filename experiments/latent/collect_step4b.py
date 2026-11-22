"""Pool step 4b's arms and apply its decision rule (``PLAN_latent_step4b.md``
version 1, §5).

    python experiments/latent/collect_step4b.py --root $PMEDM_VB_RUNS/latent/step4b \
        --step4 $PMEDM_VB_RUNS/latent/step4 --step3b $PMEDM_VB_RUNS/latent/step3b

Reads ``step4b_overdispersion.py``'s units: arm ``K1`` (``<root>/K1/``) and
arm ``K2`` assembled per simulation and region from the ``c`` in
``<root>/K2_choice.json`` (``<root>/K2_c<c>/``, or ``K1`` for ``c = 1``), and
step 4's run as the ``κ = ∞`` arm (``--step4``, reported beside). Writes,
under the root, ``summary.txt``, ``decision.json``, ``calibration.csv`` (by
arm, simulation and functional class), ``pit.csv`` and ``map_accuracy.csv``.

**Calibration** per arm, as step 4's collector
(:func:`~pmedm_vb.latent.calibration.summarize`), with the zone totals as the
class ``total``. **Decision (§5):** each arm judged by step 4's §6 bands on
simulation S (G the diagnostic): an arm is calibrated when every functional
class except rare groups is within [0.85, 0.95] at 90% and [0.42, 0.58] at
50% (step 4 §6; ``total`` counts as a class); rare groups are reported
separately. K1 calibrated: adopt K1 (model version 4, ``κ = n_A``); else K2
calibrated: adopt K2 with ``c`` chosen by B1; else stop and review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.latent.calibration import decide, pit_histogram, summarize
from pmedm_vb.progress import write_text

FILES = ("posterior.csv.gz", "scores.csv", "rare.csv", "misfit.csv", "transfer.csv",
         "tau.csv")


def read_units(dirs: list[Path], name: str) -> pd.DataFrame:
    paths = sorted(p for d in dirs for p in d.glob(name))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)


def arm_dirs(root: Path) -> dict[str, list[Path]]:
    """Unit directories of each arm: K1, K2 (chosen c per simulation and region)."""
    out = {"K1": sorted((root / "K1").glob("*/*/r*"))}
    path = root / "K2_choice.json"
    if path.exists():
        choice = json.loads(path.read_text())
        dirs = []
        for key, val in choice.items():
            sim, rid = key.split("/")
            c = float(val["c"])
            label = "K1" if c == 1.0 else f"K2_c{c:g}"
            dirs += sorted((root / label / sim / rid).glob("r*"))
        out["K2"] = dirs
    return out


def accuracy(frames: dict) -> pd.DataFrame:
    s, rare = frames["scores.csv"], frames["rare.csv"]
    keys = ["simulation", "model"]
    acc = s.groupby(keys)[["lor_mae", "tvd"]].mean()
    ok = rare[(rare["truth"] > 0) & (rare["pred"] > 0)]
    acc = acc.join(np.sqrt((np.log(ok["pred"] / ok["truth"]) ** 2)
                           .groupby([ok[k] for k in keys]).mean()).rename("rare_rms"))
    acc = acc.join(frames["misfit.csv"].groupby(keys)["rms_z"].mean().rename("misfit_rms_z"))
    t = frames["transfer.csv"]
    for k in ("B1", "B2"):
        acc = acc.join(t[t["scored_set"] == k].groupby(keys)["rms_z"].mean()
                       .rename(f"{k.lower()}_rms_z"))
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--step4", type=Path, default=None)
    parser.add_argument("--step3b", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    arms = arm_dirs(root)
    if args.step4 is not None:
        arms["inf (step 4)"] = sorted(args.step4.glob("*/*/r*"))
    tables, pits, accs, verdicts, lines = [], [], [], {}, []
    for arm, dirs in arms.items():
        if not dirs:
            continue
        frames = {name: read_units(dirs, name) for name in FILES}
        post = frames["posterior.csv.gz"]
        if post.empty:
            continue
        for col in ("cover50", "cover90", "below_zero"):
            post[col] = post[col].astype(bool)
        t = summarize(post)
        tables.append(t.assign(arm=arm).set_index("arm", append=True))
        pits.append(pit_histogram(post).assign(arm=arm))
        accs.append(accuracy(frames).assign(arm=arm))
        verdict, arm_lines = decide(t)
        verdicts[arm] = verdict
        lines += [f"Arm {arm}:", *(f"  {x}" for x in arm_lines)]
    calibrated = {a: v["per_simulation"].get("S", {}).get("others_pass", False)
                  for a, v in verdicts.items() if a in ("K1", "K2")}
    if calibrated.get("K1"):
        decision = "adopt K1: model version 4 with kappa = n_A"
    elif calibrated.get("K2"):
        decision = "adopt K2: model version 4 with kappa = c n_A, c chosen by B1"
    else:
        decision = "stop and review: neither K1 nor K2 is calibrated on simulation S"
    rare = {a: v["per_simulation"].get("S", {}).get("rare_pass") for a, v in verdicts.items()}
    lines += ["", "Step 4b decision (section 5; S primary, step 4's section 6 bands, every class "
              "except rare groups): " + ", ".join(f"{a} {'calibrated' if v else 'not calibrated'}"
                                                   for a, v in calibrated.items()),
              "  rare groups on S: " + ", ".join(f"{a} {'pass' if v else 'fail'}"
                                                  for a, v in rare.items() if v is not None),
              f"DECISION: {decision}"]
    table = pd.concat(tables).reorder_levels(["arm", "simulation", "cls"]).sort_index()
    pit = pd.concat(pits)
    acc = pd.concat(accs).set_index("arm", append=True)
    choice = json.loads((root / "K2_choice.json").read_text()) \
        if (root / "K2_choice.json").exists() else {}
    tau = read_units(arms.get("K1", []) + arms.get("K2", []), "tau.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 1000):
        text = ["Step 4b: overdispersed realized level (model version 4)", "", *lines, "",
                "1. Calibration by arm, simulation and class", table.round(3).to_string(), "",
                "2. PIT histograms", pit.round(3).to_string(), "",
                "3. MAP point accuracy (realized MAP at the B1-chosen tau)",
                acc.round(4).to_string(), ""]
        if args.step3b is not None and (args.step3b / "pooled_models.csv").exists():
            pm = pd.read_csv(args.step3b / "pooled_models.csv").set_index("model")
            if "latent_kern30" in pm.index:
                text += ["   step 3b, noise-free latent_kern30:",
                         pm.loc[["latent_kern30"]].round(4).to_string(), ""]
        if choice:
            text += ["4. K2: c per simulation and region (B1 predictive log density summed "
                     "over the selection replicates)",
                     pd.DataFrame({k: {"c": v["c"], **{f"logdens_c{c}": d for c, d in
                                                      v["b1_logdens"].items()}}
                                   for k, v in choice.items()}).T.to_string(
                         float_format=lambda x: f"{x:.6g}"), ""]
        if len(tau):
            ch = tau[tau["chosen_b1"].astype(bool)]
            text += ["5. tau chosen by B1 (multiples of tau_ref^g), by arm and simulation",
                     ch.groupby(["arm", "simulation", "tau_multiple"]).size()
                     .unstack(fill_value=0).to_string(), ""]
        done = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/*/r*/done.json"))]
        if done:
            d = pd.DataFrame(done)
            text += ["6. Timings by arm directory (seconds per unit)",
                     d.groupby("arm")["seconds"].agg(["count", "median", "max", "sum"])
                     .round(0).to_string(),
                     f"   total {d['seconds'].sum() / 3600:.1f} CPU-hours; peak memory "
                     f"{d['peak_gb'].max():.1f} GB", ""]
    write_text(root / "summary.txt", "\n".join(text) + "\n")
    write_text(root / "decision.json", json.dumps({
        "decision": decision, "calibrated": calibrated, "rare_pass": rare,
        "per_arm": verdicts, "K2_choice": choice}, indent=1, default=str))
    write_text(root / "calibration.csv", table.reset_index().to_csv(index=False))
    write_text(root / "pit.csv", pit.reset_index().to_csv(index=False))
    write_text(root / "map_accuracy.csv", acc.reset_index().to_csv(index=False))
    print("\n".join(lines))
    print(f"\nfull summary in {root / 'summary.txt'}")


if __name__ == "__main__":
    main()
