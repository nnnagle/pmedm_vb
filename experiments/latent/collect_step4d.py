"""Pool step 4d's free-shares arm and apply its decision rule
(``PLAN_latent_step4d.md`` version 1, §3).

    python experiments/latent/collect_step4d.py --root $PMEDM_VB_RUNS/latent/step4d \
        --step4c $PMEDM_VB_RUNS/latent/step4c

Reads ``step4d_free_shares.py``'s units (``<root>/FS/<region>/r<k>/``) and,
for comparison, step 4c's S (the selection replicates at S's chosen pair and
the final replicates, as ``collect_step4c.py`` pools them) and P0. Writes,
under the root, ``summary.txt``, ``decision.json``, ``calibration.csv``,
``pit.csv``, ``map_accuracy.csv`` and ``totals.csv``.

* **Calibration** as step 4c (:func:`~pmedm_vb.latent.calibration.summarize`):
  classes fitted, B1, B2, joint, rare and the zone totals, for FS, S and P0.
* **Decision** (§3): FS is primary; every class within step 4c's bands
  (:func:`~pmedm_vb.latent.calibration.decide_bands`); S and P0 reported.
* **Zone totals**: the MAP's, the survey's and the realized MAP's relative
  RMS errors against the true totals, and the posterior sd relative to the
  truth, from each unit's ``done.json``.
* **Fit diagnostics**: MAP convergence and the spread of its two starts, the
  realized MAP's outer iterations, compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from collect_step4c import FILES, accuracy, read, unit_dirs
from pmedm_vb.latent.calibration import decide_bands, pit_histogram, summarize
from pmedm_vb.progress import write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="step 4d output root")
    parser.add_argument("--step4c", required=True, type=Path, help="step 4c output root")
    args = parser.parse_args()
    choice = json.loads((args.step4c / "choice.json").read_text())
    groups4c = unit_dirs(args.step4c, choice)
    groups = {"FS": sorted(args.root.glob("FS/*/r*"))}
    for label in ("S", "P0"):
        if groups4c.get(label):
            groups[label] = groups4c[label]
    tables, pits, accs = [], [], []
    for label, dirs in groups.items():
        frames = {name: read(dirs, name) for name in FILES}
        post = frames["posterior.csv.gz"]
        if post.empty:
            continue
        post = post.assign(simulation=label)
        for col in ("cover50", "cover90", "below_zero"):
            post[col] = post[col].astype(bool)
        tables.append(summarize(post))
        pits.append(pit_histogram(post))
        accs.append(accuracy(frames, label))
    table = pd.concat(tables)
    verdict, lines = decide_bands(table, primary="FS")
    lines = lines[:-1]                       # step 4c's decision line; 4d's below
    fails = [c for c, v in verdict["per_simulation"].get("FS", {}).get("classes", {}).items()
             if not v]
    n_expected = sum(1 for k in choice if k.startswith("S/")) * 10
    done = [json.loads((d / "done.json").read_text()) for d in groups["FS"]
            if (d / "done.json").exists()]
    if len(done) < n_expected:
        decision = f"incomplete ({len(done)} of {n_expected} units)"
    elif fails:
        decision = f"stop and review (failing on FS: {', '.join(fails)})"
    else:
        decision = ("calibrated: model version 5 (free shares) with step 4c's (tau, c) "
                    "is the posterior; merge the draft into MODEL_latent.qmd and go to step 6")
    lines.append(f"DECISION (step 4d section 3, arm FS): {decision}")

    d = pd.DataFrame(done)
    tot = pd.DataFrame([{"region": u["region"], "replicate": u["replicate"],
                         **{f"rel_rms_{k}": v for k, v in u["totals_rel_rms"].items()},
                         "posterior_sd_rel": u["totals_posterior_sd_rel_mean"]} for u in done])
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 1000):
        text = ["Step 4d: free shares (model version 5 draft) on simulation S", "", *lines, "",
                "1. Calibration (FS: all replicates at S's chosen pair; S and P0 from step 4c)",
                table.round(3).to_string(), ""]
        if "total" in table.index.get_level_values("cls"):
            zt = table.xs("total", level="cls")[["n", "cover90", "cover50", "rms_z",
                                                   "median_width90"]]
            text += ["   zone totals side by side:",
                     zt.reindex([x for x in ("FS", "S", "P0") if x in zt.index])
                     .round(3).to_string(), ""]
        text += ["2. Zone totals: relative RMS error against the truth (mean over units) of "
                 "the MAP shares, the survey's shares and the realized MAP's shares; "
                 "posterior sd relative to the truth",
                 tot.drop(columns="replicate").groupby("region").mean().round(4).to_string(),
                 "   all units: " + ", ".join(f"{c} {tot[c].mean():.4f}"
                                              for c in tot.columns if c not in
                                              ("region", "replicate")), "",
                 "3. PIT histograms", pd.concat(pits).round(3).to_string(), "",
                 "4. MAP point accuracy (realized MAP)", pd.concat(accs).round(4).to_string(), ""]
        if not d.empty:
            text += ["5. Fit diagnostics (FS)",
                     f"   MAP converged in {int(d['map_converged'].sum())} of {len(d)}; "
                     f"largest gradient {d['map_grad_max'].max():.2e} "
                     f"(U {d['map_grad_max_U'].max():.2e}, psi {d['map_grad_max_psi'].max():.2e}); "
                     f"spread of the two starts: median {d['spread'].median():.2e}, "
                     f"max {d['spread'].max():.2e}",
                     f"   realized MAP converged in {int(d['realized_converged'].sum())} of "
                     f"{len(d)}; solver calls median {d['realized_outer_iterations'].median():.0f}, "
                     f"max {d['realized_outer_iterations'].max():.0f}",
                     f"   negative conditional variances (clipped at 0): "
                     f"{int(d['negative_variances'].sum())}",
                     f"   compute: {d['seconds'].sum() / 3600:.1f} CPU-hours over {len(d)} units; "
                     f"median {d['seconds'].median():.0f}s, max {d['seconds'].max():.0f}s"]
    write_text(args.root / "summary.txt", "\n".join(text) + "\n")
    write_text(args.root / "decision.json", json.dumps({"decision": decision, **verdict},
                                                       indent=1, default=str))
    write_text(args.root / "calibration.csv", table.reset_index().to_csv(index=False))
    write_text(args.root / "pit.csv", pd.concat(pits).reset_index().to_csv(index=False))
    write_text(args.root / "map_accuracy.csv", pd.concat(accs).reset_index().to_csv(index=False))
    write_text(args.root / "totals.csv", tot.to_csv(index=False))
    print("\n".join(lines))
    print(f"\nfull summary in {args.root / 'summary.txt'}")


if __name__ == "__main__":
    main()
