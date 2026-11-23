"""Pool step 4c and apply its decision rule (``PLAN_latent_step4c.md`` version 2,
§2-§3).

    python experiments/latent/collect_step4c.py --root $PMEDM_VB_RUNS/latent/step4c \
        [--step4b $PMEDM_VB_RUNS/latent/step4b]

Reads ``step4c_joint.py``'s output: for each simulation and region, the
selection replicates at the chosen pair (``select/.../c<c>/r<k>/t<m>/``) and
the final replicates (``final/``), pooled as the simulation's posterior; S-cal
(``Scal/``) and P0 (``P0/``) when present, with zone totals' coverage for S,
P0 and S-cal side by side. Writes, under the root, ``summary.txt``,
``decision.json``, ``calibration.csv``, ``calibration_final_only.csv`` (the
final replicates alone, all with the full number of draws), ``pit.csv``,
``choice.csv`` and ``map_accuracy.csv``.

* **Choice** (§2): per simulation and region, the chosen pair, the pair the
  summed Laplace evidence prefers, the truth-best pair, and a flag when ``τ``
  is at the grid's top; the B1 log density and evidence of every pair.
* **Calibration** as step 4 (:func:`~pmedm_vb.latent.calibration.summarize`),
  classes fitted, B1, B2, joint, rare and the zone totals (``total``).
* **Decision** (§3, :func:`~pmedm_vb.latent.calibration.decide_bands`): S
  primary; every class within step 4's bands, B2 within [0.80, 0.95] at 90%
  and [0.40, 0.60] at 50%; calibrated: go to step 6; otherwise stop and
  review, naming the failing classes. S-cal and P0 are reported with the
  same bands, not decided on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.latent.calibration import decide_bands, pit_histogram, summarize
from pmedm_vb.progress import write_text

FILES = ("posterior.csv.gz", "scores.csv", "rare.csv", "misfit.csv", "transfer.csv")


def read(dirs: list[Path], name: str) -> pd.DataFrame:
    paths = [d / name for d in dirs if (d / name).exists()]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)


def unit_dirs(root: Path, choice: dict) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for key, val in choice.items():
        sim, rid = key.split("/")
        c, m = val["c"], val["tau_multiple"]
        sel = sorted((root / "select" / sim / rid / f"c{c:g}").glob(f"r*/t{m:g}"))
        fin = sorted((root / "final" / sim / rid).glob("r*"))
        out.setdefault(sim, []).extend([*sel, *fin])
        out.setdefault(f"{sim} (final only)", []).extend(fin)
    for extra in ("Scal", "P0"):
        dirs = sorted(root.glob(f"{extra}/*/r*"))
        if dirs:
            out[extra] = dirs
    return out


def accuracy(frames: dict, label: str) -> pd.DataFrame:
    s, rare = frames["scores.csv"], frames["rare.csv"]
    s = s[s["model"] == "latent_kern30"]
    rare = rare[rare["model"] == "latent_kern30"]
    ok = rare[(rare["truth"] > 0) & (rare["pred"] > 0)]
    t = frames["transfer.csv"]
    t = t[t["model"] == "latent_kern30"]
    return pd.DataFrame({
        "lor_mae": [s["lor_mae"].mean()], "tvd": [s["tvd"].mean()],
        "rare_rms": [float(np.sqrt(np.mean(np.log(ok["pred"] / ok["truth"]) ** 2)))],
        "b1_rms_z": [t.loc[t["scored_set"] == "B1", "rms_z"].mean()],
        "b2_rms_z": [t.loc[t["scored_set"] == "B2", "rms_z"].mean()]}, index=[label])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--step4b", type=Path, default=None,
                        help="step 4b root: its pooled calibration printed beside")
    args = parser.parse_args()
    root = args.root
    choice = json.loads((root / "choice.json").read_text())
    groups = unit_dirs(root, choice)
    tables, finals, pits, accs = [], [], [], []
    for label, dirs in groups.items():
        frames = {name: read(dirs, name) for name in FILES}
        post = frames["posterior.csv.gz"]
        if post.empty:
            continue
        post = post.assign(simulation=label)
        for col in ("cover50", "cover90", "below_zero"):
            post[col] = post[col].astype(bool)
        t = summarize(post)
        if label.endswith("(final only)"):
            finals.append(t)
            continue
        tables.append(t)
        pits.append(pit_histogram(post))
        accs.append(accuracy(frames, label))
    table = pd.concat(tables)
    decision, lines = decide_bands(table, primary="S")
    rows = []
    for key, val in choice.items():
        sim, rid = key.split("/")
        rows.append({"simulation": sim, "region": rid, "c": val["c"],
                     "tau_multiple": val["tau_multiple"], "tau_at_top": val["tau_at_top"],
                     "evidence_c": val["evidence_pair"][0],
                     "evidence_tau": val["evidence_pair"][1],
                     "truth_best_c": val["truth_best_pair"][0],
                     "truth_best_tau": val["truth_best_pair"][1]})
    ch = pd.DataFrame(rows)
    pairs = pd.DataFrame([{"simulation": k.split("/")[0], "region": k.split("/")[1], **p}
                          for k, v in choice.items() for p in v["pairs"]])
    reused = [json.loads(p.read_text()) for p in root.glob("**/done.json")]
    n_reused = sum(1 for d in reused if d.get("reused_from"))
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 1000):
        text = ["Step 4c: tau and kappa chosen jointly (model version 4)", "", *lines, "",
                "1. Choice per simulation and region (B1 log density over the selection "
                "replicates); evidence-preferred and truth-best pairs beside",
                ch.to_string(index=False), "",
                f"   tau at the grid's top: {int(ch['tau_at_top'].sum())} of {len(ch)}; "
                f"evidence agrees with B1 in "
                f"{float(((ch['evidence_c'] == ch['c']) & (ch['evidence_tau'] == ch['tau_multiple'])).mean()):.0%}"
                f", truth-best agrees in "
                f"{float(((ch['truth_best_c'] == ch['c']) & (ch['truth_best_tau'] == ch['tau_multiple'])).mean()):.0%}",
                "",
                "2. Every pair: B1 log density, Laplace evidence (both summed over the "
                "selection replicates), mean LOR error",
                pairs.round(2).to_string(index=False), "",
                "3. Calibration (selection replicates at the chosen pair and final "
                "replicates)", table.round(3).to_string(), ""]
        if finals:
            text += ["4. Calibration, final replicates only (all with the full draws)",
                     pd.concat(finals).round(3).to_string(), ""]
        if args.step4b is not None and (args.step4b / "calibration.csv").exists():
            b = pd.read_csv(args.step4b / "calibration.csv")
            b = b[b["arm"] == "K2"].set_index(["simulation", "cls"])
            text += ["5. Step 4b, K2 (for comparison)",
                     b[["cover50", "cover90", "rms_z"]].round(3).to_string(), ""]
        if "total" in table.index.get_level_values("cls"):
            tot = table.xs("total", level="cls")[["n", "cover90", "cover50", "rms_z",
                                                    "median_width90"]]
            text += ["   zone totals side by side (S, and P0 / S-cal when run):",
                     tot.reindex([x for x in ("S", "P0", "Scal", "G") if x in tot.index])
                     .round(3).to_string(), ""]
        text += ["6. PIT histograms", pd.concat(pits).round(3).to_string(), "",
                 "7. MAP point accuracy (realized MAP at the chosen pair)",
                 pd.concat(accs).round(4).to_string(), "",
                 f"8. Units: {len(reused)} written, {n_reused} reused from step 4b"]
        done = [d for d in reused if "seconds" in d and d.get("seconds") is not None
                and not d.get("reused_from")]
        if done:
            d = pd.DataFrame(done)
            text += [f"   computed here: {len(d)} fits, {d['seconds'].sum() / 3600:.1f} "
                     f"CPU-hours; by phase:",
                     d.groupby("phase")["seconds"].agg(["count", "median", "max", "sum"])
                     .round(0).to_string()]
    write_text(root / "summary.txt", "\n".join(text) + "\n")
    write_text(root / "decision.json", json.dumps({**decision, "choice": rows},
                                                  indent=1, default=str))
    write_text(root / "calibration.csv", table.reset_index().to_csv(index=False))
    if finals:
        write_text(root / "calibration_final_only.csv",
                   pd.concat(finals).reset_index().to_csv(index=False))
    write_text(root / "pit.csv", pd.concat(pits).reset_index().to_csv(index=False))
    write_text(root / "choice.csv", pairs.to_csv(index=False))
    write_text(root / "map_accuracy.csv", pd.concat(accs).reset_index().to_csv(index=False))
    print("\n".join(lines))
    print(f"\nfull summary in {root / 'summary.txt'}")


if __name__ == "__main__":
    main()
