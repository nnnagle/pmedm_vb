"""Pool step 5's regional run (arm C3) and apply the plan's phase 3 rules
(``PLAN_latent_step5.md`` version 3, §5: agreement with stage 2, and
calibration).

    python experiments/latent/collect_step5.py --root $PMEDM_VB_RUNS/latent/step5 \\
        --step4e $PMEDM_VB_RUNS/latent/step4e --step4d $PMEDM_VB_RUNS/latent/step4d \\
        --step4c $PMEDM_VB_RUNS/latent/step4c

Reads ``step5_regional.py``'s finished units (``<root>/C3/<region>/r<k>/``
with ``done.json``) and, for comparison, step 4e's FScut, step 4d's FS and
step 4c's S and P0, and writes under ``<root>``:

* **Calibration** (:func:`collect_step4d.report`, with C3 primary, as
  ``collect_step4e.py`` does): ``summary.txt``, ``decision.json``,
  ``calibration.csv``, ``pit.csv``, ``map_accuracy.csv``, ``totals.csv``;
  every class within step 4c's bands. The class ``hhsize`` (households by
  size) is C3's only (steps 4c–4e did not score it): it is in C3's rows and
  in the verdict on all of C3's classes, and the verdict is also given on
  the classes the other arms share (without ``hhsize``), so that C3 can be
  read beside them.
* **Agreement with stage 2** at the MAP parameters (§5's phase 3 rule), the
  units' cells pooled (each unit's ``map_functionals.csv.gz``): cells with
  stage 2 mean ≥ 50, ``|mean difference| ≤ 0.1`` stage 2 sd, variance ratio
  in ``[0.8, 1.25]``; the share of cells passing both, overall and by class
  (and by region), against the rule's 90% (:func:`step5_pilot.agreement`,
  applied to the pooled cells). ``agreement.csv``.
* **Mixing** at the MAP chains, pooled per class: median and minimum bulk and
  tail ESS per sweep, ``R̂`` (max, share below 1.01 and 1.05), the version 3
  rule's ``R̂`` part (≥ 95% below 1.01 and all below 1.05); the draw chains'
  re-burn (the largest first crossing, the share of chains crossing after
  the re-burn used), the stored states' checks and compute. ``mixing.csv``.
* **Decision** (§5, phase 3): agreement and calibration;
  ``decision_step5.json``, and the text appended to ``summary.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import step5_pilot as s5p
from collect_step4d import report, step4c_groups
from pmedm_vb.progress import write_text

ARM = "C3"
#: Classes C3 scores that steps 4c–4e did not.
C3_ONLY = ("hhsize",)


def unit_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.glob(f"{ARM}/*/r*") if (d / "done.json").exists())


def read_cells(dirs: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(d / "map_functionals.csv.gz") for d in dirs],
                     ignore_index=True)


def pooled_agreement(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """:func:`step5_pilot.agreement` on the pooled cells (one row of cells)."""
    def arr(col):
        return cells[col].to_numpy(dtype=float)[None, :]

    return s5p.agreement(arr("mean"), arr("var"), arr("ess_bulk"), arr("stage2_mean"),
                         arr("stage2_var"), cells["cls"].to_numpy())


def pooled_mixing(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in list(pd.unique(cells["cls"])) + ["all"]:
        sub = cells if c == "all" else cells[cells["cls"] == c]
        live = sub[~sub["constant"].astype(bool)]
        row = {"cls": c, "n": len(sub), "constant": int(sub["constant"].astype(bool).sum())}
        if len(live):
            rh = live["rhat"].to_numpy(float)
            row.update({
                "ess_bulk_per_sweep_median": float(live["ess_bulk_per_sweep"].median()),
                "ess_bulk_per_sweep_min": float(live["ess_bulk_per_sweep"].min()),
                "ess_tail_per_sweep_median": float(live["ess_tail_per_sweep"].median()),
                "ess_tail_per_sweep_min": float(live["ess_tail_per_sweep"].min()),
                "rhat_max": float(np.max(rh)),
                "rhat_share_below": float(np.mean(rh < s5p.RHAT_MAX)),
                "rhat_share_below_1.05": float(np.mean(rh < s5p.RHAT_ALL_MAX))})
            row["rhat_rule_v3"] = bool(row["rhat_share_below"] >= s5p.RHAT_SHARE_MIN
                                       and row["rhat_max"] < s5p.RHAT_ALL_MAX)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="step 5 regional output root")
    parser.add_argument("--step4e", required=True, type=Path, help="step 4e output root")
    parser.add_argument("--step4d", required=True, type=Path, help="step 4d output root")
    parser.add_argument("--step4c", required=True, type=Path, help="step 4c output root")
    args = parser.parse_args()
    dirs = unit_dirs(args.root)
    if not dirs:
        raise SystemExit(f"no finished {ARM} units under {args.root / ARM}")
    others, n_expected = step4c_groups(args.step4c)
    groups = {ARM: dirs, "FScut": sorted(args.step4e.glob("FScut/*/r*")),
              "FS": sorted(args.step4d.glob("FS/*/r*")), **others}
    adopt = ("calibrated: every class within step 4c's bands (see the phase 3 decision below, "
             "which also needs the agreement with stage 2)")
    report(args.root, ARM, groups, n_expected,
           "Step 5, phase 3: the realized-population sampler (C3) on simulation S",
           adopt, "calibration, as step 4e's")

    # -- calibration verdicts -------------------------------------------------
    verdict = json.loads((args.root / "decision.json").read_text())
    classes = verdict["per_simulation"].get(ARM, {}).get("classes", {})
    shared = {c: v for c, v in classes.items() if c not in C3_ONLY}
    calib = pd.read_csv(args.root / "calibration.csv")
    c3cal = calib[calib["simulation"] == ARM].set_index("cls")
    below = c3cal["share_below_zero"].to_dict()

    # -- agreement and mixing (pooled cells) ------------------------------------
    cells = read_cells(dirs)
    agree_tab, agree = pooled_agreement(cells)
    by_region = []
    for rid, sub in cells.groupby("region"):
        t_, v_ = pooled_agreement(sub)
        allrow = t_[t_["cls"] == "all"]
        by_region.append({"region": rid, "units": int(sub["replicate"].nunique()),
                          "n": v_["n"], "share_both_ok": float(allrow["share_both_ok"].iloc[0])
                          if len(allrow) else np.nan, "pass": v_["pass"]})
    by_region = pd.DataFrame(by_region)
    mix = pooled_mixing(cells)
    done = [json.loads((d / "done.json").read_text()) for d in dirs]
    units = pd.DataFrame([{
        "region": u["region"], "replicate": u["replicate"],
        "agreement_pass": u["agreement_pass"],
        "reburn_max": u["reburn"]["max"],
        "reburn_share_after": u["reburn"]["share_crossing_after_reburn"],
        "log_target_rhat": u["map_chains"]["log_target"]["rhat"],
        "suggested_burn": u["map_chains"]["burn_in"]["suggested_burn"],
        "checks_ok": u["checks"]["ok"], "seconds": u["seconds"], "peak_gb": u["peak_gb"],
        **{f"t_{k}": v for k, v in u["timings"].items()}} for u in done])
    reburn_used = {u["draw_chains"]["reburn"] for u in done}

    # -- the decision (§5, phase 3) --------------------------------------------
    allrow = agree_tab[agree_tab["cls"] == "all"]
    share = float(allrow["share_both_ok"].iloc[0]) if len(allrow) else float("nan")
    agree_ok = bool(agree["pass"])
    calib_ok = bool(classes) and all(classes.values())
    calib_shared_ok = bool(shared) and all(shared.values())
    complete = len(done) >= n_expected
    lines = [
        f"Agreement with stage 2 at the MAP parameters (§5, phase 3; {len(done)} units pooled): "
        f"cells with stage 2 mean >= {s5p.AGREE_MIN_MEAN:g}: {agree['n']}; share with |dm| <= "
        f"{s5p.AGREE_MEAN_SD} sd and variance ratio in {list(s5p.AGREE_VAR)}: {share:.3f} "
        f"(rule: >= {s5p.AGREE_SHARE:.0%}) -> {'PASS' if agree_ok else 'FAIL'}",
        "Calibration (§5, phase 3; step 4c's bands, every class of C3): "
        + ", ".join(f"{c} {'pass' if v else 'FAIL'}" for c, v in classes.items())
        + f" -> {'PASS' if calib_ok else 'FAIL'}; on the classes shared with steps 4c-4e "
        f"(without {', '.join(C3_ONLY)}): {'PASS' if calib_shared_ok else 'FAIL'}",
        "Share of 90% intervals below zero (C3, by class): "
        + ", ".join(f"{c} {v:.3f}" for c, v in below.items())]
    if not complete:
        decision = f"incomplete ({len(done)} of {n_expected} units)"
    elif agree_ok and calib_ok:
        decision = ("agreement and calibration: the sampler is the stage 3 for step 6 (the real "
                    "application), and step 6 reports draws of C; the share of intervals below "
                    "zero is 0 by construction")
    else:
        fails = ([] if agree_ok else ["agreement with stage 2 (a larger gap is reviewed, since "
                                      "one of the two is wrong there)"]) + \
                ([] if calib_ok else ["calibration (failing: "
                                      + ", ".join(c for c, v in classes.items() if not v) + ")"])
        decision = "review: " + "; ".join(fails)
    lines.append(f"DECISION (PLAN_latent_step5.md §5, phase 3, arm {ARM}): {decision}")

    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 1000):
        text = ["", "=" * 80, "Step 5, phase 3 (PLAN_latent_step5.md §5)", *lines, "",
                f"6. Agreement with stage 2 at the MAP parameters, cells pooled over {len(done)} "
                "units (MC se from the MAP chains' bulk ESS)",
                agree_tab.round(4).to_string(index=False), "", "   by region:",
                by_region.round(4).to_string(index=False),
                f"   units passing on their own: {int(units['agreement_pass'].sum())} of "
                f"{len(units)}", "",
                "7. Mixing at the MAP chains, pooled per class (ESS per sweep = ESS / (chains x "
                "kept sweeps); rhat_rule_v3: >= 95% below 1.01 and all below 1.05)",
                mix.round(4).to_string(index=False), "",
                f"   draw chains' re-burn (used {sorted(reburn_used)}): largest first crossing "
                f"{int(units['reburn_max'].max())}; units with a chain crossing after it "
                f"{int((units['reburn_share_after'] > 0).sum())}; log target R-hat at the MAP "
                f"chains max {units['log_target_rhat'].max():.4f}; suggested burn-in max "
                f"{units['suggested_burn'].max()}",
                f"   stored-state checks (totals exact, no negative count, traced = stored): "
                f"{int(units['checks_ok'].sum())} of {len(units)} units ok",
                "   compute per unit (s; peak GB):",
                units.drop(columns=["replicate", "agreement_pass", "checks_ok"])
                .groupby("region").agg(["median", "max"]).round(1).to_string()]
    summary = (args.root / "summary.txt").read_text()
    write_text(args.root / "summary.txt", summary + "\n".join(text) + "\n")
    write_text(args.root / "agreement.csv", agree_tab.to_csv(index=False))
    write_text(args.root / "mixing.csv", mix.to_csv(index=False))
    write_text(args.root / "decision_step5.json", json.dumps({
        "decision": decision, "units": len(done), "expected": n_expected,
        "agreement": {"pass": agree_ok, "share_both_ok": share, "n": agree["n"],
                      "by_class": agree_tab.to_dict(orient="records"),
                      "by_region": by_region.to_dict(orient="records")},
        "calibration": {"pass": calib_ok, "pass_shared_classes": calib_shared_ok,
                        "classes": classes, "share_below_zero": below},
        "mixing": mix.to_dict(orient="records")}, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
