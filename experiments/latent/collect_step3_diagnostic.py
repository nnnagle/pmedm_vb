"""Collect the MAP diagnostic and apply the rules of §4.3b
(``PLAN_latent_step3.md`` version 3).

    python experiments/latent/collect_step3_diagnostic.py \
        --root $PMEDM_VB_RUNS/latent/step3_diagnostic

Reads ``<root>/fits.csv`` (``step3_diagnostic.py``) and writes, under the
root, ``diagnostic_summary.txt`` and ``selection.json``.

**Agreement** (§4.3b): within a region and ``τ``, the exact fits agree when
their final exact objectives are within 1 log-likelihood unit of each other
(largest minus smallest ``≤ 1``); the fixed-S fits agree when their final
fixed-S objectives are. The families are the four L-BFGS fits of each
objective (``m10``, ``m20``, ``gn``, ``warm``; a choice made here: the LM
fits are judged by their own rule below). A family agrees when it does for
every region and ``τ``.

**Start set:**

* exact fits agree (whether or not the fixed-S fits do; the exact objective
  defines the MAP, so its agreement decides): ``plain2``;
* exact fits disagree, fixed-S fits agree: ``fixedS``;
* both disagree: ``all4``.

**LM rule:** if ``exact_lm`` ends within 1 unit of the lowest exact objective
of all fits, for every region and ``τ``, LM replaces plain L-BFGS (memory 10)
everywhere in the chosen set (``--optimizer lm``); if ``fixedS_lm`` ends
within 1 unit of the lowest fixed-S objective of all fixed-S fits, for every
region and ``τ``, LM replaces L-BFGS for the fixed-S fit
(``--fixed-s-optimizer lm``, used by the ``fixedS`` set). Without LM fits
(a version 2 folder not yet resumed) the rule is reported as not available
and ``m10`` kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pmedm_vb.progress import write_text

TOLERANCE = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    fits = pd.read_csv(args.root / "fits.csv")
    if fits.empty:
        raise SystemExit(f"no fits in {args.root / 'fits.csv'}")
    key = {"exact": "objective", "fixedS": "fixed_s_objective"}
    rows = []
    lm = fits["fit"].str.endswith("_lm")
    for (region, m), group in fits.groupby(["region", "tau_multiple"]):
        row = {"region": region, "tau_multiple": m}
        for kind, column in key.items():
            sub = group[(group["objective_kind"] == kind) & ~lm[group.index]]
            spread = float(sub[column].max() - sub[column].min()) if len(sub) else float("nan")
            row[f"{kind}_n"] = len(sub)
            row[f"{kind}_spread"] = spread
            row[f"{kind}_agree"] = bool(len(sub) and spread <= TOLERANCE)
        row["best_exact_fit"] = group.loc[group["objective"].idxmin(), "fit"]
        for kind, column, family in (("exact", "objective", group),
                                     ("fixedS", "fixed_s_objective",
                                      group[group["objective_kind"] == "fixedS"])):
            fit = group[group["fit"] == f"{kind}_lm"]
            gap = float(fit[column].iloc[0] - family[column].min()) if len(fit) else float("nan")
            row[f"{kind}_lm_gap"] = gap
            row[f"{kind}_lm_ok"] = bool(len(fit) and gap <= TOLERANCE)
        rows.append(row)
    table = pd.DataFrame(rows)
    exact_ok, fixed_ok = bool(table["exact_agree"].all()), bool(table["fixedS_agree"].all())
    if exact_ok:
        chosen = "plain2"
    elif fixed_ok:
        chosen = "fixedS"
    else:
        chosen = "all4"
    have_lm = bool(lm.any())
    lm_exact = have_lm and bool(table["exact_lm_ok"].all())
    lm_fixed = have_lm and bool(table["fixedS_lm_ok"].all())
    optimizer, fixed_optimizer = ("lm" if lm_exact else "m10"), ("lm" if lm_fixed else "m10")
    options = f"--map-starts {chosen} --optimizer {optimizer}"
    if chosen == "fixedS":
        options += f" --fixed-s-optimizer {fixed_optimizer}"
    lm_line = ("   LM rule: no LM fits in this folder (version 2 run): not available, m10 kept"
               if not have_lm else
               f"   LM rule: exact LM within 1 of the best exact objective everywhere: {lm_exact} "
               f"-> {optimizer}; fixed-S LM within 1 of the best fixed-S objective everywhere: "
               f"{lm_fixed} -> {fixed_optimizer}" + ("" if chosen == "fixedS"
                                                      else " (used only by the fixedS set)"))
    columns = ["region", "tau_multiple", "fit", "origin", "objective", "fixed_s_objective",
               "fitted_grad_max", "grad_max", "iterations", "evaluations", "seconds",
               "converged", "lm_accepted", "lm_monotone", "dU_to_best", "max_share_change"]
    view = fits[[c for c in columns if c in fits]].sort_values(["region", "tau_multiple", "fit"])
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 500):
        text = [
            "MAP diagnostic (PLAN_latent_step3.md section 4.3a)", "",
            "1. Agreement (section 4.3b): spread = largest minus smallest final objective "
            f"over the family's fits; agree when <= {TOLERANCE:g}",
            table.round(3).to_string(index=False), "",
            f"   exact fits agree everywhere: {exact_ok}; fixed-S fits agree everywhere: "
            f"{fixed_ok}",
            f"   SELECTED START SET (section 4.3b): {chosen}"
            + ("  (exact fits agree: they decide, plan version 3)"
               if exact_ok and not fixed_ok else ""),
            lm_line,
            f"   step3_regional.py options: {options}", "",
            "2. Fits: exact objective F, fixed-S objective F_S (fixed-S fits), largest gradient "
            "component of the fitted objective and of F, iterations, evaluations, seconds, "
            "distance to the fit with the lowest F (||dU||, largest change in fitted zone "
            "shares on the scoring tables)",
            view.round(4).to_string(index=False), "",
            "3. Cost by fit family (sums over regions and tau)",
            fits.groupby("fit")[["evaluations", "seconds"]].sum().round(0).to_string(), ""]
    write_text(args.root / "diagnostic_summary.txt", "\n".join(text) + "\n")
    write_text(args.root / "selection.json", json.dumps({
        "selected": chosen, "optimizer": optimizer, "fixed_s_optimizer": fixed_optimizer,
        "lm_available": have_lm, "lm_exact_ok": lm_exact, "lm_fixed_s_ok": lm_fixed,
        "step3_regional_options": options, "exact_agree": exact_ok, "fixedS_agree": fixed_ok,
        "tolerance": TOLERANCE, "per_region_tau": table.to_dict(orient="records")},
        indent=1, default=float))
    print("\n".join(text[:9]))
    print(f"full summary in {args.root / 'diagnostic_summary.txt'}")


if __name__ == "__main__":
    main()
