"""Collect the MAP diagnostic and apply the rule of §4.3b
(``PLAN_latent_step3.md`` version 2).

    python experiments/latent/collect_step3_diagnostic.py \
        --root $PMEDM_VB_RUNS/latent/step3_diagnostic

Reads ``<root>/fits.csv`` (``step3_diagnostic.py``) and writes, under the
root, ``diagnostic_summary.txt`` and ``selection.json``.

**Agreement** (§4.3b): within a region and ``τ``, the exact fits agree when
their final exact objectives are within 1 log-likelihood unit of each other
(largest minus smallest ``≤ 1``); the fixed-S fits agree when their final
fixed-S objectives are. A family agrees when it does for every region and
``τ``.

**Rule:**

* exact fits agree: ``plain2`` (plain L-BFGS, memory 10, from ``U = 0`` and
  from the previous ``τ``'s solution);
* exact fits disagree, fixed-S fits agree: ``fixedS`` (the fixed-S solution
  polished on the exact objective, and plain L-BFGS from ``U = 0``);
* both disagree: ``all4`` (the four exact fits).

The plan's bullets overlap when the exact fits agree and the fixed-S fits do
not; the exact fits' agreement is taken to decide then (``plain2``), and the
summary says so.
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
    for (region, m), group in fits.groupby(["region", "tau_multiple"]):
        row = {"region": region, "tau_multiple": m}
        for kind, column in key.items():
            sub = group[group["objective_kind"] == kind]
            spread = float(sub[column].max() - sub[column].min()) if len(sub) else float("nan")
            row[f"{kind}_n"] = len(sub)
            row[f"{kind}_spread"] = spread
            row[f"{kind}_agree"] = bool(len(sub) and spread <= TOLERANCE)
        row["best_exact_fit"] = group.loc[group["objective"].idxmin(), "fit"]
        rows.append(row)
    table = pd.DataFrame(rows)
    exact_ok, fixed_ok = bool(table["exact_agree"].all()), bool(table["fixedS_agree"].all())
    note = ""
    if exact_ok:
        chosen = "plain2"
        if not fixed_ok:
            note = ("the exact fits agree while the fixed-S fits do not: the plan's bullets "
                    "overlap; the exact fits' agreement decides (plain2)")
    elif fixed_ok:
        chosen = "fixedS"
    else:
        chosen = "all4"
    columns = ["region", "tau_multiple", "fit", "origin", "objective", "fixed_s_objective",
               "fitted_grad_max", "grad_max", "iterations", "evaluations", "seconds",
               "converged", "dU_to_best", "max_share_change"]
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
            f"   SELECTED START SET (section 4.3b): {chosen}" + (f"  [{note}]" if note else ""),
            "",
            "2. Fits: exact objective F, fixed-S objective F_S (fixed-S fits), largest gradient "
            "component of the fitted objective and of F, iterations, evaluations, seconds, "
            "distance to the fit with the lowest F (||dU||, largest change in fitted zone "
            "shares on the scoring tables)",
            view.round(4).to_string(index=False), "",
            "3. Cost by fit family (sums over regions and tau)",
            fits.groupby("fit")[["evaluations", "seconds"]].sum().round(0).to_string(), ""]
    write_text(args.root / "diagnostic_summary.txt", "\n".join(text) + "\n")
    write_text(args.root / "selection.json", json.dumps({
        "selected": chosen, "exact_agree": exact_ok, "fixedS_agree": fixed_ok, "note": note,
        "tolerance": TOLERANCE, "per_region_tau": table.to_dict(orient="records")},
        indent=1, default=float))
    print("\n".join(text[:7]))
    print(f"full summary in {args.root / 'diagnostic_summary.txt'}")


if __name__ == "__main__":
    main()
