"""Does the model know a PUMA total as well as the survey does?

For each constrained category, the PUMA total is the sum of that category's
cells at one level (tract or block group). Its survey variance is the SDR
variance of the summed replicates, ``4/80 * ||sum_r L_r||^2`` with ``L`` the
replicate deviations the inputs carry -- the same as summing the variance
replicate estimates, and exact whatever the correlations between areas. The
model's variance for the same total is ``a' Sigma(alpha) a`` with ``a`` the
indicator of those rows:

- ``Sigma = D + B B'``, ``D = v - (1 - alpha) s``, ``B = sqrt((1 - alpha) 4/80) L``
  (``assemble/sigma.py``), with ``v`` the variances (floored or not) and ``s``
  the replicate variances;
- **tapered by tract** (what the fits use), ``B B'`` is kept only inside a
  tract, so ``a' Sigma a = sum_r D_r + (1 - alpha) 4/80 sum_t ||sum_{r in t} L_r||^2``;
- **untapered**, the last sum is over one group holding every row.

At ``alpha = 1`` both are ``sum_r v_r``, the independent-cells variance. The
ratio model / survey says how loosely (> 1) or tightly (< 1) the model holds
the PUMA total. A zero cell has no replicate spread, so it adds nothing to
the survey variance but its modelled (or floored) variance to the model's.

Per PUMA, level and category: the PUMA estimate, its survey SE and CV, the
independent-cells variance, and the model variance for each alpha, taper and
variance floor, with its ratio to the survey variance. Writes
``<out>/aggregate_variance.csv`` and ``<out>/summary.txt`` (ratio quantiles
over categories; the categories where survey and independent-cells variance
differ most). From the assembled inputs alone; seconds::

    $CONDA_PREFIX/bin/python experiments/aggregate_variance.py \\
        --out /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/aggregate_variance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.data.variance import SDR_FACTOR

PUMAS = ("4701501", "4701502", "4701503", "4701504")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="+", default=list(PUMAS))
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 0.1, 0.01])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def summed_by_group(l: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``(n_groups, 80)``: the replicate deviations summed within each group."""
    labels, index = np.unique(groups, return_inverse=True)
    out = np.zeros((labels.size, l.shape[1]))
    np.add.at(out, index, l)
    return out


def puma_rows(inputs: PMEDMInputs, alphas) -> list[dict]:
    if inputs.sigma_l is None:
        raise ValueError(f"{inputs.puma}: inputs carry no replicate deviations")
    l = inputs.sigma_l
    targets = inputs.targets()
    tracts = inputs.constraint_tracts()
    s = SDR_FACTOR * np.square(l).sum(axis=1)  # per-cell replicate variance
    variances = {"none": inputs.floored_variances(None), "zero": inputs.floored_variances("zero")}
    n_t, n_b = inputs.Y_T.shape[0], inputs.Y_B.shape[0]
    blocks = [("tract", 0, n_t, inputs.tract_constraints),
              ("bg", inputs.Y_T.size, n_b, inputs.bg_constraints)]
    rows = []
    for level, offset, n_areas, names in blocks:
        for k, name in enumerate(names):
            r = np.arange(offset + k * n_areas, offset + (k + 1) * n_areas)
            estimate = float(targets[r].sum())
            survey = SDR_FACTOR * float(np.square(l[r].sum(axis=0)).sum())
            within = SDR_FACTOR * float(np.square(summed_by_group(l[r], tracts[r])).sum())
            row = {
                "puma": inputs.puma, "level": level, "constraint": name, "areas": n_areas,
                "zero_cells": int((targets[r] == 0).sum()),
                "estimate": estimate, "survey_var": survey, "survey_se": np.sqrt(survey),
                "survey_cv": np.sqrt(survey) / estimate if estimate > 0 else np.nan,
                "sum_cell_var_published": float(variances["none"][r].sum()),
                "sum_replicate_var": float(s[r].sum()),
            }
            row["independent_over_survey"] = row["sum_cell_var_published"] / survey if survey > 0 else np.nan
            for floor, v in variances.items():
                for alpha in alphas:
                    d = float((v[r] - (1 - alpha) * s[r]).sum())
                    for taper, low_rank in (("tract", within), ("none", survey)):
                        model = d + (1 - alpha) * low_rank
                        key = f"floor_{floor}_a{alpha:g}_taper_{taper}"
                        row[f"model_var_{key}"] = model
                        row[f"ratio_{key}"] = model / survey if survey > 0 else np.nan
            rows.append(row)
    return rows


def quantiles(x: pd.Series) -> str:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if not len(x):
        return "no categories"
    q = np.quantile(x, [0.1, 0.5, 0.9])
    return f"p10 {q[0]:.2f}  median {q[1]:.2f}  p90 {q[2]:.2f}  (n={len(x)})"


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for puma in args.puma:
        rows += puma_rows(PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma), args.alpha)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "aggregate_variance.csv", index=False)

    lines = ["Model variance of each category's PUMA total over its survey (summed-replicate) variance.",
             "Ratio > 1: the model holds the PUMA total more loosely than the survey measures it.",
             "Quantiles over categories with a nonzero survey variance.", ""]
    for (puma, level), group in frame.groupby(["puma", "level"], sort=False):
        lines.append(f"== {puma} {level}: {len(group)} categories, "
                     f"survey CV median {group.survey_cv.median():.3f}")
        lines.append(f"   independent cells (sum of published variances) / survey: "
                     f"{quantiles(group.independent_over_survey)}")
        for floor in ("none", "zero"):
            for alpha in args.alpha:
                for taper in ("tract", "none"):
                    key = f"floor_{floor}_a{alpha:g}_taper_{taper}"
                    lines.append(f"   floor {floor:<4} alpha {alpha:<5g} taper {taper:<5}: "
                                 f"{quantiles(group[f'ratio_{key}'])}")
        ranked = group.dropna(subset=["independent_over_survey"]).sort_values("independent_over_survey")
        show = ["constraint", "estimate", "survey_se", "independent_over_survey",
                f"ratio_floor_zero_a{min(args.alpha):g}_taper_tract"]
        lines += ["   largest independent/survey (survey tighter than independent cells):",
                  ranked[show].tail(5).iloc[::-1].to_string(index=False, float_format=lambda v: f"{v:,.2f}"),
                  "   smallest:",
                  ranked[show].head(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"), ""]
    text = "\n".join(lines) + "\n"
    (args.out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
