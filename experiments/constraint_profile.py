"""Profile the constraint categories behind the weight blow-ups, one PUMA at a time.

Findings 2, 5 and 8 of ``solvers_synopsis.md`` trace the draws that put an
implausible share of ``N`` on one (block group, record) cell to rare
categories carried by a few PUMS households, some of them in bulk: race
(``nh_other_race``, ``nh_nhpi``, ``nh_asian``, ``nh_aian``) and commute mode
(``bicycle``, ``taxicab``, ``other_means``). This sets the two sides of each
such constraint next to each other, from the assembled inputs alone -- no fit:

- **Block group side**: per block group, the published count, its standard
  error (published, and floored at the zero-count variance as the fits use),
  whether it is a published zero, and the block group's total population (the
  sum of its ``B01001`` cells), so the count can be read as a share.
- **PUMS side**: per record carrying the category, its loading (persons in the
  unit with the characteristic; 0/1 for a household-level one), household size
  (sum of its ``B01001`` counts), design weight, whether it is group quarters,
  and its share of the category's weighted PUMS total.

Writes ``<out>/<puma>_bg.csv``, ``<out>/<puma>_pums.csv`` and
``<out>/<puma>_summary.txt``. Run where the inputs are (``$PMEDM_VB_DATA``);
it takes seconds::

    $CONDA_PREFIX/bin/python experiments/constraint_profile.py --puma 4701502 \\
        --out /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/constraint_profile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir

#: The categories findings 2, 5 and 8 name as carrying the walls.
DEFAULT_CATEGORIES = (
    "B03002.nh_other_race",
    "B03002.nh_nhpi",
    "B03002.nh_asian",
    "B03002.nh_aian",
    "B08301.bicycle",
    "B08301.taxicab",
    "B08301.other_means",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", required=True)
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--category", nargs="+", default=list(DEFAULT_CATEGORIES),
                        help="block group constraint names, '<table>.<category>'")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def quantiles(x: np.ndarray, fmt: str = "{:.1f}") -> str:
    qs = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return "  ".join(f"{label} {fmt.format(v)}"
                     for label, v in zip(["min", "p25", "p50", "p75", "p90", "max"], qs))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)

    names = list(inputs.bg_constraints)
    missing = [c for c in args.category if c not in names]
    if missing:
        raise SystemExit(f"not block group constraints of PUMA {args.puma}: {missing}")

    X = sp.csc_matrix(inputs.X_B)
    n_zones = inputs.n_zones
    split = inputs.Y_T.size
    se = np.sqrt(inputs.sigma_v[split:]).reshape(inputs.Y_B.shape, order="F")
    se_floor = np.sqrt(inputs.floored_variances("zero")[split:]).reshape(inputs.Y_B.shape, order="F")
    age_sex = [k for k, name in enumerate(names) if name.startswith("B01001.")]
    bg_population = inputs.Y_B[:, age_sex].sum(axis=1)
    household_size = np.asarray(X[:, age_sex].sum(axis=1)).ravel()
    zones = inputs.zones.iloc[:, 0].astype(str).to_numpy()
    tracts = inputs.tracts.iloc[:, 0].astype(str).to_numpy()[inputs.zone_tracts()]
    units = inputs.units
    weight = units["weight"].to_numpy(float)
    gq = units["is_group_quarters"].to_numpy(bool)

    bg_rows, pums_rows, lines = [], [], []
    lines.append(f"PUMA {args.puma}: {n_zones} block groups, {inputs.n_units:,} units "
                 f"({int(gq.sum()):,} group quarters), N = {inputs.N:,.0f}")
    for name in args.category:
        k = names.index(name)
        y = inputs.Y_B[:, k]
        bg_rows.append(pd.DataFrame({
            "category": name, "block_group": zones, "tract": tracts,
            "published": y, "se": se[:, k], "se_floored": se_floor[:, k],
            "is_zero": y == 0, "bg_population": bg_population,
            "share_of_bg": np.divide(y, bg_population, out=np.full(n_zones, np.nan),
                                     where=bg_population > 0),
        }))

        column = X[:, k].toarray().ravel()
        carriers = np.flatnonzero(column > 0)
        weighted = weight[carriers] * column[carriers]
        total = weighted.sum()
        frame = pd.DataFrame({
            "category": name,
            "serialno": units["SERIALNO"].to_numpy()[carriers],
            "loading": column[carriers],
            "household_size": household_size[carriers],
            "weight": weight[carriers],
            "is_gq": gq[carriers],
            "weighted_loading": weighted,
            "share_of_weighted_total": weighted / total if total > 0 else np.nan,
        }).sort_values("weighted_loading", ascending=False)
        pums_rows.append(frame)

        published_total = y.sum()
        loads = pd.Series(column[carriers]).value_counts().sort_index()
        lines += [
            "",
            f"== {name}",
            "   block groups:",
            f"     published zero in {int((y == 0).sum())} of {n_zones}; "
            f"summed published count {published_total:,.0f}",
            f"     count (nonzero BGs)   {quantiles(y[y > 0]) if (y > 0).any() else 'none'}",
            f"     share of BG pop (nonzero BGs) "
            f"{quantiles(y[y > 0] / bg_population[y > 0], '{:.3f}') if (y > 0).any() else 'none'}",
            f"     SE, published         {quantiles(se[:, k])}",
            f"     SE, zero-floored      {quantiles(se_floor[:, k])}",
            "   PUMS records:",
            f"     carriers {carriers.size:,} of {inputs.n_units:,} units "
            f"({int(gq[carriers].sum())} group quarters)",
            f"     loading (persons per carrier): "
            + ", ".join(f"{int(v)}: {c}" for v, c in loads.items()),
            f"     weighted PUMS total {total:,.0f} against summed published {published_total:,.0f}",
        ]
        if carriers.size:
            top = frame.head(5)
            lines += [
                f"     largest carrier holds {100 * frame['share_of_weighted_total'].iloc[0]:.1f}% "
                f"of the weighted total; top 5 hold "
                f"{100 * top['share_of_weighted_total'].sum():.1f}%",
                "     top carriers:",
            ]
            lines += ["       " + row for row in top[
                ["serialno", "loading", "household_size", "weight", "is_gq",
                 "share_of_weighted_total"]].to_string(index=False, float_format="%.3f").splitlines()]

    pd.concat(bg_rows).to_csv(args.out / f"{args.puma}_bg.csv", index=False)
    pd.concat(pums_rows).to_csv(args.out / f"{args.puma}_pums.csv", index=False)
    text = "\n".join(lines) + "\n"
    (args.out / f"{args.puma}_summary.txt").write_text(text)
    print(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
