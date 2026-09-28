"""Where do MAP fits under different hierarchy levels miss the published tables?

For each ``run_map.py solve`` result given, recomputes the fitted tract, block
group and PUMA totals from its saved ``W`` and scores them against the
published values:

- tract and block group cells: ``z = (fitted - Y) / SE`` with the SE the fit used
  (``--variance-floor``, default ``zero``);
- PUMA totals of each tract category: against the summed tract estimates, with
  the SDR SE of the summed tract replicates.

Writes ``<out>/hierarchy_check.txt``: per fit, |z| quantiles by level, the
``--top`` worst cells (level, area, category, published, SE, fitted, z,
whether a published zero) and the PUMA totals with the largest |z|; and
``<out>/hierarchy_check.csv``, every cell of every fit, for R::

    $CONDA_PREFIX/bin/python experiments/hierarchy_check.py --puma 4701502 \\
        --fit $RUNS/<job>/4701502_tract_a1.npz ... --out $RUNS/hierarchy_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.data.variance import SDR_FACTOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", required=True)
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--fit", type=Path, nargs="+", required=True, help="solve-step .npz results")
    parser.add_argument("--variance-floor", default="zero")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def cells(inputs: PMEDMInputs, W: np.ndarray, floor) -> pd.DataFrame:
    """One row per scored cell: level, area, category, published, SE, fitted."""
    v = inputs.floored_variances(floor)
    split = inputs.Y_T.size
    frames = []
    for level, Y, fitted, names, areas, var in (
        ("tract", inputs.Y_T, np.asarray(inputs.A_T @ (W @ inputs.X_T)), inputs.tract_constraints,
         inputs.tracts.iloc[:, 0].astype(str), v[:split]),
        ("block group", inputs.Y_B, np.asarray(W @ inputs.X_B), inputs.bg_constraints,
         inputs.block_groups.iloc[:, 0].astype(str), v[split:]),
    ):
        frames.append(pd.DataFrame({
            "level": level,
            "area": np.tile(areas.to_numpy(), Y.shape[1]),
            "category": np.repeat(names, Y.shape[0]),
            "published": Y.ravel(order="F"),
            "se": np.sqrt(var),
            "fitted": fitted.ravel(order="F"),
        }))
    n_tracts, c_t = inputs.Y_T.shape
    summed = np.stack([inputs.sigma_l[k * n_tracts + np.arange(n_tracts)].sum(axis=0)
                       for k in range(c_t)])
    frames.append(pd.DataFrame({
        "level": "puma", "area": inputs.puma, "category": inputs.tract_constraints,
        "published": inputs.Y_T.sum(axis=0),
        "se": np.sqrt(SDR_FACTOR * np.square(summed).sum(axis=1)),
        "fitted": np.asarray(W.sum(axis=0) @ inputs.X_T).ravel(),
    }))
    out = pd.concat(frames, ignore_index=True)
    out["z"] = (out.fitted - out.published) / out.se.where(out.se > 0)
    out["published_zero"] = out.published == 0
    return out


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    floor = None if args.variance_floor == "none" else (
        "zero" if args.variance_floor == "zero" else float(args.variance_floor))
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    lines, frames = [], []
    for path in args.fit:
        with np.load(path) as saved:
            W = saved["W"]
            label = (f"{path.parent.name}/{path.stem}  hierarchy="
                     f"{saved['hierarchy'] if 'hierarchy' in saved.files else 'none'}")
        table = cells(inputs, W, floor)
        table.insert(0, "fit", str(path))
        frames.append(table)
        lines.append(f"== {label}")
        lines.append(f"   largest cell of W: {W.max():,.1f} people ({100 * W.max() / inputs.N:.3f}% of N)")
        for level, group in table.groupby("level", sort=False):
            az = group.z.abs().dropna()
            q = np.quantile(az, [0.5, 0.9, 0.99, 1.0])
            lines.append(f"   {level:<12} |z| median {q[0]:.2f}  p90 {q[1]:.2f}  p99 {q[2]:.2f}  "
                         f"max {q[3]:.2f}  (cells over 3: {int((az > 3).sum())} of {az.size})")
        worst = table.reindex(table.z.abs().sort_values(ascending=False).index).head(args.top)
        lines += ["   worst cells:"] + ["     " + row for row in worst[
            ["level", "area", "category", "published", "se", "fitted", "z", "published_zero"]
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}").splitlines()]
        puma = table[table.level == "puma"]
        top = puma.reindex(puma.z.abs().sort_values(ascending=False).index).head(10)
        lines += ["   PUMA totals, largest |z|:"] + ["     " + row for row in top[
            ["category", "published", "se", "fitted", "z"]
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}").splitlines()]
        lines.append("")
    pd.concat(frames).to_csv(args.out / "hierarchy_check.csv", index=False)
    text = "\n".join(lines) + "\n"
    (args.out / "hierarchy_check.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
